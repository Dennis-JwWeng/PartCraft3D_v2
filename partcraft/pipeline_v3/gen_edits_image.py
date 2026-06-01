"""Step s1 (variant) — Generate edit instructions via VLM (image_semantic, Mode A).

This is the **overview-image** variant of :mod:`gen_edits` (which runs Mode B,
text-only).  Copy-before-edit: the canonical text-mode file is left untouched;
the TRELLIS.2 shard opts into this variant with ``EDIT_GEN_MODE=image``.

What it restores
----------------
The VLM generates edits while *looking at* the overview 分割图 (segmentation
map: a 5×2 grid — TOP row = 5 RGB photos, BOTTOM row = the same views with every
part painted a distinct palette colour), instead of reasoning purely from text
captions.

Per-object flow (``gen_edits_streaming``):

    1. Render + save ``phase1/overview.png``        (reuses gen_edits.backfill_overviews)
    2. Rule-based visibility pre-pass: count each part's pixels across the 5
       segmentation views (qc_rules.count_part_pixels_in_overview).  A part with
       < ``VIS_MIN_PIXELS`` pixels in *every* view is judged 看不到 / 没必要存在
       and dropped BEFORE the VLM — no edit is generated for it.
       (Scope: edit-generation only — the mesh / downstream are NOT touched.)
    3. Build the Mode-A menu (part_id | palette-colour | description) from the
       **visible parts only**, and the edit quota from the visible count.
    4. Call ``call_vlm_image_async`` with the overview PNG + ``SYSTEM_PROMPT_A``.
    5. Write ``phase1/parsed.json`` (+ ``phase1/visibility.json`` audit record).

``SYSTEM_PROMPT_A`` is derived from the live ``SYSTEM_PROMPT_B`` by swapping
*only* its INPUT section for image-input instructions — the OUTPUT schema is
byte-for-byte identical, so the existing ``validate_edit_json`` applies unchanged.

``VIS_MIN_PIXELS`` (default 1 = drop only truly-zero parts, matching the
"pixel_counts all zero" definition) is overridable via the env var of the same
name.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

from partcraft.pipeline_v3.vlm_core import (  # noqa: E402
    SYSTEM_PROMPT_B, _QUOTA_LINE,
    build_semantic_list,
    call_vlm_image_async, extract_json_object, validate_edit_json,
    compute_edit_quota, MAX_PARTS, _pick_global_edit_note,
)
from partcraft.render.overview import _PALETTE_NAMES

from .paths import ObjectContext
from .status import update_step, step_done, STATUS_OK, STATUS_FAIL, STATUS_SKIP
# Reuse the proven overview renderer + result dataclass from the text-mode file
# (rendering an overview is identical regardless of how edits are generated).
from .gen_edits import Phase1Result, backfill_overviews


# ───────────────────── visibility threshold (rule-based) ─────────────────────
# A part is "visible" iff at least one of the 5 segmentation views shows
# >= VIS_MIN_PIXELS palette pixels for it.  Default 1 ⇒ drop only parts that are
# strictly invisible (pixel_counts all zero — the zero_visible_pixels case).
VIS_MIN_PIXELS = int(os.environ.get("VIS_MIN_PIXELS", "1"))

# ───────────────────── white-model (白模) detection ──────────────────────────
# Some inputs are untextured white/grey meshes ("白模").  Their TOP-row RGB has
# no real colour/material, so the VLM would only *hallucinate* material / color
# edits (observed: every colour edit falls back to the same "deep crimson red").
# When detected we ZERO the material+color quota and tell the VLM to reason by
# shape only — deletion / modification / scale / global still apply.
#
# Detection: of the foreground pixels (object region, taken from the BOTTOM-row
# segmentation mask, sampled in the aligned TOP-row RGB), what fraction have a
# meaningful HSV saturation (> WHITE_MODEL_SAT).  A textured object has a large
# colourful fraction; a 白模 has ~0.  Calibrated on shard-08 overviews: pure
# white models score 0%, partially-coloured objects >= 14%.
WHITE_MODEL_COLORFUL_FRAC = float(os.environ.get("WHITE_MODEL_COLORFUL_FRAC", "0.05"))
WHITE_MODEL_SAT = int(os.environ.get("WHITE_MODEL_SAT", "40"))


def white_model_colorful_frac(ov_img) -> float:
    """Fraction of object (foreground) pixels that are colourful (HSV S > sat).

    Uses the bottom-row segmentation mask to locate the object, then measures
    saturation in the spatially-aligned top-row RGB cell.  Returns 0.0 when no
    foreground is found (treated as white/empty)."""
    import numpy as _np
    import cv2 as _cv2
    from partcraft.pipeline_v3.qc_rules import _N_VIEWS, _COL_SEP, _ROW_SEP
    H, W = ov_img.shape[:2]
    Wc = (W - (_N_VIEWS - 1) * _COL_SEP) // _N_VIEWS
    Hc = (H - _ROW_SEP) // 2
    colourful = 0
    total = 0
    for v in range(_N_VIEWS):
        x0 = v * (Wc + _COL_SEP)
        top = ov_img[0:Hc, x0:x0 + Wc]
        bot = ov_img[Hc + _ROW_SEP:Hc + _ROW_SEP + Hc, x0:x0 + Wc]
        fg = ~_np.all(bot.astype(_np.int32) > 230, axis=2)   # object region (seg mask)
        n = int(fg.sum())
        if n < 50:
            continue
        sat = _cv2.cvtColor(top, _cv2.COLOR_BGR2HSV)[:, :, 1][fg]
        colourful += int((sat > WHITE_MODEL_SAT).sum())
        total += n
    return (colourful / total) if total else 0.0


# ───────────────────── Mode-A prompts (built from the live Mode-B prompt) ────
_INPUT_B = (
    "INPUT (caption list — no image):\n"
    "  You will receive a semantic part list with columns: part_id | description.\n"
    "  Reason about parts purely from their text descriptions."
)
_INPUT_A = (
    "INPUT (5×2 overview image + semantic menu):\n"
    "  You receive a 5×2 grid IMAGE — TOP row: 5 RGB photos (views 0–4);\n"
    "  BOTTOM row: the same 5 cameras re-rendered with every part painted a\n"
    "  distinct palette colour (one colour per part).  You also receive a part\n"
    "  menu with columns: part_id | palette-colour | description.\n"
    "  • Locate each part_id by finding its palette colour in the BOTTOM row,\n"
    "    then read its true appearance from the matching region in the TOP-row photos.\n"
    "  • Palette colour names are INTERNAL labels — NEVER use them in any output\n"
    "    text field; describe real colour/appearance using the TOP-row photos.\n"
    "  • The menu lists ONLY parts that are visible in the overview; generate\n"
    "    edits exclusively for the listed part_ids."
)
SYSTEM_PROMPT_A = SYSTEM_PROMPT_B.replace(_INPUT_B, _INPUT_A)
assert SYSTEM_PROMPT_A != SYSTEM_PROMPT_B, (
    "SYSTEM_PROMPT_A derivation failed: the Mode-B INPUT block text drifted — "
    "update _INPUT_B in gen_edits_image.py to match vlm_core.SYSTEM_PROMPT_B."
)

USER_PROMPT_IMAGE_SEMANTIC = (
    "[Image: 5×2 grid — TOP row = 5 RGB photos, BOTTOM row = same 5 views with "
    "parts colour-coded by palette ID. Palette colours are INTERNAL labels — do "
    "NOT use them in output text.]\n\n"
    "# PART MENU  (id · palette-colour · description)\n"
    "{part_menu}\n\n"
) + _QUOTA_LINE


# ───────────────────── menu / visibility helpers ─────────────────────────────
import re as _re

_MENU_LINE = _re.compile(r'part_(\d+)\s+"(.*)"')


def _pid_names(mesh_npz: Path, img_npz: Path, anno_obj_dir: "Path | None") -> tuple[list[int], dict[int, str]]:
    """Reuse the Mode-B builder for the canonical (pid set, descriptions), then
    parse its regular ``part_{id}  "{name}"`` lines back into a {pid: name} map.
    Guarantees the Mode-A menu carries the exact same pids + descriptions as the
    validated text path — we only add the palette-colour column on top."""
    pids, text_menu = build_semantic_list(mesh_npz, img_npz, anno_obj_dir=anno_obj_dir)
    names: dict[int, str] = {}
    for ln in text_menu.splitlines():
        m = _MENU_LINE.search(ln)
        if m:
            names[int(m.group(1))] = m.group(2)
    return pids, names


def _image_menu(visible_pids: list[int], names: dict[int, str]) -> str:
    """part_id | palette-colour | description — palette colour matches the
    BOTTOM-row overlay so the VLM can cross-reference the image."""
    lines = []
    for pid in visible_pids:
        color = _PALETTE_NAMES[pid % len(_PALETTE_NAMES)]
        desc = names.get(pid, f"part_{pid}")
        lines.append(f'  part_{pid:<3d}   {color:<8s}  "{desc}"')
    return "\n".join(lines)


def _format_user_prompt(menu: str, quota: dict, seed: int) -> str:
    n_global = quota.get("global", 0)
    roster = _pick_global_edit_note(seed, n_global) if n_global > 0 else ""
    return USER_PROMPT_IMAGE_SEMANTIC.format(
        part_menu=menu,
        n_total=sum(quota.values()),
        n_deletion=quota.get("deletion", 0),
        n_modification=quota.get("modification", 0),
        n_scale=quota.get("scale", 0),
        n_material=quota.get("material", 0),
        n_color=quota.get("color", 0),
        n_global=quota.get("global", 0),
        global_note=roster,
    )


def _halve_edit_quota(quota: dict) -> dict:
    """Reduced quota for the retry attempt (shorter response → less truncation).
    Mirrors gen_edits._halve_edit_quota."""
    out: dict = {}
    for k, v in quota.items():
        out[k] = max(1, v // 2) if k in ("deletion", "modification") else max(0, v // 2)
    return out


def _format_retry_prompt(menu: str, quota: dict) -> str:
    """Rebuild the user prompt with a halved quota; salt the roster seed so the
    retry sees a different global-style shuffle."""
    return _format_user_prompt(menu, quota, seed=hash(menu[:32]) ^ 0xA11CE)


# ───────────────────── prerender worker (process pool) ───────────────────────

def _prepare_edit_menu_image_worker(args: tuple):
    """Top-level pickleable worker.  Runs the rule-based visibility pre-pass and
    builds the Mode-A image menu for ONE object.

    args = (mesh_npz, image_npz, overview_path, anno_obj_dir_str)
    The overview PNG is expected to already exist on disk (rendered by the
    ``backfill_overviews`` pre-pass that runs before any worker fires).

    Returns ``(user_msg, visible_pids, quota, menu)`` on success, or a string
    skip-reason (``"too_many_parts"`` / ``"no_overview"`` / ``"none_visible"``).
    """
    import numpy as _np
    import cv2 as _cv2
    from pathlib import Path as _P
    from partcraft.pipeline_v3.qc_rules import count_part_pixels_in_overview, _N_VIEWS

    mesh_p = _P(args[0]); img_p = _P(args[1]); ov_p = _P(args[2])
    anno_p = _P(args[3]) if args[3] else None

    all_pids, names = _pid_names(mesh_p, img_p, anno_p)
    if len(all_pids) > MAX_PARTS:
        return "too_many_parts"

    if not ov_p.is_file() or ov_p.stat().st_size < 1000:
        return "no_overview"
    ov_img = _cv2.imdecode(_np.frombuffer(ov_p.read_bytes(), _np.uint8), _cv2.IMREAD_COLOR)
    if ov_img is None:
        return "no_overview"

    # Rule-based per-part visibility: best (max) pixel count across the 5 views.
    pixel_counts: dict[int, int] = {}
    for pid in all_pids:
        best = 0
        for v in range(_N_VIEWS):
            c = count_part_pixels_in_overview(ov_img, v, [pid])
            if c > best:
                best = c
        pixel_counts[pid] = best

    visible = [p for p in all_pids if pixel_counts[p] >= VIS_MIN_PIXELS]
    dropped = [p for p in all_pids if pixel_counts[p] < VIS_MIN_PIXELS]

    # White-model (白模) detection: untextured mesh → no real colour/material to
    # see, so the VLM can only hallucinate material/color edits.  Downgrade by
    # zeroing those two quotas; geometry/structure edits are unaffected.
    cf = white_model_colorful_frac(ov_img)
    is_white = cf < WHITE_MODEL_COLORFUL_FRAC

    # Audit record next to the overview — ties back to the zero_visible analysis.
    try:
        (ov_p.parent / "visibility.json").write_text(json.dumps({
            "all_part_ids": all_pids,
            "visible_part_ids": visible,
            "dropped_part_ids": dropped,
            "pixel_counts": pixel_counts,
            "vis_min_pixels": VIS_MIN_PIXELS,
            "white_model": is_white,
            "colorful_frac": round(cf, 4),
        }, indent=2))
    except Exception:
        pass

    if not visible:
        return "none_visible"

    menu = _image_menu(visible, names)
    # Generation always uses the FULL quota — produce ALL edit types ONCE so we
    # never have to re-run gen_edits when more types are enabled downstream.
    # The active allow-list (qc.edit_types) gates only PROCESSING, not here.
    quota = compute_edit_quota(len(visible))
    if is_white:
        # 白模降级: skip material/color (would be pure guesses on an untextured mesh).
        quota = {**quota, "material": 0, "color": 0}
    user_msg = _format_user_prompt(menu, quota, seed=hash(mesh_p.stem))
    if is_white:
        user_msg = (
            "[NOTE: This is an UNTEXTURED white/grey model — it has no real "
            "colours or materials. Reason from SHAPE only; do NOT infer or "
            "mention specific colours/materials in any field.]\n\n"
        ) + user_msg
    return user_msg, visible, quota, menu


# ───────────────────── single-object VLM call (image) ────────────────────────

async def _call_vlm_for_one(client, ctx: ObjectContext, user_msg: str,
                            valid_pids: list[int], quota: dict, model: str,
                            sem: asyncio.Semaphore,
                            part_menu: str = "") -> Phase1Result:
    """Up to 2 image+text VLM attempts for one object (Mode A).

    Identical control flow to gen_edits._call_vlm_for_one, except the VLM is
    handed the overview PNG via ``call_vlm_image_async`` + ``SYSTEM_PROMPT_A``.
    """
    async with sem:
        t0 = time.time()
        last_error: str = ""

        try:
            ov_png = ctx.overview_path.read_bytes()
        except Exception as e:  # overview vanished between pre-pass and call
            update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                        error=f"overview_read:{e}")
            return Phase1Result(ctx.obj_id, ok=False, error="no_overview")

        for attempt in range(2):
            eff_quota = quota if attempt == 0 else _halve_edit_quota(quota)
            eff_msg = user_msg if attempt == 0 else _format_retry_prompt(part_menu, eff_quota)

            try:
                raw = await call_vlm_image_async(
                    client, ov_png, SYSTEM_PROMPT_A, eff_msg, model, max_tokens=12288,
                )
            except Exception as e:
                last_error = str(e)
                if attempt == 0:
                    continue
                update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                            error=last_error, attempts=2)
                return Phase1Result(ctx.obj_id, ok=False, error=last_error)

            ctx.raw_response_path.write_text(raw)
            parsed = extract_json_object(raw)
            if parsed is None:
                last_error = "parse_error"
                if attempt == 0:
                    continue
                update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                            error=last_error, raw_len=len(raw), attempts=2)
                return Phase1Result(ctx.obj_id, ok=False, error=last_error)

            # Normalize: VLM sometimes nests edits inside object rather than
            # as a top-level sibling.  Hoist to the expected schema.
            if "edits" not in parsed and isinstance(
                (parsed.get("object") or {}).get("edits"), list
            ):
                parsed["edits"] = parsed["object"].pop("edits")

            dt = time.time() - t0
            rep = validate_edit_json(parsed, set(valid_pids), quota=eff_quota)
            out = {
                "obj_id": ctx.obj_id,
                "shard": ctx.shard,
                "validation": rep,
                "parsed": parsed,
            }
            ctx.parsed_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            update_step(
                ctx, "s1_phase1",
                status=STATUS_OK if rep["ok"] else STATUS_FAIL,
                n_edits=len(parsed.get("edits") or []),
                n_kept=rep["n_kept_edits"],
                type_counts=rep.get("type_counts"),
                wall_s=round(dt, 2),
                attempts=attempt + 1,
                mode="image_semantic",
            )
            return Phase1Result(
                ctx.obj_id, ok=rep["ok"], n_kept=rep["n_kept_edits"],
                n_total=sum(eff_quota.values()),
                type_counts=rep.get("type_counts"),
            )

        update_step(ctx, "s1_phase1", status=STATUS_FAIL,
                    error=last_error or "unknown", attempts=2)
        return Phase1Result(ctx.obj_id, ok=False, error=last_error)


# ───────────────────── public entrypoints ────────────────────────────────────

def gen_edits_for_one(
    ctx: ObjectContext,
    *,
    blender: str,
    vlm_url: str,
    vlm_model: str,
) -> Phase1Result:
    """Synchronous one-object run (single VLM server) — debug helper."""
    from openai import AsyncOpenAI

    # Render overview up front (it is the VLM input in Mode A).
    backfill_overviews([ctx], blender=blender, n_workers=1, force=False)
    pre = _prepare_edit_menu_image_worker(
        (str(ctx.mesh_npz), str(ctx.image_npz), str(ctx.overview_path), "")
    )
    if isinstance(pre, str):
        update_step(ctx, "s1_phase1", status=STATUS_SKIP, reason=pre)
        return Phase1Result(ctx.obj_id, ok=False, error=pre)
    user_msg, pids, quota, menu = pre

    async def _go():
        client = AsyncOpenAI(base_url=vlm_url, api_key="EMPTY")
        sem = asyncio.Semaphore(1)
        return await _call_vlm_for_one(client, ctx, user_msg, pids, quota,
                                       vlm_model, sem, part_menu=menu)

    return asyncio.run(_go())


async def gen_edits_streaming(
    ctxs: Iterable[ObjectContext],
    *,
    blender: str,
    vlm_urls: list[str],
    vlm_model: str,
    n_prerender_workers: int = 8,
    force: bool = False,
    log_every: int = 20,
    post_object_fn=None,
    anno_dir: "Path | None" = None,
) -> list[Phase1Result]:
    """Producer-consumer streaming s1 (Mode A / image_semantic).

    Same orchestration as gen_edits.gen_edits_streaming, with two differences:
      * the overview backfill is mandatory (it is the VLM input, not just a
        Gate-A aid) — objects without an overview are skipped;
      * each prerender worker runs the visibility pre-pass + builds the image
        menu, and the consumer calls the image VLM.

    Resume rule unchanged: any obj with ``parsed.json`` already on disk is
    skipped.
    """
    from openai import AsyncOpenAI
    import logging
    log = logging.getLogger("pipeline_v3.s1.stream.image")

    ctxs = list(ctxs)
    todo: list[ObjectContext] = []
    results: list[Phase1Result] = []

    for ctx in ctxs:
        if not force and ctx.parsed_path.is_file():
            try:
                _j = json.loads(ctx.parsed_path.read_text())
                _p = _j.get("parsed") or {}
                if "edits" not in _p and isinstance((_p.get("object") or {}).get("edits"), list):
                    _p["edits"] = _p["object"].pop("edits")
                    _j["parsed"] = _p
                    ctx.parsed_path.write_text(json.dumps(_j, indent=2, ensure_ascii=False))
                if (_j.get("parsed") or {}).get("edits") is not None:
                    if not step_done(ctx, "s1_phase1"):
                        update_step(
                            ctx, "s1_phase1", status=STATUS_OK,
                            n_edits=len(_j["parsed"].get("edits") or []),
                            resumed=True,
                        )
                    results.append(Phase1Result(ctx.obj_id, ok=True))
                    continue
            except Exception:
                pass
        if not force and step_done(ctx, "s1_phase1"):
            results.append(Phase1Result(ctx.obj_id, ok=True))
            continue
        if ctx.mesh_npz is None or ctx.image_npz is None:
            results.append(Phase1Result(ctx.obj_id, ok=False, error="no_input"))
            continue
        todo.append(ctx)

    log.info("s1 streaming (image): todo=%d resume=%d  vlm_servers=%d  workers=%d  vis_min_px=%d",
             len(todo), len(results), len(vlm_urls), n_prerender_workers, VIS_MIN_PIXELS)

    # ── Overview render pre-pass (mandatory in Mode A) ────────────────────────
    # Renders + saves phase1/overview.png for every ctx before any VLM call.
    # This is the segmentation map the VLM reads AND the image the visibility
    # pre-pass measures.  Idempotent (skips existing files).
    try:
        ov_stats = backfill_overviews(
            ctxs, blender=blender, n_workers=n_prerender_workers,
            force=False, log=log,
        )
        log.info("s1 streaming (image): overview render done — ok=%d skip=%d err=%d",
                 ov_stats["ok"], ov_stats["skip"], ov_stats["err"])
    except Exception as exc:
        log.warning("overview render failed: %s — objects without overview will skip", exc)

    if not todo:
        return results

    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=n_prerender_workers)
    queue: asyncio.Queue = asyncio.Queue(maxsize=2 * len(vlm_urls))

    clients = [AsyncOpenAI(base_url=u, api_key="EMPTY") for u in vlm_urls]
    sems = [asyncio.Semaphore(1) for _ in clients]

    n_done = 0
    n_total = len(todo)

    async def build_one(ctx: ObjectContext):
        try:
            _anno_dir = (anno_dir / ctx.obj_id) if anno_dir else None
            pre = await loop.run_in_executor(
                pool, _prepare_edit_menu_image_worker,
                (str(ctx.mesh_npz), str(ctx.image_npz), str(ctx.overview_path),
                 str(_anno_dir) if _anno_dir else ""),
            )
        except Exception as e:
            log.warning("build_menu %s: %s", ctx.obj_id[:12], e)
            update_step(ctx, "s1_phase1", status=STATUS_FAIL, error=str(e))
            return
        if isinstance(pre, str):  # skip reason
            update_step(ctx, "s1_phase1", status=STATUS_SKIP, reason=pre)
            return
        ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
        await queue.put((ctx, pre))

    async def producer():
        sem = asyncio.Semaphore(n_prerender_workers * 2)

        async def _wrap(c):
            async with sem:
                await build_one(c)

        await asyncio.gather(*[_wrap(c) for c in todo])
        for _ in range(len(clients)):
            await queue.put(None)

    async def consumer(idx: int):
        nonlocal n_done
        client = clients[idx]
        sem = sems[idx]
        while True:
            item = await queue.get()
            if item is None:
                return
            ctx, pre = item
            user_msg, pids, quota, menu = pre
            res = await _call_vlm_for_one(client, ctx, user_msg, pids, quota,
                                          vlm_model, sem, part_menu=menu)
            results.append(res)
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                log.info("s1 stream (image): %d/%d  ok_so_far=%d",
                         n_done, n_total, sum(1 for r in results if r.ok))
            if post_object_fn is not None and res.error != "too_many_parts":
                try:
                    await post_object_fn(ctx, vlm_urls[idx])
                except Exception as _hook_exc:
                    log.warning("post_object_fn %s: %s", ctx.obj_id[:12], _hook_exc)

    try:
        await asyncio.gather(producer(), *[consumer(i) for i in range(len(clients))])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return results


__all__ = [
    "SYSTEM_PROMPT_A", "USER_PROMPT_IMAGE_SEMANTIC", "VIS_MIN_PIXELS",
    "gen_edits_for_one", "gen_edits_streaming",
]
