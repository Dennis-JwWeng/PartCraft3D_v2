"""Step ``del_add`` — v2-native deletion + inverse-addition latent re-encode (GPU).

The active edit pipeline only generates mod/scale (``qc.edit_types``).  Deletion
specs DO exist in ``phase1/parsed.json`` (prompt + ``selected_part_ids``) but,
on the mod/scale-only shards, their ``gate_a`` is recorded ``deferred`` — so
``specs.iter_deletion_specs`` skips them and ``run_deletion_batch`` /
``link_slat_assets_batch`` (the v1 Blender+DINOv2 path) are inert.  This stage
therefore sources deletions DIRECTLY from ``parsed.json`` and produces, per
deletion spec, a matched **del + inverse-add** pair of TRELLIS.2 latents written
into the prod object tree exactly like ``trellis2_3d`` writes mod/scale::

    edits_3d/<del_id>/latents/{shape_slat,tex_slat,ss}.npz   after = deleted
    edits_3d/<add_id>/latents/{shape_slat,tex_slat,ss}.npz   after = original
    edits_3d/<*_id>/after_view_<name>.png                    5 named PBR views
    edit_status.json  edits.<id>.stages.del_add = {status:done, verdict:{best_view}}

Per object:
  * merge the surviving parts into a TEMP ``after_new.glb`` (tempdir — NEVER
    persisted) via ``mesh_deletion._merge_surviving_parts_from_npz``;
  * encode shape+tex SLat @512 (32³) in the ORIGINAL mesh's frame so the deleted
    coords share the e512 'before' grid (``encode_after_512``);
  * build the ss grid (pad=4 → keep16/edit_grid) reproducing prod
    (``part_struct_grids``);
  * deletion: before=original e512, after=deleted; addition: the inverse
    (before=deleted, after=original; prompt inverted via ``invert_delete_prompt``).

best_view = overview-pixel argmax (deletion never ran gate_a → no VLM best_view;
argmax needs only ``phase1/overview.png`` + ``selected_part_ids`` which always
exist).  GPU stage, single-GPU per worker; the orchestrator slices objects via
``dispatch_gpus`` (CUDA_VISIBLE_DEVICES children + ``--gpu-shard k/N`` round-robin),
identical to ``trellis2_encode``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from .paths import ObjectContext
from . import services_cfg as psvc
from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP
from .edit_status_io import update_edit_stage, flush_edit_status

from partcraft.render.ovox_views import VIEW_ORDER

S1_PAD = 4          # matches config trellis2_s1_pad
S1_THRESH = 0.1     # matches config s1_thresh
_LAT_FILES = ("shape_slat.npz", "tex_slat.npz", "ss.npz", "ss_latent.npz", "mask.npz")


@dataclass
class DelAddResult:
    obj_id: str
    n_del: int = 0
    n_add: int = 0
    n_fail: int = 0
    n_skip: int = 0
    error: str | None = None


# ───────────────────────── kernel (lifted from export script) ─────────────────────────
def encode_after_512(encoders, orig_mesh_npz, after_glb, canonical, grid_size=512):
    """shape+tex SLat (@32³) + after SS latent (@16³) for ``after_new.glb``, normalized
    with the ORIGINAL mesh's M so the deleted-result coords share the same 32³ grid as
    the e512 'before'.  The after SS latent = ``ss_enc`` of the deleted mesh's 64³
    occupancy (byte-compatible with ``p1_encode/ss.npz``); it is del.after = add.before's
    condition latent — the S1 training target for the deleted state."""
    import trimesh, torch, o_voxel
    import trellis2.modules.sparse as sp
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    _, M = OVR._normalized_groups(OVR.load_full_scene(Path(orig_mesh_npz)), canonical=canonical)
    scene = trimesh.load(str(after_glb), file_type="glb", process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    groups, _ = OVR._normalized_groups(scene, canonical=canonical, M=M)
    merged = trimesh.util.concatenate(groups)
    verts = torch.from_numpy(np.asarray(merged.vertices)).float()
    faces = torch.from_numpy(np.asarray(merged.faces)).long()
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        verts.cpu(), faces.cpu(), grid_size=grid_size, aabb=aabb,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dual_local = (dv * grid_size - vi).clamp(0., 1.).float()
    if inter.dim() == 2 and inter.shape[1] == 3:
        inter3 = inter.float()
    else:
        b = inter.view(-1).to(torch.uint8)
        inter3 = torch.stack([(b & 1).bool(), ((b >> 1) & 1).bool(), ((b >> 2) & 1).bool()], -1).float()
    sh_coords = torch.cat([torch.zeros_like(vi[:, :1]), vi], -1).int()
    vsp = sp.SparseTensor(feats=dual_local, coords=sh_coords).cuda()
    isp = vsp.replace(inter3.bool().float().cuda())
    with torch.no_grad():
        shape_slat = encoders["shape"](vsp, isp)
    coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        trimesh.Scene(groups), grid_size=grid_size, aabb=aabb)
    def _f(x): return (x.float() / 255.0) if x.dtype == torch.uint8 else x.float()
    feats6 = torch.cat([_f(attr["base_color"]), _f(attr["metallic"]),
                        _f(attr["roughness"]), _f(attr["alpha"])], -1).float()
    tx_coords = torch.cat([torch.zeros_like(coord[:, :1]), coord], -1).int()
    tsp = sp.SparseTensor(feats=feats6, coords=tx_coords).cuda()
    with torch.no_grad():
        tex_slat = encoders["tex"](tsp)
    # after SS latent: ss_enc of the 64³ occupancy, the SAME path
    # trellis2_encode.encode_shape_tex_ss uses for p1_encode/ss.npz → [1,8,16³].
    # The 64³ occupancy = the SHAPE-ENCODER OUTPUT coords at grid_size=1024 (the
    # encoder compresses the fine dual grid 1024→64³), NOT the raw voxelization
    # (raw vi is on the fine 0..1023 grid → would index occ[64] out of range).
    vi1, dv1, inter1 = o_voxel.convert.mesh_to_flexible_dual_grid(
        verts.cpu(), faces.cpu(), grid_size=1024, aabb=aabb,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dual1 = (dv1 * 1024 - vi1).clamp(0., 1.).float()
    if inter1.dim() == 2 and inter1.shape[1] == 3:
        inter3_1 = inter1.float()
    else:
        b1 = inter1.view(-1).to(torch.uint8)
        inter3_1 = torch.stack([(b1 & 1).bool(), ((b1 >> 1) & 1).bool(),
                                ((b1 >> 2) & 1).bool()], -1).float()
    sh_coords_1 = torch.cat([torch.zeros_like(vi1[:, :1]), vi1], -1).int()
    vsp1 = sp.SparseTensor(feats=dual1, coords=sh_coords_1).cuda()
    isp1 = vsp1.replace(inter3_1.bool().float().cuda())
    with torch.no_grad():
        shape_slat_64 = encoders["shape"](vsp1, isp1)
    c64 = shape_slat_64.coords[:, 1:].long()
    occ = torch.zeros(1, 1, 64, 64, 64, device=shape_slat_64.device)
    occ[0, 0, c64[:, 0], c64[:, 1], c64[:, 2]] = 1.0
    with torch.no_grad():
        z_ss = encoders["ss"](occ.float())
    return {
        "shape_feats": shape_slat.feats.cpu().numpy().astype(np.float32),
        "shape_coords": shape_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
        "tex_feats": tex_slat.feats.cpu().numpy().astype(np.float32),
        "tex_coords": tex_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
        "ss_latent": z_ss.detach().cpu().numpy().astype(np.float32),
    }


def part_struct_grids(mesh_npz, pids, canonical, *, s1_pad=S1_PAD, s1_thresh=S1_THRESH):
    """Pipeline-exact part region: 64³ edit grid (pad=4) → 16³ keep mask +
    32³ edit_grid coords.  Reproduces prod ss.npz keep16/edit_grid (validated IoU=1)."""
    import torch
    from partcraft.pipeline_v3.trellis2_part_mask import (
        part_edit_grid_64, edit_grid_64_to_keep16, downsample_edit_grid)
    g64 = part_edit_grid_64(mesh_npz, pids, pad=s1_pad, canonical=canonical)
    keep16 = edit_grid_64_to_keep16(g64, thresh=s1_thresh).cpu().numpy().astype(bool)
    g32 = downsample_edit_grid(g64, 2)
    edit_grid = torch.nonzero(g32).to(torch.int16).cpu().numpy()
    return keep16, edit_grid


def mask_from_ss(ss):
    """v1-style keep masks from an ss dict (coords0, coords_new, edit_grid, keep16, parts)."""
    eg = {tuple(int(x) for x in c) for c in np.asarray(ss["edit_grid"]).tolist()}
    def keep(coords):
        return np.fromiter((tuple(int(x) for x in c) not in eg
                            for c in np.asarray(coords).tolist()), dtype=np.uint8,
                           count=len(coords))
    return {
        "mask_keep_ss": np.asarray(ss["keep16"]).astype(np.uint8),
        "mask_keep_slat": keep(ss["coords_new"]),
        "mask_keep_slat_before": keep(ss["coords0"]),
        "selected_part_ids": np.asarray(ss["parts"], dtype=np.int32),
    }


def compute_best_view(overview_path, selected_part_ids):
    """argmax of per-view selected-part pixels over phase1/overview.png (CPU, no VLM)."""
    try:
        import cv2
        from partcraft.pipeline_v3.qc_rules import count_part_pixels_in_overview, _N_VIEWS
        if not Path(overview_path).is_file() or not selected_part_ids:
            return 0, VIEW_ORDER[0]
        ov = cv2.imdecode(np.frombuffer(Path(overview_path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        px = [count_part_pixels_in_overview(ov, v, selected_part_ids) for v in range(_N_VIEWS)]
        bv = int(np.argmax(px)) if any(p > 0 for p in px) else 0
        return bv, VIEW_ORDER[bv]
    except Exception as e:  # noqa: BLE001
        logging.getLogger("pipeline_v3.del_add").warning("best_view failed: %s", e)
        return 0, VIEW_ORDER[0]


def load_parsed(prod_obj_dir):
    """Read parsed.json → (object full_desc, [deletion specs], [non-deletion specs])."""
    d = json.loads((Path(prod_obj_dir) / "phase1" / "parsed.json").read_text())
    p = d.get("parsed") or {}
    edits = p.get("edits") or []
    obj_desc = (p.get("object") or {}).get("full_desc", "")
    dels = [e for e in edits if e.get("edit_type") == "deletion"]
    non_dels = [e for e in edits if e.get("edit_type") != "deletion"]
    return obj_desc, dels, non_dels


# ───────────────────────── prod writers (mirror trellis2_3d) ─────────────────────────
def _slat_from_arrays(feats, coords):
    import torch
    import trellis2.modules.sparse as sp
    f = torch.from_numpy(np.ascontiguousarray(feats)).float().cuda()
    c = torch.from_numpy(np.ascontiguousarray(coords)).int()
    c = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.int32), c], 1).cuda()
    return sp.SparseTensor(feats=f, coords=c)


def _write_del_add_latents(edit_dir, shape_after, tex_after, ss_dict, logger,
                           *, ss_latent=None, write_mask=True):
    """Write edit_dir/latents/{ss,shape_slat,tex_slat,ss_latent,mask}.npz in the SAME
    schema as trellis2_3d._save_edit_latents (so del/add sit next to mod/scale
    uniformly).  ``ss_latent`` ([1,8,16,16,16]) is the after SS latent; ``mask.npz`` is
    the pure-function keep mask from the ss region pack."""
    from partcraft.pipeline_v3.trellis2_3d import _save_edit_latents
    latents = {
        "coords0": ss_dict["coords0"],
        "coords_new": ss_dict["coords_new"],
        "edit_type": ss_dict["edit_type"],
        "parts": ss_dict["parts"],
        "edit_grid": ss_dict["edit_grid"],
        "keep16": ss_dict["keep16"],
        "s1_pad": ss_dict["s1_pad"],
        "s1_thresh": ss_dict["s1_thresh"],
        "shape_feats": shape_after["feats"], "shape_coords": shape_after["coords"],
        "tex_feats": tex_after["feats"], "tex_coords": tex_after["coords"],
        "ss_latent": ss_latent,   # after SS latent → latents/ss_latent.npz
    }
    _save_edit_latents(latents, edit_dir, logger)
    if write_mask:
        np.savez_compressed(edit_dir / "latents" / "mask.npz", **mask_from_ss(ss_dict))


def _latents_complete(edit_dir: Path) -> bool:
    out = edit_dir / "latents"
    return all((out / f).is_file() and (out / f).stat().st_size > 0 for f in _LAT_FILES)


def _object_has_pending(ctx: ObjectContext, force: bool) -> bool:
    """True iff this object has deletion specs whose del/add latents still need
    (re)writing — used to lazy-load the GPU models only when there is real work,
    so a full-resume shard pays no model-load cost (mirrors trellis2_encode.run)."""
    try:
        _, dels, _ = load_parsed(ctx.dir)
    except Exception:  # noqa: BLE001 — missing/garbage parsed.json → nothing to do
        return False
    if not dels:
        return False
    if force:
        return True
    for seq in range(len(dels)):
        del_dir = ctx.edit_3d_dir(ctx.edit_id("deletion", seq))
        add_dir = ctx.edit_3d_dir(ctx.edit_id("addition", seq))
        if not (_latents_complete(del_dir) and _latents_complete(add_dir)):
            return True
    return False


# ───────────────────────── addition meta shim ─────────────────────────
@dataclass
class _DelSpecShim:
    """Adapt a parsed.json deletion dict to the attribute API _write_addition_meta needs."""
    edit_id: str
    selected_part_ids: list
    view_index: int
    prompt: str
    target_part_desc: str
    object_desc: str
    part_labels: list = field(default_factory=list)


# ───────────────────────── per-object ─────────────────────────
def run_del_add_for_object(ctx: ObjectContext, *, encoders, pipeline, p25_cfg,
                           canonical, render_after_views, write_mask, force, logger):
    from partcraft.pipeline_v3.mesh_deletion import (
        _merge_surviving_parts_from_npz, _write_addition_meta)
    res = DelAddResult(ctx.obj_id)

    # Deletions come from parsed.json directly (iter_deletion_specs skips deferred).
    obj_desc, dels_parsed, _ = load_parsed(ctx.dir)
    if not dels_parsed:
        update_step(ctx, "s_del_add", status=STATUS_OK, n=0, reason="no_deletions")
        return res

    p1 = ctx.dir / "p1_encode"
    sh_e512 = p1 / "shape_slat_e512.npz"
    tx_e512 = p1 / "tex_slat_e512.npz"
    if not (sh_e512.is_file() and tx_e512.is_file()):
        update_step(ctx, "s_del_add", status=STATUS_FAIL, error="missing e512 sidecars")
        res.error = "missing e512 sidecars"; res.n_fail = len(dels_parsed); return res
    orig_sh = dict(np.load(sh_e512))   # feats, coords (32³) = del.before / add.after
    orig_tx = dict(np.load(tx_e512))
    # original SS latent (p1) = del.before / add.after's SS latent; reused verbatim
    # for the addition's after (no re-encode — add.after IS the original object).
    p1_ss = p1 / "ss.npz"
    orig_ss = (np.load(p1_ss)["ss"].astype(np.float32) if p1_ss.is_file() else None)
    gate_dir = ctx.dir / "gate_views"

    for seq, spec in enumerate(dels_parsed):
        del_id = ctx.edit_id("deletion", seq)
        add_id = ctx.edit_id("addition", seq)
        del_dir = ctx.edit_3d_dir(del_id)
        add_dir = ctx.edit_3d_dir(add_id)
        if not force and _latents_complete(del_dir) and _latents_complete(add_dir):
            res.n_skip += 1
            continue
        pids = [int(x) for x in (spec.get("selected_part_ids") or [])]
        try:
            with tempfile.TemporaryDirectory() as td:
                if not _merge_surviving_parts_from_npz(ctx.mesh_npz, pids, Path(td),
                                                       force=True, logger=logger):
                    logger.warning("[del_add] %s %s merge failed", ctx.obj_id, del_id)
                    res.n_fail += 1
                    update_edit_stage(ctx, del_id, "deletion", "del_add",
                                      status="error", reason="merge_failed")
                    continue
                enc = encode_after_512(encoders, ctx.mesh_npz, Path(td) / "after_new.glb",
                                       canonical)
            keep16, edit_grid = part_struct_grids(ctx.mesh_npz, pids, canonical)
            bv, vname = compute_best_view(ctx.overview_path, pids)
            parts = np.asarray(pids, dtype=np.int32)
            sh_after_del = {"feats": enc["shape_feats"], "coords": enc["shape_coords"].astype(np.int16)}
            tx_after_del = {"feats": enc["tex_feats"], "coords": enc["tex_coords"].astype(np.int16)}
            sh_after_add = {"feats": orig_sh["feats"], "coords": orig_sh["coords"].astype(np.int16)}
            tx_after_add = {"feats": orig_tx["feats"], "coords": orig_tx["coords"].astype(np.int16)}

            # ── render after-views: ONLY the best_view (= before's selected view).
            #    del = decode deleted → render bv; add = copy original before_view_bv ──
            if render_after_views:
                _render_del_after_views(pipeline, p25_cfg, enc, del_dir, vname, logger)
                _copy_before_to_add_after_views(gate_dir, add_dir, vname, logger)

            # ── deletion: before = original e512, after = deleted ──
            #    after SS latent = ss_enc(deleted 64³) (= add.before's condition latent)
            ss_del = {"coords0": orig_sh["coords"].astype(np.int16),
                      "coords_new": enc["shape_coords"].astype(np.int16),
                      "edit_grid": edit_grid, "keep16": keep16, "parts": parts,
                      "edit_type": "deletion", "s1_pad": S1_PAD, "s1_thresh": S1_THRESH}
            _write_del_add_latents(del_dir, sh_after_del, tx_after_del, ss_del, logger,
                                   ss_latent=enc.get("ss_latent"), write_mask=write_mask)
            update_edit_stage(ctx, del_id, "deletion", "del_add", status="done",
                              verdict={"best_view": bv, "view_name": vname,
                                       "view_source": "overview_part_pixel_argmax"})
            res.n_del += 1

            # ── addition: inverse (before = deleted, after = original) ──
            _write_addition_meta(ctx, _DelSpecShim(
                edit_id=del_id, selected_part_ids=pids, view_index=bv,
                prompt=spec.get("prompt", ""), target_part_desc=spec.get("target_part_desc", ""),
                object_desc=obj_desc), seq, force=force, logger=logger)
            #    after SS latent = original object's SS latent (p1_encode/ss.npz) verbatim
            ss_add = {"coords0": enc["shape_coords"].astype(np.int16),
                      "coords_new": orig_sh["coords"].astype(np.int16),
                      "edit_grid": edit_grid, "keep16": keep16, "parts": parts,
                      "edit_type": "addition", "s1_pad": S1_PAD, "s1_thresh": S1_THRESH}
            _write_del_add_latents(add_dir, sh_after_add, tx_after_add, ss_add, logger,
                                   ss_latent=orig_ss, write_mask=write_mask)
            update_edit_stage(ctx, add_id, "addition", "del_add", status="done",
                              verdict={"best_view": bv, "view_name": vname,
                                       "paired_deletion_edit_id": del_id,
                                       "view_source": "overview_part_pixel_argmax"})
            res.n_add += 1
        except Exception as e:  # noqa: BLE001 — never let one edit kill the worker's slice
            logger.exception("[del_add] %s %s failed: %s", ctx.obj_id, del_id, e)
            res.n_fail += 1
            update_edit_stage(ctx, del_id, "deletion", "del_add",
                              status="error", reason=str(e)[:200])

    update_step(ctx, "s_del_add",
                status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
                n_del=res.n_del, n_add=res.n_add, n_fail=res.n_fail, n_skip=res.n_skip)
    return res


def _render_del_after_views(pipeline, p25_cfg, enc, del_dir, vname, logger):
    """Decode the deleted latents → mesh → the ONE best-view PBR after-view in del_dir.
    del/add don't run gate-E, so only the best_view (= before's selected view) is kept."""
    from partcraft.render import ovox_views as _ov
    from partcraft.pipeline_v3.trellis2_3d import _get_envmap
    from PIL import Image
    del_dir.mkdir(parents=True, exist_ok=True)
    for old in del_dir.glob("after_view_*.png"):   # drop stale multi-view renders
        old.unlink()
    mesh = pipeline.decode_latent(
        _slat_from_arrays(enc["shape_feats"], enc["shape_coords"]),
        _slat_from_arrays(enc["tex_feats"], enc["tex_coords"]), 512)[0]
    env = _get_envmap(p25_cfg, logger)
    res = int(p25_cfg.get("trellis2_gate_view_res", 512))
    imgs = _ov.render_sample(mesh, view_names=[vname], envmap=env, resolution=res, bg=(1, 1, 1))
    for name, rgb in imgs.items():
        Image.fromarray(rgb).save(del_dir / f"after_view_{name}.png")
    logger.info("[del_add] %s rendered after-view %s", del_dir.name, vname)


def _copy_before_to_add_after_views(gate_dir: Path, add_dir: Path, vname, logger):
    """addition's after == the ORIGINAL object, already rendered at encode as
    gate_views/before_view_<bv>.png — copy ONLY the best_view to add_dir/after_view_<bv>.png."""
    import shutil
    add_dir.mkdir(parents=True, exist_ok=True)
    for old in add_dir.glob("after_view_*.png"):   # drop stale multi-view copies
        old.unlink()
    n = 0
    for v in (vname,):
        src = gate_dir / f"before_view_{v}.png"
        if src.is_file():
            shutil.copy2(src, add_dir / f"after_view_{v}.png")
            n += 1
    if n:
        logger.info("[del_add] %s ← %d add after-views (copied original before-views)",
                    add_dir.name, n)


# ───────────────────────── batch entrypoint (single GPU) ─────────────────────────
def run(
    ctxs: Iterable[ObjectContext],
    *,
    cfg: dict,
    images_root: Path | None = None,
    mesh_root: Path | None = None,
    shard: str = "01",
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[DelAddResult]:
    from partcraft.pipeline_v3.trellis2_encode import _ensure_encoders
    log = logger or logging.getLogger("pipeline_v3.del_add")
    log.info("[del_add] CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))

    p25_cfg = psvc.trellis_image_edit_flat(cfg)
    canonical = bool(p25_cfg.get("trellis2_canonical_frame", False))
    render_after_views = bool(p25_cfg.get("trellis2_render_gate_views", True))
    write_mask = bool(p25_cfg.get("del_add_write_mask", True))   # mask.npz lives in prod

    # Lazy-load the GPU models only when the first object actually needs work, so
    # a full-resume shard (everything already encoded) loads nothing.
    encoders = None
    pipeline = None

    t0 = time.time()
    n_del = n_add = n_fail = n_skip = n_bad = 0
    results: list[DelAddResult] = []
    ctxs = list(ctxs)
    # Shared bad-mesh registry: skip meshes known to hard-crash any GPU step
    # (process-fatal SIGSEGV in o_voxel voxelizer / renderer).  Under a respawn
    # supervisor the in-flight guard also detects + records new ones.
    from . import bad_mesh as _bm
    _root = ctxs[0].root if ctxs else None
    guard = _bm.make_guard(_root, shard, "del_add", log) if _root else None
    bad = guard.bad if guard is not None else (_bm.load_bad(_root) if _root else set())
    for ctx in ctxs:
        if ctx.obj_id in bad:
            log.warning("[del_add/badmesh] skipping known-bad mesh %s", ctx.obj_id)
            update_step(ctx, "s_del_add", status=STATUS_SKIP, reason="bad_mesh")
            results.append(DelAddResult(ctx.obj_id, error="bad_mesh"))
            n_bad += 1
            continue
        try:
            if encoders is None and _object_has_pending(ctx, force):
                encoders = _ensure_encoders(p25_cfg, log)
                if render_after_views:
                    from partcraft.pipeline_v3.trellis2_3d import _ensure_pipeline
                    pipeline = _ensure_pipeline(p25_cfg, log)
            if guard is not None:
                guard.beat(ctx.obj_id)
            r = run_del_add_for_object(
                ctx, encoders=encoders, pipeline=pipeline, p25_cfg=p25_cfg,
                canonical=canonical, render_after_views=render_after_views,
                write_mask=write_mask, force=force, logger=log)
        except Exception as exc:  # noqa: BLE001 — one bad object must not kill the worker
            log.exception("[del_add] %s failed with unhandled error: %s", ctx.obj_id, exc)
            update_step(ctx, "s_del_add", status=STATUS_FAIL, error=str(exc)[:200])
            r = DelAddResult(ctx.obj_id, error=str(exc))
        finally:
            if guard is not None:
                guard.clear()
        results.append(r)
        n_del += r.n_del; n_add += r.n_add; n_fail += r.n_fail; n_skip += r.n_skip
    flush_edit_status()
    log.info("[del_add] done: del=%d add=%d fail=%d skip=%d wall=%.1fs",
             n_del, n_add, n_fail, n_skip, time.time() - t0)
    return results


__all__ = ["DelAddResult", "run_del_add_for_object", "run",
           "encode_after_512", "part_struct_grids", "mask_from_ss",
           "compute_best_view", "load_parsed"]
