#!/usr/bin/env python3
"""Render before / after for each 3D edit in ONE shared frame and build a
comparison sheet.

Both meshes are rendered in the **partverse (full.glb) world frame** at the
edit's real best-view camera (the same camera that produced the 2D input the
FLUX edit was based on), so the 3D panels line up with the 2D panels:

    [ 2D input | 2D FLUX edit | 3D before (full.glb) | 3D after ]

  * before = the original ``full.glb`` (already in the partverse frame).
  * after  = ``after.glb`` rigidly reframed from TRELLIS's export frame back
             into the partverse frame (undo the Y↔Z export swap + ×scale).
             This mirrors ``trellis2_3d._partverse_reframe_matrix`` so existing
             after.glb files (exported in the old TRELLIS frame) display
             correctly without re-running the GPU pipeline.

after.glb uses full.glb as ``--ref_object`` so scale edits visibly change size.
One PNG per edit plus a stacked contact sheet.

Run under an env with a working Pillow (e.g. pipeline_server):
    BLENDER_PATH=/path/to/blender RENDER_CUDA=0 \
    /mnt/zsn/miniconda3/envs/pipeline_server/bin/python \
        scripts/standalone/render_before_after_compare.py \
        --root data/Pxform_v2 --shard 08 \
        --ids configs/_gen/smoke10_shard08_ids.txt
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.pipeline_v3.preview_render import (  # noqa: E402
    _best_view_slot_for_edit, _read_camera_views_from_npz,
    _render_glb_views, _encode_asset_script,
)
from partcraft.pipeline_v3.paths import PipelineRoot  # noqa: E402

BLENDER_SCRIPT = str(_ROOT / "scripts" / "blender_render.py")
PANEL = 384
PAD = 8
CAP_H = 54
DEFAULT_SLOT = 4  # fallback best-view slot when edit_status has none

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Inverse of the TRELLIS export rotation: export → o-voxel frame ((x,y,z)→
# (x,-z,y) = R_X90).  Combined with /scale + center it sends after.glb back to
# the full.glb world frame.  Mirrors trellis2_3d._partverse_reframe_matrix.
_R_X90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)


def _concat(g):
    import trimesh
    if isinstance(g, trimesh.Scene):
        return trimesh.util.concatenate(
            [m for m in g.geometry.values() if isinstance(m, trimesh.Trimesh)])
    return g


def _full_center_scale(mesh_npz: Path):
    import trimesh
    d = np.load(str(mesh_npz), allow_pickle=True)
    g = _concat(trimesh.load(io.BytesIO(d["full.glb"].tobytes()),
                             file_type="glb", process=False))
    v = np.asarray(g.vertices, dtype=np.float64)
    vmin, vmax = v.min(0), v.max(0)
    return (vmin + vmax) / 2.0, float(0.99999 / (vmax - vmin).max())


def _extract_full_glb(mesh_npz: Path, dst: Path):
    """Write the original whole-object full.glb (partverse frame) to dst."""
    if not mesh_npz.is_file():
        return None
    try:
        import trimesh
        g = _concat(trimesh.load(
            io.BytesIO(np.load(str(mesh_npz), allow_pickle=True)["full.glb"].tobytes()),
            file_type="glb", process=False))
        g.export(str(dst))
        return dst
    except Exception as e:
        sys.stderr.write(f"[compare] full.glb extract failed {mesh_npz}: {e}\n")
        return None


def _reframe_after_to_partverse(after_glb: Path, mesh_npz: Path, dst: Path):
    """after.glb (TRELLIS export frame) → full.glb world frame → dst."""
    if not after_glb.is_file() or not mesh_npz.is_file():
        return None
    try:
        import trimesh
        center, scale = _full_center_scale(mesh_npz)
        m = np.eye(4)
        m[:3, :3] = _R_X90 / scale
        m[:3, 3] = center
        g = _concat(trimesh.load(str(after_glb), process=False))
        g.apply_transform(m)
        g.export(str(dst))
        return dst
    except Exception as e:
        sys.stderr.write(f"[compare] after reframe failed {after_glb}: {e}\n")
        return None


def _font(sz: int, bold: bool = False):
    p = FONT_BOLD if bold else FONT_PATH
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def _render_prerender(glb: Path, image_npz: Path, slot: int, blender: str,
                      resolution: int, cuda: str | None):
    """Render GLB at one canonical slot via the SAME path/normalization the 2D
    prerender used (encode_asset render + image_npz ``--normalize_scale/offset``
    = vd_scale 0.5005 → object at [-0.5,0.5]).  This makes the 3D panels match
    the 2D panels in scale AND pose, instead of blender_render.py's [-1,1] fit
    (which rendered the object 2× too big → clipped / "not fully visible").
    Returns white-bg RGB PIL.
    """
    try:
        out = _render_glb_views(
            glb, image_npz, _encode_asset_script(), blender, resolution,
            view_slots=[slot], cuda_device=cuda)
    except Exception as e:
        sys.stderr.write(f"[render] FAIL {glb.name}: {e}\n")
        return None
    bgr = out.get(slot)
    if bgr is None:
        return None
    return Image.fromarray(bgr[:, :, ::-1])  # BGR→RGB (already white-composited)


def _load_png(p: Path):
    return Image.open(str(p)).convert("RGB") if p.is_file() else None


def _fit(img, size: int):
    canvas = Image.new("RGB", (size, size), (242, 242, 242))
    if img is None:
        d = ImageDraw.Draw(canvas)
        d.text((size // 2 - 6, size // 2 - 8), "—", fill=(150, 150, 150), font=_font(28))
        return canvas
    img = img.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def _prompt_for(parsed: dict, edit_id: str) -> str:
    p = parsed.get("parsed") if isinstance(parsed, dict) else None
    edits = p.get("edits") if isinstance(p, dict) else None
    if isinstance(edits, list):
        # edit ids are <type>_<obj>_<NNN>; match by ordinal+type best-effort
        for e in edits:
            pr = e.get("prompt") or e.get("instruction") or ""
            if pr and e.get("edit_type", "")[:3] == edit_id.split("_")[0][:3]:
                return pr[:150]
    return ""


def _compose_row(obj_id, edit_id, etype, prompt, panels) -> Image.Image:
    labels = ["2D input", "2D FLUX edit", "3D before", "3D after"]
    n = len(panels)
    W = n * PANEL + (n + 1) * PAD
    H = PANEL + 2 * PAD + CAP_H + 22
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 6), f"{obj_id[:12]}  |  {edit_id}  |  {etype}",
           fill=(0, 0, 0), font=_font(18, bold=True))
    if prompt:
        d.text((PAD, 30), prompt, fill=(70, 70, 70), font=_font(14))
    y0 = CAP_H
    for i, (img, lab) in enumerate(zip(panels, labels)):
        x = PAD + i * (PANEL + PAD)
        sheet.paste(_fit(img, PANEL), (x, y0))
        d.text((x + 4, y0 + PANEL + 2), lab, fill=(40, 40, 40), font=_font(14))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--blender", default=os.environ.get("BLENDER_PATH", "blender"))
    ap.add_argument("--cuda-device", default=os.environ.get("RENDER_CUDA", None))
    args = ap.parse_args()

    root = Path(args.root)
    proot = PipelineRoot(root=root)
    out_dir = Path(args.out) if args.out else (root / "_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = [ln.strip() for ln in Path(args.ids).read_text().splitlines()
           if ln.strip() and not ln.startswith("#")]
    print(f"[compare] {len(ids)} objects  blender={args.blender}  cuda={args.cuda_device}")

    rows = []
    n_edits = 0
    with tempfile.TemporaryDirectory(prefix="cmp_render_") as _tmp:
        tmp = Path(_tmp)
        for obj_id in ids:
            obj_dir = root / "objects" / args.shard / obj_id
            edits_3d = obj_dir / "edits_3d"
            edits_2d = obj_dir / "edits_2d"
            if not edits_3d.is_dir():
                print(f"[compare] {obj_id}: no edits_3d, skip")
                continue
            mesh_npz = Path(args.mesh_root) / args.shard / f"{obj_id}.npz"
            image_npz = Path(args.images_root) / args.shard / f"{obj_id}.npz"
            ctx = proot.context(args.shard, obj_id,
                                mesh_npz=mesh_npz, image_npz=image_npz)
            parsed = {}
            if ctx.parsed_path.is_file():
                try:
                    parsed = json.loads(ctx.parsed_path.read_text())
                except Exception:
                    pass

            # before = original full.glb (already partverse frame)
            before_glb = _extract_full_glb(mesh_npz, tmp / f"{obj_id}_full.glb")

            for edit_dir in sorted(edits_3d.iterdir()):
                if not edit_dir.is_dir():
                    continue
                after = edit_dir / "after.glb"
                if not after.is_file():
                    continue
                edit_id = edit_dir.name
                etype = edit_id.split("_")[0]

                # per-edit best-view canonical slot (matches the 2D input)
                try:
                    slot = _best_view_slot_for_edit(ctx, edit_id, default=DEFAULT_SLOT)
                except Exception:
                    slot = DEFAULT_SLOT

                after_pv = _reframe_after_to_partverse(
                    after, mesh_npz, tmp / f"{edit_id}_after_pv.glb")

                b_pil = None
                if before_glb is not None:
                    b_pil = _render_prerender(before_glb, image_npz, slot,
                                              args.blender, args.resolution,
                                              args.cuda_device)
                a_pil = None
                if after_pv is not None:
                    a_pil = _render_prerender(after_pv, image_npz, slot,
                                              args.blender, args.resolution,
                                              args.cuda_device)

                inp = _load_png(edits_2d / f"{edit_id}_input.png")
                ed = _load_png(edits_2d / f"{edit_id}_edited.png")
                row = _compose_row(obj_id, edit_id, etype,
                                   _prompt_for(parsed, edit_id),
                                   [inp, ed, b_pil, a_pil])
                rpath = out_dir / f"{obj_id}__{edit_id}.png"
                row.save(str(rpath))
                rows.append(row)
                n_edits += 1
                print(f"[compare] wrote {rpath.name}  (slot {slot})")

    if rows:
        W = max(r.width for r in rows)
        H = sum(r.height for r in rows) + PAD * (len(rows) + 1)
        sheet = Image.new("RGB", (W, H), (250, 250, 250))
        y = PAD
        for r in rows:
            sheet.paste(r, (0, y)); y += r.height + PAD
        sp = out_dir / "_contact_sheet.png"
        sheet.save(str(sp))
        print(f"[compare] contact sheet: {sp}  ({n_edits} edits)")
    else:
        print("[compare] no after.glb found")


if __name__ == "__main__":
    main()
