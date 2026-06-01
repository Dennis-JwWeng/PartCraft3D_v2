#!/usr/bin/env python3
"""Diagnostic: render after.glb brought BACK into the full.glb (partverse)
frame, at the SAME camera that produced the 2D best-view input image, so the
edit is directly comparable to the 2D input / the original mesh.

The TRELLIS export swaps Y<->Z and scales x0.5 (tips the object onto its end);
this undoes it with M_inv = inv(_PARTVERSE_TO_TRELLIS) and renders at the real
best-view camera (image_npz transforms.json), composing:

    [ 2D input | 2D FLUX edit | 3D full.glb @cam | 3D after(in full frame) @cam ]

Run under pipeline_server python (working Pillow):
    BLENDER_PATH=... /mnt/zsn/miniconda3/envs/pipeline_server/bin/python \
      scripts/standalone/diag_after_in_full_frame.py --shard 08 \
      --obj bde1b486ee284e4d94f54bdbb3b3d6d7 --edit mod_..._000 --slot 3
"""
from __future__ import annotations
import argparse, io, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from partcraft.pipeline_v3.preview_render import _read_camera_views_from_npz  # noqa

BLENDER_SCRIPT = str(_ROOT / "scripts" / "blender_render.py")

# full.glb (partverse) -> after.glb (TRELLIS export) calibrated transform.
_R_X90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
_PARTVERSE_TO_TRELLIS = np.eye(4)
_PARTVERSE_TO_TRELLIS[:3, :3] = 0.5 * _R_X90.T
_TRELLIS_TO_PARTVERSE = np.linalg.inv(_PARTVERSE_TO_TRELLIS)  # after -> full frame


def _concat(g):
    import trimesh
    if isinstance(g, trimesh.Scene):
        return trimesh.util.concatenate(
            [m for m in g.geometry.values() if isinstance(m, trimesh.Trimesh)])
    return g


def _full_glb(mesh_npz: Path, dst: Path) -> Path:
    import trimesh
    d = np.load(str(mesh_npz), allow_pickle=True)
    g = _concat(trimesh.load(io.BytesIO(d["full.glb"].tobytes()),
                             file_type="glb", process=False))
    g.export(str(dst)); return dst


def _after_in_full(after_glb: Path, dst: Path) -> Path:
    import trimesh
    g = _concat(trimesh.load(str(after_glb), process=False))
    g.apply_transform(_TRELLIS_TO_PARTVERSE)
    g.export(str(dst)); return dst


def _render(glb: Path, view: dict, out_dir: Path, blender: str, res: int,
            ref: Path | None, cuda: str | None) -> Image.Image | None:
    cmd = [blender, "-b", "-P", BLENDER_SCRIPT, "--",
           "--object", str(glb), "--output_folder", str(out_dir),
           "--views", json.dumps([view]), "--resolution", str(res)]
    if ref is not None:
        cmd += ["--ref_object", str(ref)]
    env = dict(os.environ)
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    png = out_dir / "000.png"
    if r.returncode != 0 or not png.is_file():
        sys.stderr.write(f"[render] FAIL {glb.name}: {r.stderr[-500:]}\n"); return None
    im = Image.open(str(png)).convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", im.size, (255,)*4), im).convert("RGB")


def _load(p: Path):
    return Image.open(str(p)).convert("RGB") if p.is_file() else None


def _fit(img, sz):
    c = Image.new("RGB", (sz, sz), (242, 242, 242))
    if img is None:
        return c
    img = img.copy(); img.thumbnail((sz, sz), Image.LANCZOS)
    c.paste(img, ((sz - img.width) // 2, (sz - img.height) // 2)); return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--blender", default=os.environ.get("BLENDER_PATH", "blender"))
    ap.add_argument("--cuda", default=os.environ.get("RENDER_CUDA"))
    a = ap.parse_args()

    root = Path(a.root)
    obj_dir = root / "objects" / a.shard / a.obj
    edit_dir = obj_dir / "edits_3d" / a.edit
    after = edit_dir / "after.glb"
    mesh_npz = Path(a.mesh_root) / a.shard / f"{a.obj}.npz"
    image_npz = Path(a.images_root) / a.shard / f"{a.obj}.npz"
    edits_2d = obj_dir / "edits_2d"
    out = Path(a.out) if a.out else (root / "_compare" / f"{a.obj}__{a.edit}__fullframe.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    views = _read_camera_views_from_npz(image_npz, slots=[a.slot])
    if not views:
        sys.exit(f"no camera for slot {a.slot}")
    view = views[0]
    print(f"[diag] slot {a.slot} camera: yaw={view['yaw']:.3f} pitch={view['pitch']:.3f} "
          f"radius={view['radius']:.3f} fov={view['fov']:.3f}")

    with tempfile.TemporaryDirectory(prefix="diag_ff_") as _t:
        tmp = Path(_t)
        full_glb = _full_glb(mesh_npz, tmp / "full.glb")
        after_ff = _after_in_full(after, tmp / "after_ff.glb")
        d1 = tmp / "f"; d1.mkdir(); d2 = tmp / "a"; d2.mkdir()
        full_im = _render(full_glb, view, d1, a.blender, a.res, None, a.cuda)
        after_im = _render(after_ff, view, d2, a.blender, a.res, full_glb, a.cuda)

    inp = _load(edits_2d / f"{a.edit}_input.png")
    ed = _load(edits_2d / f"{a.edit}_edited.png")
    panels = [inp, ed, full_im, after_im]
    labels = ["2D input", "2D FLUX edit", "3D full.glb @cam", "3D after (full frame) @cam"]
    P = 460; PAD = 8; CAP = 40
    W = len(panels) * P + (len(panels) + 1) * PAD
    H = P + 2 * PAD + CAP
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        f = ImageFont.load_default()
    dr.text((PAD, 4), f"{a.obj[:12]} | {a.edit} | slot {a.slot}", fill=(0, 0, 0), font=f)
    for i, (im, lab) in enumerate(zip(panels, labels)):
        x = PAD + i * (P + PAD); y = CAP
        sheet.paste(_fit(im, P), (x, y))
        dr.text((x + 4, y + P + 2), lab, fill=(40, 40, 40), font=f)
    sheet.save(str(out))
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
