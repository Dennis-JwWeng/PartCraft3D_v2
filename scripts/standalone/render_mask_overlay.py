#!/usr/bin/env python3
"""Overlay the dumped 3D edit-mask voxels onto the original full.glb and render
at the SAME best-view camera as the 2D input, so mask placement is verifiable
against the 2D image.

Voxel (i,j,k) in 0..63 -> world (full.glb frame):
    p = (ijk + 0.5)/64 - 0.5        # [-0.5,0.5] o-voxel frame
    world = p / scale + center      # undo encode normalize -> full.glb coords

Panels: [2D input | full.glb | +part_raw | +edit_grid(pad) | +edit16].

Run under pipeline_server python:
  BLENDER_PATH=... RENDER_CUDA=0 /mnt/zsn/miniconda3/envs/pipeline_server/bin/python \
    scripts/standalone/render_mask_overlay.py --shard 08 --obj <id> \
    --edit mod_..._000 --mask /tmp/mask_<id>_p4.npz --slot 3
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


def _concat(g):
    import trimesh
    if isinstance(g, trimesh.Scene):
        return trimesh.util.concatenate(
            [m for m in g.geometry.values() if isinstance(m, trimesh.Trimesh)])
    return g


def _voxel_world(vox: np.ndarray, center: np.ndarray, scale: float,
                 grid: int, block: int = 1) -> np.ndarray:
    """voxel idx (0..grid-1, block-sized cells) -> world centers (full.glb)."""
    p = ((vox.astype(np.float64) + 0.5) * block) / 64.0 - 0.5
    return (p / scale) + center[None, :]


def _voxel_mesh(centers: np.ndarray, size: float, color):
    """Build one mesh of cubes at centers with a flat PBR baseColor."""
    import trimesh
    from trimesh.visual.material import PBRMaterial
    if len(centers) == 0:
        return None
    base = trimesh.creation.box(extents=(size, size, size))
    bv, bf = np.asarray(base.vertices), np.asarray(base.faces)
    nV = bv.shape[0]
    allV = (bv[None, :, :] + centers[:, None, :]).reshape(-1, 3)
    allF = (bf[None, :, :] + (np.arange(len(centers)) * nV)[:, None, None]).reshape(-1, 3)
    m = trimesh.Trimesh(vertices=allV, faces=allF, process=False)
    r, g, b = color
    m.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(baseColorFactor=[r, g, b, 255],
                             metallicFactor=0.0, roughnessFactor=0.8))
    return m


def _grey(g):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    g = g.copy()
    g.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(baseColorFactor=[205, 205, 205, 255],
                             metallicFactor=0.0, roughnessFactor=0.9))
    return g


def _render(glb: Path, view: dict, out_dir: Path, blender: str, res: int,
            ref: Path, cuda):
    cmd = [blender, "-b", "-P", BLENDER_SCRIPT, "--",
           "--object", str(glb), "--output_folder", str(out_dir),
           "--views", json.dumps([view]), "--resolution", str(res),
           "--ref_object", str(ref)]
    env = dict(os.environ)
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    png = out_dir / "000.png"
    if r.returncode != 0 or not png.is_file():
        sys.stderr.write(f"[mask] render FAIL {glb.name}: {r.stderr[-400:]}\n"); return None
    im = Image.open(str(png)).convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", im.size, (255,)*4), im).convert("RGB")


def _fit(img, sz):
    c = Image.new("RGB", (sz, sz), (242, 242, 242))
    if img is None:
        return c
    img = img.copy(); img.thumbnail((sz, sz), Image.LANCZOS)
    c.paste(img, ((sz - img.width) // 2, (sz - img.height) // 2)); return c


def main():
    import trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--mask", required=True)
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
    mesh_npz = Path(a.mesh_root) / a.shard / f"{a.obj}.npz"
    image_npz = Path(a.images_root) / a.shard / f"{a.obj}.npz"
    edits_2d = obj_dir / "edits_2d"
    out = Path(a.out) if a.out else (root / "_compare" / f"{a.obj}__{a.edit}__maskviz.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    m = np.load(a.mask)
    center = m["center"].astype(np.float64); scale = float(m["scale"])
    view = _read_camera_views_from_npz(image_npz, slots=[a.slot])[0]
    print(f"[mask] slot {a.slot} cam yaw={view['yaw']:.3f} pitch={view['pitch']:.3f} "
          f"r={view['radius']:.3f} fov={view['fov']:.3f}")

    cell = (1.0 / 64.0) / scale          # one 64-cell width in world units
    d = np.load(str(mesh_npz), allow_pickle=True)
    full = _grey(_concat(trimesh.load(io.BytesIO(d["full.glb"].tobytes()),
                                      file_type="glb", process=False)))

    stages = [
        ("full.glb", None, None, 1),
        ("+part_raw", "part_raw", (220, 40, 40), 1),
        ("+edit_grid(pad)", "edit_grid", (245, 150, 30), 1),
        ("+edit16(S1)", "edit16", (50, 90, 230), 4),
    ]
    panels, labels = [], []
    inp = Image.open(str(edits_2d / f"{a.edit}_input.png")).convert("RGB") \
        if (edits_2d / f"{a.edit}_input.png").is_file() else None
    panels.append(inp); labels.append("2D input")

    with tempfile.TemporaryDirectory(prefix="maskviz_") as _t:
        tmp = Path(_t)
        ref = tmp / "ref_full.glb"; full.export(str(ref))   # shared normalization anchor
        for lab, key, color, block in stages:
            scene = [full]
            if key is not None and m[key].shape[0] > 0:
                centers = _voxel_world(m[key], center, scale, 64, block=block)
                sz = cell * (block * 0.9)
                vm = _voxel_mesh(centers, sz, color)
                if vm is not None:
                    scene.append(vm)
            comb = trimesh.util.concatenate(scene) if len(scene) > 1 else full
            gp = tmp / f"{lab}.glb"; comb.export(str(gp))
            od = tmp / f"r_{len(panels)}"; od.mkdir()
            im = _render(gp, view, od, a.blender, a.res, ref, a.cuda)
            panels.append(im); labels.append(lab)

    P = 430; PAD = 6; CAP = 36
    W = len(panels) * P + (len(panels) + 1) * PAD
    H = P + 2 * PAD + CAP
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        f = ImageFont.load_default()
    dr.text((PAD, 4), f"{a.obj[:12]} | {a.edit} | slot {a.slot} | mask voxels on full.glb (partverse frame)",
            fill=(0, 0, 0), font=f)
    for i, (im, lab) in enumerate(zip(panels, labels)):
        x = PAD + i * (P + PAD); y = CAP
        sheet.paste(_fit(im, P), (x, y))
        dr.text((x + 4, y + P + 2), lab, fill=(40, 40, 40), font=f)
    sheet.save(str(out))
    print(f"[mask] wrote {out}")


if __name__ == "__main__":
    main()
