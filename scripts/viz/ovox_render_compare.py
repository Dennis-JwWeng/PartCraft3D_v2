"""Validate the o-voxel render front-end against the partverse pre-renders.

Loads one object's mesh.npz, voxelises to a coloured o-voxel, renders the
packed view indices from the o-voxel (no Blender), and tiles each o-voxel
render next to the matching packed PNG so frame/viewpoint alignment can be
eyeballed.

    CUDA_VISIBLE_DEVICES=1 /mnt/zsn/miniconda3/envs/trellis2/bin/python \
        scripts/viz/ovox_render_compare.py \
        --mesh data/partverse/inputs/mesh/00/<id>.npz \
        --images data/partverse/inputs/images/00/<id>.npz \
        --out /mnt/zsn/zsn_workspace/PartCraft3D_v2/data/Pxform_v2/_scratch/ovox_cmp \
        --grid 512 --res 512
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def _packed_png(img_npz: Path, view_index: int) -> np.ndarray | None:
    z = np.load(str(img_npz), allow_pickle=True)
    key = f"{view_index:03d}.png"
    if key not in z.files:
        return None
    import cv2
    raw = z[key]
    buf = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        a = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = rgb * a + 255.0 * (1 - a)
        arr = rgb.astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--views", default="", help="csv of packed view indices; "
                    "default = first 4 packed PNG slots")
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--no-canonical", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    from partcraft.pipeline_v3 import trellis2_ovox_render as ovr

    # pick view indices
    if args.views:
        view_indices = [int(x) for x in args.views.split(",") if x.strip()]
    else:
        z = np.load(str(args.images), allow_pickle=True)
        pngs = sorted(int(f[:-4]) for f in z.files if f.endswith(".png"))
        view_indices = pngs[:4]
    print(f"[ovox-cmp] views={view_indices} grid={args.grid} res={args.res}")

    t0 = time.time()
    coords, attr = ovr.mesh_to_colored_ovox(
        args.mesh, grid_size=args.grid, canonical=not args.no_canonical)
    t_vox = time.time() - t0
    print(f"[ovox-cmp] voxelised: {coords.shape[0]} voxels in {t_vox:.2f}s")

    t1 = time.time()
    yp = [ovr.sphere_hammersley_sequence(i, ovr.PRERENDER_NUM_VIEWS) for i in view_indices]
    imgs = ovr.render_ovox_views(
        coords, attr["base_color"], args.grid,
        [v[0] for v in yp], [v[1] for v in yp],
        resolution=args.res, ssaa=args.ssaa)
    t_render = time.time() - t1
    print(f"[ovox-cmp] rendered {len(imgs)} views in {t_render:.2f}s "
          f"({1000*t_render/max(1,len(imgs)):.0f} ms/view)")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for vi, img in zip(view_indices, imgs):
        packed = _packed_png(args.images, vi)
        ovox_pil = Image.fromarray(img)
        if packed is not None:
            packed_pil = Image.fromarray(packed).resize((args.res, args.res))
            row = np.concatenate([np.asarray(packed_pil), np.asarray(ovox_pil)], axis=1)
        else:
            row = np.asarray(ovox_pil)
        Image.fromarray(img).save(args.out / f"ovox_{vi:03d}.png")
        rows.append(row)
    if rows:
        H = max(r.shape[0] for r in rows)
        W = max(r.shape[1] for r in rows)
        canvas = np.full((H * len(rows), W, 3), 255, np.uint8)
        for k, r in enumerate(rows):
            canvas[k * H:k * H + r.shape[0], :r.shape[1]] = r
        Image.fromarray(canvas).save(args.out / "compare_packed_vs_ovox.png")
        print(f"[ovox-cmp] wrote {args.out/'compare_packed_vs_ovox.png'} "
              f"(left=packed, right=o-voxel)")


if __name__ == "__main__":
    main()
