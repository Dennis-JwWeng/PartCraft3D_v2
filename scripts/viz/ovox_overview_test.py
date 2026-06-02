"""Test: build the phase-1 overview rows from the o-voxel (no Blender, no
packed input images).

Per mesh: render the 5 overview viewpoints from a coloured o-voxel (top row) and
from per-part occupancy o-voxels (bottom row = palette highlight), plus one
gate-style highlight (first part RED, rest GREY) to confirm part selection
renders.  Saves a viewpoint json so downstream keeps the camera info.

    CUDA_VISIBLE_DEVICES=3 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/viz/ovox_overview_test.py \
      --shard 00 --n 4 \
      --out data/Pxform_v2/_scratch/ovox_overview
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

SEP = 4
SEP_RGB = (200, 200, 200)


def _row(imgs, sep=SEP, color=SEP_RGB):
    H = imgs[0].shape[0]
    col = np.full((H, sep, 3), color, np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.concatenate([out, col, im], axis=1)
    return out


def _grid(rgb, highlight):
    rt, rb = _row(rgb), _row(highlight)
    band = np.full((6, rt.shape[1], 3), 180, np.uint8)
    return np.concatenate([rt, band, rb], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="00")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ids", default="", help="csv of specific obj ids")
    ap.add_argument("--mesh-root", type=Path,
                    default=_ROOT / "data/partverse/inputs/mesh")
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "data/Pxform_v2/_scratch/ovox_overview")
    ap.add_argument("--color-grid", type=int, default=512)
    ap.add_argument("--part-grid", type=int, default=256)
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    from PIL import Image
    from partcraft.pipeline_v3 import trellis2_ovox_render as ovr

    mesh_dir = args.mesh_root / args.shard
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        mesh_paths = [mesh_dir / f"{i}.npz" for i in ids]
    else:
        mesh_paths = sorted(mesh_dir.glob("*.npz"))[: args.n]
    print(f"[ovox-ov] {len(mesh_paths)} meshes from {mesh_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for mp in mesh_paths:
        oid = mp.stem
        t0 = time.time()
        try:
            res = ovr.render_overview_from_ovox(
                mp, color_grid=args.color_grid, part_grid=args.part_grid,
                resolution=args.res)
        except Exception as e:
            print(f"[ovox-ov] {oid} FAILED: {e}")
            continue
        # gate-style highlight: first part as target (red), rest grey
        tgt = res["part_ids"][:1]
        gate = ovr.render_overview_from_ovox(
            mp, color_grid=args.color_grid, part_grid=args.part_grid,
            resolution=args.res, target_ids=tgt)
        dt = time.time() - t0

        odir = args.out / oid
        odir.mkdir(parents=True, exist_ok=True)
        grid = _grid(res["rgb"], res["highlight"])
        gate_grid = _grid(gate["rgb"], gate["highlight"])
        Image.fromarray(grid).save(odir / "overview_ovox.png")
        Image.fromarray(gate_grid).save(odir / "overview_gate_part0.png")
        (odir / "viewpoints.json").write_text(json.dumps(
            {"views": res["views"], "cameras": res["cam"],
             "part_ids": res["part_ids"]}, indent=2))
        print(f"[ovox-ov] {oid}: {len(res['part_ids'])} parts, {dt:.1f}s "
              f"→ {odir/'overview_ovox.png'}")
        summary_rows.append(grid)

    if summary_rows:
        W = max(r.shape[1] for r in summary_rows)
        band = 8
        tiles = []
        for r in summary_rows:
            if r.shape[1] < W:
                pad = np.full((r.shape[0], W - r.shape[1], 3), 255, np.uint8)
                r = np.concatenate([r, pad], axis=1)
            tiles.append(r)
            tiles.append(np.full((band, W, 3), 120, np.uint8))
        Image.fromarray(np.concatenate(tiles[:-1], axis=0)).save(args.out / "ALL_overviews.png")
        print(f"[ovox-ov] wrote {args.out/'ALL_overviews.png'}")


if __name__ == "__main__":
    main()
