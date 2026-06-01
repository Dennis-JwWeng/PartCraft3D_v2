#!/usr/bin/env python3
"""Recompute the 3D edit-mask voxels for one edit and dump them (in 0..63 grid
coords + the full.glb center/scale that maps them back to world) so a separate
renderer can overlay them on the original mesh.

Stages dumped (all in the SAME 64^3 o-voxel frame the masked edit runs in —
partverse orientation, NO rotation):
  coords0       object occupancy (from p1_encode/shape_slat.npz)
  part_raw      target parts voxelized at 64^3 (undilated)  -> _target_block_keys_64
  edit_grid     part_edit_grid_64(pad=3)  (the S2 geometry/material edit region)
  edit16        16^3 blocks the S1 structure stage is free to repaint (~keep16)

Run under the trellis2 env (has o_voxel):
  TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
  /mnt/zsn/miniconda3/envs/trellis2/bin/python \
    scripts/standalone/dump_edit_mask_voxels.py --shard 08 --obj <id> \
    --parts 4 --pad 3 --out /tmp/mask_<id>.npz
"""
from __future__ import annotations
import argparse, io, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def main():
    import torch, trimesh
    from partcraft.pipeline_v3.trellis2_part_mask import (
        part_edit_grid_64, _target_block_keys_64, _full_center_scale,
        edit_grid_64_to_keep16, GRID_LO,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--parts", type=int, nargs="+", required=True)
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=0.1,
                    help="16^3 keep-mask threshold (higher → tighter S1 region)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mesh_npz = Path(a.mesh_root) / a.shard / f"{a.obj}.npz"
    p1 = Path(a.root) / "objects" / a.shard / a.obj / "p1_encode" / "shape_slat.npz"

    coords0 = np.load(str(p1))["coords"].astype(np.int32)  # [N,3] 0..63

    d = np.load(str(mesh_npz), allow_pickle=True)
    center, scale = _full_center_scale(d)
    center = center.cpu().numpy().astype(np.float32)
    scale = float(scale)

    # part raw (undilated) 64^3 blocks
    keys = _target_block_keys_64(mesh_npz, a.parts).cpu()
    g = GRID_LO
    part_raw = np.stack([(keys // (g * g)).numpy(),
                         ((keys // g) % g).numpy(),
                         (keys % g).numpy()], axis=-1).astype(np.int32)

    # dilated edit grid (pad)
    eg = part_edit_grid_64(mesh_npz, a.parts, pad=a.pad)  # [64,64,64] bool
    edit_grid = torch.nonzero(eg).cpu().numpy().astype(np.int32)

    # 16^3 structure-edit blocks (~keep16), upsampled to 64 coords (block*4)
    keep16 = edit_grid_64_to_keep16(eg, thresh=a.thresh)   # True=preserve
    edit16 = torch.nonzero(~keep16).cpu().numpy().astype(np.int32)  # [M,3] in 0..15

    np.savez(str(a.out),
             coords0=coords0, part_raw=part_raw, edit_grid=edit_grid,
             edit16=edit16, center=center, scale=np.float32(scale),
             grid_lo=np.int32(GRID_LO))
    print(f"[dump] obj={a.obj} parts={a.parts} pad={a.pad}")
    print(f"  coords0  : {coords0.shape[0]} voxels  bbox {coords0.min(0)}..{coords0.max(0)}")
    print(f"  part_raw : {part_raw.shape[0]} blocks  bbox "
          f"{part_raw.min(0) if len(part_raw) else '-'}..{part_raw.max(0) if len(part_raw) else '-'}")
    print(f"  edit_grid: {edit_grid.shape[0]} voxels (pad={a.pad})  bbox "
          f"{edit_grid.min(0) if len(edit_grid) else '-'}..{edit_grid.max(0) if len(edit_grid) else '-'}")
    print(f"  edit16   : {edit16.shape[0]} blocks/16  center={center.round(3)} scale={scale:.4f}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
