#!/usr/bin/env python3
"""SS A/B prep — compute the SHARED inputs for the TRELLIS.1-vs-TRELLIS.2 SS
(sparse-structure / S1) flow comparison, ONCE, in the trellis2 env.

The whole point of this experiment is a FAIR S1 comparison: identical original
occupancy, identical (no-dilation, hard) edit mask, identical schedule — the
ONLY variable is the SS flow model (T1 ss_flow_img_dit_L_16l8 vs T2
ss_flow_img_dit_1_3B_64, each with its own image cond).  So the mask + occupancy
are computed here with the validated pipeline code and saved; both the T1 and T2
runners load these exact arrays.

For each modification/scale edit in the source smoke tree it writes
``<out>/inputs/<obj>/<edit>.npz`` with:
    coords0    [N,3] int   — original 64³ occupancy (from p1_encode/shape_slat)
    edit_grid64[M,3] int   — pad=0 part-id edit region (no dilation)
    keep16     [16,16,16]  — bool, True = preserve (anchor to inverted original)
    parts, edit_type, input_png, edited_png

    /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/experiments/ss_ab/prep.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
sys.path.insert(0, str(ROOT))
TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
if TRELLIS2_DIR not in sys.path:
    sys.path.insert(0, TRELLIS2_DIR)

S1_TYPES = {"modification", "scale"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/Pxform_v2/_exp_masked_posthoc_r1024")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--out", default="data/Pxform_v2/_scratch/ss_ab")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--keep-thresh", type=float, default=0.1)
    ap.add_argument("--canonical", type=int, default=1)
    ap.add_argument("--pad", type=int, default=0,
                    help="Chebyshev dilation of the 64³ edit grid (match the "
                         "pipeline's trellis2_s1_pad for a fair T1/T2 A/B)")
    ap.add_argument("--only", default=None, help="substring filter on obj id")
    args = ap.parse_args()

    import torch
    from partcraft.pipeline_v3.trellis2_part_mask import (
        part_edit_grid_64, edit_grid_64_to_keep16)

    src = ROOT / args.src / "objects" / args.shard
    out_root = ROOT / args.out / "inputs"
    mesh_root = ROOT / args.mesh_root / args.shard

    n = 0
    for objdir in sorted(p for p in src.iterdir() if p.is_dir()):
        obj = objdir.name
        if args.only and args.only not in obj:
            continue
        ss_p = objdir / "p1_encode" / "shape_slat.npz"
        mesh_npz = mesh_root / f"{obj}.npz"
        if not ss_p.is_file() or not mesh_npz.is_file():
            continue
        coords0 = np.load(ss_p)["coords"]
        coords0 = (coords0[:, 1:] if coords0.shape[1] == 4 else coords0).astype(np.int32)
        for ed in sorted((objdir / "edits_3d").glob("*/")):
            eid = ed.name
            ssn = ed / "latents" / "ss.npz"
            if not ssn.is_file():
                continue
            d = np.load(ssn, allow_pickle=True)
            etype = str(d["edit_type"]).lower()
            if etype not in S1_TYPES:
                continue
            parts = [int(x) for x in np.asarray(d["parts"]).tolist()]
            ip = objdir / "edits_2d" / f"{eid}_input.png"
            ep = objdir / "edits_2d" / f"{eid}_edited.png"
            if not (ip.is_file() and ep.is_file()):
                print(f"  skip (no 2D pair): {obj}/{eid}")
                continue

            can = bool(args.canonical)
            # DIRECT per-target-part voxelization, pad=0 — the part's OWN dual-grid
            # (connected, matches the part).  NOT Voronoi-over-all-parts (that lets
            # neighbour parts compete and scatters an internal part) and NOT a solid
            # fill (parts are non-watertight, fill leaks).  S2 build_coord_bridge
            # point-queries this; S1 keep16 = its 16³ downsample.
            eg64 = part_edit_grid_64(mesh_npz, parts, pad=args.pad, canonical=can)
            keep16 = edit_grid_64_to_keep16(eg64, thresh=args.keep_thresh)  # True=preserve

            od = out_root / obj
            od.mkdir(parents=True, exist_ok=True)
            np.savez(
                od / f"{eid}.npz",
                coords0=coords0,
                edit_grid64_dense=eg64.cpu().numpy().astype(bool),   # dense, for S2 bridge + viz
                keep16=keep16.cpu().numpy().astype(bool),            # for S1
                parts=np.asarray(parts, np.int32),
                edit_type=etype,
                input_png=str(ip), edited_png=str(ep),
                canonical=can, keep_thresh=float(args.keep_thresh),
            )
            cov16 = 100.0 * (1.0 - keep16.float().mean().item())
            print(f"{obj}/{eid:42s} occ={coords0.shape[0]:6d} "
                  f"edit_vox={int(eg64.sum()):5d} keep16_edit={cov16:5.1f}% parts={parts}")
            n += 1
    print(f"\nwrote {n} edit inputs -> {out_root}")


if __name__ == "__main__":
    main()
