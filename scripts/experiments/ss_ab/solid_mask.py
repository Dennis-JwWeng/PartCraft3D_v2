#!/usr/bin/env python3
"""Build a SOLID 64³ edit mask from the (shell) part voxelization, robustly for
possibly-non-watertight part meshes, and report per-part fill stats.

The S2 coord bridge tests edit membership per-voxel against `edit_grid64`, which
is a dual-grid SURFACE shell — regenerated voxels off that shell get mislabeled
"preserve" and anchored to stale SLat (a fragmentation source).  Filling the
shell solid (interior only, NO outward growth) fixes the membership test.

`scipy.ndimage.binary_fill_holes` only fills cavities fully enclosed by the
surface, so:
  * watertight / near-watertight part  → interior filled (solid >> shell)
  * non-watertight (open boundary) part → cavity leaks to the border → NOT
    filled → safely degrades back to ~the shell (solid ≈ shell).
The solid/shell ratio is thus an empirical watertightness probe.

Writes `edit_solid64` into each inputs/<obj>/<edit>.npz (additive).

    /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/experiments/ss_ab/solid_mask.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
IO = ROOT / "data/Pxform_v2/_scratch/ss_ab/inputs"


def solid_fill(shell64: np.ndarray) -> tuple[np.ndarray, bool]:
    """Interior fill (no outward growth). Returns (solid, looks_watertight)."""
    solid = ndimage.binary_fill_holes(shell64)
    grew = int(solid.sum()) - int(shell64.sum())
    watertight = grew > 0.15 * int(shell64.sum())   # filled a real interior
    return solid, watertight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    rows = []
    for p in sorted(IO.glob("*/*.npz")):
        if args.only and args.only not in str(p):
            continue
        d = dict(np.load(p, allow_pickle=True))
        eg = d["edit_grid64"].astype(int)
        shell = np.zeros((64, 64, 64), bool)
        shell[eg[:, 0], eg[:, 1], eg[:, 2]] = True
        solid, wt = solid_fill(shell)
        d["edit_solid64"] = np.argwhere(solid).astype(np.int32)
        np.savez(p, **d)
        ns, no = int(shell.sum()), int(solid.sum())
        rows.append((p.parent.name, p.stem, ns, no, no / max(ns, 1), wt))

    print(f"{'obj':12s} {'edit':40s} {'shell':>6s} {'solid':>6s} {'x':>5s}  watertight?")
    for obj, eid, ns, no, r, wt in rows:
        print(f"{obj[:12]:12s} {eid[:40]:40s} {ns:6d} {no:6d} {r:5.2f}  {'YES' if wt else 'no (open?)'}")
    nwt = sum(1 for *_, wt in rows if wt)
    print(f"\n{len(rows)} edits · {nwt} look watertight (interior filled), "
          f"{len(rows)-nwt} stayed ~shell (non-watertight / thin)")


if __name__ == "__main__":
    main()
