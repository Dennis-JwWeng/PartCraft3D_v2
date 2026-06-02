#!/usr/bin/env python3
"""Pre-step for the MASKED TRELLIS.1-SS edit (Exp 1 of the SS-bridge study).

The masked TRELLIS.1 SS edit must run in the ``vinedresser3d`` env, but the
edit-region voxelisation (``part_edit_grid_64`` → needs ``o_voxel``) only imports
in the ``trellis2`` env.  So this tiny script (trellis2 env, NO GPU model load)
dumps, per geometry edit, the two inputs the masked SS needs:

  * ``coords0``    — original occupancy ``[N,3]`` @64³ (from the canonical p1
                     cache; same frame the bridge uses)
  * ``edit_grid``  — ``[64,64,64]`` bool part-edit region (pad-dilated), packed

to ``<out>/<obj>/<edit_id>/mask_inputs.npz``.  ``trellis1_ss_coords.py --masked``
then loads these + the orig/edited images and runs TRELLIS.1 inversion+repaint.

  /mnt/zsn/miniconda3/envs/trellis2/bin/python \
    scripts/standalone/dump_ss_mask_inputs.py \
      --shard 08 --objs <ids> \
      --p1-cache data/Pxform_v2/_rerun_v2/08/_p1_canon \
      --out-dir data/Pxform_v2/_rerun_v2/08_ss1coords_masked
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dump-mask")

GEOM_TYPES = {"modification", "scale"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--objs", default="all")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--p1-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--s1-pad", type=int, default=3)
    ap.add_argument("--canonical", action="store_true", default=True)
    args = ap.parse_args()

    from partcraft.pipeline_v3.paths import PipelineRoot
    from partcraft.pipeline_v3.specs import iter_flux_specs
    from partcraft.pipeline_v3.trellis2_part_mask import part_edit_grid_64

    root = Path(args.root)
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    p1_cache = Path(args.p1_cache)
    shard_dir = root / "objects" / args.shard

    if args.objs == "all":
        objs = sorted(p.name for p in shard_dir.iterdir() if p.is_dir())
    else:
        objs = [o.strip() for o in args.objs.split(",") if o.strip()]

    n_ok = 0
    for obj in objs:
        p1 = p1_cache / f"{obj}.npz"
        if not p1.is_file():
            log.warning("%s: no p1 cache, skip", obj); continue
        coords0 = np.load(str(p1))["coords"].astype(np.int16)   # [N,3]
        mesh_npz = Path(args.mesh_root) / args.shard / f"{obj}.npz"
        image_npz = Path(args.images_root) / args.shard / f"{obj}.npz"
        ctx = PipelineRoot(root=root).context(
            args.shard, obj, mesh_npz=mesh_npz, image_npz=image_npz)
        for spec in iter_flux_specs(ctx):
            if (spec.edit_type or "").lower() not in GEOM_TYPES:
                continue
            parts = list(getattr(spec, "selected_part_ids", []) or [])
            if not parts:
                continue
            grid = part_edit_grid_64(
                mesh_npz, parts, pad=args.s1_pad,
                canonical=args.canonical).cpu().numpy().astype(bool)
            out_npz = out_root / obj / spec.edit_id / "mask_inputs.npz"
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_npz, coords0=coords0,
                edit_grid=np.packbits(grid.ravel()),
                grid_shape=np.array(grid.shape, dtype=np.int32),
                parts=np.array(parts, dtype=np.int32))
            n_ok += 1
            log.info("%s %s: coords0=%d, edit_vox=%d, parts=%s",
                     obj[:10], spec.edit_id, coords0.shape[0], int(grid.sum()), parts)
    log.info("DONE: %d edits → %s", n_ok, out_root)


if __name__ == "__main__":
    main()
