#!/usr/bin/env python3
"""SS A/B — TRELLIS.2 side.  Runs ONLY the S1 (sparse-structure) masked edit
with TRELLIS.2's own SS flow model, on the SHARED inputs from prep.py, and
saves the decoded occupancy ``coords_new``.

Fairness contract (identical to run_t1.py except the flow model + image cond):
  * occupancy + edit mask + hard keep16  ← loaded from prep.py (NOT recomputed)
  * SS VAE encode / decode               ← ss_enc/ss_dec_conv3d_16l8 (shared w/ T1)
  * schedule                             ← UNIFIED (steps25/cfg5/[.5,1]/rt3)
  * invert under ORIGINAL image, masked repaint under EDITED image
  * ONLY T2-specific thing: sparse_structure_flow_model (ss_flow_img_dit_1_3B_64)
    + T2's get_cond image embedding.

Reuses the validated ``trellis2_structure.edit_structure`` (it recomputes the
SAME hard keep16 from edit_grid64 via edit_grid_64_to_keep16(thresh=0.1)).

    CUDA_VISIBLE_DEVICES=6 /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/experiments/ss_ab/run_t2.py
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
sys.path.insert(0, str(ROOT))

UNIFIED = {"steps": 25, "guidance_strength": 5.0,
           "guidance_interval": [0.5, 1.0], "rescale_t": 3.0,
           "guidance_rescale": 0.0}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ss_ab_t2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default="data/Pxform_v2/_scratch/ss_ab")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--codebase", default="/mnt/zsn/3dobject/TRELLIS.2")
    ap.add_argument("--only", default=None, help="substring filter on obj/edit")
    args = ap.parse_args()

    import torch
    from PIL import Image
    p25 = {"trellis2_codebase": args.codebase, "trellis2_ckpt": args.ckpt}
    from partcraft.pipeline_v3.trellis2_3d import _ensure_pipeline
    from partcraft.pipeline_v3.trellis2_structure import get_ss_encoder, edit_structure
    from partcraft.pipeline_v3.trellis2_masked_sampler import (
        MaskedFlowEulerGuidanceIntervalSampler)

    pipeline = _ensure_pipeline(p25, log)
    ss_enc = get_ss_encoder(pipeline, p25, log)
    sampler = MaskedFlowEulerGuidanceIntervalSampler(1e-5)
    dev = "cuda"

    in_root = ROOT / args.io / "inputs"
    out_root = ROOT / args.io / "out" / "t2"
    inputs = sorted(in_root.glob("*/*.npz"))
    if args.only:
        inputs = [p for p in inputs if args.only in str(p)]

    for p in inputs:
        obj, eid = p.parent.name, p.stem
        d = np.load(p, allow_pickle=True)
        coords0 = torch.from_numpy(d["coords0"].astype("int64")).int().to(dev)
        eg = d["edit_grid64"].astype("int64")
        edit_grid = torch.zeros(64, 64, 64, dtype=torch.bool, device=dev)
        edit_grid[eg[:, 0], eg[:, 1], eg[:, 2]] = True
        orig = pipeline.preprocess_image(Image.open(str(d["input_png"])).convert("RGB"))
        edit = pipeline.preprocess_image(Image.open(str(d["edited_png"])).convert("RGB"))
        cond_orig = pipeline.get_cond([orig], 512)
        cond_edit = pipeline.get_cond([edit], 512)

        coords_new = edit_structure(
            pipeline, ss_enc, sampler, coords0, edit_grid,
            cond_orig, cond_edit, log,
            keep_thresh=float(d["keep_thresh"]), soft_feather=0.0,
            contact_mask=None, contact_sigma=None,
            ss_param_override=UNIFIED,
        )
        cn = coords_new.detach().cpu().numpy()
        cn = (cn[:, 1:] if cn.shape[1] == 4 else cn).astype(np.int32)
        od = out_root / obj
        od.mkdir(parents=True, exist_ok=True)
        np.savez(od / f"{eid}.npz", coords_new=cn, coords0=d["coords0"],
                 edit_grid64=d["edit_grid64"], backend="t2", sched=str(UNIFIED))
        log.info("[t2] %s/%s  occ %d -> %d", obj, eid, coords0.shape[0], cn.shape[0])

    log.info("done -> %s", out_root)


if __name__ == "__main__":
    main()
