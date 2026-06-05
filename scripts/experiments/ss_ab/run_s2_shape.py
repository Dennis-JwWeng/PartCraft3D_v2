#!/usr/bin/env python3
"""SS A/B — S2 SHAPE stage (mask-only), shared across both S1 occupancies.

Takes the S1 occupancy `coords_new` from BOTH backends (out/t1, out/t2) and runs
the SAME TRELLIS.2 S2 shape-SLat masked edit on each, then decodes shape-only to
a white-model mesh (skips PBR/tex) and renders named views.  This isolates how
the two S1 occupancies (TRELLIS.1 vs TRELLIS.2 SS flow) fare through ONE identical
S2 shape recipe.

S2 recipe (identical for both backends — only coords_new differs):
  * anchor_mode = "perstep"  (pure per-step mask anchor — the "只带 mask" baseline)
  * edit region = direct per-part voxelization (edit_grid64_dense from prep)
  * shape SLat flow @res=1024 (T2's shape_slat_flow_model_1024), native sampler
  * invert under INPUT image, masked repaint under EDITED image
  * decode = build_white_model_mesh (shape decoder + zero tex → flat grey)

Parallel: shard the (backend × edit) work list with --gpu-shard i/n, one process
per GPU.

    CUDA_VISIBLE_DEVICES=2 /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/experiments/ss_ab/run_s2_shape.py --gpu-shard 0/6
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ss_ab_s2")

VIEWS = ["front", "right", "back", "left", "down"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default="data/Pxform_v2/_scratch/ss_ab")
    ap.add_argument("--src", default="data/Pxform_v2/_exp_masked_posthoc_r1024")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--codebase", default="/mnt/zsn/3dobject/TRELLIS.2")
    ap.add_argument("--anchor", default="perstep")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--backends", default="t1,t2")
    ap.add_argument("--gpu-shard", default="0/1", help="i/n — process work items where idx%n==i")
    ap.add_argument("--only", default=None, help="comma substrings; keep work items matching any")
    ap.add_argument("--render-res", type=int, default=512)
    ap.add_argument("--ssaa", type=int, default=2)
    args = ap.parse_args()
    i, n = (int(x) for x in args.gpu_shard.split("/"))

    import torch
    from PIL import Image
    p25 = {"trellis2_codebase": args.codebase, "trellis2_ckpt": args.ckpt,
           "trellis2_gate_view_res": 512}
    from partcraft.pipeline_v3.trellis2_3d import _ensure_pipeline, _get_envmap
    from partcraft.pipeline_v3.trellis2_structure import get_ss_encoder  # noqa (load order)
    from partcraft.pipeline_v3 import trellis2_edit_stages as t2e
    from partcraft.pipeline_v3.trellis2_white import build_white_model_mesh
    from partcraft.pipeline_v3.trellis2_masked_sampler import (
        MaskedFlowEulerGuidanceIntervalSampler)
    from partcraft.render import ovox_views as _ov

    src = ROOT / args.src / "objects" / args.shard
    io = ROOT / args.io
    backends = args.backends.split(",")

    # work list = (backend, obj, eid) over every saved S1 occupancy
    work = []
    for b in backends:
        for p in sorted((io / "out" / b).glob("*/*.npz")):
            work.append((b, p.parent.name, p.stem))
    work = [w for k, w in enumerate(work) if k % n == i]
    if args.only:
        subs = args.only.split(",")
        work = [w for w in work if any(s in f"{w[0]}/{w[1]}/{w[2]}" for s in subs)]
    log.info("[s2 shard %d/%d] %d items", i, n, len(work))
    if not work:
        return

    pipeline = _ensure_pipeline(p25, log)
    sampler = MaskedFlowEulerGuidanceIntervalSampler(1e-5)
    env = _get_envmap(p25, log)
    dev = "cuda"

    for b, obj, eid in work:
        cn = np.load(io / "out" / b / obj / f"{eid}.npz", allow_pickle=True)["coords_new"]
        coords_new = torch.from_numpy(cn.astype("int64")).int().to(dev)
        inp = np.load(io / "inputs" / obj / f"{eid}.npz", allow_pickle=True)
        edit_grid = torch.from_numpy(inp["edit_grid64_dense"].astype(bool)).to(dev)
        # original shape latent (feats on coords0) from p1_encode
        p1 = np.load(src / obj / "p1_encode" / "shape_slat.npz")
        p1_feats = torch.from_numpy(p1["feats"]).float().to(dev)
        c0 = p1["coords"]; c0 = c0[:, 1:] if c0.shape[1] == 4 else c0
        coords0 = torch.from_numpy(c0.astype("int64")).int().to(dev)
        orig = pipeline.preprocess_image(Image.open(str(inp["input_png"])).convert("RGB"))
        edit = pipeline.preprocess_image(Image.open(str(inp["edited_png"])).convert("RGB"))
        cond_orig = pipeline.get_cond([orig], args.res)
        cond_edit = pipeline.get_cond([edit], args.res)

        shape_new = t2e.masked_shape_slat(
            pipeline, sampler, p1_feats, coords0, coords_new, edit_grid,
            cond_orig, cond_edit, log,
            warmstart=False, nn_init=False, anchor_mode=args.anchor,
            contact_mask=None, contact_sigma=None, res=args.res)
        mesh = build_white_model_mesh(pipeline, shape_new, log, res=args.res)

        outd = io / "s2_shape" / b / obj / eid
        outd.mkdir(parents=True, exist_ok=True)
        imgs = _ov.render_sample(mesh, view_names=VIEWS, envmap=env,
                                 resolution=args.render_res, ssaa=args.ssaa,
                                 bg=(1, 1, 1))
        for name, rgb in imgs.items():
            Image.fromarray(rgb).save(outd / f"mesh_view_{name}.png")
        sn = shape_new.coords.detach().cpu().numpy()
        np.savez(outd / "shape_slat.npz",
                 feats=shape_new.feats.detach().cpu().numpy(),
                 coords=sn[:, 1:] if sn.shape[1] == 4 else sn)
        log.info("[s2/%s] %s/%s  shape %d tok -> mesh + %d views",
                 b, obj, eid, coords_new.shape[0], len(imgs))

    log.info("[s2 shard %d/%d] done -> %s", i, n, io / "s2_shape")


if __name__ == "__main__":
    main()
