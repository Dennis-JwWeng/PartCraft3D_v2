#!/usr/bin/env python3
"""SS A/B — TRELLIS.1 side.  Runs ONLY the S1 (sparse-structure) masked edit
with TRELLIS.1's own image SS flow model (ss_flow_img_dit_L_16l8), on the SHARED
inputs from prep.py, and saves the decoded occupancy ``coords_new``.

Ported verbatim from the old PartCraft3D vinedresser3d recipe
(third_party/interweave_Trellis.py): RF inversion under the ORIGINAL image
(cfg off) → masked forward repaint under the EDITED image, re-injecting the
inverted trajectory in the keep region (hard keep16).

Fairness contract (identical to run_t2.py except the flow model + image cond):
  * occupancy + edit mask + hard keep16  ← loaded from prep.py (NOT recomputed)
  * SS VAE encode / decode               ← ss_enc/ss_dec_conv3d_16l8 (shared w/ T2)
  * schedule                             ← UNIFIED (steps25/cfg5/[.5,1]/rt3)
  * ONLY T1-specific thing: sparse_structure_flow_model (ss_flow_img_dit_L_16l8)
    + TRELLIS.1 DINOv2 get_cond.

Runs in the `trellis` conda env (TRELLIS.1 codebase).

    CUDA_VISIBLE_DEVICES=6 /mnt/zsn/miniconda3/envs/trellis/bin/python \
      scripts/experiments/ss_ab/run_t1.py
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
T1_REPO = Path("/mnt/zsn/zsn_workspace/PartCraft3D")
sys.path.insert(0, str(T1_REPO / "third_party"))

# UNIFIED schedule (same numbers as run_t2.py; T1 sampler uses cfg_strength key)
STEPS, CFG, CFG_INTERVAL, RESCALE_T = 25, 5.0, (0.5, 1.0), 3.0


# ── RF sampler, ported verbatim from interweave_Trellis.py ───────────────────
def get_times(steps, rescale_t, int_len, num_iter, inverse):
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_seq = t_seq[::-1]
    t_seq_new = []
    for i in range(0, steps + 1, int_len):
        interval = t_seq[i:min(i + int_len, steps + 1)]
        if len(interval) == 1:
            t_seq_new.extend(interval); continue
        for cnt in range(num_iter):
            t_seq_new.extend(interval)
            if cnt < num_iter - 1:
                t_seq_new.extend(interval[::-1][1:-1])
    t_seq = np.array(t_seq_new[::-1])
    if inverse:
        t_seq = t_seq[::-1]
    t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
    return t_seq, t_pairs


def _infer(model, x_t, t, cond=None, **kw):
    import torch
    tt = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
    if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
        cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
    return model(x_t, tt, cond, **kw)


def sample_once(model, x_t, t, t_prev, cond=None, neg_cond=None,
                cfg_strength=3.0, cfg_interval=(0.0, 1.0), **kw):
    if cfg_interval[0] <= t <= cfg_interval[1]:
        pred = _infer(model, x_t, t, cond, **kw)
        neg = _infer(model, x_t, t, neg_cond, **kw)
        pred_v = (1 + cfg_strength) * pred - cfg_strength * neg
    else:
        pred_v = _infer(model, x_t, t, cond, **kw)
    return pred_v


def rf_sample_once(model, x_t, t_curr, t_prev, **kw):
    pred = sample_once(model, x_t, t_curr, t_prev, **kw)
    mid = x_t + (t_prev - t_curr) / 2 * pred
    pred_mid = sample_once(model, mid, (t_curr + t_prev) / 2, t_prev, **kw)
    first = (pred_mid - pred) / ((t_prev - t_curr) / 2)
    return x_t + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default="data/Pxform_v2/_scratch/ss_ab")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS-image-large")
    ap.add_argument("--ss-enc",
                    default="/mnt/zsn/ckpts/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline
    import trellis.models as t1_models

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.ckpt)
    pipeline.cuda()
    ss_enc = t1_models.from_pretrained(args.ss_enc).eval().cuda()
    ss_flow = pipeline.models["sparse_structure_flow_model"]
    ss_dec = pipeline.models["sparse_structure_decoder"]
    dev = "cuda"
    print(f"[t1] loaded image pipeline {args.ckpt} + ss_enc")

    in_root = ROOT / args.io / "inputs"
    out_root = ROOT / args.io / "out" / "t1"
    inputs = sorted(in_root.glob("*/*.npz"))
    if args.only:
        inputs = [p for p in inputs if args.only in str(p)]

    for p in inputs:
        obj, eid = p.parent.name, p.stem
        d = np.load(p, allow_pickle=True)
        coords0 = d["coords0"].astype("int64")
        keep16 = torch.from_numpy(d["keep16"].astype(bool)).to(dev)        # [16,16,16] True=preserve
        keep = keep16[None, None].float().expand(1, 8, 16, 16, 16)         # broadcast to z_s
        orig = pipeline.preprocess_image(Image.open(str(d["input_png"])).convert("RGB"))
        edit = pipeline.preprocess_image(Image.open(str(d["edited_png"])).convert("RGB"))
        c_orig = pipeline.get_cond([orig]); c_edit = pipeline.get_cond([edit])

        # occupancy → SS latent (shared VAE, identical weights to T2)
        occ = torch.zeros(1, 1, 64, 64, 64, device=dev)
        occ[0, 0, coords0[:, 0], coords0[:, 1], coords0[:, 2]] = 1.0
        z_s0 = ss_enc(occ.float())

        # RF inversion under ORIGINAL image (cfg off)
        _, t_pairs_inv = get_times(STEPS, RESCALE_T, 1, 1, True)
        sample = z_s0
        inv = {}
        for t_curr, t_prev in t_pairs_inv:
            sample = rf_sample_once(ss_flow, sample, t_curr, t_prev,
                                    cond=c_orig["cond"], neg_cond=c_orig["neg_cond"],
                                    cfg_strength=0.0, cfg_interval=(0.0, 1.0))
            inv[round(float(t_prev), 6)] = sample
        s1_noise = sample      # fully-noised inverted latent (repaint start)

        # masked forward repaint under EDITED image; anchor keep region to inv
        _, t_pairs = get_times(STEPS, RESCALE_T, 1, 1, False)
        sample = s1_noise
        for t_curr, t_prev in t_pairs:
            x = rf_sample_once(ss_flow, sample, t_curr, t_prev,
                               cond=c_edit["cond"], neg_cond=c_edit["neg_cond"],
                               cfg_strength=CFG, cfg_interval=CFG_INTERVAL)
            inv_feats = inv.get(round(float(t_prev), 6))
            if inv_feats is not None:
                x = x * (1.0 - keep) + inv_feats * keep
            sample = x
        z_s_new = sample

        dec = ss_dec(z_s_new) > 0          # [1,1,64,64,64]
        cn = torch.argwhere(dec)[:, 2:5].int().cpu().numpy().astype(np.int32)
        od = out_root / obj
        od.mkdir(parents=True, exist_ok=True)
        eg = d["edit_grid64_dense"] if "edit_grid64_dense" in d.files else d.get("edit_grid64")
        np.savez(od / f"{eid}.npz", coords_new=cn, coords0=d["coords0"],
                 edit_grid64_dense=eg, backend="t1",
                 sched=str(dict(steps=STEPS, cfg=CFG, interval=CFG_INTERVAL, rt=RESCALE_T)))
        print(f"[t1] {obj}/{eid}  occ {coords0.shape[0]} -> {cn.shape[0]}")

    print(f"done -> {out_root}")


if __name__ == "__main__":
    main()
