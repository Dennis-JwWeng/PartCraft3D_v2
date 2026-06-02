#!/usr/bin/env python3
"""Step 1 of the TRELLIS.1-SS → TRELLIS.2-S2 bridge.

Generate sparse-structure occupancy coords with the ORIGINAL TRELLIS.1 image
pipeline (DINOv2 cond + ``ss_flow_img_dit_L_16l8`` + the shared
``ss_dec_conv3d_16l8`` decoder) for every geometry edit of a shard, and cache
them to ``<out>/<obj>/<edit_id>/ss1_coords.npz``.

Why: TRELLIS.2's masked S1 over-inflates the edit region and invents unreasonable
structure for large/whole-object edits.  TRELLIS.1's SS flow may produce cleaner
geometry.  Since BOTH share the same 16³×8 SS latent + 64³ ``ss_dec`` decoder,
TRELLIS.1's 64³ coords are drop-in compatible with a TRELLIS.2 second stage —
a later step (in the trellis2 env) loads these coords as the SLat structure and
runs TRELLIS.2 shape+texture S2 on them (``--ss1-coords-dir``).

This first version is VANILLA: it runs SS on each *edited* image (whole object,
no mask), to test the geometry-quality hypothesis directly.  A masked-edit
variant (v1 interweave inversion+repaint at SS) is a follow-up.

Runs in the ``vinedresser3d`` conda env (which has the TRELLIS.1 deps); the v1
repo's ``third_party/`` is prepended to sys.path for the vendored ``trellis``
package + the bundled DINOv2.  All weights are local/offline.

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  /mnt/zsn/miniconda3/envs/vinedresser3d/bin/python \
    scripts/standalone/trellis1_ss_coords.py \
      --shard 08 --objs all \
      --out-dir data/Pxform_v2/_rerun_v2/08_ss1coords
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("SPCONV_ALGO", "auto")

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# v2 repo root (for `partcraft` imports in masked mode)
_ROOT = Path(__file__).resolve().parents[2]

# v1 repo third_party/ → vendored `trellis` package + bundled DINOv2 (dinov2_hub)
V1_THIRD_PARTY = "/mnt/zsn/zsn_workspace/PartCraft3D/third_party"
if V1_THIRD_PARTY not in sys.path:
    sys.path.insert(0, V1_THIRD_PARTY)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("trellis1-ss")

# edit-id prefixes that change geometry (the only ones where a fresh SS helps).
# mat/clr/glb/global are S2-only (material/colour/export) — skipped.
GEOM_PREFIXES = ("mod", "scl", "add", "rem")


def _masked_ss(pipe, ss_enc, ss_flow, ss_dec, msampler, make_cb,
               edited_png, input_png, mask_npz, thresh, gs, gi, rt, steps, seed):
    """MASKED TRELLIS.1 SS edit — a faithful mirror of trellis2_structure.
    edit_structure, but on TRELLIS.1's SS flow + DINOv2 cond.  Invert the
    original occupancy under the INPUT image, then repaint the edit region
    under the EDITED image with the preserved blocks anchored to the inversion
    trajectory.  Returns coords_new ``[M,3]`` int16 @64³."""
    import numpy as np
    import torch
    from PIL import Image

    dev = "cuda"
    d = np.load(mask_npz, allow_pickle=True)
    coords0 = torch.from_numpy(d["coords0"].astype("int64")).to(dev)
    shp = tuple(int(x) for x in d["grid_shape"])
    grid = np.unpackbits(d["edit_grid"])[:int(np.prod(shp))].reshape(shp).astype(bool)
    edit_grid = torch.from_numpy(grid).to(dev)

    cond_orig = pipe.get_cond(
        [pipe.preprocess_image(Image.open(input_png).convert("RGB"))])
    cond_edit = pipe.get_cond(
        [pipe.preprocess_image(Image.open(edited_png).convert("RGB"))])

    occ = torch.zeros(1, 1, 64, 64, 64, device=dev)
    occ[0, 0, coords0[:, 0], coords0[:, 1], coords0[:, 2]] = 1.0
    z_s0 = ss_enc(occ.float())

    torch.manual_seed(seed)
    inv = msampler.invert_clean(
        ss_flow, z_s0, cond=cond_orig["cond"], neg_cond=cond_orig["neg_cond"],
        guidance_strength=1.0, guidance_interval=gi, guidance_rescale=0.0,
        steps=steps, rescale_t=rt, verbose=False)
    # 16³ keep-mask (1=preserve, 0=edit), hard threshold like the baseline
    g16 = edit_grid.float().reshape(16, 4, 16, 4, 16, 4).sum(dim=(1, 3, 5)) / 64.0
    keep16 = (~(g16 >= thresh))[None, None]
    cb = make_cb(inv, keep16)
    z_s_new = msampler.sample(
        ss_flow, inv[1.0], cond=cond_edit["cond"], neg_cond=cond_edit["neg_cond"],
        steps=steps, rescale_t=rt, guidance_strength=gs, guidance_interval=gi,
        guidance_rescale=0.0, verbose=False, x_init=inv[1.0], step_callback=cb,
    ).samples
    decoded = ss_dec(z_s_new) > 0
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]][:, 1:].to(torch.int16).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--objs", default="all",
                    help="'all' or comma-separated object ids")
    ap.add_argument("--edits-2d-subdir", default="edits_2d")
    ap.add_argument("--image-ckpt", default="/mnt/zsn/ckpts/TRELLIS-image-large")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ss-steps", type=int, default=0,
                    help="override SS sampler steps (0 = ckpt default, 25).")
    ap.add_argument("--cfg-strength", type=float, default=-1.0,
                    help="override SS guidance strength (<0 = ckpt default).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", default=False)
    # ── masked mode (Exp 1): TRELLIS.1 inversion+repaint on the edit region ──
    ap.add_argument("--masked", action="store_true", default=False,
                    help="MASKED TRELLIS.1 SS edit (mirror of edit_structure): "
                         "invert the original occupancy under the input image and "
                         "repaint the edit region under the edited image. Needs "
                         "--mask-inputs-dir (from dump_ss_mask_inputs.py).")
    ap.add_argument("--mask-inputs-dir", default=None,
                    help="dir of <obj>/<edit_id>/mask_inputs.npz (coords0+edit_grid).")
    ap.add_argument("--ss-enc-ckpt",
                    default="/mnt/zsn/ckpts/TRELLIS-image-large/ckpts/"
                            "ss_enc_conv3d_16l8_fp16",
                    help="TRELLIS.1 SS encoder ckpt (to encode the original occ).")
    ap.add_argument("--s1-thresh", type=float, default=0.1,
                    help="16³ keep-mask threshold (match the baseline edit).")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.root) / "objects" / args.shard

    if args.objs == "all":
        objs = sorted(p.name for p in shard_dir.iterdir() if p.is_dir())
    else:
        objs = [o.strip() for o in args.objs.split(",") if o.strip()]
    if args.limit:
        objs = objs[:args.limit]

    masked = args.masked
    if masked and not args.mask_inputs_dir:
        log.error("--masked requires --mask-inputs-dir"); return
    mi_root = Path(args.mask_inputs_dir) if args.mask_inputs_dir else None

    # work-list: (obj, eid, edited_png, input_png, out_npz, mask_npz)
    work = []
    for obj in objs:
        e2d = shard_dir / obj / args.edits_2d_subdir
        if not e2d.is_dir():
            continue
        for png in sorted(e2d.glob("*_edited.png")):
            eid = png.name[:-len("_edited.png")]
            if not eid.split("_")[0].lower().startswith(GEOM_PREFIXES):
                continue
            out_npz = out_root / obj / eid / "ss1_coords.npz"
            if out_npz.is_file() and not args.overwrite:
                continue
            inp = e2d / f"{eid}_input.png"
            mnpz = (mi_root / obj / eid / "mask_inputs.npz") if mi_root else None
            if masked and (mnpz is None or not mnpz.is_file()):
                continue  # no precomputed mask inputs for this edit
            work.append((obj, eid, png, inp, out_npz, mnpz))
    log.info("shard %s [%s]: %d objects → %d edits → %s", args.shard,
             "masked" if masked else "vanilla", len(objs), len(work), out_root)
    if not work:
        log.info("nothing to do."); return

    log.info("loading TRELLIS.1 image pipeline from %s ...", args.image_ckpt)
    t0 = time.time()
    pipe = TrellisImageTo3DPipeline.from_pretrained(args.image_ckpt)
    pipe.cuda()
    ss_params = dict(pipe.sparse_structure_sampler_params)
    if args.ss_steps > 0:
        ss_params["steps"] = args.ss_steps
    if args.cfg_strength >= 0:
        ss_params["cfg_strength"] = args.cfg_strength
    log.info("pipeline ready in %.1fs; SS params=%s", time.time() - t0, ss_params)

    # masked-mode extras: SS encoder + the (TRELLIS.2) masked flow sampler
    make_cb = msampler = ss_enc = ss_flow = ss_dec = None
    gs = gi = rt = steps = None
    if masked:
        import trellis.models as tm
        ss_enc = tm.from_pretrained(args.ss_enc_ckpt).cuda().eval()
        os.environ.setdefault("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
        for p in (os.environ["TRELLIS2_DIR"], str(_ROOT)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from partcraft.pipeline_v3.trellis2_masked_sampler import (
            MaskedFlowEulerGuidanceIntervalSampler, make_inverse_anchored_callback)
        make_cb = make_inverse_anchored_callback
        msampler = MaskedFlowEulerGuidanceIntervalSampler(1e-5)
        ss_flow = pipe.models["sparse_structure_flow_model"]
        ss_dec = pipe.models["sparse_structure_decoder"]
        steps = int(ss_params["steps"]); rt = float(ss_params.get("rescale_t", 1.0))
        gs = float(ss_params.get("cfg_strength", 5.0))
        gi = ss_params.get("cfg_interval", (0.0, 1.0))
        log.info("masked: ss_enc ready; steps=%d cfg=%.1f interval=%s rt=%.1f",
                 steps, gs, gi, rt)

    n_ok = n_fail = 0
    for i, (obj, eid, png, inp, out_npz, mnpz) in enumerate(work):
        try:
            t1 = time.time()
            if not masked:
                img = pipe.preprocess_image(Image.open(png).convert("RGB"))
                cond = pipe.get_cond([img])
                torch.manual_seed(args.seed)
                c3 = pipe.sample_sparse_structure(
                    cond, 1, ss_params)[:, 1:].to(torch.int16).cpu().numpy()
                tag = "trellis1_ss_vanilla"
            else:
                c3 = _masked_ss(pipe, ss_enc, ss_flow, ss_dec, msampler, make_cb,
                                png, inp, mnpz, args.s1_thresh,
                                gs, gi, rt, steps, args.seed)
                tag = "trellis1_ss_masked"
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_npz, coords=c3, meta=np.array([obj, eid, tag], dtype=object))
            n_ok += 1
            log.info("[%d/%d] %s: %d voxels (%.1fs)",
                     i + 1, len(work), eid, c3.shape[0], time.time() - t1)
        except Exception as e:
            n_fail += 1
            log.exception("[%d/%d] %s FAILED: %s", i + 1, len(work), eid, e)

    log.info("DONE shard %s: %d ok, %d failed → %s",
             args.shard, n_ok, n_fail, out_root)


if __name__ == "__main__":
    main()
