#!/usr/bin/env python3
"""Ceiling test: VANILLA TRELLIS.2 full regen from the edited image.

Runs ``pipeline.run(edited_img)`` (no masking, no anchoring — the model
generates the whole shape SLat self-consistently), exports the GLB and reports
the same dome-vs-body open_frac + alpha as the masked path.  If the dome here is
SOLID (open_frac ~ body) the see-through dome is caused by our MASKED editing
(forced anchoring breaks edit-region consistency); if it is ALSO holey the
white-clay edited image is itself ambiguous for shape generation.
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
TRELLIS2_DIR = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
if TRELLIS2_DIR not in sys.path:
    sys.path.insert(0, TRELLIS2_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vanilla")


def _tile(frames, cols=2, bg=20):
    frames = list(frames)
    rows = (len(frames) + cols - 1) // cols
    h, w, _ = frames[0].shape
    canvas = np.full((rows * h, cols * w, 3), bg, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit-id", required=True)
    ap.add_argument("--use-input", action="store_true",
                    help="regen from the *_input.png (original) instead of edited")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--hdri", default=f"{TRELLIS2_DIR}/assets/hdri/forest.exr")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import torch
    from PIL import Image
    from partcraft.pipeline_v3 import trellis2_3d as T
    from scripts.standalone.render_masked_edit_native import (
        _dump_alpha_stats, _export_and_inspect_glb)
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap

    p25_cfg = {
        "trellis2_codebase": TRELLIS2_DIR, "trellis2_ckpt": args.ckpt,
        "trellis2_pipeline_type": "1024_cascade",
    }
    pipeline = T._ensure_pipeline(p25_cfg, log)

    e2d = Path(args.root) / "objects" / args.shard / args.obj / "edits_2d"
    tag = "input" if args.use_input else "edited"
    img = Image.open(e2d / f"{args.edit_id}_{tag}.png").convert("RGB")
    log.info("VANILLA regen from %s (%s)", f"{args.edit_id}_{tag}.png", img.size)

    meshes = pipeline.run(img, pipeline_type="1024_cascade", seed=args.seed,
                          num_samples=1)
    mesh = meshes[0]
    mesh.simplify(16_777_216)

    # dummy coords for the alpha helper (it classifies by spatial z, not coords)
    dummy = torch.zeros(1, 3, dtype=torch.int32)
    _dump_alpha_stats(mesh, dummy, dummy, log)
    _export_and_inspect_glb(mesh, out_dir, p25_cfg, log, bands=(1,))

    hdr = cv2.cvtColor(cv2.imread(args.hdri, cv2.IMREAD_UNCHANGED),
                       cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr, dtype=torch.float32, device="cuda"))
    snap = render_utils.render_snapshot(mesh, resolution=args.resolution,
                                        r=2.0, fov=40.0, nviews=4, envmap=envmap)
    shaded = snap["shaded"] if "shaded" in snap else next(iter(snap.values()))
    Image.fromarray(_tile(shaded, cols=2)).save(out_dir / "vanilla_shaded.png")
    if "normal" in snap:
        Image.fromarray(_tile(snap["normal"], cols=2)).save(
            out_dir / "vanilla_normal.png")
    log.info("DONE → %s", out_dir)


if __name__ == "__main__":
    main()
