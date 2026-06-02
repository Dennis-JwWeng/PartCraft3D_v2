#!/usr/bin/env python3
"""Render the saved masked-edit latents NATIVELY (SLat → MeshWithVoxel), no GLB.

Re-decodes each edit's persisted `latents/{shape_slat,tex_slat}.npz` (denormalized,
on coords_new) via `pipeline.decode_latent` and renders with TRELLIS's own
`render_utils` (PbrMeshRenderer + HDR): a turntable mp4 + a hi-res multiview grid.
This is the "view it directly from the SLat" path — no o_voxel.to_glb, no blender,
no reframe. Cheap: only decode+render, the diffusion edit is NOT re-run.

  CUDA_VISIBLE_DEVICES=4 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
  python scripts/viz/render_latents_turntable.py \
    --in-root data/Pxform_v2/_rerun_v2/08 --objs all --frames 48 --res 768
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("latent-render")


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
    ap.add_argument("--in-root", required=True,
                    help="dir of <obj>/<edit_id>/latents/ (e.g. _rerun_v2/08)")
    ap.add_argument("--objs", default="all", help="'all' or comma list")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--hdri", default=f"{TRELLIS2_DIR}/assets/hdri/forest.exr")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--nviews", type=int, default=6)
    args = ap.parse_args()

    import cv2
    import torch
    import imageio
    from PIL import Image
    from partcraft.pipeline_v3 import trellis2_3d as T
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    import trellis2.modules.sparse as sp

    in_root = Path(args.in_root)
    obj_dirs = sorted(p for p in in_root.iterdir()
                      if p.is_dir() and not p.name.startswith("_"))
    if args.objs != "all":
        keep = set(o.strip() for o in args.objs.split(","))
        obj_dirs = [p for p in obj_dirs if p.name in keep]
    edits = [(o.name, ed) for o in obj_dirs for ed in sorted(o.iterdir())
             if (ed / "latents" / "shape_slat.npz").is_file()]
    log.info("%d edits to render from latents → %s", len(edits), in_root)

    p25_cfg = {"trellis2_codebase": TRELLIS2_DIR, "trellis2_ckpt": args.ckpt,
               "trellis2_pipeline_type": "1024_cascade"}
    pipeline = T._ensure_pipeline(p25_cfg, log)
    hdr = cv2.cvtColor(cv2.imread(args.hdri, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.tensor(hdr, dtype=torch.float32, device="cuda"))

    def _sparse(npz):
        d = np.load(npz)
        feats = torch.from_numpy(d["feats"]).float().cuda()
        c3 = torch.from_numpy(d["coords"]).int()
        if c3.shape[1] == 4:
            c3 = c3[:, 1:]
        coords = torch.cat([torch.zeros(c3.shape[0], 1, dtype=torch.int32), c3], 1).cuda()
        return sp.SparseTensor(feats=feats, coords=coords)

    n_ok = 0
    for obj, ed in edits:
        try:
            shape = _sparse(ed / "latents" / "shape_slat.npz")
            tex = _sparse(ed / "latents" / "tex_slat.npz")
            mesh = pipeline.decode_latent(shape, tex, 1024)[0]
            # multiview grid
            snap = render_utils.render_snapshot(
                mesh, resolution=args.res, r=2.0, fov=40.0,
                nviews=args.nviews, envmap=envmap)
            shaded = snap["shaded"] if "shaded" in snap else next(iter(snap.values()))
            Image.fromarray(_tile(shaded, cols=3)).save(ed / "slat_multiview.png")
            # turntable
            if args.frames > 0:
                raw = render_utils.render_video(
                    mesh, resolution=args.res, num_frames=args.frames,
                    r=2.0, fov=40.0, envmap=envmap)
                vid = raw["shaded"] if "shaded" in raw else next(iter(raw.values()))
                imageio.mimsave(ed / "slat_turntable.mp4", list(vid), fps=18,
                                macro_block_size=1)
            n_ok += 1
            log.info("  ✓ %s/%s", obj, ed.name)
        except Exception as e:
            log.exception("  ✗ %s/%s: %s", obj, ed.name, e)
    log.info("DONE: %d/%d rendered from latents", n_ok, len(edits))


if __name__ == "__main__":
    main()
