#!/usr/bin/env python3
"""Batch re-run the masked 3-stage edit for a whole shard with the FIXED config.

This regenerates ``after.glb`` for every flux edit of every object in a shard
using the validated fixes:

  * ``--canonical``    encode + mask in TRELLIS Z-up frame (upright result)
  * ``--s2-nn-init``   warm-start + nearest-neighbor-init the edited shape tokens
                       (kills the turret spikes)
  * ``--s1-densify N`` dilate the edited-region S1 occupancy by N cells so the
                       shape decoder CLOSES the surface (fixes the see-through
                       holey "transparent grid ball" dome — a GEOMETRY issue, NOT
                       a texture/PBR one: the decoded alpha is already 1.0).

It re-encodes each object's full mesh in the canonical frame (the production
``p1_encode`` is non-canonical) into a per-object cache, then exports the
reframed GLB (partverse world frame) + a 4-view PBR contact sheet to a REVIEW
directory — it does NOT overwrite the production ``edits_3d/`` until promoted.

  CUDA_VISIBLE_DEVICES=2 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
  /mnt/zsn/miniconda3/envs/trellis2/bin/python \
    scripts/standalone/rerun_shard_masked_edit.py \
      --shard 08 --objs all --s1-densify 2 \
      --out-dir data/Pxform_v2/_rerun_v2/08
"""
from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

TRELLIS2_DIR = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
if TRELLIS2_DIR not in sys.path:
    sys.path.insert(0, TRELLIS2_DIR)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rerun-shard")


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
    ap.add_argument("--objs", default="all",
                    help="'all' or comma-separated object ids")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--edits-2d-subdir", default="edits_2d")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--hdri", default=f"{TRELLIS2_DIR}/assets/hdri/forest.exr")
    ap.add_argument("--canonical", action="store_true", default=True)
    ap.add_argument("--no-canonical", dest="canonical", action="store_false")
    ap.add_argument("--s2-nn-init", action="store_true", default=True)
    ap.add_argument("--no-s2-nn-init", dest="s2_nn_init", action="store_false")
    ap.add_argument("--s1-densify", type=int, default=0)
    ap.add_argument("--s2-anchor-mode", default="posthoc",
                    choices=["perstep", "release_late", "posthoc", "free"])
    ap.add_argument("--s2-anchor-cutoff", type=float, default=0.3)
    ap.add_argument("--s1-pad", type=int, default=3)
    ap.add_argument("--s1-thresh", type=float, default=0.1)
    ap.add_argument("--s1-soft-feather", type=float, default=0.0,
                    help="feather the S1 keep mask by N 16³ blocks (0=hard cut). "
                         "Softens the body↔edit boundary so the SS occupancy "
                         "doesn't tear at the junction.")
    ap.add_argument("--render", action="store_true", default=True)
    ap.add_argument("--no-render", dest="render", action="store_false")
    ap.add_argument("--export-glb", action="store_true", default=False,
                    help="also export reframed after.glb (SLOW: ~2-3min/edit "
                         "to_glb remesh). Default off = fast render-only review.")
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--decimation-target", type=int, default=500_000)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--p1-cache", default=None,
                    help="dir for canonical p1 caches (default <out-dir>/_p1_canon)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    p1_cache = Path(args.p1_cache) if args.p1_cache else out_root / "_p1_canon"
    p1_cache.mkdir(parents=True, exist_ok=True)

    import cv2
    import torch
    from PIL import Image

    from partcraft.pipeline_v3 import trellis2_3d as T
    from partcraft.pipeline_v3 import trellis2_encode as TE
    from partcraft.pipeline_v3.paths import PipelineRoot
    from partcraft.pipeline_v3.specs import iter_flux_specs
    from partcraft.pipeline_v3.trellis2_white import read_white_model_flag
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap

    root = Path(args.root)
    shard_dir = root / "objects" / args.shard
    if args.objs == "all":
        objs = sorted(p.name for p in shard_dir.iterdir() if p.is_dir())
    else:
        objs = [o.strip() for o in args.objs.split(",") if o.strip()]
    if args.limit:
        objs = objs[:args.limit]
    log.info("shard %s: %d objects → %s", args.shard, len(objs), out_root)

    p25_cfg = {
        "trellis2_codebase": TRELLIS2_DIR,
        "trellis2_ckpt": args.ckpt,
        "trellis2_pipeline_type": "1024_cascade",
        "trellis2_s1_pad": args.s1_pad,
        "trellis2_s1_keep_thresh": args.s1_thresh,
        "trellis2_s1_soft_feather": args.s1_soft_feather,
        "trellis2_canonical_frame": args.canonical,
        "trellis2_s2_warmstart": args.s2_nn_init,
        "trellis2_s2_nn_init": args.s2_nn_init,
        "trellis2_s1_densify": args.s1_densify,
        "trellis2_s2_anchor_mode": args.s2_anchor_mode,
        "trellis2_s2_anchor_cutoff": args.s2_anchor_cutoff,
        "trellis2_texture_size": args.texture_size,
        "trellis2_decimation_target": args.decimation_target,
    }
    log.info("config: canonical=%s anchor_mode=%s nn_init=%s s1_densify=%d "
             "s1_pad=%d thresh=%.2f", args.canonical, args.s2_anchor_mode,
             args.s2_nn_init, args.s1_densify, args.s1_pad, args.s1_thresh)

    pipeline = T._ensure_pipeline(p25_cfg, log)
    enc = None
    envmap = None
    if args.render:
        hdr = cv2.cvtColor(cv2.imread(args.hdri, cv2.IMREAD_UNCHANGED),
                           cv2.COLOR_BGR2RGB)
        envmap = EnvMap(torch.tensor(hdr, dtype=torch.float32, device="cuda"))

    n_ok = n_fail = 0
    for oi, obj in enumerate(objs):
        try:
            mesh_npz = Path(args.mesh_root) / args.shard / f"{obj}.npz"
            image_npz = Path(args.images_root) / args.shard / f"{obj}.npz"
            ctx = PipelineRoot(root=root).context(
                args.shard, obj, mesh_npz=mesh_npz, image_npz=image_npz)
            specs = list(iter_flux_specs(ctx))
            if not specs:
                log.info("[%d/%d] %s: no specs, skip", oi + 1, len(objs), obj)
                continue

            # canonical re-encode (cached)
            p1_path = p1_cache / f"{obj}.npz"
            if not p1_path.is_file():
                if enc is None:
                    enc = TE._ensure_encoder(p25_cfg, log)
                feats, coords = TE.encode_full_mesh(
                    enc, mesh_npz, canonical=args.canonical)
                np.savez_compressed(p1_path, feats=feats, coords=coords)
                log.info("[%d/%d] %s: canon-encoded %d tokens",
                         oi + 1, len(objs), obj, coords.shape[0])
            d = np.load(str(p1_path))
            p1_feats = torch.from_numpy(d["feats"]).float()
            p1_coords3 = torch.from_numpy(d["coords"]).int()
            white_model = read_white_model_flag(ctx)

            for spec in specs:
                t0 = time.time()
                e2d = ctx.dir / args.edits_2d_subdir
                ip = e2d / f"{spec.edit_id}_input.png"
                ep = e2d / f"{spec.edit_id}_edited.png"
                if not (ip.is_file() and ep.is_file()):
                    log.warning("  %s: missing 2D imgs, skip", spec.edit_id)
                    continue
                orig_img = Image.open(ip).convert("RGB")
                edited_img = Image.open(ep).convert("RGB")
                mesh, latents = T._build_p4_mesh(
                    pipeline, spec, edited_img, orig_img,
                    p1_feats, p1_coords3, mesh_npz, p25_cfg, log,
                    white_model=white_model)

                ed_dir = out_root / obj / spec.edit_id
                ed_dir.mkdir(parents=True, exist_ok=True)
                if args.export_glb:
                    T._run_and_export(
                        pipeline, None, ed_dir / "after.glb", p25_cfg, log,
                        mesh_obj=mesh, reframe_mesh_npz=mesh_npz)
                T._save_edit_latents(latents, ed_dir, log)

                if args.render:
                    snap = render_utils.render_snapshot(
                        mesh, resolution=args.resolution, r=2.0, fov=40.0,
                        nviews=4, envmap=envmap)
                    shaded = snap["shaded"] if "shaded" in snap else \
                        next(iter(snap.values()))
                    Image.fromarray(_tile(shaded, cols=2)).save(
                        ed_dir / "after_shaded.png")
                    if "normal" in snap:
                        Image.fromarray(_tile(snap["normal"], cols=2)).save(
                            ed_dir / "after_normal.png")
                n_ok += 1
                log.info("  ✓ %s (%s) %.1fs", spec.edit_id, spec.edit_type,
                         time.time() - t0)
        except Exception as e:
            n_fail += 1
            log.exception("[%d/%d] %s FAILED: %s", oi + 1, len(objs), obj, e)

    log.info("DONE shard %s: %d edits ok, %d objects failed → %s",
             args.shard, n_ok, n_fail, out_root)


if __name__ == "__main__":
    main()
