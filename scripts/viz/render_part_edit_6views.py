#!/usr/bin/env python3
"""Render 6-view before / part-mask / after for one part-edit case.

Logic (3D part edit):
  - ``selected_part_ids`` → edit target (highlight red in mask row)
  - all other parts → preserved / masked (grey in mask row)
  - before = original mesh PBR at SIX_VIEW_ORDER cameras
  - after  = decoded post-edit mesh at the same cameras

Outputs under ``<out>/six_views/<edit_id>/``:
  before_view_{front,back,left,right,top,bottom}.png
  mask_view_{...}.png          (red=selected, grey=rest)
  after_view_{...}.png

Usage:
  CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 \\
    python scripts/viz/render_part_edit_6views.py \\
      --root data/Pxform_v2 --shard 08 --obj <obj_id> --edit <edit_id_substr>
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description="6-view part-edit before/mask/after render")
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit", default="", help="edit_id substring; default=first flux spec")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--p1-cache", default="")
    ap.add_argument("--out", default="", help="default: <root>/<shard>/<obj>/six_views")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("render_part_edit_6views")

    import numpy as np
    import torch
    from PIL import Image

    from partcraft.pipeline_v3 import trellis2_3d as T
    from partcraft.pipeline_v3 import trellis2_encode as TE
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    from partcraft.pipeline_v3.paths import PipelineRoot
    from partcraft.pipeline_v3.specs import iter_flux_specs
    from partcraft.pipeline_v3.trellis2_white import read_white_model_flag
    from partcraft.render import ovox_views as ov

    views = ov.SIX_VIEW_ORDER
    p25_cfg = {
        "trellis2_codebase": TRELLIS2_DIR,
        "trellis2_ckpt": args.ckpt,
        "trellis2_pipeline_type": "1024_cascade",
        "trellis2_s1_pad": 3,
        "trellis2_s1_keep_thresh": 0.1,
        "trellis2_canonical_frame": True,
        "trellis2_s2_warmstart": True,
        "trellis2_s2_nn_init": True,
        "trellis2_s2_anchor_mode": "posthoc",
        "trellis2_s1_mode": "masked",
        "trellis2_texture_size": 2048,
        "trellis2_decimation_target": 500000,
        "trellis2_gate_view_res": args.res,
    }

    mesh_npz = Path(args.mesh_root) / args.shard / f"{args.obj}.npz"
    image_npz = Path(args.images_root) / args.shard / f"{args.obj}.npz"
    ctx = PipelineRoot(root=Path(args.root)).context(
        args.shard, args.obj, mesh_npz=mesh_npz, image_npz=image_npz
    )
    specs = [s for s in iter_flux_specs(ctx) if (not args.edit or args.edit in s.edit_id)]
    if not specs:
        log.error("no matching edit spec")
        return
    spec = specs[0]
    pids = list(spec.selected_part_ids)
    log.info("edit=%s type=%s selected_part_ids=%s", spec.edit_id, spec.edit_type, pids)

    out_base = Path(args.out) if args.out else ctx.dir / "six_views"
    out_dir = out_base / spec.edit_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = T._ensure_pipeline(p25_cfg, log)
    env = ov.load_envmap(f"{TRELLIS2_DIR}/assets/hdri/forest.exr")

    p1_cache = Path(args.p1_cache) if args.p1_cache else Path(args.root) / "_p1_canon" / args.shard
    p1p = p1_cache / f"{args.obj}.npz"
    if p1p.is_file():
        d = np.load(str(p1p))
        p1_feats = torch.from_numpy(d["feats"]).float()
        p1_coords3 = torch.from_numpy(d["coords"]).int()
    else:
        enc = TE._ensure_encoder(p25_cfg, log)
        feats, coords = TE.encode_full_mesh(enc, mesh_npz, canonical=True)
        p1p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p1p, feats=feats, coords=coords)
        p1_feats = torch.from_numpy(feats).float()
        p1_coords3 = torch.from_numpy(coords).int()

    e2d = ctx.dir / "edits_2d"
    orig_img = Image.open(e2d / f"{spec.edit_id}_input.png").convert("RGB")
    edited_img = Image.open(e2d / f"{spec.edit_id}_edited.png").convert("RGB")

    # BEFORE — decode original latents
    edit_res = int(p25_cfg.get("trellis2_edit_res", 1024))
    d = ctx.dir / "p1_encode"
    suffix = "" if edit_res == 1024 else f"_e{edit_res}"
    before_mesh = pipeline.decode_latent(
        T._slat_from_npz(d / f"shape_slat{suffix}.npz"),
        T._slat_from_npz(d / f"tex_slat{suffix}.npz"),
        edit_res,
    )[0]
    t0 = time.time()
    before = ov.render_sample(before_mesh, view_names=views, envmap=env, resolution=args.res)
    for nm, rgb in before.items():
        Image.fromarray(rgb).save(out_dir / f"before_view_{nm}.png")
    log.info("before 6 views in %.1fs", time.time() - t0)

    # MASK — selected part red, rest grey (edit region vs preserved)
    t1 = time.time()
    part_coords, _ = OVR.part_occupancy_coords(mesh_npz, grid_size=512)
    pc, pcol = OVR.colorize_parts(part_coords, target_ids=pids)
    mask = ov.render_voxel_positions(
        pc, pcol, 1.0 / 512, views, resolution=args.res, bg=(1, 1, 1)
    )
    for nm, rgb in mask.items():
        Image.fromarray(rgb).save(out_dir / f"mask_view_{nm}.png")
    log.info("mask 6 views in %.1fs", time.time() - t1)

    # AFTER — masked 3D edit decode
    t2 = time.time()
    after_mesh, _ = T._build_p4_mesh(
        pipeline, spec, edited_img, orig_img,
        p1_feats, p1_coords3, mesh_npz, p25_cfg, log,
        white_model=read_white_model_flag(ctx),
    )
    after = ov.render_sample(after_mesh, view_names=views, envmap=env, resolution=args.res)
    for nm, rgb in after.items():
        Image.fromarray(rgb).save(out_dir / f"after_view_{nm}.png")
    log.info("after 6 views in %.1fs", time.time() - t2)

    meta = {
        "edit_id": spec.edit_id,
        "edit_type": spec.edit_type,
        "selected_part_ids": pids,
        "views": views,
        "prompt": getattr(spec, "prompt", "") or "",
    }
    import json
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
