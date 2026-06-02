"""Validate the gate-E render pair on a real decoded edit.

Re-derives the decoded post-edit mesh (``_build_p4_mesh``) for one edit, then
renders the named views two ways:
  - AFTER  = render_sample(decoded mesh, envmap)   ← post-edit latents (PBR)
  - BEFORE = o-voxel of the original mesh           ← gate-E top row
on the SAME named cameras, and tiles BEFORE-over-AFTER per view.

    CUDA_VISIBLE_DEVICES=3 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/viz/validate_after_views.py \
      --obj bde1b486ee284e4d94f54bdbb3b3d6d7 --edit mod_..._000 \
      --out data/Pxform_v2/_scratch/after_views
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit", default="", help="edit_id substring; default=first")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--p1-cache", default="data/Pxform_v2/_rerun_v2/08/_p1_canon")
    ap.add_argument("--out", default="data/Pxform_v2/_scratch/after_views")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    import logging; logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("validate_after")
    import torch
    from PIL import Image
    from partcraft.pipeline_v3 import trellis2_3d as T
    from partcraft.pipeline_v3 import trellis2_encode as TE
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    from partcraft.pipeline_v3.paths import PipelineRoot
    from partcraft.pipeline_v3.specs import iter_flux_specs
    from partcraft.pipeline_v3.trellis2_white import read_white_model_flag
    from partcraft.render import ovox_views as ov

    p25_cfg = {
        "trellis2_codebase": TRELLIS2_DIR, "trellis2_ckpt": args.ckpt,
        "trellis2_pipeline_type": "1024_cascade",
        "trellis2_s1_pad": 3, "trellis2_s1_keep_thresh": 0.1,
        "trellis2_canonical_frame": True,
        "trellis2_s2_warmstart": True, "trellis2_s2_nn_init": True,
        "trellis2_s2_anchor_mode": "posthoc", "trellis2_s1_mode": "masked",
        "trellis2_texture_size": 2048, "trellis2_decimation_target": 500000,
    }
    out = Path(args.out) / args.obj; out.mkdir(parents=True, exist_ok=True)

    pipeline = T._ensure_pipeline(p25_cfg, log)
    envmap = ov.load_envmap(f"{TRELLIS2_DIR}/assets/hdri/forest.exr")

    mesh_npz = Path(args.mesh_root) / args.shard / f"{args.obj}.npz"
    image_npz = Path(args.images_root) / args.shard / f"{args.obj}.npz"
    ctx = PipelineRoot(root=Path(args.root)).context(
        args.shard, args.obj, mesh_npz=mesh_npz, image_npz=image_npz)
    specs = [s for s in iter_flux_specs(ctx)
             if (not args.edit or args.edit in s.edit_id)]
    if not specs:
        log.error("no matching spec"); return
    spec = specs[0]
    log.info("edit=%s  type=%s  view_name=%s", spec.edit_id, spec.edit_type,
             getattr(spec, "view_name", "?"))

    # P1 canonical latent (cache or encode)
    p1p = Path(args.p1_cache) / f"{args.obj}.npz"
    if p1p.is_file():
        d = np.load(str(p1p)); p1_feats = torch.from_numpy(d["feats"]).float()
        p1_coords3 = torch.from_numpy(d["coords"]).int()
    else:
        enc = TE._ensure_encoder(p25_cfg, log)
        feats, coords = TE.encode_full_mesh(enc, mesh_npz, canonical=True)
        p1p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p1p, feats=feats, coords=coords)
        p1_feats = torch.from_numpy(feats).float(); p1_coords3 = torch.from_numpy(coords).int()

    e2d = ctx.dir / "edits_2d"
    orig_img = Image.open(e2d / f"{spec.edit_id}_input.png").convert("RGB")
    edited_img = Image.open(e2d / f"{spec.edit_id}_edited.png").convert("RGB")

    t0 = time.time()
    mesh, _lat = T._build_p4_mesh(pipeline, spec, edited_img, orig_img,
                                  p1_feats, p1_coords3, mesh_npz, p25_cfg, log,
                                  white_model=read_white_model_flag(ctx))
    log.info("decoded mesh in %.1fs", time.time() - t0)

    # AFTER — render the decoded post-edit mesh at named views
    t1 = time.time()
    after = ov.render_sample(mesh, envmap=envmap, resolution=args.res)
    log.info("render_sample (after) %d views in %.1fs", len(after), time.time() - t1)
    for nm, rgb in after.items():
        Image.fromarray(rgb).save(out / f"after_view_{nm}.png")

    # BEFORE — o-voxel of the original mesh at the same named views
    coords, attr = OVR.mesh_to_colored_ovox(mesh_npz, grid_size=512)
    before = ov.render_ovoxel(coords, attr["base_color"], 512, ov.VIEW_ORDER,
                              resolution=args.res)

    # tile: rows = views (before | after)
    pad = np.full((args.res, 6, 3), 200, np.uint8)
    rows = []
    for nm in ov.VIEW_ORDER:
        rows.append(np.concatenate([before[nm], pad, after[nm]], axis=1))
    sep = np.full((6, rows[0].shape[1], 3), 120, np.uint8)
    canvas = rows[0]
    for r in rows[1:]:
        canvas = np.concatenate([canvas, sep, r], axis=0)
    Image.fromarray(canvas).save(out / "before_after_named.png")
    log.info("wrote %s  (left=before o-voxel, right=after latents; rows=%s)",
             out / "before_after_named.png", ov.VIEW_ORDER)


if __name__ == "__main__":
    main()
