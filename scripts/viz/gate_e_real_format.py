"""Produce the REAL gate-E VLM input (2x5 before/after collage, white bg).

Both rows via TRELLIS's native PbrMeshRenderer at the 5 named views:
  BEFORE = original glb → glb_to_pbr_mesh
  AFTER  = decoded edited mesh (_build_p4_mesh)
composited on WHITE, then assembled with the pipeline's own
``_make_before_after_collage`` (the exact format the gate-E judge sees).

    CUDA_VISIBLE_DEVICES=3 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/viz/gate_e_real_format.py \
      --obj bde54221d35c4341b80e9576f4e379ef --edit mod_..._000
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit", default="")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--images-root", default="data/partverse/inputs/images")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--p1-cache", default="data/Pxform_v2/_rerun_v2/08/_p1_canon")
    ap.add_argument("--out", default="data/Pxform_v2/_scratch/gate_e_real")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    import logging; logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("gate_e_real")
    import cv2, torch
    from PIL import Image
    from partcraft.pipeline_v3 import trellis2_3d as T
    from partcraft.pipeline_v3 import trellis2_encode as TE
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    from partcraft.pipeline_v3.paths import PipelineRoot
    from partcraft.pipeline_v3.specs import iter_flux_specs
    from partcraft.pipeline_v3.trellis2_white import read_white_model_flag
    from partcraft.pipeline_v3.vlm_core import _make_before_after_collage
    from partcraft.render import ovox_views as ov

    p25_cfg = {
        "trellis2_codebase": TRELLIS2_DIR, "trellis2_ckpt": args.ckpt,
        "trellis2_pipeline_type": "1024_cascade",
        "trellis2_s1_pad": 3, "trellis2_s1_keep_thresh": 0.1,
        "trellis2_canonical_frame": True, "trellis2_s2_warmstart": True,
        "trellis2_s2_nn_init": True, "trellis2_s2_anchor_mode": "posthoc",
        "trellis2_s1_mode": "masked", "trellis2_ss_align_t1": True,
        "trellis2_texture_size": 2048, "trellis2_decimation_target": 500000,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pipeline = T._ensure_pipeline(p25_cfg, log)
    envmap = ov.load_envmap(f"{TRELLIS2_DIR}/assets/hdri/forest.exr")

    mesh_npz = Path(args.mesh_root) / args.shard / f"{args.obj}.npz"
    image_npz = Path(args.images_root) / args.shard / f"{args.obj}.npz"
    ctx = PipelineRoot(root=Path(args.root)).context(
        args.shard, args.obj, mesh_npz=mesh_npz, image_npz=image_npz)
    specs = [s for s in iter_flux_specs(ctx) if (not args.edit or args.edit in s.edit_id)]
    if not specs:
        log.error("no spec"); return
    spec = specs[0]; log.info("edit=%s type=%s", spec.edit_id, spec.edit_type)

    p1p = Path(args.p1_cache) / f"{args.obj}.npz"
    if p1p.is_file():
        d = np.load(str(p1p))
        p1_feats = torch.from_numpy(d["feats"]).float(); p1_coords3 = torch.from_numpy(d["coords"]).int()
    else:
        enc = TE._ensure_encoder(p25_cfg, log)
        feats, coords = TE.encode_full_mesh(enc, mesh_npz, canonical=True)
        p1p.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(p1p, feats=feats, coords=coords)
        p1_feats = torch.from_numpy(feats).float(); p1_coords3 = torch.from_numpy(coords).int()

    e2d = ctx.dir / "edits_2d"
    orig_img = Image.open(e2d / f"{spec.edit_id}_input.png").convert("RGB")
    edited_img = Image.open(e2d / f"{spec.edit_id}_edited.png").convert("RGB")

    t0 = time.time()
    after_mesh, _ = T._build_p4_mesh(pipeline, spec, edited_img, orig_img, p1_feats, p1_coords3,
                                     mesh_npz, p25_cfg, log, white_model=read_white_model_flag(ctx))
    before_mesh = OVR.glb_to_pbr_mesh(mesh_npz)
    log.info("meshes ready in %.1fs", time.time() - t0)

    # both via PbrMeshRenderer at the 5 named views, composited on WHITE
    before = ov.render_sample(before_mesh, ov.VIEW_ORDER, envmap=envmap, resolution=args.res, bg=(1, 1, 1))
    after = ov.render_sample(after_mesh, ov.VIEW_ORDER, envmap=envmap, resolution=args.res, bg=(1, 1, 1))

    before_bgr = [cv2.cvtColor(before[v], cv2.COLOR_RGB2BGR) for v in ov.VIEW_ORDER]
    after_bgr = [cv2.cvtColor(after[v], cv2.COLOR_RGB2BGR) for v in ov.VIEW_ORDER]

    # save the EXACT gate-E collage the VLM sees
    png = _make_before_after_collage(before_bgr, after_bgr)
    (out / f"{spec.edit_id}_gateE_collage.png").write_bytes(png)
    log.info("wrote %s  (REAL gate-E format: 2x5, top=before, bottom=after, white bg)",
             out / f"{spec.edit_id}_gateE_collage.png")


if __name__ == "__main__":
    main()
