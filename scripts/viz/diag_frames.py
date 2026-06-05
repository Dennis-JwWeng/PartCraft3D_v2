"""Verify the unified-frame fix: decoded RGB vs part-seg must align (no transform).

Runs the REAL pipeline path (``trellis2_ovox_render.render_pbr_overview``) — decode
the encoded latents → RGB (top row) + part-mesh palette segmentation (bottom row),
both through ``render_sample`` at the named cameras with NO extra transform.  If the
``_CANON_ROT`` reasoning is right, top and bottom now sit in ONE frame (the object is
upright and identically oriented in both rows, column-for-column).

    CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/viz/diag_frames.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)

OBJS = ["bde54221d35c4341b80e9576f4e379ef", "be004a4739ca4fefb121e9898459b2ed"]
CKPT = "/mnt/zsn/ckpts/TRELLIS.2-4B"


def main() -> None:
    import logging; logging.basicConfig(level=logging.INFO); log = logging.getLogger("d")
    import torch
    import cv2
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from partcraft.pipeline_v3.trellis2_compat import patch_dinov3_extractor
    from partcraft.pipeline_v3 import trellis2_encode as TE
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    from partcraft.render import ovox_views as ov

    out = Path("data/Pxform_v2/_scratch/diag_frames"); out.mkdir(parents=True, exist_ok=True)
    patch_dinov3_extractor()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(CKPT); pipeline.cuda()
    encs = TE._ensure_encoders({"trellis2_codebase": TRELLIS2_DIR}, log)
    envmap = ov.load_envmap(f"{TRELLIS2_DIR}/assets/hdri/forest.exr")

    def lab(im, t):
        im = np.ascontiguousarray(im.copy())
        cv2.putText(im, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return im

    rows = []
    for obj in OBJS:
        mesh_npz = Path("data/partverse/inputs/mesh/08") / f"{obj}.npz"
        eo = TE.encode_shape_tex_ss(encs, mesh_npz, 1024, canonical=True)
        shape = TE._slat_from_arrays(eo["shape_feats"], eo["shape_coords"])
        tex = TE._slat_from_arrays(eo["tex_feats"], eo["tex_coords"])
        r = OVR.render_pbr_overview(pipeline, mesh_npz, shape, tex, envmap, resolution=384)
        top = np.concatenate([lab(r["rgb"][v], f"rgb {v}") for v in ov.VIEW_ORDER], axis=1)
        bot = np.concatenate([lab(r["seg"][v], f"seg {v}") for v in ov.VIEW_ORDER], axis=1)
        rows.append(np.concatenate([top, bot], axis=0))
        log.info("%s done", obj[:8])
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "frames.png")
    log.info("wrote %s", out / "frames.png")


if __name__ == "__main__":
    main()
