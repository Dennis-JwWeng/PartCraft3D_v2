"""Empirical: does decode grid resolution (1024 vs 512) change mesh fragmentation?

Decodes the SAME saved edit latents (shape+tex SLat) at res 1024 and 512 and
renders the condition view, for a few edits in both recipes (A=_exp_flowedit_free_r1024,
B=_exp_masked_posthoc_r1024).  No re-edit — pure decode/render from stored latents.

Grid per edit row:  BEFORE | A@1024 | A@512 | B@1024 | B@512   (condition view)

    CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/viz/decode_res_compare.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/mnt/zsn/3dobject/TRELLIS.2")
A = ROOT / "data/Pxform_v2/_exp_flowedit_free_r1024/objects/08"
B = ROOT / "data/Pxform_v2/_exp_masked_posthoc_r1024/objects/08"
OUT = ROOT / "data/Pxform_v2/_scratch/ab_compare/decode_res.png"
CKPT = "/mnt/zsn/ckpts/TRELLIS.2-4B"
RES_IMG = 384
# (obj, edit_id, condition view)  — a fragmented-A case + a coherent case
EDITS = [
    ("be004a4739ca4fefb121e9898459b2ed", "mod_be004a4739ca4fefb121e9898459b2ed_000", "front"),
    ("bde54221d35c4341b80e9576f4e379ef", "mod_bde54221d35c4341b80e9576f4e379ef_000", "right"),
    ("be393abd76474c4287d3bcec890367c6", "mod_be393abd76474c4287d3bcec890367c6_000", "front"),
]


def main() -> None:
    import logging; logging.basicConfig(level=logging.WARNING)
    import torch, cv2
    import trellis2.modules.sparse as sp
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from partcraft.pipeline_v3.trellis2_compat import patch_dinov3_extractor
    from partcraft.render import ovox_views as ov

    patch_dinov3_extractor()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(CKPT); pipe.cuda()
    env = ov.load_envmap("/mnt/zsn/3dobject/TRELLIS.2/assets/hdri/forest.exr")

    def slat(npz):
        z = np.load(npz)
        f = torch.from_numpy(z["feats"].astype(np.float32)).cuda()
        c = torch.from_numpy(z["coords"].astype(np.int32)).cuda()
        c = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.int32, device=c.device), c], 1)
        return sp.SparseTensor(feats=f, coords=c)

    def render(latdir, res, view):
        sh = slat(latdir / "shape_slat.npz"); tx = slat(latdir / "tex_slat.npz")
        mesh = pipe.decode_latent(sh, tx, res)[0]
        d = ov.render_sample(mesh, [view], envmap=env, resolution=RES_IMG, key="shaded", bg=(1, 1, 1))
        return cv2.cvtColor(d[view], cv2.COLOR_RGB2BGR)

    def lab(im, t):
        im = im.copy(); cv2.putText(im, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2); return im

    rows = []
    for o, eid, view in EDITS:
        before = cv2.imread(str(A / o / "gate_views" / f"before_view_{view}.png"))
        before = cv2.resize(before, (RES_IMG, RES_IMG)) if before is not None else np.full((RES_IMG, RES_IMG, 3), 235, np.uint8)
        cells = [lab(before, f"BEFORE {view}")]
        for tag, base in (("A", A), ("B", B)):
            ld = base / o / "edits_3d" / eid / "latents"
            for res in (1024, 512):
                cells.append(lab(render(ld, res, view), f"{tag}@{res}"))
        rows.append(np.concatenate(cells, axis=1))
        print(f"  {o[:8]} {eid.split('_')[-1]} done")
    cv2.imwrite(str(OUT), np.concatenate(rows, axis=0))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
