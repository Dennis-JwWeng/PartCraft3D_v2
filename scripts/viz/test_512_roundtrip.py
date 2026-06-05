"""Validate the 512 SLat infrastructure before wiring a 512 edit.

Encodes the original mesh at grid 512 (→ 32³ shape/tex SLat) and grid 1024 (→ 64³),
decodes each at its matching res (512 / 1024), and renders front view side by side.
Confirms: (a) grid 512 → 32³ coords, shape/tex share coords; (b) decode_latent at
res=512 on a 32³ SLat produces a clean mesh (the earlier crash was a 64³-SLat @
res-512 MISMATCH).  This is reconstruction only — fragmentation differences come
from the *edit* sampler, tested separately.

    CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/viz/test_512_roundtrip.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/mnt/zsn/3dobject/TRELLIS.2")
CKPT = "/mnt/zsn/ckpts/TRELLIS.2-4B"
OBJS = ["bde54221d35c4341b80e9576f4e379ef", "be004a4739ca4fefb121e9898459b2ed"]
OUT = ROOT / "data/Pxform_v2/_scratch/ab_compare/roundtrip_512v1024.png"


def encode_shape_tex(encoders, mesh_npz, grid, canonical=True):
    """Shape+tex SLat at the given grid (→ grid/16 ³ coords).  No SS."""
    import torch, trimesh, o_voxel
    import trellis2.modules.sparse as sp
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    scene = OVR.load_full_scene(Path(mesh_npz))
    groups, _ = OVR._normalized_groups(scene, canonical=canonical)
    merged = trimesh.util.concatenate(groups)
    verts = torch.from_numpy(np.asarray(merged.vertices)).float()
    faces = torch.from_numpy(np.asarray(merged.faces)).long()
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        verts.cpu(), faces.cpu(), grid_size=grid, aabb=aabb,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dual_local = (dv * grid - vi).clamp(0., 1.).float()
    b = inter.view(-1).to(torch.uint8) if not (inter.dim() == 2 and inter.shape[1] == 3) else None
    inter3 = inter.float() if b is None else torch.stack(
        [(b & 1).bool(), ((b >> 1) & 1).bool(), ((b >> 2) & 1).bool()], -1).float()
    shc = torch.cat([torch.zeros_like(vi[:, :1]), vi], -1).int()
    vsp = sp.SparseTensor(feats=dual_local, coords=shc).cuda()
    isp = vsp.replace(inter3.bool().float().cuda())
    with torch.no_grad():
        shape = encoders["shape"](vsp, isp)
    coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        trimesh.Scene(groups), grid_size=grid, aabb=aabb)
    def _f(x): return (x.float() / 255.0) if x.dtype == torch.uint8 else x.float()
    feats6 = torch.cat([_f(attr["base_color"]), _f(attr["metallic"]),
                        _f(attr["roughness"]), _f(attr["alpha"])], -1).float()
    txc = torch.cat([torch.zeros_like(coord[:, :1]), coord], -1).int()
    tsp = sp.SparseTensor(feats=feats6, coords=txc).cuda()
    with torch.no_grad():
        tex = encoders["tex"](tsp)
    return shape, tex


def main() -> None:
    import logging; logging.basicConfig(level=logging.WARNING); log = logging.getLogger("d")
    import torch, cv2
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from partcraft.pipeline_v3.trellis2_compat import patch_dinov3_extractor
    from partcraft.pipeline_v3 import trellis2_encode as TE
    from partcraft.render import ovox_views as ov
    patch_dinov3_extractor()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(CKPT); pipe.cuda()
    encs = TE._ensure_encoders({"trellis2_codebase": "/mnt/zsn/3dobject/TRELLIS.2"}, log)
    env = ov.load_envmap("/mnt/zsn/3dobject/TRELLIS.2/assets/hdri/forest.exr")

    def lab(im, t):
        im = im.copy(); cv2.putText(im, t, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2); return im

    rows = []
    for o in OBJS:
        npz = ROOT / "data/partverse/inputs/mesh/08" / f"{o}.npz"
        cells = []
        for grid, res in ((1024, 1024), (512, 512)):
            sh, tx = encode_shape_tex(encs, npz, grid)
            nshape, ntex = sh.coords.shape[0], tx.coords.shape[0]
            cmax = int(sh.coords[:, 1:].max())
            mesh = pipe.decode_latent(sh, tx, res)[0]
            d = ov.render_sample(mesh, ["front"], envmap=env, resolution=384, key="shaded", bg=(1, 1, 1))
            im = cv2.cvtColor(d["front"], cv2.COLOR_RGB2BGR)
            cells.append(lab(im, f"grid{grid} res{res} coords={nshape}(max{cmax})"))
            print(f"  {o[:8]} grid{grid}: shape={nshape} tex={ntex} coordmax={cmax}")
        rows.append(np.concatenate(cells, 1))
    cv2.imwrite(str(OUT), np.concatenate(rows, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
