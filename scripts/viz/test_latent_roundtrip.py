"""Verify the latents-level round-trip: encode original mesh (shape+tex) → decode → render.

If this works, the gate-E "before" can be decode(encoded shape, tex) — fully
latents-level, same source as the after (no glb rendering).  Key unknown:
do shape_enc (dual grid) and tex_enc (volumetric attr) land on compatible
coords so decode_latent works.

    CUDA_VISIBLE_DEVICES=5 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/viz/test_latent_roundtrip.py --mesh data/partverse/inputs/mesh/08/<id>.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)

SHAPE_ENC = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TEX_ENC = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="data/partverse/inputs/mesh/08/bde54221d35c4341b80e9576f4e379ef.npz")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--grid", type=int, default=1024)
    ap.add_argument("--out", default="data/Pxform_v2/_scratch/latent_roundtrip")
    args = ap.parse_args()

    import logging; logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("rt")
    import os, torch, trimesh
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    from PIL import Image
    import trellis2.models as t2_models
    import trellis2.modules.sparse as sp
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from partcraft.pipeline_v3.trellis2_compat import patch_dinov3_extractor
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    from partcraft.render import ovox_views as ov

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    patch_dinov3_extractor()
    log.info("loading pipeline + encoders ...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt); pipeline.cuda()
    shape_enc = t2_models.from_pretrained(SHAPE_ENC).eval().cuda()
    tex_enc = t2_models.from_pretrained(TEX_ENC).eval().cuda()
    envmap = ov.load_envmap(f"{TRELLIS2_DIR}/assets/hdri/forest.exr")

    # ONE consistently-normalized canonical mesh (so shape+tex share the frame)
    scene = OVR.load_full_scene(Path(args.mesh))
    groups, M = OVR._normalized_groups(scene, canonical=True)
    merged = trimesh.util.concatenate(groups)
    verts = torch.from_numpy(np.asarray(merged.vertices)).float()
    faces = torch.from_numpy(np.asarray(merged.faces)).long()

    # --- SHAPE encode (dual grid) ---
    vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        verts.cpu(), faces.cpu(), grid_size=args.grid,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    sh_coords = torch.cat([torch.zeros_like(vi[:, :1]), vi], -1)
    dual_local = (dv * args.grid - vi).clamp(0., 1.).float()
    if inter.dim() == 2 and inter.shape[1] == 3:
        inter3 = inter.float()
    else:
        b = inter.view(-1).to(torch.uint8)
        inter3 = torch.stack([(b & 1).bool(), ((b >> 1) & 1).bool(), ((b >> 2) & 1).bool()], -1).float()
    vertices_sp = sp.SparseTensor(feats=dual_local, coords=sh_coords.int()).cuda()
    inter_sp = vertices_sp.replace(inter3.bool().float().cuda())
    with torch.no_grad():
        shape_slat = shape_enc(vertices_sp, inter_sp)
    log.info("shape_slat: %s coords, %d feats", tuple(shape_slat.coords.shape), shape_slat.feats.shape[1])

    # --- TEX encode (volumetric pbr attr) ---
    coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        trimesh.Scene(groups), grid_size=args.grid, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    def f(x): return (x.float() / 255.0) if x.dtype == torch.uint8 else x.float()
    feats6 = torch.cat([f(attr["base_color"]), f(attr["metallic"]), f(attr["roughness"]), f(attr["alpha"])], -1)
    tx_coords = torch.cat([torch.zeros_like(coord[:, :1]), coord], -1)
    tex_in = sp.SparseTensor(feats=feats6.float(), coords=tx_coords.int()).cuda()
    with torch.no_grad():
        tex_slat = tex_enc(tex_in)
    log.info("tex_slat: %s coords, %d feats", tuple(tex_slat.coords.shape), tex_slat.feats.shape[1])

    # coord alignment between shape & tex slats
    sc = set(map(tuple, shape_slat.coords[:, 1:].cpu().numpy().tolist()))
    tc = set(map(tuple, tex_slat.coords[:, 1:].cpu().numpy().tolist()))
    log.info("coord overlap: shape=%d tex=%d inter=%d (shape⊆tex=%s)",
             len(sc), len(tc), len(sc & tc), sc.issubset(tc))

    # --- DECODE both → mesh ---
    try:
        meshes = pipeline.decode_latent(shape_slat, tex_slat, 1024)
        mesh = meshes[0]
        log.info("decode OK: %d verts", mesh.vertices.shape[0])
    except Exception as e:
        log.error("decode_latent FAILED: %s", e); return

    # --- render (named views, white) ---
    imgs = ov.render_sample(mesh, ov.VIEW_ORDER, envmap=envmap, resolution=512, bg=(1, 1, 1))
    row = np.concatenate([imgs[v] for v in ov.VIEW_ORDER], axis=1)
    Image.fromarray(row).save(out / f"{Path(args.mesh).stem[:8]}_decoded_from_encoded_latents.png")
    log.info("wrote %s", out / f"{Path(args.mesh).stem[:8]}_decoded_from_encoded_latents.png")


if __name__ == "__main__":
    main()
