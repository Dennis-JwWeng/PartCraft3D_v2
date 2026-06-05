"""Prototype: unified mesh/PBR rendering for overview (RGB + segmentation).

RGB  = encode shape+tex → decode_latent → PbrMeshRenderer 'shaded' (realistic).
SEG  = each part GLB → mesh → PbrMeshRenderer mask+depth → depth-composite into a
       FLAT pure-colour palette (captures thin geometry like crib slats, no
       o-voxel solid-block loss).
Both via TRELLIS PbrMeshRenderer at the 5 named views, white bg.

    CUDA_VISIBLE_DEVICES=6 OPENCV_IO_ENABLE_OPENEXR=1 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/viz/proto_unified_render.py --obj be004a4739ca4fefb121e9898459b2ed
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="be004a4739ca4fefb121e9898459b2ed")
    ap.add_argument("--shard", default="08")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default="data/Pxform_v2/_scratch/proto_unified")
    args = ap.parse_args()

    import logging; logging.basicConfig(level=logging.INFO); log = logging.getLogger("p")
    import torch, trimesh
    from PIL import Image
    import trellis2.models as t2_models
    import trellis2.modules.sparse as sp
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.renderers import PbrMeshRenderer, EnvMap
    from partcraft.pipeline_v3.trellis2_compat import patch_dinov3_extractor
    from partcraft.pipeline_v3 import trellis2_encode as TE
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR
    from partcraft.render import ovox_views as ov

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mesh_npz = Path("data/partverse/inputs/mesh") / args.shard / f"{args.obj}.npz"
    patch_dinov3_extractor()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt); pipeline.cuda()
    encs = TE._ensure_encoders({"trellis2_codebase": TRELLIS2_DIR}, log)
    hdr = __import__("cv2").cvtColor(__import__("cv2").imread(f"{TRELLIS2_DIR}/assets/hdri/forest.exr", -1), 4)
    envmap = EnvMap(torch.tensor(hdr, dtype=torch.float32, device="cuda"))
    extr, intr = ov.named_cameras(ov.VIEW_ORDER)

    # ── RGB: encode → decode → PBR shaded ──
    enc_out = TE.encode_shape_tex_ss(encs, mesh_npz, 1024, canonical=True)
    def _slat(f, c):
        feats = torch.from_numpy(f).float().cuda()
        coords = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.int32), torch.from_numpy(c).int()], 1).cuda()
        return sp.SparseTensor(feats=feats, coords=coords)
    shape_slat = _slat(enc_out["shape_feats"], enc_out["shape_coords"])
    tex_slat = _slat(enc_out["tex_feats"], enc_out["tex_coords"])
    decoded = pipeline.decode_latent(shape_slat, tex_slat, 1024)[0]
    rgb = ov.render_sample(decoded, ov.VIEW_ORDER, envmap=envmap, resolution=args.res, key="shaded", bg=(1, 1, 1))

    # ── SEG: ONE mesh, per-part solid palette material → 'base_color' channel ──
    # (flat pure colours, occlusion via z-buffer, true part-mesh geometry).
    from trellis2.representations.mesh.base import MeshWithPbrMaterial, PbrMaterial, AlphaMode
    scene = OVR.load_full_scene(mesh_npz)
    _, M = OVR._normalized_groups(scene, canonical=True)        # shared frame
    parts = OVR.load_part_scenes(mesh_npz)
    pal = np.array(OVR.OVERVIEW_PALETTE, np.float32) / 255.0
    all_v, all_f, all_mid, materials = [], [], [], []
    start = 0
    for i, pid in enumerate(sorted(parts)):
        groups, _ = OVR._normalized_groups(parts[pid], M=M, fix_textures=False)
        merged = trimesh.util.concatenate(groups)
        v = torch.from_numpy(np.asarray(merged.vertices)).float()
        f = torch.from_numpy(np.asarray(merged.faces)).long()
        all_v.append(v); all_f.append(f + start)
        all_mid.append(torch.full((f.shape[0],), i, dtype=torch.long)); start += v.shape[0]
        materials.append(PbrMaterial(base_color_factor=[float(x) for x in pal[pid % len(pal)]],
                                     metallic_factor=0.0, roughness_factor=1.0, alpha_mode=AlphaMode.OPAQUE))
    V = torch.cat(all_v); F = torch.cat(all_f); MID = torch.cat(all_mid)
    seg_mesh = MeshWithPbrMaterial(
        vertices=V, faces=F, material_ids=MID,
        uv_coords=torch.zeros(F.shape[0], 3, 2), materials=materials).cuda()
    seg = ov.render_sample(seg_mesh, ov.VIEW_ORDER, envmap=envmap, resolution=args.res, key="base_color", bg=(1, 1, 1))

    # stitch RGB row + SEG row
    def row(d): return np.concatenate([d[v] for v in ov.VIEW_ORDER], axis=1)
    band = np.full((6, row(rgb).shape[1], 3), 180, np.uint8)
    grid = np.concatenate([row(rgb), band, row(seg)], axis=0)
    Image.fromarray(grid).save(out / f"{args.obj[:8]}_unified.png")
    log.info("wrote %s  (top=RGB latents PBR, bottom=part-mesh seg)", out / f"{args.obj[:8]}_unified.png")


if __name__ == "__main__":
    main()
