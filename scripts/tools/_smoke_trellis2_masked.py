"""Model-level smoke test for the masked 3-layer TRELLIS.2 edit path.

Encodes a real partverse mesh → P1 shape latent, then runs _build_p4_mesh for a
`modification` edit (SS structure + geometry + material with coord bridge) and a
`material` edit (S2-only), asserting a non-empty mesh comes out. Uses the same
view image for orig+edited (we only verify the code paths run, not edit quality).

Run:  CUDA_VISIBLE_DEVICES=0 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/tools/_smoke_trellis2_masked.py
"""
import io
import os
import sys
import logging
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

CODEBASE = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
sys.path.insert(0, CODEBASE)
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("smoke")

MESH = "/mnt/zsn/data/partverse/inputs/mesh/08/bdd36c94f3f74f22b02b8a069c8d97b7.npz"
IMG = "/mnt/zsn/data/partverse/inputs/images/08/bdd36c94f3f74f22b02b8a069c8d97b7.npz"
CKPT = "/mnt/zsn/ckpts/TRELLIS.2-4B"


def encode_p1(enc):
    import trellis2.modules.sparse as sp
    import o_voxel
    d = np.load(MESH, allow_pickle=True)
    scene = trimesh.load(io.BytesIO(d["full.glb"].tobytes()), file_type="glb", process=False)
    mesh = (trimesh.util.concatenate([g for g in scene.geometry.values()
            if isinstance(g, trimesh.Trimesh)]) if isinstance(scene, trimesh.Scene) else scene)
    v = torch.from_numpy(np.asarray(mesh.vertices)).float()
    f = torch.from_numpy(np.asarray(mesh.faces)).long()
    vmin, vmax = v.min(0)[0], v.max(0)[0]
    v = (v - (vmin + vmax) / 2) * (0.99999 / (vmax - vmin).max())
    vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=v.float(), faces=f.long(), grid_size=1024,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dual_local = (dv * 1024 - vi).clamp(0., 1.).float()
    inter3 = inter.float() if (inter.dim() == 2 and inter.shape[1] == 3) else None
    coords = torch.cat([torch.zeros_like(vi[:, :1]), vi], dim=-1).to(torch.int32)
    vsp = sp.SparseTensor(dual_local, coords)
    isp = vsp.replace(inter3.bool().float())
    with torch.no_grad():
        z = enc(vsp.cuda(), isp.cuda())
    return z.feats.float().cpu(), z.coords[:, 1:].int().cpu()


def load_view():
    d = np.load(IMG, allow_pickle=True)
    return Image.open(io.BytesIO(d[d.files[0]].tobytes())).convert("RGB")


def _patch_dinov3():
    """transformers 5.9 nests the DINOv3 encoder at model.model.layer; the
    shipped extractor still uses model.layer. Patch for the smoke test only."""
    import torch.nn.functional as F
    from trellis2.modules import image_feature_extractor as ife

    def extract_features(self, image):
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        layers = getattr(self.model, "layer", None)
        if layers is None:
            layers = self.model.model.layer
        for layer_module in layers:
            hidden_states = layer_module(hidden_states,
                                         position_embeddings=position_embeddings)
        return F.layer_norm(hidden_states, hidden_states.shape[-1:])

    ife.DinoV3FeatureExtractor.extract_features = extract_features
    log.info("patched DinoV3FeatureExtractor for transformers 5.9")


def main():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import trellis2.models as t2_models
    from partcraft.pipeline_v3 import trellis2_3d as T

    _patch_dinov3()
    log.info("loading pipeline %s ...", CKPT)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(CKPT); pipe.cuda()
    log.info("loading shape encoder ...")
    enc = t2_models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16").eval().cuda()

    p1_feats, p1_coords = encode_p1(enc)
    log.info("P1: %d tokens, feats %s", p1_coords.shape[0], tuple(p1_feats.shape))
    del enc; torch.cuda.empty_cache()

    img = load_view()
    p25 = {"trellis2_codebase": CODEBASE}

    for et, parts in [("modification", [0]), ("material", [0])]:
        spec = SimpleNamespace(edit_id=f"smoke_{et}", edit_type=et, selected_part_ids=parts)
        log.info("=== _build_p4_mesh  edit_type=%s parts=%s ===", et, parts)
        mesh, _latents = T._build_p4_mesh(pipe, spec, img, img, p1_feats, p1_coords,
                                          Path(MESH), p25, log, white_model=False)
        nv = int(mesh.vertices.shape[0]); nf = int(mesh.faces.shape[0])
        log.info("RESULT %s: verts=%d faces=%d", et, nv, nf)
        assert nv > 0 and nf > 0, f"{et}: empty mesh"
    log.info("SMOKE_PASS")


if __name__ == "__main__":
    main()
