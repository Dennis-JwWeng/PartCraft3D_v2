"""S1 — TRELLIS.1 sparse-structure (SS) flow as a first-class, in-process option.

The masked SS edit (occ → ss_enc → RF-invert under orig img → masked forward
repaint under edited img → ss_dec → coords_new) is identical for TRELLIS.1 and
TRELLIS.2 — see ``trellis2_structure.edit_structure``.  Only three things differ
between the two: the SS *flow* model, the *image conditioner*, and (shared) the
SS VAE.  This module supplies the TRELLIS.1 flow + conditioner so ``edit_structure``
can run the T1 SS edit **inside the trellis2 process**, replacing the old offline
cross-env bridge (``scripts/experiments/ss_ab/{prep,run_t1}.py`` in the
vinedresser3d env + ``trellis2_ss1_coords_dir``).  See memory ``t1-ss-mask-bridge``.

Key facts (verified — ``scripts/experiments/ss_ab/verify_inprocess.py``):
  * The T1 SS-flow ckpt ``ss_flow_img_dit_L_16l8_fp16`` loads cleanly into
    trellis2's OWN ``SparseStructureFlowModel`` (missing=0/unexpected=0), so NO
    ``import trellis`` is needed (that would pull flexicubes→kaolin, absent here).
    ``convert_to(fp16)`` puts the torso blocks in fp16 for flash_attn while the
    input_layer / t_embedder stay fp32 (the model's own ``manual_cast`` handles it).
  * T1 conditions on **DINOv2-L** (``dinov2_vitl14_reg``, 1024-dim) — NOT trellis2's
    DinoV3 — so we load T1's DINOv2 via the bundled offline loader
    ``third_party/encode_asset/dinov2_hub`` and reproduce T1's preprocess + get_cond
    (ported from ``third_party/trellis/pipelines/trellis_image_to_3d.py``).
  * The SS VAE (``ss_enc/ss_dec_conv3d_16l8``) is shared; ``edit_structure`` uses
    ``pipeline.models["sparse_structure_decoder"]`` + ``get_ss_encoder``.

Sampling itself is done by ``edit_structure`` (the trellis2
``MaskedFlowEulerGuidanceIntervalSampler``) with this module's flow injected via
``ss_flow=`` — T1 and T2 share one masked code path.  This means T1-native output
is NOT bit-identical to the old bridge (run_t1's hand-rolled Heun RF sampler), but
the masked-edit semantics are the same and per-edit IoU is high.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_T1_SS_FLOW = (
    "/mnt/zsn/ckpts/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16")
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def load_t1_ss_flow(pipeline, p25_cfg: dict, logger):
    """Load + cache the TRELLIS.1 SS flow inside trellis2's SparseStructureFlowModel.

    Blocks → fp16 (flash_attn); input_layer / t_embedder stay fp32.  Cached on the
    pipeline object so it loads once per worker.
    """
    flow = getattr(pipeline, "_pcv2_t1_ss_flow", None)
    if flow is not None:
        return flow
    codebase = str(Path(p25_cfg.get(
        "trellis2_codebase", "/mnt/zsn/3dobject/TRELLIS.2")).resolve())
    if codebase not in sys.path:
        sys.path.insert(0, codebase)
    from trellis2.models.sparse_structure_flow import SparseStructureFlowModel
    from safetensors.torch import load_file

    ckpt = str(p25_cfg.get("trellis2_t1_ss_flow", DEFAULT_T1_SS_FLOW))
    logger.info("[s5/S1] loading TRELLIS.1 SS flow %s (trellis2 class)", ckpt)
    args = json.load(open(ckpt + ".json"))["args"]
    flow = SparseStructureFlowModel(**args)
    msg = flow.load_state_dict(load_file(ckpt + ".safetensors"), strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        raise RuntimeError(
            f"T1 SS-flow ckpt mismatch vs trellis2 SparseStructureFlowModel: "
            f"missing={msg.missing_keys[:3]} unexpected={msg.unexpected_keys[:3]}")
    flow = flow.eval().cuda()
    flow.convert_to(torch.float16)   # torso blocks → fp16 for flash_attn
    pipeline._pcv2_t1_ss_flow = flow
    return flow


def load_t1_dino(pipeline, logger):
    """Load + cache TRELLIS.1's DINOv2-L image conditioner (bundled, offline)."""
    dino = getattr(pipeline, "_pcv2_t1_dino", None)
    if dino is not None:
        return dino
    for sub in ("third_party/encode_asset", "third_party/dinov2"):
        p = str(ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from dinov2_hub import load_dinov2_vitl14_reg
    logger.info("[s5/S1] loading TRELLIS.1 DINOv2-L (dinov2_vitl14_reg, bundled)")
    dino = load_dinov2_vitl14_reg(pretrained=True).eval().cuda()
    pipeline._pcv2_t1_dino = dino
    return dino


def get_rembg_session(pipeline, logger):
    """Cached rembg u2net session; None if rembg unavailable (white-bg fallback)."""
    if hasattr(pipeline, "_pcv2_rembg"):
        return pipeline._pcv2_rembg
    try:
        import rembg
        sess = rembg.new_session("u2net")
        logger.info("[s5/S1] rembg u2net session ready (matches bridge preprocess)")
    except Exception as e:  # noqa: BLE001
        sess = None
        logger.warning("[s5/S1] rembg unavailable (%s); white-bg alpha fallback",
                       type(e).__name__)
    pipeline._pcv2_rembg = sess
    return sess


def t1_preprocess(input_img, rembg_session):
    """TRELLIS.1 image preprocess (ported from trellis_image_to_3d.preprocess_image).

    Background removal (rembg u2net, or white-threshold fallback for clean white-bg
    renders) → crop to 1.2× object bbox → resize 518² → premultiply alpha.
    """
    from PIL import Image
    has_alpha = (input_img.mode == "RGBA"
                 and not np.all(np.array(input_img)[:, :, 3] == 255))
    if has_alpha:
        output = input_img
    elif rembg_session is not None:
        import rembg
        inp = input_img.convert("RGB")
        scale = min(1, 1024 / max(inp.size))
        if scale < 1:
            inp = inp.resize((int(inp.width * scale), int(inp.height * scale)),
                             Image.Resampling.LANCZOS)
        output = rembg.remove(inp, session=rembg_session)
    else:
        rgb = np.array(input_img.convert("RGB"))
        a = (~np.all(rgb >= 248, axis=-1)).astype(np.uint8) * 255
        output = Image.fromarray(np.dstack([rgb, a]), mode="RGBA")
    arr = np.array(output)
    alpha = arr[:, :, 3]
    bb = np.argwhere(alpha > 0.8 * 255)
    x0, y0, x1, y1 = (np.min(bb[:, 1]), np.min(bb[:, 0]),
                      np.max(bb[:, 1]), np.max(bb[:, 0]))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sz = int(max(x1 - x0, y1 - y0) * 1.2)
    output = output.crop((cx - sz // 2, cy - sz // 2, cx + sz // 2, cy + sz // 2))
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    o = np.array(output).astype(np.float32) / 255
    o = o[:, :, :3] * o[:, :, 3:4]
    return Image.fromarray((o * 255).astype(np.uint8))


@torch.no_grad()
def t1_get_cond(dino, img_pil, dev: str = "cuda") -> dict:
    """TRELLIS.1 image conditioning: DINOv2-L x_prenorm patch tokens (layer-normed).

    Mirrors trellis_image_to_3d.encode_image / get_cond.  ``img_pil`` is the output
    of :func:`t1_preprocess`.  Returns ``{"cond": [1,T,1024], "neg_cond": zeros}``.
    """
    from torchvision import transforms
    norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    im = img_pil.resize((518, 518)).convert("RGB")
    x = torch.from_numpy(np.array(im).astype(np.float32) / 255).permute(2, 0, 1)[None]
    x = norm(x).to(dev)
    feats = dino(x, is_training=True)["x_prenorm"]
    cond = torch.nn.functional.layer_norm(feats, feats.shape[-1:])
    return {"cond": cond, "neg_cond": torch.zeros_like(cond)}


__all__ = ["load_t1_ss_flow", "load_t1_dino", "get_rembg_session",
           "t1_preprocess", "t1_get_cond", "DEFAULT_T1_SS_FLOW"]
