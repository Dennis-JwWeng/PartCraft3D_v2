"""S1 — TRELLIS.2 sparse-structure (SS) masked editing.

The modification / scale path must change the *active-voxel set* (occupancy),
not just repaint geometry inside the original footprint — otherwise a part can
only be reshaped within the voxels it already occupies, never grown or shrunk
(this is exactly what v1's ``refiner.build_part_mask`` + ``interweave_Trellis``
S1 repaint did, and what the current P4 path dropped).

TRELLIS.2 reuses TRELLIS.1's SS VAE verbatim — ``ss_enc/dec_conv3d_16l8`` — a
dense ``16³ × 8`` latent; only the SS *flow* model is new (image-conditioned
3.3B DiT, ``ss_flow_img_dit_1_3B_64``).  So the v1 S1-editing recipe ports
directly, swapping the text condition for the original / edited image:

    occupancy(C0 @ 64³) → ss_enc → z_s (16³×8)
      → RF-invert under the ORIGINAL image (cfg off)
      → masked forward repaint under the EDITED image, preserving the 16³
        keep-mask region by re-injecting the inverted trajectory
      → ss_dec → coords_new (64³, voxels grown / shrunk in the edit region)

``coords_new`` then drives the geometry + material stages (see
``trellis2_edit_stages``), which reuse the original latent outside the edit
region via a coord bridge.

Uses ``pipeline_type='1024'`` semantics: SS decodes to 64³ occupancy, which is
exactly the coord resolution the 1024 shape/tex flow models consume — so all
three stages share one 64³ coord set (no cascade upsampling to bridge).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from .trellis2_masked_sampler import make_inverse_anchored_callback
from .trellis2_part_mask import (edit_grid_64_to_keep16,
                                 edit_grid_64_to_keep16_soft)


# v1-shared SS VAE encoder (same checkpoint TRELLIS.2 pairs its SS decoder with).
SS_ENC_NAME = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"


def get_ss_encoder(pipeline, p25_cfg: dict, logger):
    """Load + cache the SS encoder once on the pipeline object."""
    enc = getattr(pipeline, "_pcv2_ss_enc", None)
    if enc is not None:
        return enc
    codebase = str(Path(p25_cfg.get(
        "trellis2_codebase", "/mnt/zsn/3dobject/TRELLIS.2")).resolve())
    if codebase not in sys.path:
        sys.path.insert(0, codebase)
    import trellis2.models as t2_models
    name = p25_cfg.get("trellis2_ss_enc", SS_ENC_NAME)
    logger.info("[s5/S1] loading SS encoder %s", name)
    enc = t2_models.from_pretrained(name).eval().cuda()
    pipeline._pcv2_ss_enc = enc
    return enc


@torch.no_grad()
def edit_structure(
    pipeline,
    ss_enc,
    sampler,
    coords0: torch.Tensor,
    edit_grid64: torch.Tensor,
    cond_orig: dict,
    cond_edit: dict,
    logger,
    keep_thresh: float = 0.1,
    soft_feather: float = 0.0,
    contact_mask: torch.Tensor | None = None,
    contact_sigma: float | None = None,
    ss_param_override: dict | None = None,
) -> torch.Tensor:
    """Masked-edit the sparse structure; return ``coords_new`` ``[M,3]`` int (0..63).

    Args:
        pipeline:    Trellis2ImageTo3DPipeline (provides ss flow/decoder + params).
        ss_enc:      SS encoder from :func:`get_ss_encoder`.
        sampler:     a ``MaskedFlowEulerGuidanceIntervalSampler``.
        coords0:     ``[N,3]`` original active-voxel indices (0..63).
        edit_grid64: dense ``[64,64,64]`` bool edit region (part + pad dilation).
        cond_orig:   512-res image cond for the ORIGINAL view (inversion).
        cond_edit:   512-res image cond for the EDITED view (forward repaint).
        keep_thresh: 16³ block is 'edit' iff ≥ this fraction of its 64 cells are
                     in the 64³ edit region (higher → tighter S1 edit region).
    """
    dev = "cuda"
    ss_flow = pipeline.models["sparse_structure_flow_model"]
    ss_dec = pipeline.models["sparse_structure_decoder"]
    # SS sampler params: ckpt default (steps12/cfg7.5/interval[.6,1]/rt5),
    # optionally overridden — e.g. to benchmark against TRELLIS.1's gentler
    # schedule (steps25/cfg5/interval[.5,1]/rt3), which is far more robust to
    # collapse on LARGE edit regions.
    params = {**pipeline.sparse_structure_sampler_params, **(ss_param_override or {})}
    gi = params.get("guidance_interval", (0.0, 1.0))
    grescale = params.get("guidance_rescale", 0.0)
    steps = int(params["steps"])
    rescale_t = float(params.get("rescale_t", 1.0))
    gs_fwd = float(params.get("guidance_strength", 7.5))
    if ss_param_override:
        logger.info("[s5/S1] SS sampler override: steps=%d cfg=%.1f "
                    "interval=%s rt=%.1f grescale=%.2f", steps, gs_fwd, gi,
                    rescale_t, grescale)

    # occupancy [1,1,64,64,64] from the original active voxels.
    # Feed fp32 (matches data_toolkit/encode_ss_latent.py) so the SS latent is
    # fp32 — the SS flow is normally driven with fp32 noise.
    c = coords0.long().to(dev)
    occ = torch.zeros(1, 1, 64, 64, 64, device=dev)
    occ[0, 0, c[:, 0], c[:, 1], c[:, 2]] = 1.0
    z_s0 = ss_enc(occ.float())

    if pipeline.low_vram:
        ss_flow.to(dev)
    # RF inversion under the ORIGINAL image (guidance_strength=1.0 == cfg off)
    inv = sampler.invert_clean(
        ss_flow, z_s0,
        cond=cond_orig["cond"], neg_cond=cond_orig["neg_cond"],
        guidance_strength=1.0, guidance_interval=gi, guidance_rescale=grescale,
        steps=steps, rescale_t=rescale_t, verbose=False, tqdm_desc="S1 inv",
    )
    # 16³ preserve mask (downsampled ×4 from the 64³ edit region).  Hard =
    # bool (True outside edit, torch.where replace); soft = float keep-weight
    # in [0,1] feathered across the boundary (blend) to heal the junction seam.
    if contact_mask is not None:
        # v1-faithful: contact-aware distance-transform soft mask.  Decay is
        # measured from the contact boundary (edit↔preserved interface) with the
        # dynamic sigma from compute_contact_boundary, so blocks far from any
        # contact stay fully anchored and the junction feathers exactly where
        # the edit meets preserved geometry (mirrors interweave get_s1_soft_mask).
        from .trellis2_contact_mask import get_s1_soft_mask
        sigma = float(contact_sigma) if contact_sigma is not None else 3.0
        keep16 = get_s1_soft_mask(
            edit_grid64, sigma=sigma, contact_mask=contact_mask
        ).to(dev)[None, None]
        logger.info("[s5/S1] contact-soft boundary: sigma=%.2f, "
                    "keep-weight range [%.2f, %.2f]", sigma,
                    float(keep16.min()), float(keep16.max()))
    elif soft_feather > 0:
        keep16 = edit_grid_64_to_keep16_soft(
            edit_grid64, thresh=keep_thresh, feather=soft_feather
        ).to(dev)[None, None]
        logger.info("[s5/S1] soft boundary: feather=%.1f block(s), "
                    "keep-weight range [%.2f, %.2f]", soft_feather,
                    float(keep16.min()), float(keep16.max()))
    else:
        keep16 = edit_grid_64_to_keep16(
            edit_grid64, thresh=keep_thresh).to(dev)[None, None]
    x_init = inv[1.0]
    cb = make_inverse_anchored_callback(inv, keep16)
    z_s_new = sampler.sample(
        ss_flow, x_init,
        cond=cond_edit["cond"], neg_cond=cond_edit["neg_cond"],
        steps=steps, rescale_t=rescale_t,
        guidance_strength=gs_fwd, guidance_interval=gi, guidance_rescale=grescale,
        verbose=False, tqdm_desc="S1 fwd",
        x_init=x_init, step_callback=cb,
    ).samples
    if pipeline.low_vram:
        ss_flow.cpu()

    if pipeline.low_vram:
        ss_dec.to(dev)
    decoded = ss_dec(z_s_new) > 0
    if pipeline.low_vram:
        ss_dec.cpu()
    # argwhere → (b, c, x, y, z); keep (x, y, z)
    coords_new = torch.argwhere(decoded)[:, [0, 2, 3, 4]][:, 1:].int().cpu()
    # Degenerate-edit guard: when the edit region ≈ the whole object (e.g.
    # scaling the main body), the masked S1 repaint can collapse the structure
    # to (near-)empty. Downstream coord-bridge / x_init then hit an empty-tensor
    # max(). Fail here with a clear, catchable reason so the per-edit handler
    # records it cleanly instead of a cryptic error. Threshold: < 32 voxels
    # (a real edited structure has hundreds–thousands).
    n_new = int(coords_new.shape[0])
    n_in = int(c.shape[0])
    if n_new < 32:
        raise ValueError(
            f"S1 structure collapsed: {n_in} → {n_new} voxels "
            f"(edit region too large — likely scaling the main body). "
            f"Skipping this edit."
        )
    return coords_new


__all__ = ["get_ss_encoder", "edit_structure", "SS_ENC_NAME"]
