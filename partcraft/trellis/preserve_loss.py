"""Optional preserve losses for fine-tuning / research (not used by inference).

PartCraft3D's default pipeline only **infers** with ``TrellisRefiner.edit`` and
``interweave_Trellis_TI``; it does **not** ship a first-party TRELLIS training
loop.  Use the helpers below when wiring a custom trainer.

**Loss spaces (pick one or combine)**

1. ``PreserveLossSpace.SLAT_SPARSE`` — compare SLAT features at occupied
   voxels.  Index with a per-point boolean ``preserve`` derived from the final
   64³ edit mask: ``preserve = ~mask[slat.coords[:,1], coords[:,2], coords[:,3]]``.
   This matches the mask passed into ``interweave_Trellis_TI`` (editable =
   ``mask``).

2. ``PreserveLossSpace.SS_LATENT`` — compare ``z_s`` tensors from
   ``sparse_structure_encoder`` (shape ``[C, R, R, R]``).  A coarse keep-mask
   can be obtained via
   :attr:`partcraft.pipeline_v3.mask_materialization.V2StructMaskBuilder` (``keep16``)
   (16³) **if** ``R == 16``; otherwise resize / reproject to match ``z_s`` spatial
   dims.

3. ``PreserveLossSpace.DECODED_GEOMETRY`` — apply losses after
   ``decode_slat`` (e.g. Gaussian / mesh).  Most expensive; define your own
   renderer or point-cloud term.

See also
:meth:`partcraft.pipeline_v3.mask_materialization.V2StructMaskBuilder.build`
for exporting masks + per-SLAT keep flags.
"""

from __future__ import annotations

import enum
from typing import Literal

import torch

Reduction = Literal["mean", "sum", "none"]


class PreserveLossSpace(str, enum.Enum):
    """Where a preserve / identity loss is computed."""

    SLAT_SPARSE = "slat_sparse"
    SS_LATENT = "ss_latent"
    DECODED_GEOMETRY = "decoded_geometry"


def slat_sparse_preserve_loss(
    feats_before: torch.Tensor,
    feats_after: torch.Tensor,
    preserve: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """L1 loss on SLAT features where ``preserve`` is True.

    Parameters
    ----------
    feats_before, feats_after
        ``[N, C]`` sparse features (same ``coords`` ordering).
    preserve
        ``[N]`` bool — **True** means "do not change this voxel".
    """
    if feats_before.shape != feats_after.shape:
        raise ValueError(
            f"feat shape mismatch: {feats_before.shape} vs {feats_after.shape}"
        )
    if preserve.ndim != 1 or preserve.shape[0] != feats_before.shape[0]:
        raise ValueError(
            f"preserve must be [N], N={feats_before.shape[0]}; got {preserve.shape}"
        )
    if not preserve.any():
        return feats_before.new_zeros(())
    diff = (feats_after - feats_before).abs().mean(dim=-1)
    masked = diff[preserve]
    if reduction == "none":
        return masked
    if reduction == "sum":
        return masked.sum()
    return masked.mean()


def ss_latent_preserve_loss(
    z_before: torch.Tensor,
    z_after: torch.Tensor,
    mask_keep: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """MSE between SS latents inside ``mask_keep`` (True = preserve).

    Parameters
    ----------
    z_before, z_after
        ``[C, R, R, R]`` (or batched ``[B, C, R, R, R]``).
    mask_keep
        Bool tensor matching spatial ``(R, R, R)``, or ``(B, R, R, R)`` when
        ``z_*`` is 5-D.
    """
    if z_before.shape != z_after.shape:
        raise ValueError(
            f"latent shape mismatch: {z_before.shape} vs {z_after.shape}"
        )
    spatial = z_before.shape[-3:]
    m = mask_keep if mask_keep.dtype is torch.bool else mask_keep > 0
    if m.ndim == 3:
        if m.shape != spatial:
            raise ValueError(f"mask_keep spatial {m.shape} != z spatial {spatial}")
        if z_before.ndim == 5:
            m = m.unsqueeze(0).expand(z_before.shape[0], *spatial)
    channel_dim = 1 if z_before.ndim == 5 else 0
    diff = (z_after - z_before).pow(2).mean(dim=channel_dim)
    sel = diff[m]
    if sel.numel() == 0:
        return z_before.new_zeros(())
    if reduction == "none":
        return sel
    if reduction == "sum":
        return sel.sum()
    return sel.mean()


def dense64_preserve_loss(
    grid_before: torch.Tensor,
    grid_after: torch.Tensor,
    mask_keep: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Example helper for dense 64³ grids (e.g. occupancy logits)."""
    if grid_before.shape != grid_after.shape:
        raise ValueError("grid shape mismatch")
    m = mask_keep if mask_keep.dtype is torch.bool else mask_keep > 0
    if m.shape != grid_before.shape[-3:]:
        raise ValueError("mask_keep must match last 3 dims of grids")
    diff = (grid_after - grid_before).pow(2)
    while diff.ndim > 3:
        diff = diff.mean(dim=0)
    sel = diff[m]
    if sel.numel() == 0:
        return grid_before.new_zeros(())
    if reduction == "none":
        return sel
    if reduction == "sum":
        return sel.sum()
    return sel.mean()


def training_integration_note() -> str:
    """Return a short note for downstream trainers (no runtime deps)."""

    return (
        "PartCraft3D has no built-in TRELLIS fine-tune entrypoint. "
        "Wire PreserveLossSpace helpers after your forward pass; keep masks in "
        "the same frame as TrellisRefiner.build_part_mask (64³ VD grid, SLAT-aligned)."
    )


def training_entrypoints_inventory() -> dict[str, list[str]]:
    """Read-only map for fine-tune wiring (this repo is inference-first).

    PartCraft3D ships V3 orchestration and Trellis **inference** helpers; it does
    not include an upstream TRELLIS training script.  Use this map to where
    masks, encoders, and losses connect when you bring your own trainer.
    """

    return {
        "mask_export": [
            "partcraft/pipeline_v3/mask_materialization.py",
        ],
        "trellis_inference_core": [
            "partcraft/trellis/refiner.py",
            "partcraft/pipeline_v3/trellis2_3d.py",
        ],
        "ss_encode_util": [
            "partcraft/io/npz_utils.py (encode_ss)",
        ],
        "notes": [
            "No first-party PyTorch Lightning / Accelerate trainer lives under "
            "partcraft/ for fine-tuning interweave_Trellis_TI.",
            "third_party/TRELLIS training code (if any) is maintained upstream.",
        ],
    }
