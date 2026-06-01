"""Shared NPZ save/load and SS encoding utilities.

Used by ``migrate_slat_to_npz.py`` and ``trellis_refine.py`` to avoid
duplicating tensor → numpy → npz logic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_npz(
    path: Path,
    feats: torch.Tensor,
    coords: torch.Tensor,
    ss: torch.Tensor | None = None,
) -> None:
    """Save SLAT + optional SS to NPZ.

    Parameters
    ----------
    feats : Tensor [N, C]
    coords : Tensor [N, 4]  (batch, x, y, z)
    ss : Tensor [C, R, R, R] or None
    """
    data: dict[str, np.ndarray] = {
        "slat_feats": feats.detach().cpu().float().numpy(),
        "slat_coords": coords.detach().cpu().int().numpy(),
    }
    if ss is not None:
        data["ss"] = ss.detach().cpu().float().numpy()
    np.savez(path, **data)


@torch.no_grad()
def encode_ss(
    encoder: torch.nn.Module,
    coords: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode voxel coordinates into SS VAE latent ``z_s [C, R, R, R]``.

    Builds a 64^3 binary occupancy grid from ``coords`` and runs the
    sparse-structure encoder.
    """
    occ = torch.zeros(1, 1, 64, 64, 64, device=device)
    occ[0, 0, coords[:, 1], coords[:, 2], coords[:, 3]] = 1
    z_s = encoder(occ)
    return z_s.squeeze(0)


def load_ss_encoder(ckpt_root: Path, device: str = "cuda"):
    """Load only the sparse-structure VAE encoder.

    The old implementation instantiated the full TRELLIS text pipeline and then
    pulled ``sparse_structure_encoder`` from it. For batch latent backfills this
    wastes startup time and GPU memory; SS encoding only needs this one module.
    """
    import logging

    import trellis.models as trellis_models

    log = logging.getLogger(__name__)
    ckpt_root = Path(ckpt_root)
    candidates = [
        ckpt_root / "TRELLIS-text-xlarge" / "ckpts" / "ss_enc_conv3d_16l8_fp16",
        ckpt_root / "TRELLIS-image-large" / "ckpts" / "ss_enc_conv3d_16l8_fp16",
    ]
    ss_enc_path = next(
        (p for p in candidates if p.with_suffix(".json").is_file() or p.with_suffix(".safetensors").is_file()),
        candidates[0],
    )
    log.info("Loading TRELLIS sparse-structure encoder from %s ...", ss_enc_path)
    encoder = trellis_models.from_pretrained(str(ss_enc_path)).eval()
    encoder = encoder.cuda() if device == "cuda" else encoder.cpu()
    log.info("SS encoder ready on %s", device)
    return encoder
