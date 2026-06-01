#!/usr/bin/env python3
"""TRELLIS SLAT rendering helpers used by pipeline v3 preview stages.

Historically these helpers lived as a standalone script.  Several pipeline
entrypoints still import this module by name, so keep the small public surface
here instead of duplicating the rendering code at every call site.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

_LAST_SLAT_ID: int | None = None
_LAST_GAUSSIAN: Any = None


def _torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_slat(npz_path: str | Path):
    """Load a TRELLIS/PartCraft SLAT ``.npz`` into a sparse tensor.

    Supports both historical key pairs:
    - ``feats`` + ``coords``
    - ``slat_feats`` + ``slat_coords``
    """
    import torch
    from trellis.modules import sparse as sp

    with np.load(str(npz_path)) as data:
        if "feats" in data and "coords" in data:
            feats_np = np.asarray(data["feats"], dtype=np.float32)
            coords_np = np.asarray(data["coords"], dtype=np.int32)
        elif "slat_feats" in data and "slat_coords" in data:
            feats_np = np.asarray(data["slat_feats"], dtype=np.float32)
            coords_np = np.asarray(data["slat_coords"], dtype=np.int32)
        else:
            raise KeyError("npz must contain feats+coords or slat_feats+slat_coords")

    device = _torch_device()
    feats = torch.from_numpy(feats_np).float().to(device)
    coords = torch.from_numpy(coords_np).int().to(device)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"unexpected SLAT coords shape: {tuple(coords.shape)}")
    if coords.shape[1] == 3:
        batch = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=device)
        coords = torch.cat([batch, coords], dim=1)
    return sp.SparseTensor(feats=feats, coords=coords)


def _frame_pose(frame: dict[str, Any]) -> tuple[float, float, float, float]:
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"invalid transform_matrix shape: {matrix.shape}")
    x, y, z = matrix[:3, 3].tolist()
    radius = float(math.sqrt(x * x + y * y + z * z))
    if radius <= 0:
        raise ValueError("camera radius must be positive")
    # TRELLIS yaw convention: origin = [sin(yaw), cos(yaw), sin(pitch)] * r.
    yaw = float(math.atan2(x, y))
    pitch = float(math.atan2(z, math.sqrt(x * x + y * y)))
    fov = float(frame.get("camera_angle_x", math.radians(40.0)))
    if fov <= math.pi:
        fov = math.degrees(fov)
    return yaw, pitch, radius, fov


def frame_to_extrinsic_intrinsic(frame: dict[str, Any]):
    """Convert a PartVerse ``transforms.json`` frame to TRELLIS cameras."""
    from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    yaw, pitch, radius, fov = _frame_pose(frame)
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, radius, fov)


def _decode_gaussian(pipeline, slat):
    global _LAST_SLAT_ID, _LAST_GAUSSIAN

    sid = id(slat)
    if _LAST_SLAT_ID == sid and _LAST_GAUSSIAN is not None:
        return _LAST_GAUSSIAN

    import torch

    with torch.no_grad():
        decoded = pipeline.decode_slat(slat, ["gaussian"])
    gaussian = decoded["gaussian"][0]
    _LAST_SLAT_ID = sid
    _LAST_GAUSSIAN = gaussian
    return gaussian


def render_one_view(pipeline, slat, frame: dict[str, Any], resolution: int = 518) -> np.ndarray:
    """Render one RGB uint8 image from a TRELLIS SLAT at a PartVerse camera."""
    import torch
    from trellis.utils.render_utils import get_renderer

    gaussian = _decode_gaussian(pipeline, slat)
    extrinsic, intrinsic = frame_to_extrinsic_intrinsic(frame)
    renderer = get_renderer(gaussian, resolution=resolution, bg_color=(1, 1, 1))
    with torch.no_grad():
        res = renderer.render(gaussian, extrinsic, intrinsic)
    color = res["color"].detach().cpu().numpy().transpose(1, 2, 0)
    if color.dtype != np.uint8:
        color = (color * 255).clip(0, 255).astype(np.uint8)
    return color


__all__ = ["frame_to_extrinsic_intrinsic", "load_slat", "render_one_view"]
