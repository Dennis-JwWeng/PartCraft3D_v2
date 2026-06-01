"""Scene normalization (scale + offset) from packed image NPZ transforms.json.

Used to replay the *same* normalization as the original full-mesh prerender when
rendering partial meshes (e.g. deletion ``after_new.glb``) for SLAT/UniLat
pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class SceneNormalizationError(ValueError):
    """Raised when transforms.json lacks usable scale/offset."""


@dataclass(frozen=True)
class SceneNormalization:
    """VD / prerender scene normalization parameters."""

    scale: float
    offset: tuple[float, float, float]
    source: str

    def as_blender_args(self) -> list[str]:
        ox, oy, oz = self.offset
        return [
            "--normalize_scale",
            f"{float(self.scale):.12g}",
            "--normalize_offset",
            f"{ox:.12g}",
            f"{oy:.12g}",
            f"{oz:.12g}",
        ]


def _json_from_npz_value(raw: Any) -> dict[str, Any]:
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, np.ndarray) and raw.dtype == np.uint8:
        return json.loads(raw.tobytes().decode("utf-8"))
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return json.loads(bytes(raw).decode("utf-8"))
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"unsupported transforms.json payload type: {type(raw)!r}")


def load_transforms_dict_from_image_npz(image_npz: Path | str) -> dict[str, Any]:
    """Load the transforms dict from a packed images NPZ."""
    p = Path(image_npz)
    if not p.is_file():
        raise FileNotFoundError(f"image_npz not found: {p}")
    z = np.load(str(p), allow_pickle=True)
    if "transforms.json" not in z.files:
        raise SceneNormalizationError(f"{p}: missing transforms.json")
    return _json_from_npz_value(z["transforms.json"])


def read_scene_normalization_from_image_npz(image_npz: Path | str) -> SceneNormalization:
    """Read scale + offset from ``image_npz`` transforms.json.

    Missing or invalid fields raise ``SceneNormalizationError``.
    """
    p = Path(image_npz)
    meta = load_transforms_dict_from_image_npz(p)
    scale = meta.get("scale")
    offset = meta.get("offset")
    if scale is None or offset is None:
        raise SceneNormalizationError(
            f"{p}: transforms.json missing scale/offset "
            f"(scale={scale!r}, offset={offset!r})"
        )
    try:
        off_list = [float(v) for v in offset]
    except (TypeError, ValueError) as e:
        raise SceneNormalizationError(f"{p}: invalid offset {offset!r}") from e
    if len(off_list) != 3:
        raise SceneNormalizationError(
            f"{p}: offset must have length 3, got {len(off_list)}: {off_list!r}"
        )
    try:
        sc = float(scale)
    except (TypeError, ValueError) as e:
        raise SceneNormalizationError(f"{p}: invalid scale {scale!r}") from e
    if sc <= 0.0:
        raise SceneNormalizationError(f"{p}: scale must be positive, got {sc}")
    src = f"{p}::transforms.json"
    return SceneNormalization(scale=sc, offset=(off_list[0], off_list[1], off_list[2]), source=src)


__all__ = [
    "SceneNormalization",
    "SceneNormalizationError",
    "load_transforms_dict_from_image_npz",
    "read_scene_normalization_from_image_npz",
]
