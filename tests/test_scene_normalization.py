"""Tests for partcraft.io.scene_normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from partcraft.io.scene_normalization import (
    SceneNormalizationError,
    load_transforms_dict_from_image_npz,
    read_scene_normalization_from_image_npz,
)


def _write_npz(path: Path, transforms: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tf = json.dumps(transforms).encode("utf-8")
    np.savez(path, **{"transforms.json": np.frombuffer(tf, dtype=np.uint8)})


def test_read_ok(tmp_path: Path) -> None:
    p = tmp_path / "obj.npz"
    _write_npz(p, {"scale": 0.5, "offset": [0.1, -0.2, 0.3], "frames": []})
    n = read_scene_normalization_from_image_npz(p)
    assert n.scale == 0.5
    assert n.offset == (0.1, -0.2, 0.3)
    assert str(p) in n.source
    args = n.as_blender_args()
    assert "--normalize_scale" in args
    assert "--normalize_offset" in args


def test_missing_transforms_key(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    np.savez(p, foo=np.array([1]))
    with pytest.raises(SceneNormalizationError, match="missing transforms"):
        load_transforms_dict_from_image_npz(p)


def test_missing_scale(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    _write_npz(p, {"offset": [0, 0, 0]})
    with pytest.raises(SceneNormalizationError, match="missing scale"):
        read_scene_normalization_from_image_npz(p)


def test_bad_offset_length(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    _write_npz(p, {"scale": 1.0, "offset": [0, 0]})
    with pytest.raises(SceneNormalizationError, match="length 3"):
        read_scene_normalization_from_image_npz(p)


def test_non_positive_scale(tmp_path: Path) -> None:
    p = tmp_path / "bad.npz"
    _write_npz(p, {"scale": 0.0, "offset": [0, 0, 0]})
    with pytest.raises(SceneNormalizationError, match="positive"):
        read_scene_normalization_from_image_npz(p)
