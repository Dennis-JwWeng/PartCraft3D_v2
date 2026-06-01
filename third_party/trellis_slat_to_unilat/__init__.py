"""
TRELLIS SLAT (.npz) → DINO voxel features → UniLat3D latent bridge.

Vendored from UniLat3D `slat_to_dino_voxel.py`, wired to this repo’s
`third_party/trellis` and `third_party/unilat3d` on `install_import_paths()`.
"""

from .bridge import (
    SlatToUnilatConfig,
    StageTimings,
    decode_slat_to_gaussian,
    encode_with_unilat,
    install_import_paths,
    load_slat,
    render_and_aggregate_dino,
    slat_file_to_unilat,
    third_party_dir,
    dino_from_render_dir,
    render_dir_to_unilat,
)

__all__ = [
    "SlatToUnilatConfig",
    "StageTimings",
    "decode_slat_to_gaussian",
    "encode_with_unilat",
    "install_import_paths",
    "load_slat",
    "render_and_aggregate_dino",
    "slat_file_to_unilat",
    "third_party_dir",
    "dino_from_render_dir",
    "render_dir_to_unilat",
]
