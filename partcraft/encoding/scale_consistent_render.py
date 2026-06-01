"""Scale-consistent Blender multi-view rendering for SLAT / UniLat pipelines.

``encode_asset.render_img_for_enc.render`` does not pass ``--normalize_*``;
this module calls ``third_party/encode_asset/blender_script/render.py`` directly
so partial meshes reuse the original prerender normalization.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from partcraft.io.scene_normalization import SceneNormalization, read_scene_normalization_from_image_npz

_LOG = logging.getLogger(__name__)

# ---- Low-discrepancy views (must match encode_asset/render_img_for_enc.py) ----

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def _radical_inverse(base: int, n: int) -> float:
    val = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def _halton_sequence(dim: int, n: int) -> list[float]:
    return [_radical_inverse(_PRIMES[dim], n) for dim in range(dim)]


def _hammersley_sequence(dim: int, n: int, num_samples: int) -> list[float]:
    return [n / num_samples] + _halton_sequence(dim - 1, n)


def _sphere_hammersley_sequence(
    n: int, num_samples: int, offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    u, v = _hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = float(np.arccos(1 - 2 * u) - np.pi / 2)
    phi = float(v * 2 * np.pi)
    return phi, theta


def build_hammersley_views(num_views: int) -> list[dict[str, Any]]:
    """Return view dicts matching ``encode_asset.render_img_for_enc.render``."""
    views: list[dict[str, Any]] = []
    for i in range(int(num_views)):
        yaw, pitch = _sphere_hammersley_sequence(i, int(num_views))
        views.append(
            {
                "yaw": yaw,
                "pitch": pitch,
                "radius": 2,
                "fov": 40 / 180 * np.pi,
            }
        )
    return views


def repo_root() -> Path:
    """Repository root (``PartCraft3D/``)."""
    return Path(__file__).resolve().parents[2]


def default_encode_asset_blender_script() -> Path:
    p = repo_root() / "third_party" / "encode_asset" / "blender_script" / "render.py"
    if not p.is_file():
        raise FileNotFoundError(f"encode_asset blender script not found: {p}")
    return p


def resolve_blender_path(explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    return os.environ.get("BLENDER_PATH", "/usr/local/bin/blender")


def build_scale_consistent_blender_cmd(
    *,
    mesh_path: Path,
    output_folder: Path,
    num_views: int,
    resolution: int,
    blender_path: str,
    blender_script: Path,
    normalization: SceneNormalization | None,
    engine: str = "CYCLES",
    save_mesh: bool = True,
    allow_self_normalize: bool = False,
    blender_threads: int | None = None,
) -> list[str]:
    """Build argv for ``blender -b -P render.py -- ...`` (for tests / debugging)."""
    if not allow_self_normalize and normalization is None:
        raise ValueError(
            "normalization is required unless allow_self_normalize=True "
            "(partial mesh renders must reuse reference scale/offset)"
        )
    views = build_hammersley_views(int(num_views))
    cmd: list[str] = [
        str(blender_path),
        "-b",
    ]
    if blender_threads is not None and int(blender_threads) > 0:
        cmd += ["-t", str(int(blender_threads))]
    cmd += [
        "-P",
        str(blender_script),
        "--",
        "--views",
        json.dumps(views),
        "--object",
        str(mesh_path.resolve()),
        "--resolution",
        str(int(resolution)),
        "--output_folder",
        str(output_folder),
        "--engine",
        str(engine),
    ]
    if save_mesh:
        cmd.append("--save_mesh")
    if normalization is not None:
        cmd.extend(normalization.as_blender_args())
    return cmd


def render_mesh_for_slat_encode(
    mesh_path: Path,
    name: str,
    work_dir: Path,
    *,
    reference_image_npz: Path | None,
    num_views: int = 40,
    resolution: int = 512,
    blender_path: str | None = None,
    blender_script: Path | None = None,
    engine: str = "CYCLES",
    save_mesh: bool = True,
    allow_self_normalize: bool = False,
    subprocess_timeout: int = 3600,
) -> Path:
    """Run Blender multi-view render + voxelize; returns render output directory.

    The output directory is ``work_dir / name`` (same layout as legacy
    ``migrate_slat_to_npz._render_ply_views``).
    """
    mesh_path = Path(mesh_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    render_out = work_dir / name
    if render_out.exists():
        # Caller may pass unique names; if exists, reuse (resume not handled here).
        pass
    render_out.mkdir(parents=True, exist_ok=True)

    bpath = resolve_blender_path(blender_path)
    script = blender_script or default_encode_asset_blender_script()

    normalization: SceneNormalization | None = None
    if reference_image_npz is not None:
        normalization = read_scene_normalization_from_image_npz(reference_image_npz)
    elif not allow_self_normalize:
        raise ValueError(
            "reference_image_npz is required unless allow_self_normalize=True"
        )

    threads_raw = os.environ.get("BLENDER_THREADS", "0")
    try:
        threads = int(threads_raw)
    except ValueError:
        threads = 0

    cmd = build_scale_consistent_blender_cmd(
        mesh_path=mesh_path,
        output_folder=render_out,
        num_views=num_views,
        resolution=resolution,
        blender_path=bpath,
        blender_script=script,
        normalization=normalization,
        engine=engine,
        save_mesh=save_mesh,
        allow_self_normalize=allow_self_normalize,
        blender_threads=threads,
    )
    _LOG.debug("scale_consistent_render cmd: %s", " ".join(cmd))
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(subprocess_timeout),
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Blender render failed (exit {res.returncode}) for {mesh_path}:\n"
            f"{res.stderr[-4000:]}"
        )
    mesh_ply = render_out / "mesh.ply"
    if not mesh_ply.is_file():
        raise FileNotFoundError(f"Blender produced no mesh.ply under {render_out}")

    # Voxelize using encode_asset helper (matches legacy migration path).
    _third_party = repo_root() / "third_party"
    import sys

    third_s = str(_third_party)
    if third_s not in sys.path:
        sys.path.insert(0, third_s)
    if blender_path:
        os.environ["BLENDER_PATH"] = str(blender_path)
    import encode_asset.render_img_for_enc as _rim  # type: ignore

    if blender_path:
        _rim.BLENDER_PATH = str(blender_path)
    from encode_asset.render_img_for_enc import voxelize  # type: ignore

    voxelize(str(mesh_ply), name, str(render_out))
    return render_out


__all__ = [
    "build_hammersley_views",
    "build_scale_consistent_blender_cmd",
    "default_encode_asset_blender_script",
    "render_mesh_for_slat_encode",
    "repo_root",
    "resolve_blender_path",
]
