"""
TRELLIS sparse latent (SLAT) .npz → DINOv2 per-voxel features → UniLat3D encoder.

Supports ``feats``+``coords`` (N,3) TRELLIS export, or PartCraft ``slat_feats``+``slat_coords`` (N,4, batch in col0).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path as PathType
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

Path = PathType

@dataclass
class SlatToUnilatConfig:
    trellis_ckpt_dir: Union[str, Path]
    unilat_ckpt_dir: Union[str, Path]
    num_views: int = 150
    render_resolution: int = 512
    fov: float = 40.0
    cam_distance: float = 2.0
    device: str = "cuda:0"
    # If True, saves per-view PNGs under ``<output_dir>/renders/``.
    save_renders: bool = False
    save_dino_pt: bool = True
    save_voxels_ply: bool = True
    save_unilat_pt: bool = True
    skip_unilat: bool = False
    # Internal model cache – populated on first use, reused across items.
    _trellis_decoder: Any = field(default=None, init=False, repr=False, compare=False)
    _dinov2: Any = field(default=None, init=False, repr=False, compare=False)
    _unilat_encoder: Any = field(default=None, init=False, repr=False, compare=False)


@dataclass
class StageTimings:
    load_slat_s: float = 0.0
    decode_gs_s: float = 0.0
    dino_render_s: float = 0.0
    save_aux_s: float = 0.0
    unilat_encode_s: float = 0.0
    total_s: float = 0.0


def third_party_dir() -> Path:
    """``PartCraft3D/third_party`` (parent of this package)."""
    return Path(__file__).resolve().parent.parent


def install_import_paths(third: Optional[Path] = None) -> Path:
    """
    Insert ``third_party`` on ``sys.path`` so ``trellis`` and ``unilat3d`` import
    from this repository. Call once per process before importing those packages.
    """
    root = third or third_party_dir()
    s = str(root.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def _voxel_indices_xyz_from_npz_data(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """(N,3) grid indices in [0, 64) for DINO and UniLat encoder paths."""
    if "coords" in data:
        c = np.asarray(data["coords"], dtype=np.int32)
    elif "slat_coords" in data:
        c = np.asarray(data["slat_coords"], dtype=np.int32)
    else:
        raise KeyError("npz must contain 'coords' or 'slat_coords'")
    if c.ndim != 2 or c.shape[1] not in (3, 4):
        raise ValueError(f"unexpected coord shape: {c.shape}")
    if c.shape[1] == 4:
        return c[:, 1:4].copy()
    return c.copy()


def load_slat(npz_path: Union[str, Path], device: torch.device) -> Any:
    install_import_paths()
    from trellis.modules import sparse as sp

    data = np.load(str(npz_path))
    if "feats" in data and "coords" in data:
        feats = torch.from_numpy(np.asarray(data["feats"], np.float32)).float().to(device)
        c = np.asarray(data["coords"], dtype=np.int32)
    elif "slat_feats" in data and "slat_coords" in data:
        feats = torch.from_numpy(np.asarray(data["slat_feats"], np.float32)).float().to(device)
        c = np.asarray(data["slat_coords"], dtype=np.int32)
    else:
        raise KeyError("npz must have feats+coords or slat_feats+slat_coords")
    if c.shape[1] == 3:
        batch_idx = torch.zeros(c.shape[0], 1, dtype=torch.int32, device=device)
        coords = torch.from_numpy(c).int().to(device)
        coords_4d = torch.cat([batch_idx, coords], dim=1)
    else:
        coords_4d = torch.from_numpy(c).int().to(device)
    return sp.SparseTensor(feats=feats, coords=coords_4d)


def decode_slat_to_gaussian(slat, ckpt_dir: Union[str, Path], device: torch.device, *, _cached_decoder: Any = None) -> Any:
    install_import_paths()
    import trellis.models as trellis_models

    if _cached_decoder is not None:
        decoder = _cached_decoder
        own_decoder = False
    else:
        ck = Path(ckpt_dir) / "slat_dec_gs_swin8_B_64l8gs32_fp16"
        decoder = trellis_models.from_pretrained(str(ck)).to(device).eval()
        own_decoder = True
    gaussians = decoder(slat)
    if own_decoder:
        del decoder
        torch.cuda.empty_cache()
    gs = gaussians[0]
    gs.aabb = gs.aabb.to(device)
    gs.scale_bias = gs.scale_bias.to(device)
    gs.rots_bias = gs.rots_bias.to(device)
    gs.opacity_bias = gs.opacity_bias.to(device)
    return gs


def _build_cameras(
    num_views: int, r: float = 2.0, fov: float = 40.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    install_import_paths()
    from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    from trellis.utils.random_utils import sphere_hammersley_sequence

    cams = [sphere_hammersley_sequence(i, num_views) for i in range(num_views)]
    yaws = [c[0] for c in cams]
    pitchs = [c[1] for c in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    return extrinsics, intrinsics


def _load_dinov2(device: torch.device) -> Any:
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
    dinov2.eval().to(device)
    return dinov2


@torch.no_grad()
def render_and_aggregate_dino(
    gaussian: Any,
    coords_int: np.ndarray,
    num_views: int = 150,
    render_resolution: int = 512,
    fov: float = 40.0,
    r: float = 2.0,
    device: Optional[torch.device] = None,
    save_renders_dir: Optional[Union[str, Path]] = None,
    *,
    _cached_dinov2: Any = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from PIL import Image
    import utils3d

    install_import_paths()
    from trellis.utils.render_utils import get_renderer

    N = coords_int.shape[0]
    positions = torch.from_numpy(coords_int).float().to(device) / 64.0 - 0.5
    extrinsics, intrinsics = _build_cameras(num_views, r=r, fov=fov)
    renderer = get_renderer(gaussian, resolution=render_resolution, bg_color=(1, 1, 1))
    if _cached_dinov2 is not None:
        dinov2 = _cached_dinov2
    else:
        dinov2 = _load_dinov2(device)
    dino_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    n_patch = 518 // 14

    if save_renders_dir is not None:
        os.makedirs(str(save_renders_dir), exist_ok=True)

    feat_accum = np.zeros((N, 1024), dtype=np.float64)

    for i in tqdm(range(num_views), desc="Render+DINOv2"):
        color = renderer.render(gaussian, extrinsics[i], intrinsics[i])["color"]
        if save_renders_dir is not None:
            img_np = (color.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(os.path.join(str(save_renders_dir), f"{i:03d}.png"))
        img = F.interpolate(color.unsqueeze(0), size=(518, 518), mode="bilinear", align_corners=False)
        img = dino_norm(img.squeeze(0)).unsqueeze(0)
        features = dinov2(img, is_training=True)
        patchtokens = (
            features["x_prenorm"][:, dinov2.num_register_tokens + 1 :]
            .permute(0, 2, 1)
            .reshape(1, 1024, n_patch, n_patch)
        )
        uv = utils3d.torch.project_cv(
            positions,
            extrinsics[i].unsqueeze(0).to(device),
            intrinsics[i].unsqueeze(0).to(device),
        )[0] * 2 - 1
        sampled = (
            F.grid_sample(
                patchtokens,
                uv.unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(2)
            .permute(0, 2, 1)
        )
        feat_accum += sampled[0].detach().cpu().numpy()

    del dinov2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (feat_accum / num_views).astype(np.float16)


def save_voxels_ply(coords_int: np.ndarray, path: Union[str, Path]) -> None:
    from plyfile import PlyData, PlyElement

    pos = coords_int.astype(np.float32) / 64.0 - 0.5
    verts = np.array([(p[0], p[1], p[2]) for p in pos], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    PlyData([PlyElement.describe(verts, "vertex")]).write(str(path))


@torch.no_grad()
def encode_with_unilat(
    dino_feats: torch.Tensor,
    coords_int: np.ndarray,
    ckpt_dir: Union[str, Path],
    device: torch.device,
    *,
    _cached_encoder: Any = None,
) -> torch.Tensor:
    install_import_paths()
    from unilat3d import models as unilat_models
    from unilat3d.modules.sparse import SparseTensor

    if _cached_encoder is not None:
        encoder = _cached_encoder
        own_encoder = False
    else:
        enc_path = str(Path(ckpt_dir) / "encoder")
        encoder = unilat_models.from_pretrained(enc_path).to(device).eval()
        own_encoder = True
    indices = torch.from_numpy(coords_int).long().to(device)
    batch_idx = torch.zeros(indices.shape[0], 1, dtype=torch.long, device=device)
    full_coords = torch.cat([batch_idx, indices], dim=1).int()
    sparse_input = SparseTensor(feats=dino_feats.float().to(device), coords=full_coords)
    unilat = encoder(sparse_input, sample_posterior=False)
    if own_encoder:
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return unilat


@torch.no_grad()
def slat_file_to_unilat(
    slat_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: SlatToUnilatConfig,
    timings: Optional[StageTimings] = None,
) -> Dict[str, Any]:
    os.environ.setdefault("SPCONV_ALGO", "native")
    t0 = time.perf_counter()
    out: Dict[str, Any] = {}
    if timings is None:
        timings = StageTimings()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slat_path = Path(slat_path)
    device = torch.device(config.device)
    tr_dir = Path(config.trellis_ckpt_dir)
    un_dir = Path(config.unilat_ckpt_dir)

    t1 = time.perf_counter()
    d_npz = np.load(str(slat_path))
    slat = load_slat(slat_path, device)
    coords_int = _voxel_indices_xyz_from_npz_data(d_npz)
    timings.load_slat_s = time.perf_counter() - t1

    t1 = time.perf_counter()
    if config._trellis_decoder is None:
        install_import_paths()
        import trellis.models as trellis_models
        ck = Path(config.trellis_ckpt_dir) / "slat_dec_gs_swin8_B_64l8gs32_fp16"
        config._trellis_decoder = trellis_models.from_pretrained(str(ck)).to(device).eval()
    gaussian = decode_slat_to_gaussian(slat, tr_dir, device, _cached_decoder=config._trellis_decoder)
    n_gs = gaussian.get_xyz.shape[0]
    out["num_gaussians"] = int(n_gs)
    del slat
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timings.decode_gs_s = time.perf_counter() - t1

    renders_path = (output_dir / "renders") if config.save_renders else None
    t1 = time.perf_counter()
    if config._dinov2 is None:
        config._dinov2 = _load_dinov2(device)
    dino_np = render_and_aggregate_dino(
        gaussian,
        coords_int,
        num_views=config.num_views,
        render_resolution=config.render_resolution,
        fov=config.fov,
        r=config.cam_distance,
        device=device,
        save_renders_dir=renders_path,
        _cached_dinov2=config._dinov2,
    )
    del gaussian
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timings.dino_render_s = time.perf_counter() - t1

    dino_tensor = torch.from_numpy(dino_np)
    out["dino"] = dino_tensor
    t_save = time.perf_counter()
    if config.save_dino_pt:
        dp = output_dir / "dino_voxel_mean.pt"
        torch.save(dino_tensor, dp)
        out["dino_voxel_mean_path"] = str(dp)
    if config.save_voxels_ply:
        vp = output_dir / "voxels.ply"
        save_voxels_ply(coords_int, vp)
        out["voxels_ply_path"] = str(vp)
    timings.save_aux_s = time.perf_counter() - t_save

    if not config.skip_unilat and config.save_unilat_pt:
        t1 = time.perf_counter()
        if config._unilat_encoder is None:
            install_import_paths()
            from unilat3d import models as unilat_models
            enc_path = str(Path(config.unilat_ckpt_dir) / "encoder")
            config._unilat_encoder = unilat_models.from_pretrained(enc_path).to(device).eval()
        unilat = encode_with_unilat(dino_tensor, coords_int, un_dir, device, _cached_encoder=config._unilat_encoder)
        out["unilat"] = unilat
        up = output_dir / "unilat.pt"
        torch.save(unilat.cpu(), up)
        out["unilat_path"] = str(up)
        timings.unilat_encode_s = time.perf_counter() - t1
    else:
        out["unilat"] = None

    timings.total_s = time.perf_counter() - t0
    out["timings"] = timings
    return out


# ---------------------------------------------------------------------------
# Fast path: from existing Blender render dir (skips SLAT → Gaussian → render)
# ---------------------------------------------------------------------------

@torch.no_grad()
def dino_from_render_dir(
    render_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    num_views: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate DINOv2 features from a pre-rendered Blender work dir.

    The render dir must contain:
        <NNN>.png        — Blender-rendered views (000.png, 001.png, …)
        transforms.json  — NeRF-convention c2w matrices + camera_angle_x
        voxels.ply       — Open3D 64³ voxel grid (PLY with x,y,z in [-0.5, 0.5])

    Camera convention: Blender c2w → OpenCV w2c via ``c2w[:3, 1:3] *= -1``.
    This matches ``encode_asset.encode_into_SLAT.encode_into_SLAT`` exactly.

    Returns:
        dino_feats  — (N, 1024) float16 DINOv2 voxel features
        indices     — (N, 3) int64 voxel grid indices in [0, 64)
    """
    import json as _json
    import math as _math

    from PIL import Image as _Image
    from plyfile import PlyData as _PlyData

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    render_dir = Path(render_dir)

    # --- voxels.ply → positions ---
    ply = _PlyData.read(render_dir / "voxels.ply")
    vd = ply["vertex"].data
    ply_pos = np.stack([vd["x"], vd["y"], vd["z"]], axis=1).astype(np.float64)  # (N,3)
    # Invert: PLY stored as coords/64 - 0.5, recover integer indices
    indices = np.round((ply_pos + 0.5) * 64).astype(np.int64)  # (N,3)
    positions = torch.from_numpy(
        (indices.astype(np.float32) / 64.0 - 0.5)
    ).to(device)  # (N,3)

    # --- transforms.json ---
    transforms = _json.loads((render_dir / "transforms.json").read_text())
    frames = transforms["frames"]
    if num_views is None:
        num_views = len(frames)
    else:
        num_views = min(num_views, len(frames))

    # --- DINOv2 ---
    install_import_paths()
    dinov2 = _load_dinov2(device)
    from torchvision import transforms as _tvt
    dino_norm = _tvt.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    n_patch = 518 // 14
    N = positions.shape[0]
    feat_accum = np.zeros((N, 1024), dtype=np.float64)

    for i in range(num_views):
        frame = frames[i]
        png = render_dir / frame["file_path"]
        img = _Image.open(png).convert("RGB").resize((518, 518), _Image.LANCZOS)
        img_t = torch.from_numpy(
            np.array(img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        img_t = dino_norm(img_t).unsqueeze(0).to(device)

        features = dinov2(img_t, is_training=True)
        patch = (
            features["x_prenorm"][:, dinov2.num_register_tokens + 1:]
            .permute(0, 2, 1)
            .reshape(1, 1024, n_patch, n_patch)
        )

        # Blender c2w → OpenCV w2c
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        extrinsic = torch.inverse(c2w).unsqueeze(0).to(device)
        fov = frame["camera_angle_x"]
        intrinsic = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(float(fov)), torch.tensor(float(fov))).unsqueeze(0).to(device)

        uv_cv, _ = utils3d.torch.project_cv(positions, extrinsic, intrinsic)
        uv = uv_cv * 2 - 1
        sampled = (
            F.grid_sample(patch, uv.unsqueeze(1), mode="bilinear", align_corners=False)
            .squeeze(2)
            .permute(0, 2, 1)
        )
        feat_accum += sampled[0].detach().cpu().numpy()

    del dinov2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dino_feats = (feat_accum / num_views).astype(np.float16)
    return dino_feats, indices





@torch.no_grad()
def render_dir_to_unilat(
    render_dir: Union[str, Path],
    output_dir: Union[str, Path],
    unilat_ckpt_dir: Union[str, Path],
    device: str = "cuda:0",
    num_views: Optional[int] = None,
    save_dino_pt: bool = False,
    save_unilat_pt: bool = True,
    timings: Optional[StageTimings] = None,
) -> dict:
    """Fast UniLat encode from an existing Blender render dir.

    Skips SLAT loading, Gaussian decoding, and Trellis rendering entirely.
    Requires ``voxels.ply`` + ``transforms.json`` + ``<NNN>.png`` in *render_dir*.

    Args:
        render_dir:       Path to the Blender render work directory.
        output_dir:       Where to write ``unilat.pt`` (and optionally ``dino_voxel_mean.pt``).
        unilat_ckpt_dir:  UniLat3D checkpoint directory.
        device:           CUDA device string (default ``"cuda:0"``).
        num_views:        How many views to use (default: all in transforms.json).
        save_dino_pt:     Also save ``dino_voxel_mean.pt``.
        save_unilat_pt:   Save ``unilat.pt`` (default True).
        timings:          Optional StageTimings to fill.

    Returns:
        dict with ``unilat``, ``dino``, ``unilat_path``, ``dino_voxel_mean_path``, ``timings``.
    """
    t0 = time.perf_counter()
    if timings is None:
        timings = StageTimings()
    out: dict = {}

    render_dir = Path(render_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    # --- DINOv2 from existing renders ---
    t1 = time.perf_counter()
    dino_np, indices = dino_from_render_dir(render_dir, dev, num_views)
    timings.dino_render_s = time.perf_counter() - t1

    dino_tensor = torch.from_numpy(dino_np)
    out["dino"] = dino_tensor
    if save_dino_pt:
        dp = output_dir / "dino_voxel_mean.pt"
        torch.save(dino_tensor, dp)
        out["dino_voxel_mean_path"] = str(dp)

    # --- UniLat encode ---
    t1 = time.perf_counter()
    coords_int = indices.astype(np.int32)  # (N,3) int32 grid indices
    unilat = encode_with_unilat(dino_tensor, coords_int, unilat_ckpt_dir, dev)
    timings.unilat_encode_s = time.perf_counter() - t1

    out["unilat"] = unilat
    if save_unilat_pt:
        up = output_dir / "unilat.pt"
        torch.save(unilat.cpu(), up)
        out["unilat_path"] = str(up)

    timings.total_s = time.perf_counter() - t0
    out["timings"] = timings
    return out
