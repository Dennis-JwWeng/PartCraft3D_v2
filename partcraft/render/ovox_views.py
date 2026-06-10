"""Reusable view renderer with explicit **named** camera transforms.

Single source of truth for the pipeline's render viewpoints.  Replaces the old
``overview column → VIEW_INDICES[col] → sphere_hammersley(frame, 150)`` index
mapping with explicit named cameras (``front / right / back / left / down``), so
renders no longer depend on the packed 150-view set or input images.

Two render backends share the SAME named cameras, so a "before" (mesh → o-voxel)
and an "after" (edited latents → decoded mesh) render line up view-for-view:

  - :func:`render_ovoxel`  — ``o_voxel.rasterize.VoxelRenderer`` (coloured sparse
    voxels; the mesh / part-highlight path).
  - :func:`render_sample`  — TRELLIS ``render_frames`` / ``PbrMeshRenderer`` (a
    decoded SLat mesh; the gate-E "after" path).

Frame: TRELLIS / o-voxel canonical **Z-up**, camera looks at the origin with
up = +Z.  yaw 0 = +X side; pitch > 0 = above (looking down), pitch < 0 = below
(looking up).  Camera azimuth uses ``(cos, sin)`` (Blender pre-render
convention) so both backends agree.  Runs in the ``trellis2`` env (needs
``utils3d``; ``render_ovoxel`` needs ``o_voxel``; ``render_sample`` needs the
TRELLIS codebase).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

VIEW_RADIUS = 2.0
VIEW_FOV_DEG = 40.0

# name → (yaw_deg, pitch_deg).  up = +Z, yaw 0 = +X side, pitch > 0 = above
# (looking down).  front/right/back/left are AXIS-ALIGNED (0/90/180/270) so each
# directly faces one side of the object (orthographic-like front view, NOT the
# old ~45°-offset corner/3-quarter angles), at a slight overhead tilt (~22°).
# ``down`` = low front-upward (legacy overview column 4).
# ``top`` / ``bottom`` = near-orthogonal vertical views for 6-view before/after.
NAMED_VIEWS: dict[str, tuple[float, float]] = {
    "front": (0.0,    22.0),
    "right": (90.0,   22.0),
    "back":  (180.0,  22.0),
    "left":  (270.0,  22.0),
    "down":  (22.5000, -63.2951),   # legacy overview column 4
    "top":   (0.0,    89.0),        # +Z down (俯视)
    "bottom": (0.0,  -89.0),        # -Z up (仰视)
}
# Gate-A / overview pipeline (5 columns, unchanged).
VIEW_ORDER: list[str] = ["front", "right", "back", "left", "down"]
# Part-edit before/after compare: 前后左右上下.
SIX_VIEW_ORDER: list[str] = ["front", "back", "left", "right", "top", "bottom"]


def view_yaw_pitch(name: str) -> tuple[float, float]:
    """Named view → (yaw, pitch) in radians."""
    yd, pd = NAMED_VIEWS[name]
    return math.radians(yd), math.radians(pd)


def named_cameras(
    view_names: Sequence[str],
    *,
    r: float = VIEW_RADIUS,
    fov_deg: float = VIEW_FOV_DEG,
    device: str = "cuda",
):
    """(extrinsics, intrinsics) lists for the named views (look-at, Z-up)."""
    import utils3d

    extr_list, intr_list = [], []
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    look = torch.tensor([0.0, 0.0, 0.0], device=device)
    for nm in view_names:
        yaw, pitch = view_yaw_pitch(nm)
        fov = torch.deg2rad(torch.tensor(float(fov_deg), device=device))
        y = torch.tensor(yaw, device=device)
        p = torch.tensor(pitch, device=device)
        orig = torch.tensor([
            torch.cos(y) * torch.cos(p),
            torch.sin(y) * torch.cos(p),
            torch.sin(p),
        ], device=device) * r
        extr_list.append(utils3d.torch.extrinsics_look_at(orig, look, up))
        intr_list.append(utils3d.torch.intrinsics_from_fov_xy(fov, fov))
    return extr_list, intr_list


def camera_transforms(
    view_names: Sequence[str] | None = None,
    *,
    r: float = VIEW_RADIUS,
    fov_deg: float = VIEW_FOV_DEG,
) -> list[dict]:
    """Serializable per-view camera record (for ``viewpoints.json``)."""
    view_names = list(view_names or VIEW_ORDER)
    out = []
    for nm in view_names:
        yaw, pitch = view_yaw_pitch(nm)
        out.append({"name": nm, "yaw": yaw, "pitch": pitch,
                    "radius": float(r), "fov": math.radians(fov_deg)})
    return out


def shade(
    base_color: np.ndarray,
    normal: np.ndarray,
    *,
    light: tuple[float, float, float] = (0.5, 0.4, 1.0),
    ambient: float = 0.45,
    diffuse: float = 0.7,
) -> np.ndarray:
    """Lambertian-shade per-voxel ``base_color`` by ``normal`` → uint8 [N,3].

    Fixed world-space light (canonical frame) so the shading is view-independent
    and a "before" and "after" voxel shade identically.  Accepts uint8 (0..255)
    or float (0..1) inputs; ``normal`` may be raw uint8 (0..255 encoding n*.5+.5)
    or float [-1,1].
    """
    bc = np.asarray(base_color, np.float32)
    if bc.max() > 1.5:
        bc = bc / 255.0
    nm = np.asarray(normal, np.float32)
    if nm.max() > 1.5:
        nm = nm / 255.0 * 2.0 - 1.0
    L = np.asarray(light, np.float32); L = L / (np.linalg.norm(L) + 1e-8)
    d = np.clip((nm * L).sum(-1, keepdims=True), 0.0, 1.0)
    shaded = np.clip(bc * (ambient + diffuse * d), 0.0, 1.0)
    return (shaded * 255).astype(np.uint8)


def render_ovoxel(
    coords: np.ndarray,
    colors: np.ndarray,
    grid_size: int,
    view_names: Sequence[str] | None = None,
    *,
    resolution: int = 512,
    ssaa: int = 2,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Rasterise coloured voxels at named views → ``{name: RGB uint8 [H,W,3]}``.

    ``colors`` may be uint8 (0..255) or float (0..1).  Empty pixels composite
    onto ``bg``.
    """
    import o_voxel

    view_names = list(view_names or VIEW_ORDER)
    pos = torch.from_numpy(np.asarray(coords, np.float32)).to(device) / grid_size - 0.5
    c = np.asarray(colors)
    if c.dtype not in (np.float32, np.float64):
        c = c.astype(np.float32) / 255.0
    attrs = torch.from_numpy(c.astype(np.float32)).to(device)

    return render_voxel_positions(pos, attrs, 1.0 / grid_size, view_names,
                                  resolution=resolution, ssaa=ssaa, bg=bg, device=device)


def render_voxel_positions(
    position,
    attrs,
    voxel_size: float,
    view_names: Sequence[str] | None = None,
    *,
    resolution: int = 512,
    ssaa: int = 2,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Rasterise voxels given **world positions** (N,3) + per-voxel ``attrs``.

    The lower-level path shared by the mesh o-voxel ("before") and a decoded
    ``MeshWithVoxel``'s own voxel ("after"), so both render identically.
    """
    import o_voxel

    view_names = list(view_names or VIEW_ORDER)
    pos = position if torch.is_tensor(position) else torch.from_numpy(np.asarray(position, np.float32))
    pos = pos.to(device).float()
    a = attrs if torch.is_tensor(attrs) else torch.from_numpy(np.asarray(attrs, np.float32))
    a = a.to(device).float()
    if a.max() > 1.5:
        a = a / 255.0
    extr, intr = named_cameras(view_names, device=device)
    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": resolution, "ssaa": ssaa})
    bg_t = torch.tensor(bg, device=device).view(3, 1, 1)
    out: dict[str, np.ndarray] = {}
    for nm, e, i in zip(view_names, extr, intr):
        ret = renderer.render(pos, a, float(voxel_size), e, i)
        alpha = ret.alpha.clamp(0, 1)[None]
        comp = ret.attr * alpha + bg_t * (1 - alpha)
        out[nm] = (comp.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return out


def load_envmap(hdri_path: str, device: str = "cuda"):
    """Load an HDR ``.exr`` into a TRELLIS ``EnvMap`` (for the PBR mesh render)."""
    import cv2
    from trellis2.renderers import EnvMap
    hdr = cv2.cvtColor(cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED),
                       cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(hdr, dtype=torch.float32, device=device))


def render_sample(
    sample,
    view_names: Sequence[str] | None = None,
    *,
    envmap=None,
    resolution: int = 512,
    ssaa: int = 2,
    key: str = "shaded",
    bg: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    transformation=None,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Render a decoded TRELLIS sample (Mesh / MeshWithVoxel) at named views.

    Returns ``{name: RGB uint8 [H,W,3]}`` for the requested ``key`` channel
    (``shaded`` = textured PBR render, ``normal`` = normals).  Uses TRELLIS
    ``render_frames`` (the same PbrMeshRenderer), so the "after" lands on the
    SAME named cameras as the "before".  When ``bg`` is given (default white),
    the object is composited onto that background using the render mask — so
    gate-E gets the overview-style white bg.

    A textured ``MeshWithVoxel`` / ``MeshWithPbrMaterial`` needs an ``envmap``
    (see :func:`load_envmap`) for the shaded channel.
    """
    from trellis2.utils import render_utils

    view_names = list(view_names or VIEW_ORDER)
    extr, intr = named_cameras(view_names, device=device)
    kw = {} if envmap is None else {"envmap": envmap}
    if transformation is not None:
        t = transformation
        if not torch.is_tensor(t):
            t = torch.from_numpy(np.asarray(t, np.float32))
        kw["transformation"] = t.float().to(device)
    rets = render_utils.render_frames(
        sample, extr, intr,
        {"resolution": resolution, "ssaa": ssaa},
        verbose=False, **kw)
    chan = key if key in rets else ("shaded" if "shaded" in rets else next(iter(rets)))
    frames = rets[chan]
    masks = rets.get("mask") or rets.get("alpha")
    out: dict[str, np.ndarray] = {}
    for k, nm in enumerate(view_names):
        img = frames[k]
        if bg is not None and masks is not None:
            m = masks[k].astype(np.float32) / 255.0
            bg_arr = np.array(bg, np.float32) * 255.0
            img = (img.astype(np.float32) * m + bg_arr * (1.0 - m)).clip(0, 255).astype(np.uint8)
        out[nm] = img
    return out


__all__ = [
    "NAMED_VIEWS", "VIEW_ORDER", "SIX_VIEW_ORDER", "VIEW_RADIUS", "VIEW_FOV_DEG",
    "view_yaw_pitch", "named_cameras", "camera_transforms",
    "render_ovoxel", "shade", "render_sample", "load_envmap",
]
