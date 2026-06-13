"""o-voxel front-end: mesh → coloured o-voxel → multiview render (no Blender).

Replaces the slow ``GLB → Blender(Cycles) → image`` path used to obtain the
conditioning / preview multiview.  The recipe:

  1. **Extract o-voxel once.**  ``o_voxel.convert.textured_mesh_to_volumetric_attr``
     voxelises the textured mesh into a sparse coloured grid carrying
     ``base_color`` (+ metallic/roughness/normal).  This is the "extract o-voxel
     up-front" step the pipeline now leads with.
  2. **Render from o-voxel.**  ``o_voxel.rasterize.VoxelRenderer`` (CUDA) rasterises
     that grid from arbitrary cameras.  We reuse TRELLIS's
     ``yaw / pitch / r / fov`` convention which is *identical* to the partverse
     pre-render convention (``sphere_hammersley_sequence``, r=2, fov=40°,
     look-at-origin, up=+Z).  So o-voxel renders land on the **same viewpoints**
     as the packed multiview — "保留视角信息" — without ever touching Blender.

Frame: partverse meshes are Y-up; TRELLIS's render/latent frame is Z-up.  We
center+scale the mesh to ``[-0.5, 0.5]^3`` and (``canonical=True``) apply the
same Y-up→Z-up rotation as :mod:`trellis2_encode` so the o-voxel sits in the
camera frame the packed views were rendered in.

Runs in the ``trellis2`` env (needs ``o_voxel`` + ``utils3d``).  Pure rendering
path — does **not** import the heavy TRELLIS renderer/representation stack.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import trimesh

from scripts.data_prep.mesh_sources import open_mesh


# Y-up (partverse) → Z-up (TRELLIS) canonical rotation, row-vertex convention:
# (x, y, z) → (x, -z, y).  Identical to trellis2_encode._CANON_ROT so o-voxel
# renders share the masked-edit / encode frame.
_CANON_ROT = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

# partverse pre-render camera defaults (encode_asset/render_img_for_enc.py).
PRERENDER_NUM_VIEWS = 150
PRERENDER_RADIUS = 2.0
PRERENDER_FOV_DEG = 40.0


# ───────────────────────── camera convention (TRELLIS) ─────────────────────────
# Re-implemented locally (matches trellis2.utils.random_utils /
# render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics) to avoid importing the
# heavy renderer stack.

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


def sphere_hammersley_sequence(n: int, num_samples: int,
                               offset: tuple[float, float] = (0, 0)) -> list[float]:
    """``[yaw(phi), pitch(theta)]`` — exact copy of TRELLIS's sequence."""
    u = n / num_samples + offset[0] / num_samples
    v = _radical_inverse(_PRIMES[0], n) + offset[1]
    theta = float(np.arccos(1 - 2 * u) - np.pi / 2)
    phi = float(v * 2 * np.pi)
    return [phi, theta]


def yaw_pitch_to_extrinsics_intrinsics(
    yaws: Sequence[float],
    pitchs: Sequence[float],
    r: float = PRERENDER_RADIUS,
    fov_deg: float = PRERENDER_FOV_DEG,
    device: str = "cuda",
):
    """Look-at cameras (Z-up) matching the **partverse Blender pre-render**.

    Camera position follows ``blender_script/render.py``:
    ``(r·cosθ·cosφ, r·sinθ·cosφ, r·sinφ)`` for yaw θ, pitch φ — note this swaps
    x/y vs TRELLIS's ``yaw_pitch_r_fov_to_extrinsics_intrinsics`` (which uses
    sin/cos), so renders land on the SAME azimuth as the packed views.
    Returns (extrinsics, intrinsics) lists.
    """
    import utils3d

    extr_list, intr_list = [], []
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    look = torch.tensor([0.0, 0.0, 0.0], device=device)
    for yaw, pitch in zip(yaws, pitchs):
        fov = torch.deg2rad(torch.tensor(float(fov_deg), device=device))
        y = torch.tensor(float(yaw), device=device)
        p = torch.tensor(float(pitch), device=device)
        orig = torch.tensor([
            torch.cos(y) * torch.cos(p),
            torch.sin(y) * torch.cos(p),
            torch.sin(p),
        ], device=device) * r
        extr = utils3d.torch.extrinsics_look_at(orig, look, up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extr_list.append(extr)
        intr_list.append(intr)
    return extr_list, intr_list


# ─────────────────────────── mesh → coloured o-voxel ───────────────────────────

def load_full_scene(mesh_npz: Path) -> trimesh.Scene:
    """Load ``full.glb`` from a partverse mesh.npz, **keeping materials**."""
    d = open_mesh(mesh_npz)
    if "full.glb" not in d.files:
        raise KeyError(f"no 'full.glb' in {mesh_npz}; have {d.files}")
    scene = trimesh.load(io.BytesIO(d["full.glb"].tobytes()),
                         file_type="glb", process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    return scene


def _nearest_pow2(n: int) -> int:
    if n < 1:
        return 1
    if n & (n - 1) == 0:
        return n
    lo = 1 << (n.bit_length() - 1)
    hi = 1 << n.bit_length()
    return lo if (n - lo) < (hi - n) else hi


def _fix_material_textures(g: trimesh.Trimesh) -> None:
    """Resize PBR textures to a square power-of-two in place.

    ``textured_mesh_to_volumetric_attr`` (unlike the blender_dump path) requires
    square pow2 textures; partverse baseColor maps are arbitrary-sized.
    """
    from PIL import Image

    mat = getattr(g.visual, "material", None)
    if mat is None:
        return
    for name in ("baseColorTexture", "metallicRoughnessTexture",
                 "emissiveTexture", "normalTexture", "occlusionTexture"):
        tex = getattr(mat, name, None)
        if tex is None or not hasattr(tex, "size"):
            continue
        w, h = tex.size
        if w == h and (w & (w - 1) == 0):
            continue
        s = _nearest_pow2(max(w, h))
        setattr(mat, name, tex.resize((s, s), Image.LANCZOS))


def _norm_matrix(allv: np.ndarray, canonical: bool = True) -> np.ndarray:
    """4x4 transform: center + scale to [-0.5,0.5] (by max extent) + optional canon rot."""
    vmin, vmax = allv.min(0), allv.max(0)
    center = (vmin + vmax) / 2.0
    scale = 0.99999 / float((vmax - vmin).max())
    M = np.eye(4)
    M[:3, :3] = np.eye(3) * scale
    M[:3, 3] = -center * scale
    if canonical:
        # trellis2_encode rotates row-vectors (``verts @ _CANON_ROT``); trimesh
        # apply_transform uses the column convention (``R @ v``), so use the
        # transpose to land on the SAME (x,y,z)→(x,-z,y) mapping.
        R = np.eye(4)
        R[:3, :3] = _CANON_ROT.T
        M = R @ M
    return M


def _scene_groups(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    groups = scene.dump() if isinstance(scene, trimesh.Scene) else [scene]
    groups = [g for g in groups if isinstance(g, trimesh.Trimesh) and len(g.vertices)]
    if not groups:
        raise RuntimeError("no triangle geometry in scene")
    return groups


def _normalized_groups(
    scene: trimesh.Scene,
    canonical: bool = True,
    M: np.ndarray | None = None,
    fix_textures: bool = True,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Bake scene-graph transforms then apply normalization ``M``.

    If ``M`` is None it is computed from this scene's own bounds; pass an
    explicit ``M`` (e.g. the full mesh's) so parts land in the SAME frame.
    Returns ``(groups, M)`` — world-space ``Trimesh`` groups + the transform used.
    """
    groups = _scene_groups(scene)
    if M is None:
        allv = np.concatenate([np.asarray(g.vertices) for g in groups], axis=0)
        M = _norm_matrix(allv, canonical=canonical)
    out = []
    for g in groups:
        g = g.copy()
        g.apply_transform(M)
        if fix_textures:
            _fix_material_textures(g)
        out.append(g)
    return out, M


def mesh_to_colored_ovox(
    mesh_npz: Path,
    grid_size: int = 512,
    canonical: bool = True,
    M: np.ndarray | None = None,
    return_M: bool = False,
):
    """Voxelise a partverse mesh → coloured sparse o-voxel.

    Returns ``(coords, attr)`` where ``coords`` is int32 ``[N, 3]`` in
    ``0..grid_size-1`` and ``attr`` is the dict from
    ``textured_mesh_to_volumetric_attr`` (``base_color`` uint8 ``[N,3]`` etc).
    Voxel centres in the render frame are ``coords / grid_size - 0.5``.
    With ``return_M`` also returns the 4x4 normalization (reuse it for parts).
    """
    import o_voxel

    scene = load_full_scene(mesh_npz)
    groups, M = _normalized_groups(scene, canonical=canonical, M=M)
    coords, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        trimesh.Scene(groups),
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    )
    coords = coords.cpu().numpy().astype(np.int32) if torch.is_tensor(coords) else np.asarray(coords, np.int32)
    if return_M:
        return coords, attr, M
    return coords, attr


def load_part_scenes(mesh_npz: Path) -> dict[int, trimesh.Scene]:
    """Load ``part_<i>.glb`` from a partverse mesh.npz → ``{pid: Scene}``.

    Parts share the raw frame of ``full.glb`` (verified: union bounds match), so
    applying the full mesh's normalization ``M`` lands them in the shared frame.
    """
    import re
    d = open_mesh(mesh_npz)
    out = {}
    for k in d.files:
        m = re.match(r"^part_(\d+)\.glb$", k)
        if not m:
            continue
        sc = trimesh.load(io.BytesIO(d[k].tobytes()), file_type="glb", process=False)
        if isinstance(sc, trimesh.Trimesh):
            sc = trimesh.Scene(sc)
        out[int(m.group(1))] = sc
    return out


def part_occupancy_coords(
    mesh_npz: Path,
    M: np.ndarray,
    grid_size: int = 256,
) -> dict[int, np.ndarray]:
    """Voxelise each part's geometry into the shared (full-mesh ``M``) frame.

    Geometry-only (``mesh_to_flexible_dual_grid``) — no material needed — so the
    part-highlight row is robust to parts that lack PBR textures.  Returns
    ``{pid: coords int32[N,3]}`` in ``0..grid_size-1``.
    """
    import o_voxel

    parts = load_part_scenes(mesh_npz)
    out: dict[int, np.ndarray] = {}
    for pid, scene in parts.items():
        groups, _ = _normalized_groups(scene, M=M, fix_textures=False)
        merged = trimesh.util.concatenate(groups)
        verts = torch.from_numpy(np.asarray(merged.vertices)).float()
        faces = torch.from_numpy(np.asarray(merged.faces)).long()
        vox, _dual, _inter = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=verts, faces=faces, grid_size=grid_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2,
            regularization_weight=1e-2, timing=False,
        )
        vox = vox.cpu().numpy().astype(np.int32) if torch.is_tensor(vox) else np.asarray(vox, np.int32)
        out[pid] = vox
    return out


# ──────────────────────────────── render ──────────────────────────────────────

def render_ovox_views(
    coords: np.ndarray,
    base_color: np.ndarray,
    grid_size: int,
    yaws: Sequence[float],
    pitchs: Sequence[float],
    *,
    r: float = PRERENDER_RADIUS,
    fov_deg: float = PRERENDER_FOV_DEG,
    resolution: int = 512,
    ssaa: int = 2,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
) -> list[np.ndarray]:
    """Rasterise a coloured o-voxel from each (yaw, pitch).  Returns RGB uint8.

    ``base_color`` may be uint8 ``[N,3]`` (0..255) or float ``[N,3]`` (0..1).
    Empty pixels are composited onto ``bg``.
    """
    import o_voxel

    pos = torch.from_numpy(coords.astype(np.float32)).to(device) / grid_size - 0.5
    bc = base_color
    if torch.is_tensor(bc):
        bc = bc.detach().cpu().numpy()
    bc = np.asarray(bc)
    if bc.dtype != np.float32 and bc.dtype != np.float64:
        bc = bc.astype(np.float32) / 255.0
    attrs = torch.from_numpy(bc.astype(np.float32)).to(device)

    extr_list, intr_list = yaw_pitch_to_extrinsics_intrinsics(
        yaws, pitchs, r=r, fov_deg=fov_deg, device=device)
    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": resolution, "ssaa": ssaa})
    bg_t = torch.tensor(bg, device=device).view(3, 1, 1)

    imgs = []
    for extr, intr in zip(extr_list, intr_list):
        out = renderer.render(pos, attrs, 1.0 / grid_size, extr, intr)
        color = out.attr                       # [3, H, W] in 0..1
        alpha = out.alpha.clamp(0, 1)[None]    # [1, H, W]
        comp = color * alpha + bg_t * (1 - alpha)
        img = (comp.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imgs.append(img)
    return imgs


def render_object_multiview(
    mesh_npz: Path,
    view_indices: Iterable[int],
    *,
    num_views: int = PRERENDER_NUM_VIEWS,
    grid_size: int = 512,
    resolution: int = 512,
    ssaa: int = 2,
    canonical: bool = True,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
):
    """High-level: mesh.npz → coloured o-voxel → RGB renders at packed viewpoints.

    ``view_indices`` are the partverse frame indices (e.g. 8 for ``008.png``);
    each maps to ``sphere_hammersley_sequence(i, num_views)``.  Returns
    ``(images, coords, attr, cam)`` — ``cam`` is the per-view yaw/pitch/r/fov so
    the viewpoint metadata travels with the renders.
    """
    coords, attr = mesh_to_colored_ovox(mesh_npz, grid_size=grid_size, canonical=canonical)
    view_indices = list(view_indices)
    yp = [sphere_hammersley_sequence(i, num_views) for i in view_indices]
    yaws = [v[0] for v in yp]
    pitchs = [v[1] for v in yp]
    imgs = render_ovox_views(
        coords, attr["base_color"], grid_size, yaws, pitchs,
        resolution=resolution, ssaa=ssaa, bg=bg, device=device)
    cam = [{"view_index": int(i), "yaw": y, "pitch": p,
            "radius": PRERENDER_RADIUS, "fov": np.deg2rad(PRERENDER_FOV_DEG)}
           for i, y, p in zip(view_indices, yaws, pitchs)]
    return imgs, coords, attr, cam


# ───────────────────────────── overview (5×2 grid) ─────────────────────────────
# Mirrors partcraft.render.overview: top row = RGB, bottom row = per-part
# palette highlight.  Both rows now come from the o-voxel — no Blender, no
# packed input images.

# part palette (RGB) — kept in sync with partcraft.render.overview._PALETTE.
OVERVIEW_PALETTE = [
    [220, 30, 30], [255, 140, 0], [255, 220, 0], [130, 220, 30],
    [30, 160, 50], [0, 150, 150], [60, 220, 230], [30, 90, 240],
    [20, 30, 130], [140, 40, 200], [230, 40, 200], [255, 150, 200],
    [130, 70, 30], [220, 180, 130], [30, 30, 30], [130, 130, 130],
]
HIGHLIGHT_RED = [220, 30, 30]
HIGHLIGHT_GREY = [170, 170, 170]


def colorize_parts(
    part_coords: dict[int, np.ndarray],
    target_ids: Iterable[int] | None = None,
):
    """Flatten ``{pid: coords}`` → ``(coords[N,3], colors[N,3] uint8)``.

    ``target_ids=None`` → each part gets ``OVERVIEW_PALETTE[pid % 16]`` (the
    phase-1 overview).  Otherwise target parts are RED and the rest GREY (the
    gate / selection highlight).
    """
    tset = None if target_ids is None else set(int(t) for t in target_ids)
    all_c, all_col = [], []
    for pid, c in sorted(part_coords.items()):
        if c is None or len(c) == 0:
            continue
        if tset is None:
            col = OVERVIEW_PALETTE[pid % len(OVERVIEW_PALETTE)]
        else:
            col = HIGHLIGHT_RED if pid in tset else HIGHLIGHT_GREY
        all_c.append(np.asarray(c, np.int32))
        all_col.append(np.tile(np.array(col, np.uint8), (len(c), 1)))
    if not all_c:
        return np.zeros((0, 3), np.int32), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_c, 0), np.concatenate(all_col, 0)


def render_overview_from_ovox(
    mesh_npz: Path,
    *,
    view_names: Sequence[str] | None = None,
    color_grid: int = 512,
    part_grid: int = 256,
    resolution: int = 512,
    ssaa: int = 2,
    canonical: bool = True,
    target_ids: Iterable[int] | None = None,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
    skip_rgb: bool = False,
):
    """Render the overview rows from the o-voxel at the **named views**.

    No Blender, no packed images, no view-index→frame mapping — the columns are
    the explicit named cameras in :data:`ovox_views.VIEW_ORDER`
    (front/right/back/left/down).

    Returns ``dict`` with:
      ``rgb``        list[H,W,3] uint8  — coloured o-voxel at each view (top row)
      ``highlight``  list[H,W,3] uint8  — per-part palette (or red/grey if
                     ``target_ids``) at each view (bottom row)
      ``cam``        list of per-view {name, yaw, pitch, radius, fov}
      ``views``      ordered view names (== overview column order)
      ``part_ids``   sorted part ids
    """
    from partcraft.render import ovox_views as _ov

    views = list(view_names or _ov.VIEW_ORDER)

    if skip_rgb:
        # Caller supplies the (PBR) top row separately → skip the expensive
        # full-mesh colour voxelisation + o-voxel RGB render; only compute the
        # normalisation M (lightweight) needed for the part-occupancy seg row.
        scene = load_full_scene(mesh_npz)
        _, M = _normalized_groups(scene, canonical=canonical)
        rgb = None
    else:
        # top row — coloured full mesh; reuse its normalization M for the parts.
        fc, fattr, M = mesh_to_colored_ovox(
            mesh_npz, grid_size=color_grid, canonical=canonical, return_M=True)
        rgb_d = _ov.render_ovoxel(fc, fattr["base_color"], color_grid, views,
                                  resolution=resolution, ssaa=ssaa, bg=bg, device=device)
        rgb = [rgb_d[v] for v in views]

    # bottom row — per-part occupancy in the SAME frame, palette-coloured.
    part_coords = part_occupancy_coords(mesh_npz, M, grid_size=part_grid)
    pc, pcol = colorize_parts(part_coords, target_ids=target_ids)
    hl_d = _ov.render_ovoxel(pc, pcol, part_grid, views,
                             resolution=resolution, ssaa=ssaa, bg=bg, device=device)

    highlight = [hl_d[v] for v in views]
    cam = _ov.camera_transforms(views)
    return {"rgb": rgb, "highlight": highlight, "cam": cam,
            "views": views, "part_ids": sorted(part_coords.keys())}


def glb_to_pbr_mesh(
    mesh_npz_or_scene,
    *,
    canonical: bool = True,
    M: np.ndarray | None = None,
    device: str = "cuda",
):
    """Build a TRELLIS ``MeshWithPbrMaterial`` from a partverse glb.

    Lets the original mesh be rendered by the SAME native ``PbrMeshRenderer`` as
    the decoded "after", so before/after share renderer + lighting.  Normalised
    to the canonical ``[-0.5,0.5]`` frame (same as the decoded mesh).  Falls back
    to ``base_color_factor`` (solid colour) when a part has no PBR texture.
    """
    from trellis2.representations.mesh.base import (
        MeshWithPbrMaterial, PbrMaterial, Texture,
        TextureFilterMode, TextureWrapMode, AlphaMode)

    scene = (load_full_scene(Path(mesh_npz_or_scene))
             if isinstance(mesh_npz_or_scene, (str, Path)) else mesh_npz_or_scene)
    groups, M = _normalized_groups(scene, canonical=canonical, M=M)

    materials, all_v, all_f, all_mid, all_uv = [], [], [], [], []
    start = 0
    for gi, g in enumerate(groups):
        v = np.asarray(g.vertices, np.float32)
        f = np.asarray(g.faces, np.int64)
        uv = getattr(g.visual, "uv", None)
        uvf = (np.asarray(uv, np.float32)[f] if uv is not None
               else np.zeros((f.shape[0], 3, 2), np.float32))
        # TRELLIS's PbrMeshRenderer (like its training data) expects v-flipped
        # UVs; partverse glb UVs are not flipped → flip v, else the texture
        # samples the (mostly empty/black) atlas and renders dark.
        uvf = uvf.copy()
        uvf[..., 1] = 1.0 - uvf[..., 1]
        mat = getattr(g.visual, "material", None)
        tex = None
        bct = getattr(mat, "baseColorTexture", None) if mat is not None else None
        if bct is not None:
            arr = np.asarray(bct.convert("RGB"), np.float32) / 255.0
            tex = Texture(torch.from_numpy(arr),
                          filter_mode=TextureFilterMode.LINEAR,
                          wrap_mode=TextureWrapMode.REPEAT)
        bcf = getattr(mat, "baseColorFactor", None) if mat is not None else None
        bcf = ([float(x) / 255.0 for x in np.asarray(bcf)[:3]]
               if bcf is not None else [1.0, 1.0, 1.0])
        # metallic-roughness texture: glTF packs roughness in G, metallic in B.
        mrt = getattr(mat, "metallicRoughnessTexture", None) if mat is not None else None
        metal_tex = rough_tex = None
        if mrt is not None:
            mr = np.asarray(mrt.convert("RGB"), np.float32) / 255.0
            metal_tex = Texture(torch.from_numpy(np.ascontiguousarray(mr[..., 2:3])),
                                filter_mode=TextureFilterMode.LINEAR,
                                wrap_mode=TextureWrapMode.REPEAT)
            rough_tex = Texture(torch.from_numpy(np.ascontiguousarray(mr[..., 1:2])),
                                filter_mode=TextureFilterMode.LINEAR,
                                wrap_mode=TextureWrapMode.REPEAT)
        mf = getattr(mat, "metallicFactor", None) if mat is not None else None
        rf = getattr(mat, "roughnessFactor", None) if mat is not None else None
        # glTF: factor defaults to 1.0 (multiplies the texture) when present.
        metal_f = float(mf) if mf is not None else (1.0 if metal_tex is not None else 0.0)
        rough_f = float(rf) if rf is not None else 1.0
        materials.append(PbrMaterial(
            base_color_texture=tex, base_color_factor=bcf,
            metallic_texture=metal_tex, metallic_factor=metal_f,
            roughness_texture=rough_tex, roughness_factor=rough_f,
            alpha_mode=AlphaMode.OPAQUE, alpha_cutoff=0.5))
        all_v.append(v); all_f.append(f + start)
        all_mid.append(np.full(f.shape[0], gi, np.int64)); all_uv.append(uvf)
        start += len(v)
    materials.append(PbrMaterial(base_color_factor=[0.8, 0.8, 0.8],
                                 metallic_factor=0.0, roughness_factor=0.5))

    mesh = MeshWithPbrMaterial(
        vertices=torch.from_numpy(np.concatenate(all_v, 0)),
        faces=torch.from_numpy(np.concatenate(all_f, 0)),
        material_ids=torch.from_numpy(np.concatenate(all_mid, 0)),
        uv_coords=torch.from_numpy(np.concatenate(all_uv, 0)),
        materials=materials,
    )
    return mesh.to(device)


def render_mesh_named_view(
    mesh_npz: Path,
    view_name: str,
    *,
    grid_size: int = 512,
    resolution: int = 512,
    ssaa: int = 2,
    canonical: bool = True,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
) -> np.ndarray:
    """Render the coloured mesh o-voxel at a single named view → RGB uint8 [H,W,3].

    The FLUX-input / single-view path (replaces ``get_image_bytes`` on packed
    images): voxelise the mesh and rasterise it from the named camera.
    """
    from partcraft.render import ovox_views as _ov

    coords, attr = mesh_to_colored_ovox(mesh_npz, grid_size=grid_size, canonical=canonical)
    d = _ov.render_ovoxel(coords, attr["base_color"], grid_size, [view_name],
                          resolution=resolution, ssaa=ssaa, bg=bg, device=device)
    return d[view_name]


# ───────────────── unified PBR overview (decode RGB + part-mesh seg) ─────────
# Replaces the o-voxel overview: realistic RGB from decoded latents, and a flat
# pure-colour part segmentation from the actual part meshes (captures thin
# geometry like crib slats that o-voxel solid-blocks lose).  Both via TRELLIS's
# native PbrMeshRenderer at the named views.

def build_part_palette_mesh(mesh_npz: Path, M: np.ndarray,
                            target_ids: Iterable[int] | None = None,
                            device: str = "cuda"):
    """Combined part-mesh, each part a solid material → flat segmentation.

    ``target_ids=None`` → every part gets ``OVERVIEW_PALETTE[pid%16]`` (overview
    segmentation).  Otherwise target parts are RED, the rest GREY (gate-A
    highlight).  Render its ``base_color`` channel for a flat, geometry-accurate
    map (occlusion via the renderer's z-buffer).
    """
    import trimesh
    from trellis2.representations.mesh.base import MeshWithPbrMaterial, PbrMaterial, AlphaMode

    parts = load_part_scenes(Path(mesh_npz))
    tset = None if target_ids is None else set(int(t) for t in target_ids)
    all_v, all_f, all_mid, materials = [], [], [], []
    start = 0
    for i, pid in enumerate(sorted(parts)):
        groups, _ = _normalized_groups(parts[pid], M=M, fix_textures=False)
        merged = trimesh.util.concatenate(groups)
        v = torch.from_numpy(np.asarray(merged.vertices)).float()
        f = torch.from_numpy(np.asarray(merged.faces)).long()
        all_v.append(v); all_f.append(f + start)
        all_mid.append(torch.full((f.shape[0],), i, dtype=torch.long)); start += v.shape[0]
        if tset is None:
            col = np.array(OVERVIEW_PALETTE[pid % len(OVERVIEW_PALETTE)], np.float32) / 255.0
        else:
            col = np.array(HIGHLIGHT_RED if pid in tset else HIGHLIGHT_GREY, np.float32) / 255.0
        materials.append(PbrMaterial(base_color_factor=[float(x) for x in col],
                                     metallic_factor=0.0, roughness_factor=1.0,
                                     alpha_mode=AlphaMode.OPAQUE))
    if not all_v:
        return None
    return MeshWithPbrMaterial(
        vertices=torch.cat(all_v), faces=torch.cat(all_f),
        material_ids=torch.cat(all_mid),
        uv_coords=torch.zeros(torch.cat(all_f).shape[0], 3, 2),
        materials=materials).to(device)


def render_pbr_overview(pipeline, mesh_npz: Path, shape_slat, tex_slat, envmap,
                        *, view_names: Sequence[str] | None = None,
                        resolution: int = 512, decode_res: int = 1024,
                        target_ids: Iterable[int] | None = None,
                        device: str = "cuda"):
    """Unified overview render.  Returns dict:
      ``rgb``       {name: RGB}  decoded latents → PbrMeshRenderer 'shaded'
      ``seg``       {name: RGB}  part-mesh palette → 'base_color' (flat)
      ``highlight`` {name: RGB}  part-mesh red/grey (only if target_ids)
      ``cam``/``M``/``part_ids``
    """
    from partcraft.render import ovox_views as _ov

    views = list(view_names or _ov.VIEW_ORDER)
    decoded = pipeline.decode_latent(shape_slat, tex_slat, decode_res)[0]
    # decoded mesh and the part-mesh segmentation are encoded/normalized with the
    # SAME canonical _CANON_ROT (Y-up→Z-up), so both already sit in ONE frame —
    # render both straight through render_sample at the named cameras, no extra
    # transform.  (A rotation on only one side is what tipped RGB vs seg before.)
    rgb = _ov.render_sample(decoded, views, envmap=envmap, resolution=resolution,
                            key="shaded", bg=(1, 1, 1), device=device)

    scene = load_full_scene(Path(mesh_npz))
    _, M = _normalized_groups(scene, canonical=True)
    part_ids = sorted(load_part_scenes(Path(mesh_npz)).keys())

    def _seg(tids):
        m = build_part_palette_mesh(mesh_npz, M, target_ids=tids, device=device)
        if m is None:
            white = (np.ones((resolution, resolution, 3), np.uint8) * 255)
            return {v: white.copy() for v in views}
        return _ov.render_sample(m, views, envmap=envmap, resolution=resolution,
                                 key="base_color", bg=(1, 1, 1), device=device)

    out = {"rgb": rgb, "seg": _seg(None), "cam": _ov.camera_transforms(views),
           "M": M, "part_ids": part_ids}
    if target_ids is not None:
        out["highlight"] = _seg(target_ids)
    return out


def save_ovox(path: Path, coords: np.ndarray, attr: dict, grid_size: int) -> None:
    """Persist the coloured o-voxel so downstream stages reuse one voxelisation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save = {"coords": coords.astype(np.int32), "grid_size": np.int32(grid_size)}
    for k, v in attr.items():
        save[k] = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    np.savez_compressed(path, **save)


__all__ = [
    "sphere_hammersley_sequence",
    "yaw_pitch_to_extrinsics_intrinsics",
    "load_full_scene",
    "load_part_scenes",
    "mesh_to_colored_ovox",
    "part_occupancy_coords",
    "colorize_parts",
    "render_ovox_views",
    "render_object_multiview",
    "render_overview_from_ovox",
    "render_mesh_named_view",
    "glb_to_pbr_mesh",
    "build_part_palette_mesh",
    "render_pbr_overview",
    "OVERVIEW_PALETTE",
    "save_ovox",
]
