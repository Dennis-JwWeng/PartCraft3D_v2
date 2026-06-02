"""P5 — derive a per-token keep_mask from part annotations.

The TRELLIS.2 cascade emits HR shape SLat at 64³ internal resolution
(== input 1024³ downsampled 16× by the f16 encoder).  P1's saved
coords live in this same 64³ space.

Given the original mesh's ``part_X.glb`` files inside ``mesh.npz`` and
a list of target part ids (= parts the user WANTS TO EDIT), we voxelize
each target part at the SAME 1024³ grid P1 used, integer-divide by 16
to get coarse block ids, and mark any P1 coord that falls in that set
as "edit" (keep_mask = False).  All other tokens preserve the original.

This is approximate — the f16 encoder has a receptive field, so a token
at (X,Y,Z) does not perfectly equal "voxels in the [16X, 16X+16) × ...
block".  In practice the 16× block heuristic is what data_toolkit
itself uses for part masks (mirror of dual_grid token alignment) and
is the same approach Vinedresser3D uses (see ``get_s1_mask``).

API:
    coords_64, keep_mask = part_keep_mask(
        mesh_npz_path, p1_coords, target_part_ids)

    # coords_64 : LongTensor [N, 3]  in 0..63  (= input p1 coords, returned
    #              for verification)
    # keep_mask : BoolTensor [N]      True  → preserve via masked sampling
    #                                  False → free for the model to repaint
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh

_T2 = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
if _T2 not in sys.path:
    sys.path.insert(0, _T2)

import o_voxel                                                          # noqa: E402

GRID_HR = 1024            # voxelization resolution (matches P1)
GRID_LO = 64              # cascade output internal resolution
DOWNSAMPLE = GRID_HR // GRID_LO

# Canonical-frame rotation R_X90 (partverse Y-up → TRELLIS Z-up); right-mult so
# a row vertex (x,y,z) → (x,-z,y).  MUST equal trellis2_encode._CANON_ROT and be
# applied at the SAME stage (after center+scale, before voxelize) so the edit
# mask stays byte-aligned with the canonical-encoded coords0.  See the long note
# in trellis2_encode for why masked editing needs the latent in this frame.
_CANON_ROT = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)


def _load_part_mesh(mesh_npz_path: Path, part_id: int) -> trimesh.Trimesh:
    d = np.load(mesh_npz_path, allow_pickle=True)
    key = f"part_{part_id}.glb"
    if key not in d.files:
        raise KeyError(f"{key} not in {mesh_npz_path}; have {d.files}")
    g = trimesh.load(io.BytesIO(d[key].tobytes()),
                     file_type="glb", process=False)
    if isinstance(g, trimesh.Scene):
        meshes = [m for m in g.geometry.values() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"{key} has no Trimesh geometry")
        return trimesh.util.concatenate(meshes)
    return g


def _normalize_to_unit(verts: torch.Tensor) -> torch.Tensor:
    """Center + scale to [-0.5, 0.5]^3 (same recipe as trellis2_encode_one)."""
    vmin = verts.min(dim=0)[0]
    vmax = verts.max(dim=0)[0]
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    return (verts - center) * scale


def _voxelize_to_grid_hr(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Return ``[K, 3]`` int voxel indices at GRID_HR resolution.

    Uses ``o_voxel.convert.mesh_to_flexible_dual_grid`` to match P1's
    encoder input voxel set exactly.
    """
    voxel_indices, _dual, _intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=verts.float(),
        faces=faces.long(),
        grid_size=GRID_HR,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )
    return voxel_indices.int()


def _coords3_to_key(coords3: torch.Tensor, grid: int) -> torch.Tensor:
    """Flatten 3-D voxel indices into a single int64 key for set lookup."""
    g = int(grid)
    return (coords3[:, 0].long() * g * g
            + coords3[:, 1].long() * g
            + coords3[:, 2].long())


def part_keep_mask(
    mesh_npz_path: Path,
    p1_coords: torch.Tensor,
    target_part_ids: Iterable[int],
    *,
    full_mesh_for_normalize: bool = True,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Build a per-token keep_mask aligned with ``p1_coords``.

    Args:
        mesh_npz_path:  partcraft ``mesh.npz`` path containing ``full.glb``
                        and ``part_X.glb`` blobs.
        p1_coords:      LongTensor ``[N, 3]`` or ``[N, 4]`` from P1
                        (if 4 cols, batch-idx column is dropped).  Values
                        in 0..63 (the 64³ HR-cascade-internal space).
        target_part_ids: ids of parts the user wants to edit. Voxels
                        belonging to ANY of these become edit-region.
        full_mesh_for_normalize: if True (default), compute the
                        center+scale from FULL.glb so part voxelization
                        lives in the same canonical frame as P1's
                        encoding.  Set False to use the union of target
                        parts' own bbox (only valid if you know parts
                        were already centered against the full mesh).

    Returns:
        keep_mask:  BoolTensor ``[N]``.  True = preserve.
        info:       dict with diagnostic counts.
    """
    # Strip batch column if present.
    if p1_coords.dim() == 2 and p1_coords.shape[1] == 4:
        p1_coords3 = p1_coords[:, 1:]
    else:
        p1_coords3 = p1_coords
    p1_coords3 = p1_coords3.long()

    # Determine the unit-normalization (center+scale) from the FULL mesh
    # so part voxelization is in the same frame P1 was encoded under.
    d = np.load(mesh_npz_path, allow_pickle=True)
    if full_mesh_for_normalize:
        g = trimesh.load(io.BytesIO(d["full.glb"].tobytes()),
                         file_type="glb", process=False)
        if isinstance(g, trimesh.Scene):
            ms = [m for m in g.geometry.values() if isinstance(m, trimesh.Trimesh)]
            full_mesh = trimesh.util.concatenate(ms)
        else:
            full_mesh = g
        full_v = torch.from_numpy(np.asarray(full_mesh.vertices)).float()
        vmin = full_v.min(dim=0)[0]
        vmax = full_v.max(dim=0)[0]
        center = (vmin + vmax) / 2
        scale = 0.99999 / (vmax - vmin).max()
    else:
        center, scale = None, None

    # Union all target parts' voxels at HR.
    target_voxels_hr_list = []
    for pid in target_part_ids:
        m = _load_part_mesh(mesh_npz_path, pid)
        v = torch.from_numpy(np.asarray(m.vertices)).float()
        f = torch.from_numpy(np.asarray(m.faces)).long()
        if center is not None:
            v = (v - center) * scale
        else:
            v = _normalize_to_unit(v)
        idx_hr = _voxelize_to_grid_hr(v, f)
        target_voxels_hr_list.append(idx_hr)
    if not target_voxels_hr_list:
        # No target parts → preserve everything.
        keep = torch.ones(p1_coords3.shape[0], dtype=torch.bool,
                          device=p1_coords3.device)
        return keep, {"n_tokens": p1_coords3.shape[0],
                      "n_edit_voxels_hr": 0,
                      "n_edit_blocks_64": 0,
                      "n_kept": int(keep.sum().item()),
                      "n_target_parts": 0}
    target_hr = torch.cat(target_voxels_hr_list, dim=0)

    # Coarsen to 64³ (integer division).
    target_lo = (target_hr // DOWNSAMPLE).long().clamp(0, GRID_LO - 1)
    target_keys = _coords3_to_key(target_lo, GRID_LO).unique()
    target_set = set(target_keys.cpu().tolist())

    p1_keys = _coords3_to_key(p1_coords3.cpu(), GRID_LO).cpu().tolist()
    is_edit = torch.tensor(
        [k in target_set for k in p1_keys],
        dtype=torch.bool, device=p1_coords3.device,
    )
    keep = ~is_edit

    info = {
        "n_tokens": int(p1_coords3.shape[0]),
        "n_edit_voxels_hr": int(target_hr.shape[0]),
        "n_edit_blocks_64": int(target_keys.shape[0]),
        "n_target_parts": len(target_voxels_hr_list),
        "n_edit_tokens": int(is_edit.sum().item()),
        "n_kept": int(keep.sum().item()),
    }
    if verbose:
        print(f"[p5] part_keep_mask: targets={list(target_part_ids)}  "
              f"tokens={info['n_tokens']}  "
              f"edit_voxels_hr={info['n_edit_voxels_hr']}  "
              f"edit_blocks_64={info['n_edit_blocks_64']}  "
              f"edit_tokens={info['n_edit_tokens']}  "
              f"kept={info['n_kept']} "
              f"({info['n_kept']/info['n_tokens']*100:.1f}%)")
    return keep, info


# ─────────────────── structure-editing helpers (S1) ─────────────────
# These support the modification / scale path, which (matching v1's
# refiner.build_part_mask + interweave_Trellis S1 repaint) must edit the
# TRELLIS.2 sparse-structure stage so the part can grow / shrink new voxels,
# not just be repainted inside its original footprint.


def _full_center_scale(d) -> tuple[torch.Tensor, torch.Tensor]:
    """center+scale that maps full.glb into [-0.5, 0.5]^3 (P1 frame)."""
    g = trimesh.load(io.BytesIO(d["full.glb"].tobytes()),
                     file_type="glb", process=False)
    if isinstance(g, trimesh.Scene):
        ms = [m for m in g.geometry.values() if isinstance(m, trimesh.Trimesh)]
        full_mesh = trimesh.util.concatenate(ms)
    else:
        full_mesh = g
    v = torch.from_numpy(np.asarray(full_mesh.vertices)).float()
    vmin, vmax = v.min(0)[0], v.max(0)[0]
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    return center, scale


def _target_block_keys_64(
    mesh_npz_path: Path,
    target_part_ids: Iterable[int],
    *,
    full_mesh_for_normalize: bool = True,
    canonical: bool = False,
) -> torch.Tensor:
    """Union of target parts' occupied 64³ block keys (int64 [K]).

    ``canonical=True`` applies :data:`_CANON_ROT` after center+scale (same as the
    canonical encode) so the mask aligns with canonical-frame coords0.
    """
    d = np.load(mesh_npz_path, allow_pickle=True)
    center, scale = (_full_center_scale(d) if full_mesh_for_normalize
                     else (None, None))
    blocks: list[torch.Tensor] = []
    for pid in target_part_ids:
        m = _load_part_mesh(mesh_npz_path, pid)
        v = torch.from_numpy(np.asarray(m.vertices)).float()
        f = torch.from_numpy(np.asarray(m.faces)).long()
        v = (v - center) * scale if center is not None else _normalize_to_unit(v)
        if canonical:
            v = v @ _CANON_ROT.to(v.dtype)
        idx_hr = _voxelize_to_grid_hr(v, f)
        idx_lo = (idx_hr // DOWNSAMPLE).long().clamp(0, GRID_LO - 1)
        blocks.append(_coords3_to_key(idx_lo, GRID_LO))
    if not blocks:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(blocks).unique()


def _keys_to_grid(keys: torch.Tensor) -> torch.Tensor:
    """Dense ``[64,64,64]`` bool grid from flat int64 block keys."""
    g = GRID_LO
    grid = torch.zeros(g, g, g, dtype=torch.bool)
    if keys.numel():
        grid[(keys // (g * g)).long(),
             ((keys // g) % g).long(),
             (keys % g).long()] = True
    return grid


def _all_part_ids(mesh_npz_path: Path) -> list[int]:
    """Every ``part_<id>.glb`` id stored in the mesh npz, sorted."""
    import re
    d = np.load(mesh_npz_path, allow_pickle=True)
    ids = []
    for f in d.files:
        m = re.match(r"part_(\d+)\.glb$", f)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _dilate_grid(grid: torch.Tensor, pad: int) -> torch.Tensor:
    """Chebyshev (box) dilation of a dense [G,G,G] bool grid by ``pad`` cells."""
    if pad <= 0:
        return grid
    import torch.nn.functional as F
    out = F.max_pool3d(grid.float()[None, None], kernel_size=2 * pad + 1,
                       stride=1, padding=pad)
    return (out[0, 0] > 0)


def part_edit_grid_64(
    mesh_npz_path: Path,
    target_part_ids: Iterable[int],
    *,
    pad: int = 3,
    full_mesh_for_normalize: bool = True,
    canonical: bool = False,
    subtract_preserved: bool = False,
) -> torch.Tensor:
    """Dense ``[64,64,64]`` bool edit region for the *structure* stage.

    The target parts are voxelized at 64³ then dilated by ``pad`` cells so the
    S1 repaint has empty room to generate a part of a different size/shape
    (mirrors v1 ``_compute_editing_region(..., pad=3)``).  Independent of the
    current active-voxel set, so newly-grown voxels are permitted.

    ``canonical`` is forwarded to :func:`_target_block_keys_64` so the region
    matches canonical-frame coords0 when canonical encoding is on.

    ``subtract_preserved`` (v1 anti-inflation): voxelize the OTHER (non-target)
    GT parts and remove them from the dilated region, so the pad/dilation never
    eats into preserved geometry (mirrors v1 ``mask & ~preserved_parts``).  For a
    single-part / whole-object asset this is a no-op (no preserved parts exist).
    """
    keys = _target_block_keys_64(
        mesh_npz_path, target_part_ids,
        full_mesh_for_normalize=full_mesh_for_normalize,
        canonical=canonical)
    grid = _dilate_grid(_keys_to_grid(keys), pad)
    if subtract_preserved:
        tgt = {int(t) for t in target_part_ids}
        pres_ids = [p for p in _all_part_ids(mesh_npz_path) if p not in tgt]
        if pres_ids:
            pres_keys = _target_block_keys_64(
                mesh_npz_path, pres_ids,
                full_mesh_for_normalize=full_mesh_for_normalize,
                canonical=canonical)
            grid = grid & ~_keys_to_grid(pres_keys).to(grid.device)
    return grid


def edit_grid_64_to_keep16(grid64: torch.Tensor, thresh: float = 0.1) -> torch.Tensor:
    """Downsample a 64³ edit grid to a 16³ *keep* (preserve) mask.

    A 4×4×4 block is 'edit' if ≥``thresh`` of its 64 cells are edit (mirrors
    ``interweave_Trellis.get_s1_mask``).  Returns ``[16,16,16]`` bool where
    True = preserve (anchor to the inverted original SS latent).
    """
    g = grid64.float().reshape(16, 4, 16, 4, 16, 4).sum(dim=(1, 3, 5)) / 64.0
    return ~(g >= thresh)


def edit_grid_64_to_keep16_soft(
    grid64: torch.Tensor, thresh: float = 0.1, feather: float = 1.0,
) -> torch.Tensor:
    """Soft (feathered) version of :func:`edit_grid_64_to_keep16`.

    Returns a FLOAT ``[16,16,16]`` *keep weight* in ``[0,1]`` (1 = fully
    preserve / anchor to the inverted original SS latent, 0 = fully free).
    The hard binary keep mask is box-blurred so the body↔edit boundary ramps
    smoothly over ``~feather`` 16³ blocks instead of a one-block step.  This
    heals the occupancy crater that the hard ``torch.where`` mask cuts at the
    junction (the SS latent is only 16³, so a binary cut = a 4-voxel step that
    ``ss_dec`` decodes as a torn boundary).  Deep body stays exactly 1.0 and
    deep edit stays exactly 0.0; only a thin boundary shell blends.

    ``feather`` controls the blur radius ``r = round(feather)`` (kernel
    ``2r+1``); ``feather<=0`` falls back to the hard mask (as float).

    The feather is ONE-SIDED: the preserve region stays EXACTLY 1.0 (the body
    is never touched — a symmetric blur would pull preserve-side weights below 1
    and let the S1 repaint DELETE body occupancy → the top gets removed).  Only
    the EDIT side ramps 1→0 over the boundary band, giving the edit structure a
    soft "lead-in" of partial anchoring so it stays attached to the body.
    """
    g = grid64.float().reshape(16, 4, 16, 4, 16, 4).sum(dim=(1, 3, 5)) / 64.0
    keep = (g < thresh).float()                       # [16,16,16] 1=preserve
    if feather <= 0:
        return keep
    r = max(1, int(round(feather)))
    k = 2 * r + 1
    w = keep[None, None]                              # [1,1,16,16,16]
    w = torch.nn.functional.pad(w, (r,) * 6, mode="replicate")
    kernel = torch.ones(1, 1, k, k, k, device=grid64.device) / float(k ** 3)
    blur = torch.nn.functional.conv3d(w, kernel)[0, 0]
    # one-sided: preserve stays 1.0, edit side gets the (decaying) blur lead-in
    return torch.maximum(keep, blur).clamp(0.0, 1.0)


def coord_keys_64(coords3: torch.Tensor) -> torch.Tensor:
    """[N,3] int voxel indices (0..63) → [N] int64 flat keys."""
    return _coords3_to_key(coords3.long(), GRID_LO)


def build_coord_bridge(
    coords0: torch.Tensor,
    coords_new: torch.Tensor,
    edit_grid64: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map preserved tokens between original coords ``C0`` and edited ``C1``.

    A new-coord token is *preserved* iff it also exists in ``C0`` AND lies
    outside the edit region.  Everything else (new voxels grown by S1, or old
    voxels inside the edit region) is free to be regenerated.

    Args:
        coords0:    ``[N0,3]`` original active-voxel indices (0..63).
        coords_new: ``[N1,3]`` edited active-voxel indices (0..63).
        edit_grid64: dense ``[64,64,64]`` bool edit region.

    Returns:
        preserved_new: bool ``[N1]`` — which coords_new tokens to anchor.
        src_index:     long ``[P]`` — row in ``coords0`` feeding each preserved
                       token (in coords_new order), ``P == preserved_new.sum()``.
    """
    # unify device (coords0 may be CPU from P1 npz while coords_new/edit are CUDA)
    dev = coords_new.device
    coords0 = coords0.long().to(dev)
    coords_new = coords_new.long().to(dev)
    edit_grid64 = edit_grid64.to(dev)
    k0 = coord_keys_64(coords0)
    k1 = coord_keys_64(coords_new)
    edit_at = edit_grid64[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]]
    order = torch.argsort(k0)
    k0s = k0[order]
    pos = torch.searchsorted(k0s, k1).clamp(max=max(k0s.numel() - 1, 0))
    in_c0 = (k0s[pos] == k1) if k0s.numel() else torch.zeros_like(k1, dtype=torch.bool)
    preserved_new = in_c0 & (~edit_at)
    src_index = order[pos[preserved_new]]
    return preserved_new, src_index


def incore_edit_bridge(
    coords0: torch.Tensor,
    coords_new: torch.Tensor,
    edit_grid64: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokens present in BOTH ``C0`` and the edit region → ``(mask[N1], src[P])``.

    Complement of :func:`build_coord_bridge` (which masks ``in_c0 & ~edit``):
    these are the edit-region voxels that already existed, so a masked-edit
    forward can WARM-START them from their inverted original latent (instead of
    pure noise) and then let them evolve UN-anchored toward the edited target —
    smoother than hallucinating the part from scratch.
    """
    dev = coords_new.device
    coords0 = coords0.long().to(dev)
    coords_new = coords_new.long().to(dev)
    edit_grid64 = edit_grid64.to(dev)
    k0 = coord_keys_64(coords0)
    k1 = coord_keys_64(coords_new)
    edit_at = edit_grid64[coords_new[:, 0], coords_new[:, 1], coords_new[:, 2]]
    order = torch.argsort(k0)
    k0s = k0[order]
    pos = torch.searchsorted(k0s, k1).clamp(max=max(k0s.numel() - 1, 0))
    in_c0 = (k0s[pos] == k1) if k0s.numel() else torch.zeros_like(k1, dtype=torch.bool)
    warm_new = in_c0 & edit_at
    src_index = order[pos[warm_new]]
    return warm_new, src_index


def densify_edit_occupancy(
    coords_new: torch.Tensor,
    edit_grid64: torch.Tensor,
    iters: int = 1,
) -> torch.Tensor:
    """Thicken the EDITED-region occupancy so the shape decoder can close it.

    The S1 sparse structure for a *grown* part is frequently a 1-voxel-thick
    shell (neighbour degree ~4); flexicubes then leaves an open / holey surface
    there (the "transparent grid ball").  We morphologically dilate the active
    voxels by ``iters`` cells **but only inside the (dilated) edit region**, so:

      * the preserved body stays byte-identical (no voxel added outside the
        edit region → ``build_coord_bridge`` still anchors it to the original);
      * the edited part gets a thicker, closeable band; the added voxels are
        'grown' (not in ``C0``) so the masked shape/tex flows regenerate them
        (and ``nn_init`` seeds them from the nearest existing token).

    Returns the new ``coords_new`` as an ``[M,3]`` int32 tensor (superset of the
    input) on the same device.  ``iters<=0`` returns the input unchanged.
    """
    if iters <= 0:
        return coords_new
    dev = coords_new.device
    cn = coords_new.long()
    if cn.shape[1] == 4:
        cn = cn[:, 1:]
    g = GRID_LO
    occ = torch.zeros(g, g, g, dtype=torch.bool, device=dev)
    occ[cn[:, 0], cn[:, 1], cn[:, 2]] = True
    region = _dilate_grid(edit_grid64.to(dev).bool(), iters)  # confine growth
    grown = _dilate_grid(occ, iters)
    new_occ = occ | (grown & region)
    return new_occ.nonzero(as_tuple=False).to(torch.int32)


__all__ = [
    "part_keep_mask",
    "part_edit_grid_64",
    "edit_grid_64_to_keep16",
    "edit_grid_64_to_keep16_soft",
    "coord_keys_64",
    "build_coord_bridge",
    "incore_edit_bridge",
    "densify_edit_occupancy",
]
