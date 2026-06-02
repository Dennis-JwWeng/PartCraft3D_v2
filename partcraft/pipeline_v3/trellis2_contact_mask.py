"""Contact-aware soft masks for TRELLIS.2 masked editing — a faithful port of
the v1 (``interweave_Trellis.py``) tricks that v2 dropped.

v1's SS-mask flow editing did NOT cut a hard boundary at the edit region.  It
analyzed how the edit surface *contacts* the preserved geometry and feathered
the preserve↔edit transition with a **distance-transform soft mask** whose width
(``sigma``) scaled with the contact ratio:

  * deeply embedded part (leg in body, ratio≈0.5) → wide blend → closes the hole;
  * loosely attached part (hat on head, ratio≈0.05) → narrow blend → mostly
    preserve.

This module ports the four functions that implement that, adapted to v2's
conventions (``[N,3]`` int coords instead of v1's batched ``[N,4]`` SLAT coords,
and a dense ``[64,64,64]`` bool edit grid instead of v1's ``mask``):

  * :func:`compute_contact_boundary` — contact_64 + dynamic S1/S2 sigma
  * :func:`get_s1_soft_mask`         — 16³ float keep-weight (1=preserve)
  * :func:`get_s2_soft_mask`         — per-preserved-token float weight
  * :func:`remove_small_components`  — drop floating specks in the edit region

The formulas / constants are kept identical to v1 so the edited structure
matches the TRELLIS.1 pipeline.  See ``third_party/interweave_Trellis.py`` in the
v1 repo for the originals.
"""
from __future__ import annotations

import numpy as np
import torch

GRID_LO = 64
GRID_S1 = 16


def _coords3(coords: torch.Tensor) -> torch.Tensor:
    """Accept ``[N,3]`` or batched ``[N,4]`` coords; return the ``xyz`` columns."""
    return coords[:, 1:] if coords.shape[1] == 4 else coords


def compute_contact_boundary(
    edit_grid64: torch.Tensor,
    coords0: torch.Tensor,
    device=None,
):
    """Analyze how the edit region connects to preserved geometry (v1 port).

    Args:
        edit_grid64: dense ``[64,64,64]`` bool edit region.
        coords0:     ``[N,3]`` (or ``[N,4]``) original active-voxel indices —
                     the preserved geometry outside the edit region.

    Returns:
        contact_64:    ``[64,64,64]`` bool — edit voxels 6-adjacent to a
                       preserved voxel.
        contact_ratio: float in [0,1] — fraction of the edit surface that
                       touches preserved geometry.
        s1_sigma:      dynamic S1 (16³) blend width ∈ [1.5, 5.5].
        s2_sigma:      dynamic S2 (64³) blend width ∈ [2.0, 12.0].
    """
    from scipy import ndimage

    device = device or edit_grid64.device
    mask = edit_grid64.to(device).bool()
    sc = _coords3(coords0).long().to(device)

    # preserved occupancy = original voxels OUTSIDE the edit region
    preserved = torch.zeros(64, 64, 64, dtype=torch.bool, device=device)
    if sc.shape[0] > 0:
        in_mask = mask[sc[:, 0], sc[:, 1], sc[:, 2]]
        pres_sc = sc[~in_mask]
        if pres_sc.shape[0] > 0:
            preserved[pres_sc[:, 0], pres_sc[:, 1], pres_sc[:, 2]] = True

    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    pres_dilated = torch.from_numpy(
        ndimage.binary_dilation(
            preserved.cpu().numpy(), structure=struct, iterations=1)
    ).to(device)
    contact_64 = mask & pres_dilated

    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_dilated = ndimage.binary_dilation(mask_np, structure=struct, iterations=1)
    edit_surface = (torch.from_numpy(mask_dilated).to(device) & ~mask)
    edit_surface_count = max(int(edit_surface.sum()), 1)

    contact_count = int(contact_64.sum())
    contact_ratio = min(contact_count / edit_surface_count, 1.0)

    # more contact → wider transition needed (v1 constants, verbatim)
    s1_sigma = 1.5 + contact_ratio * 4.0    # [1.5, 5.5]
    s2_sigma = 2.0 + contact_ratio * 10.0   # [2.0, 12.0]
    return contact_64, contact_ratio, s1_sigma, s2_sigma


def get_s1_soft_mask(
    edit_grid64: torch.Tensor,
    sigma: float = 3.0,
    contact_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft S1 (16³) keep-weight with distance-based decay (v1 port).

    Returns a ``[16,16,16]`` float tensor where 1.0 = fully preserve (far from
    the contact boundary), 0.0 = fully editable (inside the edit region), and
    ``(0,1)`` = smooth transition.  When ``contact_mask`` is given the decay is
    measured from the contact boundary only, so preserved blocks far from any
    contact stay fully preserved (== anchored to the inverted original).

    NOTE: v1 returned ``[1,8,16,16,16]``; here we return the bare ``[16,16,16]``
    and let the caller add ``[None, None]`` to broadcast over the 8 SS channels.
    """
    from scipy import ndimage

    mask_input = edit_grid64.bool()
    mask_reshaped = mask_input.float().reshape(16, 4, 16, 4, 16, 4)
    edit_frac = mask_reshaped.mean(dim=(1, 3, 5))
    edit_16 = (edit_frac >= 0.1).cpu().numpy().astype(np.uint8)

    contact_16 = None
    if contact_mask is not None:
        contact_reshaped = contact_mask.float().reshape(16, 4, 16, 4, 16, 4)
        contact_frac = contact_reshaped.mean(dim=(1, 3, 5))
        contact_16 = (contact_frac > 0).cpu().numpy().astype(np.uint8)

    if contact_16 is not None and contact_16.sum() > 0:
        # distance from the contact boundary (edit↔preserved interface)
        dist = ndimage.distance_transform_edt(1 - contact_16).astype(np.float32)
    else:
        # No contact (e.g. a whole-object part — nothing to anchor against):
        # fall back to distance from the edit boundary so the edit region is
        # FREE (soft→0 there) instead of wrongly preserving the interior.
        # NOTE: EDT of an all-nonzero array measures distance to the array edge,
        # so guarding the empty-contact case here is essential — otherwise a
        # whole-object edit's core would get a high preserve weight and never
        # change.  edit_16≈all → preserved_16≈∅ → dist=0 → soft=0 → free gen.
        dist = ndimage.distance_transform_edt(1 - edit_16).astype(np.float32)

    soft = 1.0 - np.exp(-dist / max(sigma, 0.1))
    soft[edit_16 == 1] = 0.0
    return torch.from_numpy(soft).to(edit_grid64.device).float()


def get_s2_soft_mask(
    preserved_coords3: torch.Tensor,
    edit_grid64: torch.Tensor,
    sigma: float = 5.0,
    contact_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-preserved-token soft weight for S2 (64³) (v1 port).

    Distance is measured from the contact boundary (or the whole edit region if
    no contact_mask): preserved tokens far from any contact get weight≈0 (fully
    preserve / anchor to inverted original), those near contact get weight≈1
    (allow generation to blend → heals the seam).

    Returns ``[P]`` float aligned with ``preserved_coords3`` rows.

    Uses ``scipy.ndimage.distance_transform_edt`` on the 64³ grid (Euclidean)
    rather than v1's per-point Manhattan KNN — the trellis2 env has no sklearn,
    and the grid EDT is both faster and consistent with :func:`get_s1_soft_mask`.
    The exp decay ``w = exp(-dist/sigma)`` keeps the same near→blend, far→preserve
    behaviour; ``sigma`` is the same dynamic value from
    :func:`compute_contact_boundary`.
    """
    from scipy import ndimage

    pc = _coords3(preserved_coords3).long()
    if pc.shape[0] == 0:
        return torch.zeros(0, device=preserved_coords3.device, dtype=torch.float32)
    ref = (contact_mask if contact_mask is not None else edit_grid64).bool()
    ref_np = ref.cpu().numpy()
    if not ref_np.any():
        return torch.zeros(pc.shape[0], device=preserved_coords3.device,
                           dtype=torch.float32)
    # distance from each voxel to the nearest ref (contact) voxel: 0 on ref,
    # growing outward into the preserved region.
    dist = ndimage.distance_transform_edt(~ref_np).astype(np.float32)
    cn = pc.cpu().numpy()
    d = dist[cn[:, 0], cn[:, 1], cn[:, 2]]
    w = np.exp(-d / max(float(sigma), 1e-6)).astype(np.float32)
    return torch.from_numpy(w).to(preserved_coords3.device)


def remove_small_components(
    coords3: torch.Tensor, min_size: int = 50,
) -> torch.Tensor:
    """Return a bool ``[N]`` keep-mask dropping voxels in 6-connected components
    smaller than ``min_size`` (v1 port; uses ``scipy.ndimage.label`` for speed
    instead of v1's Python BFS — identical 6-connectivity result)."""
    from scipy import ndimage

    c = _coords3(coords3).long()
    if c.shape[0] == 0:
        return torch.ones(0, dtype=torch.bool, device=coords3.device)
    grid = np.zeros((64, 64, 64), dtype=np.uint8)
    cn = c.cpu().numpy()
    grid[cn[:, 0], cn[:, 1], cn[:, 2]] = 1
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    labels, n = ndimage.label(grid, structure=struct)
    if n == 0:
        return torch.zeros(c.shape[0], dtype=torch.bool, device=coords3.device)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0                                   # background
    keep_label = sizes >= min_size                 # per-label keep flag
    vox_label = labels[cn[:, 0], cn[:, 1], cn[:, 2]]
    keep = keep_label[vox_label]
    return torch.from_numpy(keep).to(coords3.device)


__all__ = [
    "compute_contact_boundary",
    "get_s1_soft_mask",
    "get_s2_soft_mask",
    "remove_small_components",
]
