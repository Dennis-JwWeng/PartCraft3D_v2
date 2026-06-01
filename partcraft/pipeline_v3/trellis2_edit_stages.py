"""S2 — TRELLIS.2 geometry + material masked editing on (possibly new) coords.

After :mod:`trellis2_structure` decides ``coords_new`` (which may differ from the
original P1 coords ``C0`` where the structure was edited), these helpers run the
shape-SLat and texture-SLat flows with Vinedresser3D-style masked replacement:

    * preserved tokens (in ``C0`` and outside the edit region) are anchored to
      the ORIGINAL-image inversion trajectory via a coord bridge;
    * edit-region / newly-grown tokens are regenerated under the EDITED image.

All inversions use the ORIGINAL view image; all forward passes use the EDITED
image — so the edit signal is the conditioning change, exactly as the migration
doc prescribes.

Coords convention: every SparseTensor here lives on a single 64³ coord set
(``C0`` for the original references, ``coords_new`` for the outputs), so the
shape and texture latents that feed ``pipeline.decode_latent`` share coords.
"""
from __future__ import annotations

import torch

from .trellis2_masked_sampler import make_bridged_anchor_callback
from .trellis2_part_mask import build_coord_bridge, incore_edit_bridge


def _sp():
    import trellis2.modules.sparse as sp  # lazy: needs codebase on sys.path
    return sp


def _norm_pair(norm: dict, dev: str):
    mean = torch.tensor(norm["mean"], device=dev).float()
    std = torch.tensor(norm["std"], device=dev).float()
    return mean, std


def sparse_denorm_shape(pipeline, feats: torch.Tensor, coords3: torch.Tensor):
    """Wrap the raw (denormalized) P1 shape latent as a SparseTensor on ``C0``."""
    sp = _sp()
    dev = "cuda"
    coords = torch.cat([torch.zeros_like(coords3[:, :1]), coords3],
                       dim=1).to(dev).int()
    return sp.SparseTensor(feats=feats.to(dev).float(), coords=coords)


def _sparse_on(coords3: torch.Tensor, feats: torch.Tensor):
    sp = _sp()
    dev = "cuda"
    coords = torch.cat([torch.zeros_like(coords3[:, :1]), coords3],
                       dim=1).to(dev).int()
    return sp.SparseTensor(feats=feats.to(dev), coords=coords.to(dev))


@torch.no_grad()
def masked_shape_slat(
    pipeline, sampler,
    p1_feats: torch.Tensor,
    coords0: torch.Tensor,
    coords_new: torch.Tensor,
    edit_grid64: torch.Tensor,
    cond_orig: dict,
    cond_edit: dict,
    logger,
    warmstart: bool = False,
    nn_init: bool = False,
    anchor_mode: str = "perstep",
    anchor_cutoff: float = 0.3,
):
    """Edit shape SLat on ``coords_new``; preserve non-edit tokens.

    Returns the denormalized shape SLat (SparseTensor on ``coords_new``).

    ``warmstart`` initializes edit-region tokens that already existed in ``C0``
    from their inverted original latent (rather than pure ``randn``), leaving
    them UN-anchored so the flow morphs them toward the edit instead of
    hallucinating from scratch — reduces surface jitter / spikes in the edited
    part (the occupancy is a clean shell; the spikes come from noisy repaint
    feats).  Newly-grown voxels (not in ``C0``) still start from noise.

    ``nn_init`` (implies warmstart) ALSO seeds those newly-grown edit voxels —
    which warmstart cannot reach because S1 relocated them — from their spatially
    nearest already-seeded token's feats, so the WHOLE edit region has a coherent
    init instead of a noisy core.  This is what finishes the turret-core spikes
    that warmstart alone leaves behind.

    ``anchor_mode`` controls HOW the preserved (body) tokens are pinned to the
    inverted original — which determines edit-region surface quality:

      * ``"perstep"`` (legacy): overwrite preserved tokens with the inverted
        trajectory at EVERY step.  Exact body preservation, but the edit-region
        tokens attend to off-distribution (inverted-original) neighbours the
        whole way → the generated part decodes as a HOLEY / see-through shell
        (a free ``pipeline.run`` of the same image makes a SOLID part, proving
        the per-step anchor is what breaks it).
      * ``"release_late"``: anchor only while ``t >= anchor_cutoff`` (high
        noise), then release so ALL tokens denoise together for the last
        ``anchor_cutoff`` fraction → the part heals into a closed surface while
        the body, structure-locked early, barely drifts.
      * ``"posthoc"``: NO per-step anchor (free, vanilla-quality generation of
        the whole field under the edited image), then hard-overwrite the
        preserved tokens with the original clean latent at the end → solid part
        + exact body, at the cost of a possible 1-voxel boundary seam.
      * ``"free"``: like ``posthoc`` but WITHOUT the final body paste — the
        whole shape SLat is generated freely on ``coords_new`` under the edited
        image.  The body is preserved only STRUCTURALLY (the original occupancy
        coords carried through masked S1), so the surface is fully coherent and
        seamless, at the cost of the body not being latent-identical.  Use when
        a clean seamless mesh matters more than bit-exact body preservation.
    """
    dev = "cuda"
    flow = pipeline.models["shape_slat_flow_model_1024"]
    params = pipeline.shape_slat_sampler_params
    mean, std = _norm_pair(pipeline.shape_slat_normalization, dev)
    gi = params.get("guidance_interval", (0.0, 1.0))
    grescale = params.get("guidance_rescale", 0.0)
    steps = int(params["steps"])
    rescale_t = float(params.get("rescale_t", 1.0))
    gs_fwd = float(params.get("guidance_strength", 7.5))

    clean = _sparse_on(coords0, ((p1_feats.to(dev) - mean) / std).contiguous())

    if pipeline.low_vram:
        flow.to(dev)
    inv = sampler.invert_clean(
        flow, clean,
        cond=cond_orig["cond"], neg_cond=cond_orig["neg_cond"],
        guidance_strength=1.0, guidance_interval=gi, guidance_rescale=grescale,
        steps=steps, rescale_t=rescale_t, verbose=False, tqdm_desc="S2 shape inv",
    )

    preserved, src_idx = build_coord_bridge(coords0, coords_new, edit_grid64)
    preserved, src_idx = preserved.to(dev), src_idx.to(dev)
    feats_init = torch.randn(coords_new.shape[0], flow.in_channels, device=dev)
    feats_init[preserved] = inv[1.0].feats[src_idx]
    seeded = preserved.clone()
    if warmstart or nn_init:
        warm, warm_src = incore_edit_bridge(coords0, coords_new, edit_grid64)
        warm, warm_src = warm.to(dev), warm_src.to(dev)
        feats_init[warm] = inv[1.0].feats[warm_src]
        seeded = seeded | warm
        logger.info("[s5/S2] shape warm-start: %d edit tokens seeded from "
                    "inverted original (%d newly-grown %s)",
                    int(warm.sum()), int((~seeded).sum()),
                    "→ nearest-neighbor init" if nn_init else "stay noise")
    if nn_init:
        unseeded = ~seeded
        if bool(unseeded.any()) and bool(seeded.any()):
            cN = coords_new.to(dev).float()
            if cN.shape[1] == 4:
                cN = cN[:, 1:]
            src_pts = cN[seeded]            # [S,3] already-seeded coords
            qry_pts = cN[unseeded]          # [U,3] newly-grown coords
            nn = torch.cdist(qry_pts, src_pts).argmin(dim=1)  # [U]
            feats_init[unseeded] = feats_init[seeded][nn]
            logger.info("[s5/S2] shape NN-init: %d newly-grown tokens seeded "
                        "from nearest of %d seeded",
                        int(unseeded.sum()), int(seeded.sum()))
    if anchor_mode in ("posthoc", "free"):
        # free (vanilla-quality) generation of the whole field — start every
        # token from pure noise so the edit region is sampled exactly like a
        # fresh pipeline.run (no inverted-original neighbours to fight).
        # "posthoc" pastes the body latent back at the end (exact preservation);
        # "free" keeps the whole field as generated — the body is preserved only
        # STRUCTURALLY, by reusing the original occupancy coords from masked S1,
        # so the surface is fully coherent (no boundary seam) at the cost of the
        # body no longer being latent-identical to the original.
        feats_init = torch.randn(coords_new.shape[0], flow.in_channels, device=dev)
        cb = None
    elif anchor_mode == "release_late":
        cb = make_bridged_anchor_callback(
            inv, preserved, src_idx, schedule="early", cutoff_t=float(anchor_cutoff))
    else:  # "perstep"
        cb = make_bridged_anchor_callback(inv, preserved, src_idx)
    x_init = _sparse_on(coords_new, feats_init)
    logger.info("[s5/S2] shape anchor_mode=%s%s", anchor_mode,
                f" (cutoff_t={anchor_cutoff})" if anchor_mode == "release_late" else "")

    out = sampler.sample(
        flow, x_init,
        cond=cond_edit["cond"], neg_cond=cond_edit["neg_cond"],
        steps=steps, rescale_t=rescale_t,
        guidance_strength=gs_fwd, guidance_interval=gi, guidance_rescale=grescale,
        verbose=False, tqdm_desc="S2 shape",
        x_init=x_init, step_callback=cb,
    ).samples
    if anchor_mode == "posthoc":
        # hard-preserve the body: overwrite preserved tokens with the original
        # clean latent (exact reconstruction outside the edit region).
        new_feats = out.feats.clone()
        new_feats[preserved] = clean.feats[src_idx]
        out = out.replace(new_feats)
    if pipeline.low_vram:
        flow.cpu()
    return out * std + mean


@torch.no_grad()
def masked_tex_slat(
    pipeline, sampler,
    shape0_denorm,
    shape_new_denorm,
    coords0: torch.Tensor,
    coords_new: torch.Tensor,
    edit_grid64: torch.Tensor,
    cond_orig: dict,
    cond_edit: dict,
    logger,
    anchor_mode: str = "perstep",
):
    """Edit texture SLat on ``coords_new``; preserve non-edit material.

    The reference texture is the model's own texture for the ORIGINAL image on
    ``C0`` (so coords always align with the original shape); it is inverted and
    re-injected outside the edit region while the edit region regenerates under
    the EDITED image + edited shape.  Returns denormalized tex SLat on
    ``coords_new``.
    """
    dev = "cuda"
    flow = pipeline.models["tex_slat_flow_model_1024"]
    params = pipeline.tex_slat_sampler_params
    s_mean, s_std = _norm_pair(pipeline.shape_slat_normalization, dev)
    t_mean, t_std = _norm_pair(pipeline.tex_slat_normalization, dev)
    gi = params.get("guidance_interval", (0.0, 1.0))
    grescale = params.get("guidance_rescale", 0.0)
    steps = int(params["steps"])
    rescale_t = float(params.get("rescale_t", 1.0))
    gs = float(params.get("guidance_strength", 1.0))

    shape0_norm = (shape0_denorm - s_mean) / s_std
    shape_new_norm = (shape_new_denorm - s_mean) / s_std
    tex_dim = flow.in_channels - shape0_norm.feats.shape[1]

    if anchor_mode == "free":
        # fully free material on coords_new under the EDITED image + edited
        # shape — no original-texture reference, no inversion, no per-step
        # anchor → seamless texture matching the seamless free shape.
        if pipeline.low_vram:
            flow.to(dev)
        feats_init = torch.randn(coords_new.shape[0], tex_dim, device=dev)
        x_init = shape_new_norm.replace(feats_init)
        logger.info("[s5/S2] tex anchor_mode=free (no reference inversion)")
        out = sampler.sample(
            flow, x_init,
            cond=cond_edit["cond"], neg_cond=cond_edit["neg_cond"],
            steps=steps, rescale_t=rescale_t,
            guidance_strength=gs, guidance_interval=gi, guidance_rescale=grescale,
            concat_cond=shape_new_norm,
            verbose=False, tqdm_desc="tex (free)",
            x_init=x_init, step_callback=None,
        ).samples
        if pipeline.low_vram:
            flow.cpu()
        return out * t_std + t_mean

    # 1. reference: original-image texture on C0 (conditioned on original shape)
    tex0 = pipeline.sample_tex_slat(cond_orig, flow, shape0_denorm, {})
    tex0_norm = (tex0 - t_mean) / t_std

    if pipeline.low_vram:
        flow.to(dev)
    # 2. invert reference under ORIGINAL image + original shape
    inv = sampler.invert_clean(
        flow, tex0_norm,
        cond=cond_orig["cond"], neg_cond=cond_orig["neg_cond"],
        guidance_strength=gs, guidance_interval=gi, guidance_rescale=grescale,
        concat_cond=shape0_norm,
        steps=steps, rescale_t=rescale_t, verbose=False, tqdm_desc="tex inv",
    )

    # 3. masked forward on coords_new under EDITED image + edited shape
    preserved, src_idx = build_coord_bridge(coords0, coords_new, edit_grid64)
    preserved, src_idx = preserved.to(dev), src_idx.to(dev)
    feats_init = torch.randn(coords_new.shape[0], tex_dim, device=dev)
    feats_init[preserved] = inv[1.0].feats[src_idx]
    x_init = shape_new_norm.replace(feats_init)
    cb = make_bridged_anchor_callback(inv, preserved, src_idx)

    out = sampler.sample(
        flow, x_init,
        cond=cond_edit["cond"], neg_cond=cond_edit["neg_cond"],
        steps=steps, rescale_t=rescale_t,
        guidance_strength=gs, guidance_interval=gi, guidance_rescale=grescale,
        concat_cond=shape_new_norm,
        verbose=False, tqdm_desc="tex",
        x_init=x_init, step_callback=cb,
    ).samples
    if pipeline.low_vram:
        flow.cpu()
    return out * t_std + t_mean


__all__ = ["sparse_denorm_shape", "masked_shape_slat", "masked_tex_slat"]
