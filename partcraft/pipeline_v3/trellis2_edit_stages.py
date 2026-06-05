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
    contact_mask: torch.Tensor | None = None,
    contact_sigma: float | None = None,
    res: int = 1024,
):
    """Edit shape SLat on ``coords_new``; preserve non-edit tokens.

    ``res`` selects the SLat flow-model resolution (``shape_slat_flow_model_{res}``)
    and the coord grid (``res//16``: 64³ for 1024, 32³ for 512).  ``coords0`` /
    ``coords_new`` / ``edit_grid64`` must already live on that grid.

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
    grid = res // 16
    flow = pipeline.models[f"shape_slat_flow_model_{res}"]
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

    preserved, src_idx = build_coord_bridge(coords0, coords_new, edit_grid64, grid=grid)
    preserved, src_idx = preserved.to(dev), src_idx.to(dev)
    feats_init = torch.randn(coords_new.shape[0], flow.in_channels, device=dev)
    feats_init[preserved] = inv[1.0].feats[src_idx]
    seeded = preserved.clone()
    if warmstart or nn_init:
        warm, warm_src = incore_edit_bridge(coords0, coords_new, edit_grid64, grid=grid)
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
    elif anchor_mode == "contact_soft":
        # v1-faithful: preserved tokens NEAR the contact boundary are allowed to
        # blend with the generation (heals the seam); tokens far from contact are
        # fully anchored to the inverted original.  Per-token weight from the
        # contact-distance soft mask (mirrors interweave get_s2_soft_mask).
        from .trellis2_contact_mask import get_s2_soft_mask
        c3 = coords_new.to(dev)
        c3 = c3[:, 1:] if c3.shape[1] == 4 else c3
        sigma = float(contact_sigma) if contact_sigma is not None else 5.0
        soft_w = get_s2_soft_mask(
            c3[preserved], edit_grid64, sigma=sigma, contact_mask=contact_mask
        ).to(dev)
        cb = make_bridged_anchor_callback(
            inv, preserved, src_idx, soft_w=soft_w)
        logger.info("[s5/S2] shape contact-soft: %d preserved tokens, "
                    "blend-weight range [%.2f, %.2f] (sigma=%.2f)",
                    int(preserved.sum()), float(soft_w.min()) if soft_w.numel() else 0.0,
                    float(soft_w.max()) if soft_w.numel() else 0.0, sigma)
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
    contact_mask: torch.Tensor | None = None,
    contact_sigma: float | None = None,
    res: int = 1024,
    before_tex_denorm: torch.Tensor | None = None,
):
    """Edit texture SLat on ``coords_new``; preserve non-edit material.

    ``res`` selects ``tex_slat_flow_model_{res}`` and the coord grid (``res//16``);
    all coords / ``edit_grid64`` must already live on that grid.

    The reference texture is the model's own texture for the ORIGINAL image on
    ``C0`` (so coords always align with the original shape); it is inverted and
    re-injected outside the edit region while the edit region regenerates under
    the EDITED image + edited shape.  Returns denormalized tex SLat on
    ``coords_new``.

    ``before_tex_denorm`` ([N0, tex_dim] aligned to ``coords0``) is the P1
    VAE-**encoded** original texture latent (``tex_slat_e{res}.npz``).  When given
    with ``anchor_mode="posthoc"`` it is hard-pasted onto the preserved tokens at
    the end (S1 ``restore_preserved``-style) instead of the re-sampled reference —
    the kept region then decodes to the EXACT original material, not a re-render.
    """
    dev = "cuda"
    grid = res // 16
    flow = pipeline.models[f"tex_slat_flow_model_{res}"]
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

    _restore = (anchor_mode == "posthoc" and before_tex_denorm is not None
                and before_tex_denorm.shape[0] == coords0.shape[0])
    if anchor_mode == "posthoc" and before_tex_denorm is not None and not _restore:
        logger.warning("[s5/S2] tex restore skipped: before_tex rows (%d) != "
                       "coords0 (%d) — falling back to re-sampled posthoc",
                       int(before_tex_denorm.shape[0]), int(coords0.shape[0]))
    if _restore:
        # RESTORE-style (mirrors S1 restore_preserved): generate the whole
        # material field FREELY under the edited image + edited shape, then
        # hard-paste the P1-ENCODED *original* tex latent onto the preserved
        # tokens — so the kept region decodes to the EXACT original material
        # (not a re-rendered guess).  No reference re-sample, no inversion.
        preserved, src_idx = build_coord_bridge(coords0, coords_new, edit_grid64, grid=grid)
        preserved, src_idx = preserved.to(dev), src_idx.to(dev)
        before = before_tex_denorm.to(dev).float()
        before_norm = (before - t_mean) / t_std
        if pipeline.low_vram:
            flow.to(dev)
        feats_init = torch.randn(coords_new.shape[0], tex_dim, device=dev)
        x_init = shape_new_norm.replace(feats_init)
        logger.info("[s5/S2] tex anchor_mode=posthoc-restore (free gen + paste "
                    "BEFORE-encoded tex, %d/%d preserved tokens)",
                    int(preserved.sum()), coords_new.shape[0])
        out = sampler.sample(
            flow, x_init,
            cond=cond_edit["cond"], neg_cond=cond_edit["neg_cond"],
            steps=steps, rescale_t=rescale_t,
            guidance_strength=gs, guidance_interval=gi, guidance_rescale=grescale,
            concat_cond=shape_new_norm,
            verbose=False, tqdm_desc="tex (posthoc-restore)",
            x_init=x_init, step_callback=None,
        ).samples
        new_feats = out.feats.clone()
        new_feats[preserved] = before_norm[src_idx]
        out = out.replace(new_feats)
        if pipeline.low_vram:
            flow.cpu()
        return out * t_std + t_mean

    # 1. reference: original-image texture on C0 (conditioned on original shape)
    tex0 = pipeline.sample_tex_slat(cond_orig, flow, shape0_denorm, {})
    tex0_norm = (tex0 - t_mean) / t_std

    preserved, src_idx = build_coord_bridge(coords0, coords_new, edit_grid64, grid=grid)
    preserved, src_idx = preserved.to(dev), src_idx.to(dev)

    if anchor_mode == "posthoc":
        # Free (vanilla-quality) generation of the whole material field under the
        # EDITED image + edited shape, then HARD-OVERWRITE the preserved tokens
        # with the ORIGINAL-image reference texture (``tex0``) at the end.  This
        # gives exact body material (no edited-image colour/lighting drift bleeding
        # into the kept region, no inverted-original neighbours dragging the edit)
        # + a clean edit region.  Mirrors shape ``posthoc``; possible 1-voxel
        # boundary seam (use ``contact_soft`` if that matters).  No inversion
        # needed (we paste the clean ``tex0``, not an inverted trajectory) → also
        # skips the costly invert_clean pass.
        if pipeline.low_vram:
            flow.to(dev)
        feats_init = torch.randn(coords_new.shape[0], tex_dim, device=dev)
        x_init = shape_new_norm.replace(feats_init)
        logger.info("[s5/S2] tex anchor_mode=posthoc (free gen + final body paste, "
                    "%d/%d preserved tokens)", int(preserved.sum()),
                    coords_new.shape[0])
        out = sampler.sample(
            flow, x_init,
            cond=cond_edit["cond"], neg_cond=cond_edit["neg_cond"],
            steps=steps, rescale_t=rescale_t,
            guidance_strength=gs, guidance_interval=gi, guidance_rescale=grescale,
            concat_cond=shape_new_norm,
            verbose=False, tqdm_desc="tex (posthoc)",
            x_init=x_init, step_callback=None,
        ).samples
        new_feats = out.feats.clone()
        new_feats[preserved] = tex0_norm.feats[src_idx]
        out = out.replace(new_feats)
        if pipeline.low_vram:
            flow.cpu()
        return out * t_std + t_mean

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
    feats_init = torch.randn(coords_new.shape[0], tex_dim, device=dev)
    feats_init[preserved] = inv[1.0].feats[src_idx]
    x_init = shape_new_norm.replace(feats_init)
    if anchor_mode == "contact_soft":
        from .trellis2_contact_mask import get_s2_soft_mask
        c3 = coords_new.to(dev)
        c3 = c3[:, 1:] if c3.shape[1] == 4 else c3
        sigma = float(contact_sigma) if contact_sigma is not None else 5.0
        soft_w = get_s2_soft_mask(
            c3[preserved], edit_grid64, sigma=sigma, contact_mask=contact_mask
        ).to(dev)
        cb = make_bridged_anchor_callback(
            inv, preserved, src_idx, soft_w=soft_w)
        logger.info("[s5/S2] tex contact-soft: %d preserved tokens (sigma=%.2f)",
                    int(preserved.sum()), sigma)
    else:
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
