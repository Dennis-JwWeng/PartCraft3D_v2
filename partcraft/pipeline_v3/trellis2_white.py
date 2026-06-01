"""White-model (白模) handling for the TRELLIS.2 s5 stage.

Some inputs are untextured white/grey meshes.  For these, Phase-1 already drops
material/color edits (see ``gen_edits_image.py``); here we additionally **skip
the TRELLIS.2 third stage — texture SLat generation** (``sample_tex_slat``, the
expensive iterative flow sampling).  The output GLB keeps the edited geometry
and carries a flat white/grey PBR material, matching the untextured input.

Why we still call ``decode_latent`` (not pure geometry): the per-voxel structure
that ``MeshWithVoxel`` needs is produced by the texture *decoder* (guided by the
shape substructures), not by the texture *sampler*.  So we feed a dummy (zero)
texture latent — which costs one cheap decoder forward pass but skips the costly
sampler — then overwrite the decoded attrs with a constant grey PBR.
"""
from __future__ import annotations

import json


# Flat-grey PBR (overrides the decoded texture attrs).  base_color in [0,1].
WHITE_BASE_COLOR = 0.60   # neutral grey albedo
WHITE_METALLIC = 0.0
WHITE_ROUGHNESS = 0.7     # matte
WHITE_ALPHA = 1.0


def read_white_model_flag(ctx) -> bool:
    """True if Phase-1 flagged this object as an untextured white model.

    Reads ``phase1/visibility.json`` (written by gen_edits_image's pre-pass).
    Missing/unreadable → False (treat as textured; safe default)."""
    try:
        p = ctx.phase1_dir / "visibility.json"
        return bool(json.loads(p.read_text()).get("white_model", False))
    except Exception:
        return False


def build_white_model_mesh(pipeline, shape_slat, logger=None):
    """Decode ``shape_slat`` into a MeshWithVoxel with a flat grey PBR material,
    skipping the texture SLat sampler (TRELLIS.2 third stage).

    Mirrors ``pipeline.decode_latent`` but supplies a zero texture latent and
    then forces constant grey attrs, so no texture is generated/hallucinated.
    """
    import torch

    # Dummy texture latent: same sparse coords as the shape latent (this is what
    # sample_tex_slat would have produced), zero feats of the tex-latent dim.
    tex_ch = len(pipeline.tex_slat_normalization["mean"])
    dummy_tex = shape_slat.replace(
        feats=torch.zeros(
            shape_slat.coords.shape[0], tex_ch,
            device=shape_slat.device, dtype=shape_slat.feats.dtype,
        )
    )

    meshes = pipeline.decode_latent(shape_slat, dummy_tex, 1024)
    mesh = meshes[0]

    # Overwrite the (garbage) decoded attrs with a constant grey PBR.
    lay = pipeline.pbr_attr_layout
    a = mesh.attrs
    if "base_color" in lay:
        a[..., lay["base_color"]] = WHITE_BASE_COLOR
    if "metallic" in lay:
        a[..., lay["metallic"]] = WHITE_METALLIC
    if "roughness" in lay:
        a[..., lay["roughness"]] = WHITE_ROUGHNESS
    if "alpha" in lay:
        a[..., lay["alpha"]] = WHITE_ALPHA

    if logger is not None:
        logger.info("[s5/P4] white-model decode: skipped tex sampler, "
                    "flat grey PBR (%d voxels)", int(a.shape[0]))
    return mesh


__all__ = ["read_white_model_flag", "build_white_model_mesh",
           "WHITE_BASE_COLOR", "WHITE_METALLIC", "WHITE_ROUGHNESS", "WHITE_ALPHA"]
