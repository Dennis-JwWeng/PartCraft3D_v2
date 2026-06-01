"""Step s5 (trellis2 variant) — image-driven 3D regeneration via TRELLIS.2.

Parallel to ``trellis_3d.py`` but uses the TRELLIS.2 image-to-3D pipeline
instead of v1's SLAT-space Flow Inversion / mask repaint. Because v2 does
not expose a latent editing API, the "edit" here is implemented as a full
re-generation from the FLUX-edited 2D image:

    edits_2d/{edit_id}_edited.png ──► Trellis2ImageTo3DPipeline.run ──► mesh ──► after.glb

For symmetry with the v1 contract, the original view image is also
re-generated to produce ``before.glb`` (you may turn this off via
``p25_cfg['emit_before'] = False`` to halve runtime).

Outputs per edit_id::

    ctx.edit_3d_dir(edit_id)/before.glb
    ctx.edit_3d_dir(edit_id)/after.glb

Downstream stages (``preview_flux``, ``render_3d``) currently read v1's
``after.npz`` — they will need a sibling trellis2 variant or to be
adapted to read GLB. That is out of scope for this file.

This runner is **single-GPU**; the orchestrator slices contexts across
GPUs via ``CUDA_VISIBLE_DEVICES`` subprocesses.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from .paths import ObjectContext
from . import services_cfg as psvc
from .specs import EditSpec, iter_flux_specs
from .status import update_step, STATUS_OK, STATUS_FAIL
from .qc_io import is_gate_a_failed
from .edit_status_io import edit_needs_step, update_edit_stage, obj_needs_stage


# v2 has no in-SLAT "edit type". All edit_types are handled the same way:
# regenerate from the edited 2D image. We keep the same GPU_TYPES set so
# the stage gating logic upstream still selects the same edits.
GPU_TYPES = frozenset({"modification", "scale", "material", "global"})


@dataclass
class Trellis2Result:
    obj_id: str
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0
    error: str | None = None


# ─────────────────── pipeline construction ──────────────────────────

def _ensure_pipeline(p25_cfg: dict, logger):
    """Load Trellis2ImageTo3DPipeline once per process."""
    # Lazy imports so the module can be imported even when the trellis2
    # codebase / weights are not present (e.g., dry runs).
    sys.path.insert(0, str(Path(p25_cfg.get(
        "trellis2_codebase", "/mnt/zsn/3dobject/TRELLIS.2")).resolve()))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from partcraft.pipeline_v3.trellis2_compat import patch_dinov3_extractor
    patch_dinov3_extractor()  # transformers 5.x DINOv3 layer nesting

    ckpt = p25_cfg.get("trellis2_ckpt", "/mnt/zsn/ckpts/TRELLIS.2-4B")
    logger.info("[s5] loading Trellis2ImageTo3DPipeline from %s", ckpt)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(ckpt)
    pipeline.cuda()
    logger.info("[s5] TRELLIS.2 pipeline ready")
    return pipeline


def _full_center_scale_np(mesh_npz: Path):
    """center+scale mapping full.glb → [-0.5,0.5]^3 (SAME recipe as encode /
    part_mask ``_full_center_scale``).  Returned as numpy (center[3], scale)."""
    import io as _io
    import numpy as _np
    import trimesh
    d = _np.load(str(mesh_npz), allow_pickle=True)
    g = trimesh.load(_io.BytesIO(d["full.glb"].tobytes()),
                     file_type="glb", process=False)
    if isinstance(g, trimesh.Scene):
        g = trimesh.util.concatenate(
            [m for m in g.geometry.values() if isinstance(m, trimesh.Trimesh)])
    v = _np.asarray(g.vertices, dtype=_np.float64)
    vmin, vmax = v.min(0), v.max(0)
    center = (vmin + vmax) / 2.0
    scale = 0.99999 / (vmax - vmin).max()
    return center, float(scale)


def _partverse_reframe_matrix(mesh_npz: Path):
    """4×4 mapping the TRELLIS-export GLB back into the full.glb world frame.

    TRELLIS's ``to_glb`` exports in a canonical frame that is the o-voxel
    (encode) frame rotated by ``R = R_X90.T`` ((x,y,z)→(x,z,-y), tipping the
    object onto its end) and normalized to [-0.5,0.5].  The o-voxel frame is
    ``(full_world - center) * scale``.  So to undo both::

        full_world = R⁻¹ @ export / scale + center        (R⁻¹ = R_X90)

    Applying this puts after.glb back exactly on top of the original full.glb,
    so 2D condition / mask overlays / before / after all share one frame and
    the existing partverse render/VLM/preview cameras work unchanged.
    """
    import numpy as _np
    center, scale = _full_center_scale_np(mesh_npz)
    r_inv = _np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=_np.float64)  # R_X90
    m = _np.eye(4)
    m[:3, :3] = r_inv / scale
    m[:3, 3] = center
    return m


def _run_and_export(pipeline, image, out_path: Path, p25_cfg: dict, logger,
                    *, mesh_obj=None, reframe_mesh_npz: Path | None = None):
    """Run trellis2 on a single PIL image and export GLB to ``out_path``.

    If ``mesh_obj`` is given, skip the pipeline call and export that mesh
    directly (used by the P4 branch which builds the mesh outside).

    If ``reframe_mesh_npz`` is given, the exported GLB is rigidly transformed
    from TRELLIS's canonical export frame back into the original full.glb
    (partverse) world frame via :func:`_partverse_reframe_matrix`, so the
    output shares one coordinate frame with the 2D condition and the edit mask.
    """
    import o_voxel  # type: ignore

    pipeline_type = p25_cfg.get("trellis2_pipeline_type", "1024_cascade")
    seed = int(p25_cfg.get("trellis2_seed", 1))
    num_samples = int(p25_cfg.get("trellis2_num_samples", 1))
    decimation_target = int(p25_cfg.get("trellis2_decimation_target", 1_000_000))
    texture_size = int(p25_cfg.get("trellis2_texture_size", 4096))
    aabb = p25_cfg.get("trellis2_aabb",
                       [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

    if mesh_obj is not None:
        meshes = [mesh_obj]
    else:
        meshes = pipeline.run(
            image,
            pipeline_type=pipeline_type,
            seed=seed,
            num_samples=num_samples,
        )
    if not meshes:
        raise RuntimeError("trellis2 returned no meshes")
    mesh = meshes[0]
    # nvdiffrast vertex-count cap.
    mesh.simplify(16_777_216)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=aabb,
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )
    if reframe_mesh_npz is not None:
        try:
            m = _partverse_reframe_matrix(reframe_mesh_npz)
            glb.apply_transform(m)
            logger.info("[s5] reframed GLB → partverse frame "
                        "(diag=%s)", [round(float(x), 4) for x in m.diagonal()[:3]])
        except Exception as e:
            logger.warning("[s5] partverse reframe failed (%s); exporting "
                           "in TRELLIS frame", e)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    webp = bool(p25_cfg.get("trellis2_extension_webp", True))
    try:
        glb.export(str(out_path), extension_webp=webp)
    except Exception as e:
        # Some Pillow builds lack webp (no HAVE_WEBPANIM) — fall back to PNG
        # textures so the GLB still writes.
        if webp:
            logger.warning("[s5] webp texture export failed (%s); "
                           "retrying with PNG textures", e)
            glb.export(str(out_path), extension_webp=False)
        else:
            raise
    logger.info("[s5] wrote %s", out_path)


def _load_view_image(spec, obj_record, p25_cfg: dict):
    """Best-effort load of the original (un-edited) input image.

    Falls back to None if the view cannot be resolved; the caller may
    then skip the "before" export.
    """
    from PIL import Image  # noqa: F401  (used downstream via scripts.run_2d_edit)
    try:
        from scripts.run_2d_edit import prepare_input_image
        if hasattr(spec, "npz_view") and spec.npz_view >= 0:
            _, pil = prepare_input_image(obj_record, spec.npz_view)
            return pil.convert("RGB")
    except Exception:
        pass
    return None


# ─────────────────── P4: masked sampling with P1 anchor ─────────────

def _load_p1_slat(ctx: ObjectContext):
    """Load this object's P1 shape SLat → (feats Tensor [N,32], coords [N,3])."""
    p1_path = ctx.dir / "p1_encode" / "shape_slat.npz"
    if not p1_path.is_file():
        raise FileNotFoundError(
            f"P4 requires P1 encode output; missing: {p1_path}")
    d = __import__("numpy").load(p1_path)
    import torch as _t
    return (_t.from_numpy(d["feats"]).float(),
            _t.from_numpy(d["coords"]).int())


# ─────────────────── latent retention (SS / SLat / o-voxel) ──────────
# The masked edit produces three intermediate latents that the old path threw
# away after ``decode_latent``: the SS-edited occupancy (coords_new), the masked
# shape SLat, and the masked tex SLat.  Persist them per edit so the structure
# change is auditable and the result can be re-decoded / re-rendered (incl. with
# TRELLIS.2's native renderers) without re-running the GPU edit.

def _sparse_to_np(st):
    """SparseTensor → (feats float32 [N,C], coords3 int16 [N,3]) numpy."""
    if st is None:
        return None, None
    import torch as _t
    f = st.feats.detach().cpu().float().numpy()
    c = st.coords
    c3 = c[:, 1:] if c.shape[1] == 4 else c
    return f, c3.detach().cpu().to(_t.int16).numpy()


def _collect_edit_latents(*, edit_type, target_part_ids, coords0, coords_new,
                          shape_new, tex_new, edit_grid, keep16,
                          s1_pad, s1_thresh):
    """Gather the per-edit latents into plain numpy (for ``_save_edit_latents``)."""
    import numpy as _np
    import torch as _t

    def _c3(t):
        return None if t is None else t.detach().cpu().to(_t.int16).numpy()

    sf, sc = _sparse_to_np(shape_new)
    tf, tc = _sparse_to_np(tex_new)
    eg = (_t.nonzero(edit_grid).detach().cpu().to(_t.int16).numpy()
          if edit_grid is not None else None)
    return {
        "edit_type": edit_type,
        "parts": _np.asarray(list(target_part_ids), dtype=_np.int32),
        "coords0": _c3(coords0),
        "coords_new": _c3(coords_new),
        "shape_feats": sf, "shape_coords": sc,
        "tex_feats": tf, "tex_coords": tc,
        "edit_grid": eg,
        "keep16": (keep16.detach().cpu().numpy() if keep16 is not None else None),
        "s1_pad": s1_pad, "s1_thresh": s1_thresh,
    }


def _save_edit_latents(latents: dict, edit_dir: Path, logger):
    """Write the retained latents under ``edit_dir/latents/`` (ss/shape/tex npz)."""
    import numpy as _np
    out = edit_dir / "latents"
    out.mkdir(parents=True, exist_ok=True)

    ss = {
        "coords0": latents["coords0"],
        "coords_new": latents["coords_new"],
        "edit_type": _np.asarray(latents["edit_type"] or ""),
        "parts": latents["parts"],
    }
    if latents["edit_grid"] is not None:
        ss["edit_grid"] = latents["edit_grid"]
    if latents["keep16"] is not None:
        ss["keep16"] = latents["keep16"]
    if latents["s1_pad"] is not None:
        ss["s1_pad"] = _np.int32(latents["s1_pad"])
    if latents["s1_thresh"] is not None:
        ss["s1_thresh"] = _np.float32(latents["s1_thresh"])
    _np.savez_compressed(out / "ss.npz", **ss)

    if latents["shape_feats"] is not None:
        _np.savez_compressed(out / "shape_slat.npz",
                             feats=latents["shape_feats"],
                             coords=latents["shape_coords"])
    if latents["tex_feats"] is not None:
        _np.savez_compressed(out / "tex_slat.npz",
                             feats=latents["tex_feats"],
                             coords=latents["tex_coords"])
    logger.info(
        "[s5/P4] saved latents → %s (ss_new=%s shape=%s tex=%s)", out,
        None if latents["coords_new"] is None else latents["coords_new"].shape,
        None if latents["shape_feats"] is None else latents["shape_feats"].shape,
        None if latents["tex_feats"] is None else latents["tex_feats"].shape)


def _build_p4_mesh(pipeline, spec, edited_img, orig_img, p1_feats, p1_coords3,
                   mesh_npz_path, p25_cfg, logger, white_model=False):
    """Masked 3-layer edit → MeshWithVoxel (same shape as ``pipeline.run()[0]``).

    Implements the Vinedresser3D idea on TRELLIS.2's three latents, routed by
    edit type (matching v1's ``edit_types``):

      * **modification / scale** (``S1_S2_TYPES``) — *structure changes*. Edit
        the SS latent so the part can grow / shrink voxels (``coords0`` →
        ``coords_new``), then masked geometry + masked material on
        ``coords_new`` with the preserved region anchored to the ORIGINAL-image
        inversion via a coord bridge.
      * **material / color / global** (``S2_ONLY_TYPES``) — geometry locked.
        Reuse the original shape latent untouched (no resample → zero drift),
        coords fixed, and masked material only (global = whole-object material).

    All inversions use the ORIGINAL view image (``orig_img``); all forward
    passes use the EDITED image (``edited_img``).

    ``white_model=True`` skips the texture stage and emits a flat grey PBR.
    """
    import torch
    from partcraft.edit_types import S1_S2_TYPES, S2_ONLY_TYPES
    from partcraft.pipeline_v3.trellis2_masked_sampler import (
        MaskedFlowEulerGuidanceIntervalSampler,
    )
    from partcraft.pipeline_v3.trellis2_part_mask import (
        part_edit_grid_64, edit_grid_64_to_keep16, densify_edit_occupancy)
    from partcraft.pipeline_v3 import trellis2_structure as t2s
    from partcraft.pipeline_v3 import trellis2_edit_stages as t2e

    dev = "cuda"
    sampler = MaskedFlowEulerGuidanceIntervalSampler(1e-5)
    edit_type = (getattr(spec, "edit_type", "") or "").lower()
    target_part_ids = list(getattr(spec, "selected_part_ids", []) or [])
    coords0 = p1_coords3.int()
    # Latent-retention bookkeeping (saved by run_for_object when
    # trellis2_save_latents is on).  edit_grid is set in both branches;
    # keep16 / s1_* only exist on the S1 (structure-edit) path.
    edit_grid = keep16 = None
    s1_pad = s1_thresh = None
    # canonical Z-up encode/mask frame (must match the P1 encode's flag)
    canonical = bool(p25_cfg.get("trellis2_canonical_frame", False))

    # ── conditioning: original (inversion) vs edited (forward) ────────
    orig_proc = pipeline.preprocess_image(orig_img)
    edit_proc = pipeline.preprocess_image(edited_img)
    cond_orig_1024 = pipeline.get_cond([orig_proc], 1024)
    cond_edit_1024 = pipeline.get_cond([edit_proc], 1024)

    # original shape latent (raw/denormalized) — reused as geometry reference
    shape0 = t2e.sparse_denorm_shape(pipeline, p1_feats, coords0)

    if edit_type in S1_S2_TYPES and target_part_ids:
        # ── S1: edit structure → coords_new ───────────────────────────
        # Edit-region tightness knobs (the S1 SS latent is only 16³, so an
        # over-dilated / low-threshold region engulfs the whole part zone and
        # the structure repaint returns a coarse blob — see maskviz diag).
        s1_pad = int(p25_cfg.get("trellis2_s1_pad", 3))
        s1_thresh = float(p25_cfg.get("trellis2_s1_keep_thresh", 0.1))
        edit_grid = part_edit_grid_64(mesh_npz_path, target_part_ids,
                                      pad=s1_pad, canonical=canonical).to(dev)
        cond_orig_512 = pipeline.get_cond([orig_proc], 512)
        cond_edit_512 = pipeline.get_cond([edit_proc], 512)
        ss_enc = t2s.get_ss_encoder(pipeline, p25_cfg, logger)
        coords_new = t2s.edit_structure(
            pipeline, ss_enc, sampler, coords0, edit_grid,
            cond_orig_512, cond_edit_512, logger,
            keep_thresh=s1_thresh,
            soft_feather=float(p25_cfg.get("trellis2_s1_soft_feather", 0.0))
            ).to(dev)
        logger.info("[s5/P4] %s S1 structure: %d → %d voxels (parts=%s)",
                    spec.edit_id, coords0.shape[0], coords_new.shape[0],
                    target_part_ids)
        keep16 = edit_grid_64_to_keep16(edit_grid, thresh=s1_thresh)
        # ── densify the edited region so flexicubes can CLOSE the surface ─
        # (a grown part's S1 shell is ~1-voxel thin → holey/see-through decode;
        # thicken only inside the edit region so the body stays untouched).
        s1_densify = int(p25_cfg.get("trellis2_s1_densify", 0))
        if s1_densify > 0:
            n_before = coords_new.shape[0]
            coords_new = densify_edit_occupancy(
                coords_new, edit_grid, iters=s1_densify).to(dev)
            logger.info("[s5/P4] %s S1 densify(iters=%d): %d → %d voxels",
                        spec.edit_id, s1_densify, n_before, coords_new.shape[0])
        # ── S2 geometry on coords_new ─────────────────────────────────
        shape_new = t2e.masked_shape_slat(
            pipeline, sampler, p1_feats, coords0, coords_new, edit_grid,
            cond_orig_1024, cond_edit_1024, logger,
            warmstart=bool(p25_cfg.get("trellis2_s2_warmstart", False)),
            nn_init=bool(p25_cfg.get("trellis2_s2_nn_init", False)),
            anchor_mode=str(p25_cfg.get("trellis2_s2_anchor_mode", "perstep")),
            anchor_cutoff=float(p25_cfg.get("trellis2_s2_anchor_cutoff", 0.3)))
    else:
        # ── material / color / global: geometry preserved, coords fixed ──
        coords_new = coords0
        if edit_type == "global":
            edit_grid = torch.ones(64, 64, 64, dtype=torch.bool, device=dev)
        elif target_part_ids:
            edit_grid = part_edit_grid_64(mesh_npz_path, target_part_ids,
                                          pad=0, canonical=canonical).to(dev)
        else:
            edit_grid = torch.zeros(64, 64, 64, dtype=torch.bool, device=dev)
        shape_new = shape0   # reuse original geometry verbatim
        logger.info("[s5/P4] %s S2-only (%s): geometry locked, %d voxels",
                    spec.edit_id, edit_type, coords0.shape[0])

    # ── material + decode ─────────────────────────────────────────────
    tex_new = None
    if white_model:
        from partcraft.pipeline_v3.trellis2_white import build_white_model_mesh
        logger.info("[s5/P4] %s white-model — skipping texture stage",
                    spec.edit_id)
        mesh = build_white_model_mesh(pipeline, shape_new, logger)
    else:
        tex_new = t2e.masked_tex_slat(
            pipeline, sampler, shape0, shape_new, coords0, coords_new,
            edit_grid, cond_orig_1024, cond_edit_1024, logger,
            anchor_mode=str(p25_cfg.get("trellis2_s2_anchor_mode", "perstep")))
        torch.cuda.empty_cache()
        meshes = pipeline.decode_latent(shape_new, tex_new, 1024)
        mesh = meshes[0]
    mesh.simplify(16_777_216)

    # ── retain intermediate latents (SS occupancy + shape/tex SLat) ───
    latents = _collect_edit_latents(
        edit_type=edit_type, target_part_ids=target_part_ids,
        coords0=coords0, coords_new=coords_new,
        shape_new=shape_new, tex_new=tex_new,
        edit_grid=edit_grid, keep16=keep16,
        s1_pad=s1_pad, s1_thresh=s1_thresh)
    return mesh, latents


# ─────────────────── per-object processing ───────────────────────────

def run_for_object(
    ctx: ObjectContext,
    *,
    pipeline,
    dataset,
    p25_cfg: dict,
    seed: int = 1,
    debug: bool = False,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> Trellis2Result:
    log = logger or logging.getLogger("pipeline_v3.trellis2")
    res = Trellis2Result(obj_id=ctx.obj_id)

    all_specs: list[EditSpec] = []
    pending: list[EditSpec] = []
    for spec in iter_flux_specs(ctx):
        all_specs.append(spec)
        if prereq_map is not None:
            if not edit_needs_step(ctx, spec.edit_id, "s5", prereq_map, force=force):
                res.n_skip += 1
                continue
        else:
            if is_gate_a_failed(ctx, spec.edit_id):
                res.n_skip += 1
                continue
            after = ctx.edit_3d_dir(spec.edit_id) / "after.glb"
            before = ctx.edit_3d_dir(spec.edit_id) / "before.glb"
            if after.is_file() and before.is_file() and not force:
                res.n_skip += 1
                continue
        pending.append(spec)

    if not all_specs:
        update_step(ctx, "s5_trellis2", status=STATUS_OK, n=0, reason="no_specs")
        return res
    if not pending:
        update_step(ctx, "s5_trellis2", status=STATUS_OK,
                    n=res.n_skip, n_skip=res.n_skip)
        return res

    try:
        obj_record = dataset.load_object(ctx.shard, ctx.obj_id)
    except Exception as e:
        log.error("[s5] %s load_object failed: %s", ctx.obj_id, e)
        update_step(ctx, "s5_trellis2", status=STATUS_FAIL,
                    error=f"load_object: {e}")
        res.error = str(e); res.n_fail = len(pending); return res

    emit_before = bool(p25_cfg.get("emit_before", True))
    # After.glb is exported in TRELLIS's canonical frame (Y↔Z swapped, ×0.5,
    # object tipped onto its end).  Reframe it back onto the original full.glb
    # world frame so the dataset/condition/preview all share one frame.
    reframe_on = bool(p25_cfg.get("trellis2_export_partverse_frame", True))
    reframe_npz = ctx.mesh_npz if (reframe_on and ctx.mesh_npz is not None) else None
    # Masked latent editing (Vinedresser3D-style: preserve outside the part,
    # re-flow inside) is the default.  Set use_mask=False (legacy key: use_p4)
    # to fall back to naive full re-generation from the edited 2D image.
    use_p4 = bool(p25_cfg.get("use_mask", p25_cfg.get("use_p4", True)))
    edits_2d_subdir = p25_cfg.get("edits_2d_subdir", "edits_2d")
    t0 = time.time()

    from PIL import Image

    p1_feats = p1_coords3 = None
    white_model = False
    if use_p4:
        p1_feats, p1_coords3 = _load_p1_slat(ctx)
        from partcraft.pipeline_v3.trellis2_white import read_white_model_flag
        white_model = read_white_model_flag(ctx)
        log.info("[s5/P4] %s loaded P1 SLat (%d tokens)  white_model=%s",
                 ctx.obj_id, int(p1_coords3.shape[0]), white_model)

    for spec in pending:
        try:
            edited_path = ctx.dir / edits_2d_subdir / f"{spec.edit_id}_edited.png"
            if not edited_path.exists():
                raise FileNotFoundError(f"missing edited 2D image: {edited_path}")
            edited_img = Image.open(edited_path).convert("RGB")

            pair_dir = ctx.edit_3d_dir(spec.edit_id)
            after_glb = pair_dir / "after.glb"
            if use_p4:
                # The original (pre-edit) view drives the RF inversion that
                # anchors the preserved region; the edited view drives the
                # forward edit.  flux_2d wrote both as *_input/_edited.png.
                input_path = (ctx.dir / edits_2d_subdir /
                              f"{spec.edit_id}_input.png")
                if not input_path.exists():
                    raise FileNotFoundError(
                        f"masked edit needs the original view: {input_path}")
                orig_img = Image.open(input_path).convert("RGB")
                mesh_obj, latents = _build_p4_mesh(
                    pipeline, spec, edited_img, orig_img,
                    p1_feats, p1_coords3,
                    ctx.mesh_npz, p25_cfg, log,
                    white_model=white_model,
                )
                if bool(p25_cfg.get("trellis2_save_latents", True)):
                    try:
                        _save_edit_latents(latents, pair_dir, log)
                    except Exception as e:
                        log.warning("[s5] %s latent save failed: %s",
                                    spec.edit_id, e)
                _run_and_export(pipeline, edited_img, after_glb, p25_cfg, log,
                                mesh_obj=mesh_obj, reframe_mesh_npz=reframe_npz)
            else:
                _run_and_export(pipeline, edited_img, after_glb, p25_cfg, log,
                                reframe_mesh_npz=reframe_npz)

            if emit_before and not use_p4:
                before_img = _load_view_image(spec, obj_record, p25_cfg)
                if before_img is None:
                    input_path = ctx.dir / edits_2d_subdir / f"{spec.edit_id}_input.png"
                    if input_path.exists():
                        before_img = Image.open(input_path).convert("RGB")
                if before_img is not None:
                    before_glb = pair_dir / "before.glb"
                    _run_and_export(pipeline, before_img, before_glb, p25_cfg, log,
                                    reframe_mesh_npz=reframe_npz)
                else:
                    log.warning("[s5] %s no input image for before.glb (skip)",
                                spec.edit_id)

            res.n_ok += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5",
                              status="done")
            log.info("[s5] %s ok", spec.edit_id)
        except Exception as e:
            log.error("[s5] %s failed: %s", spec.edit_id, e)
            res.n_fail += 1
            update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5",
                              status="error", reason=str(e)[:200])

    obj_record.close()
    update_step(
        ctx, "s5_trellis2",
        status=STATUS_OK if res.n_fail == 0 else STATUS_FAIL,
        n_ok=res.n_ok, n_fail=res.n_fail, n_skip=res.n_skip,
        wall_s=round(time.time() - t0, 2),
    )
    return res


# ─────────────────── batch entrypoint (single GPU) ───────────────────

def run(
    ctxs: Iterable[ObjectContext],
    *,
    cfg: dict,
    images_root: Path,
    mesh_root: Path,
    shard: str = "01",
    seed: int = 1,
    debug: bool = False,
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> list[Trellis2Result]:
    log = logger or logging.getLogger("pipeline_v3.trellis2")
    log.info("[s5] CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Reuse the same services_cfg flattener as v1; YAML keys for trellis2
    # live under the same services.image_edit / pipeline.* section unless
    # you split them. Extra v2-specific keys are read with defaults inside
    # _ensure_pipeline / _run_and_export.
    p25_cfg = psvc.trellis_image_edit_flat(cfg)

    from partcraft.io.hy3d_loader import HY3DPartDataset
    dataset = HY3DPartDataset(str(images_root), str(mesh_root), [shard])

    pipeline = _ensure_pipeline(p25_cfg, log)

    results: list[Trellis2Result] = []
    for ctx in list(ctxs):
        edit_ids = [sp.edit_id for sp in iter_flux_specs(ctx)]
        if edit_ids and not force and not obj_needs_stage(
            ctx, edit_ids, "s5", prereq_map, force=force
        ):
            results.append(Trellis2Result(ctx.obj_id))
            continue
        results.append(run_for_object(
            ctx, pipeline=pipeline, dataset=dataset, p25_cfg=p25_cfg,
            seed=seed, debug=debug,
            prereq_map=prereq_map, force=force, logger=log,
        ))
    return results


__all__ = ["GPU_TYPES", "Trellis2Result", "run_for_object", "run"]
