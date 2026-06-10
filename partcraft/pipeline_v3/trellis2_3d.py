"""Step s5 (trellis2 variant) — image-driven 3D regeneration via TRELLIS.2.

Parallel to ``trellis_3d.py`` but uses the TRELLIS.2 cascade. The "edit" is
masked latent editing (Vinedresser3D-style: RF-invert the original view to
anchor the preserved region, re-flow only the edited part), implemented in
:func:`_build_p4_mesh`:

    edits_2d/{edit_id}_{input,edited}.png ──► _build_p4_mesh ──► mesh ──► after.glb

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
from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP
from .qc_io import is_gate_a_failed
from .edit_status_io import edit_needs_step, update_edit_stage, obj_needs_stage


# v2 has no in-SLAT "edit type". All edit_types are handled the same way:
# regenerate from the edited 2D image. We keep the same GPU_TYPES set so
# the stage gating logic upstream still selects the same edits.
GPU_TYPES = frozenset({"modification", "scale", "material", "global"})


_WEBP_OK: bool | None = None   # cached once-per-process webp probe


def _webp_textures_work() -> bool:
    """Probe whether this env's Pillow can actually encode WEBP.

    The trellis2 env ships a tangled Pillow (pip metadata 12.2.0 but on-disk
    PIL 9.5.0.post2 with a mismatched ``_webp`` C ext — ``WebPEncode`` takes 9
    args while the Python layer passes 11), so every ``glb.export(..., webp)``
    throws and falls back to PNG.  Probe ONCE in-memory and cache it so we skip
    the doomed webp attempt on every edit; auto-recovers if Pillow is fixed.
    """
    global _WEBP_OK
    if _WEBP_OK is None:
        try:
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buf, "WEBP", quality=80)
            _WEBP_OK = True
        except Exception:
            _WEBP_OK = False
    return _WEBP_OK


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

    # Residency vs low-VRAM offload.  With low_vram=True (the upstream default)
    # EVERY flow/decoder is shuttled CPU→GPU before each sample and GPU→CPU
    # after — repeated per object, per stage (SS/shape/shape_lr/tex + decoders
    # + DINOv3).  On these 144 GB cards there is ample room to keep the whole
    # ~4B pipeline resident, eliminating all that PCIe churn.  Both the vendored
    # sampler AND our edit stages honour pipeline.low_vram, so flipping it off
    # is consistent end-to-end.  Override with services.image_edit.low_vram:true
    # on memory-tight machines.
    low_vram = bool(p25_cfg.get("low_vram", False))
    pipeline.low_vram = low_vram
    pipeline.cuda()  # low_vram=False → moves all models onto GPU, resident
    logger.info("[s5] TRELLIS.2 pipeline ready  (low_vram=%s → %s)",
                low_vram, "offload per-stage" if low_vram else "all models resident")
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


def _export_edit_glb(mesh_obj, out_path: Path, p25_cfg: dict, logger,
                     *, reframe_mesh_npz: Path | None = None):
    """Export a pre-built masked-edit mesh to GLB at ``out_path``.

    The mesh is produced by :func:`_build_p4_mesh` (masked latent editing); this
    helper only bakes + writes it.  If ``reframe_mesh_npz`` is given, the GLB is
    rigidly transformed from TRELLIS's canonical export frame back into the
    original full.glb (partverse) world frame via
    :func:`_partverse_reframe_matrix`, so the output shares one coordinate frame
    with the 2D condition and the edit mask.
    """
    import o_voxel  # type: ignore

    decimation_target = int(p25_cfg.get("trellis2_decimation_target", 1_000_000))
    texture_size = int(p25_cfg.get("trellis2_texture_size", 4096))
    aabb = p25_cfg.get("trellis2_aabb",
                       [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

    mesh = mesh_obj
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
    # This env's Pillow webp encoder is broken (see _webp_textures_work); probe
    # once and skip straight to PNG so we don't throw + retry on every edit.
    if webp:
        first_probe = _WEBP_OK is None
        if not _webp_textures_work():
            if first_probe:
                logger.info("[s5] webp unavailable in this Pillow build → "
                            "exporting GLB with PNG textures")
            webp = False
    try:
        glb.export(str(out_path), extension_webp=webp)
    except Exception as e:
        # Fallback guard in case webp passes the probe but still fails at export.
        if webp:
            logger.warning("[s5] webp texture export failed (%s); "
                           "retrying with PNG textures", e)
            glb.export(str(out_path), extension_webp=False)
        else:
            raise
    logger.info("[s5] wrote %s", out_path)


# ─────────────────── P4: masked sampling with P1 anchor ─────────────

def _load_p1_slat(ctx: ObjectContext, res: int = 1024, which: str = "shape"):
    """Load this object's P1 ``{which}`` SLat → (feats Tensor [N,C], coords [N,3]).

    ``which`` is ``"shape"`` or ``"tex"``.  ``res != 1024`` loads the grid-``res``
    sidecar ``{which}_slat_e{res}.npz`` (re-encoded at grid ``res`` → ``res//16``³
    coords) that the ``_{res}`` SLat flow models consume; ``1024`` loads the
    canonical 64³ ``{which}_slat.npz``.  Shape and tex share coords (same order),
    so a tex latent loaded this way aligns with the matching shape ``coords``.
    """
    stem = f"{which}_slat"
    fname = f"{stem}.npz" if res == 1024 else f"{stem}_e{res}.npz"
    p1_path = ctx.dir / "p1_encode" / fname
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


def _ss_latent_from_occ64(ss_enc, coords_new, dev):
    """``ss_enc`` of the 64³ occupancy defined by ``[N,3]`` (or ``[N,4]``) int voxel
    coords → np float32 ``[1,8,16,16,16]``.

    BYTE-COMPATIBLE with the P1 ``ss.npz`` before-latent
    (``trellis2_encode.encode_shape_tex_ss``): zero ``[1,1,64³]`` occupancy, set
    active voxels to 1, run the SAME ``ss_enc`` (default
    ``ss_enc_conv3d_16l8_fp16``).  Captured from the 64³ ``coords_new`` BEFORE the
    ``_to_s2`` downsample so the after-latent describes the true 64³ structure —
    the training target for the S1 (SS structure) stage."""
    import torch as _t
    import numpy as _np
    c = coords_new.long()
    if c.shape[1] == 4:
        c = c[:, 1:]
    c = c.to(dev)
    occ = _t.zeros(1, 1, 64, 64, 64, device=dev)
    occ[0, 0, c[:, 0], c[:, 1], c[:, 2]] = 1.0
    with _t.no_grad():
        z = ss_enc(occ.float())
    return z.detach().cpu().numpy().astype(_np.float32)


def _collect_edit_latents(*, edit_type, target_part_ids, coords0, coords_new,
                          shape_new, tex_new, edit_grid, keep16,
                          s1_pad, s1_thresh, ss_latent=None):
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
        # after SS latent (dense [1,8,16,16,16]) — S1 training target; None on
        # S2-only edit types (structure unchanged → after == before).
        "ss_latent": ss_latent,
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
    # after SS latent (dense [1,8,16,16,16]) — the S1 stage's training target,
    # ss_enc of the 64³ edited occupancy.  Named ``ss_latent.npz`` to disambiguate
    # from ``ss.npz`` (the coords/region pack).  Byte-compatible with the P1
    # before-latent ``p1_encode/ss.npz``.
    if latents.get("ss_latent") is not None:
        _np.savez_compressed(out / "ss_latent.npz", ss=latents["ss_latent"])
    logger.info(
        "[s5/P4] saved latents → %s (ss_new=%s shape=%s tex=%s)", out,
        None if latents["coords_new"] is None else latents["coords_new"].shape,
        None if latents["shape_feats"] is None else latents["shape_feats"].shape,
        None if latents["tex_feats"] is None else latents["tex_feats"].shape)


def _build_p4_mesh(pipeline, spec, edited_img, orig_img, p1_feats, p1_coords3,
                   mesh_npz_path, p25_cfg, logger, white_model=False,
                   p1_feats_s2=None, p1_coords_s2=None, p1_tex_s2=None,
                   edit_res=1024, ss_latent_only=False):
    """Masked 3-layer edit → MeshWithVoxel (same shape as ``pipeline.run()[0]``).

    ``edit_res`` (1024 default / 512) sets the S2 SLat resolution.  S1 (the SS
    structure stage) ALWAYS runs at 64³ — the TRELLIS.1 SS VAE is fixed 64³→16³,
    and the native 512 pipeline just max-pools the decoded occupancy 64³→32³.  So
    for ``edit_res=512`` we keep S1 on the 64³ body (``p1_feats``/``p1_coords3``),
    then downsample ``coords_new``/``edit_grid`` 64³→32³ (``slat_grid=res//16``)
    and feed S2 the grid-512 re-encoded body (``p1_feats_s2``/``p1_coords_s2``,
    32³).  For 1024 the s2 body == the 64³ body and nothing is downsampled.

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

    from partcraft.pipeline_v3.trellis2_part_mask import (
        downsample_coords, downsample_edit_grid, restore_preserved_occupancy)

    dev = "cuda"
    sampler = MaskedFlowEulerGuidanceIntervalSampler(1e-5)
    edit_type = (getattr(spec, "edit_type", "") or "").lower()
    target_part_ids = list(getattr(spec, "selected_part_ids", []) or [])
    if ss_latent_only and (edit_type not in S1_S2_TYPES or not target_part_ids):
        # S2-only edit types (material/color/global) don't change structure →
        # the after SS latent equals the before; nothing to re-encode here.
        return None, {"ss_latent_only": True, "ss_latent": None}
    # S1 runs on the 64³ body; S2 on the grid-(edit_res) body (== 64³ for 1024).
    slat_grid = edit_res // 16
    factor = 64 // slat_grid
    if p1_feats_s2 is None:
        p1_feats_s2 = p1_feats
    if p1_coords_s2 is None:
        p1_coords_s2 = p1_coords3
    coords0_s1 = p1_coords3.int()        # 64³ — S1 occupancy / edit_grid / keep16
    coords0 = p1_coords_s2.int()         # slat_grid — S2 body anchor

    def _to_s2(cn64, eg64):
        """64³ coords_new + edit_grid → slat_grid (no-op when edit_res==1024)."""
        if factor <= 1:
            return cn64.to(dev), eg64.to(dev)
        return (downsample_coords(cn64, factor).to(dev),
                downsample_edit_grid(eg64, factor).to(dev))
    # Latent-retention bookkeeping (saved by run_for_object when
    # trellis2_save_latents is on).  edit_grid is set in both branches;
    # keep16 / s1_* only exist on the S1 (structure-edit) path.
    edit_grid = keep16 = None
    s1_pad = s1_thresh = None
    ss_after_latent = None   # after SS latent (dense [1,8,16,16,16]); set on the S1 path
    # canonical Z-up encode/mask frame (must match the P1 encode's flag)
    canonical = bool(p25_cfg.get("trellis2_canonical_frame", False))

    # ── v1-faithful contact-aware soft-mask path (one master switch) ──────
    # When ``trellis2_s1_contact_soft`` is on we reproduce the TRELLIS.1
    # interweave recipe: contact-aware distance-transform soft masks at S1 & S2
    # with a dynamic sigma, preserved-part subtraction, and small-component
    # cleanup.  Off → legacy hard/box-blur behaviour (unchanged).
    contact_soft = bool(p25_cfg.get("trellis2_s1_contact_soft", False))
    contact_64 = None
    s2_sigma_eff = None
    # default S2 anchor from config; the structure path overrides to
    # "contact_soft" when the master switch is on (S2-only edits keep config).
    s2_anchor = str(p25_cfg.get("trellis2_s2_anchor_mode", "perstep"))
    # Texture can use a different anchor than shape (e.g. shape=perstep for a
    # solid edited part, tex=posthoc to hard-paste the original body material so
    # the preserved region keeps its exact colour).  Defaults to the shape mode.
    s2_tex_anchor = str(p25_cfg.get("trellis2_s2_tex_anchor_mode", s2_anchor))

    # ── conditioning: original (inversion) vs edited (forward) ────────
    orig_proc = pipeline.preprocess_image(orig_img)
    edit_proc = pipeline.preprocess_image(edited_img)
    # S2 conds at the edit resolution (512 conds drive the _512 SLat flow models).
    cond_orig_s2 = pipeline.get_cond([orig_proc], edit_res)
    cond_edit_s2 = pipeline.get_cond([edit_proc], edit_res)

    # original shape latent (raw/denormalized) — reused as geometry reference
    # (on the S2 grid: 64³ for 1024, 32³ for 512).
    shape0 = t2e.sparse_denorm_shape(pipeline, p1_feats_s2, coords0)

    # ── TRELLIS.1-SS bridge (optional) ────────────────────────────────────
    # Externally-generated TRELLIS.1 occupancy (precomputed offline in the
    # vinedresser3d env, TRELLIS.1 SS flow) injected as ``coords_new``.  When set,
    # it REPLACES the S1 sampler inside the main masked path below but keeps the
    # SAME recipe (real edit_grid + same-frame restore + configured S2 anchor +
    # edit_res), so it's a fair T1-vs-T2 SS-flow A/B (the only variable is the
    # flow; SS VAE / edit region / restore / S2 are identical).
    ss1_dir = p25_cfg.get("trellis2_ss1_coords_dir", None)
    ss1_npz = None
    if ss1_dir:
        cand = Path(ss1_dir) / spec.obj_id / spec.edit_id / "ss1_coords.npz"
        if cand.is_file():
            ss1_npz = cand

    if p25_cfg.get("trellis2_ss_vanilla", False) and edit_type in S1_S2_TYPES:
        # Exp 2: TRELLIS.2's OWN SS flow in VANILLA whole-object mode (no mask),
        # then free S2 — the control that isolates "vanilla vs masked" (mechanism)
        # from "TRELLIS.1 vs TRELLIS.2" (model).  SS uses the 512-res edited cond.
        cond_edit_512 = pipeline.get_cond([edit_proc], 512)
        cn = pipeline.sample_sparse_structure(
            cond_edit_512, 64, num_samples=1)
        coords_new = (cn[:, 1:] if cn.shape[1] == 4 else cn).int().to(dev)
        edit_grid = torch.ones(64, 64, 64, dtype=torch.bool, device=dev)
        s2_anchor = s2_tex_anchor = "free"   # whole-object vanilla: free shape AND tex
        coords_new, edit_grid = _to_s2(coords_new, edit_grid)
        logger.info("[s5/P4] %s TRELLIS.2-vanilla-SS: %d voxels → free S2",
                    spec.edit_id, coords_new.shape[0])
        shape_new = t2e.masked_shape_slat(
            pipeline, sampler, p1_feats_s2, coords0, coords_new, edit_grid,
            cond_orig_s2, cond_edit_s2, logger,
            warmstart=False, nn_init=False, anchor_mode="free", res=edit_res)
    elif edit_type in S1_S2_TYPES and target_part_ids:
        # ── S1: edit structure → coords_new ───────────────────────────
        # Edit-region tightness knobs (the S1 SS latent is only 16³, so an
        # over-dilated / low-threshold region engulfs the whole part zone and
        # the structure repaint returns a coarse blob — see maskviz diag).
        # Default pad=0: edit region == the raw part-id mask, NO dilation.
        # Set trellis2_s1_pad>0 to re-enable Chebyshev box growth (v1 used 3).
        s1_pad = int(p25_cfg.get("trellis2_s1_pad", 0))
        s1_thresh = float(p25_cfg.get("trellis2_s1_keep_thresh", 0.1))
        restore_pres = bool(p25_cfg.get("trellis2_s2_restore_preserved", False))
        coords_orig_ss = None  # same-frame pre-edit SS occupancy (masked path only)
        sub_pres = bool(p25_cfg.get("trellis2_mask_subtract_preserved",
                                    contact_soft))
        edit_grid = part_edit_grid_64(
            mesh_npz_path, target_part_ids, pad=s1_pad, canonical=canonical,
            subtract_preserved=sub_pres).to(dev)
        # contact analysis (shared by S1 + S2) — dynamic blend sigma from how
        # much the edit surface touches preserved geometry (v1 recipe).
        s1_sigma_eff = None
        if contact_soft:
            from partcraft.pipeline_v3.trellis2_contact_mask import (
                compute_contact_boundary)
            contact_64, c_ratio, s1_dyn, s2_dyn = compute_contact_boundary(
                edit_grid, coords0_s1, dev)
            s1_cfg = p25_cfg.get("trellis2_s1_soft_sigma", None)
            s1_sigma_eff = float(s1_cfg) if s1_cfg is not None else s1_dyn
            # S2 deliberately does NOT use the contact soft mask.  A per-step
            # (even soft) anchor on TRELLIS.2's S2 makes the edit-region shell
            # holey/see-through (the void/破碎 regression) — exactly what posthoc
            # was built to avoid.  So contact-soft is now an S1-ONLY structure
            # trick; S2 keeps the configured anchor (default posthoc = free gen
            # + body paste), which stays the validated solid path.
            s2_sigma_eff = None
            logger.info("[s5/P4] %s contact ratio=%.2f → s1_sigma=%.2f "
                        "(S1 contact mask only; S2 anchor=%s)",
                        spec.edit_id, c_ratio, s1_sigma_eff, s2_anchor)
        cond_orig_512 = pipeline.get_cond([orig_proc], 512)
        cond_edit_512 = pipeline.get_cond([edit_proc], 512)
        ss_enc = t2s.get_ss_encoder(pipeline, p25_cfg, logger)
        # SS sampler override — benchmark T2's own masked SS against TRELLIS.1's
        # gentler schedule (the robustness win on large parts may be the sampler,
        # not the model → would remove the cross-env bridge entirely).
        ss_override = {}
        if p25_cfg.get("trellis2_ss_align_t1", False):
            ss_override = {"steps": 25, "guidance_strength": 5.0,
                           "guidance_interval": [0.5, 1.0], "rescale_t": 3.0,
                           "guidance_rescale": 0.0}
        if p25_cfg.get("trellis2_ss_steps"):
            ss_override["steps"] = int(p25_cfg["trellis2_ss_steps"])
        if p25_cfg.get("trellis2_ss_cfg") is not None:
            ss_override["guidance_strength"] = float(p25_cfg["trellis2_ss_cfg"])
        # S1 mode: "masked" (inversion + keep-mask repaint, default) or
        # "flowedit" (source/target velocity-difference ODE, no inversion / no
        # SS keep mask — the edit region emerges from the conditioning change).
        # edit_grid is still computed above and still drives the S2 coord bridge;
        # only the S1 *latent* edit changes.
        s1_mode = str(p25_cfg.get("trellis2_s1_mode", "masked")).lower()
        # S1 SS *flow* model: "t2" (default, TRELLIS.2's own SS flow) or "t1"
        # (TRELLIS.1's SS flow + DINOv2 cond, in-process — the native replacement
        # for the old offline ss1_coords_dir bridge; see trellis1_ss.py).
        s1_ss_model = str(p25_cfg.get("trellis2_s1_ss_model", "t2")).lower()
        if ss1_npz is not None:
            # TRELLIS.1-SS bridge: external occupancy REPLACES the S1 sampler,
            # but the rest of the recipe (edit_grid above, restore below, S2
            # anchor, edit_res) is identical to the T2 masked run → fair A/B.
            import numpy as _np
            coords_new = torch.from_numpy(
                _np.load(ss1_npz, allow_pickle=True)["coords"].astype("int64")
            ).int().to(dev)
            logger.info("[s5/P4] %s TRELLIS.1-SS bridge: external %d voxels → "
                        "recipe S2 (anchor=%s restore=%s res=%d)", spec.edit_id,
                        coords_new.shape[0], s2_anchor, restore_pres, edit_res)
            if restore_pres:
                # same-frame reference is the SS-VAE roundtrip (flow-independent)
                coords_orig_ss = t2s.ss_roundtrip_occupancy(
                    pipeline, ss_enc, coords0_s1, dev).to(dev)
        elif s1_mode == "flowedit":
            # gs_src defaults to gs_tgt (SYMMETRIC CFG): the edit drive should
            # come from the source→target CONDITION difference only, not a CFG
            # strength gap.  Asymmetric gs injects a spurious (pos−neg) push that
            # is nonzero even under identical conditioning and shreds occupancy on
            # detailed structures (smoke: tank identity recall 0.15 @ gs 3/7.5 vs
            # 1.00 @ gs 7.5/7.5).  Override trellis2_s1_fe_gs_src to go asymmetric.
            gs_tgt = float(p25_cfg.get("trellis2_s1_fe_gs_tgt", 7.5))
            gs_src = float(p25_cfg.get("trellis2_s1_fe_gs_src", gs_tgt))
            coords_new = t2s.flowedit_structure(
                pipeline, ss_enc, sampler, coords0_s1,
                cond_orig_512, cond_edit_512, logger,
                gs_src=gs_src, gs_tgt=gs_tgt,
                n_avg=int(p25_cfg.get("trellis2_s1_fe_navg", 1)),
                seed=int(p25_cfg.get("trellis2_seed", 1)),
                ss_param_override=(ss_override or None),
                ).to(dev)
        else:
            # T1-native: swap in TRELLIS.1's SS flow + DINOv2 conds (the SS VAE /
            # masked machinery in edit_structure are shared).  T2 path unchanged.
            t1_flow = None
            c_orig_s1, c_edit_s1 = cond_orig_512, cond_edit_512
            if s1_ss_model == "t1":
                from partcraft.pipeline_v3 import trellis1_ss as t1ss
                t1_flow = t1ss.load_t1_ss_flow(pipeline, p25_cfg, logger)
                dino = t1ss.load_t1_dino(pipeline, logger)
                rsess = t1ss.get_rembg_session(pipeline, logger)
                c_orig_s1 = t1ss.t1_get_cond(dino, t1ss.t1_preprocess(orig_img, rsess), dev)
                c_edit_s1 = t1ss.t1_get_cond(dino, t1ss.t1_preprocess(edited_img, rsess), dev)
                if not ss_override:   # force T1's gentler schedule (== ss_align_t1)
                    ss_override = {"steps": 25, "guidance_strength": 5.0,
                                   "guidance_interval": [0.5, 1.0], "rescale_t": 3.0,
                                   "guidance_rescale": 0.0}
                logger.info("[s5/P4] %s S1 = TRELLIS.1 SS flow (in-process, native)",
                            spec.edit_id)
            _s1 = t2s.edit_structure(
                pipeline, ss_enc, sampler, coords0_s1, edit_grid,
                c_orig_s1, c_edit_s1, logger,
                keep_thresh=s1_thresh,
                soft_feather=float(p25_cfg.get("trellis2_s1_soft_feather", 0.0)),
                contact_mask=contact_64, contact_sigma=s1_sigma_eff,
                ss_param_override=(ss_override or None),
                return_orig_occ=restore_pres, ss_flow=t1_flow)
            if restore_pres:
                coords_new, coords_orig_ss = _s1
                coords_new = coords_new.to(dev)
                coords_orig_ss = coords_orig_ss.to(dev)
            else:
                coords_new = _s1.to(dev)
        logger.info("[s5/P4] %s S1 structure: %d → %d voxels (parts=%s)",
                    spec.edit_id, coords0_s1.shape[0], coords_new.shape[0],
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
        # ── drop floating specks (v1 remove_small_components) ─────────────
        # Applied to the FULL occupancy (safer than v1's edit-only split: can't
        # sever a small grown part still attached to the body) with v1's size.
        s2_remove_small = int(p25_cfg.get("trellis2_s2_remove_small", 0))
        if s2_remove_small > 0:
            from partcraft.pipeline_v3.trellis2_contact_mask import (
                remove_small_components)
            keep = remove_small_components(
                coords_new, min_size=s2_remove_small).to(coords_new.device)
            n_before = coords_new.shape[0]
            coords_new = coords_new[keep]
            logger.info("[s5/P4] %s remove_small(<%d vox): %d → %d voxels",
                        spec.edit_id, s2_remove_small, n_before,
                        coords_new.shape[0])
        # ── restore preserved body occupancy lost to S1 — SAME-FRAME (64³ SS) ──
        # Re-insert body voxels (outside the edit region) that the masked S1 edit
        # dropped, using the PRE-EDIT SS occupancy decoded by the SAME ss_dec as
        # the reference (coords_orig_ss) — NOT the shape-VAE sidecar (different
        # encoder → misaligned/floating paste).  Done at 64³ before _to_s2 so the
        # restored voxels are guaranteed adjacent to / aligned with coords_new.
        if restore_pres and coords_orig_ss is not None:
            n_before = coords_new.shape[0]
            coords_new, n_restored = restore_preserved_occupancy(
                coords_orig_ss, coords_new, edit_grid, grid=64)
            coords_new = coords_new.to(dev)
            logger.info("[s5/P4] %s S1 restore-preserved (64³ SS same-frame): "
                        "+%d voxels (%d → %d)", spec.edit_id, n_restored,
                        n_before, coords_new.shape[0])
        # after SS latent (S1 training target): ss_enc of the TRUE 64³ edited
        # occupancy, captured BEFORE the _to_s2 downsample.  Byte-compatible with
        # the P1 before-latent (p1_encode/ss.npz).  In ss_latent_only (backfill)
        # mode we stop here — S2 (shape/tex/decode/render) is skipped entirely.
        ss_after_latent = _ss_latent_from_occ64(
            t2s.get_ss_encoder(pipeline, p25_cfg, logger), coords_new, dev)
        if ss_latent_only:
            return None, {
                "ss_latent_only": True,
                "ss_latent": ss_after_latent,
                "coords_new_64": coords_new.detach().cpu().to(torch.int16).numpy(),
            }
        # ── S2 geometry on coords_new (downsampled 64³→slat_grid for 512) ──
        coords_new, edit_grid = _to_s2(coords_new, edit_grid)
        shape_new = t2e.masked_shape_slat(
            pipeline, sampler, p1_feats_s2, coords0, coords_new, edit_grid,
            cond_orig_s2, cond_edit_s2, logger,
            warmstart=bool(p25_cfg.get("trellis2_s2_warmstart", False)),
            nn_init=bool(p25_cfg.get("trellis2_s2_nn_init", False)),
            anchor_mode=s2_anchor,
            anchor_cutoff=float(p25_cfg.get("trellis2_s2_anchor_cutoff", 0.3)),
            contact_mask=(contact_64 if factor == 1 else None),
            contact_sigma=s2_sigma_eff, res=edit_res)
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
        # coords_new == coords0 already lives on the S2 grid; bring the 64³
        # edit_grid down to match (no-op for 1024).
        if factor > 1:
            edit_grid = downsample_edit_grid(edit_grid, factor).to(dev)
        shape_new = shape0   # reuse original geometry verbatim
        logger.info("[s5/P4] %s S2-only (%s): geometry locked, %d voxels",
                    spec.edit_id, edit_type, coords0.shape[0])

    # ── material + decode ─────────────────────────────────────────────
    tex_new = None
    if white_model:
        from partcraft.pipeline_v3.trellis2_white import build_white_model_mesh
        logger.info("[s5/P4] %s white-model — skipping texture stage",
                    spec.edit_id)
        mesh = build_white_model_mesh(pipeline, shape_new, logger, res=edit_res)
    else:
        tex_new = t2e.masked_tex_slat(
            pipeline, sampler, shape0, shape_new, coords0, coords_new,
            edit_grid, cond_orig_s2, cond_edit_s2, logger,
            anchor_mode=s2_tex_anchor,
            contact_mask=(contact_64 if factor == 1 else None),
            contact_sigma=s2_sigma_eff, res=edit_res,
            before_tex_denorm=p1_tex_s2)
        torch.cuda.empty_cache()
        meshes = pipeline.decode_latent(shape_new, tex_new, edit_res)
        mesh = meshes[0]
    mesh.simplify(16_777_216)

    # ── retain intermediate latents (SS occupancy + shape/tex SLat) ───
    latents = _collect_edit_latents(
        edit_type=edit_type, target_part_ids=target_part_ids,
        coords0=coords0, coords_new=coords_new,
        shape_new=shape_new, tex_new=tex_new,
        edit_grid=edit_grid, keep16=keep16,
        s1_pad=s1_pad, s1_thresh=s1_thresh, ss_latent=ss_after_latent)
    return mesh, latents


# ─────────────────── gate-E "after" render (post-edit latents) ────────
# Render the decoded post-edit mesh at the named views (front/right/back/left/
# down) so gate-E judges the EDITED-LATENTS result instead of packed previews,
# on the SAME cameras as the o-voxel "before".  Gated off by default until
# gate-E is re-enabled.

_ENVMAP_CACHE: dict = {}


def _get_envmap(p25_cfg: dict, logger):
    if "env" in _ENVMAP_CACHE:
        return _ENVMAP_CACHE["env"]
    from partcraft.render import ovox_views as _ov
    cb = p25_cfg.get("trellis2_codebase", "/mnt/zsn/3dobject/TRELLIS.2")
    hdri = p25_cfg.get("trellis2_hdri", f"{cb}/assets/hdri/forest.exr")
    env = None
    try:
        env = _ov.load_envmap(hdri)
    except Exception as e:
        logger.warning("[s5] envmap load failed (%s); after-views unlit", e)
    _ENVMAP_CACHE["env"] = env
    return env


def _render_after_named_views(mesh_obj, pair_dir: Path, p25_cfg: dict, logger):
    """Render the decoded post-edit mesh at the named views → ``after_view_{name}.png``.

    Shaded PBR render (PbrMeshRenderer + envmap, white bg) on the named cameras,
    same renderer/lighting as the "before", so gate-E compares like-for-like.
    Consumes the in-memory decoded mesh — no GLB round-trip.
    """
    from PIL import Image
    from partcraft.render import ovox_views as _ov
    env = _get_envmap(p25_cfg, logger)
    res = int(p25_cfg.get("trellis2_gate_view_res", 512))
    imgs = _ov.render_sample(mesh_obj, envmap=env, resolution=res, bg=(1, 1, 1))
    for name, rgb in imgs.items():
        Image.fromarray(rgb).save(pair_dir / f"after_view_{name}.png")
    logger.info("[s5] %s rendered %d after-views", pair_dir.name, len(imgs))


def _slat_from_npz(npz_path: Path):
    """Reconstruct a SparseTensor from a saved p1 latent npz (feats, coords)."""
    import numpy as _np
    import torch
    import trellis2.modules.sparse as sp
    z = _np.load(str(npz_path))
    feats = torch.from_numpy(z["feats"]).float().cuda()
    c = torch.from_numpy(z["coords"]).int()
    coords = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.int32), c], 1).cuda()
    return sp.SparseTensor(feats=feats, coords=coords)


def _render_before_named_views(pipeline, ctx, gate_dir: Path, p25_cfg: dict, logger):
    """Render the ORIGINAL mesh at the named views → ``before_view_{name}.png``.

    Fully latents-level (no glb): decode the p1-encoded shape+tex SLat →
    MeshWithVoxel → SAME PbrMeshRenderer + envmap + white bg as the "after".
    So before/after are the same source (both ``decode_latent`` outputs).
    """
    from PIL import Image
    from partcraft.render import ovox_views as _ov
    # Decode the ORIGINAL at the edit resolution so before/after match (512 uses
    # the grid-512 sidecar; 1024 uses the canonical 64³ latents).
    edit_res = int(p25_cfg.get("trellis2_edit_res", 1024))
    d = Path(ctx.dir) / "p1_encode"
    suffix = "" if edit_res == 1024 else f"_e{edit_res}"
    shape_slat = _slat_from_npz(d / f"shape_slat{suffix}.npz")
    tex_slat = _slat_from_npz(d / f"tex_slat{suffix}.npz")
    mesh = pipeline.decode_latent(shape_slat, tex_slat, edit_res)[0]
    env = _get_envmap(p25_cfg, logger)
    res = int(p25_cfg.get("trellis2_gate_view_res", 512))
    imgs = _ov.render_sample(mesh, envmap=env, resolution=res, bg=(1, 1, 1))
    gate_dir.mkdir(parents=True, exist_ok=True)
    for name, rgb in imgs.items():
        Image.fromarray(rgb).save(gate_dir / f"before_view_{name}.png")
    logger.info("[s5] %s rendered %d before-views (decoded latents) → %s",
                ctx.obj_id, len(imgs), gate_dir.name)


# ─────────────────── per-object processing ───────────────────────────

def _finish_ss_latent_only(ctx, spec, pair_dir, latents, log) -> bool:
    """Backfill writer: persist ss_latent.npz (+ mask.npz) for one edit, verify the
    S1 re-run reproduced the original occupancy.  Returns False on hard failure."""
    import numpy as _np
    out = pair_dir / "latents"
    ss_after = latents.get("ss_latent")
    if ss_after is None:
        log.warning("[s5/ss_latent] %s no after SS latent (S2-only type) — skip",
                    spec.edit_id)
        return False
    _np.savez_compressed(out / "ss_latent.npz", ss=ss_after)

    # consistency: downsample2(re-run coords_new@64³) vs saved ss.npz coords_new@32³.
    # seed + cached edited image are fixed → S1 is deterministic → IoU should be 1.0.
    iou = None
    try:
        saved = _np.load(out / "ss.npz", allow_pickle=True)
        cn32_saved = {tuple(int(x) for x in c)
                      for c in _np.asarray(saved["coords_new"]).tolist()}
        cn64 = latents.get("coords_new_64")
        cn32_re = {tuple(int(x) // 2 for x in c)
                   for c in _np.asarray(cn64).tolist()}
        inter = len(cn32_saved & cn32_re)
        union = len(cn32_saved | cn32_re) or 1
        iou = inter / union
    except Exception as e:  # noqa: BLE001
        log.warning("[s5/ss_latent] %s consistency check failed: %s",
                    spec.edit_id, e)

    # mask.npz: pure function of the existing ss.npz region pack (CPU).
    try:
        from partcraft.pipeline_v3.del_add_reencode import mask_from_ss
        ss = _np.load(out / "ss.npz", allow_pickle=True)
        _np.savez_compressed(out / "mask.npz", **mask_from_ss(ss))
    except Exception as e:  # noqa: BLE001
        log.warning("[s5/ss_latent] %s mask save failed: %s", spec.edit_id, e)

    log.info("[s5/ss_latent] %s ss_latent%s + mask saved; coords IoU=%s",
             spec.edit_id, tuple(ss_after.shape),
             "n/a" if iou is None else f"{iou:.4f}")
    update_edit_stage(ctx, spec.edit_id, spec.edit_type, "s5_ss_latent",
                      status="done",
                      verdict={"coords_iou": iou} if iou is not None else None)
    return True


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

    # Backfill mode: re-run S1 ONLY on already-edited mod/scale edits to add the
    # after SS latent (ss_latent.npz) + mask.npz, skipping S2/decode/render.
    # Selects edits that PASSED gate-E (valid training data), HAVE latents/ss.npz,
    # but LACK latents/ss_latent.npz.
    ss_latent_only = bool(p25_cfg.get("trellis2_ss_latent_only", False))
    _es_edits: dict = {}
    if ss_latent_only:
        from .edit_status_io import load_edit_status
        _es_edits = (load_edit_status(ctx) or {}).get("edits", {}) or {}

    all_specs: list[EditSpec] = []
    pending: list[EditSpec] = []
    for spec in iter_flux_specs(ctx):
        all_specs.append(spec)
        if ss_latent_only:
            # only valid (gate-E pass) mod/scale edits are training data
            st = (_es_edits.get(spec.edit_id) or {}).get("stages", {})
            ge = (st.get("gate_e") or st.get("gate_quality") or {}).get("status")
            if ge != "pass":
                res.n_skip += 1
                continue
            ldir = ctx.edit_3d_dir(spec.edit_id) / "latents"
            if not (ldir / "ss.npz").is_file():
                res.n_skip += 1
                continue
            if (ldir / "ss_latent.npz").is_file() and not force:
                res.n_skip += 1
                continue
            pending.append(spec)
            continue
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

    # emit_glb=False → skip the HEAVY to_glb export (remesh + decimate + 4096²
    # texture bake + webp).  The edit stage then only decodes the latents and
    # renders the named gate views from them (PbrMeshRenderer) — fully
    # latents-level, no GLB asset written.  Turn off for fast eval-only smoke;
    # turn on when you actually want the editable GLB asset.
    emit_glb = bool(p25_cfg.get("trellis2_emit_glb", True))
    # After.glb is exported in TRELLIS's canonical frame (Y↔Z swapped, ×0.5,
    # object tipped onto its end).  Reframe it back onto the original full.glb
    # world frame so the dataset/condition/preview all share one frame.
    reframe_on = bool(p25_cfg.get("trellis2_export_partverse_frame", True))
    reframe_npz = ctx.mesh_npz if (reframe_on and ctx.mesh_npz is not None) else None
    edits_2d_subdir = p25_cfg.get("edits_2d_subdir", "edits_2d")
    t0 = time.time()

    from PIL import Image

    # Edit resolution: 1024 (default) or 512.  S1 always uses the 64³ body;
    # for 512, S2 additionally needs the grid-512 (32³) sidecar body.
    edit_res = int(p25_cfg.get("trellis2_edit_res", 1024))

    # Masked latent editing (Vinedresser3D-style: preserve outside the part,
    # re-flow inside) is the only path — load the P1-encoded original SLat that
    # the inversion anchors to.
    p1_feats, p1_coords3 = _load_p1_slat(ctx)          # 64³ — S1
    if edit_res != 1024:
        p1_feats_s2, p1_coords_s2 = _load_p1_slat(ctx, res=edit_res)  # 32³ — S2
    else:
        p1_feats_s2, p1_coords_s2 = p1_feats, p1_coords3
    # P1-encoded ORIGINAL tex latent (aligned to p1_coords_s2) for the
    # restore-style tex posthoc; optional (older trees may lack the sidecar).
    try:
        p1_tex_s2, _ = _load_p1_slat(ctx, res=edit_res, which="tex")
    except FileNotFoundError:
        p1_tex_s2 = None
    from partcraft.pipeline_v3.trellis2_white import read_white_model_flag
    # Phase-1 may flag an object as an untextured white model; the
    # trellis2_force_white_model config knob forces it for ALL objects
    # (shape-only experiments: stop after S2 shape, decode a grey 512 mesh,
    # skip the texture SLat stage).
    white_model = (read_white_model_flag(ctx)
                   or bool(p25_cfg.get("trellis2_force_white_model", False)))
    log.info("[s5/P4] %s loaded P1 SLat (S1 %d tok @64³, S2 %d tok @%d³)  "
             "edit_res=%d white_model=%s", ctx.obj_id,
             int(p1_coords3.shape[0]), int(p1_coords_s2.shape[0]),
             edit_res // 16, edit_res, white_model)

    # gate-E "before": render the ORIGINAL mesh at the named views once per
    # object (same PbrMeshRenderer + envmap + white bg as the per-edit "after"),
    # so gate-E sees one consistent render style.
    render_gate_views = bool(p25_cfg.get("trellis2_render_gate_views", False))
    if render_gate_views and not ss_latent_only:
        try:
            _render_before_named_views(pipeline, ctx, ctx.dir / "gate_views", p25_cfg, log)
        except Exception as e:
            log.warning("[s5] %s before-view render failed: %s", ctx.obj_id, e)

    for spec in pending:
        try:
            edited_path = ctx.dir / edits_2d_subdir / f"{spec.edit_id}_edited.png"
            if not edited_path.exists():
                raise FileNotFoundError(f"missing edited 2D image: {edited_path}")
            edited_img = Image.open(edited_path).convert("RGB")

            pair_dir = ctx.edit_3d_dir(spec.edit_id)
            after_glb = pair_dir / "after.glb"
            # The original (pre-edit) view drives the RF inversion that anchors
            # the preserved region; the edited view drives the forward edit.
            # flux_2d wrote both as *_input/_edited.png.
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
                p1_feats_s2=p1_feats_s2, p1_coords_s2=p1_coords_s2,
                p1_tex_s2=p1_tex_s2,
                edit_res=edit_res,
                ss_latent_only=ss_latent_only,
            )
            if ss_latent_only:
                # S1-only backfill: write ss_latent.npz + mask.npz, verify the
                # re-run reproduced the original occupancy, skip glb/render.
                ok = _finish_ss_latent_only(ctx, spec, pair_dir, latents, log)
                res.n_ok += 1 if ok else 0
                res.n_fail += 0 if ok else 1
                continue
            if bool(p25_cfg.get("trellis2_save_latents", True)):
                try:
                    _save_edit_latents(latents, pair_dir, log)
                except Exception as e:
                    log.warning("[s5] %s latent save failed: %s",
                                spec.edit_id, e)
            if emit_glb:
                _export_edit_glb(mesh_obj, after_glb, p25_cfg, log,
                                 reframe_mesh_npz=reframe_npz)
            # gate-E "after" = render the post-edit latents at named views.
            if render_gate_views:
                try:
                    _render_after_named_views(mesh_obj, pair_dir, p25_cfg, log)
                except Exception as e:
                    log.warning("[s5] %s after-view render failed: %s",
                                spec.edit_id, e)

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
    # _ensure_pipeline / _export_edit_glb.
    p25_cfg = psvc.trellis_image_edit_flat(cfg)

    from partcraft.io.hy3d_loader import HY3DPartDataset
    dataset = HY3DPartDataset(str(images_root), str(mesh_root), [shard])

    pipeline = _ensure_pipeline(p25_cfg, log)

    # Backfill mode re-runs S1 on edits that ALREADY finished s5, so the
    # object-level "s5 done → skip" gate must NOT short-circuit it; let
    # run_for_object's own per-edit ss_latent.npz selection decide.
    ss_latent_only = bool(p25_cfg.get("trellis2_ss_latent_only", False))

    results: list[Trellis2Result] = []
    for ctx in list(ctxs):
        edit_ids = [sp.edit_id for sp in iter_flux_specs(ctx)]
        if (not ss_latent_only and edit_ids and not force
                and not obj_needs_stage(ctx, edit_ids, "s5", prereq_map, force=force)):
            results.append(Trellis2Result(ctx.obj_id))
            continue
        # Per-object guard: a single bad object (e.g. missing P1 encode
        # output from a segfaulted encode worker — FileNotFoundError in
        # _load_p1_slat) must NOT propagate to main() and kill this GPU
        # worker, which would abandon the rest of its 1/N object slice.
        # Record the failure and continue to the next object.
        try:
            results.append(run_for_object(
                ctx, pipeline=pipeline, dataset=dataset, p25_cfg=p25_cfg,
                seed=seed, debug=debug,
                prereq_map=prereq_map, force=force, logger=log,
            ))
        except FileNotFoundError as exc:
            # Missing prerequisite (P1 encode) — object is unprocessable as-is;
            # mark skip so resume/validate can account for it and move on.
            log.error("[s5] %s skipped — missing prerequisite: %s",
                      ctx.obj_id, exc)
            update_step(ctx, "s5_trellis2", status=STATUS_SKIP,
                        reason=f"missing_prereq: {exc}")
            results.append(Trellis2Result(ctx.obj_id, error=str(exc)))
        except Exception as exc:  # noqa: BLE001 — never let one object kill the worker
            log.exception("[s5] %s failed with unhandled error: %s",
                          ctx.obj_id, exc)
            update_step(ctx, "s5_trellis2", status=STATUS_FAIL,
                        error=str(exc))
            results.append(Trellis2Result(ctx.obj_id, error=str(exc)))
    return results


__all__ = ["GPU_TYPES", "Trellis2Result", "run_for_object", "run"]
