"""Step s4b — TRELLIS.2 P1: encode original mesh → shape SLat.

Per-object: reads ``ctx.mesh_npz`` (full.glb blob), voxelizes to a 1024³
dual grid (the f16 encoder downsamples 16× internally so the SLat lives
at 64³), runs the shape encoder, and stores

    ctx.dir / p1_encode / shape_slat.npz

with keys ``feats`` (float32 [N, 32]) and ``coords`` (int32 [N, 3]).

Downstream :mod:`trellis2_3d` (P4 branch) reads this latent + the
target part ids to drive masked sampling.

Single-GPU; the orchestrator slices objects across GPUs via
``CUDA_VISIBLE_DEVICES`` subprocesses just like :mod:`trellis2_3d`.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
from scripts.data_prep.mesh_sources import open_mesh, mesh_available  # noqa: E402
sys.path.insert(0, str(_ROOT))

from .paths import ObjectContext
from . import services_cfg as psvc
from .status import update_step, STATUS_OK, STATUS_FAIL, STATUS_SKIP


SHAPE_ENC_NAME = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
TEX_ENC_NAME = "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
SS_ENC_NAME = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"


def _ensure_encoder(p25_cfg: dict, logger):
    """Load + cache the shape encoder once per process (shape only — legacy)."""
    sys.path.insert(0, str(Path(p25_cfg.get(
        "trellis2_codebase", "/mnt/zsn/3dobject/TRELLIS.2")).resolve()))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    import trellis2.models as t2_models
    name = p25_cfg.get("trellis2_shape_enc", SHAPE_ENC_NAME)
    logger.info("[s4b] loading shape encoder %s", name)
    enc = t2_models.from_pretrained(name).eval().cuda()
    return enc


def _ensure_encoders(p25_cfg: dict, logger):
    """Load shape + tex + ss encoders (dict).  Latents are saved together so the
    original mesh round-trips through ``decode_latent`` (shape+tex share coords,
    verified) for fully latents-level before/after rendering."""
    sys.path.insert(0, str(Path(p25_cfg.get(
        "trellis2_codebase", "/mnt/zsn/3dobject/TRELLIS.2")).resolve()))
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import trellis2.models as t2_models
    names = {"shape": p25_cfg.get("trellis2_shape_enc", SHAPE_ENC_NAME),
             "tex": p25_cfg.get("trellis2_tex_enc", TEX_ENC_NAME),
             "ss": p25_cfg.get("trellis2_ss_enc", SS_ENC_NAME)}
    encs = {}
    for k, name in names.items():
        logger.info("[s4b] loading %s encoder %s", k, name)
        encs[k] = t2_models.from_pretrained(name).eval().cuda()
    return encs


def encode_shape_tex_ss(encoders: dict, mesh_npz: Path, grid_size: int = 1024,
                        canonical: bool = True) -> dict:
    """Encode the original mesh → shape SLat + tex SLat + SS latent.

    shape (dual-grid geometry) and tex (volumetric PBR attr) land on the SAME
    64³ coords, so ``decode_latent(shape, tex)`` reconstructs the original mesh.
    Returns numpy arrays ready to save.
    """
    import trellis2.modules.sparse as sp
    import o_voxel
    import trimesh
    from partcraft.pipeline_v3 import trellis2_ovox_render as OVR

    scene = OVR.load_full_scene(Path(mesh_npz))
    groups, _M = OVR._normalized_groups(scene, canonical=canonical)
    merged = trimesh.util.concatenate(groups)
    verts = torch.from_numpy(np.asarray(merged.vertices)).float()
    faces = torch.from_numpy(np.asarray(merged.faces)).long()
    aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    # ── shape (dual grid) ──
    vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        verts.cpu(), faces.cpu(), grid_size=grid_size, aabb=aabb,
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dual_local = (dv * grid_size - vi).clamp(0., 1.).float()
    if inter.dim() == 2 and inter.shape[1] == 3:
        inter3 = inter.float()
    else:
        b = inter.view(-1).to(torch.uint8)
        inter3 = torch.stack([(b & 1).bool(), ((b >> 1) & 1).bool(), ((b >> 2) & 1).bool()], -1).float()
    sh_coords = torch.cat([torch.zeros_like(vi[:, :1]), vi], -1).int()
    vsp = sp.SparseTensor(feats=dual_local, coords=sh_coords).cuda()
    isp = vsp.replace(inter3.bool().float().cuda())
    with torch.no_grad():
        shape_slat = encoders["shape"](vsp, isp)

    # ── tex (volumetric PBR attr; base_color+metallic+roughness+alpha = 6ch) ──
    coord, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        trimesh.Scene(groups), grid_size=grid_size, aabb=aabb)
    def _f(x):
        return (x.float() / 255.0) if x.dtype == torch.uint8 else x.float()
    feats6 = torch.cat([_f(attr["base_color"]), _f(attr["metallic"]),
                        _f(attr["roughness"]), _f(attr["alpha"])], -1).float()
    tx_coords = torch.cat([torch.zeros_like(coord[:, :1]), coord], -1).int()
    tsp = sp.SparseTensor(feats=feats6, coords=tx_coords).cuda()
    with torch.no_grad():
        tex_slat = encoders["tex"](tsp)

    # ── ss (occupancy @64³ from shape coords → ss_enc) ──
    c64 = shape_slat.coords[:, 1:].long()
    occ = torch.zeros(1, 1, 64, 64, 64, device=shape_slat.device)
    occ[0, 0, c64[:, 0], c64[:, 1], c64[:, 2]] = 1.0
    with torch.no_grad():
        z_s = encoders["ss"](occ.float())

    return {
        "shape_feats": shape_slat.feats.cpu().numpy().astype(np.float32),
        "shape_coords": shape_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
        "tex_feats": tex_slat.feats.cpu().numpy().astype(np.float32),
        "tex_coords": tex_slat.coords[:, 1:].cpu().numpy().astype(np.int32),
        "ss": z_s.detach().cpu().numpy().astype(np.float32),
    }


# Canonical-frame rotation R_X90: partverse **Y-up** → TRELLIS **Z-up**.
# Right-multiply matrix so a row vertex (x,y,z) → (x,-z,y).  TRELLIS.2's
# image→3D latents live in a Z-up canonical frame (its render_utils render
# with up=[0,0,1]; to_glb then Z-up→Y-up for standard GLB).  partverse meshes
# are Y-up, and the encoder does center+scale only — so coords0/SLat come out
# Y-up = NON-canonical.  That mismatch is invisible for full regen (the model
# generates in its own frame) but breaks MASKED editing, which injects coords0:
# the model's canonical prior fights the Y-up structure → spiky edits.  Applying
# this rotation (identically in encode AND part_mask, AFTER center+scale so the
# mask stays byte-aligned) puts the injected latent in the model's frame.
_CANON_ROT = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    """Center + scale to [-0.5, 0.5]^3 (matches P1 + part_mask conventions)."""
    vmin = v.min(0)[0]
    vmax = v.max(0)[0]
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    return (v - center) * scale


def _load_full_mesh(npz_path: Path) -> trimesh.Trimesh:
    d = open_mesh(npz_path)
    if "full.glb" not in d.files:
        raise KeyError(f"no 'full.glb' in {npz_path}; have {d.files}")
    scene = trimesh.load(io.BytesIO(d["full.glb"].tobytes()),
                         file_type="glb", process=False)
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(
            [g for g in scene.geometry.values()
             if isinstance(g, trimesh.Trimesh)])
    return scene


def encode_full_mesh(enc, mesh_npz: Path, grid_size: int = 1024,
                     canonical: bool = False):
    """Encode a partverse ``mesh.npz`` (full.glb) → shape SLat.

    Returns ``(feats, coords)`` numpy arrays — ``feats`` float32 ``[N,32]``,
    ``coords`` int32 ``[N,3]`` in 0..63.  Shared by the s4b stage and the
    minimal single-object runner so the encode recipe lives in one place.

    ``canonical=True`` rotates the (centered+scaled) mesh by :data:`_CANON_ROT`
    (partverse Y-up → TRELLIS Z-up) before voxelizing, so the latent is in the
    model's canonical frame for masked editing.  part_mask must use the SAME
    flag so the edit mask stays aligned with these coords.
    """
    import trellis2.modules.sparse as sp
    import o_voxel

    mesh = _load_full_mesh(mesh_npz)
    verts = torch.from_numpy(np.asarray(mesh.vertices)).float()
    faces = torch.from_numpy(np.asarray(mesh.faces)).long()
    verts = _normalize(verts)
    if canonical:
        verts = verts @ _CANON_ROT.to(verts.dtype)

    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices=verts.float(),
            faces=faces.long(),
            grid_size=grid_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=False,
        )
    )
    dual_local = (dual_vertices * grid_size - voxel_indices).clamp(0., 1.).float()
    if intersected.dim() == 2 and intersected.shape[1] == 3:
        inter3 = intersected.float()
    else:
        b = intersected.view(-1).to(torch.uint8)
        inter3 = torch.stack([
            (b & 1).bool(),
            ((b >> 1) & 1).bool(),
            ((b >> 2) & 1).bool(),
        ], dim=-1).float()

    coords = torch.cat([
        torch.zeros_like(voxel_indices[:, 0:1]),
        voxel_indices,
    ], dim=-1).to(torch.int32)
    vertices_sp = sp.SparseTensor(dual_local, coords)
    intersected_sp = vertices_sp.replace(inter3.bool().float())
    with torch.no_grad():
        z = enc(vertices_sp.cuda(), intersected_sp.cuda())
    return (z.feats.cpu().numpy().astype(np.float32),
            z.coords[:, 1:].cpu().numpy().astype(np.int32))


def _slat_from_arrays(feats: np.ndarray, coords: np.ndarray):
    import trellis2.modules.sparse as sp
    f = torch.from_numpy(feats).float().cuda()
    c = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32),
                   torch.from_numpy(coords).int()], 1).cuda()
    return sp.SparseTensor(feats=f, coords=c)


# NOTE: the old _render_overview_at_encode (render_pbr_overview path) was removed.
# Encode now renders gate_views via _render_before_named_views (PBR decode +
# render_sample, == the post-edit "after" render) and builds the overview as PBR
# RGB (reused gate_views) + robust o-voxel seg — see _encode_one below.  The
# render_pbr_overview seg pass rasterised raw part-meshes and crashed CUDA.


def _encode_one(ctx: ObjectContext, encoders: dict, p25_cfg: dict, logger,
                pipeline=None, envmap=None, force: bool = False) -> Path:
    """Encode full mesh → shape/tex/ss latents; optionally render unified overview."""
    if not mesh_available(ctx.mesh_npz):
        raise FileNotFoundError(f"missing mesh_npz: {ctx.mesh_npz}")

    grid_size = int(p25_cfg.get("trellis2_p1_grid", 1024))
    canonical = bool(p25_cfg.get("trellis2_canonical_frame", False))
    edit_res = int(p25_cfg.get("trellis2_edit_res", 1024))

    d = ctx.dir / "p1_encode"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "shape_slat.npz"

    # ── canonical 64³ encode (skip if already present — e.g. symlinked from a
    # sibling experiment tree; only the grid-(edit_res) sidecar is then needed) ──
    main_present = all((d / f).is_file() and (d / f).stat().st_size > 0
                       for f in ("shape_slat.npz", "tex_slat.npz", "ss.npz"))
    enc_out = None
    if force or not main_present:
        enc_out = encode_shape_tex_ss(encoders, ctx.mesh_npz, grid_size, canonical=canonical)
        np.savez(out, feats=enc_out["shape_feats"], coords=enc_out["shape_coords"])
        np.savez(d / "tex_slat.npz", feats=enc_out["tex_feats"], coords=enc_out["tex_coords"])
        np.savez(d / "ss.npz", ss=enc_out["ss"])
        logger.info("[s4b] %s encoded → shape+tex(%d tokens)+ss at %s",
                    ctx.obj_id, int(enc_out["shape_coords"].shape[0]), d)

    # ── grid-(edit_res) sidecar for the 512-edit body anchor (32³ shape/tex) ──
    # The _512 SLat flow models consume res//16 ³ coords; the 64³ encode above
    # can't feed them.  ss not needed (S1 always uses the 64³ body).
    if edit_res != 1024:
        sc = d / f"shape_slat_e{edit_res}.npz"
        tc = d / f"tex_slat_e{edit_res}.npz"
        if force or not (sc.is_file() and sc.stat().st_size > 0
                         and tc.is_file() and tc.stat().st_size > 0):
            e2 = encode_shape_tex_ss(encoders, ctx.mesh_npz, edit_res, canonical=canonical)
            np.savez(sc, feats=e2["shape_feats"], coords=e2["shape_coords"])
            np.savez(tc, feats=e2["tex_feats"], coords=e2["tex_coords"])
            logger.info("[s4b] %s grid-%d sidecar → shape+tex(%d tokens @%d³) at %s",
                        ctx.obj_id, edit_res, int(e2["shape_coords"].shape[0]),
                        edit_res // 16, d)

    # Render the named source views (gate_views/before_view_*) at ENCODE time via
    # the SAME robust path as the post-edit "after" render: decode the P1 latents
    # → mesh → render_sample (PbrMeshRenderer).  This is exactly what trellis2_3d
    # does for before/after (_render_before_named_views) and is production-proven.
    # It deliberately AVOIDS render_pbr_overview's extra part-palette SEGMENTATION
    # pass, which hit CUDA error 700 on some degenerate part-meshes and corrupted
    # the worker's CUDA context (cascading all subsequent renders to failure).
    # flux_2d then just LOADS these PBR PNGs — no on-the-fly render.  (gate-A's
    # RGB+seg overview is still produced by gen_edits' o-voxel backfill; the seg
    # render is the fragile part we keep off the GPU-render hot path.)
    if pipeline is not None:
        gd = ctx.dir / "gate_views"
        if force or not (gd / "before_view_front.png").is_file():
            try:
                from partcraft.pipeline_v3.trellis2_3d import _render_before_named_views
                _render_before_named_views(pipeline, ctx, gd, p25_cfg, logger)
            except Exception as e:
                logger.warning("[s4b] %s before-views render failed: %s", ctx.obj_id, e)
        # Build gate-A's overview.png = PBR RGB top (REUSE the gate_views just
        # rendered — RGB is rendered ONCE, no redundant o-voxel RGB voxelisation)
        # + robust o-voxel SEG bottom (render_overview_png(skip_rgb)).  If the PBR
        # views are missing, leave the overview to gen_edits' o-voxel backfill.
        if force or not ctx.overview_path.is_file():
            try:
                from PIL import Image as _Img
                from partcraft.render import ovox_views as _ov2
                from partcraft.pipeline_v3.vlm_core import render_overview_png as _rop
                pngs = [gd / f"before_view_{v}.png" for v in _ov2.VIEW_ORDER]
                if all(p.is_file() for p in pngs):
                    rgb_top = [np.array(_Img.open(p).convert("RGB")) for p in pngs]
                    ctx.phase1_dir.mkdir(parents=True, exist_ok=True)
                    png = _rop(ctx.mesh_npz,
                               save_viewpoints=ctx.phase1_dir / "viewpoints.json",
                               rgb_override=rgb_top)
                    ctx.overview_path.write_bytes(png)
                    logger.info("[s4b] %s overview = PBR RGB + o-voxel seg", ctx.obj_id)
            except Exception as e:
                logger.warning("[s4b] %s overview build failed: %s", ctx.obj_id, e)
    return out


# the encode stage now needs all 3 latents present to skip an object.
def _p1_complete(ctx: ObjectContext, edit_res: int = 1024) -> bool:
    d = ctx.dir / "p1_encode"
    files = ["shape_slat.npz", "tex_slat.npz", "ss.npz"]
    if edit_res != 1024:
        # 512-edit also needs the grid-(edit_res) sidecar (32³ body for S2).
        files += [f"shape_slat_e{edit_res}.npz", f"tex_slat_e{edit_res}.npz"]
    return all((d / f).is_file() and (d / f).stat().st_size > 0 for f in files)


# ─────────────────── batch entrypoint (single GPU) ───────────────────

def run(
    ctxs: Iterable[ObjectContext],
    *,
    cfg: dict,
    images_root: Path | None = None,
    mesh_root: Path | None = None,
    shard: str = "01",
    prereq_map: dict[str, str | None] | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or logging.getLogger("pipeline_v3.t2_encode")
    log.info("[s4b] CUDA_VISIBLE_DEVICES=%s",
             os.environ.get("CUDA_VISIBLE_DEVICES"))
    p25_cfg = psvc.trellis_image_edit_flat(cfg)

    # When overview rendering is on, also load the decode pipeline + envmap so
    # the encode stage produces the unified PBR overview (RGB from decoded
    # latents + part-mesh segmentation) — front-loaded, no Blender, no o-voxel.
    render_overview = bool(p25_cfg.get("trellis2_encode_render_overview", False))
    edit_res = int(p25_cfg.get("trellis2_edit_res", 1024))
    encoders = None
    pipeline = None
    envmap = None
    t0 = time.time()
    n_ok = n_fail = n_skip = n_bad = 0
    ctxs = list(ctxs)
    # Bad-mesh registry: skip meshes already known to hard-crash a GPU step
    # (process-fatal SIGSEGV in the o_voxel voxelizer / renderer — uncatchable).
    # Under a respawn supervisor the in-flight guard also DETECTS new ones: the
    # object stamped before a crash is recorded bad on the next respawn.
    from . import bad_mesh as _bm
    _root = ctxs[0].root if ctxs else None
    bad_set = _bm.load_bad(_root) if _root else set()
    guard = _bm.make_guard(_root, shard, "trellis2_encode", log) if _root else None
    if guard is not None:
        bad_set = guard.bad
    for ctx in ctxs:
        if ctx.obj_id in bad_set:
            n_bad += 1
            update_step(ctx, "s4b_t2_encode", status=STATUS_SKIP, reason="bad_mesh")
            continue
        # When render_overview is on, the object is only "done" once the named
        # source views (gate_views/before_view_*) exist — flux_2d loads these
        # instead of doing a concurrent on-the-fly render.  Check the views, NOT
        # phase1/overview.png (gate-A writes that separately), so an already-
        # encoded object still gets its views rendered (without re-encoding).
        ov_done = (not render_overview) or (
            (ctx.dir / "gate_views" / "before_view_front.png").is_file()
            and ctx.overview_path.is_file())
        if _p1_complete(ctx, edit_res) and ov_done and not force:
            n_skip += 1
            update_step(ctx, "s4b_t2_encode", status=STATUS_OK, reason="exists")
            continue
        try:
            if encoders is None:
                encoders = _ensure_encoders(p25_cfg, log)
                if render_overview:
                    from partcraft.pipeline_v3.trellis2_3d import _ensure_pipeline
                    from partcraft.render import ovox_views as _ov
                    pipeline = _ensure_pipeline(p25_cfg, log)
                    cb = p25_cfg.get("trellis2_codebase", "/mnt/zsn/3dobject/TRELLIS.2")
                    hdri = p25_cfg.get("trellis2_hdri", f"{cb}/assets/hdri/forest.exr")
                    envmap = _ov.load_envmap(hdri)
            # stamp the object in-flight BEFORE the crash-prone voxelize/render;
            # if it hard-segfaults, the next respawn records it bad and skips it.
            if guard is not None:
                guard.beat(ctx.obj_id)
            _encode_one(ctx, encoders, p25_cfg, log, pipeline=pipeline,
                        envmap=envmap, force=force)
            n_ok += 1
            update_step(ctx, "s4b_t2_encode", status=STATUS_OK)
        except Exception as e:
            log.error("[s4b] %s failed: %s", ctx.obj_id, e)
            n_fail += 1
            update_step(ctx, "s4b_t2_encode", status=STATUS_FAIL,
                        error=str(e)[:200])
        finally:
            # object survived (ok or caught failure) → not the segfault culprit
            if guard is not None:
                guard.clear()
    log.info(
        "[s4b] done: ok=%d fail=%d skip=%d bad_mesh=%d wall=%.1fs",
        n_ok, n_fail, n_skip, n_bad, time.time() - t0,
    )


__all__ = ["run"]
