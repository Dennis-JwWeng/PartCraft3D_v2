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

from .paths import ObjectContext
from . import services_cfg as psvc
from .status import update_step, STATUS_OK, STATUS_FAIL


SHAPE_ENC_NAME = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"


def _ensure_encoder(p25_cfg: dict, logger):
    """Load + cache the shape encoder once per process."""
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
    d = np.load(npz_path, allow_pickle=True)
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


def _encode_one(ctx: ObjectContext, enc, p25_cfg: dict, logger) -> Path:
    """Encode this object's full mesh → shape_slat.npz."""
    if ctx.mesh_npz is None or not ctx.mesh_npz.is_file():
        raise FileNotFoundError(f"missing mesh_npz: {ctx.mesh_npz}")

    grid_size = int(p25_cfg.get("trellis2_p1_grid", 1024))
    canonical = bool(p25_cfg.get("trellis2_canonical_frame", False))
    feats, coords = encode_full_mesh(enc, ctx.mesh_npz, grid_size,
                                     canonical=canonical)

    out = ctx.dir / "p1_encode" / "shape_slat.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, feats=feats, coords=coords)
    logger.info("[s4b] %s encoded → %d tokens at %s",
                ctx.obj_id, int(coords.shape[0]), out)
    return out


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

    enc = None
    t0 = time.time()
    n_ok = n_fail = n_skip = 0
    ctxs = list(ctxs)
    for ctx in ctxs:
        out = ctx.dir / "p1_encode" / "shape_slat.npz"
        if out.is_file() and out.stat().st_size > 0 and not force:
            n_skip += 1
            update_step(ctx, "s4b_t2_encode", status=STATUS_OK, reason="exists")
            continue
        try:
            if enc is None:
                enc = _ensure_encoder(p25_cfg, log)
            _encode_one(ctx, enc, p25_cfg, log)
            n_ok += 1
            update_step(ctx, "s4b_t2_encode", status=STATUS_OK)
        except Exception as e:
            log.error("[s4b] %s failed: %s", ctx.obj_id, e)
            n_fail += 1
            update_step(ctx, "s4b_t2_encode", status=STATUS_FAIL,
                        error=str(e)[:200])
    log.info(
        "[s4b] done: ok=%d fail=%d skip=%d wall=%.1fs",
        n_ok, n_fail, n_skip, time.time() - t0,
    )


__all__ = ["run"]
