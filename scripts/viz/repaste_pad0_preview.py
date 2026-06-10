#!/usr/bin/env python3
"""Re-paste a finished posthoc edit with a NEW (default pad=0) mask — no generation.

Takes the prod pad4 run's saved occupancy AS-IS (``latents/ss.npz`` coords_new,
already includes the in-run 64³ restore) and only re-runs the posthoc body
paste with a freshly built pad=0 edit grid: preserved = in_C0 & ~grid, then
``after[preserved] = before_e512[src]`` for BOTH shape and tex (denorm space —
identical to the in-pipeline paste).  Decodes the original-after and the
re-pasted-after latents at 512 and renders 6 views of each for comparison.

Outputs under ``<edit_dir>/repaste_pad{P}/``:
  after_orig_view_{front,...}.png     decode of the saved (pad4-pasted) latents
  after_pad{P}_view_{...}.png         decode of the re-pasted latents
  compare_{...}.png                   top=orig, bottom=re-pasted
  meta.json                           paste stats

Usage:
  CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 \\
    python scripts/viz/repaste_pad0_preview.py \\
      --root data/Pxform_v2/prod_posthoc_no2dqc --shard 06 \\
      --obj 727f33155d8c46fc9f01a7ee7c346538 [--edit mod_] [--pad 0]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description="pad0 re-paste preview (no generation)")
    ap.add_argument("--root", default="data/Pxform_v2/prod_posthoc_no2dqc")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--edit", default="", help="edit_id substring; default=ALL edits with latents")
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--pad", type=int, default=0, help="new edit-grid dilation (default 0)")
    ap.add_argument("--no-subtract-preserved", action="store_true",
                    help="match runs without contact_soft (prod default subtracts)")
    ap.add_argument("--res", type=int, default=512, help="render resolution")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("repaste_pad0")

    import numpy as np
    import torch
    from PIL import Image
    import trellis2.modules.sparse as sp

    from partcraft.pipeline_v3 import trellis2_3d as T
    from partcraft.pipeline_v3.trellis2_part_mask import (
        build_coord_bridge, downsample_edit_grid, part_edit_grid_64)
    from partcraft.render import ovox_views as ov

    EDIT_RES = 512
    GRID = EDIT_RES // 16          # 32
    views = ov.SIX_VIEW_ORDER

    obj_dir = Path(args.root) / "objects" / args.shard / args.obj
    mesh_npz = Path(args.mesh_root) / args.shard / f"{args.obj}.npz"
    if not mesh_npz.is_file():
        log.error("mesh npz missing: %s", mesh_npz)
        return

    # before (P1-encoded) latents on coords0 — the paste source
    p1 = obj_dir / "p1_encode"
    bshape = np.load(p1 / "shape_slat_e512.npz")
    btex = np.load(p1 / "tex_slat_e512.npz")

    edit_dirs = sorted(
        d.parent for d in (obj_dir / "edits_3d").glob("*/latents/ss.npz")
        if not args.edit or args.edit in d.parent.parent.name)
    if not edit_dirs:
        log.error("no edits with latents matching %r under %s", args.edit, obj_dir)
        return
    log.info("%d edit(s): %s", len(edit_dirs), [d.parent.name for d in edit_dirs])

    p25_cfg = {"trellis2_codebase": TRELLIS2_DIR, "trellis2_ckpt": args.ckpt}
    pipeline = T._ensure_pipeline(p25_cfg, log)
    env = ov.load_envmap(f"{TRELLIS2_DIR}/assets/hdri/forest.exr")

    def _sparse(coords_np, feats):
        c = torch.from_numpy(np.asarray(coords_np)).int()
        coords = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.int32), c], 1).cuda()
        return sp.SparseTensor(feats=feats.float().cuda(), coords=coords)

    def _render(mesh, out_dir, prefix):
        imgs = ov.render_sample(mesh, view_names=views, envmap=env,
                                resolution=args.res, bg=(1, 1, 1))
        for nm, rgb in imgs.items():
            Image.fromarray(rgb).save(out_dir / f"{prefix}_view_{nm}.png")
        return imgs

    for lat_dir in edit_dirs:
        edit_dir = lat_dir.parent
        edit_id = edit_dir.name
        t0 = time.time()
        ss = np.load(lat_dir / "ss.npz", allow_pickle=True)
        coords0 = torch.from_numpy(ss["coords0"].astype("int64"))
        coords_new = torch.from_numpy(ss["coords_new"].astype("int64"))
        parts = [int(p) for p in ss["parts"]]
        s1_pad = int(ss["s1_pad"]) if "s1_pad" in ss else -1
        ashape = np.load(lat_dir / "shape_slat.npz")
        atex = np.load(lat_dir / "tex_slat.npz")
        if (bshape["feats"].shape[0] != coords0.shape[0]
                or ashape["feats"].shape[0] != coords_new.shape[0]):
            log.warning("%s: row mismatch (before %d vs coords0 %d, after %d vs "
                        "coords_new %d) — skipped", edit_id,
                        bshape["feats"].shape[0], coords0.shape[0],
                        ashape["feats"].shape[0], coords_new.shape[0])
            continue

        # new edit grid at the requested pad, downsampled to the 32³ SLat grid
        grid64 = part_edit_grid_64(
            mesh_npz, parts, pad=args.pad, canonical=True,
            subtract_preserved=not args.no_subtract_preserved)
        grid32 = downsample_edit_grid(grid64, 64 // GRID).cuda()

        preserved, src_idx = build_coord_bridge(
            coords0.cuda(), coords_new.cuda(), grid32, grid=GRID)

        # old preserved set (saved run grid) for stats — ss.npz stores the
        # edit_grid POST-_to_s2, i.e. already on the 32³ SLat grid for 512 runs
        # (sparse coords); infer the resolution instead of assuming 64³.
        eg = ss["edit_grid"].astype("int64")
        G_saved = GRID if eg.max() < GRID else 64
        old = torch.zeros(G_saved, G_saved, G_saved, dtype=torch.bool)
        old[eg[:, 0], eg[:, 1], eg[:, 2]] = True
        if G_saved != GRID:
            old = downsample_edit_grid(old, G_saved // GRID)
        old_pres, _ = build_coord_bridge(coords0.cuda(), coords_new.cuda(), old.cuda(), grid=GRID)
        n_new, n_old = int(preserved.sum()), int(old_pres.sum())
        log.info("%s parts=%s run_pad=%d: preserved %d → %d (+%d ring tokens "
                 "of %d total)", edit_id, parts, s1_pad, n_old, n_new,
                 n_new - n_old, coords_new.shape[0])

        # re-paste in denorm space (== in-pipeline normalized paste)
        sh = torch.from_numpy(ashape["feats"]).float().cuda()
        tx = torch.from_numpy(atex["feats"]).float().cuda()
        sh[preserved] = torch.from_numpy(bshape["feats"]).float().cuda()[src_idx]
        tx[preserved] = torch.from_numpy(btex["feats"]).float().cuda()[src_idx]

        out_dir = edit_dir / f"repaste_pad{args.pad}"
        out_dir.mkdir(parents=True, exist_ok=True)

        mesh_orig = pipeline.decode_latent(
            _sparse(ashape["coords"], torch.from_numpy(ashape["feats"])),
            _sparse(atex["coords"], torch.from_numpy(atex["feats"])), EDIT_RES)[0]
        mesh_orig.simplify(16_777_216)
        imgs_o = _render(mesh_orig, out_dir, "after_orig")

        mesh_new = pipeline.decode_latent(
            _sparse(ashape["coords"], sh.cpu()),
            _sparse(atex["coords"], tx.cpu()), EDIT_RES)[0]
        mesh_new.simplify(16_777_216)
        imgs_n = _render(mesh_new, out_dir, f"after_pad{args.pad}")

        for nm in views:
            top, bot = imgs_o[nm], imgs_n[nm]
            Image.fromarray(np.concatenate([top, bot], axis=0)).save(
                out_dir / f"compare_{nm}.png")

        (out_dir / "meta.json").write_text(json.dumps({
            "edit_id": edit_id, "parts": parts, "run_pad": s1_pad,
            "repaste_pad": args.pad, "tokens_total": int(coords_new.shape[0]),
            "preserved_old": n_old, "preserved_new": n_new,
            "ring_tokens_repasted": n_new - n_old,
        }, indent=2))
        log.info("%s done in %.1fs → %s", edit_id, time.time() - t0, out_dir)


if __name__ == "__main__":
    main()
