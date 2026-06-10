#!/usr/bin/env python3
"""Batch pad0 re-paste over a prod shard — gate-E-passed edits only, no generation.

Per edit (``stages.gate_e.status == "pass"`` with saved latents):
  1. occupancy AS-IS: ``latents/ss.npz`` coords_new (pad4 run, in-run restore kept)
  2. rebuild the edit grid at ``--pad`` (default 0) on 64³ → max-pool to 32³
  3. preserved = in_C0 & ~grid; hard-paste ``p1_encode/{shape,tex}_slat_e512`` rows
  4. save re-pasted latents + decode @512 + render ONLY the gate-A best view
     (``VIEW_ORDER[gate_a.vlm.best_view]`` — the FLUX condition camera, matching
     ``edits_2d/<id>_input.png``)

Outputs under ``edits_3d/<edit_id>/repaste_pad{P}/``:
  shape_slat.npz / tex_slat.npz     re-pasted latents (coords_new frame, denorm)
  after_view_<name>.png             single best-view render (sentinel)
  meta.json                         paste stats + view name

Multi-GPU: run one process per GPU with --slice I/N (objects round-robin by
index, del_add-style).  Resume: edits whose after_view PNG already exists are
skipped unless --force.

Usage (single worker):
  CUDA_VISIBLE_DEVICES=0 OPENCV_IO_ENABLE_OPENEXR=1 \\
    python scripts/repaste_pad0_batch.py --shard 00 --slice 0/8
Driver (8 GPUs): bash scripts/run_repaste_pad0_shard.sh 00
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
TRELLIS2_DIR = "/mnt/zsn/3dobject/TRELLIS.2"
sys.path.insert(0, TRELLIS2_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description="batch pad0 re-paste (gate-E pass only)")
    ap.add_argument("--root", default="data/Pxform_v2/prod_posthoc_no2dqc")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--mesh-root", default="data/partverse/inputs/mesh")
    ap.add_argument("--ckpt", default="/mnt/zsn/ckpts/TRELLIS.2-4B")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--no-subtract-preserved", action="store_true")
    ap.add_argument("--res", type=int, default=512, help="render resolution")
    ap.add_argument("--slice", default="0/1", help="I/N object round-robin slice")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    si, sn = (int(x) for x in args.slice.split("/"))
    logging.basicConfig(level=logging.INFO,
                        format=f"%(levelname)s [{args.shard}:{si}] %(message)s")
    log = logging.getLogger("repaste_batch")

    import numpy as np
    import torch

    from partcraft.pipeline_v3.trellis2_part_mask import (
        build_coord_bridge, downsample_edit_grid, part_edit_grid_64)
    from partcraft.render.ovox_views import VIEW_ORDER

    EDIT_RES = 512
    GRID = EDIT_RES // 16          # 32

    shard_dir = Path(args.root) / "objects" / args.shard
    obj_dirs = sorted(d for d in shard_dir.iterdir()
                      if (d / "edit_status.json").is_file())
    obj_dirs = [d for i, d in enumerate(obj_dirs) if i % sn == si]
    log.info("%d objects in slice %s", len(obj_dirs), args.slice)

    # collect work first so workers with an empty slice never load the pipeline
    work = []   # (obj_dir, edit_id, view_name)
    for od in obj_dirs:
        es = json.loads((od / "edit_status.json").read_text())
        for eid, e in (es.get("edits") or {}).items():
            stages = e.get("stages") or {}
            if (stages.get("gate_e") or {}).get("status") != "pass":
                continue
            if not (od / "edits_3d" / eid / "latents" / "ss.npz").is_file():
                continue
            bv = (((stages.get("gate_a") or {}).get("verdict") or {})
                  .get("vlm") or {}).get("best_view")
            if not (isinstance(bv, int) and 0 <= bv < len(VIEW_ORDER)):
                log.warning("%s/%s: gate_e pass but no valid best_view — skipped",
                            od.name, eid)
                continue
            view_name = VIEW_ORDER[bv]
            out_dir = od / "edits_3d" / eid / f"repaste_pad{args.pad}"
            if not args.force and (out_dir / f"after_view_{view_name}.png").is_file():
                continue
            work.append((od, eid, view_name))
    log.info("%d edits to process", len(work))
    if not work:
        return

    from PIL import Image
    import trellis2.modules.sparse as sp
    from partcraft.pipeline_v3 import trellis2_3d as T
    from partcraft.render import ovox_views as ov

    pipeline = T._ensure_pipeline(
        {"trellis2_codebase": TRELLIS2_DIR, "trellis2_ckpt": args.ckpt}, log)
    env = ov.load_envmap(f"{TRELLIS2_DIR}/assets/hdri/forest.exr")

    def _sparse(coords_np, feats_t):
        c = torch.from_numpy(np.asarray(coords_np)).int()
        coords = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.int32), c], 1).cuda()
        return sp.SparseTensor(feats=feats_t.float().cuda(), coords=coords)

    n_done = n_err = 0
    grid_cache: dict[tuple, "torch.Tensor"] = {}
    t_start = time.time()
    for od, eid, view_name in work:
        t0 = time.time()
        try:
            lat_dir = od / "edits_3d" / eid / "latents"
            ss = np.load(lat_dir / "ss.npz", allow_pickle=True)
            coords0 = torch.from_numpy(ss["coords0"].astype("int64"))
            coords_new = torch.from_numpy(ss["coords_new"].astype("int64"))
            parts = tuple(int(p) for p in ss["parts"])
            ashape = np.load(lat_dir / "shape_slat.npz")
            bshape = np.load(od / "p1_encode" / "shape_slat_e512.npz")
            # white-model objects (phase1 visibility.json white_model=true) have
            # NO tex latents by design — shape-only repaste + white decode.
            white = not (lat_dir / "tex_slat.npz").is_file()
            if white:
                vis = od / "phase1" / "visibility.json"
                if not (vis.is_file()
                        and json.loads(vis.read_text()).get("white_model")):
                    log.warning("%s/%s: tex_slat missing but NOT white_model — "
                                "skipped", od.name, eid)
                    n_err += 1
                    continue
            else:
                atex = np.load(lat_dir / "tex_slat.npz")
                btex = np.load(od / "p1_encode" / "tex_slat_e512.npz")
            if (bshape["feats"].shape[0] != coords0.shape[0]
                    or ashape["feats"].shape[0] != coords_new.shape[0]):
                log.warning("%s/%s: row mismatch — skipped", od.name, eid)
                n_err += 1
                continue

            key = (od.name, parts)
            grid32 = grid_cache.get(key)
            if grid32 is None:
                mesh_npz = Path(args.mesh_root) / args.shard / f"{od.name}.npz"
                grid64 = part_edit_grid_64(
                    mesh_npz, list(parts), pad=args.pad, canonical=True,
                    subtract_preserved=not args.no_subtract_preserved)
                grid32 = downsample_edit_grid(grid64, 64 // GRID).cuda()
                grid_cache[key] = grid32
            preserved, src_idx = build_coord_bridge(
                coords0.cuda(), coords_new.cuda(), grid32, grid=GRID)

            sh = torch.from_numpy(ashape["feats"]).float().cuda()
            sh[preserved] = torch.from_numpy(bshape["feats"]).float().cuda()[src_idx]

            out_dir = od / "edits_3d" / eid / f"repaste_pad{args.pad}"
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / "shape_slat.npz",
                                feats=sh.cpu().numpy(), coords=ss["coords_new"])
            if white:
                from partcraft.pipeline_v3.trellis2_white import (
                    build_white_model_mesh)
                mesh = build_white_model_mesh(
                    pipeline, _sparse(ashape["coords"], sh.cpu()),
                    log, res=EDIT_RES)
            else:
                tx = torch.from_numpy(atex["feats"]).float().cuda()
                tx[preserved] = torch.from_numpy(btex["feats"]).float().cuda()[src_idx]
                np.savez_compressed(out_dir / "tex_slat.npz",
                                    feats=tx.cpu().numpy(), coords=ss["coords_new"])
                mesh = pipeline.decode_latent(
                    _sparse(ashape["coords"], sh.cpu()),
                    _sparse(atex["coords"], tx.cpu()), EDIT_RES)[0]
            mesh.simplify(16_777_216)
            imgs = ov.render_sample(mesh, view_names=[view_name], envmap=env,
                                    resolution=args.res, bg=(1, 1, 1))
            (out_dir / "meta.json").write_text(json.dumps({
                "edit_id": eid, "parts": list(parts),
                "repaste_pad": args.pad, "view_name": view_name,
                "tokens_total": int(coords_new.shape[0]),
                "preserved": int(preserved.sum()),
                "white_model": white,
            }, indent=2))
            # PNG last — it is the sentinel, so a crash mid-edit re-runs cleanly
            Image.fromarray(imgs[view_name]).save(
                out_dir / f"after_view_{view_name}.png")
            n_done += 1
            if n_done % 25 == 0:
                rate = n_done / max(time.time() - t_start, 1e-6)
                log.info("%d/%d done (%.1f edit/min, last %s/%s %.1fs)",
                         n_done, len(work), rate * 60, od.name, eid,
                         time.time() - t0)
            torch.cuda.empty_cache()
        except Exception:
            n_err += 1
            log.exception("%s/%s failed", od.name, eid)
    log.info("slice %s finished: %d done, %d failed, %.1f min",
             args.slice, n_done, n_err, (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
