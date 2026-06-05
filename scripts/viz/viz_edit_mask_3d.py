#!/usr/bin/env python3
"""3D voxel visualization of the S1 edit mask (+ shape latent) for masked edits.

Renders each edit's SAVED occupancy as colored 3D voxels with TRELLIS's own
VoxelRenderer (GPU — but only the renderer, NO 16GB pipeline load), from
``--nviews`` viewpoints, so you can judge IN 3D whether the part mask targets
the right region (vs the orthographic-projection viz which collapses depth):

  * mask-on-body : coords0      — green = preserved body,
                                  red   = body voxel INSIDE the edit region
  * coords_new   : SS-decoded   — blue  = preserved voxel,
                                  red   = edit / newly-grown voxel
  * shape |feat| : coords_new colored by shape-SLat L2 norm (blue=low->red=high)

Panel per edit: [ 2D edited | mask 3D | coords_new 3D | shape 3D ] (each a
2x2 multiview tile) + coverage %.  Also writes ``_INDEX3D.png`` (2D edited +
mask-3D, sorted by coverage).  Reads only the saved latents — no re-run.

  CUDA_VISIBLE_DEVICES=6 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
  /mnt/zsn/miniconda3/envs/trellis2/bin/python \
    scripts/viz/viz_edit_mask_3d.py \
      --in-root data/Pxform_v2/_rerun_v2/08 --objs all
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
TRELLIS2_DIR = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
if TRELLIS2_DIR not in sys.path:
    sys.path.insert(0, TRELLIS2_DIR)

G = 64  # default / legacy 1024-edit grid; per-edit grid is auto-detected below


def _grid_of(*arrs):
    """Detect the SLat coord grid (32 for 512-edit-res, 64 for 1024) from coords.

    coords live on ``res//16``: 1024->64³ (coords 0..63), 512->32³ (0..31).
    """
    cmax = 0
    for a in arrs:
        a = np.asarray(a)
        if a.size and a.ndim >= 2:
            c = a[:, 1:] if a.shape[1] == 4 else a
            cmax = max(cmax, int(c.max()))
    return 32 if cmax < 32 else 64


def _edit_grid_dense(ss, g_res=G):
    eg = np.asarray(ss["edit_grid"])
    if eg.ndim == 3:
        return eg.astype(bool)
    g = np.zeros((g_res, g_res, g_res), bool)
    e = eg.astype(int)
    g[e[:, 0], e[:, 1], e[:, 2]] = True
    return g


def _c3(a):
    a = np.asarray(a)
    return a[:, 1:] if a.shape[1] == 4 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--objs", default="all", help="'all' or comma list of obj ids")
    ap.add_argument("--objects-root", default=None)
    ap.add_argument("--edits-2d-subdir", default="edits_2d")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--nviews", type=int, default=4)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--r", type=float, default=2.0)
    ap.add_argument("--fov", type=float, default=40.0)
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from trellis2.utils import render_utils
    from trellis2.representations import Voxel

    in_root = Path(args.in_root)
    shard = in_root.name
    obj_root = (Path(args.objects_root) if args.objects_root
                else in_root.parents[1] / "objects" / shard)
    out_dir = Path(args.out_dir) if args.out_dir else in_root / "_mask_viz_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    def vox(coords3, colors01, res=G):
        c = torch.from_numpy(np.ascontiguousarray(coords3)).int().cuda()
        col = torch.from_numpy(np.ascontiguousarray(colors01)).float().cuda()
        return Voxel(origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / res,
                     coords=c, attrs=col, layout={"color": slice(0, 3)},
                     device="cuda")

    def render4(v):
        snap = render_utils.render_snapshot(
            v, resolution=args.res, r=args.r, fov=args.fov, nviews=args.nviews)
        frames = snap["color"] if "color" in snap else next(iter(snap.values()))
        return list(frames)

    def tile2x2(frames):
        f = frames[:4]
        while len(f) < 4:
            f.append(np.zeros_like(f[0]))
        h, w, _ = f[0].shape
        canvas = np.full((2 * h, 2 * w, 3), 18, np.uint8)
        for i, fr in enumerate(f):
            r, c = divmod(i, 2)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
        return canvas

    def ctx(path, size):
        im0 = Image.new("RGB", (size, size), (245, 245, 245))
        try:
            im = Image.open(path).convert("RGB")
            if "after_shaded" in str(path):
                im = im.crop((0, 0, im.width // 2, im.height // 2))
            im.thumbnail((size, size))
            im0.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        except Exception:
            ImageDraw.Draw(im0).text((6, size // 2), "-", fill=(150, 0, 0))
        return im0

    def pca_rgb(feats):
        """Project [N,C] latents to RGB via top-3 PCA (shows latent structure)."""
        X = feats.astype(np.float32)
        X = X - X.mean(0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            proj = X @ Vt[:3].T
        except np.linalg.LinAlgError:
            proj = X[:, :3]
        lo = np.percentile(proj, 2, axis=0)
        hi = np.percentile(proj, 98, axis=0)
        return np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)

    objs = sorted(p for p in in_root.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))
    if args.objs != "all":
        keep = {o.strip() for o in args.objs.split(",")}
        objs = [p for p in objs if p.name in keep]

    GREEN = np.array([0.16, 0.78, 0.16], np.float32)
    RED = np.array([0.90, 0.23, 0.23], np.float32)
    BLUE = np.array([0.24, 0.47, 0.90], np.float32)

    index = []
    for obj in objs:
        # current pipeline nests edits under <obj>/edits_3d/; legacy trees put
        # them directly under <obj>/.  Support both.
        edits_parent = obj / "edits_3d" if (obj / "edits_3d").is_dir() else obj
        for ed in sorted(edits_parent.iterdir()):
            if not ed.is_dir():
                continue
            ld = ed / "latents"
            if not (ld / "ss.npz").is_file():
                continue
            ss = np.load(ld / "ss.npz", allow_pickle=True)
            coords0 = _c3(ss["coords0"]).astype(np.int32)
            coords_new = _c3(ss["coords_new"]).astype(np.int32)
            g_res = _grid_of(coords0, coords_new, ss["edit_grid"])  # 32 (512) or 64 (1024)
            blk = g_res // 16                                       # //4 @64³, //2 @32³
            edit = _edit_grid_dense(ss, g_res)
            keep16 = np.asarray(ss["keep16"]).astype(bool)     # [16,16,16]
            et = str(ss["edit_type"]) if "edit_type" in ss.keys() else "?"

            in0 = edit[coords0[:, 0], coords0[:, 1], coords0[:, 2]]
            cov = 100.0 * in0.mean()

            # ── STAGE 1: SS mask @16^3 — the actual resolution the SS flow masks.
            #    Render the object's occupied 16^3 blocks, colored by keep16.
            b16 = np.unique(coords0 // blk, axis=0).astype(np.int32)
            keep_b = keep16[b16[:, 0], b16[:, 1], b16[:, 2]]    # True=preserve
            col16 = np.where(keep_b[:, None], GREEN, RED)

            # ── STAGE 2: SLat mask @64^3 on the BEFORE shape (coords0).
            #    The S2 preserve reference is the ORIGINAL SLat: the inversion is
            #    done on coords0 and preserved tokens are seeded FROM coords0, so
            #    the edit/preserve mask lives on the before shape (the edit region
            #    is defined on the original geometry). green=keep, red=edit — the
            #    same region the 16^3 S1 mask coarsens, just at full 64^3.
            colS2 = np.where(in0[:, None], RED, GREEN)

            # ── STAGE 2 features @64^3 (PCA->RGB of the 32-dim latents) ──
            shp = np.load(ld / "shape_slat.npz", allow_pickle=True)
            shc = _c3(shp["coords"]).astype(np.int32)
            col_shape = pca_rgb(np.asarray(shp["feats"]))
            tex_p = ld / "tex_slat.npz"
            if tex_p.is_file():
                tx = np.load(tex_p, allow_pickle=True)
                txc = _c3(tx["coords"]).astype(np.int32)
                col_tex = pca_rgb(np.asarray(tx["feats"]))
                t_tex = tile2x2(render4(vox(txc, col_tex, res=g_res)))
            else:
                t_tex = None

            t_s1 = tile2x2(render4(vox(b16, col16, res=16)))
            t_s2 = tile2x2(render4(vox(coords0, colS2, res=g_res)))
            t_shp = tile2x2(render4(vox(shc, col_shape, res=g_res)))

            e2d = obj_root / obj.name / args.edits_2d_subdir
            sz = t_s1.shape[0]
            pad, hdr = 8, 30
            tex_img = (Image.fromarray(t_tex) if t_tex is not None
                       else ctx(None, sz))
            cells = [ctx(e2d / f"{ed.name}_input.png", sz),
                     ctx(e2d / f"{ed.name}_edited.png", sz),
                     Image.fromarray(t_s1), Image.fromarray(t_s2),
                     Image.fromarray(t_shp), tex_img]
            labels = ["2D original", "2D edited",
                      "S1 mask @16^3 (before) grn=keep red=edit",
                      f"S2 mask @{g_res}^3 (before) grn=keep red=edit",
                      f"shape SLat PCA @{g_res}^3 (after)",
                      f"tex SLat PCA @{g_res}^3 (after)" + ("" if t_tex is not None else " (white-model: none)")]
            W = len(cells) * sz + (len(cells) + 1) * pad
            H = hdr + sz + 22 + pad
            sheet = Image.new("RGB", (W, H), (255, 255, 255))
            d = ImageDraw.Draw(sheet)
            flag = "  <<< MASK ~ WHOLE OBJECT" if cov >= 60 else ""
            d.text((pad, 8), f"{obj.name[:8]} {ed.name.replace(obj.name,'').strip('_')} "
                   f"[{et}]  grid={g_res}^3  coverage={cov:.1f}%{flag}", fill=(0, 0, 0))
            for i, (im, lb) in enumerate(zip(cells, labels)):
                x = pad + i * (sz + pad)
                sheet.paste(im, (x, hdr))
                d.text((x + 4, hdr + sz + 4), lb, fill=(20, 20, 20))
            sheet.save(out_dir / f"{ed.name}.png")
            index.append((cov, obj.name, ed.name, et,
                          e2d / f"{ed.name}_edited.png", t_s2))
            print(f"{cov:6.1f}%  {ed.name}")

    # index sheet: 2D edited + mask-3D, sorted by coverage
    index.sort(key=lambda r: -r[0])
    TH = 220
    W = TH * 2 + 30
    H = 34 + len(index) * (TH + 30)
    idx = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(idx)
    d.text((10, 10), f"{shard}: {len(index)} edits by coverage  "
           f"[2D edited | mask 3D]  >=60% = mask~whole object", fill=(0, 0, 0))
    for i, (cov, obj, eid, et, ep, t_mask) in enumerate(index):
        y = 34 + i * (TH + 30)
        idx.paste(ctx(ep, TH), (10, y))
        mk = Image.fromarray(t_mask); mk.thumbnail((TH, TH))
        idx.paste(mk, (10 + TH + 10, y))
        col = (200, 0, 0) if cov >= 60 else (0, 0, 0)
        d.text((10, y + TH + 6),
               f"{cov:5.1f}%  {obj[:8]} {eid.replace(obj,'').strip('_')} [{et}]",
               fill=col)
    idx.save(out_dir / "_INDEX3D.png")
    print(f"\nwrote {len(index)} panels + _INDEX3D.png -> {out_dir}")


if __name__ == "__main__":
    main()
