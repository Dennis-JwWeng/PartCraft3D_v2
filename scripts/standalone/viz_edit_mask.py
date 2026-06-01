#!/usr/bin/env python3
"""Visualize the S1 edit mask (+ shape latent) for every masked edit in a dir.

Purpose: eyeball whether the part mask is given correctly. For each edit that
saved structure latents (``latents/ss.npz``) we render, from the saved data
only (NO GPU, NO re-run), a panel with:

  row 0  CONTEXT     : 2D input | 2D edited | 3D after_shaded (what the edit is)
  row 1  S1 MASK     : XY/XZ/YZ projection of the edit region over the body —
                       green=body(preserved), red=edit-region-only(empty),
                       yellow=body INSIDE the edit region (what gets regenerated)
  row 2  coords_new  : XY/XZ/YZ of the SS-decoded occupancy —
                       blue=preserved voxel, red=edit/new voxel
  row 3  shape |feat|: XY/XZ/YZ heatmap of the shape-SLat L2 norm on coords_new

plus ``coverage`` = % of the body's voxels that fall inside the edit region.
A high coverage (≈ whole object) means the part mask is too coarse to be a
local edit (the masked S1 then regenerates ~everything and tears).

Also writes ``_INDEX.png`` sorting every edit by coverage so the bad masks
(near-whole-object) surface at the top.

  python scripts/standalone/viz_edit_mask.py \
    --in-root data/Pxform_v2/_rerun_v2/08 \
    --objects-root data/Pxform_v2/objects/08 \
    --out-dir data/Pxform_v2/_rerun_v2/08/_mask_viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

G = 64
SC = 5                                  # voxel upscale
AXES = [(2, "XY"), (1, "XZ"), (0, "YZ")]


def _dense_occ(coords, shape=(G, G, G)):
    c = coords.astype(int)
    v = np.zeros(shape, bool)
    v[c[:, 0], c[:, 1], c[:, 2]] = True
    return v


def _edit_grid_dense(ss):
    eg = np.asarray(ss["edit_grid"])
    if eg.ndim == 3:
        return eg.astype(bool)
    g = np.zeros((G, G, G), bool)
    e = eg.astype(int)
    g[e[:, 0], e[:, 1], e[:, 2]] = True
    return g


def _up(rgb):
    h, w, _ = rgb.shape
    return Image.fromarray(rgb).resize((w * SC, h * SC), Image.NEAREST)


def _panel_mask(coords0, edit, axis):
    body = _dense_occ(coords0).any(axis=axis)
    ed = edit.any(axis=axis)
    h, w = body.shape
    rgb = np.full((h, w, 3), 25, np.uint8)
    rgb[body & ~ed] = (40, 200, 40)
    rgb[ed & ~body] = (220, 40, 40)
    rgb[body & ed] = (240, 230, 40)
    return _up(rgb)


def _panel_newocc(coords_new, edit, axis):
    cn = _dense_occ(coords_new)
    c = coords_new.astype(int)
    inside = edit[c[:, 0], c[:, 1], c[:, 2]]
    new = np.zeros((G, G, G), bool)
    new[c[inside, 0], c[inside, 1], c[inside, 2]] = True
    pr = (cn & ~new).any(axis=axis)
    nw = new.any(axis=axis)
    h, w = pr.shape
    rgb = np.full((h, w, 3), 25, np.uint8)
    rgb[pr] = (60, 120, 230)
    rgb[nw] = (230, 60, 60)
    return _up(rgb)


def _panel_shape(coords_new, feats, axis):
    norm = np.linalg.norm(feats.astype(np.float32), axis=1)
    vol = np.full((G, G, G), -1.0, np.float32)
    c = coords_new.astype(int)
    np.maximum.at(vol, (c[:, 0], c[:, 1], c[:, 2]), norm)
    m = vol.max(axis=axis)
    occ = (vol > -1).any(axis=axis)
    lo, hi = np.percentile(norm, [2, 98])
    t = np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)
    r = np.clip(1.5 * t, 0, 1)
    g = np.clip(1.5 * (1 - abs(t - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - t), 0, 1)
    col = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    h, w = m.shape
    rgb = np.full((h, w, 3), 25, np.uint8)
    rgb[occ] = col[occ]
    return _up(rgb)


def _ctx(path, size):
    c = Image.new("RGB", (size, size), (245, 245, 245))
    try:
        im = Image.open(path).convert("RGB")
        if "after_shaded" in str(path):           # 2x2 grid → top-left view
            im = im.crop((0, 0, im.width // 2, im.height // 2))
        im.thumbnail((size, size))
        c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    except Exception:
        ImageDraw.Draw(c).text((6, size // 2), "-", fill=(150, 0, 0))
    return c


def _coverage(coords0, edit):
    c = coords0.astype(int)
    return 100.0 * edit[c[:, 0], c[:, 1], c[:, 2]].mean()


def viz_one(latents_dir, e2d_dir, after_png, title, out_png):
    ss = np.load(latents_dir / "ss.npz", allow_pickle=True)
    coords0 = np.asarray(ss["coords0"])
    coords_new = np.asarray(ss["coords_new"])
    edit = _edit_grid_dense(ss)
    shp = np.load(latents_dir / "shape_slat.npz", allow_pickle=True)
    shc = np.asarray(shp["coords"])
    shf = np.asarray(shp["feats"])
    if shc.shape[1] == 4:
        shc = shc[:, 1:]
    cov = _coverage(coords0, edit)

    voxel_rows = [
        ("S1 mask  grn=body red=edit ylw=overlap",
         [_panel_mask(coords0, edit, ax) for ax, _ in AXES]),
        ("coords_new  blu=preserved red=new",
         [_panel_newocc(coords_new, edit, ax) for ax, _ in AXES]),
        ("shape |feat|  blue=low red=high",
         [_panel_shape(shc, shf, ax) for ax, _ in AXES]),
    ]
    cw, ch = voxel_rows[0][1][0].size
    eid = latents_dir.parent.name
    ctx_imgs = [
        ("2D input", e2d_dir / f"{eid}_input.png"),
        ("2D edited", e2d_dir / f"{eid}_edited.png"),
        ("3D after", after_png),
    ]
    ctx = [(lbl, _ctx(p, ch)) for lbl, p in ctx_imgs]

    pad, lab, hdr = 8, 165, 40
    W = lab + 3 * cw + 4 * pad
    H = hdr + 4 * (ch + 24) + 2 * pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    flag = "  <<< MASK ~ WHOLE OBJECT" if cov >= 60 else ""
    d.text((pad, 8), f"{title}   coverage={cov:.1f}%{flag}", fill=(0, 0, 0))
    # context row
    y = hdr
    d.text((pad, y + ch // 2), "context", fill=(0, 0, 0))
    for ci, (lbl, im) in enumerate(ctx):
        x = lab + pad + ci * (cw + pad)
        sheet.paste(im, (x, y))
        d.text((x + 4, y - 14), lbl, fill=(0, 0, 0))
    # voxel rows
    for ri, (rlabel, panels) in enumerate(voxel_rows):
        y = hdr + (ri + 1) * (ch + 24)
        d.text((pad, y + ch // 2), rlabel, fill=(0, 0, 0))
        for ci, (ax, axname) in enumerate(AXES):
            x = lab + pad + ci * (cw + pad)
            sheet.paste(panels[ci], (x, y))
            d.text((x + 4, y - 14), axname, fill=(0, 0, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True,
                    help="dir of <obj>/<edit_id>/latents/ss.npz")
    ap.add_argument("--objects-root", default=None,
                    help="objects/<shard> dir for the 2D edits_2d images "
                         "(default: infer data/Pxform_v2/objects/<shard>)")
    ap.add_argument("--edits-2d-subdir", default="edits_2d")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    shard = in_root.name
    obj_root = (Path(args.objects_root) if args.objects_root
                else in_root.parents[1] / "objects" / shard)
    out_dir = Path(args.out_dir) if args.out_dir else in_root / "_mask_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for obj in sorted(p for p in in_root.iterdir()
                      if p.is_dir() and not p.name.startswith("_")):
        for ed in sorted(obj.iterdir()):
            ld = ed / "latents"
            if not (ld / "ss.npz").is_file():
                continue
            e2d = obj_root / obj.name / args.edits_2d_subdir
            after = ed / "after_shaded.png"
            ss = np.load(ld / "ss.npz", allow_pickle=True)
            et = str(ss["edit_type"]) if "edit_type" in ss.keys() else "?"
            title = f"{obj.name[:8]}  {ed.name.replace(obj.name, '').strip('_')}  [{et}]"
            cov = viz_one(ld, e2d, after, title, out_dir / f"{ed.name}.png")
            rows.append((cov, obj.name, ed.name, et, e2d, after))
            print(f"{cov:6.1f}%  {ed.name}")

    # index sheet sorted by coverage
    rows.sort(reverse=True)
    TH = 200
    cell_h = TH + 40
    cols = 1
    W = TH * 3 + 4 * 10
    H = 36 + len(rows) * (cell_h + 8)
    idx = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(idx)
    d.text((10, 10), f"{shard}: {len(rows)} edits by coverage  "
           f"[2D edited | S1 mask XY | 3D after]   >=60% = mask~whole object",
           fill=(0, 0, 0))
    for i, (cov, obj, eid, et, e2d, after) in enumerate(rows):
        y = 36 + i * (cell_h + 8)
        ss = np.load(in_root / obj / eid / "latents" / "ss.npz", allow_pickle=True)
        mask_xy = _panel_mask(np.asarray(ss["coords0"]),
                              _edit_grid_dense(ss), 2).resize((TH, TH))
        idx.paste(_ctx(e2d / f"{eid}_edited.png", TH), (10, y))
        idx.paste(mask_xy, (10 + TH + 10, y))
        idx.paste(_ctx(after, TH), (10 + 2 * (TH + 10), y))
        col = (200, 0, 0) if cov >= 60 else (0, 0, 0)
        d.text((10, y + TH + 4),
               f"{cov:5.1f}%  {obj[:8]} {eid.replace(obj,'').strip('_')} [{et}]",
               fill=col)
    idx.save(out_dir / "_INDEX.png")
    print(f"\nwrote {len(rows)} panels + _INDEX.png → {out_dir}")


if __name__ == "__main__":
    main()
