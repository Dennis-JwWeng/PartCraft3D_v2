#!/usr/bin/env python3
"""AFTER SS-decoded OCCUPANCY (coords_new), 1024 vs 512 — render + HTML in one.

The structure stage (S1) decodes the SS latent into the occupancy voxel set
``coords_new`` — the set of voxels that EXIST in the edited object.  The shape /
texture SLat then only fill those voxels.  So if an edit looks broken, the first
question is: **is the occupancy itself wrong** (holes, fragmentation, missing /
exploded part) or is it the SLat/decode on top?  This renders ``coords_new``
alone as solid voxels (edit region red, preserved body blue) from ``--nviews``
viewpoints, for both the 1024 (64³) and 512 (32³) edit-res trees, and stacks
them per edit in one standalone base64 HTML.

Unlike ``viz_edit_mask_3d.py`` (which renders ``coords0`` = the BEFORE shape,
colored by the edit mask), this renders ``coords_new`` = the AFTER occupancy.

    CUDA_VISIBLE_DEVICES=6 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
    /mnt/zsn/miniconda3/envs/trellis2/bin/python \
      scripts/viz/ab_after_occ_html.py            # masked (default)
      scripts/viz/ab_after_occ_html.py flowedit   # _exp_flowedit_free_r1024 vs _exp_flowedit_free_r512
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import base64
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
sys.path.insert(0, str(ROOT))
TRELLIS2_DIR = os.environ.get("TRELLIS2_DIR", "/mnt/zsn/3dobject/TRELLIS.2")
if TRELLIS2_DIR not in sys.path:
    sys.path.insert(0, TRELLIS2_DIR)

# recipe -> (1024 tree, 512 tree, title)
RECIPES = {
    "masked": dict(a="_exp_masked_posthoc_r1024", b="_exp_masked_posthoc_r512_pad0",
                   title="TRELLIS.2 masked-edit · AFTER SS occupancy (coords_new) · 1024 (64³) vs 512 (32³)"),
    "flowedit": dict(a="_exp_flowedit_free_r1024", b="_exp_flowedit_free_r512",
                     title="TRELLIS.2 FlowEdit-edit · AFTER SS occupancy (coords_new) · 1024 (64³) vs 512 (32³)"),
}

NVIEWS = 6
RES = 320
RED = np.array([0.90, 0.23, 0.23], np.float32)   # edit / newly-grown voxel
BLUE = np.array([0.24, 0.47, 0.90], np.float32)   # preserved body voxel


def _c3(a):
    a = np.asarray(a)
    return a[:, 1:] if a.shape[1] == 4 else a


def _grid_of(*arrs):
    cmax = 0
    for a in arrs:
        a = np.asarray(a)
        if a.size and a.ndim >= 2:
            c = a[:, 1:] if a.shape[1] == 4 else a
            cmax = max(cmax, int(c.max()))
    return 32 if cmax < 32 else 64


def _edit_dense(eg, g):
    eg = np.asarray(eg)
    if eg.ndim == 3:
        return eg.astype(bool)
    d = np.zeros((g, g, g), bool)
    e = eg.astype(int)
    d[e[:, 0], e[:, 1], e[:, 2]] = True
    return d


def b64_png(rgb_u8) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    recipe = sys.argv[1] if len(sys.argv) > 1 else "masked"
    cfg = RECIPES[recipe]
    A = ROOT / f"data/Pxform_v2/{cfg['a']}/objects/08"
    B = ROOT / f"data/Pxform_v2/{cfg['b']}/objects/08"
    out = ROOT / f"data/Pxform_v2/_scratch/ab_compare/after_occ_1024_vs_512_{recipe}.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from trellis2.utils import render_utils
    from trellis2.representations import Voxel

    def render_occ(npz_path: Path):
        """coords_new occupancy multiview tile + (#edit, #preserved, grid)."""
        ss = np.load(npz_path, allow_pickle=True)
        cn = _c3(ss["coords_new"]).astype(np.int32)
        g = _grid_of(ss["coords0"], ss["coords_new"], ss["edit_grid"])
        edit = _edit_dense(ss["edit_grid"], g)
        in_edit = edit[cn[:, 0], cn[:, 1], cn[:, 2]]
        col = np.where(in_edit[:, None], RED, BLUE).astype(np.float32)
        v = Voxel(origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / g,
                  coords=torch.from_numpy(np.ascontiguousarray(cn)).int().cuda(),
                  attrs=torch.from_numpy(np.ascontiguousarray(col)).float().cuda(),
                  layout={"color": slice(0, 3)}, device="cuda")
        snap = render_utils.render_snapshot(v, resolution=RES, r=2.0, fov=40.0, nviews=NVIEWS)
        frames = list(snap["color"] if "color" in snap else next(iter(snap.values())))
        h, w, _ = frames[0].shape
        strip = np.concatenate(frames, axis=1)
        return strip, int(in_edit.sum()), int((~in_edit).sum()), g

    # common edits present in both trees
    def edits_of(root):
        out = {}
        for obj in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            ep = obj / "edits_3d"
            if not ep.is_dir():
                continue
            for ed in sorted(ep.iterdir()):
                if (ed / "latents" / "ss.npz").is_file():
                    out[ed.name] = ed / "latents" / "ss.npz"
        return out

    ea, eb = edits_of(A), edits_of(B)
    eids = sorted(set(ea) & set(eb))

    blocks = []
    for eid in eids:
        sa, na_e, na_p, ga = render_occ(ea[eid])
        sb, nb_e, nb_p, gb = render_occ(eb[eid])
        blocks.append(f"""
        <div class="block">
          <div class="eid">{eid}</div>
          <div class="row"><span class="tag a">1024 · {ga}³<br>edit {na_e}<br>keep {na_p}</span><img src="{b64_png(sa)}"></div>
          <div class="row"><span class="tag b">512 · {gb}³<br>edit {nb_e}<br>keep {nb_p}</span><img src="{b64_png(sb)}"></div>
        </div>""")
        print(f"{eid}  1024:{na_e}+{na_p}@{ga}³  512:{nb_e}+{nb_p}@{gb}³")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{cfg['title']}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 12px}}
 .note{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:12px 14px;margin-bottom:18px;line-height:1.6;font-size:13px;max-width:1200px;border-left:3px solid #ffb454}}
 .note code{{background:#0d0f12;padding:1px 5px;border-radius:4px;color:#7fd1ff;font-size:12px}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}}
 .row{{display:flex;align-items:center;gap:8px;margin:3px 0}}
 .row img{{max-width:100%;height:auto;border-radius:3px;display:block;background:#000}}
 .tag{{font-size:10px;font-weight:600;text-align:right;line-height:1.4;min-width:64px;white-space:nowrap}}
 .tag.a{{color:#7fd1ff}} .tag.b{{color:#ffb454}}
</style></head><body>
<h1>{cfg['title']}</h1>
<div class="note">
<p>渲染的是 <b>S1 decode 出来的 AFTER 占据 <code>coords_new</code></b>（哪些 voxel 存在），
<span style="color:#e36">红=编辑/新长出</span>、<span style="color:#69f">蓝=保留 body</span>，
每行 {NVIEWS} 视角。看占据本身是否碎裂/空洞/塌缩——这是 SLat/decode 之前的问题源。
上 1024（64³），下 512（32³，max-pool 降采样后喂 S2）。</p>
<p style="color:#8a8f98">{len(eids)} edits · base64 内嵌单文件。</p></div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"\nwrote {out}  ({len(eids)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
