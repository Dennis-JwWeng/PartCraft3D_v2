#!/usr/bin/env python3
"""SS A/B viz — TRELLIS.1 vs TRELLIS.2 S1 occupancy, render + HTML in one.

Reads the decoded occupancy from scripts/experiments/ss_ab/{run_t1,run_t2}.py
(out/t1, out/t2) and the shared original occupancy from out/inputs, and for each
edit renders 3 occupancy strips (multiview voxels, edit region red / body blue):

    BEFORE        (coords0, original occupancy)
    T1  ss_flow_img_dit_L_16l8   (after)
    T2  ss_flow_img_dit_1_3B_64  (after)

Everything else (encode, decode, no-dilation hard mask, schedule, input images)
is identical between T1 and T2 — so any difference IS the SS flow model.
One standalone base64 HTML.

    CUDA_VISIBLE_DEVICES=6 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
    /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/viz/ab_ss_flow_html.py
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

IO = ROOT / "data/Pxform_v2/_scratch/ss_ab"
NVIEWS, RES = 6, 320
RED = np.array([0.90, 0.23, 0.23], np.float32)
BLUE = np.array([0.24, 0.47, 0.90], np.float32)
GREY = np.array([0.62, 0.66, 0.72], np.float32)


def _c3(a):
    a = np.asarray(a)
    return (a[:, 1:] if a.ndim >= 2 and a.shape[1] == 4 else a).astype(np.int32)


def b64_png(rgb_u8) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    out = IO.parent / "ab_compare" / "ss_flow_t1_vs_t2.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from trellis2.utils import render_utils
    from trellis2.representations import Voxel

    def strip(coords3, edit_dense, color_edit=True):
        cn = _c3(coords3)
        if color_edit and edit_dense is not None:
            ie = edit_dense[cn[:, 0], cn[:, 1], cn[:, 2]]
            col = np.where(ie[:, None], RED, BLUE).astype(np.float32)
        else:
            col = np.tile(GREY, (cn.shape[0], 1)).astype(np.float32)
        v = Voxel(origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / 64,
                  coords=torch.from_numpy(np.ascontiguousarray(cn)).int().cuda(),
                  attrs=torch.from_numpy(np.ascontiguousarray(col)).float().cuda(),
                  layout={"color": slice(0, 3)}, device="cuda")
        snap = render_utils.render_snapshot(v, resolution=RES, r=2.0, fov=40.0, nviews=NVIEWS)
        fr = list(snap["color"] if "color" in snap else next(iter(snap.values())))
        return np.concatenate(fr, axis=1)

    # edits present in BOTH t1 and t2 outputs
    t1d = {p.parent.name + "/" + p.stem: p for p in (IO / "out/t1").glob("*/*.npz")}
    t2d = {p.parent.name + "/" + p.stem: p for p in (IO / "out/t2").glob("*/*.npz")}
    keys = sorted(set(t1d) & set(t2d))

    blocks = []
    for k in keys:
        obj, eid = k.split("/")
        d1 = np.load(t1d[k], allow_pickle=True)
        d2 = np.load(t2d[k], allow_pickle=True)
        eg = d1["edit_grid64"].astype(int)
        edit = np.zeros((64, 64, 64), bool)
        edit[eg[:, 0], eg[:, 1], eg[:, 2]] = True
        c0 = _c3(d1["coords0"]); c1 = _c3(d1["coords_new"]); c2 = _c3(d2["coords_new"])
        s0 = strip(c0, edit); s1 = strip(c1, edit); s2 = strip(c2, edit)
        blocks.append(f"""
        <div class="block">
          <div class="eid">{eid} <span class="obj">({obj[:10]})</span></div>
          <div class="row"><span class="tag g">BEFORE<br>{c0.shape[0]}</span><img src="{b64_png(s0)}"></div>
          <div class="row"><span class="tag a">T1 flow<br>{c1.shape[0]}</span><img src="{b64_png(s1)}"></div>
          <div class="row"><span class="tag b">T2 flow<br>{c2.shape[0]}</span><img src="{b64_png(s2)}"></div>
        </div>""")
        print(f"{k}  before {c0.shape[0]}  T1 {c1.shape[0]}  T2 {c2.shape[0]}")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SS flow T1 vs T2</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 12px}}
 .note{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:12px 14px;margin-bottom:18px;line-height:1.6;font-size:13px;max-width:1200px;border-left:3px solid #ffb454}}
 .note code{{background:#0d0f12;padding:1px 5px;border-radius:4px;color:#7fd1ff;font-size:12px}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}} .obj{{color:#8a8f98;font-weight:400}}
 .row{{display:flex;align-items:center;gap:8px;margin:3px 0}}
 .row img{{max-width:100%;height:auto;border-radius:3px;display:block;background:#000}}
 .tag{{font-size:11px;font-weight:600;text-align:right;line-height:1.4;min-width:62px}}
 .tag.g{{color:#aeb4bd}} .tag.a{{color:#7fd1ff}} .tag.b{{color:#ffb454}}
</style></head><body>
<h1>TRELLIS.1 vs TRELLIS.2 — S1 sparse-structure flow (fair A/B)</h1>
<div class="note">
<p>只跑 <b>S1(占据)</b>。<b>唯一变量 = SS flow 模型</b>:
<span style="color:#7fd1ff">T1 <code>ss_flow_img_dit_L_16l8</code></span> vs
<span style="color:#ffb454">T2 <code>ss_flow_img_dit_1_3B_64</code></span>(各带自己的 image cond)。
encode/decode(共用 <code>ss_*_conv3d_16l8</code> VAE)、<b>不膨胀硬 mask(pad=0)</b>、
统一调度(steps25/cfg5/[.5,1]/rt3)、输入图(input→反演 / edited→重绘)全部一致。
红=编辑区 / 蓝=保留 body,每行 {NVIEWS} 视角。数字 = 占据 voxel 数。</p>
<p style="color:#8a8f98">{len(keys)} edits · base64 内嵌单文件。</p></div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"\nwrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
