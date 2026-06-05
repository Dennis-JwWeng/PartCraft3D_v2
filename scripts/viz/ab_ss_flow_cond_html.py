#!/usr/bin/env python3
"""SS A/B viz WITH condition images — 64³ voxel, render + HTML in one.

Per edit, embeds (all base64, single file):
  * condition images:  INPUT (original, drives inversion) | EDITED (target, drives repaint)
  * SS-decode voxel of the CONDITION SOURCE (coords0 / before occupancy)  — shared
  * SS-decode voxel AFTER, T1 flow  (ss_flow_img_dit_L_16l8)
  * SS-decode voxel AFTER, T2 flow  (ss_flow_img_dit_1_3B_64)

All voxels rendered at the native 64³ grid (edit region red / body blue).
Reads scripts/experiments/ss_ab/{inputs,out/t1,out/t2}.

    CUDA_VISIBLE_DEVICES=6 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
    /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/viz/ab_ss_flow_cond_html.py
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
NVIEWS, RES, IMG = 4, 300, 300
RED = np.array([0.90, 0.23, 0.23], np.float32)
BLUE = np.array([0.24, 0.47, 0.90], np.float32)


def _c3(a):
    a = np.asarray(a)
    return (a[:, 1:] if a.ndim >= 2 and a.shape[1] == 4 else a).astype(np.int32)


def b64_png(rgb_u8) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb_u8).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_img(path) -> str:
    from PIL import Image
    try:
        im = Image.open(str(path)).convert("RGB")
        im.thumbnail((IMG, IMG))
        return b64_png(np.asarray(im))
    except Exception:
        return ""


def main() -> None:
    out = IO.parent / "ab_compare" / "ss_flow_cond_t1_vs_t2.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from trellis2.utils import render_utils
    from trellis2.representations import Voxel

    def strip(coords3, edit_dense):
        cn = _c3(coords3)
        ie = edit_dense[cn[:, 0], cn[:, 1], cn[:, 2]]
        col = np.where(ie[:, None], RED, BLUE).astype(np.float32)
        v = Voxel(origin=[-0.5, -0.5, -0.5], voxel_size=1.0 / 64,
                  coords=torch.from_numpy(np.ascontiguousarray(cn)).int().cuda(),
                  attrs=torch.from_numpy(np.ascontiguousarray(col)).float().cuda(),
                  layout={"color": slice(0, 3)}, device="cuda")
        snap = render_utils.render_snapshot(v, resolution=RES, r=2.0, fov=40.0, nviews=NVIEWS)
        fr = list(snap["color"] if "color" in snap else next(iter(snap.values())))
        return np.concatenate(fr, axis=1)

    t1d = {p.parent.name + "/" + p.stem: p for p in (IO / "out/t1").glob("*/*.npz")}
    t2d = {p.parent.name + "/" + p.stem: p for p in (IO / "out/t2").glob("*/*.npz")}
    keys = sorted(set(t1d) & set(t2d))

    blocks = []
    for k in keys:
        obj, eid = k.split("/")
        inp = np.load(IO / "inputs" / obj / f"{eid}.npz", allow_pickle=True)
        d1 = np.load(t1d[k], allow_pickle=True)
        d2 = np.load(t2d[k], allow_pickle=True)
        if "edit_grid64_dense" in inp.files:        # direct per-part voxelization (dense)
            edit = inp["edit_grid64_dense"].astype(bool)
        else:                                       # back-compat: idx form
            eg = _c3(inp["edit_grid64"]).astype(int)
            edit = np.zeros((64, 64, 64), bool)
            edit[eg[:, 0], eg[:, 1], eg[:, 2]] = True
        c0 = _c3(inp["coords0"]); c1 = _c3(d1["coords_new"]); c2 = _c3(d2["coords_new"])
        img_in = b64_img(str(inp["input_png"]))
        img_ed = b64_img(str(inp["edited_png"]))
        s0 = strip(c0, edit); s1 = strip(c1, edit); s2 = strip(c2, edit)
        blocks.append(f"""
        <div class="block">
          <div class="eid">{eid} <span class="obj">({obj[:10]})</span></div>
          <div class="grid">
            <div class="conds">
              <div class="cap">condition: INPUT (→invert)</div><img class="ci" src="{img_in}">
              <div class="cap">condition: EDITED (→repaint)</div><img class="ci" src="{img_ed}">
            </div>
            <div class="vox">
              <div class="row"><span class="tag g">SS decode<br>SOURCE<br>{c0.shape[0]}</span><img src="{b64_png(s0)}"></div>
              <div class="row"><span class="tag a">AFTER · T1<br>{c1.shape[0]}</span><img src="{b64_png(s1)}"></div>
              <div class="row"><span class="tag b">AFTER · T2<br>{c2.shape[0]}</span><img src="{b64_png(s2)}"></div>
            </div>
          </div>
        </div>""")
        print(f"{k}  source {c0.shape[0]}  T1 {c1.shape[0]}  T2 {c2.shape[0]}")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SS flow T1 vs T2 + condition</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 12px}}
 .note{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:12px 14px;margin-bottom:18px;line-height:1.6;font-size:13px;max-width:1200px;border-left:3px solid #ffb454}}
 .note code{{background:#0d0f12;padding:1px 5px;border-radius:4px;color:#7fd1ff;font-size:12px}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}} .obj{{color:#8a8f98;font-weight:400}}
 .grid{{display:flex;gap:14px;align-items:flex-start}}
 .conds{{flex:0 0 auto}} .cap{{font-size:10px;color:#aeb4bd;margin:2px 0}}
 .ci{{display:block;border-radius:4px;margin-bottom:6px;max-width:300px}}
 .vox{{flex:1 1 auto;min-width:0}}
 .row{{display:flex;align-items:center;gap:8px;margin:3px 0}}
 .row img{{max-width:100%;height:auto;border-radius:3px;display:block;background:#000}}
 .tag{{font-size:11px;font-weight:600;text-align:right;line-height:1.4;min-width:62px}}
 .tag.g{{color:#aeb4bd}} .tag.a{{color:#7fd1ff}} .tag.b{{color:#ffb454}}
</style></head><body>
<h1>TRELLIS.1 vs TRELLIS.2 — S1 flow with condition images (64³)</h1>
<div class="note">
<p>左列 = <b>条件图</b>(INPUT 驱动反演 / EDITED 驱动重绘);右列 = <b>SS-decode 的 64³ occupancy</b>:
<span style="color:#aeb4bd">SOURCE(条件源 / before)</span> →
<span style="color:#7fd1ff">AFTER T1 <code>ss_flow_img_dit_L_16l8</code></span> /
<span style="color:#ffb454">AFTER T2 <code>ss_flow_img_dit_1_3B_64</code></span>。
红=编辑区 / 蓝=保留 body(<b>按目标 part 的直接体素化上色</b>,连通、贴合 part;SOURCE 连通,AFTER 按原壳点查、移位重生 voxel 可能稀疏),每行 {NVIEWS} 视角,数字=voxel 数。
唯一变量=SS flow 模型(encode/decode/不膨胀硬 mask/统一调度/输入图全一致)。</p>
<p style="color:#8a8f98">{len(keys)} edits · base64 内嵌单文件。</p></div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"\nwrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
