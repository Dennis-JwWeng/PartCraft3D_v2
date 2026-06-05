#!/usr/bin/env python3
"""SS A/B — S2 shape-only decode, TRELLIS.1-S1 vs TRELLIS.2-S1, embedded HTML.

Embeds the white-model meshes rendered by run_s2_shape.py (no GPU needed).  For
each edit: condition images + T1-S1→S2 white mesh (5 views) + T2-S1→S2 white
mesh (5 views).  Both went through the SAME T2 S2 shape recipe (perstep mask,
res=1024) — so the geometry difference reflects only the upstream S1 occupancy
(TRELLIS.1 vs TRELLIS.2 SS flow).

    python scripts/viz/ab_s2_shape_html.py
"""
from __future__ import annotations
import base64
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
IO = ROOT / "data/Pxform_v2/_scratch/ss_ab"
VIEWS = ["front", "right", "back", "left", "down"]
IMG = 300


def b64(path: Path) -> str:
    if not path.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def b64_thumb(path) -> str:
    from PIL import Image
    import io as _io
    try:
        im = Image.open(str(path)).convert("RGB"); im.thumbnail((IMG, IMG))
        buf = _io.BytesIO(); im.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def main() -> None:
    out = IO.parent / "ab_compare" / "s2_shape_t1_vs_t2.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    def complete(p):  # require the rendered views (skip empty dirs from a failed render)
        return p.is_dir() and (p / "mesh_view_front.png").is_file()
    t1 = {p.parent.name + "/" + p.name: p for p in (IO / "s2_shape/t1").glob("*/*") if complete(p)}
    t2 = {p.parent.name + "/" + p.name: p for p in (IO / "s2_shape/t2").glob("*/*") if complete(p)}
    keys = sorted(set(t1) & set(t2))

    def strip(d: Path) -> str:
        cells = "".join(
            f'<td><img src="{b64(d / f"mesh_view_{v}.png")}"></td>' for v in VIEWS)
        return f"<table><tr>{cells}</tr></table>"

    blocks = []
    for k in keys:
        obj, eid = k.split("/")
        inp = np.load(IO / "inputs" / obj / f"{eid}.npz", allow_pickle=True)
        ci = b64_thumb(str(inp["input_png"])); ce = b64_thumb(str(inp["edited_png"]))
        blocks.append(f"""
        <div class="block">
          <div class="eid">{eid} <span class="obj">({obj[:10]})</span></div>
          <div class="grid">
            <div class="conds">
              <div class="cap">INPUT</div><img class="ci" src="{ci}">
              <div class="cap">EDITED</div><img class="ci" src="{ce}">
            </div>
            <div class="vox">
              <div class="vr"><span class="tag a">S1=T1 → S2 shape</span>{strip(t1[k])}</div>
              <div class="vr"><span class="tag b">S1=T2 → S2 shape</span>{strip(t2[k])}</div>
            </div>
          </div>
        </div>""")

    head = "".join(f"<th>{v}</th>" for v in VIEWS)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>S2 shape T1 vs T2</title>
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
 .vr{{margin:3px 0}} .tag{{font-size:11px;font-weight:600;display:block;margin:2px 0}}
 .tag.a{{color:#7fd1ff}} .tag.b{{color:#ffb454}}
 table{{border-collapse:collapse}} td{{padding:2px}} img{{display:block;border-radius:3px;width:200px;background:#fff}}
</style></head><body>
<h1>S2 shape-only decode — S1=TRELLIS.1 vs S1=TRELLIS.2 (same T2 S2 recipe)</h1>
<div class="note">
<p>两侧用<b>同一套 T2 S2 shape recipe</b>(<code>anchor=perstep</code> 只带 mask、shape SLat @1024、白模 decode 跳过 PBR),
唯一差别 = 上游 <b>S1 占据来自 T1 还是 T2 的 SS flow</b>。
<span style="color:#7fd1ff">蓝标=S1 用 TRELLIS.1</span> / <span style="color:#ffb454">橙标=S1 用 TRELLIS.2</span>。
灰白模反映纯几何(无纹理)。</p>
<p style="color:#8a8f98">{len(keys)} edits · base64 内嵌单文件。</p></div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"wrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
