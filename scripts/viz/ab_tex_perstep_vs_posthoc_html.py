#!/usr/bin/env python3
"""Texture anchor A/B: perstep vs posthoc (preserved-region colour fidelity).

Same pipeline (native T1-SS, pad4, restore, 512, shape=perstep), the ONLY
difference is the TEXTURE anchor:
  * perstep  — preserved tex tokens anchored every step to the inverted
               re-sampled reference (the body colour drifts under the edited
               image's global condition).
  * posthoc  — free generation under the edited image, then HARD-PASTE the
               original-image reference texture onto preserved tokens at the
               end (body colour locked, only the edit region regenerates).

    python scripts/viz/ab_tex_perstep_vs_posthoc_html.py
"""
from __future__ import annotations
import base64
import io as _io
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
DATA = ROOT / "data/Pxform_v2"
SHARD = "08"
VIEWS = ["front", "right", "back", "left", "down"]
TREES = [
    ("perstep tex (drift)",          "_exp_t1ss_native_r512_pad4_full",        "#ffb454"),
    ("posthoc tex (re-sample)",      "_exp_t1ss_native_r512_pad4_texposthoc",  "#7fd1ff"),
    ("posthoc-RESTORE (before enc)", "_exp_t1ss_native_r512_pad4_texrestore",  "#9be564"),
]


def b64(path: Path) -> str:
    return ("data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
            if path.is_file() else "")


def thumb(path: Path, px: int = 220) -> str:
    from PIL import Image
    if not path.is_file():
        return ""
    im = Image.open(str(path)).convert("RGB"); im.thumbnail((px, px))
    buf = _io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def strip(view_dir: Path, prefix: str) -> str:
    cells = "".join(f'<td><img src="{b64(view_dir / f"{prefix}_{v}.png")}"></td>' for v in VIEWS)
    return f"<table><tr>{cells}</tr></table>"


def main() -> None:
    # keys = edits present in the (smaller) restore test tree
    post = DATA / TREES[-1][1]
    keys = []
    for ed in sorted(post.glob(f"objects/{SHARD}/*/edits_3d/mod_*")):
        if ed.name == "latents" or not (ed / "after_view_front.png").is_file():
            continue
        keys.append((ed.parent.parent.name, ed.name))

    blocks = []
    for obj, eid in keys:
        od = post / "objects" / SHARD / obj
        ci = thumb(od / "edits_2d" / f"{eid}_input.png")
        ce = thumb(od / "edits_2d" / f"{eid}_edited.png")
        rows = ""
        gv = od / "gate_views"
        if (gv / "before_view_front.png").is_file():
            rows += (f'<div class="vr"><span class="tag" style="color:#aeb4bd">BEFORE (512 decode)</span>'
                     f'{strip(gv, "before_view")}</div>')
        for lbl, d, col in TREES:
            ed = DATA / d / "objects" / SHARD / obj / "edits_3d" / eid
            if (ed / "after_view_front.png").is_file():
                rows += (f'<div class="vr"><span class="tag" style="color:{col}">AFTER · {lbl}</span>'
                         f'{strip(ed, "after_view")}</div>')
        blocks.append(f"""
        <div class="block">
          <div class="eid">{eid} <span class="obj">({obj[:10]})</span></div>
          <div class="grid">
            <div class="conds">
              <div class="cap">INPUT</div><img class="ci" src="{ci}">
              <div class="cap">EDITED</div><img class="ci" src="{ce}">
            </div>
            <div class="vox">{rows}</div>
          </div>
        </div>""")

    legend = " · ".join(f'<b style="color:{c}">{lbl}</b>' for lbl, _, c in TREES)
    out = DATA / "_scratch" / "ab_compare" / "tex_perstep_vs_posthoc_vs_restore.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>tex perstep vs posthoc</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 6px}} .note{{font-size:13px;color:#aeb4bd;margin-bottom:16px;line-height:1.6}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}} .obj{{color:#8a8f98;font-weight:400}}
 .grid{{display:flex;gap:14px;align-items:flex-start}} .conds{{flex:0 0 auto}}
 .cap{{font-size:10px;color:#aeb4bd;margin:2px 0}} .ci{{display:block;border-radius:4px;margin-bottom:6px;max-width:200px}}
 .vox{{flex:1 1 auto;min-width:0}} .vr{{margin:3px 0}} .tag{{font-size:11px;font-weight:600;display:block;margin:2px 0}}
 table{{border-collapse:collapse}} td{{padding:2px}} img{{display:block;border-radius:3px;width:176px;background:#fff}}
</style></head><body>
<h1>TEXTURE anchor A/B · perstep vs posthoc · native T1-SS pad4 512</h1>
<div class="note">同管线(shape=perstep),唯一差别是纹理锚定方式。
<b style="color:#ffb454">perstep</b>:保留 token 每步锚到反演的重采样参考,编辑图全局条件让 body 颜色漂;
<b style="color:#9be564">posthoc</b>:编辑图下自由生成,终点把原图参考纹理硬贴回保留 token → body 颜色锁住,只编辑区重画。
对比保留区(非编辑部件)的颜色是否更接近 BEFORE。{legend}。{len(keys)} edits。</div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"wrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
