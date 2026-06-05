#!/usr/bin/env python3
"""Results gallery for the native T1-SS pad4 FULL-texture run, embedded HTML.

One row per edit: INPUT + EDITED 2D conditions, then the BEFORE named views
(decoded original latents) and the AFTER named views (decoded post-edit,
textured 512 mesh).  S1 = TRELLIS.1 SS flow in-process (native), S2 shape+tex
both masked per-step, pad4 + restore, edit_res=512, textured (no white-model).

    python scripts/viz/t1native_pad4_full_results_html.py
"""
from __future__ import annotations
import base64
import io as _io
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
DATA = ROOT / "data/Pxform_v2"
TREE = DATA / "_exp_t1ss_native_r512_pad4_full"
SHARD = "08"
VIEWS = ["front", "right", "back", "left", "down"]


def b64(path: Path) -> str:
    return ("data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
            if path.is_file() else "")


def thumb(path: Path, px: int = 240) -> str:
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
    keys = []
    for ed in sorted(TREE.glob(f"objects/{SHARD}/*/edits_3d/mod_*")):
        if ed.name == "latents" or not (ed / "after_view_front.png").is_file():
            continue
        keys.append((ed.parent.parent.name, ed.name, ed))

    blocks = []
    for obj, eid, ed in keys:
        od = TREE / "objects" / SHARD / obj
        ci = thumb(od / "edits_2d" / f"{eid}_input.png")
        ce = thumb(od / "edits_2d" / f"{eid}_edited.png")
        rows = ""
        gv = od / "gate_views"
        if (gv / "before_view_front.png").is_file():
            rows += (f'<div class="vr"><span class="tag" style="color:#aeb4bd">BEFORE (512 decode)</span>'
                     f'{strip(gv, "before_view")}</div>')
        rows += (f'<div class="vr"><span class="tag" style="color:#9be564">AFTER (textured, native T1-SS pad4)</span>'
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

    out = DATA / "_scratch" / "ab_compare" / "t1native_pad4_full_results.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>T1-SS native pad4 full-texture results</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 6px}} .note{{font-size:13px;color:#aeb4bd;margin-bottom:16px;line-height:1.6}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}} .obj{{color:#8a8f98;font-weight:400}}
 .grid{{display:flex;gap:14px;align-items:flex-start}} .conds{{flex:0 0 auto}}
 .cap{{font-size:10px;color:#aeb4bd;margin:2px 0}} .ci{{display:block;border-radius:4px;margin-bottom:6px;max-width:220px}}
 .vox{{flex:1 1 auto;min-width:0}} .vr{{margin:3px 0}} .tag{{font-size:11px;font-weight:600;display:block;margin:2px 0}}
 table{{border-collapse:collapse}} td{{padding:2px}} img{{display:block;border-radius:3px;width:180px;background:#fff}}
</style></head><body>
<h1>TRELLIS.1-SS native · pad4 + restore · 512 · FULL texture</h1>
<div class="note">S1 = TRELLIS.1 SS flow + DINOv2(进程内 native);S2 shape <b>和</b> texture 都 masked per-step;
edit_res=512;无 white-model → 带纹理 512 mesh。视图顺序:front · right · back · left · down。共 {len(keys)} edits。</div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"wrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
