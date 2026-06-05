#!/usr/bin/env python3
"""512 T1-SS dilation-pad sweep — pad3 / pad4 / pad5 / pad7, embedded HTML.

For each (obj, edit) shows: condition input/edited + BEFORE 5 views + the
white-model S2-shape AFTER 5 views for each pad tree.  ALL trees use the
IDENTICAL recipe — TRELLIS.1 SS flow (via the offline bridge) + same-frame 64³
restore + perstep S2 + edit_res=512 + force_white_model.  The ONLY variable is
the 64³ edit-grid Chebyshev dilation pad (3/4/5/7), so geometry differences
isolate how much the dilation grows / heals the edited part.

    python scripts/viz/ab_pad_t1ss_html.py
"""
from __future__ import annotations
import base64
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
DATA = ROOT / "data/Pxform_v2"
VIEWS = ["front", "right", "back", "left", "down"]
TREES = [
    ("pad3", "_exp_t1ss_perstep_r512_pad3_restore", "#7fd1ff"),
    ("pad4", "_exp_t1ss_perstep_r512_pad4_restore", "#9be564"),
    ("pad5", "_exp_t1ss_perstep_r512_pad5_restore", "#ffb454"),
    ("pad7", "_exp_t1ss_perstep_r512_pad7_restore", "#ff7a90"),
]
SHARD = "08"


def b64(path: Path) -> str:
    return ("data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
            if path.is_file() else "")


def thumb(path: Path, px: int = 260) -> str:
    from PIL import Image
    import io as _io
    if not path.is_file():
        return ""
    im = Image.open(str(path)).convert("RGB"); im.thumbnail((px, px))
    buf = _io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def strip(view_dir: Path, prefix: str) -> str:
    cells = "".join(f'<td><img src="{b64(view_dir / f"{prefix}_{v}.png")}"></td>' for v in VIEWS)
    return f"<table><tr>{cells}</tr></table>"


def main() -> None:
    present = [(lbl, DATA / d, c) for lbl, d, c in TREES if (DATA / d).is_dir()]
    keys = set()
    for _, tdir, _ in present:
        for ed in tdir.glob(f"objects/{SHARD}/*/edits_3d/*"):
            if (ed / "after_view_front.png").is_file():
                keys.add((ed.parent.parent.name, ed.name))
    keys = sorted(keys)

    blocks = []
    for obj, eid in keys:
        cond_in = cond_ed = before_dir = None
        for _, tdir, _ in present:
            od = tdir / "objects" / SHARD / obj
            if cond_in is None and (od / "edits_2d" / f"{eid}_input.png").is_file():
                cond_in = od / "edits_2d" / f"{eid}_input.png"
                cond_ed = od / "edits_2d" / f"{eid}_edited.png"
            if before_dir is None and (od / "gate_views" / "before_view_front.png").is_file():
                before_dir = od / "gate_views"
        rows = ""
        if before_dir is not None:
            rows += (f'<div class="vr"><span class="tag" style="color:#aeb4bd">BEFORE (512 decode)</span>'
                     f'{strip(before_dir, "before_view")}</div>')
        for lbl, tdir, col in present:
            ed = tdir / "objects" / SHARD / obj / "edits_3d" / eid
            if (ed / "after_view_front.png").is_file():
                rows += (f'<div class="vr"><span class="tag" style="color:{col}">AFTER · {lbl}</span>'
                         f'{strip(ed, "after_view")}</div>')
        ci = thumb(cond_in) if cond_in else ""
        ce = thumb(cond_ed) if cond_ed else ""
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

    legend = " · ".join(f'<b style="color:{c}">{lbl}</b>' for lbl, _, c in present)
    out = DATA / "_scratch" / "ab_compare" / "pad_sweep_t1ss_r512.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>512 T1-SS pad sweep</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 6px}} .note{{font-size:13px;color:#aeb4bd;margin-bottom:16px;line-height:1.6}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}} .obj{{color:#8a8f98;font-weight:400}}
 .grid{{display:flex;gap:14px;align-items:flex-start}} .conds{{flex:0 0 auto}}
 .cap{{font-size:10px;color:#aeb4bd;margin:2px 0}} .ci{{display:block;border-radius:4px;margin-bottom:6px;max-width:240px}}
 .vox{{flex:1 1 auto;min-width:0}} .vr{{margin:3px 0}} .tag{{font-size:11px;font-weight:600;display:block;margin:2px 0}}
 table{{border-collapse:collapse}} td{{padding:2px}} img{{display:block;border-radius:3px;width:180px;background:#fff}}
</style></head><body>
<h1>512 TRELLIS.1-SS — dilation-pad sweep (pad3/4/5/7)</h1>
<div class="note">同一配方(<b>TRELLIS.1 SS flow</b> + 同帧 64³ restore + perstep S2 + edit_res=512 + 白模),唯一变量 = 64³ 编辑网格 Chebyshev 膨胀 pad。{legend}。{len(keys)} edits。</div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"wrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
