#!/usr/bin/env python3
"""Build a SELF-CONTAINED HTML gallery of the per-edit mask/feature 3D panels.

Embeds each ``<edit_id>.png`` panel from ``--viz-dir`` as a base64 JPEG data URI
so the output is a single portable ``.html`` (no sidecar files). Edits are
sorted by edit-region coverage (whole-object masks float to the top, flagged);
coverage / edit-type are recomputed from the saved ``ss.npz`` under ``--in-root``.
A search box filters by edit id; coverage-bucket buttons filter by severity.

  python scripts/standalone/build_mask_viz_html.py \
    --viz-dir data/Pxform_v2/_rerun_v2/08/_mask_viz_3d \
    --in-root data/Pxform_v2/_rerun_v2/08
"""
from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

import numpy as np
from PIL import Image

G = 64


def _edit_grid_dense(ss):
    eg = np.asarray(ss["edit_grid"])
    if eg.ndim == 3:
        return eg.astype(bool)
    g = np.zeros((G, G, G), bool)
    e = eg.astype(int)
    g[e[:, 0], e[:, 1], e[:, 2]] = True
    return g


def _coverage_and_type(ss):
    c0 = np.asarray(ss["coords0"]).astype(int)
    edit = _edit_grid_dense(ss)
    cov = 100.0 * edit[c0[:, 0], c0[:, 1], c0[:, 2]].mean()
    et = str(ss["edit_type"]) if "edit_type" in ss.keys() else "?"
    return cov, et


def _embed_jpeg(png_path: Path, quality: int) -> str:
    im = Image.open(png_path).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


CSS = """
:root{--bg:#0d0f12;--card:#171a1f;--line:#272b31;--txt:#e6e8ea;--mut:#9aa3ad}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
 font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial}
header{position:sticky;top:0;z-index:5;background:#0d0f12ee;backdrop-filter:blur(6px);
 border-bottom:1px solid var(--line);padding:14px 20px}
h1{margin:0 0 6px;font-size:18px}
.legend{color:var(--mut);font-size:12px;margin:6px 0}
.legend b{color:var(--txt)}
.controls{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input[type=search]{background:#10131722;border:1px solid var(--line);color:var(--txt);
 border-radius:8px;padding:7px 10px;width:260px;outline:none}
button{background:var(--card);border:1px solid var(--line);color:var(--txt);
 border-radius:8px;padding:7px 11px;cursor:pointer}
button.on{background:#2a3340;border-color:#3c4a5c}
.wrap{padding:18px 20px;display:flex;flex-direction:column;gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.bar{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid var(--line)}
.badge{font-weight:700;border-radius:6px;padding:2px 9px;font-variant-numeric:tabular-nums}
.b-hi{background:#3a1515;color:#ff8a8a;border:1px solid #5a2020}
.b-mid{background:#3a3115;color:#ffd98a;border:1px solid #5a4a20}
.b-lo{background:#16321b;color:#8ee6a3;border:1px solid #205a2c}
.eid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--mut)}
.type{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:5px;padding:1px 6px}
.flag{margin-left:auto;color:#ff8a8a;font-weight:600;font-size:12px}
img{display:block;width:100%;height:auto;background:#000}
.empty{padding:40px;text-align:center;color:var(--mut)}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];
let bucket='all';
function apply(){
 const q=document.getElementById('q').value.toLowerCase();
 let n=0;
 for(const c of cards){
  const cov=parseFloat(c.dataset.cov);
  const okB = bucket==='all' || (bucket==='hi'&&cov>=60) ||
              (bucket==='mid'&&cov>=30&&cov<60) || (bucket==='lo'&&cov<30);
  const okQ = c.dataset.eid.includes(q);
  const show = okB && okQ; c.style.display = show?'':'none'; if(show)n++;
 }
 document.getElementById('count').textContent=n+' shown';
}
function setBucket(b,el){bucket=b;
 document.querySelectorAll('.controls button').forEach(x=>x.classList.remove('on'));
 el.classList.add('on');apply();}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-dir", required=True)
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    viz_dir = Path(args.viz_dir)
    in_root = Path(args.in_root)
    out = Path(args.out) if args.out else viz_dir / "index.html"
    shard = in_root.name

    rows = []
    for obj in sorted(p for p in in_root.iterdir()
                      if p.is_dir() and not p.name.startswith("_")):
        for ed in sorted(obj.iterdir()):
            ss_p = ed / "latents" / "ss.npz"
            panel = viz_dir / f"{ed.name}.png"
            if not (ss_p.is_file() and panel.is_file()):
                continue
            ss = np.load(ss_p, allow_pickle=True)
            cov, et = _coverage_and_type(ss)
            rows.append((cov, obj.name, ed.name, et, panel))
            print(f"{cov:6.1f}%  {ed.name}")

    rows.sort(key=lambda r: -r[0])
    n_hi = sum(1 for r in rows if r[0] >= 60)

    cards = []
    for cov, obj, eid, et, panel in rows:
        cls = "b-hi" if cov >= 60 else ("b-mid" if cov >= 30 else "b-lo")
        flag = ('<span class="flag">&#9664; mask ~ whole object</span>'
                if cov >= 60 else "")
        short = eid.replace(obj, "").strip("_")
        src = _embed_jpeg(panel, args.quality)
        cards.append(
            f'<div class="card" data-eid="{html.escape(eid.lower())}" '
            f'data-cov="{cov:.1f}">'
            f'<div class="bar">'
            f'<span class="badge {cls}">{cov:.1f}%</span>'
            f'<span class="eid">{html.escape(obj[:10])} &middot; {html.escape(short)}</span>'
            f'<span class="type">{html.escape(et)}</span>{flag}</div>'
            f'<img loading="lazy" alt="{html.escape(eid)}" src="{src}"></div>')

    legend = (
        '<div class="legend">6 columns &rarr; '
        '<b>2D original</b> | <b>2D edited</b> | '
        '<b>S1 mask @16&sup3; (before)</b> grn=keep red=edit | '
        '<b>S2 mask @64&sup3; (before)</b> grn=keep red=edit | '
        '<b>shape SLat PCA @64&sup3; (after)</b> | '
        '<b>tex SLat PCA @64&sup3; (after)</b></div>'
        '<div class="legend">coverage = % of body voxels inside the edit region; '
        '<b>&ge;60% = mask &asymp; whole object</b> (masked edit degenerates &rarr; '
        'route to vanilla). masks read verbatim from each edit&rsquo;s saved '
        '<code>ss.npz</code> (the tensors the pipeline actually used).</div>')

    doc = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>shard {shard} &mdash; edit mask / SLat viz</title>"
        f"<style>{CSS}</style></head><body>"
        f"<header><h1>shard {shard} &mdash; per-stage edit mask &amp; SLat feature viz "
        f"({len(rows)} edits, {n_hi} whole-object)</h1>{legend}"
        "<div class=controls>"
        "<input id=q type=search placeholder='filter by edit id&hellip;' oninput=apply()>"
        "<button class=on onclick=\"setBucket('all',this)\">all</button>"
        "<button onclick=\"setBucket('hi',this)\">&ge;60%</button>"
        "<button onclick=\"setBucket('mid',this)\">30&ndash;60%</button>"
        "<button onclick=\"setBucket('lo',this)\">&lt;30%</button>"
        "<span id=count class=legend></span></div></header>"
        f"<div class=wrap>{''.join(cards) or '<div class=empty>no panels found</div>'}</div>"
        f"<script>{JS}\napply();</script></body></html>")

    out.write_text(doc, encoding="utf-8")
    mb = out.stat().st_size / 1024 / 1024
    print(f"\nwrote {out}  ({len(rows)} edits, {mb:.1f} MB self-contained)")


if __name__ == "__main__":
    main()
