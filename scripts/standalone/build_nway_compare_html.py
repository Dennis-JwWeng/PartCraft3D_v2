#!/usr/bin/env python3
"""Self-contained N-way before/after HTML comparing several masked-edit runs.

Unlike ``build_contact_soft_compare_html.py`` (fixed OLD-vs-NEW), this tiles an
arbitrary number of labelled run directories side by side so you can eyeball a
multi-arm ablation in one page.  Each edit becomes one row:

    [ 2D input | 2D edited | run_1 mesh | run_2 mesh | ... ]

using the ``after_shaded.png`` multiview grids the runs already saved (no
re-render).  Images are embedded as base64 JPEG → one portable .html.

  python scripts/standalone/build_nway_compare_html.py \
    --objects-root data/Pxform_v2/objects/08 \
    --out data/Pxform_v2/_rerun_v2/s1mask_compare.html \
    "a posthoc (ref)=data/Pxform_v2/_rerun_v2/08" \
    "b S1 mask=data/Pxform_v2/_rerun_v2/08_s1mask_b" \
    "c +subtract=data/Pxform_v2/_rerun_v2/08_s1mask_c"

Each positional arg is ``LABEL=DIR``.  The row set is the union of every edit
that has an ``after_shaded.png`` in any run; columns a run is missing show a
grey placeholder.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

from PIL import Image


def _embed(img: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load(path: Path, size: int, label: str) -> Image.Image:
    im = Image.new("RGB", (size, size), (245, 245, 245))
    try:
        src = Image.open(path).convert("RGB")
        src.thumbnail((size, size))
        im.paste(src, ((size - src.width) // 2, (size - src.height) // 2))
    except Exception:
        from PIL import ImageDraw
        ImageDraw.Draw(im).text((6, size // 2), f"(no {label})", fill=(150, 150, 150))
    return im


CSS = """
body{margin:0;background:#0d0f12;color:#e6e8ea;font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial}
header{position:sticky;top:0;background:#0d0f12ee;backdrop-filter:blur(6px);
 border-bottom:1px solid #272b31;padding:12px 18px;z-index:5}
h1{margin:0;font-size:17px}.leg{color:#9aa3ad;font-size:12px;margin-top:4px}
.wrap{padding:16px;display:flex;flex-direction:column;gap:14px}
.card{background:#171a1f;border:1px solid #272b31;border-radius:10px;overflow:hidden}
.bar{padding:8px 12px;border-bottom:1px solid #272b31;font-family:ui-monospace,Menlo,monospace;
 font-size:12px;color:#cdd3da;display:flex;gap:10px;align-items:center}
.type{color:#9aa3ad;border:1px solid #272b31;border-radius:5px;padding:1px 6px;font-size:11px}
.row{display:grid;gap:6px;padding:8px}
.cell{display:flex;flex-direction:column;gap:4px}
.cap{font-size:11px;color:#9aa3ad;text-align:center}
.cap.ref{color:#ffd27f}.cap.run{color:#8ee6a3}
img{width:100%;height:auto;display:block;background:#000;border-radius:4px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=DIR per run (order = column order)")
    ap.add_argument("--objects-root", required=True)
    ap.add_argument("--edits-2d-subdir", default="edits_2d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()

    runs = []  # (label, Path)
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"run spec must be LABEL=DIR, got: {spec!r}")
        label, d = spec.split("=", 1)
        runs.append((label.strip(), Path(d.strip())))
    obj_root = Path(args.objects_root)
    out = Path(args.out)
    sz = args.size

    # union of (obj, edit) across all runs that have an after_shaded.png
    edits: dict[tuple[str, str], None] = {}
    for _, rd in runs:
        if not rd.is_dir():
            continue
        for obj in sorted(p for p in rd.iterdir()
                          if p.is_dir() and not p.name.startswith("_")):
            for ed in sorted(obj.iterdir()):
                if (ed / "after_shaded.png").is_file():
                    edits[(obj.name, ed.name)] = None

    # column template: 2D input | 2D edited | one per run
    ncol_runs = len(runs)
    template = "1fr 1fr " + " ".join(["1.6fr"] * ncol_runs)

    cards = []
    for (obj, eid) in sorted(edits):
        et = eid.split("_")[0]
        e2d = obj_root / obj / args.edits_2d_subdir
        cells = [
            ("2D input", _load(e2d / f"{eid}_input.png", sz, "input"), ""),
            ("2D edited", _load(e2d / f"{eid}_edited.png", sz, "edited"), ""),
        ]
        for i, (label, rd) in enumerate(runs):
            cls = "ref" if i == 0 else "run"
            cells.append((label, _load(rd / obj / eid / "after_shaded.png",
                                       sz * 2, label), cls))
        row = "".join(
            f'<div class="cell"><div class="cap {cls}">{html.escape(cap)}</div>'
            f'<img src="{_embed(im, args.quality)}"></div>'
            for cap, im, cls in cells)
        short = eid.replace(obj, "").strip("_")
        cards.append(
            f'<div class="card"><div class="bar">'
            f'<b>{html.escape(obj[:10])}</b> &middot; {html.escape(short)}'
            f'<span class="type">{html.escape(et)}</span></div>'
            f'<div class="row" style="grid-template-columns:{template}">{row}</div></div>')
        print(f"  {obj[:10]} {eid}")

    legend = " | ".join(
        f"<b style='color:{'#ffd27f' if i == 0 else '#8ee6a3'}'>{html.escape(lbl)}</b>"
        for i, (lbl, _) in enumerate(runs))
    doc = (
        "<!doctype html><meta charset=utf-8>"
        f"<style>{CSS}</style>"
        f"<header><h1>S1-mask ablation &mdash; {len(edits)} edits, {ncol_runs} runs</h1>"
        f"<div class=leg>per row: 2D input | 2D edited | {legend} &mdash; "
        "after_shaded multiview; look for torn/holey shells &amp; broken small parts.</div>"
        "</header>"
        f"<div class=wrap>{''.join(cards)}</div>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    mb = out.stat().st_size / 1024 / 1024
    print(f"\nwrote {out}  ({len(edits)} edits, {ncol_runs} runs, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
