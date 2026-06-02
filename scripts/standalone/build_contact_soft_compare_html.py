#!/usr/bin/env python3
"""Self-contained before/after HTML comparing two masked-edit render runs.

For every edit present in ``--new-dir`` (and, when available, ``--old-dir``)
build one row:

    [ 2D input | 2D edited | OLD mesh (after_shaded) | NEW mesh (after_shaded) ]

so you can eyeball whether a recipe change (e.g. the v1 contact-aware soft
masks, ``--contact-soft``) fixed the tearing without re-rendering anything —
it just tiles the ``after_shaded.png`` multiview grids the runs already saved.
Images are embedded as base64 JPEG so the output is a single portable .html.

  python scripts/standalone/build_contact_soft_compare_html.py \
    --old-dir data/Pxform_v2/_rerun_v2/08 \
    --new-dir data/Pxform_v2/_rerun_v2/08_contact_soft \
    --objects-root data/Pxform_v2/objects/08
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
        ImageDraw.Draw(im).text((6, size // 2), f"(no {label})", fill=(170, 0, 0))
    return im


CSS = """
body{margin:0;background:#0d0f12;color:#e6e8ea;font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial}
header{position:sticky;top:0;background:#0d0f12ee;backdrop-filter:blur(6px);
 border-bottom:1px solid #272b31;padding:12px 18px}
h1{margin:0;font-size:17px}.leg{color:#9aa3ad;font-size:12px;margin-top:4px}
.wrap{padding:16px;display:flex;flex-direction:column;gap:14px}
.card{background:#171a1f;border:1px solid #272b31;border-radius:10px;overflow:hidden}
.bar{padding:8px 12px;border-bottom:1px solid #272b31;font-family:ui-monospace,Menlo,monospace;
 font-size:12px;color:#cdd3da;display:flex;gap:10px;align-items:center}
.type{color:#9aa3ad;border:1px solid #272b31;border-radius:5px;padding:1px 6px;font-size:11px}
.row{display:grid;grid-template-columns:1fr 1fr 1.6fr 1.6fr;gap:6px;padding:8px}
.cell{display:flex;flex-direction:column;gap:4px}
.cap{font-size:11px;color:#9aa3ad;text-align:center}
.cap.new{color:#8ee6a3}.cap.old{color:#ff8a8a}
img{width:100%;height:auto;display:block;background:#000;border-radius:4px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--objects-root", required=True)
    ap.add_argument("--edits-2d-subdir", default="edits_2d")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()

    new_root = Path(args.new_dir)
    old_root = Path(args.old_dir)
    obj_root = Path(args.objects_root)
    out = Path(args.out) if args.out else new_root / "compare.html"
    sz = args.size

    cards = []
    n = 0
    for obj in sorted(p for p in new_root.iterdir()
                      if p.is_dir() and not p.name.startswith("_")):
        for ed in sorted(obj.iterdir()):
            new_png = ed / "after_shaded.png"
            if not new_png.is_file():
                continue
            eid = ed.name
            et = eid.split("_")[0]
            e2d = obj_root / obj.name / args.edits_2d_subdir
            old_png = old_root / obj.name / eid / "after_shaded.png"
            cells = [
                ("2D input", _load(e2d / f"{eid}_input.png", sz, "input"), ""),
                ("2D edited", _load(e2d / f"{eid}_edited.png", sz, "edited"), ""),
                ("OLD (perstep/posthoc)", _load(old_png, sz * 2, "old"), "old"),
                ("NEW (contact-soft)", _load(new_png, sz * 2, "new"), "new"),
            ]
            row = "".join(
                f'<div class="cell"><div class="cap {cls}">{html.escape(cap)}</div>'
                f'<img src="{_embed(im, args.quality)}"></div>'
                for cap, im, cls in cells)
            short = eid.replace(obj.name, "").strip("_")
            cards.append(
                f'<div class="card"><div class="bar">'
                f'<b>{html.escape(obj.name[:10])}</b> &middot; {html.escape(short)}'
                f'<span class="type">{html.escape(et)}</span></div>'
                f'<div class="row">{row}</div></div>')
            n += 1
            print(f"  {eid}")

    doc = (
        "<!doctype html><meta charset=utf-8>"
        f"<style>{CSS}</style>"
        f"<header><h1>contact-soft compare &mdash; {n} edits</h1>"
        "<div class=leg>per row: 2D input | 2D edited | "
        "<b style='color:#ff8a8a'>OLD</b> mesh | "
        "<b style='color:#8ee6a3'>NEW (--contact-soft)</b> mesh &mdash; "
        "after_shaded multiview; look for torn/holey shells in OLD vs NEW.</div>"
        "</header>"
        f"<div class=wrap>{''.join(cards)}</div>")
    out.write_text(doc, encoding="utf-8")
    mb = out.stat().st_size / 1024 / 1024
    print(f"\nwrote {out}  ({n} edits, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
