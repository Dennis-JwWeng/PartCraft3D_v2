#!/usr/bin/env python3
"""HTML report: pad4-run after vs pad0 re-pasted after, single best view per edit.

Per edit (gate-E pass, repaste done) one row of 4 same-camera images:
  Before                <obj>/gate_views/before_view_<view>.png
  2D condition          <obj>/edits_2d/<edit_id>_edited.png  (FLUX output)
  After pad4 (prod)     <obj>/edits_3d/<id>/after_view_<view>.png
  After pad0 (repaste)  <obj>/edits_3d/<id>/repaste_pad{P}/after_view_<view>.png

view = the gate-A best view (the FLUX condition camera) — identical for all
columns.  Images are downscaled to JPEG thumbnails and base64-embedded; output
is paginated (index.html + page_NNN.html) to keep each file openable.

Usage:
  python scripts/viz/build_repaste_compare_html.py --shard 00 \\
    --out-dir reports/repaste_pad0_shard00 [--per-page 100] [--thumb 320]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image


def thumb_b64(p: Path, size: int, quality: int) -> str:
    if not p.is_file():
        return ""
    im = Image.open(p).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def collect(shard_dir: Path, pads: list[int],
            ext: dict[int, Path]) -> list[dict]:
    """Rows keyed on the FIRST pad in ``pads`` (defines the subset); the other
    pads contribute extra columns (blank cell when that repaste is missing).
    ``ext`` maps pad → external root holding <obj_id>/<edit_id>/ outputs;
    pads not in ``ext`` are read in-place (edits_3d/<id>/repaste_pad{P})."""
    primary = pads[0]

    def pad_dir(pad: int, obj: str, eid: str) -> Path:
        if pad in ext:
            return ext[pad] / obj / eid
        return shard_dir / obj / "edits_3d" / eid / f"repaste_pad{pad}"

    if primary in ext:
        metas = sorted(ext[primary].glob("*/*/meta.json"))
        keys = [(m.parents[1].name, m.parents[0].name) for m in metas]
    else:
        metas = sorted(shard_dir.glob(f"*/edits_3d/*/repaste_pad{primary}/meta.json"))
        keys = [(m.parents[3].name, m.parents[1].name) for m in metas]

    rows = []
    for meta_p, (obj, eid) in zip(metas, keys):
        rp_dir = meta_p.parent
        obj_dir = shard_dir / obj
        edit_dir = obj_dir / "edits_3d" / eid
        meta = json.loads(meta_p.read_text())
        view = meta.get("view_name", "")
        png = rp_dir / f"after_view_{view}.png"
        if not png.is_file():
            continue
        prompt = ""
        pp = edit_dir / "refined_prompt.json"
        if pp.is_file():
            try:
                d = json.loads(pp.read_text())
                prompt = d.get("improved_prompt") or d.get("original_prompt") or ""
            except Exception:
                pass
        pastes = {}   # pad -> (png path, preserved count)
        for p in pads:
            d = pad_dir(p, obj, eid)
            mp = d / "meta.json"
            n_pres = -1
            if mp.is_file():
                try:
                    n_pres = json.loads(mp.read_text()).get("preserved", -1)
                except Exception:
                    pass
            pastes[p] = (d / f"after_view_{view}.png", n_pres)
        rows.append({
            "obj": obj_dir.name,
            "edit_id": edit_dir.name,
            "view": view,
            "parts": meta.get("parts", []),
            "tokens": meta.get("tokens_total", 0),
            "white": bool(meta.get("white_model", False)),
            "prompt": prompt,
            "before": obj_dir / "gate_views" / f"before_view_{view}.png",
            "cond": obj_dir / "edits_2d" / f"{edit_dir.name}_edited.png",
            "after_pad4": edit_dir / f"after_view_{view}.png",
            "pastes": pastes,
        })
    return rows


CSS = """
  body { font-family: system-ui, sans-serif; margin: 24px 32px; background: #0f1117; color: #e8eaed; }
  h1 { font-size: 1.3rem; }
  .intro, .nav { color: #9aa0a6; margin-bottom: 16px; line-height: 1.6; font-size: 0.9rem; }
  .nav a { color: #6ea8fe; margin-right: 10px; text-decoration: none; }
  table { width: 100%; border-collapse: collapse; }
  th, td { border: 1px solid #2a2f3d; padding: 6px; vertical-align: top; text-align: center; }
  th { background: #222633; font-size: 0.85rem; position: sticky; top: 0; }
  img { max-width: 300px; width: 100%; background: #111; border-radius: 6px; }
  .eid { font-size: 0.78rem; color: #6ea8fe; word-break: break-all; text-align: left; max-width: 200px; }
  .meta { font-size: 0.74rem; color: #9aa0a6; text-align: left; }
  .prompt { font-size: 0.78rem; color: #d2d6db; text-align: left; margin-top: 6px; }
  .white { color: #fdd663; }
"""


def page_html(rows: list[dict], pads: list[int], title: str, nav: str,
              thumb: int, quality: int) -> str:
    trs = []
    for r in rows:
        white = ' <span class="white">[white]</span>' if r["white"] else ""
        stats = " · ".join(
            f"pad{p}: {n}/{r['tokens']}" for p, (_, n) in r["pastes"].items() if n >= 0)
        paste_tds = "".join(
            f'<td><img src="{thumb_b64(png, thumb, quality)}" loading="lazy"></td>'
            for png, _ in r["pastes"].values())
        trs.append(f"""
      <tr>
        <td class="eid">{html.escape(r['edit_id'])}{white}
          <div class="meta">obj {html.escape(r['obj'])}<br>
            parts {r['parts']} · view <b>{html.escape(r['view'])}</b><br>
            preserved {html.escape(stats)}</div>
          <div class="prompt">{html.escape(r['prompt'])}</div></td>
        <td><img src="{thumb_b64(r['before'], thumb, quality)}" loading="lazy"></td>
        <td><img src="{thumb_b64(r['cond'], thumb, quality)}" loading="lazy"></td>
        <td><img src="{thumb_b64(r['after_pad4'], thumb, quality)}" loading="lazy"></td>
        {paste_tds}
      </tr>""")
    paste_ths = "".join(f"<th>After pad{p}（重贴）</th>" for p in pads)
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="nav">{nav}</div>
<p class="intro">同一 best-view 相机（FLUX condition 视角）。Before = 原始 mesh 渲染；
2D condition = FLUX 编辑图；After pad4 = 生产 pad4 贴回；After padN = 用 padN 掩码
离线重贴（pad 越小保留区越大，零重新生成）。</p>
<table>
  <thead><tr><th>edit</th><th>Before</th><th>2D condition</th><th>After pad4（生产）</th>{paste_ths}</tr></thead>
  <tbody>{''.join(trs)}</tbody>
</table>
<div class="nav">{nav}</div>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Pxform_v2/prod_posthoc_no2dqc")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--pads", default="0",
                    help="comma list of repaste pads; FIRST defines the row "
                         "subset, the rest add columns (e.g. '2,0')")
    ap.add_argument("--ext", action="append", default=[],
                    help="pad:root for externally-stored outputs, e.g. "
                         "'2:reports/repaste_pad2_shard00' (repeatable)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--thumb", type=int, default=320)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    shard_dir = Path(args.root) / "objects" / args.shard
    pads = [int(p) for p in args.pads.split(",")]
    ext = {int(e.split(":", 1)[0]): Path(e.split(":", 1)[1]) for e in args.ext}
    rows = collect(shard_dir, pads, ext)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no repaste rows found", file=sys.stderr)
        sys.exit(1)

    pads_label = "/".join(f"pad{p}" for p in pads)
    out_dir = Path(args.out_dir
                   or f"reports/repaste_pad{pads[0]}_shard{args.shard}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = [rows[i: i + args.per_page] for i in range(0, len(rows), args.per_page)]
    links = " ".join(f'<a href="page_{i + 1:03d}.html">p{i + 1}</a>'
                     for i in range(len(pages)))
    for i, pr in enumerate(pages):
        title = (f"pad4 vs {pads_label} 重贴对比 — shard {args.shard} · "
                 f"page {i + 1}/{len(pages)} · edits {i * args.per_page + 1}-"
                 f"{i * args.per_page + len(pr)}/{len(rows)}")
        nav = f'<a href="index.html">index</a> {links}'
        (out_dir / f"page_{i + 1:03d}.html").write_text(
            page_html(pr, pads, title, nav, args.thumb, args.quality),
            encoding="utf-8")
        print(f"page_{i + 1:03d}.html  {len(pr)} rows")

    n_white = sum(r["white"] for r in rows)
    idx_rows = "".join(
        f'<li><a href="page_{i + 1:03d}.html">page {i + 1}</a> — '
        f'{html.escape(pr[0]["edit_id"])} … {html.escape(pr[-1]["edit_id"])} '
        f'({len(pr)} edits)</li>'
        for i, pr in enumerate(pages))
    (out_dir / "index.html").write_text(f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>repaste compare index</title>
<style>{CSS} li {{ margin: 4px 0; }} a {{ color: #6ea8fe; }}</style></head><body>
<h1>pad4 vs {pads_label} 重贴对比 — shard {args.shard}</h1>
<p class="intro">{len(rows)} edits（gate-E pass，重贴完成），其中 white-model {n_white} 个 ·
每页 {args.per_page} 行 · 列：Before / 2D condition / After pad4（生产） /
{' / '.join(f'After pad{p}（重贴）' for p in pads)}，全部同一 best-view 相机。</p>
<ul>{idx_rows}</ul>
</body></html>""", encoding="utf-8")
    print(f"index.html → {out_dir}  ({len(rows)} rows, {len(pages)} pages, "
          f"{n_white} white)")


if __name__ == "__main__":
    main()
