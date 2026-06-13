#!/usr/bin/env python3
"""Self-contained HTML: before / 2D condition / after for mod+scale edits.

Per edit entry:
  - 2D condition: edits_2d/{edit_id}_input.png + _edited.png
  - 5 named views: gate_views/before_view_{v}.png vs edits_3d/{id}/after_view_{v}.png

Images are JPEG-thumbnail base64-embedded (portable single-file report).

Usage:
  python scripts/viz/build_xl_gatee_compare_html.py \\
    --root data/Pxform_v2/partversexl_posthoc_no2dqc \\
    --shard 00 \\
    --obj-ids-file configs/partversexl_smoke_10_shard00.txt \\
    --out data/Pxform_v2/partversexl_posthoc_no2dqc/_global/xl_smoke10_gatee_30.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.render.ovox_views import VIEW_ORDER

VIEW_ZH = {
    "front": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "down": "下",
}


def thumb_b64(p: Path, size: int, quality: int) -> str:
    if not p.is_file():
        return ""
    from PIL import Image

    im = Image.open(p).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def img_cell(p: Path, size: int, quality: int, alt: str) -> str:
    src = thumb_b64(p, size, quality)
    if not src:
        return '<span class="miss">missing</span>'
    return f'<img src="{src}" alt="{html.escape(alt)}" loading="lazy">'


def load_rows(root: Path, shard: str, obj_ids: list[str]) -> list[dict]:
    shard_dir = root / "objects" / shard
    rows: list[dict] = []
    for oid in obj_ids:
        obj_dir = shard_dir / oid
        es_path = obj_dir / "edit_status.json"
        parsed_path = obj_dir / "phase1" / "parsed.json"
        if not es_path.is_file():
            continue
        es = json.loads(es_path.read_text())
        parsed_by_type_seq: dict[tuple[str, int], dict] = {}
        if parsed_path.is_file():
            flux_seq = del_seq = 0
            for e in (json.loads(parsed_path.read_text()).get("parsed") or {}).get("edits") or []:
                et = e.get("edit_type", "")
                if et in ("modification", "scale", "material", "color", "global"):
                    parsed_by_type_seq[(et, flux_seq)] = e
                    flux_seq += 1
                elif et == "deletion":
                    parsed_by_type_seq[("deletion", del_seq)] = e
                    del_seq += 1

        for eid, ed in sorted((es.get("edits") or {}).items()):
            if not eid.startswith(("mod_", "scl_")):
                continue
            edit_dir = obj_dir / "edits_3d" / eid
            if not all((edit_dir / f"after_view_{v}.png").is_file() for v in VIEW_ORDER):
                continue

            et = ed.get("edit_type", "")
            seq = int(eid.rsplit("_", 1)[-1])
            pe = parsed_by_type_seq.get((et, seq), {})

            ga = (ed.get("stages") or {}).get("gate_a") or {}
            ge = (ed.get("stages") or {}).get("gate_e") or {}
            gav = (ga.get("verdict") or {}).get("vlm") or {}
            gev = (ge.get("verdict") or {}).get("vlm") or {}
            best_view = gav.get("view_name") or VIEW_ORDER[gav.get("best_view", 0)] if gav.get("best_view") is not None else ""

            rows.append({
                "obj_id": oid,
                "edit_id": eid,
                "edit_type": et,
                "prompt": pe.get("prompt") or "",
                "gate_a": ga.get("status", ""),
                "gate_e": ge.get("status", ""),
                "gate_e_score": gev.get("score"),
                "gate_e_reason": (gev.get("reason") or "")[:240],
                "best_view": best_view,
                "cond_in": obj_dir / "edits_2d" / f"{eid}_input.png",
                "cond_ed": obj_dir / "edits_2d" / f"{eid}_edited.png",
                "before": {v: obj_dir / "gate_views" / f"before_view_{v}.png" for v in VIEW_ORDER},
                "after": {v: edit_dir / f"after_view_{v}.png" for v in VIEW_ORDER},
            })
    return rows


CSS = """
  :root { --bg:#0f1117; --card:#181b24; --border:#2a2f3d; --text:#e8eaed; --muted:#9aa0a6;
          --ok:#3dd68c; --fail:#f87171; --link:#6ea8fe; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 0; padding: 20px 28px 48px;
         background: var(--bg); color: var(--text); }
  h1 { font-size: 1.35rem; margin: 0 0 8px; }
  .intro { color: var(--muted); font-size: 0.9rem; line-height: 1.55; max-width: 1100px; margin-bottom: 20px; }
  .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; font-size: 0.85rem; }
  .stat { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
  .toc { margin-bottom: 28px; font-size: 0.82rem; line-height: 1.8; }
  .toc a { color: var(--link); text-decoration: none; margin-right: 10px; }
  .entry { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
           padding: 14px 16px 18px; margin-bottom: 22px; }
  .entry-hd { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; margin-bottom: 10px; }
  .eid { font-weight: 600; font-size: 0.95rem; color: var(--link); }
  .badge { font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--border); }
  .badge.pass { color: var(--ok); border-color: #2d6b4f; }
  .badge.fail { color: var(--fail); border-color: #7f3b3b; }
  .meta { font-size: 0.78rem; color: var(--muted); }
  .prompt { font-size: 0.82rem; margin: 8px 0 12px; line-height: 1.45; }
  .reason { font-size: 0.78rem; color: var(--muted); margin-bottom: 12px; line-height: 1.4; }
  h3 { font-size: 0.82rem; margin: 12px 0 8px; color: var(--muted); font-weight: 600; }
  table { width: 100%; border-collapse: collapse; }
  th, td { border: 1px solid var(--border); padding: 6px; text-align: center; vertical-align: middle; }
  th { background: #222633; font-size: 0.78rem; }
  .vlabel { font-size: 0.8rem; white-space: nowrap; }
  img { max-width: 220px; width: 100%; height: auto; display: block; margin: 0 auto;
        background: #0a0c10; border-radius: 4px; }
  .miss { color: #666; font-size: 0.75rem; }
  .cond-row td { width: 50%; }
"""


def render_entry(r: dict, thumb: int, quality: int) -> str:
    ge_cls = "pass" if r["gate_e"] == "pass" else "fail"
    view_rows = []
    for v in VIEW_ORDER:
        zh = VIEW_ZH.get(v, v)
        star = " ★" if r["best_view"] == v else ""
        view_rows.append(f"""
        <tr>
          <td class="vlabel">{html.escape(zh)}<br><span class="meta">{html.escape(v)}{star}</span></td>
          <td>{img_cell(r['before'][v], thumb, quality, f"before {v}")}</td>
          <td>{img_cell(r['after'][v], thumb, quality, f"after {v}")}</td>
        </tr>""")

    score = r["gate_e_score"]
    score_s = f" · score {score}" if score is not None else ""
    reason = html.escape(r["gate_e_reason"]) if r["gate_e_reason"] else ""

    return f"""
    <section class="entry" id="{html.escape(r['edit_id'])}">
      <div class="entry-hd">
        <span class="eid">{html.escape(r['edit_id'])}</span>
        <span class="badge">{html.escape(r['edit_type'])}</span>
        <span class="badge {ge_cls}">gate_e {html.escape(r['gate_e'] or '?')}{score_s}</span>
        <span class="meta">obj {html.escape(r['obj_id'][:12])}… · best_view {html.escape(r['best_view'] or '?')}</span>
      </div>
      <div class="prompt">{html.escape(r['prompt'])}</div>
      {f'<div class="reason">{reason}</div>' if reason else ''}
      <h3>2D Condition（FLUX input → edited）</h3>
      <table class="cond-table">
        <thead><tr><th>input</th><th>edited（3D 条件）</th></tr></thead>
        <tbody><tr class="cond-row">
          <td>{img_cell(r['cond_in'], thumb, quality, 'input')}</td>
          <td>{img_cell(r['cond_ed'], thumb, quality, 'edited')}</td>
        </tr></tbody>
      </table>
      <h3>3D Gate views（before → after，5 视角）</h3>
      <table>
        <thead><tr><th>视角</th><th>Before</th><th>After</th></tr></thead>
        <tbody>{''.join(view_rows)}</tbody>
      </table>
    </section>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shard", default="00")
    ap.add_argument("--obj-ids-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thumb", type=int, default=280)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    obj_ids: list[str] = []
    if args.obj_ids_file and args.obj_ids_file.is_file():
        obj_ids = [
            ln.strip() for ln in args.obj_ids_file.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    else:
        shard_dir = args.root / "objects" / args.shard
        obj_ids = sorted(d.name for d in shard_dir.iterdir() if d.is_dir())

    rows = load_rows(args.root, args.shard, obj_ids)
    n_pass = sum(1 for r in rows if r["gate_e"] == "pass")
    n_fail = sum(1 for r in rows if r["gate_e"] == "fail")

    toc = "".join(
        f'<a href="#{html.escape(r["edit_id"])}">'
        f'{"✓" if r["gate_e"]=="pass" else "✗"} {html.escape(r["edit_id"][:28])}</a>'
        for r in rows
    )
    body = "".join(render_entry(r, args.thumb, args.quality) for r in rows)

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XL smoke Gate-E · {len(rows)} edits</title>
<style>{CSS}</style>
</head><body>
<h1>PartVerse XL smoke — mod/scale Gate E 对比（{len(rows)} 条）</h1>
<p class="intro">每条 edit：上方为 FLUX 2D condition（input / edited）；下方为同一物体 5 视角 3D 渲染
（gate_views before vs trellis2 after）。★ = gate_a best_view。图片 JPEG 缩略图 base64 内嵌，单文件可离线打开。</p>
<div class="stats">
  <div class="stat"><b>{len(rows)}</b> edits</div>
  <div class="stat" style="color:var(--ok)"><b>{n_pass}</b> gate_e pass</div>
  <div class="stat" style="color:var(--fail)"><b>{n_fail}</b> gate_e fail</div>
</div>
<nav class="toc">{toc}</nav>
{body}
</body></html>"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(doc, encoding="utf-8")
    print(f"Wrote {args.out} ({len(rows)} edits, {args.out.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
