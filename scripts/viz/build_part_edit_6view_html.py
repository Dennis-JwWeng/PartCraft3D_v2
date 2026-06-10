#!/usr/bin/env python3
"""HTML report: 6-view before / part-mask / after for 3D part edits.

Expects renders from ``render_part_edit_6views.py`` or compatible layout:
  <case_dir>/before_view_{view}.png
  <case_dir>/mask_view_{view}.png
  <case_dir>/after_view_{view}.png
  <case_dir>/meta.json  (optional)

Views: front, back, left, right, top, bottom (前后左右上下).

Usage:
  python scripts/viz/build_part_edit_6view_html.py \\
    --cases data/Pxform_v2/08/<obj>/six_views/<edit_id> \\
    --out report_part_edit_6view.html

  # batch: all edits under a shard tree
  python scripts/viz/build_part_edit_6view_html.py \\
    --scan data/Pxform_v2/08 --out report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from partcraft.render.ovox_views import SIX_VIEW_ORDER

VIEW_LABELS = {
    "front": "前",
    "back": "后",
    "left": "左",
    "right": "右",
    "top": "上",
    "bottom": "下",
}


def b64(p: Path) -> str:
    if not p.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def find_case_dirs(scan: Path) -> list[Path]:
    out = []
    for meta in scan.rglob("meta.json"):
        d = meta.parent
        if (d / f"before_view_front.png").is_file() or (d / f"before_view_{SIX_VIEW_ORDER[0]}.png").is_file():
            out.append(d)
    return sorted(out)


def case_section(case_dir: Path) -> str:
    meta = {}
    mp = case_dir / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text())
    edit_id = meta.get("edit_id", case_dir.name)
    pids = meta.get("selected_part_ids", [])
    prompt = meta.get("prompt", "")

    rows = []
    for v in SIX_VIEW_ORDER:
        before = case_dir / f"before_view_{v}.png"
        mask = case_dir / f"mask_view_{v}.png"
        after = case_dir / f"after_view_{v}.png"
        label = VIEW_LABELS.get(v, v)
        rows.append(f"""
        <tr>
          <td class="vlabel">{html.escape(label)}<br><span class="vname">{html.escape(v)}</span></td>
          <td><img src="{b64(before)}" alt="before {v}"></td>
          <td><img src="{b64(mask)}" alt="mask {v}"></td>
          <td><img src="{b64(after)}" alt="after {v}"></td>
        </tr>""")

    pid_str = ", ".join(str(x) for x in pids) if pids else "（global / 全物体）"
    return f"""
    <section class="case">
      <h2>{html.escape(edit_id)}</h2>
      <p class="meta">
        <strong>编辑部件</strong> part_id = [{html.escape(pid_str)}]（红）；
        <strong>其余部件</strong> mask 保留（灰）
      </p>
      {f'<p class="prompt"><strong>Prompt:</strong> {html.escape(prompt)}</p>' if prompt else ''}
      <table class="grid">
        <thead><tr>
          <th>视角</th><th>Before（编辑前）</th><th>Part Mask（选中/保留）</th><th>After（编辑后）</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def build_html(cases: list[Path], title: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = "".join(case_section(c) for c in cases)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px 32px; background: #0f1117; color: #e8eaed; }}
    h1 {{ font-size: 1.5rem; }}
    .intro {{ color: #9aa0a6; margin-bottom: 24px; line-height: 1.6; }}
    .case {{ background: #1a1d27; border: 1px solid #2a2f3d; border-radius: 12px;
             padding: 20px; margin-bottom: 28px; }}
    .case h2 {{ color: #6ea8fe; margin: 0 0 12px; font-size: 1.1rem; }}
    .meta, .prompt {{ font-size: 0.9rem; margin: 0 0 8px; }}
    .grid {{ width: 100%; border-collapse: collapse; }}
    .grid th, .grid td {{ border: 1px solid #2a2f3d; padding: 8px; vertical-align: middle; text-align: center; }}
    .grid th {{ background: #222633; font-size: 0.85rem; }}
    .vlabel {{ font-weight: 700; min-width: 56px; }}
    .vname {{ font-size: 0.75rem; color: #9aa0a6; font-weight: 400; }}
    .grid img {{ max-width: 220px; max-height: 220px; object-fit: contain; background: #111; border-radius: 6px; }}
    .legend {{ display: flex; gap: 20px; margin: 12px 0 24px; font-size: 0.85rem; }}
    .dot-red {{ color: #f28b82; }} .dot-grey {{ color: #9aa0a6; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="intro">
    3D Part 编辑对比 · 生成时间 {html.escape(now)} · 用例 {len(cases)}<br>
    选中 part 参与编辑（mask 行红色），其余 part 保持/掩膜（灰色）。
    六视图：前 / 后 / 左 / 右 / 上 / 下。
  </p>
  <div class="legend">
    <span class="dot-red">■ 选中 part（编辑目标）</span>
    <span class="dot-grey">■ 其余 part（mask 保留）</span>
  </div>
  {body}
</body>
</html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="case dirs with before/mask/after pngs")
    ap.add_argument("--scan", default="", help="scan tree for meta.json + six_views")
    ap.add_argument("--out", default="report_part_edit_6view.html")
    ap.add_argument("--title", default="3D Part 编辑 — 六视图 Before / Mask / After")
    args = ap.parse_args()

    cases: list[Path] = [Path(c) for c in args.cases] if args.cases else []
    if args.scan:
        cases.extend(find_case_dirs(Path(args.scan)))
    cases = sorted(set(c.resolve() for c in cases if c.is_dir()))
    if not cases:
        print("no case dirs found", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(cases, args.title), encoding="utf-8")
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
