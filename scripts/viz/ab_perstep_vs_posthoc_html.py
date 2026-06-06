#!/usr/bin/env python3
"""Self-contained A/B HTML: S2 shape PERSTEP vs POSTHOC-restore.

Per edit present in BOTH trees, a 3-row × 5-view block of the rendered named
views (base64-embedded):

    BEFORE            (shared original, decoded latents)
    AFTER · perstep   (s2_anchor_mode=perstep — per-step anchor to INVERTED original)
    AFTER · posthoc   (s2_anchor_mode=posthoc — free gen SOLID + paste BEFORE-encoded body; NO inversion)

Header documents the recipe diff + the measured timing (posthoc ~1.22x/edit,
skips the shape invert_clean).  One standalone .html.

    python scripts/viz/ab_perstep_vs_posthoc_html.py
"""
from __future__ import annotations
import base64
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
VIEWS = ["front", "right", "back", "left", "down"]
SHARD = "08"

A = ROOT / "data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore"    # perstep
B = ROOT / "data/Pxform_v2/_exp_t1ss_native_r512_pad4_shaperestore"  # posthoc


def b64(p: Path) -> str:
    return ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            if p.is_file() else "")


def row(cells) -> str:
    return "<tr>" + "".join(f'<td><img src="{c}"></td>' for c in cells) + "</tr>"


def main() -> None:
    objroot_b = B / "objects" / SHARD
    blocks = []
    n = 0
    for objdir in sorted(objroot_b.glob("*/")):
        obj = objdir.name
        bdir = A / "objects" / SHARD / obj / "gate_views"   # shared BEFORE
        for ed in sorted((objdir / "edits_3d").glob("*/")):
            eid = ed.name
            a_ed = A / "objects" / SHARD / obj / "edits_3d" / eid
            if not (ed / "after_view_front.png").is_file():
                continue
            if not (a_ed / "after_view_front.png").is_file():
                continue
            n += 1
            head = "".join(f"<th>{v}</th>" for v in VIEWS)
            before = [b64(bdir / f"before_view_{v}.png") for v in VIEWS]
            a_after = [b64(a_ed / f"after_view_{v}.png") for v in VIEWS]
            b_after = [b64(ed / f"after_view_{v}.png") for v in VIEWS]
            blocks.append(f"""
            <div class="block">
              <div class="eid">{eid} <span class="obj">({obj[:12]})</span></div>
              <table>
                <tr><th class="rh"></th>{head}</tr>
                <tr><th class="rh">BEFORE</th>{''.join(f'<td><img src="{c}"></td>' for c in before)}</tr>
                <tr><th class="rh a">AFTER · perstep</th>{''.join(f'<td><img src="{c}"></td>' for c in a_after)}</tr>
                <tr><th class="rh b">AFTER · posthoc</th>{''.join(f'<td><img src="{c}"></td>' for c in b_after)}</tr>
              </table>
            </div>""")

    out = ROOT / "data/Pxform_v2/_scratch/ab_compare/s2_perstep_vs_posthoc.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>S2 shape perstep vs posthoc</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 12px}}
 .recipe{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:14px 16px;margin-bottom:20px;line-height:1.7;font-size:13px;max-width:1100px}}
 .recipe code{{background:#0d0f12;padding:1px 5px;border-radius:4px;color:#7fd1ff;font-size:12px}}
 .speed{{border-left:3px solid #54c08a;padding-left:10px;margin-top:10px}}
 .block{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:10px 12px;margin-bottom:16px;display:inline-block}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:6px}} .obj{{color:#8a8f98;font-weight:400}}
 table{{border-collapse:collapse}} td{{padding:2px}} img{{display:block;border-radius:3px;width:150px;background:#fff}}
 th{{font-size:11px;color:#aeb4bd;font-weight:500;padding:2px 4px}}
 .rh{{text-align:right;white-space:nowrap;min-width:110px}}
 .rh.a{{color:#ffb454}} .rh.b{{color:#7fdca0}}
</style></head><body>
<h1>TRELLIS.2 S2 shape:PERSTEP ↔ POSTHOC-restore · 512 edit-res · {n} edits</h1>
<div class="recipe">
 <p><b style="color:#ffb454">perstep</b>(<code>trellis2_s2_anchor_mode: perstep</code>):shape S2 先 <code>invert_clean</code> 反演原始 shape SLat,再把保留 body token <b>每步锚回反演轨迹</b>。body 近似精确,但编辑区 token 全程被反演原始邻居拽偏 → 生成部件 decode 成<b>薄壳/看穿洞</b>。</p>
 <p><b style="color:#7fdca0">posthoc</b>(<code>trellis2_s2_anchor_mode: posthoc</code>):<b>自由生成</b>整场 shape(编辑区实心,无锚定)→ 结尾把 <b>P1 编码的原始 body latent</b>(<code>shape_slat_e512.npz</code>)硬贴回保留 token(texture restore 的 shape 版)。body <b>逐位精确</b>,编辑区更实;<b>跳过 invert_clean</b>。</p>
 <p class="speed"><b>实测耗时(2 卡并发,13 对象 / 32 编辑):</b> perstep 每条 <b>17.6s</b> → posthoc <b>14.4s</b>,配对每条快 <b style="color:#7fdca0">3.2s · 1.22×</b>(省掉的就是 shape 反演的 25 步)。body 两版逐像素一致。</p>
 <p style="color:#8a8f98">每块 3 行(BEFORE / <span style="color:#ffb454">AFTER·perstep</span> / <span style="color:#7fdca0">AFTER·posthoc</span>)× 5 视角,base64 内嵌单文件。</p>
</div>
{''.join(blocks)}
</body></html>""")
    print(f"wrote {out}  ({n} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
