#!/usr/bin/env python3
"""Side-by-side Gate-E comparison: v1 vs v3 (or any two sidecar suffixes).

For each edit that has BOTH sidecars it renders one card:
  * BEFORE / AFTER 5-view strips (the VLM's visual input),
  * the 2D CONDITION images (EDIT REF / INPUT) that v3 additionally fed in,
  * two verdict panels side by side — v1 (left) and v3 (right) — each with the
    full judge badges + reason, and a PASS/FAIL chip.
Cards whose verdict FLIPPED between the two versions are highlighted and listed
in the summary.

  python scripts/viz/gate_e_compare_html.py [tree_dir] [--a SUFFIX] [--b SUFFIX]
    # defaults: A=v1 (gate_e_judge.v1.json), B=v3 (gate_e_judge.json)
    # tree default: data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore
"""
from __future__ import annotations
import base64, json, sys
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
VIEWS = ["front", "right", "back", "left", "down"]
SHARD = "08"


def b64(p: Path) -> str:
    return ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            if p.is_file() else "")


def strip(d: Path, prefix: str) -> str:
    return "<table><tr>" + "".join(
        f'<td><img src="{b64(d / f"{prefix}_{v}.png")}"></td>' for v in VIEWS) + "</tr></table>"


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def badge(label: str, val) -> str:
    if isinstance(val, bool):
        col = "#54c08a" if val else "#e0625e"
        txt = "✓ true" if val else "✗ false"
    elif val is None:
        col = "#6b7079"; txt = "—"
    else:
        col = "#aeb4bd"; txt = esc(val)
    return f'<span class="b"><span class="bk">{esc(label)}</span><span class="bv" style="color:{col}">{txt}</span></span>'


def vq_badge(vq, floor=3) -> str:
    try:
        n = int(vq)
    except (TypeError, ValueError):
        n = 0
    col = "#e0625e" if n < floor else "#54c08a"
    return f'<span class="b"><span class="bk">visual_quality</span><span class="bv" style="color:{col};font-weight:700">{esc(vq)} / 5</span></span>'


def panel(rec: dict, *, is_v3: bool) -> str:
    if rec is None:
        return '<div class="panel"><div class="miss">(no verdict)</div></div>'
    j = rec.get("judge") or {}
    passed = rec.get("pass")
    pcol = "#1f7a4d" if passed else "#a23433"
    ptxt = "PASS" if passed else "FAIL"
    badges = [
        badge("edit_executed", j.get("edit_executed")),
        badge("correct_region", j.get("correct_region")),
        badge("preserve_other", j.get("preserve_other")),
        vq_badge(j.get("visual_quality")),
    ]
    # artifact_free is the ADDED hard gate in v3 → emphasise it there
    af = j.get("artifact_free")
    if is_v3:
        col = "#54c08a" if af else "#e0625e"
        badges.append(f'<span class="b gate"><span class="bk">artifact_free⛔</span>'
                      f'<span class="bv" style="color:{col};font-weight:700">{"✓ true" if af else "✗ false"}</span></span>')
    else:
        badges.append(badge("artifact_free", af))
    badges.append(badge("prompt_quality", f'{j.get("prompt_quality")} / 5'))
    cond = rec.get("condition_imgs") or []
    cond_note = (f'<div class="cnote">+ {len(cond)} condition 图输入</div>' if cond else "")
    return (f'<div class="panel">'
            f'<div class="phead"><span class="pf" style="background:{pcol}">{ptxt}</span>'
            f'<span class="vlabel">{"v3 (v1严格 + condition + artifact_free硬闸)" if is_v3 else "v1 (基线)"}</span>{cond_note}</div>'
            f'<div class="badges">{"".join(badges)}</div>'
            f'<div class="reason"><b>reason:</b> {esc(j.get("reason",""))}</div>'
            f'</div>')


def main() -> None:
    argv = [a for a in sys.argv[1:]]
    a_suf, b_suf = ".v1", ""
    if "--a" in argv:
        a_suf = argv[argv.index("--a") + 1]
    if "--b" in argv:
        b_suf = argv[argv.index("--b") + 1]
    pos = [a for a in argv if not a.startswith("--") and a not in (a_suf, b_suf)]
    tree = Path(pos[0]) if pos else ROOT / "data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore"
    if not tree.is_absolute():
        tree = ROOT / tree
    objroot = tree / "objects" / SHARD

    def load(p: Path):
        return json.loads(p.read_text()) if p.is_file() else None

    cards = []
    n_a_pass = n_b_pass = 0
    flips = {"PF": [], "FP": []}
    for objdir in sorted(objroot.glob("*/")):
        obj = objdir.name
        bdir = objdir / "gate_views"
        e2d = objdir / "edits_2d"
        for ed in sorted((objdir / "edits_3d").glob("*/")):
            ra = load(ed / f"gate_e_judge{a_suf}.json")
            rb = load(ed / f"gate_e_judge{b_suf}.json")
            if ra is None and rb is None:
                continue
            pa = bool(ra and ra.get("pass"))
            pb = bool(rb and rb.get("pass"))
            n_a_pass += pa; n_b_pass += pb
            flip = ""
            if pa and not pb:
                flip = "PF"; flips["PF"].append(ed.name)
            elif not pa and pb:
                flip = "FP"; flips["FP"].append(ed.name)
            cards.append((obj, ed.name, ra, rb, ed, bdir, e2d, flip))

    blocks = []
    for obj, eid, ra, rb, ed, bdir, e2d, flip in cards:
        cond_cells = ""
        for tag, suf, col in [("EDIT REF", "_edited", "#ffd27f"), ("INPUT", "_input", "#9aa0a8")]:
            p = e2d / f"{eid}{suf}.png"
            if p.is_file():
                cond_cells += f'<td><div class="ctag" style="color:{col}">{tag}</div><img class="cimg" src="{b64(p)}"></td>'
        cond_row = (f'<div class="vr"><span class="tag" style="color:#ffd27f">CONDITION（v3 额外喂给 VLM）</span>'
                    f'<table><tr>{cond_cells}</tr></table></div>') if cond_cells else ""
        flip_chip = ""
        if flip == "PF":
            flip_chip = '<span class="flip pf">FLIP v1✓→v3✗ (新判破损)</span>'
        elif flip == "FP":
            flip_chip = '<span class="flip fp">FLIP v1✗→v3✓ (condition救回)</span>'
        cls = "block flip-pf" if flip == "PF" else ("block flip-fp" if flip == "FP" else "block")
        blocks.append(f"""
        <div class="{cls}">
          <div class="eid">{esc(eid)} <span class="obj">({esc(obj[:10])})</span>{flip_chip}</div>
          <div class="vr"><span class="tag" style="color:#aeb4bd">BEFORE</span>{strip(bdir,"before_view")}</div>
          <div class="vr"><span class="tag" style="color:#7fd1ff">AFTER (512 edit)</span>{strip(ed,"after_view")}</div>
          {cond_row}
          <div class="cmp">{panel(ra, is_v3=False)}{panel(rb, is_v3=True)}</div>
        </div>""")

    total = len(cards)
    fp = "、".join(s[:22] for s in flips["FP"]) or "无"
    pf = "、".join(s[:22] for s in flips["PF"]) or "无"
    summary = (f'{total} 条编辑 · <b style="color:#9aa0a8">v1: {n_a_pass}P/{total-n_a_pass}F</b> '
               f'→ <b style="color:#7fdca0">v3: {n_b_pass}P/{total-n_b_pass}F</b> · '
               f'<b style="color:#e0625e">{len(flips["PF"])} 个 v1✓→v3✗</b>(新抓破损)、'
               f'<b style="color:#54c08a">{len(flips["FP"])} 个 v1✗→v3✓</b>(condition救回)')

    out = ROOT / "data/Pxform_v2/_scratch/ab_compare" / f"gate_e_compare_v1_vs_v3_{tree.name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Gate-E v1 vs v3 · {esc(tree.name)}</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 4px}} .note{{font-size:13px;color:#aeb4bd;margin-bottom:14px;line-height:1.6}}
 .legend{{background:#13161c;border:1px solid #2c3038;border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:12.5px;line-height:1.7;color:#cdd2d9}}
 .legend code{{background:#23262d;padding:1px 5px;border-radius:4px;color:#ffb454}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:10px 12px;margin-bottom:16px}}
 .flip-pf{{border-color:#e0625e;box-shadow:0 0 0 1px #e0625e55}}
 .flip-fp{{border-color:#54c08a;box-shadow:0 0 0 1px #54c08a55}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:6px}} .obj{{color:#8a8f98;font-weight:400}}
 .flip{{font-size:11px;font-weight:700;border-radius:4px;padding:1px 8px;margin-left:10px}}
 .flip.pf{{background:#e0625e;color:#1a0d0d}} .flip.fp{{background:#54c08a;color:#0d1a12}}
 .vr{{margin:3px 0}} .tag{{font-size:11px;font-weight:600;display:block;margin:2px 0}}
 table{{border-collapse:collapse}} td{{padding:2px;vertical-align:top}} img{{display:block;border-radius:3px;width:132px;background:#fff}}
 .cimg{{width:180px}} .ctag{{font-size:10px;font-weight:600;margin:2px 0}}
 .cmp{{display:flex;gap:12px;margin-top:8px}}
 .panel{{flex:1 1 0;min-width:0;background:#0b0d10;border:1px solid #2c3038;border-radius:7px;padding:8px 10px}}
 .phead{{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}}
 .pf{{font-size:11px;font-weight:700;color:#fff;border-radius:4px;padding:1px 9px}}
 .vlabel{{font-size:11px;color:#aeb4bd}} .cnote{{font-size:10.5px;color:#ffd27f}}
 .badges{{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:7px}}
 .b{{background:#16181d;border:1px solid #2c3038;border-radius:5px;padding:2px 6px;font-size:11px}}
 .b.gate{{border-color:#5a4a2a;background:#1d1810}}
 .bk{{color:#8a8f98;margin-right:4px}} .bv{{font-family:monospace}}
 .reason{{font-size:12px;color:#cdd2d9;line-height:1.5}} .reason b{{color:#7fd1ff}}
 .miss{{color:#6b7079;font-size:12px}}
</style></head><body>
<h1>Gate-E 对比:v1 基线 ↔ v3(v1 严格 + condition + artifact_free 硬闸)· {esc(tree.name)}</h1>
<div class="note">{summary}</div>
<div class="legend">
 <b>v1</b>:PASS = <code>edit_executed</code> 且 <code>vq≥3</code> 且 <code>correct_region</code> 且 <code>preserve_other</code>。<code>artifact_free</code> 被忽略。<br>
 <b>v3</b>:<b>v1 全部条件一字不放松</b>,额外把 <code>artifact_free</code> 变成<b>硬闸</b>(看穿洞/撕裂/漂浮碎块/ghost 重影 → false → FAIL),并把 2D EDIT REF + INPUT 两张 condition 图也喂给 VLM。<br>
 每条编辑:上方 BEFORE/AFTER 5 视角(判别器视觉输入)+ CONDITION 图,下方左 v1 / 右 v3 判别;翻转卡片描边高亮。
</div>
{''.join(blocks)}
</body></html>""")
    print(f"wrote {out}")
    print(f"  {total} edits · v1 {n_a_pass}P/{total-n_a_pass}F → v3 {n_b_pass}P/{total-n_b_pass}F")
    print(f"  v1→v3 P→F (新抓破损): {pf}")
    print(f"  v1→v3 F→P (救回): {fp}")
    print(f"  size {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
