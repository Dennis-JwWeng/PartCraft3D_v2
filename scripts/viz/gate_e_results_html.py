#!/usr/bin/env python3
"""Gate-E (VLM visual-quality judge) results → self-contained HTML.

For each edit that has a ``gate_e_judge.json`` sidecar (written by the
gate_quality stage), renders one card:

  * BEFORE / AFTER 5-view strips (exactly the images the 2x5 collage is built
    from — the VLM's visual input),
  * the EXACT prompt sent to the VLM (static system prompt shown once at top,
    per-edit user message in the card),
  * the FULL VLM verdict (pass/fail, the four booleans, visual_quality 1-5,
    artifact_free, reason, prompt_quality, improved_prompt).

    python scripts/viz/gate_e_results_html.py [tree_dir]
    # default tree: data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore
"""
from __future__ import annotations
import base64, io, json, sys
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
    else:
        col = "#aeb4bd"; txt = esc(val)
    return f'<span class="b"><span class="bk">{esc(label)}</span><span class="bv" style="color:{col}">{txt}</span></span>'


def vq_badge(vq) -> str:
    try:
        n = int(vq)
    except (TypeError, ValueError):
        n = 0
    col = "#e0625e" if n <= 2 else ("#e0b04e" if n == 3 else "#54c08a")
    return f'<span class="b"><span class="bk">visual_quality</span><span class="bv" style="color:{col};font-weight:700">{esc(vq)} / 5</span></span>'


def score_badge(label: str, val, thr=None, hi_is_good: bool = True) -> str:
    """Graded 0/1-5 badge (v2): colour by value, optionally annotate threshold."""
    try:
        n = int(val)
    except (TypeError, ValueError):
        n = -1
    if n < 0:
        col = "#aeb4bd"
    elif thr is not None:
        col = "#54c08a" if n >= thr else "#e0625e"
    else:
        col = "#e0625e" if n <= 2 else ("#e0b04e" if n == 3 else "#54c08a")
    thr_txt = f' <span style="color:#6b7079">(≥{thr})</span>' if thr is not None else ""
    return (f'<span class="b"><span class="bk">{esc(label)}</span>'
            f'<span class="bv" style="color:{col};font-weight:700">{esc(val)}{thr_txt}</span></span>')


def defects_badge(defects) -> str:
    if not defects:
        return ('<span class="b"><span class="bk">mesh_defects</span>'
                '<span class="bv" style="color:#54c08a">none ✓</span></span>')
    items = ", ".join(esc(d) for d in defects) if isinstance(defects, list) else esc(defects)
    return ('<span class="b"><span class="bk">mesh_defects</span>'
            f'<span class="bv" style="color:#e0625e">{items}</span></span>')


def main() -> None:
    tree = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore"
    if not tree.is_absolute():
        tree = ROOT / tree
    objroot = tree / "objects" / SHARD

    # collect (obj, eid, judge_dict, edit_dir, before_dir)
    cards, sysprompt = [], None
    n_pass = n_fail = 0
    for objdir in sorted(objroot.glob("*/")):
        obj = objdir.name
        bdir = objdir / "gate_views"
        for ed in sorted((objdir / "edits_3d").glob("*/")):
            jp = ed / "gate_e_judge.json"
            if not jp.is_file():
                continue
            rec = json.loads(jp.read_text())
            sysprompt = sysprompt or rec.get("prompt", {}).get("system")
            cards.append((obj, ed.name, rec, ed, bdir))
            if rec.get("pass"):
                n_pass += 1
            else:
                n_fail += 1

    blocks = []
    for obj, eid, rec, ed, bdir in cards:
        j = rec.get("judge") or {}
        user = rec.get("prompt", {}).get("user", "")
        thr = rec.get("thresholds") or {}
        jver = (rec.get("judge_version") or ("v2" if "mesh_quality" in j else "v1")).lower()
        passed = rec.get("pass")
        pcol = "#54c08a" if passed else "#e0625e"
        ptxt = "PASS" if passed else "FAIL"
        if jver == "v2":
            # mesh-integrity-first, graded execution — lead with the two graded axes
            verdict = "".join([
                score_badge("mesh_quality", j.get("mesh_quality"),
                            thr=thr.get("min_mesh_quality")),
                score_badge("edit_strength", j.get("edit_strength"),
                            thr=thr.get("min_edit_strength")),
                defects_badge(j.get("mesh_defects")),
                badge("correct_region", j.get("correct_region")),
                badge("preserve_other", j.get("preserve_other")),
                vq_badge(j.get("visual_quality")),
                badge("artifact_free", j.get("artifact_free")),
                badge("prompt_quality", f'{j.get("prompt_quality")} / 5'),
            ])
        else:
            verdict = "".join([
                badge("edit_executed", j.get("edit_executed")),
                badge("correct_region", j.get("correct_region")),
                badge("preserve_other", j.get("preserve_other")),
                vq_badge(j.get("visual_quality")),
                badge("artifact_free", j.get("artifact_free")),
                badge("prompt_quality", f'{j.get("prompt_quality")} / 5'),
            ])
        rows = (f'<div class="vr"><span class="tag" style="color:#aeb4bd">BEFORE</span>{strip(bdir,"before_view")}</div>'
                f'<div class="vr"><span class="tag" style="color:#7fd1ff">AFTER (512 edit)</span>{strip(ed,"after_view")}</div>')
        # v2: show the 2D condition images that were ALSO fed to the VLM as reference
        cond_fed = rec.get("condition_imgs") or []
        if jver in ("v2", "v3") and cond_fed:
            e2d = ed.parent.parent / "edits_2d"
            edited_p, input_p = e2d / f"{eid}_edited.png", e2d / f"{eid}_input.png"
            cond_cells = ""
            if edited_p.is_file():
                cond_cells += (f'<td><div class="ctag" style="color:#ffd27f">EDIT REF (Image 2)</div>'
                               f'<img class="cimg" src="{b64(edited_p)}"></td>')
            if input_p.is_file():
                cond_cells += (f'<td><div class="ctag" style="color:#9aa0a8">INPUT (Image 3)</div>'
                               f'<img class="cimg" src="{b64(input_p)}"></td>')
            if cond_cells:
                rows += (f'<div class="vr"><span class="tag" style="color:#ffd27f">CONDITION 图(也输入给 VLM 作参考)</span>'
                         f'<table><tr>{cond_cells}</tr></table></div>')
        blocks.append(f"""
        <div class="block">
          <div class="eid">{esc(eid)} <span class="obj">({esc(obj[:10])})</span>
            <span class="pf" style="background:{pcol}">{ptxt}</span>
            <span class="jv" style="background:{'#3a5fae' if jver=='v2' else ('#2f7d4f' if jver=='v3' else '#444a55')}">{esc(jver)}</span>
            <span class="thr">{(
                f"阈值: mesh≥{esc(thr.get('min_mesh_quality','?'))} · edit≥{esc(thr.get('min_edit_strength','?'))} · vq≥{esc(thr.get('min_visual_quality','?'))}" + (' · preserve_other' if thr.get('require_preserve_other') else '')
              ) if jver=='v2' else (
                f"阈值(v1严格+mesh硬闸): edit_executed · vq≥{esc(thr.get('min_visual_quality','?'))}" + (' · preserve_other' if thr.get('require_preserve_other') else '') + ' · artifact_free'
              ) if jver=='v3' else (
                f"阈值: vq≥{esc(thr.get('min_visual_quality','?'))}" + (' · preserve_other' if thr.get('require_preserve_other') else '')
              )}</span>
          </div>
          {rows}
          <div class="cols">
            <div class="pcol"><div class="ph">VLM USER PROMPT(发给判别器的 per-edit 文本)</div><pre>{esc(user)}</pre></div>
            <div class="vcol">
              <div class="ph">VLM 判别 (gate_e_judge.json · judge)</div>
              <div class="badges">{verdict}</div>
              <div class="reason"><b>reason:</b> {esc(j.get('reason',''))}</div>
              <div class="reason"><b>improved_prompt:</b> {esc(j.get('improved_prompt',''))}</div>
              <div class="reason"><b>improved_after_desc:</b> {esc(j.get('improved_after_desc',''))}</div>
            </div>
          </div>
        </div>""")

    summary = f'{len(cards)} edits judged · <b style="color:#54c08a">{n_pass} PASS</b> · <b style="color:#e0625e">{n_fail} FAIL</b>'
    sys_block = (f"""<details class="method"><summary>VLM SYSTEM PROMPT(静态,全 batch 共用 · KV-cache)+ 判别逻辑</summary>
      <pre class="sys">{esc(sysprompt or '(missing)')}</pre>
      <p style="color:#aeb4bd;font-size:12px">判别器输入 = 上面 system + 每条 edit 的 user 文本 + 一张 2×5 拼图(上排 BEFORE 5 视角、下排 AFTER 5 视角,同列同相机)。</p>
      <p style="color:#aeb4bd;font-size:12px"><b style="color:#cdd2d9">PASS 规则 · v1</b>(<code>_passes_quality_thresholds</code>): <code>edit_executed=true</code> 且 <code>visual_quality ≥ min_visual_quality(默认3)</code>
      且 <code>correct_region=true</code> 且(若 <code>require_preserve_other</code>)<code>preserve_other=true</code>。</p>
      <p style="color:#aeb4bd;font-size:12px"><b style="color:#7fdca0">PASS 规则 · v3</b>(<code>_passes_quality_thresholds_v3</code> · v1 严格语义 + mesh 硬闸 + 喂 condition 图):
      = v1 全部条件(<code>edit_executed=true</code> 且 <code>vq≥3</code> 且 <code>correct_region</code> 且 <code>preserve_other</code>)<b>原样不放松</b>,
      再加一条硬闸 <code>artifact_free=true</code>(任何看穿洞/撕裂/漂浮碎块/ghost 重影 → false → FAIL)。判别器额外输入 EDIT REF + INPUT 两张 2D condition 图。</p>
      <p style="color:#aeb4bd;font-size:12px"><b style="color:#7fa8ff">PASS 规则 · v2</b>(<code>_passes_quality_thresholds_v2</code> · mesh 优先 + 分级执行):
      <code>mesh_quality ≥ min_mesh_quality(默认4,硬闸)</code> 且 <code>edit_strength ≥ min_edit_strength(默认2,放松)</code>
      且 <code>visual_quality ≥ min_visual_quality(默认2)</code> 且 <code>correct_region=true</code> 且(若 <code>require_preserve_other</code>)<code>preserve_other=true</code>。
      mesh 破损即便编辑完美也 FAIL;编辑只要有可见尝试(strength≥2)即放行。</p>
      </details>""")

    out = tree / "_scratch_gate_e_results.html"
    # prefer the shared ab_compare dir for discoverability
    shared = ROOT / "data/Pxform_v2/_scratch/ab_compare" / f"gate_e_{tree.name}.html"
    shared.parent.mkdir(parents=True, exist_ok=True)
    out = shared
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Gate-E · {esc(tree.name)}</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 4px}} .note{{font-size:13px;color:#aeb4bd;margin-bottom:14px}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:10px 12px;margin-bottom:16px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:6px}} .obj{{color:#8a8f98;font-weight:400}}
 .pf{{font-size:11px;font-weight:700;color:#0e1013;border-radius:4px;padding:1px 8px;margin-left:8px}}
 .jv{{font-size:10px;font-weight:700;color:#e8eaed;border-radius:4px;padding:1px 6px;margin-left:6px;letter-spacing:.5px}}
 .thr{{font-size:11px;color:#8a8f98;margin-left:8px;font-weight:400}}
 .vr{{margin:3px 0}} .tag{{font-size:11px;font-weight:600;display:block;margin:2px 0}}
 table{{border-collapse:collapse}} td{{padding:2px;vertical-align:top}} img{{display:block;border-radius:3px;width:150px;background:#fff}}
 .cimg{{width:200px}} .ctag{{font-size:10px;font-weight:600;margin:2px 0}}
 .cols{{display:flex;gap:16px;margin-top:8px;align-items:flex-start}}
 .pcol{{flex:0 0 380px}} .vcol{{flex:1 1 auto;min-width:0}}
 .ph{{font-size:11px;color:#ffd27f;font-weight:600;margin-bottom:4px}}
 pre{{background:#0b0d10;border:1px solid #2c3038;border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.5;color:#cdd2d9;white-space:pre-wrap;margin:0}}
 .badges{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}}
 .b{{background:#0b0d10;border:1px solid #2c3038;border-radius:5px;padding:2px 7px;font-size:11.5px}}
 .bk{{color:#8a8f98;margin-right:5px}} .bv{{font-family:monospace}}
 .reason{{font-size:12.5px;color:#cdd2d9;line-height:1.5;margin:3px 0}} .reason b{{color:#7fd1ff}}
 .method{{background:#13161c;border:1px solid #2c3038;border-radius:8px;padding:10px 16px;margin-bottom:18px}}
 .method summary{{cursor:pointer;font-weight:600;font-size:14px;color:#ffd27f}}
 pre.sys{{margin-top:8px;max-height:420px;overflow:auto}}
 code{{background:#23262d;padding:1px 5px;border-radius:4px;color:#ffb454;font-size:12px}}
</style></head><body>
<h1>Gate-E(VLM 视觉质量判别)· {esc(tree.name)}</h1>
<div class="note">{summary}。每块:BEFORE/AFTER 5 视角(判别器视觉输入)+ 发给 VLM 的 user prompt + VLM 完整判别 JSON。</div>
{sys_block}
{''.join(blocks)}
</body></html>""")
    print(f"wrote {out}  ({len(cards)} edits: {n_pass} pass / {n_fail} fail, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
