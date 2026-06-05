#!/usr/bin/env python3
"""T1-vs-T2 SS-flow A/B — same recipe (perstep+restore+512+white), only S1 flow differs.

T1 arm  : _exp_t1ss_perstep_r512_pad2_restore   (TRELLIS.1 ss_flow_img_dit_L_16l8, via bridge)
T2 arm  : _exp_masked_perstep_r512_pad2_restore (TRELLIS.2 ss_flow_img_dit_1_3B_64, masked S1)
Both: pad=2 edit_grid, same-frame 64³ restore, perstep S2, edit_res=512, white decode.

Per edit: condition input/edited + BEFORE 5 views + T1-S1→S2 5 views + T2-S1→S2 5 views.

    python scripts/viz/ab_t1_vs_t2_ssflow_html.py
"""
from __future__ import annotations
import base64, io
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
DATA = ROOT / "data/Pxform_v2"
VIEWS = ["front", "right", "back", "left", "down"]
SHARD = "08"
ARMS = [
    ("S1 = TRELLIS.1 flow", "_exp_t1ss_perstep_r512_pad2_restore",   "#7fd1ff"),
    ("S1 = TRELLIS.2 flow", "_exp_masked_perstep_r512_pad2_restore", "#ffb454"),
]


# ── methodology block (current config + code path), embedded in the HTML ──
METHOD = """
<details class="method" open><summary>实验配置 &amp; 代码思路（点击折叠）</summary>
<div class="mbody">

<h3>0. 目的 / baseline</h3>
<p><b>唯一变量 = S1 的 SS flow 模型</b>。两臂共享同一份 TRELLIS.1 SS VAE
（<code>ss_enc/ss_dec_conv3d_16l8</code>，T2 逐字复用 T1 的）、同一编辑区、同一膨胀、同一同帧
restore、同一 S2 策略、同一 512 解码 + 白模。仅 S1 的 flow 权重 + 图像 cond 不同：</p>
<ul>
<li><b>T1 臂</b> <code>_exp_t1ss_perstep_r512_pad2_restore</code> — <code>ss_flow_img_dit_L_16l8</code>（DINOv2 cond），
占据离线在 vinedresser3d 算好，经 <code>trellis2_ss1_coords_dir</code> 桥注入。</li>
<li><b>T2 臂</b> <code>_exp_masked_perstep_r512_pad2_restore</code> — <code>ss_flow_img_dit_1_3B_64</code>，管线内 masked S1。</li>
</ul>

<h3>1. 当前配置（config 旋钮）</h3>
<table class="cfg">
<tr><td>trellis2_edit_res</td><td>512</td><td>SLat 网格 = 512//16 = <b>32³</b>（1024→64³）</td></tr>
<tr><td>trellis2_s1_mode</td><td>masked（T2）/ ss1 桥（T1）</td><td>S1 走 masked 重绘 / 注入 T1 占据</td></tr>
<tr><td>trellis2_ss_align_t1</td><td>true</td><td>S1 采样 <b>schedule</b> 换成 T1 的（steps25 / cfg5 / interval[.5,1] / rt3）；不换权重</td></tr>
<tr><td>trellis2_s1_pad</td><td>2</td><td>64³ 编辑区 Chebyshev box 膨胀 2 格</td></tr>
<tr><td>trellis2_s2_restore_preserved</td><td>true</td><td>同帧 64³ SS 把 mask 外被删的源 body 补回</td></tr>
<tr><td>trellis2_s2_anchor_mode</td><td>perstep</td><td>S2 每步把 preserved token 锚回反演的原始隐变量</td></tr>
<tr><td>trellis2_force_white_model</td><td>true</td><td>只到 S2 shape，零纹理隐变量 → 灰模，跳过纹理阶段</td></tr>
</table>

<h3>2. Mask 策略（<code>trellis2_part_mask.py</code>）</h3>
<p><b>64³ 编辑区是纯"碰到即占据"</b>，无阈值：<code>part_edit_grid_64()</code> 把目标 part 在 64³ 体素化
（dual-grid 表面壳 → block keys → 稠密布尔图）。<b>阈值只在降到 16³ keep mask 时出现</b>：
<code>edit_grid_64_to_keep16(thresh=0.1)</code> —— 一个 4×4×4 块里 ≥10% 子格被触及才算"编辑"，其余为
<b>keep（保留）</b>，S1 重绘时锚定到反演的原始 SS 隐变量。</p>

<h3>3. 膨胀怎么做（pad=2）</h3>
<p><code>_dilate_grid(grid, pad)</code> = 对 64³ 稠密布尔图做 <b>Chebyshev box 膨胀</b>：
<code>F.max_pool3d(grid, kernel_size=2*pad+1=5, stride=1, padding=2)</code>。给 S1 重绘留出"长出不同
尺寸/形状新 part"的空房间。膨胀<b>同时</b>作用于 16³ keep mask 和 S2 编辑区（二者都从这张 64³ 图派生），
保证 S1/S2 编辑范围一致。</p>

<h3>4. S1 阶段（结构 / SS）</h3>
<p>占据 → SS VAE 编码成 16³ 隐变量 → masked 重绘：在<b>原图</b>下 RF 反演得到加噪轨迹，再在<b>编辑图</b>下
前向重绘，<b>keep 区每步用反演轨迹覆盖</b>（硬锚定），编辑区自由生成 → <code>ss_dec</code> 解出 64³ 占据
<code>coords_new</code>。T1/T2 此处<b>仅 flow 权重 + 图像 cond 不同</b>，VAE、schedule、keep mask 全一致。</p>

<h3>5. 同帧 restore（为什么 32³ 下不碎）</h3>
<p>参考帧 = <code>ss_dec(ss_enc(occ(coords0)))</code> —— 纯 VAE roundtrip，<b>flow 无关</b>，且与
<code>coords_new = ss_dec(z_s_new)</code> 用<b>同一个 64³ SS 解码帧</b>。
<code>restore_preserved_occupancy()</code> 把"在编辑区<b>外</b>、但 <code>coords_new</code> 里<b>丢了</b>"的源体素
并回（编辑区内的 part 不动，仍自由生长）。同帧 ⇒ 补回体素与编辑后 body 逐 voxel 对齐，不悬空；
之前用 grid-512 shape-VAE sidecar 做并集（不同编码器）才会错位/碎裂。补全在 64³ 完成后再 //2 到 32³。</p>

<h3>6. 降采样 → S2 阶段（perstep，<code>trellis2_edit_stages.py</code>）</h3>
<p>64³ → 32³：<code>downsample_coords(//2)</code> + <code>downsample_edit_grid(max_pool factor=2)</code>。
S2 用 <code>shape_slat_flow_model_512</code>：<code>build_coord_bridge</code> 区分 preserved / 编辑 token；
对原始 clean 隐变量 <code>invert_clean</code> 反演；<code>perstep</code> = 每步回调把 preserved token 覆盖回反演值
（body 精确保持），编辑 token 在编辑图下生成。最后 <code>decode_latent(shape, zero_tex, 512)</code> 出 512 灰模。</p>

</div></details>
"""


def b64(p: Path) -> str:
    return ("data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
            if p.is_file() else "")


def thumb(p: Path, px=240) -> str:
    from PIL import Image
    if not p.is_file():
        return ""
    im = Image.open(str(p)).convert("RGB"); im.thumbnail((px, px))
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def strip(d: Path, prefix: str) -> str:
    return "<table><tr>" + "".join(
        f'<td><img src="{b64(d / f"{prefix}_{v}.png")}"></td>' for v in VIEWS) + "</tr></table>"


def main() -> None:
    present = [(lbl, DATA / d, c) for lbl, d, c in ARMS if (DATA / d).is_dir()]
    keys = set()
    for _, t, _ in present:
        for ed in t.glob(f"objects/{SHARD}/*/edits_3d/*"):
            if (ed / "after_view_front.png").is_file():
                keys.add((ed.parent.parent.name, ed.name))
    keys = sorted(keys)

    blocks = []
    for obj, eid in keys:
        ci = ce = bdir = None
        for _, t, _ in present:
            od = t / "objects" / SHARD / obj
            if ci is None and (od / "edits_2d" / f"{eid}_input.png").is_file():
                ci = od / "edits_2d" / f"{eid}_input.png"; ce = od / "edits_2d" / f"{eid}_edited.png"
            if bdir is None and (od / "gate_views" / "before_view_front.png").is_file():
                bdir = od / "gate_views"
        rows = ""
        if bdir:
            rows += f'<div class="vr"><span class="tag" style="color:#aeb4bd">BEFORE</span>{strip(bdir,"before_view")}</div>'
        for lbl, t, col in present:
            ed = t / "objects" / SHARD / obj / "edits_3d" / eid
            if (ed / "after_view_front.png").is_file():
                rows += f'<div class="vr"><span class="tag" style="color:{col}">{lbl}</span>{strip(ed,"after_view")}</div>'
        blocks.append(f"""
        <div class="block"><div class="eid">{eid} <span class="obj">({obj[:10]})</span></div>
          <div class="grid"><div class="conds">
            <div class="cap">INPUT</div><img class="ci" src="{thumb(ci) if ci else ''}">
            <div class="cap">EDITED</div><img class="ci" src="{thumb(ce) if ce else ''}">
          </div><div class="vox">{rows}</div></div></div>""")

    legend = " vs ".join(f'<b style="color:{c}">{l}</b>' for l, _, c in present)
    out = DATA / "_scratch" / "ab_compare" / "t1_vs_t2_ssflow_r512.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>T1 vs T2 SS flow</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 6px}} .note{{font-size:13px;color:#aeb4bd;margin-bottom:16px;line-height:1.6}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}} .obj{{color:#8a8f98;font-weight:400}}
 .grid{{display:flex;gap:14px;align-items:flex-start}} .conds{{flex:0 0 auto}}
 .cap{{font-size:10px;color:#aeb4bd;margin:2px 0}} .ci{{display:block;border-radius:4px;margin-bottom:6px;max-width:220px}}
 .vox{{flex:1 1 auto;min-width:0}} .vr{{margin:3px 0}} .tag{{font-size:11px;font-weight:600;display:block;margin:2px 0}}
 table{{border-collapse:collapse}} td{{padding:2px}} img{{display:block;border-radius:3px;width:185px;background:#fff}}
 .method{{background:#13161c;border:1px solid #2c3038;border-radius:8px;padding:10px 16px;margin-bottom:18px;font-size:13px;line-height:1.65}}
 .method summary{{cursor:pointer;font-weight:600;font-size:14px;color:#ffd27f}}
 .method h3{{font-size:13.5px;margin:14px 0 4px;color:#7fd1ff}} .method p{{margin:4px 0;color:#cdd2d9}}
 .method ul{{margin:4px 0 4px 18px;color:#cdd2d9}} .method code{{background:#23262d;padding:1px 5px;border-radius:4px;color:#ffb454;font-size:12px}}
 .method .mbody{{margin-top:6px}}
 table.cfg{{border-collapse:collapse;margin:4px 0;font-size:12px}}
 table.cfg td{{border:1px solid #2c3038;padding:3px 8px;vertical-align:top}}
 table.cfg td:first-child{{color:#ffb454;font-family:monospace;white-space:nowrap}} table.cfg td:nth-child(2){{color:#9fe0a0;white-space:nowrap}}
</style></head><body>
<h1>T1 vs T2 SS-flow — 同配方(perstep+同帧restore+512+白模),唯一变量=S1 flow 模型</h1>
<div class="note">{legend}。SS VAE / 编辑区(pad2) / restore / S2 全相同;T1 走 bridge 注入占据,T2 走 masked S1。{len(keys)} edits。</div>
{METHOD}
{''.join(blocks)}</body></html>""")
    print(f"wrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
