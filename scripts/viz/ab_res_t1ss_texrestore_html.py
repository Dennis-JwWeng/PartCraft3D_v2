#!/usr/bin/env python3
"""1024-vs-512 edit-resolution A/B — t1ss-native masked recipe (pad4 · texrestore).

Same recipe both arms (native TRELLIS.1 SS flow + DINOv2 masked S1, pad4 +
same-frame restore, S2-shape per-step, S2-tex posthoc-restore, textured decode);
the ONLY variable is trellis2_edit_res → SLat grid res//16 = 64³ (1024) vs 32³ (512).
S1 is always 64³.

1024 arm : _exp_t1ss_native_r1024_pad4_texrestore
512  arm : _exp_t1ss_native_r512_pad4_texrestore

Per edit: condition input/edited (the FLUX 2D before/after) + BEFORE 5 views +
AFTER@1024 5 views + AFTER@512 5 views.  Self-contained HTML (base64-embedded).

    python scripts/viz/ab_res_t1ss_texrestore_html.py
"""
from __future__ import annotations
import base64, io
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
DATA = ROOT / "data/Pxform_v2"
VIEWS = ["front", "right", "back", "left", "down"]
SHARD = "08"
ARMS = [
    ("AFTER · edit_res 1024 (S2 @ 64³)", "_exp_t1ss_native_r1024_pad4_texrestore", "#7fd1ff"),
    ("AFTER · edit_res 512 (S2 @ 32³)",  "_exp_t1ss_native_r512_pad4_texrestore",  "#ffb454"),
]


# ── methodology block (current config + code path), embedded in the HTML ──
METHOD = """
<details class="method" open><summary>实验配置 &amp; 代码思路（点击折叠）</summary>
<div class="mbody">

<h3>0. 目的 / baseline</h3>
<p><b>唯一变量 = 编辑分辨率 <code>trellis2_edit_res</code>（1024 vs 512）</b>。两臂跑<b>完全相同的配方</b>、
同一批物体、同一份 FLUX 2D 编辑图（input/edited 条件），仅 S2 的 SLat 网格分辨率不同。
配方 = 当前最优:进程内 NATIVE TRELLIS.1 SS flow + DINOv2 masked S1(pad4 + 同帧 restore)、
S2 shape <b>per-step</b> 锚定、S2 texture <b>posthoc-restore</b>、带纹理解码(非白模)。</p>
<ul>
<li><b>1024 臂</b> <code>_exp_t1ss_native_r1024_pad4_texrestore</code> — SLat 网格 = 1024//16 = <b>64³</b>(原生)。
S2 直接读 64³ 主 <code>p1_encode/{shape,tex}_slat.npz</code>,<b>无</b> e512 sidecar、<b>无</b>降采样。</li>
<li><b>512 臂</b> <code>_exp_t1ss_native_r512_pad4_texrestore</code> — SLat 网格 = 512//16 = <b>32³</b>。
S1 仍 64³,再 max-pool 64³→32³ 喂 S2;S2 用 <code>shape/tex_slat_flow_model_512</code> + grid-512 重编码
sidecar(<code>{shape,tex}_slat_e512.npz</code>,32³)+ <code>decode_latent(.,.,512)</code>。</li>
</ul>
<p>两臂的 <code>p1_encode</code> / <code>edits_2d</code> / <code>phase1</code> 都从同一上游 symlink 复用,
所以 encode/VLM/FLUX/gate 完全一致,只有 3D 编辑阶段在各自分辨率重跑。</p>

<h3>1. 当前配置(config 旋钮)</h3>
<table class="cfg">
<tr><td>trellis2_edit_res</td><td>1024 / 512</td><td>SLat 网格 = res//16 → <b>64³ / 32³</b>(本 A/B 唯一变量)</td></tr>
<tr><td>trellis2_s1_mode</td><td>masked</td><td>反演原 SS latent → 编辑区按 16³ keep-mask 重绘 occupancy</td></tr>
<tr><td>trellis2_s1_ss_model</td><td>t1</td><td>S1 用进程内 <b>NATIVE</b> TRELLIS.1 <code>ss_flow_img_dit_L_16l8</code> + DINOv2 cond(非桥接)</td></tr>
<tr><td>trellis2_s1_contact_soft</td><td>true</td><td>contact-aware 软 mask(动态 sigma 羽化编辑/保留边界)</td></tr>
<tr><td>trellis2_ss_align_t1</td><td>true</td><td>S1 采样 schedule 换成 T1 的(steps25 / cfg5 / interval[.5,1] / rt3)</td></tr>
<tr><td>trellis2_s1_pad</td><td>4</td><td>64³ 编辑区 Chebyshev box 膨胀 4 格(给新尺寸/形状留空间)</td></tr>
<tr><td>trellis2_s2_restore_preserved</td><td>true</td><td>同帧 64³ 把 mask 外被删的源 body 体素补回</td></tr>
<tr><td>trellis2_s2_anchor_mode</td><td>perstep</td><td><b>SHAPE</b>:每步把 preserved token 锚回反演原始隐变量(实心编辑部件)</td></tr>
<tr><td>trellis2_s2_tex_anchor_mode</td><td>posthoc</td><td><b>TEXTURE</b>:posthoc-restore(见下),独立于 shape 选锚定</td></tr>
<tr><td>force_white_model</td><td>(无)</td><td>不跳纹理 → 带纹理 mesh 解码</td></tr>
</table>

<h3>2. S2 纹理 = posthoc-restore(核心,本配方的纹理保真做法)</h3>
<p><code>trellis2_s2_tex_anchor_mode: posthoc</code> + 存在 P1 编码的 before-tex 边车时,
<code>masked_tex_slat</code> 走 <b>posthoc-restore 分支</b>:在编辑图下<b>自由生成</b>整张纹理 → 终点把
<b>P1 编码的原始 tex latent</b>(<code>tex_slat.npz</code> @64³ / <code>tex_slat_e512.npz</code> @32³)
通过 shape 桥接 <code>src_idx</code> <b>硬贴回</b>保留 token。保留区 decode 出<b>逐像素原始材质</b>,只有编辑区重画。
日志:<code>tex anchor_mode=posthoc-restore (... N/M preserved tokens)</code>。前提:shape 与 tex 编码坐标逐位一致。</p>

<h3>3. S1 阶段(结构 / SS)— 1024/512 完全相同</h3>
<p>SS VAE(<code>ss_enc/ss_dec_conv3d_16l8</code>)固定 64³ occupancy ↔ 16³ latent。占据 → 编码 16³ →
masked 重绘:<b>原图</b>下 RF 反演,<b>编辑图</b>下前向重绘,keep 区每步硬锚回反演轨迹,编辑区自由生成 →
<code>ss_dec</code> 解出 64³ <code>coords_new</code>。<code>s1_pad: 4</code> 对 64³ 布尔图 max_pool(k=9) 膨胀。</p>

<h3>4. 同帧 restore + 降采样</h3>
<p>参考帧 = <code>ss_dec(ss_enc(occ(coords0)))</code>(纯 VAE roundtrip,与 coords_new 同一 64³ 解码帧)→
<code>restore_preserved_occupancy()</code> 把编辑区外、coords_new 里丢了的源体素并回(64³ 完成)。
512 臂再 <code>downsample_coords(//2)</code> + <code>downsample_edit_grid(max_pool=2)</code> 到 32³ 喂 S2;
1024 臂 64³ 直通(no-op)。</p>

<p style="color:#8a8f98">注:S1 占据逐位确定,但 S2 隐变量数值在独立进程间非 bit 级一致(fp16/flash_attn),
故同一行内细微差异不全是分辨率所致 —— 但分辨率效应是主导、也是本图要看的。</p>

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
            rows += f'<div class="vr"><span class="tag" style="color:#aeb4bd">BEFORE (原始)</span>{strip(bdir,"before_view")}</div>'
        for lbl, t, col in present:
            ed = t / "objects" / SHARD / obj / "edits_3d" / eid
            if (ed / "after_view_front.png").is_file():
                rows += f'<div class="vr"><span class="tag" style="color:{col}">{lbl}</span>{strip(ed,"after_view")}</div>'
        blocks.append(f"""
        <div class="block"><div class="eid">{eid} <span class="obj">({obj[:10]})</span></div>
          <div class="grid"><div class="conds">
            <div class="cap">INPUT (原视图 → FLUX)</div><img class="ci" src="{thumb(ci) if ci else ''}">
            <div class="cap">EDITED (FLUX 编辑图 = condition)</div><img class="ci" src="{thumb(ce) if ce else ''}">
          </div><div class="vox">{rows}</div></div></div>""")

    legend = " vs ".join(f'<b style="color:{c}">{l}</b>' for l, _, c in present)
    out = DATA / "_scratch" / "ab_compare" / "res_1024_vs_512_t1ss_texrestore.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>1024 vs 512 · t1ss texrestore</title><style>
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
<h1>1024 vs 512 edit-res — t1ss-native masked(pad4 · texrestore),唯一变量=编辑分辨率</h1>
<div class="note">{legend}。配方全同(native T1-SS masked S1 + pad4 + 同帧 restore + S2-shape perstep + S2-tex posthoc-restore + 带纹理);
S1 固定 64³,仅 S2 网格 64³(1024)vs 32³(512)。每块左列 = INPUT/EDITED 条件图,右列 = BEFORE / AFTER@1024 / AFTER@512 各 5 视角。{len(keys)} edits。</div>
{METHOD}
{''.join(blocks)}</body></html>""")
    print(f"wrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
