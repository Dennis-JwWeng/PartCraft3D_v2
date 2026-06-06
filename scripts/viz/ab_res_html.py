"""Self-contained HTML: TRELLIS.2 3D edit at 1024 vs 512 edit-resolution.

For each edit present in BOTH the 1024 tree and its 512 sibling, shows a
3-row × 5-view block of the already-rendered named views (base64-embedded,
no GPU needed):

    row BEFORE     (shared original, decoded latents)
    row AFTER@1024
    row AFTER@512

plus a header documenting the current S1 / S2 recipe and exactly what 512
changes.  One standalone .html, nothing external to load.

    # masked recipe (default): _exp_masked_posthoc_r1024 vs _exp_masked_posthoc_r512_pad0
    python scripts/viz/ab_res_html.py
    # FlowEdit + free recipe:
    python scripts/viz/ab_res_html.py flowedit
"""
from __future__ import annotations
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
VIEWS = ["front", "right", "back", "left", "down"]
CELL = 240

# recipe → (1024 tree, 512 tree, human label, S1 text, S2 text)
RECIPES = {
    "masked": dict(
        a="data/Pxform_v2/_exp_masked_posthoc_r1024/objects/08",
        b="data/Pxform_v2/_exp_masked_posthoc_r512_pad0/objects/08",
        title="TRELLIS.2 masked-edit · 1024 vs 512 edit-resolution",
        s1=(
            "<b>S1 (sparse structure / occupancy)</b> — <code>trellis2_s1_mode: masked</code>. "
            "反演原 SS latent（cfg off）后，按 16³ keep-mask 在编辑区重新去噪重绘 occupancy；"
            "<code>trellis2_s1_contact_soft: true</code> 用 contact-aware 距离变换<b>软 mask</b>（动态 sigma，"
            "按编辑面与保留几何的接触比例决定羽化）；<code>trellis2_ss_align_t1: true</code> 换 TRELLIS.1 的"
            "温和 SS 调度（steps25 / cfg5 / interval[.5,1] / rt3，抗大件塌陷）。"
            "<b>SS VAE 固定 64³ occupancy ↔ 16³ latent，1024 与 512 完全相同。</b>"),
        s2=(
            "<b>S2 (shape + texture SLat)</b> — <code>trellis2_s2_anchor_mode: posthoc</code>. "
            "整场<b>自由生成</b>（vanilla 质量，无逐步锚定）后，结尾把保留 body token <b>硬粘回原 clean latent</b> → "
            "实心编辑区 + 逐位精确 body（代价：边界可能 1-voxel seam）。tex 走 per-step 锚定 masked 路径。"),
    ),
    "t1ss": dict(
        a="data/Pxform_v2/_exp_t1ss_native_r1024_pad4_texrestore/objects/08",
        b="data/Pxform_v2/_exp_t1ss_native_r512_pad4_texrestore/objects/08",
        title="TRELLIS.2 t1ss-native masked-edit (pad4 · texrestore) · 1024 vs 512 edit-resolution",
        s1=(
            "<b>S1 (sparse structure / occupancy)</b> — <code>trellis2_s1_mode: masked</code> + "
            "<code>trellis2_s1_ss_model: t1</code>（进程内 NATIVE TRELLIS.1 SS flow + DINOv2 条件，非桥接）。"
            "反演原 SS latent 后按 16³ keep-mask 在编辑区重绘 occupancy；<code>trellis2_s1_contact_soft: true</code> "
            "contact-aware 软 mask；<code>trellis2_ss_align_t1: true</code> 用 T1 温和 SS 调度（抗大件塌陷）；"
            "<code>trellis2_s1_pad: 4</code> 把 64³ edit grid 切比雪夫膨胀 4；"
            "<code>trellis2_s2_restore_preserved: true</code> 同帧 64³ 把 mask 外被丢的源 body 体素补回。"
            "<b>SS VAE 固定 64³ occupancy ↔ 16³ latent，1024 与 512 完全相同。</b>"),
        s2=(
            "<b>S2 (shape + texture SLat)</b> — SHAPE: <code>trellis2_s2_anchor_mode: perstep</code>（masked 逐步锚定，"
            "实心编辑部件）。TEXTURE: <code>trellis2_s2_tex_anchor_mode: posthoc</code> = <b>posthoc-restore</b>："
            "编辑图下自由生成整张纹理 → 终点把 <b>P1 编码的原始 tex latent</b> 通过 shape 桥接 src_idx <b>硬贴回</b>"
            "保留 token → 保留区 decode 出逐像素原始材质，只有编辑区重画。"),
    ),
    "flowedit": dict(
        a="data/Pxform_v2/_exp_flowedit_free_r1024/objects/08",
        b="data/Pxform_v2/_exp_flowedit_free_r512/objects/08",
        title="TRELLIS.2 FlowEdit-edit · 1024 vs 512 edit-resolution",
        s1=(
            "<b>S1 (sparse structure)</b> — <code>trellis2_s1_mode: flowedit</code>. "
            "不反演、不 mask：用 源→目标 引导速度差 ODE 驱动 clean SS latent，编辑由条件之差自然涌现"
            "（对称 CFG gs_src==gs_tgt==7.5，canonical frame）。SS VAE 固定 64³↔16³，1024/512 相同。"),
        s2=(
            "<b>S2 (shape + texture SLat)</b> — <code>trellis2_s2_anchor_mode: free</code>. "
            "整场自由生成、<b>不</b>粘回 body：body 仅通过沿用原 occupancy 坐标在结构上保留，表面全连贯无缝"
            "（代价：body 非逐位 latent-identical）。"),
    ),
}

RES_NOTE = (
    "<b>512 改了什么</b>：TRELLIS.2 里 SLat 坐标网格 = 分辨率//16，所以 S2 在 <b>32³</b>（512）vs <b>64³</b>（1024），"
    "用 <code>shape/tex_slat_flow_model_512</code> + conds@512 + <code>decode_latent(.,.,512)</code>；body 锚点来自 mesh 在 "
    "grid-512 的重编码 sidecar（<code>p1_encode/shape_slat_e512.npz</code>，32³）。"
    "<b>S1 两者完全一样（64³）</b>：先在 64³ 出 coords_new，再 max-pool 64³→32³ 喂 S2。单一开关 <code>trellis2_edit_res</code>。")


def _load(p: Path):
    im = cv2.imread(str(p))
    if im is None:
        return np.full((CELL, CELL, 3), 240, np.uint8)
    return cv2.resize(im, (CELL, CELL))


def b64(im_bgr) -> str:
    rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cell(src: Path) -> str:
    return f'<td><img src="{b64(_load(src))}"></td>'


def main() -> None:
    recipe = sys.argv[1] if len(sys.argv) > 1 else "masked"
    cfg = RECIPES[recipe]
    A = ROOT / cfg["a"]; B = ROOT / cfg["b"]
    out = ROOT / f"data/Pxform_v2/_scratch/ab_compare/res_1024_vs_512_{recipe}.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    # common edits: every edit dir present in BOTH trees with rendered afters
    rows = []
    for objdir in sorted(B.glob("*/")):
        obj = objdir.name
        for ed in sorted((objdir / "edits_3d").glob("*/")):
            eid = ed.name
            a_ed = A / obj / "edits_3d" / eid
            if not (ed / "after_view_front.png").is_file():
                continue
            if not (a_ed / "after_view_front.png").is_file():
                continue
            rows.append((obj, eid, A / obj, B / obj, a_ed, ed))

    blocks = []
    for obj, eid, a_obj, b_obj, a_ed, b_ed in rows:
        head = "".join(f'<th>{v}</th>' for v in VIEWS)
        before = "".join(cell(a_obj / "gate_views" / f"before_view_{v}.png") for v in VIEWS)
        a_after = "".join(cell(a_ed / f"after_view_{v}.png") for v in VIEWS)
        b_after = "".join(cell(b_ed / f"after_view_{v}.png") for v in VIEWS)
        blocks.append(f"""
        <div class="block">
          <div class="eid">{eid} <span class="obj">({obj[:12]})</span></div>
          <table>
            <tr><th class="rh"></th>{head}</tr>
            <tr><th class="rh">BEFORE</th>{before}</tr>
            <tr><th class="rh a">AFTER · 1024</th>{a_after}</tr>
            <tr><th class="rh b">AFTER · 512</th>{b_after}</tr>
          </table>
        </div>""")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{cfg['title']}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 12px}}
 .recipe{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:14px 16px;margin-bottom:20px;line-height:1.6;font-size:13px;max-width:1100px}}
 .recipe p{{margin:6px 0}}
 .recipe code{{background:#0d0f12;padding:1px 5px;border-radius:4px;color:#7fd1ff;font-size:12px}}
 .res{{border-left:3px solid #ffb454;padding-left:10px;margin-top:10px}}
 .block{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:10px 12px;margin-bottom:16px;display:inline-block}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:6px}} .obj{{color:#8a8f98;font-weight:400}}
 table{{border-collapse:collapse}} td{{padding:2px}} img{{display:block;border-radius:3px}}
 th{{font-size:11px;color:#aeb4bd;font-weight:500;padding:2px 4px}}
 .rh{{text-align:right;white-space:nowrap;min-width:88px}}
 .rh.a{{color:#7fd1ff}} .rh.b{{color:#ffb454}}
</style></head><body>
<h1>{cfg['title']}</h1>
<div class="recipe">
  <p>{cfg['s1']}</p>
  <p>{cfg['s2']}</p>
  <p class="res">{RES_NOTE}</p>
  <p style="color:#8a8f98">{len(rows)} edits · 每块 3 行（BEFORE / <span style="color:#7fd1ff">AFTER@1024</span> / <span style="color:#ffb454">AFTER@512</span>）× 5 视角。图片全部 base64 内嵌，单文件可离线打开。</p>
</div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(rows)} edits, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
