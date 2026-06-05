"""Self-contained HTML: TRELLIS.2 SS-decoded OCCUPANCY at 1024 vs 512 edit-res.

Companion to ``ab_res_html.py`` (which compares the textured named-view
renders).  This one compares the **3D occupancy / SLat** the editor actually
operates on, by embedding the per-edit panels produced by
``viz_edit_mask_3d.py`` (2D edit | S1 mask@16³ | S2 mask | shape SLat PCA |
tex SLat PCA).  For each edit present in BOTH the 1024 and the 512 panel dir
it stacks the two panels (1024 above 512) so the grid coarsening (64³ vs 32³)
is directly visible.  base64-embedded, single file, no GPU.

Prereq — render the panels for both trees first::

    for t in _exp_masked_posthoc_r1024 _exp_masked_posthoc_r512_pad0; do
      CUDA_VISIBLE_DEVICES=6 TRELLIS2_DIR=/mnt/zsn/3dobject/TRELLIS.2 \
      /mnt/zsn/miniconda3/envs/trellis2/bin/python scripts/viz/viz_edit_mask_3d.py \
        --in-root data/Pxform_v2/$t/objects/08 --objs all \
        --out-dir data/Pxform_v2/_scratch/mask_viz_3d/$t
    done

Then::

    python scripts/viz/ab_occ_html.py            # masked (default)
    python scripts/viz/ab_occ_html.py flowedit   # after rendering those panels
"""
from __future__ import annotations
import base64
import sys
from pathlib import Path

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
VIZ = ROOT / "data/Pxform_v2/_scratch/mask_viz_3d"

# recipe -> (1024 panel dir, 512 panel dir, human title)
RECIPES = {
    "masked": dict(
        a="_exp_masked_posthoc_r1024", b="_exp_masked_posthoc_r512_pad0",
        title="TRELLIS.2 masked-edit · SS-decoded occupancy · 1024 (64³) vs 512 (32³)",
    ),
    "flowedit": dict(
        a="_exp_flowedit_free_r1024", b="_exp_flowedit_free_r512",
        title="TRELLIS.2 FlowEdit-edit · SS-decoded occupancy · 1024 (64³) vs 512 (32³)",
    ),
}

NOTE = (
    "每个 edit 两行面板（上 <b>1024 → 64³</b>，下 <b>512 → 32³</b>），每行 6 列："
    "<code>2D 原图 | 2D 编辑 | S1 mask@16³ | S2 mask | shape SLat PCA | tex SLat PCA</code>。"
    "图中 voxel 是 <b>SS-decode 出来的占据</b>（保存的 <code>coords0/coords_new</code> + shape/tex SLat 的 PCA），"
    "不是 mesh decode。S1 实际仍在 64³ 跑、之后 max-pool 降到 S2 网格；这里 <code>coords0</code> 存的是 S2 最终网格，"
    "所以反映「喂给 S2 的占据」。对比 512 行明显更<b>块状</b>（token≈1/8），且编辑区 coverage 在更粗网格上系统性偏高。"
)


def b64(p: Path) -> str:
    if not p.is_file():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def main() -> None:
    recipe = sys.argv[1] if len(sys.argv) > 1 else "masked"
    cfg = RECIPES[recipe]
    A = VIZ / cfg["a"]
    B = VIZ / cfg["b"]
    out = ROOT / f"data/Pxform_v2/_scratch/ab_compare/occ_1024_vs_512_{recipe}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not A.is_dir() or not B.is_dir():
        sys.exit(f"missing panel dir(s): {A} | {B}\n"
                 f"render them first with viz_edit_mask_3d.py (see this file's docstring).")

    eids = sorted(p.name for p in A.glob("*.png")
                  if p.name != "_INDEX3D.png" and (B / p.name).is_file())

    blocks = []
    for eid in eids:
        a_img = b64(A / eid)
        b_img = b64(B / eid)
        name = eid[:-4]
        blocks.append(f"""
        <div class="block">
          <div class="eid">{name}</div>
          <div class="row"><span class="tag a">1024 · 64³</span><img src="{a_img}"></div>
          <div class="row"><span class="tag b">512 · 32³</span><img src="{b_img}"></div>
        </div>""")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{cfg['title']}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#14161a;color:#e8eaed;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 12px}}
 .note{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:14px 16px;margin-bottom:20px;line-height:1.6;font-size:13px;max-width:1200px;border-left:3px solid #ffb454}}
 .note code{{background:#0d0f12;padding:1px 5px;border-radius:4px;color:#7fd1ff;font-size:12px}}
 .block{{background:#1e2127;border:1px solid #2c3038;border-radius:8px;padding:10px 12px;margin-bottom:16px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:6px;color:#e8eaed}}
 .row{{display:flex;align-items:center;gap:10px;margin:4px 0}}
 .row img{{max-width:100%;height:auto;border-radius:4px;display:block}}
 .tag{{font-size:11px;font-weight:600;writing-mode:vertical-rl;transform:rotate(180deg);padding:6px 2px;border-radius:4px;white-space:nowrap}}
 .tag.a{{color:#7fd1ff}} .tag.b{{color:#ffb454}}
</style></head><body>
<h1>{cfg['title']}</h1>
<div class="note"><p>{NOTE}</p>
<p style="color:#8a8f98">{len(eids)} edits · 图片全部 base64 内嵌，单文件可离线打开。</p></div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(eids)} edits, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
