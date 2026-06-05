#!/usr/bin/env python3
"""32³ voxel diagnostic for the restore remedy — is it pasting MISALIGNED voxels?

For each edit shared by the pad2 (no-restore) and pad2_restore trees, loads the
saved 32³ occupancy from edits_3d/<eid>/latents/ss.npz and renders matplotlib
voxel scatters (2 angles each):

  1. SOURCE  coords0   (sidecar shape-VAE encode, 32³)         — grey
  2. pad2    coords_new (S1//2, NO restore): body=blue, edit=red
  3. restore coords_new (S1//2 + restore):   body=blue, edit=red, RESTORED=magenta
  4. OVERLAY pad2 body (blue, faint) + RESTORED (magenta)      — do the magenta
     voxels sit ON the body shell, or float off it (错位)?

RESTORED = (pad2_restore.coords_new) \ (pad2.coords_new) — the voxels the remedy
actually added.  If they cluster away from the blue body → confirmed misalignment.

    python scripts/viz/diag_restore_voxel_32.py
"""
from __future__ import annotations
import base64, io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
DATA = ROOT / "data/Pxform_v2"
T_PAD2 = DATA / "_exp_masked_perstep_r512_pad2"
T_REST = DATA / "_exp_masked_perstep_r512_pad2_restore"
SHARD = "08"
G = 32
ANGLES = [(20, -60), (20, 40)]


def _load_ss(tree: Path, obj: str, eid: str):
    f = tree / "objects" / SHARD / obj / "edits_3d" / eid / "latents" / "ss.npz"
    if not f.is_file():
        return None
    d = np.load(f, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _keys(c):
    c = c.astype(np.int64)
    return set((c[:, 0] * G * G + c[:, 1] * G + c[:, 2]).tolist())


def _scatter(ax, coords, color, s=18, alpha=1.0):
    if len(coords) == 0:
        return
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, marker="s",
               s=s, alpha=alpha, edgecolors="none", depthshade=True)


def _panel(title, layers, angle):
    fig = plt.figure(figsize=(3.2, 3.2), dpi=90)
    ax = fig.add_subplot(111, projection="3d")
    for coords, color, s, a in layers:
        _scatter(ax, coords, color, s, a)
    ax.set_xlim(0, G); ax.set_ylim(0, G); ax.set_zlim(0, G)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=angle[0], azim=angle[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=8, color="#222")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.93)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _split_edit(coords, edit_keys):
    k = (coords.astype(np.int64)[:, 0] * G * G + coords[:, 1] * G + coords[:, 2])
    m = np.array([int(x) in edit_keys for x in k])
    return coords[~m], coords[m]   # body, edit


def main() -> None:
    keys = []
    for ed in sorted(T_REST.glob(f"objects/{SHARD}/*/edits_3d/*")):
        if (ed / "latents" / "ss.npz").is_file():
            keys.append((ed.parent.parent.name, ed.name))

    blocks = []
    for obj, eid in keys:
        r = _load_ss(T_REST, obj, eid); p = _load_ss(T_PAD2, obj, eid)
        if r is None:
            continue
        c0 = r["coords0"].astype(np.int64)
        cnR = r["coords_new"].astype(np.int64)
        eg_keys = _keys(r["edit_grid"])
        cnP = p["coords_new"].astype(np.int64) if p else cnR
        restored_k = _keys(cnR) - _keys(cnP)
        kR = (cnR[:, 0] * G * G + cnR[:, 1] * G + cnR[:, 2])
        restored = cnR[np.array([int(x) in restored_k for x in kR])] if len(cnR) else cnR

        bodyP, editP = _split_edit(cnP, eg_keys)
        bodyR, editR = _split_edit(cnR, eg_keys)

        imgs = []
        for ang in ANGLES:
            imgs.append((
                _panel(f"SOURCE coords0 ({len(c0)})", [(c0, "#888888", 16, 0.9)], ang),
                _panel(f"pad2 NO-restore ({len(cnP)})",
                       [(bodyP, "#3a7bd5", 16, 0.9), (editP, "#e74c3c", 16, 0.95)], ang),
                _panel(f"pad2+RESTORE (+{len(restored)})",
                       [(bodyR, "#3a7bd5", 16, 0.9), (editR, "#e74c3c", 16, 0.95),
                        (restored, "#ff00ff", 34, 1.0)], ang),
                _panel("OVERLAY body+restored",
                       [(bodyP, "#3a7bd5", 14, 0.35), (restored, "#ff00ff", 40, 1.0)], ang),
            ))
        cells = ""
        for row in imgs:
            cells += "<tr>" + "".join(f'<td><img src="{u}"></td>' for u in row) + "</tr>"
        blocks.append(f"""
        <div class="block"><div class="eid">{eid} <span class="obj">({obj[:10]})</span>
          &nbsp; restored=<b style="color:#c0c">{len(restored)}</b> voxels @32³</div>
          <table class="hdr"><tr><th>SOURCE coords0</th><th>pad2 (no restore)</th>
          <th>pad2+restore (magenta=added)</th><th>overlay body+restored</th></tr></table>
          <table>{cells}</table></div>""")

    out = DATA / "_scratch" / "ab_compare" / "diag_restore_voxel_32.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>restore 32³ voxel diag</title><style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e1013;color:#e8eaed;margin:0;padding:20px}}
 h1{{font-size:18px}} .note{{font-size:13px;color:#aeb4bd;margin-bottom:14px;line-height:1.6}}
 .block{{background:#16181d;border:1px solid #2c3038;border-radius:8px;padding:8px 10px;margin-bottom:14px}}
 .eid{{font-size:13px;font-weight:600;margin-bottom:4px}} .obj{{color:#8a8f98;font-weight:400}}
 table{{border-collapse:collapse}} td{{padding:1px}} th{{font-size:10px;color:#aeb4bd;font-weight:600;width:288px}}
 img{{display:block;border-radius:3px;width:280px}}
</style></head><body>
<h1>restore 占据 32³ 诊断 — 补回的体素是否错位</h1>
<div class="note">
<b style="color:#ff00ff">品红 = restore 实际补回的体素</b>(= pad2+restore 的 coords_new 减去 pad2 的 coords_new)。
蓝=body,红=编辑区。第 4 列把品红叠到淡蓝 body 上:<b>若品红贴在蓝壳上 → 对齐;若飘在外面/错层 → 错位</b>。
SOURCE 是 32³ sidecar(shape-VAE)占据,与 S1//2 占据来自不同编码器,这是错位的根源假设。{len(keys)} edits。</div>
{''.join(blocks)}
</body></html>"""
    out.write_text(html)
    print(f"wrote {out}  ({len(keys)} edits, {out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
