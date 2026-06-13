#!/usr/bin/env python
"""3D visualization of one edit's latents + mask.

Panels (each a 3D voxel plot):
  (a) before shape coords @32³            — gray
  (b) after  shape coords @32³            — keep(blue)/edit(red) by mask_keep_slat
  (c) ss structure @16³ = downsample2(after) — keep/edit by mask_keep_ss
  (d) ss latent magnitude @16³ (‖z‖ over 8 ch) — viridis heatmap

Usage: python scripts/viz/viz_latents_mask_3d.py <obj_dir> <eid> [out.png]
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


def voxgrid(coords, R, vals=None, fill=True):
    """coords[N,3] int -> bool grid [R,R,R]; optional per-voxel value grid."""
    g = np.zeros((R, R, R), dtype=bool)
    c = np.clip(coords.astype(int), 0, R - 1)
    g[c[:, 0], c[:, 1], c[:, 2]] = True
    if vals is None:
        return g
    vg = np.full((R, R, R), np.nan, dtype=float)
    vg[c[:, 0], c[:, 1], c[:, 2]] = vals
    return g, vg


def draw(ax, occ, facecolors, title, R, azim=-60):
    ax.voxels(occ, facecolors=facecolors, edgecolor=(0, 0, 0, 0.08), linewidth=0.2)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlim(0, R); ax.set_ylim(0, R); ax.set_zlim(0, R)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    obj = sys.argv[1]
    eid = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "latents_mask_3d.png"
    lat = os.path.join(obj, "edits_3d", eid, "latents")

    bsh = np.load(os.path.join(obj, "p1_encode", "shape_slat_e512.npz"))
    ash = np.load(os.path.join(lat, "shape_slat.npz"))
    ssl = np.load(os.path.join(lat, "ss_latent.npz"))["ss"][0]   # [8,16,16,16]
    mk = np.load(os.path.join(lat, "mask.npz"))
    m_slat = mk["mask_keep_slat"]            # [N_after] 1=keep 0=edit
    m_ss = mk["mask_keep_ss"].astype(bool)   # [16,16,16] 1=keep

    bc = bsh["coords"]; ac = ash["coords"]
    print(f"before N={len(bc)}  after N={len(ac)}  edit-voxels(after)={int((m_slat==0).sum())}")

    KEEP = np.array([0.27, 0.51, 0.71, 0.9])   # steelblue
    EDIT = np.array([0.86, 0.08, 0.24, 0.95])   # crimson
    GRAY = np.array([0.6, 0.6, 0.6, 0.85])

    panels = []  # (occ, facecolors, title, R)

    # (a) before
    occ_b = voxgrid(bc, 32)
    fcb = np.zeros(occ_b.shape + (4,)); fcb[occ_b] = GRAY
    panels.append((occ_b, fcb, f"(a) before shape @32³  (N={len(bc)})", 32))

    # (b) after + mask
    occ_a = voxgrid(ac, 32)
    fca = np.zeros(occ_a.shape + (4,))
    cc = np.clip(ac.astype(int), 0, 31)
    fca[cc[:, 0], cc[:, 1], cc[:, 2]] = np.where(m_slat[:, None] == 1, KEEP[None, :], EDIT[None, :])
    panels.append((occ_a, fca,
                   f"(b) after shape @32³  keep={int((m_slat==1).sum())} / edit(red)={int((m_slat==0).sum())}", 32))

    # (c) ss structure @16³ = downsample2(after) + mask_keep_ss
    c16 = np.unique(np.clip(ac.astype(int) // 2, 0, 15), axis=0)
    occ_s = voxgrid(c16, 16)
    fcs = np.zeros(occ_s.shape + (4,))
    for x, y, z in c16:
        fcs[x, y, z] = KEEP if m_ss[x, y, z] else EDIT
    panels.append((occ_s, fcs, f"(c) ss struct @16³ (=ds2 after)  keep_ss={int(m_ss.sum())} cells", 16))

    # (d) ss latent magnitude @16³
    mag = np.linalg.norm(ssl, axis=0)   # [16,16,16]
    thr = np.percentile(mag, 60)
    occ_m = mag > thr
    norm = Normalize(vmin=mag[occ_m].min(), vmax=mag.max())
    cols = cm.viridis(norm(mag)); cols[..., 3] = 0.9
    fcm = np.zeros(occ_m.shape + (4,)); fcm[occ_m] = cols[occ_m]
    panels.append((occ_m, fcm, f"(d) ‖ss latent‖ @16³  (>{thr:.2f}, max {mag.max():.2f})", 16))

    # two viewing angles (front-ish / back-ish) as two rows
    azims = [-60, 120]
    fig = plt.figure(figsize=(22, 11))
    for r, az in enumerate(azims):
        for cidx, (occ, fc, title, R) in enumerate(panels):
            ax = fig.add_subplot(2, 4, r * 4 + cidx + 1, projection="3d")
            draw(ax, occ, fc, title if r == 0 else "", R, azim=az)
            if cidx == 0:
                ax.text2D(-0.05, 0.5, f"azim={az}°", transform=ax.transAxes,
                          rotation=90, va="center", fontsize=10, color="0.4")

    fig.suptitle(f"{os.path.basename(obj)} / {eid}   "
                 f"[blue=keep / red=edit;  ss=16³ dense, shape/tex coords=32³ sparse]", fontsize=12)
    fig.tight_layout(rect=(0.01, 0, 1, 0.97))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
