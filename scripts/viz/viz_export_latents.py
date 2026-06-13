#!/usr/bin/env python
"""3D viz of an EXPORT edit dir (self-contained, flat layout).

Export edit dir holds (after-state + mask):
  ss.npz        ss[1,8,16,16,16]   -- real SS latent (S1)
  shape_slat.npz feats[N,32] coords[N,3] @32  (shape == tex coords)
  tex_slat.npz   feats[N,32] coords[N,3] @32
  mask.npz       mask_keep_ss[16,16,16] mask_keep_slat[N] mask_keep_slat_before[Nb] selected_part_ids[P]

Panels (each 3D voxels, 2 azimuths as 2 rows):
  (a) after shape @32  keep(blue)/edit(red) by mask_keep_slat
  (b) ss struct @16 (=ds2 after coords) keep/edit by mask_keep_ss
  (c) ||ss latent|| @16 viridis magnitude over the 8 channels

Usage: viz_export_latents.py <edit_dir> <out.png>
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

KEEP = np.array([0.27, 0.51, 0.71, 0.9])
EDIT = np.array([0.86, 0.08, 0.24, 0.95])


def voxgrid(coords, R):
    g = np.zeros((R, R, R), dtype=bool)
    c = np.clip(coords.astype(int), 0, R - 1)
    g[c[:, 0], c[:, 1], c[:, 2]] = True
    return g


def draw(ax, occ, fc, title, R, azim):
    ax.voxels(occ, facecolors=fc, edgecolor=(0, 0, 0, 0.08), linewidth=0.2)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlim(0, R); ax.set_ylim(0, R); ax.set_zlim(0, R)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def build_panels(ed):
    ss = np.load(os.path.join(ed, "ss.npz"))["ss"][0]      # [8,16,16,16]
    sh = np.load(os.path.join(ed, "shape_slat.npz"))
    mk = np.load(os.path.join(ed, "mask.npz"))
    ac = sh["coords"]
    m_slat = mk["mask_keep_slat"]
    m_ss = mk["mask_keep_ss"].astype(bool)

    panels = []
    # (a) after + slat mask
    occ_a = voxgrid(ac, 32)
    fca = np.zeros(occ_a.shape + (4,))
    cc = np.clip(ac.astype(int), 0, 31)
    fca[cc[:, 0], cc[:, 1], cc[:, 2]] = np.where(m_slat[:, None] == 1, KEEP[None], EDIT[None])
    panels.append((occ_a, fca,
                   f"(a) after @32  keep={int((m_slat==1).sum())} / edit={int((m_slat==0).sum())}", 32))

    # (b) ss struct @16 = ds2(after) + mask_keep_ss
    c16 = np.unique(np.clip(ac.astype(int) // 2, 0, 15), axis=0)
    occ_s = voxgrid(c16, 16)
    fcs = np.zeros(occ_s.shape + (4,))
    for x, y, z in c16:
        fcs[x, y, z] = KEEP if m_ss[x, y, z] else EDIT
    panels.append((occ_s, fcs, f"(b) ss struct @16  keep_ss={int(m_ss.sum())}", 16))

    # (c) ||ss latent|| @16
    mag = np.linalg.norm(ss, axis=0)
    thr = float(np.percentile(mag, 60))
    occ_m = mag > thr
    vmin = float(mag[occ_m].min()) if occ_m.any() else 0.0
    norm = Normalize(vmin=vmin, vmax=float(mag.max()))
    cols = cm.viridis(norm(mag)); cols[..., 3] = 0.9
    fcm = np.zeros(occ_m.shape + (4,)); fcm[occ_m] = cols[occ_m]
    panels.append((occ_m, fcm, f"(c) ||ss latent|| @16  (>{thr:.2f}, max {mag.max():.2f})", 16))
    return panels


def main():
    ed = sys.argv[1]; out = sys.argv[2]
    panels = build_panels(ed)
    azims = [-60, 120]
    fig = plt.figure(figsize=(15, 9))
    for r, az in enumerate(azims):
        for c, (occ, fc, title, R) in enumerate(panels):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
            draw(ax, occ, fc, title if r == 0 else "", R, az)
            if c == 0:
                ax.text2D(-0.04, 0.5, f"azim={az}", transform=ax.transAxes,
                          rotation=90, va="center", fontsize=9, color="0.4")
    name = "/".join(ed.rstrip("/").split("/")[-3:])
    fig.suptitle(f"{name}   [blue=keep / red=edit;  ss=16 dense, shape/tex coords=32 sparse]", fontsize=11)
    fig.tight_layout(rect=(0.01, 0, 1, 0.96))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
