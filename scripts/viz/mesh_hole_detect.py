#!/usr/bin/env python3
"""Render-space see-through-hole detector for TRELLIS.2 edit previews.

A see-through hole in a thin shell shows up, in a white-background render, as a
patch of background colour that is ENCLOSED by foreground in the 2D projection
(you can see through the front surface to the background / inner back-face).
Unlike a watertightness test this only fires on holes that are actually VISIBLE
in a render, so a non-watertight-but-solid-looking mesh is not penalised.

Algorithm per view:
  1. background mask = near-white pixels.
  2. flood-fill the background mask inward from the image border → the TRUE
     exterior background (reachable from the edge).
  3. enclosed = near-white pixels NOT reached from the border → these are
     background-coloured holes surrounded by the object.
  4. keep connected components above a min area; report their sizes.

A per-edit decision aggregates the 5 named views.

CLI:
  python scripts/viz/mesh_hole_detect.py <edit_dir> [--save-overlay]
  python scripts/viz/mesh_hole_detect.py --scan <tree_dir>   # rank all edits
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
VIEWS = ["front", "right", "back", "left", "down"]

# tuning knobs
WHITE_THR = 244        # >= this on all-channels ≈ background white
MIN_HOLE_AREA = 60     # px; ignore specks (512x512 frame)
EDGE_MARGIN = 2        # border ring used to seed the exterior flood


def detect_holes(bgr: np.ndarray) -> "tuple[list[int], np.ndarray]":
    """Return (sorted hole areas desc, enclosed-hole mask uint8)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    white = (g >= WHITE_THR).astype(np.uint8)
    # exterior background = white reachable from the border
    ext = np.zeros((H, W), np.uint8)
    ff = white.copy()
    mask = np.zeros((H + 2, W + 2), np.uint8)
    seeds = []
    for x in range(0, W, 1):
        for y in (0, H - 1):
            if white[y, x]:
                seeds.append((x, y))
    for y in range(0, H, 1):
        for x in (0, W - 1):
            if white[y, x]:
                seeds.append((x, y))
    for (x, y) in seeds:
        if ff[y, x] == 1:
            cv2.floodFill(ff, mask, (x, y), 2)
    ext = (ff == 2).astype(np.uint8)
    enclosed = ((white == 1) & (ext == 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(enclosed, 8)
    areas = []
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= MIN_HOLE_AREA:
            areas.append(a)
            keep[lab == i] = 1
    areas.sort(reverse=True)
    return areas, keep


def edit_stats(edit_dir: Path) -> dict:
    max_hole = 0
    tot_hole = 0
    n_views_hit = 0
    per_view = {}
    for v in VIEWS:
        p = edit_dir / f"after_view_{v}.png"
        im = cv2.imread(str(p))
        if im is None:
            per_view[v] = None
            continue
        areas, _ = detect_holes(im)
        mh = areas[0] if areas else 0
        per_view[v] = mh
        max_hole = max(max_hole, mh)
        tot_hole += sum(areas)
        if mh >= MIN_HOLE_AREA:
            n_views_hit += 1
    return {"max_hole": max_hole, "tot_hole": tot_hole,
            "n_views_hit": n_views_hit, "per_view": per_view}


def has_hole(stats: dict, *, min_max_area: int = 120, min_views: int = 1) -> bool:
    """Decision: a real see-through hole present."""
    return stats["max_hole"] >= min_max_area and stats["n_views_hit"] >= min_views


def _save_overlay(edit_dir: Path, out: Path) -> None:
    cells = []
    for v in VIEWS:
        im = cv2.imread(str(edit_dir / f"after_view_{v}.png"))
        if im is None:
            continue
        _, keep = detect_holes(im)
        ov = im.copy()
        ov[keep == 1] = (0, 0, 255)
        vis = cv2.addWeighted(im, 0.5, ov, 0.5, 0)
        cv2.putText(vis, v, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)
        cells.append(vis)
    if cells:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.hconcat(cells))
        print(f"  overlay → {out}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--scan":
        tree = Path(args[1])
        if not tree.is_absolute():
            tree = ROOT / tree
        rows = []
        for ed in sorted(tree.glob("objects/08/*/edits_3d/*/")):
            if not (ed / "after_view_front.png").is_file():
                continue
            s = edit_stats(ed)
            rows.append((ed.name, s, has_hole(s)))
        rows.sort(key=lambda r: -r[1]["max_hole"])
        print(f"{'edit':30}{'maxHole':>8}{'totHole':>8}{'views':>6}  HOLE")
        for name, s, h in rows:
            print(f"{name[:30]:30}{s['max_hole']:8d}{s['tot_hole']:8d}"
                  f"{s['n_views_hit']:6d}  {'YES' if h else ''}")
        return
    ed = Path(args[0])
    if not ed.is_absolute():
        ed = ROOT / ed
    s = edit_stats(ed)
    print(f"{ed.name}")
    print(f"  max_hole={s['max_hole']} tot={s['tot_hole']} views_hit={s['n_views_hit']}")
    print(f"  per_view={s['per_view']}")
    print(f"  HAS_HOLE={has_hole(s)}")
    if "--save-overlay" in args:
        _save_overlay(ed, ed / "_hole_overlay.png")


if __name__ == "__main__":
    main()
