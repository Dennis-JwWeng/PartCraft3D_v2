"""A/B compare the two 3D-edit recipes on the SAME object+edit_id.

  A = _exp_flowedit_free_r1024     (FlowEdit S1 + free S2)
  B = _exp_masked_posthoc_r1024  (masked contact-soft S1 + ss_align_t1 + posthoc S2)

Both render the named views from decoded latents (gate_views/before_view_* for the
shared "before"; edits_3d/<id>/after_view_* for each recipe's "after").  For every
edit_id present in BOTH trees we stack a 3-row collage:

    row0  BEFORE   (shared, decoded original latents)
    row1  A after  (FlowEdit + free)
    row2  B after  (masked + posthoc)

across the 5 named views, and write one PNG per edit + a per-object stacked sheet.

    python scripts/viz/compare_ab_edits.py            # all common edits
    python scripts/viz/compare_ab_edits.py <obj8> ...  # only these objects (8-char ok)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path("/mnt/zsn/zsn_workspace/PartCraft3D_v2")
A = ROOT / "data/Pxform_v2/_exp_flowedit_free_r1024/objects/08"
B = ROOT / "data/Pxform_v2/_exp_masked_posthoc_r1024/objects/08"
OUT = ROOT / "data/Pxform_v2/_scratch/ab_compare"
VIEWS = ["front", "right", "back", "left", "down"]
CELL = 256


def _load(p: Path):
    im = cv2.imread(str(p))
    if im is None:
        return np.full((CELL, CELL, 3), 240, np.uint8)
    return cv2.resize(im, (CELL, CELL))


def _row(get, label, color=(40, 40, 40)):
    cells = []
    for v in VIEWS:
        im = _load(get(v)).copy()
        cv2.putText(im, v, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)
        cells.append(im)
    row = np.concatenate(cells, axis=1)
    band = np.full((26, row.shape[1], 3), 255, np.uint8)
    cv2.putText(band, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return np.concatenate([band, row], axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    want = {a[:8] for a in sys.argv[1:]} or None
    n_obj = n_edit = 0
    for objdir in sorted(B.glob("*/")):
        obj = objdir.name
        if want and obj[:8] not in want:
            continue
        gv = A / obj / "gate_views"           # shared before (same encode)
        b_edits = sorted((objdir / "edits_3d").glob("*/"))
        obj_rows = []
        for bed in b_edits:
            eid = bed.name
            aed = A / obj / "edits_3d" / eid
            if not (aed / "after_view_front.png").is_file():
                continue                      # only edits present in BOTH
            if not (bed / "after_view_front.png").is_file():
                continue
            before = _row(lambda v: gv / f"before_view_{v}.png", "BEFORE (orig latents)")
            arow = _row(lambda v: aed / f"after_view_{v}.png", f"A  FlowEdit+free   {eid}", (150, 80, 0))
            brow = _row(lambda v: bed / f"after_view_{v}.png", f"B  masked+t1+posthoc  {eid}", (0, 110, 0))
            coll = np.concatenate([before, arow, brow], axis=0)
            cv2.imwrite(str(OUT / f"{eid}.png"), coll)
            obj_rows.append(coll)
            n_edit += 1
        if obj_rows:
            sep = np.full((6, obj_rows[0].shape[1], 3), 0, np.uint8)
            stacked = obj_rows[0]
            for r in obj_rows[1:]:
                stacked = np.concatenate([stacked, sep, r], axis=0)
            cv2.imwrite(str(OUT / f"_obj_{obj[:12]}.png"), stacked)
            n_obj += 1
    print(f"wrote {n_edit} edit comparisons across {n_obj} objects → {OUT}")


if __name__ == "__main__":
    main()
