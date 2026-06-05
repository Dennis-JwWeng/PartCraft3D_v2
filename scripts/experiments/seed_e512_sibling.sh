#!/usr/bin/env bash
# Seed a 512-edit-resolution sibling output tree from an existing 1024 smoke tree.
#
# Reuses the pre-3D products (phase1/parsed.json + overview, edits_2d, gate_views)
# by SYMLINK, copies the mutable per-object state (edit_status.json, p1_encode),
# and drops edits_3d so the 3D edit re-runs at 512.  The encode stage then only
# adds the grid-512 sidecar (shape/tex_slat_e512.npz); the 64³ main is present.
#
#   bash scripts/experiments/seed_e512_sibling.sh \
#        data/Pxform_v2/_exp_flowedit_free_r1024 data/Pxform_v2/_exp_flowedit_free_r512
set -euo pipefail
SRC="${1:?usage: $0 <src_tree> <dst_tree>}"
DST="${2:?usage: $0 <src_tree> <dst_tree>}"
SRC="$(cd "$SRC" && pwd)"

echo "[seed] $SRC -> $DST"
mkdir -p "$DST/_global" "$DST/objects"
cp -f "$SRC"/smoke15_ids.txt "$DST"/ 2>/dev/null || true
cp -f "$SRC"/_global/* "$DST"/_global/ 2>/dev/null || true

n=0
for objdir in "$SRC"/objects/*/*/; do
    rel="${objdir#"$SRC"/}"          # objects/08/<hash>/
    # skip incomplete objects (no 64³ encode → nothing to edit)
    [ -f "$objdir/p1_encode/shape_slat.npz" ] || { echo "[seed] skip (no p1_encode): $rel"; continue; }
    d="$DST/$rel"
    mkdir -p "$d"
    # read-only reuse via symlink (absolute targets so links survive moves)
    # NOTE: gate_views is NOT symlinked — s5 rewrites before_view_*.png there at
    # the edit resolution, which through a symlink would clobber the 1024 baseline.
    for sub in phase1 edits_2d debug; do
        [ -e "$objdir/$sub" ] && ln -sfn "$objdir/$sub" "$d/$sub"
    done
    mkdir -p "$d/gate_views"         # real dir: 512 before-views land here, not in src
    # mutable copies
    cp -f "$objdir/edit_status.json" "$d/" 2>/dev/null || true
    rm -rf "$d/p1_encode"; cp -r "$objdir/p1_encode" "$d/p1_encode"
    rm -rf "$d/edits_3d"             # force the 3D edit to regenerate at 512
    n=$((n+1))
done
echo "[seed] seeded $n objects under $DST"
