#!/usr/bin/env bash
# Seed a NEW masked-512 experiment tree (e.g. pad2 / pad2_restore) by cloning the
# symlink layout of an EXISTING masked-512 sibling (the template).  Reuses the
# already-computed 512 encode (p1_encode → _exp_flowedit_free_r1024, holds shape/tex_slat_e512)
# and 2D products (edits_2d/phase1 → _exp_masked_posthoc_r1024) by recreating the template's
# symlinks against their resolved absolute targets; drops edits_3d + gives a fresh
# gate_views so the 3D edit re-runs at 512.  No GPU / no re-encode.
#
#   bash scripts/experiments/seed_masked_e512_variant.sh \
#        data/Pxform_v2/_exp_masked_posthoc_r512_pad0 data/Pxform_v2/_exp_masked_perstep_r512_pad2
set -euo pipefail
TPL="${1:?usage: $0 <template_tree> <dst_tree>}"
DST="${2:?usage: $0 <template_tree> <dst_tree>}"
TPL="$(cd "$TPL" && pwd)"

echo "[seed] template $TPL -> $DST"
mkdir -p "$DST/_global" "$DST/objects"
cp -f "$TPL"/*.txt "$DST"/ 2>/dev/null || true
cp -f "$TPL"/_global/* "$DST"/_global/ 2>/dev/null || true

n=0
for objdir in "$TPL"/objects/*/*/; do
    rel="${objdir#"$TPL"/}"            # objects/08/<hash>/
    d="$DST/$rel"
    mkdir -p "$d"
    for entry in "$objdir"*; do
        name="$(basename "$entry")"
        case "$name" in
            edits_3d) mkdir -p "$d/edits_3d" ;;             # regenerate at 512
            gate_views) mkdir -p "$d/gate_views" ;;          # 512 before-views land here
            edit_status.json) cp -f "$entry" "$d/" ;;        # mutable per-object state
            edit_status.json.lock) : ;;                      # skip stale lock
            *)
                if [ -L "$entry" ]; then                     # reuse: copy stored target verbatim
                    ln -sfn "$(readlink "$entry")" "$d/$name"
                elif [ -d "$entry" ]; then
                    ln -sfn "$entry" "$d/$name"              # real dir in template → link it
                fi
                ;;
        esac
    done
    n=$((n+1))
done
echo "[seed] seeded $n objects under $DST"
