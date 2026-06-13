#!/usr/bin/env bash
# Regenerate configs/shard01_xl/allow.txt, block.tsv, need_encode.txt from disk state.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG=configs/shard01_xl/pipeline.yaml
ROSTER=configs/shard01_xl/roster.txt
OUT=configs/shard01_xl

python scripts/ops/preflight_partversexl.py \
  --config "$CFG" --shard 01 --obj-ids-file "$ROSTER" \
  --write-allow "$OUT/allow.txt" --write-block "$OUT/block.tsv" --warn-only

python3 << PY
from pathlib import Path
allow = [l.strip() for l in Path("$OUT/allow.txt").read_text().splitlines() if l.strip()]
obj = Path("data/Pxform_v2/partversexl_posthoc_no2dqc/objects/01")
need = [i for i in allow if not (obj/i/"p1_encode"/"shape_slat_e512.npz").is_file()]
Path("$OUT/need_encode.txt").write_text("\n".join(need) + ("\n" if need else ""))
print(f"allow={len(allow)} need_encode={len(need)}")
PY
