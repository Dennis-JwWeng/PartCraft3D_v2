#!/usr/bin/env bash
# PartVerse XL pipeline — shard 01 only.
#
# Usage:
#   STAGES=trellis2_encode OBJ_IDS_FILE=configs/shard01_xl/need_encode.txt \\
#     bash run_shard01_xl.sh xl_s01_enc
#
#   STAGES=text_gen_gate_a,flux_2d,trellis2_preview,gate_quality \\
#     bash run_shard01_xl.sh xl_s01_main
#
#   FULL=1 bash run_shard01_xl.sh xl_s01_full   # all stages, no STAGES=
#
# Defaults (configs/shard01_xl/run.env): SHARD=01, CONFIG, GPU 0-7, OBJ_IDS_FILE=allow.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source configs/shard01_xl/run.env
TAG="${1:?usage: $0 <tag>  (set STAGES=... or FULL=1 for all stages)}"
if [ -z "${STAGES:-}" ] && [ "${FULL:-0}" != "1" ]; then
    echo "[ERROR] Set STAGES=... or FULL=1 for end-to-end run"
    exit 1
fi
exec bash run_pipeline_v3_shard_trellis2.sh "$TAG" "$CONFIG"
