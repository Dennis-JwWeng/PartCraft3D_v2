#!/usr/bin/env bash
# pad0 re-paste driver — one worker per GPU over one shard (gate-E pass only).
#
#   bash scripts/run_repaste_pad0_shard.sh 00            # shard 00 on 8 GPUs
#   GPUS="0,1,2,3" bash scripts/run_repaste_pad0_shard.sh 03
#   PAD=1 bash scripts/run_repaste_pad0_shard.sh 00      # alternative pad
#
# Resume-safe: edits with the after_view PNG already rendered are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

SHARD="${1:?usage: run_repaste_pad0_shard.sh <shard>}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PAD="${PAD:-0}"
ROOT="${ROOT:-data/Pxform_v2/prod_posthoc_no2dqc}"
OUT_ROOT="${OUT_ROOT:-}"        # external output root (default: in-place repaste_pad$PAD)
EDITS_FILE="${EDITS_FILE:-}"    # optional '<obj>/<edit_id>' filter file

EXTRA=()
[[ -n "$OUT_ROOT" ]] && EXTRA+=(--out-root "$OUT_ROOT")
[[ -n "$EDITS_FILE" ]] && EXTRA+=(--edits-file "$EDITS_FILE")

source /mnt/zsn/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

IFS=',' read -ra G <<< "$GPUS"
N=${#G[@]}
LOGDIR="logs/repaste_pad${PAD}"
mkdir -p "$LOGDIR"

pids=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES="${G[$i]}" OPENCV_IO_ENABLE_OPENEXR=1 \
    python scripts/repaste_pad0_batch.py \
      --root "$ROOT" --shard "$SHARD" --pad "$PAD" --slice "$i/$N" \
      ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$LOGDIR/shard${SHARD}_w${i}.log" 2>&1 &
  pids+=($!)
done
echo "shard $SHARD: $N workers (gpus $GPUS), logs in $LOGDIR/shard${SHARD}_w*.log"

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "shard $SHARD done (rc=$rc)"
exit $rc
