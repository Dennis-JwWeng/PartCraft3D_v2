#!/usr/bin/env bash
# S1-only backfill of the after SS latent (ss_latent.npz) + mask.npz for every
# already-edited mod/scale edit, across all prod shards.  Idempotent (skips edits
# that already have ss_latent.npz).  Products land IN PLACE in the prod tree:
#   data/Pxform_v2/prod_posthoc_no2dqc/objects/<shard>/<obj>/edits_3d/<eid>/latents/
#       ss_latent.npz   mask.npz
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc_ss_latent_backfill.yaml
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SHARDS="${SHARDS:-00 01 02 03 04 05 06 07 08 09}"
PY=/mnt/zsn/miniconda3/envs/trellis2/bin/python
for S in $SHARDS; do
  echo "========================= shard $S  $(date '+%F %T') ========================="
  $PY -m partcraft.pipeline_v3.run_trellis2 \
      --config "$CFG" --shard "$S" --steps trellis2_3d --all --gpus "$GPUS"
  echo "----- shard $S done  $(date '+%F %T') -----"
done
echo "========================= ALL SHARDS DONE  $(date '+%F %T') ========================="
