#!/usr/bin/env bash
# Full v2-native del + inverse-add latent re-encode across all prod shards.
# Per object: reads phase1/parsed.json deletion specs → merges surviving parts
# into a TEMP after_new.glb (never persisted) → encodes shape+tex SLat @512 +
# ss_latent + mask + single best-view after_view → writes BOTH del_*/ and the
# inverse add_*/ into edits_3d/<eid>/latents/.  Idempotent (skips edits whose
# latents already exist; pass FORCE=--force to redo).  Multi-GPU via dispatch.
#   data/Pxform_v2/prod_posthoc_no2dqc/objects/<shard>/<obj>/edits_3d/{del,add}_*/latents/
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc.yaml
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SHARDS="${SHARDS:-00 01 02 03 04 05 06 07 08 09}"
FORCE="${FORCE:-}"
PY=/mnt/zsn/miniconda3/envs/trellis2/bin/python
for S in $SHARDS; do
  echo "========================= del_add shard $S  $(date '+%F %T') ========================="
  $PY -m partcraft.pipeline_v3.run_trellis2 \
      --config "$CFG" --shard "$S" --steps del_add --all --gpus "$GPUS" $FORCE
  echo "----- del_add shard $S done  $(date '+%F %T') -----"
done
echo "========================= DEL_ADD ALL SHARDS DONE  $(date '+%F %T') ========================="
