#!/usr/bin/env bash
# Full-13 pad=N TRELLIS.1-vs-TRELLIS.2 SS-flow A/B driver (generalized run_full_pad2).
#   prep(padN) -> run_t1(T1 SS flow) -> repack -> T2 arm -> T1 arm
# Same recipe both arms (perstep + same-frame 64³ restore + 512 + white);
# ONLY variables = S1 SS flow model (T1 vs T2) and the edit-grid dilation pad.
# GPUs 0,1.  Usage:  bash run_full_padN_t1t2.sh <PAD>     (PAD in 3 4 5 7 ...)
set -uo pipefail
cd /mnt/zsn/zsn_workspace/PartCraft3D_v2

PAD="${1:?usage: run_full_padN_t1t2.sh <PAD>}"
GPU="${GPU:-0}"          # single GPU id (e.g. 0 or 1) for prep/run_t1/pipeline of THIS pad
IO=data/Pxform_v2/_scratch/ss_ab_t1t2_pad${PAD}
IDS=data/Pxform_v2/_exp_masked_perstep_r512_pad0/seeded_ids.txt
PY2=/mnt/zsn/miniconda3/envs/trellis2/bin/python
PYV=/mnt/zsn/miniconda3/envs/vinedresser3d/bin/python

step(){ echo; echo "=============== $* @ $(date +%T) ==============="; }
die(){ echo "!!! FAILED at: $* (exit $?) @ $(date +%T)"; exit 1; }

step "1/5 prep --pad ${PAD} (shared inputs for both arms) [GPU ${GPU}]"
CUDA_VISIBLE_DEVICES=$GPU $PY2 scripts/experiments/ss_ab/prep.py \
  --src data/Pxform_v2/_exp_masked_posthoc_r1024 --shard 08 --pad "${PAD}" --out "$IO" \
  || die "prep"
# keep only the 13 seeded objects (drop 2 non-seeded source objs)
for x in be1691a3b8484eab823c69e135299e2f be2e3dd6ee9d43d7809ba5e0a3b56559; do
  rm -rf "$IO/inputs/$x"
done
NIN=$(find "$IO/inputs" -name '*.npz' | wc -l)
NSS1=$(find "$IO/ss1" -name 'ss1_coords.npz' 2>/dev/null | wc -l)
echo "inputs objects: $(ls "$IO/inputs" | wc -l)  edits: $NIN  existing ss1: $NSS1"

if [ "$NSS1" -ge "$NIN" ] && [ "$NIN" -gt 0 ]; then
  step "2-3/5 run_t1 + repack — SKIPPED (ss1 already complete: $NSS1/$NIN)"
else
step "2/5 run_t1 (TRELLIS.1 ss_flow_img_dit_L_16l8, vinedresser3d, GPU ${GPU})"
CUDA_VISIBLE_DEVICES=$GPU $PYV scripts/experiments/ss_ab/run_t1.py --io "$IO" || die "run_t1"

step "3/5 repack out/t1 -> ss1/<obj>/<eid>/ss1_coords.npz (key coords)"
$PY2 - "$IO" <<'PYEOF' || die "repack"
import numpy as np, glob, os, sys
IO=sys.argv[1]; n=0
for p in sorted(glob.glob(f"{IO}/out/t1/*/*.npz")):
    obj=os.path.basename(os.path.dirname(p)); eid=os.path.splitext(os.path.basename(p))[0]
    d=np.load(p, allow_pickle=True)
    od=f"{IO}/ss1/{obj}/{eid}"; os.makedirs(od, exist_ok=True)
    np.savez(f"{od}/ss1_coords.npz", coords=d["coords_new"].astype(np.int32)); n+=1
print(f"repacked {n} edits -> {IO}/ss1")
PYEOF
fi

SEED=scripts/experiments/ss_ab/../seed_masked_e512_variant.sh

if [ "${SKIP_T2:-0}" = "1" ]; then
  step "4/5 T2 arm — SKIPPED (SKIP_T2=1; TRELLIS.2 SS arm deferred)"
else
  step "3.5/5 seed T2 output tree from pad2 T2 template (pre-3D symlinks + edit_status)"
  bash "$SEED" data/Pxform_v2/_exp_masked_perstep_r512_pad2_restore \
               data/Pxform_v2/_exp_masked_perstep_r512_pad${PAD}_restore || die "seed T2"

  step "4/5 T2 arm preview (TRELLIS.2 SS flow), GPU ${GPU}"
  CUDA_VISIBLE_DEVICES=$GPU SHARD=08 OBJ_IDS_FILE=$IDS FORCE=1 STAGES=trellis2_preview \
    MACHINE_ENV=configs/machine/local_trellis2.env PIPELINE_GPUS="0" \
    bash run_pipeline_v3_shard_trellis2.sh shard08_t2pad${PAD} \
      configs/pipeline_v3_trellis2_masked_perstep_r512_pad${PAD}_restore.yaml || die "T2 arm"
fi

step "4.5/5 seed T1 output tree from pad2 T1 template (pre-3D symlinks + edit_status)"
bash "$SEED" data/Pxform_v2/_exp_t1ss_perstep_r512_pad2_restore \
             data/Pxform_v2/_exp_t1ss_perstep_r512_pad${PAD}_restore || die "seed T1"

step "5/5 T1 arm preview (TRELLIS.1 SS flow via bridge), GPU ${GPU}"
CUDA_VISIBLE_DEVICES=$GPU SHARD=08 OBJ_IDS_FILE=$IDS FORCE=1 STAGES=trellis2_preview \
  MACHINE_ENV=configs/machine/local_trellis2.env PIPELINE_GPUS="0" \
  bash run_pipeline_v3_shard_trellis2.sh shard08_t1pad${PAD} \
    configs/pipeline_v3_trellis2_t1ss_perstep_r512_pad${PAD}_restore.yaml || die "T1 arm"

step "ALL DONE pad=${PAD}"
