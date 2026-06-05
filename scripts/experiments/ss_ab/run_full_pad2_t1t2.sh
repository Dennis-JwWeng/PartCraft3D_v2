#!/usr/bin/env bash
# Full-13 pad2 TRELLIS.1-vs-TRELLIS.2 SS-flow A/B driver.
#   prep(pad2) -> run_t1(T1 SS flow) -> repack -> T2 arm -> T1 arm -> HTML
# Same recipe both arms (perstep + same-frame 64³ restore + 512 + white);
# ONLY variable = S1 SS flow model.  GPUs 0,1.  Run in background.
set -uo pipefail
cd /mnt/zsn/zsn_workspace/PartCraft3D_v2

IO=data/Pxform_v2/_scratch/ss_ab_t1t2_pad2
IDS=data/Pxform_v2/_exp_masked_perstep_r512_pad0/seeded_ids.txt
PY2=/mnt/zsn/miniconda3/envs/trellis2/bin/python
PYV=/mnt/zsn/miniconda3/envs/vinedresser3d/bin/python
PYVIZ=/mnt/zsn/miniconda3/envs/trellis2_viz/bin/python

step(){ echo; echo "=============== $* @ $(date +%T) ==============="; }
die(){ echo "!!! FAILED at: $* (exit $?) @ $(date +%T)"; exit 1; }

step "1/6 prep --pad 2 (shared inputs for both arms)"
CUDA_VISIBLE_DEVICES=0 $PY2 scripts/experiments/ss_ab/prep.py \
  --src data/Pxform_v2/_exp_masked_posthoc_r1024 --shard 08 --pad 2 --out "$IO" \
  || die "prep"
# keep only the 13 seeded objects (drop 2 non-seeded source objs from run_t1)
for x in be1691a3b8484eab823c69e135299e2f be2e3dd6ee9d43d7809ba5e0a3b56559; do
  rm -rf "$IO/inputs/$x"
done
echo "inputs objects: $(ls "$IO/inputs" | wc -l)"

step "2/6 run_t1 (TRELLIS.1 ss_flow_img_dit_L_16l8, vinedresser3d, GPU0)"
CUDA_VISIBLE_DEVICES=0 $PYV scripts/experiments/ss_ab/run_t1.py --io "$IO" || die "run_t1"

step "3/6 repack out/t1 -> ss1/<obj>/<eid>/ss1_coords.npz (key coords)"
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

step "4/6 T2 arm preview (TRELLIS.2 SS flow), GPUs 0,1"
SHARD=08 OBJ_IDS_FILE=$IDS FORCE=1 STAGES=trellis2_preview \
  MACHINE_ENV=configs/machine/local_trellis2.env PIPELINE_GPUS="0,1" \
  bash run_pipeline_v3_shard_trellis2.sh shard08_t2full \
    configs/pipeline_v3_trellis2_masked_perstep_r512_pad2_restore.yaml || die "T2 arm"

step "5/6 T1 arm preview (TRELLIS.1 SS flow via bridge), GPUs 0,1"
SHARD=08 OBJ_IDS_FILE=$IDS FORCE=1 STAGES=trellis2_preview \
  MACHINE_ENV=configs/machine/local_trellis2.env PIPELINE_GPUS="0,1" \
  bash run_pipeline_v3_shard_trellis2.sh shard08_t1full \
    configs/pipeline_v3_trellis2_t1ss_perstep_r512_pad2_restore.yaml || die "T1 arm"

step "6/6 regen comparison HTML"
$PYVIZ scripts/viz/ab_t1_vs_t2_ssflow_html.py || die "html"

step "ALL DONE"
