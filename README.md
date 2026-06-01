# PartCraft3D

**Agentic pipeline that turns part-segmented 3D assets into `(before, after, edit instruction)` training pairs**, all the way from a raw mesh to a verified edit triplet — VLMs drive the labeling, planning, and quality gates; FLUX edits 2D conditioning; TRELLIS lifts edits back into 3D latents.

Active orchestrator: **`partcraft.pipeline_v3`**.

---

## Quickstart

```bash
# 1. Install (pip installs the partcraft package + CLI deps)
pip install -e .

# 2. Copy the template and fill machine/data paths
cp configs/templates/pipeline_v3_bench.template.yaml configs/my_run.yaml
cp configs/machine/template.env configs/machine/$(hostname).env
$EDITOR configs/my_run.yaml configs/machine/$(hostname).env

# 3. Validate inputs, then run a shard end-to-end
bash scripts/tools/validate_bench_inputs.sh configs/my_run.yaml
bash scripts/tools/run_pipeline_v3_shard.sh configs/my_run.yaml
```

Direct Python CLI (useful for single-stage debugging or CI):

```bash
python -m partcraft.pipeline_v3.run --config configs/my_run.yaml --shard 08 --all
# Single stage:  --stage gate_text_align
# Scoped run:    --obj-ids-file ids.txt
# Force redo:    --force
```

---

## Pipeline

```
gen_edits         VLM proposes per-object edit specs from a part menu
   ↓
gate_text_align   VLM-judged text-image alignment, also writes best_view per edit ←single source of truth
   ↓
   ┌─────────────────────────────┬──────────────────────────────┐
   ↓ (CPU)                       ↓ (GPU)                        ↓
del_mesh                      flux_2d                         preview_del
(deletion mask + glb)           ↓                             (Blender 5-view)
                             trellis_3d   (3D latent edit)
                                ↓
                             preview_flux (5-view)
   ↓                            ↓
   └─────────────────────────────┴──────────────────────────────┘
                                ↓
                            gate_quality   (final VLM visual QC, Gate E)
```

Per-edit state is tracked in `objects/<shard>/<obj>/edit_status.json`. Each stage is resumable: an edit only re-runs when its prerequisite gate passed and its own stage is missing or errored. `gate_text_align.best_view` is the **single authority** for which view feeds FLUX — missing → hard error, no silent fallback.

Edit types: `deletion`, `modification`, `scale`, `material`, `color`, `global`, `addition` (addition is back-filled from paired deletion).

---

## Layout

```
partcraft/
  pipeline_v3/          orchestrator, scheduler, per-stage runners (s4/s5/s6p/...)
  io/                   dataset loaders (PartVerse / H3D_v1 / mesh NPZ)
  trellis/              TRELLIS refiner integration
  render/               Blender overview + previews
  cleaning/h3d_v1/      promote pipeline output → H3D_v1 dataset
configs/
  templates/            pipeline + machine env template (the only config you keep)
  h3d_v1_full_bench_scale_consistent_del.yaml
scripts/
  tools/run_pipeline_v3_shard.sh    main shell entry
  tools/validate_bench_inputs.sh    pre-flight check
  tools/run_pipeline_v3_bench.sh    bench helper (tmux)
  datasets/{partverse,partobjaverse,h3d_v1}/   one-time dataset packers
docs/
  PIPELINE.md                       full pipeline architecture
  PIPELINE_V3_VLM_PROMPTS.md        VLM prompt specs (3 calls)
  ARCH.md                           module conventions
  dataset-path-contract.md          path naming rules
  runbooks/                         H3D_v1 promote / publish / re-render
```

---

## Output → dataset

A pipeline run lands per-shard under `<output_dir>/objects/<shard>/<obj_id>/`. To promote one shard into the **H3D_v1** dataset:

```bash
# Order matters: deletion (GPU) → flux → addition → index
python -m scripts.cleaning.h3d_v1.pull_deletion --pipeline-cfg configs/my_run.yaml --shard 08 --dataset-root data/H3D_v1 --device cuda:0
python -m scripts.cleaning.h3d_v1.pull_flux     --pipeline-cfg configs/my_run.yaml --shard 08 --dataset-root data/H3D_v1
python -m scripts.cleaning.h3d_v1.pull_addition --pipeline-cfg configs/my_run.yaml --shard 08 --dataset-root data/H3D_v1
python -m scripts.cleaning.h3d_v1.build_h3d_v1_index --dataset-root data/H3D_v1 --validate
```

Full multi-machine runbook: **[`docs/runbooks/h3d-v1-promote.md`](docs/runbooks/h3d-v1-promote.md)**.

---

## More

- **[docs/PIPELINE.md](docs/PIPELINE.md)** — full architecture & cost model
- **[docs/PIPELINE_V3_VLM_PROMPTS.md](docs/PIPELINE_V3_VLM_PROMPTS.md)** — VLM prompt specs (3 calls)
- **[docs/ARCH.md](docs/ARCH.md)** — module conventions & stage scheduler
- **[docs/dataset-path-contract.md](docs/dataset-path-contract.md)** — path naming / `images_root` ↔ `image_npz_dir`
