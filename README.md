# PartCraft3D

**Agentic pipeline that turns part-segmented 3D assets into `(before, after, edit instruction)` training pairs.** VLMs drive the labeling, planning, and quality gates; FLUX edits 2D conditioning; TRELLIS.2 lifts the edit back into 3D latents.

Orchestrator: **`partcraft.pipeline_v3`**.

## Quickstart

```bash
pip install -e .

# Configure: copy templates, fill in machine/data paths
cp configs/templates/pipeline_v3_bench.template.yaml configs/my_run.yaml
cp configs/machine/template.env configs/machine/$(hostname).env

# Validate inputs, then run
bash scripts/ops/validate_bench_inputs.sh configs/my_run.yaml
bash run_pipeline_v3_shard_trellis2.sh <tag> configs/my_run.yaml   # one shard, end-to-end
bash run_all_shards_trellis2.sh                                    # all shards 00..09, serially
```

Direct CLI (single-stage debugging / CI):

```bash
python -m partcraft.pipeline_v3.run --config configs/my_run.yaml --shard 08 --all
#   --stage gate_text_align    run one stage
#   --obj-ids-file ids.txt     scoped run
#   --force                    redo
```

## Pipeline

```
gen_edits → gate_text_align → ┬ del_mesh → preview_del            (CPU)
                              └ flux_2d → trellis_3d → preview_flux (GPU)
                                                ↓
                                          gate_quality  (final VLM QC, Gate E)
```

- Per-edit state lives in `objects/<shard>/<obj>/edit_status.json`. Stages are resumable: an edit re-runs only when its prerequisite gate passed and its own stage is missing or errored.
- `gate_text_align.best_view` is the **single authority** for which view feeds FLUX — missing → hard error, no silent fallback.
- Edit types: `deletion`, `modification`, `scale`, `material`, `color`, `global`, `addition` (back-filled from paired deletion). A per-run allow-list selects active types; the rest are recorded as `deferred`.

A run lands per-shard under `<output_dir>/objects/<shard>/<obj_id>/`; promote into the **H3D_v1** dataset via [`docs/runbooks/h3d-v1-promote.md`](docs/runbooks/h3d-v1-promote.md).

## Docs

- [docs/PIPELINE.md](docs/PIPELINE.md) — full architecture & cost model
- [docs/PIPELINE_V3_VLM_PROMPTS.md](docs/PIPELINE_V3_VLM_PROMPTS.md) — VLM prompt specs (3 calls)
- [docs/ARCH.md](docs/ARCH.md) — module conventions & stage scheduler
- [docs/dataset-path-contract.md](docs/dataset-path-contract.md) — path naming / `images_root` ↔ `image_npz_dir`
