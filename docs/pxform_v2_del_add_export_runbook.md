# Pxform_v2 deletion / addition — pipeline stage + export runbook

How the **deletion** and **addition** training examples are produced and exported,
why each step exists, and how to (re)run them.

As of the `del_add` stage, del/add are a **first-class pipeline stage** that writes
TRELLIS.2 latents into the prod object tree exactly like mod/scale. The export is
then a **pure CPU copy** of prod for all four types. (Earlier this lived only in a
standalone GPU script; that path has been retired.)

---

## TL;DR

- **Stage code**: `partcraft/pipeline_v3/del_add_reencode.py` (GPU stage `del_add`).
- **Export code**: `scripts/export_pxform_v2_dataset.py` (`export_type_from_prod`, CPU-only).
- **del** = original mesh with the selected part **removed**, re-encoded in the
  TRELLIS.2 latent space at **512 edit-res (32³ slat / 16³ ss)**.
- **add** = the **inverse** of del — same geometry pair, `before`/`after` swapped,
  prompt inverted (`invert_delete_prompt`).
- Everything is **v2-native**. v1 (TRELLIS.1 / DINOv2) latents are **not** reused.
- The temp deleted GLB is **never persisted** (tempdir only).

---

## Why del/add need a re-encode stage

mod/scale are finished FLUX 3D edits (`edits_3d/<eid>/latents/*`). Deletion specs
exist in `phase1/parsed.json` (prompt + `selected_part_ids`) but on the
mod/scale-only shards their `gate_a` is recorded **`deferred`**, so
`specs.iter_deletion_specs` skips them (`specs.py:242`) and the v1
`run_deletion_batch` / `link_slat_assets_batch` (Blender + DINOv2) are inert.

The `del_add` stage therefore sources deletions **directly from `parsed.json`** and
re-encodes with the **same** TRELLIS.2 encoders the pipeline uses, at the **same**
512 edit resolution as prod. Addition is the algebraic inverse of deletion.

---

## Stage pipeline (per object)  — `del_add_reencode.py`

```
parsed.json (deletion specs) ─┐
                              ├─► _merge_surviving_parts_from_npz ─► TEMP after_new.glb
mesh npz (selected_part_ids) ─┘            (mesh_deletion.py:187; tempdir, NEVER persisted)
                                                      │
                                                      ▼
                          encode_after_512(encoders, orig_mesh_npz, after_new.glb)
                              normalize after_new with the ORIGINAL mesh's M
                              → shape_slat + tex_slat @ 32³ (grid_size=512)
                                                      │
                                                      ▼
                          part_struct_grids(mesh_npz, pids)   ── ss structure
                              part_edit_grid_64(pad=4) → keep16(16³) + edit_grid(32³)
                                                      │
                                                      ▼
                          deletion → edits_3d/del_*/latents/{shape_slat,tex_slat,ss}.npz
                          addition → edits_3d/add_*/latents/...  (ss coords0/coords_new swapped)
                          + after_view_*.png   + edit_status.json stages.del_add = done
```

### Step 1 — geometry: `_merge_surviving_parts_from_npz`
`mesh_deletion.py:187`. Concatenates the **surviving** part GLBs into a temp
`after_new.glb` (raw Y-up). The exact s5b deletion geometry — reused, not copied.
Written into a `tempfile.TemporaryDirectory()`, so **no GLB is ever persisted**.

### Step 2 — frame alignment (critical)
`encode_after_512()`. The deleted GLB can have a different bounding box than the
original, so it is normalized with the **original mesh's** transform `M`:

```python
_, M = OVR._normalized_groups(OVR.load_full_scene(orig_mesh_npz), canonical)   # M from ORIGINAL
groups, _ = OVR._normalized_groups(scene, canonical=canonical, M=M)            # reuse M for after
```

Keeps the deleted result on the original's 32³ grid (validated `frame_overlap = 1.000`).

### Step 3 — encode @512
Same `encoders["shape"]` / `encoders["tex"]` the pipeline loads (`_ensure_encoders`),
`grid_size=512` → 32³ sparse slat. shape & tex share one voxelization, so
**`shape_coords == tex_coords`** (one coordinate mask). Texture is the 6-channel PBR
attr (base_color+metallic+roughness+alpha).

### Step 4 — structure (`ss.npz`)
`part_struct_grids()` reproduces the prod `ss.npz` region exactly:

| call | result | matches prod |
|------|--------|--------------|
| `part_edit_grid_64(mesh_npz, pids, pad=4)` | 64³ part region | — |
| `edit_grid_64_to_keep16(g64, thresh=0.1)` | `keep16` 16³ | agree = 1.0 |
| `downsample_edit_grid(g64, 2)` + `nonzero` | `edit_grid` 32³ coords | IoU = 1.0 |

`ss.npz` is written via `trellis2_3d._save_edit_latents` — byte-for-byte the same
schema as mod/scale (`coords0, coords_new, edit_type, parts, edit_grid, keep16,
s1_pad, s1_thresh`).

### Step 5 — del / add are an inverse pair
|              | `coords0` (before) | `coords_new` (after) | prompt |
|--------------|--------------------|----------------------|--------|
| **deletion** | original e512      | deleted (encoded)    | `spec.prompt` ("Remove the …") |
| **addition** | deleted (encoded)  | original e512        | `invert_delete_prompt(...)` ("Add a …") |

Verified: del.`coords0` == add.`coords_new` and del.`coords_new` == add.`coords0`;
`keep16`/`edit_grid` identical.

### Step 6 — best view (argmax, not VLM)
Deletion never ran gate_a (deferred), so there is no VLM `best_view`.
`compute_best_view()` picks the view with the most selected-part pixels via
`count_part_pixels_in_overview` over `phase1/overview.png`. Recorded in
`edit_status.json` `stages.del_add.verdict.best_view` (`view_source:
overview_part_pixel_argmax`).

### Step 7 — after-views
- del: decode the deleted latents → mesh → 5 named PBR views → `del_*/after_view_*.png`.
- add: the addition result **is** the original, already rendered at encode as
  `gate_views/before_view_*.png` — copied to `add_*/after_view_*.png` (no decode).

---

## Running the stage

`del_add` is the last stage in
`configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc.yaml`
(`use_gpus: true`, `servers: none`). It is picked up automatically by the shard
and all-shards drivers — **no bash change needed**.

### Per shard, all GPUs (normal use)
```bash
SHARD=00 MACHINE_ENV=configs/machine/local_trellis2.env PIPELINE_GPUS="0,1,2,3,4,5,6,7" \
  bash run_pipeline_v3_shard_trellis2.sh deladd_shard00 \
       configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc.yaml
# (del_add runs after gate_quality; or limit with STAGES=del_add)
```

### Just the stage, on already-encoded objects
```bash
python -m partcraft.pipeline_v3.run_trellis2 \
    --config configs/pipeline_v3_trellis2_t1ss_native_r512_pad4_posthoc_no2dqc.yaml \
    --shard 00 --steps del_add --gpus 0,1,2,3,4,5,6,7 --all
# --obj-ids <id...> to target objects; --force to re-encode.
```

### How the GPU stage parallelizes
`del_add ∈ GPU_STEPS` + `use_gpus: true` → `dispatch_gpus` spawns one child per GPU
with `CUDA_VISIBLE_DEVICES=<gpu>` and `--gpu-shard k/N`; each child takes the
round-robin object slice `j % N == k` and loads encoders+pipeline once. Identical
to `trellis2_encode`. Models are **lazy-loaded** — a full-resume shard (everything
already encoded) loads nothing and finishes in seconds.

Dependencies: only `trellis2_encode` outputs (`p1_encode/{shape,tex}_slat_e512.npz`,
`phase1/{parsed.json, overview.png}`). Independent of flux/gate_2d/trellis2_3d/gate_quality.
Idempotent: skips edits whose del/add latents already exist (`--force` overrides).

---

## Exporting the dataset  — `export_pxform_v2_dataset.py` (CPU-only)

Once del/add latents are in prod, the export is a uniform CPU copy for **all four
types** (`export_type_from_prod`): copy `edits_3d/<eid>/latents/{shape,tex,ss}.npz`,
derive `mask.npz` from `ss.npz` (`mask_from_ss`), copy the best-view before/after
PNGs, write `meta.json` / `view.meta.json` / manifest line.

```bash
# 250 each, two parallel CPU object-shards
python scripts/export_pxform_v2_dataset.py --types del,add,mod,scale --shard 00 \
    --n-del 250 --n-add 250 --n-mod 250 --n-scale 250 \
    --obj-shard 0/2 --tag A &
python scripts/export_pxform_v2_dataset.py --types del,add,mod,scale --shard 00 \
    --n-del 250 --n-add 250 --n-mod 250 --n-scale 250 \
    --obj-shard 1/2 --tag B &
```

Selection + provenance:
| type | selected by | best_view | before image | quality |
|------|-------------|-----------|--------------|---------|
| mod/scale | `gate_e == pass` | gate_a VLM | `gate_views/before_view` | final_pass, gate_e score |
| deletion | `stages.del_add == done` | overview argmax | `gate_views/before_view` | alignment 1.0 |
| addition | `stages.del_add == done` | overview argmax | **paired del's `after_view`** (deleted) | alignment 1.0 |

> **Addition before-image**: the addition *starts* from the deleted object, so its
> `before.png` = the paired deletion's `after_view_<v>.png`, and its `after.png` =
> the original (`gate_views/before_view_<v>.png`). The paired id is in
> `stages.del_add.verdict.paired_deletion_edit_id`.

### Merge per-tag manifests → unified `all.jsonl`
Rebuild from on-disk `meta.json` (authoritative; tolerant of lost manifest lines):
```bash
cd data/Pxform_v2/export
python3 - <<'PY'
import json, glob
rows = [json.load(open(m)) for t in ('deletion','addition','modification','scale')
        for m in glob.glob(f'{t}/*/*/*/meta.json')]
rows.sort(key=lambda r: (r['edit_type'], r['shard'], r['edit_id']))
with open('manifests/all.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
from collections import Counter
print(len(rows), dict(Counter(r['edit_type'] for r in rows)))
PY
```

---

## Output layout (per edit, all four types uniform)

```
data/Pxform_v2/export/<edit_type>/<shard>/<obj>/<edit_id>/
    shape_slat.npz   feats[N,32], coords[N,3] @32³   (copied from prod, byte-identical)
    tex_slat.npz     feats[N,32], coords[N,3]        (shares coords w/ shape)
    ss.npz           coords0, coords_new, edit_grid, keep16, parts, edit_type, s1_pad, s1_thresh
    mask.npz         mask_keep_ss[16,16,16], mask_keep_slat[N_after],
                     mask_keep_slat_before[N_before], selected_part_ids
    before.png  after.png  meta.json  view.meta.json
_assets/<shard>/<obj>/   shape_slat.npz(e512) + tex_slat.npz(e512) + ss.npz(1,8,16,16,16)
manifests/all.<tag>.jsonl
```

mask is **not** stored in prod (it is a pure function of `ss.npz`); it is derived at
export so prod stays byte-uniform with mod/scale. (Set `del_add_write_mask: true`
in the config if a directly-trainable prod is ever needed.)

---

## Invariants to check after a run

- Each edit dir has **8 files** (3 latent + mask + 2 png + 2 json).
- Latent coords `max == 31` (32³ — i.e. **512** edit-res).
- `shape_coords == tex_coords` per edit (one shared coord mask).
- del/add inverse: del.`coords0` == add.`coords_new` for the same obj/seq; keep16 identical.
- exported latents are **byte-identical** to prod `edits_3d/<eid>/latents/*`.
- addition `before.png` == paired deletion `after_view`; addition `after.png` == original.
- No `after_new.glb` persisted anywhere (`find … -name after_new.glb` empty).
- best_view distribution is **not** all-`front` (argmax actually firing).
```
