# Pxform_v2 export — deletion / addition runbook

How the **deletion** and **addition** training examples in `data/Pxform_v2/export/`
are built, why each step exists, and how to (re)run it.

> mod/scale are a pure CPU copy of prod latents and are covered briefly at the
> end. The interesting (GPU, re-encoded) path is **del/add** — that is the focus.

---

## TL;DR

- **Code**: `scripts/export_pxform_v2_dataset.py` — `export_del_add()` (lines 253–335).
- **del** = original mesh with the selected part **removed**, re-encoded in the
  TRELLIS.2 latent space at **512 edit-res (32³ slat / 16³ ss)**.
- **add** = the **inverse** of del — same geometry pair with `before`/`after`
  swapped and the prompt inverted (`invert_delete_prompt`).
- Everything is **v2-native**. v1 (TRELLIS.1 / DINOv2) latents are **not** reused —
  different latent space *and* different prompts.

---

## Why del/add need their own path

mod/scale already exist as finished prod edits (`edits_3d/<eid>/latents/*`), so
they are copied verbatim. del/add do **not** survive prod gating in the v2 tree
(deletion is mesh-only / deferred; addition is backfilled), so there is no
ready-made latent to copy. We rebuild them from two v2-native ingredients:

1. **del specs** — prompt + `selected_part_ids` from `phase1/parsed.json`.
2. **del geometry** — the s5b merge that drops the selected part from the mesh.

Then we re-encode the resulting mesh with the **same** TRELLIS.2 encoders the
pipeline uses, at the **same** 512 edit resolution as prod.

---

## Pipeline (per object)

```
parsed.json (deletion specs) ─┐
                              ├─► _merge_surviving_parts_from_npz ─► after_new.glb
mesh npz (selected_part_ids) ─┘            (s5b geometry, raw Y-up, keeps surviving part GLBs)
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
                          mask_from_ss(ss)            ── mask sidecar
                              keep masks over coords (shape & tex share coords)
                                                      │
                                                      ▼
                          deletion edit:  before = original e512,  after = deleted
                          addition edit:  before = deleted,        after = original e512
                              (ss coords0/coords_new swapped; prompt inverted)
```

### Step 1 — geometry: `_merge_surviving_parts_from_npz`
`partcraft/pipeline_v3/mesh_deletion.py:187`. Given the mesh npz and the
`selected_part_ids`, it concatenates the **surviving** part GLBs into
`after_new.glb` (raw Y-up, no `vd_scale`). This is exactly the s5b deletion
geometry the pipeline produces — we reuse the pipeline function, not a copy.

### Step 2 — frame alignment (critical)
`encode_after_512()` (line 47). The deleted GLB can have a different bounding box
than the original, so encoding it on its own grid would shift coords and break
the before↔after correspondence. We normalize `after_new.glb` with the
**original mesh's** transform `M`:

```python
_, M = OVR._normalized_groups(OVR.load_full_scene(orig_mesh_npz), canonical)   # M from ORIGINAL
groups, _ = OVR._normalized_groups(scene, canonical=canonical, M=M)            # reuse M for after
```

This keeps the deleted result on the original's 32³ grid (validated
`frame_overlap = 1.000`). **Do not** let `after_new.glb` recompute its own M.

### Step 3 — encode @512
Same `encoders["shape"]` / `encoders["tex"]` the pipeline loads
(`_ensure_encoders`), `grid_size=512` → 32³ sparse slat. shape and tex are
encoded over the same voxelization, so **`shape_coords == tex_coords`** — one
coordinate mask covers both. No `ss_enc` is computed here: the dataset's
after-`ss` is structural metadata, not a re-encoded latent.

### Step 4 — structure (`ss.npz`) + mask
`part_struct_grids()` (line 95) reproduces the prod `ss.npz` region exactly:

| call | result | matches prod |
|------|--------|--------------|
| `part_edit_grid_64(mesh_npz, pids, pad=4)` | 64³ part region | — |
| `edit_grid_64_to_keep16(g64, thresh=0.1)` | `keep16` 16³ | agree = 1.0 |
| `downsample_edit_grid(g64, 2)` + `nonzero` | `edit_grid` 32³ coords | IoU = 1.0 |

`pad=4` / `thresh=0.1` mirror the config (`trellis2_s1_pad: 4`). `mask_from_ss()`
(line 108) then derives the keep masks: a coord is "keep" iff it is **not** in
`edit_grid`.

### Step 5 — del / add are an inverse pair
Both edits share the same encoded geometry; only direction and labels differ:

|              | `before` (coords0) | `after` (coords_new) | prompt |
|--------------|--------------------|----------------------|--------|
| **deletion** | original e512      | deleted (encoded)    | `spec.prompt` ("Remove the …") |
| **addition** | deleted (encoded)  | original e512        | `invert_delete_prompt(spec.prompt)` ("Add a …") |

`invert_delete_prompt` is at `partcraft/pipeline_v3/addition_utils.py:15`. The
mask `edit_grid`/`keep16`/`parts` is identical for both; only `coords0` and
`coords_new` swap. (Verified: del's `mask_keep_slat` == add's
`mask_keep_slat_before`, and vice-versa.)

### Step 6 — best view (argmax, not VLM)
del/add never went through gate_a, so there is no VLM `best_view`.
`compute_best_view()` (line 145) picks the view with the most selected-part
pixels via `count_part_pixels_in_overview` over `phase1/overview.png`.
`view.meta.json` records `"view_source": "overview_part_pixel_argmax"`.

### Step 7 — render
One `pipeline.decode_latent(shape_slat, tex_slat, 512)` per latent →
`_ov.render_sample` gives all 5 named views at once; we keep the best-view PNG
for `before.png` / `after.png`. Use `--no-render` to skip (latents/masks still
written).

---

## Output layout

```
data/Pxform_v2/export/
  deletion/<shard>/<obj>/<edit_id>/
      shape_slat.npz   feats[N,32], coords[N,3] @32³   (after geometry)
      tex_slat.npz     feats[N,32], coords[N,3]        (shares coords w/ shape)
      ss.npz           coords0, coords_new, edit_grid, keep16, parts, edit_type,
                       s1_pad, s1_thresh               (structural metadata)
      mask.npz         mask_keep_ss[16,16,16], mask_keep_slat[N_after],
                       mask_keep_slat_before[N_before], selected_part_ids
      before.png  after.png  meta.json  view.meta.json
  addition/<shard>/<obj>/<edit_id>/    (same files, inverse direction)
  _assets/<shard>/<obj>/   shape_slat.npz(e512) + tex_slat.npz(e512) + ss.npz   (shared "before")
  manifests/all.<tag>.jsonl            (one line per edit; merge into all.jsonl)
```

`meta.json` is field-for-field the manifest record, so `all.jsonl` can be rebuilt
from the on-disk `meta.json` files if a per-tag manifest line is lost.

---

## How to run

Two-GPU parallel (object partition via `--gpu-shard k/n`), 125+125 each:

```bash
cd /mnt/zsn/zsn_workspace/PartCraft3D_v2

# GPU 0 — half the objects
CUDA_VISIBLE_DEVICES=0 python scripts/export_pxform_v2_dataset.py \
    --types del,add --shard 00 --n-del 125 --n-add 125 \
    --gpu-shard 0/2 --tag delA \
    > data/Pxform_v2/_export_delA.log 2>&1 &

# GPU 1 — the other half
CUDA_VISIBLE_DEVICES=1 python scripts/export_pxform_v2_dataset.py \
    --types del,add --shard 00 --n-del 125 --n-add 125 \
    --gpu-shard 1/2 --tag delB \
    > data/Pxform_v2/_export_delB.log 2>&1 &
```

mod/scale (pure CPU copy, no model load) in parallel:

```bash
python scripts/export_pxform_v2_dataset.py \
    --types mod,scale --shard 00 --n-mod 250 --n-scale 250 --tag modscale \
    > data/Pxform_v2/_export_modscale.log 2>&1 &
```

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

### Key flags
| flag | meaning |
|------|---------|
| `--types` | subset of `del,add,mod,scale` (GPU loaded only if del/add present) |
| `--shard` | prod shard under `--prod-root` (default `00`) |
| `--gpu-shard k/n` | take objects where `i % n == k` (object-level partition) |
| `--n-del/add/mod/scale` | per-type cap (counts down across objects) |
| `--no-render` | skip decode/render; still writes latents + masks |
| `--tag` | manifest suffix `all.<tag>.jsonl` |
| `--canonical` | canonical-frame normalization (default 1; must match encode) |

---

## Invariants to check after a run

- Each edit dir has **8 files** (3 latent + mask + 2 png + 2 json).
- Latent coords `max == 31` (32³ — i.e. **512** edit-res, not 1024/64³).
- `shape_coords == tex_coords` per edit (one shared coord mask).
- del/add inverse: del.`coords_new` == add.`coords0` for the same obj/seq.
- best_view distribution is **not** all-`front` (argmax is actually firing).

---

## mod / scale (for completeness)

`export_mod_scale()` (line 338), pure CPU:

- Source = prod `edits_3d/<eid>/latents/{shape_slat,tex_slat,ss}.npz`, copied
  **verbatim** — already 32³, and `ss.npz` already carries
  `edit_grid`/`keep16`/`parts`.
- Only `gate_e == pass` edits are taken.
- `mask.npz` derived from the copied `ss.npz` via the same `mask_from_ss()`.
- `before.png`/`after.png` copied from prod `gate_views/before_view_<v>.png` and
  `edits_3d/<eid>/after_view_<v>.png`, where `<v>` is the **VLM** best_view from
  gate_a (`view_source": "pipeline_gate_a"`).
