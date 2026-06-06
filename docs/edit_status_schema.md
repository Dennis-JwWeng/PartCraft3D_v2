# `edit_status.json` schema — why it's messy, and how to read it correctly

**TL;DR:** the produced 3D assets and the gate verdicts are correct. The only
unreliable thing is the persisted **`final_pass`** field, which *over-counts*
passes. To count "valid edits" use the run-log `[gate_quality] … pass=` tally
or `qc_io.load_qc()` (which reconciles) — **never read the raw `final_pass`
off disk.**

---

## The four overlapping representations

A single `edit_status.json` records the same orchestration state four times:

```jsonc
{
  "obj_id": …, "shard": …, "schema_version": 2, "updated": …,

  // ① stage-level orchestration (status.py) — step resume
  "steps": { "s4b_t2_encode":…, "s1_phase1":…, "sq1_qc_A":…,
             "s4_flux_2d":…, "s5_trellis2":…, "sq3_qc_E":… },

  "edits": { "<edit_id>": {
      "edit_type": "global" | "part" | …,

      // ② per-edit, per-stage LIGHT markers (edit_status_io.update_edit_stage)
      "stages": { "gate_a": {status, ts}, "gate_e": {status, ts} },

      // ③ per-edit STRUCTURED gate payloads (qc_io.update_edit_gate / save_qc)
      "gates":  { "A": {rule, vlm}, "C": null, "E": null },

      // ④ DERIVED boolean (qc_io._sync_fail_fields)
      "final_pass": true
  } }
}
```

② and ③ encode the *same fact* (did gate X pass?) but are written by
**different functions** and are not kept consistent:

| writer | file | touches |
|---|---|---|
| `update_edit_stage(ctx, id, "gate_e", status=…)` | `edit_status_io.py:245` | only `stages.gate_e` (②) |
| `update_edit_gate(ctx, id, …, "E", vlm_result=…)` | `qc_io.py:174` | only `gates.E` (③) + recomputes `final_pass` |

## The concrete bug we hit

`gate_quality` writes the **stage marker** (`stages.gate_e.status="fail"`) but
**not** the structured `gates.E` payload, so `gates.E` stays `null`. And
`final_pass` is derived from `gates` only, with **"missing gate ⇒ pass"**:

```python
# qc_io.py:79,97
def _gp(gd): return True if gd is None else (gd not failing)
entry["final_pass"] = all(_gp(gates[g]) for g in ("A", "C", "E"))
```

→ an edit that **failed** Gate-E (per `stages`) computes `final_pass = true`
because `gates.E is None`. This is why on shard00 the raw `final_pass=True`
count was **10648** while the true Gate-E pass count was **2284**.

There *is* a reconciliation — `_build_qc_view_from_es` (`qc_io.py:127-131`)
backfills `gates[g]` from `stages.<gate>.status` when the payload is missing —
**but it runs only in the in-memory read view (`load_qc`), and is never
written back to disk.** So the file on disk disagrees with `load_qc()`.

## How to read it correctly

- **Counting valid edits (per shard):** sum the run-log lines
  `[gate_quality] <obj> done: pass=P fail=F skip=S` (dedupe by obj, take the
  last — resume can re-log). `P` summed across objects = valid edits.
  Validated on shard00 → **2284 valid edits across 989 / 1203 objects**
  (3805 edits reached 3D; ~60% Gate-E pass among rendered).
- **In code:** use `qc_io.load_qc(ctx)` — its read view reconciles `gates`
  from `stages`. Do **not** trust the persisted `final_pass`.
- `edit_status.json` accumulates edits across *all* historical runs/configs
  (one shard00 object held 13129 edit entries, no GC) — per-edit aggregates
  over the raw files mix in stale data; prefer the current run's log.

## Why it grew this way (accretion across generations)

| historical layer | residue in the schema |
|---|---|
| a separate `status.json` was merged INTO `edit_status.json` | top-level `steps` coexists awkwardly with `edits` (see `status.py` docstring: "single source of truth for both per-edit and per-step") |
| light `stages` markers predate the structured `gates` payloads | one gate recorded in two places, no enforced consistency |
| Gate C (2D quality) added, then disabled in the `no2dqc` config | `gates.C` is always `null` yet occupies a slot in every schema, every `_sync_fail_fields`, every `final_pass` |
| runs/configs accumulate, no garbage collection | thousands of stale edit entries per object |
| `schema_version=2` + migration shims | defensive `setdefault` fallbacks + the read-path backfill papering over divergence |

Each generation **added a parallel representation instead of replacing the old
one**, and the divergences are patched by `_sync_fail_fields` + the read-path
backfill rather than by a single canonical write path.

## Deferred cleanup (do AFTER the 10-shard batch — NOT mid-run)

Changing the write path mid-batch is unsafe: `final_pass` feeds
`is_edit_qc_failed` (`validators.py:268`), which decides skip-on-resume, and
the driver launches each shard against whatever code is on disk → behavioural
drift between shards. Once shards 00–09 finish:

1. **Unify the gate write path** — write `stages` and `gates` together (one
   `record_gate` that updates both), so ②/③ can't diverge.
2. **Derive `final_pass` from the reconciled view and write it back** in
   `_save_es`, eliminating the disk-vs-memory disagreement.
3. **Drop the vestigial Gate C** when `no2dqc` (don't carry an always-null slot).
4. **Optional:** GC historical edit entries not belonging to the active config.

Until then: read via the log tally or `load_qc()`. A read-only valid-edit
extractor (safe to run against live data) is the interim tool of choice.
