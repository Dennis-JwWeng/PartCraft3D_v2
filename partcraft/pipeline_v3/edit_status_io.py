"""Per-edit operational state tracking (edit_status.json).

``edit_status.json`` records which pipeline stages each edit has reached
and whether they succeeded.  It complements (does not replace)
``status.json`` (step-level aggregates) and ``qc.json`` (gate quality
signals).

Schema::

    {
      "obj_id": "...",
      "shard": "06",
      "schema_version": 1,
      "updated": "2026-04-14T07:00:00",
      "edits": {
        "mod_..._001": {
          "edit_type": "modification",
          "stages": {
            "gate_a": {"status": "pass", "ts": "..."},
            "s4":     {"status": "done", "ts": "..."},
            ...
          }
        }
      }
    }

Status values:
  - ``"pass"`` / ``"fail"`` for gate stages
  - ``"done"`` / ``"error"`` for processing stages
  - absent key = not yet reached
"""
from __future__ import annotations

import atexit
import copy
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ObjectContext

SCHEMA_VERSION = 1

# ─────────────────────────────────────────────────────────────────────────────
# Write-behind cache for edit_status.json
#
# edit_status.json is read-modify-written on the hot path (every step / every
# edit-stage transition), often from inside the asyncio event loop.  On the
# networked CPFS/FUSE backing store a single fcntl.lockf + atomic-replace can
# spike to seconds and FREEZE the whole event loop (all VLM consumers stall in
# lockstep, GPU idles).  To decouple FS latency from the hot path:
#
#   * an in-process cache holds each object's latest edit_status (source of
#     truth during the run) → reads never touch disk after the first load;
#   * mutations update the cache under a per-object threading.Lock and mark it
#     dirty → the hot path returns immediately;
#   * a background daemon thread flushes dirty entries to disk (tmp + replace),
#     coalescing repeated writes, so an FS hiccup can never block a consumer;
#   * flush_edit_status() drains synchronously for stage-end durability and is
#     registered atexit as a safety net.
#
# The cross-process fcntl.lockf is dropped: each object is owned by exactly one
# GPU shard process (run_trellis2 --gpu-shard k/N), so its edit_status.json is
# never written concurrently by two processes.  The per-object threading.Lock
# gives the in-process atomicity that step (status.py) and edit writers share.
# ─────────────────────────────────────────────────────────────────────────────

# Per-object lock is REENTRANT: the read-modify-write helpers acquire it via
# _edit_status_lock and then call load_edit_status / save_edit_status, which
# re-acquire the same key's lock on the same thread.  A plain Lock would
# self-deadlock there; RLock allows the nested same-thread acquisition while
# still excluding other threads (e.g. the background writer).
_es_locks: dict[str, "threading.RLock"] = {}
_es_locks_guard = threading.Lock()
_es_cache: dict[str, dict[str, Any]] = {}
_es_dirty: set[str] = set()
_es_cv = threading.Condition()
_es_writer_started = False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _es_path(ctx: ObjectContext) -> Path:
    return ctx.dir / "edit_status.json"


def _es_key(ctx: ObjectContext) -> str:
    return str(_es_path(ctx).resolve())


def _es_lock_for(key: str) -> "threading.RLock":
    with _es_locks_guard:
        lk = _es_locks.get(key)
        if lk is None:
            lk = _es_locks[key] = threading.RLock()
        return lk


def _es_read_disk(ctx: ObjectContext) -> dict[str, Any]:
    p = _es_path(ctx)
    _empty: dict[str, Any] = {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "schema_version": SCHEMA_VERSION,
        "updated": None,
        "edits": {},
    }
    for _attempt in range(3):
        try:
            if not p.is_file():
                return _empty
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else _empty
        except json.JSONDecodeError:
            return _empty
        except OSError:
            import time as _t
            _t.sleep(0.5 * (_attempt + 1))
    return _empty


def _es_write_disk(key: str) -> None:
    """Flush one cached entry to disk (atomic).  Runs OFF the hot path."""
    lk = _es_lock_for(key)
    with lk:
        snap = _es_cache.get(key)
        snap = copy.deepcopy(snap) if snap is not None else None
    if snap is None:
        return
    p = Path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Preserve qc_io-owned per-edit fields. qc_io writes gates/final_pass
    # straight to disk, bypassing this module's _es_cache; the cache is seeded
    # once and never re-reads disk, so flushing our snapshot would otherwise
    # clobber a best_view/gate written out-of-band after the seed. Re-read disk
    # and graft those fields back. Disk is always >= cache in freshness for
    # these keys (we never write them), so preferring disk is safe. A transient
    # CPFS read error just degrades to the old (clobber-prone) behaviour.
    try:
        _disk = json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, json.JSONDecodeError):
        _disk = {}
    _disk_edits = _disk.get("edits", {}) if isinstance(_disk, dict) else {}
    for _eid, _entry in (snap.get("edits", {}) or {}).items():
        _de = _disk_edits.get(_eid)
        if not isinstance(_de, dict):
            continue
        for _k in ("gates", "final_pass"):
            if _k in _de:
                _entry[_k] = _de[_k]
    fd, tmp = tempfile.mkstemp(prefix=".es.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _es_writer_loop() -> None:
    while True:
        with _es_cv:
            while not _es_dirty:
                _es_cv.wait()
            key = _es_dirty.pop()
        try:
            _es_write_disk(key)
        except Exception:
            # Re-queue once; never let a transient FS error kill the writer.
            with _es_cv:
                _es_dirty.add(key)
                _es_cv.wait(timeout=1.0)


def _es_ensure_writer() -> None:
    global _es_writer_started
    if _es_writer_started:
        return
    with _es_locks_guard:
        if _es_writer_started:
            return
        t = threading.Thread(target=_es_writer_loop, name="edit-status-writer",
                             daemon=True)
        t.start()
        _es_writer_started = True


def flush_edit_status(timeout: float = 60.0) -> None:
    """Synchronously drain all pending edit_status writes (stage-end durability).

    Writes on the calling thread so it works even at interpreter shutdown when
    the daemon writer may already be gone.
    """
    import time as _t
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        with _es_cv:
            if not _es_dirty:
                return
            key = _es_dirty.pop()
        try:
            _es_write_disk(key)
        except Exception:
            pass


atexit.register(flush_edit_status)


@contextmanager
def _edit_status_lock(ctx: ObjectContext):
    """Per-object in-process lock for the edit_status read-modify-write window.

    Shared by step writers (status.py) and edit-stage writers so they never
    clobber each other's slice of the same edit_status.json.  No fcntl — see
    the module note above.
    """
    lk = _es_lock_for(_es_key(ctx))
    with lk:
        yield


# --- I/O ---

def load_edit_status(ctx: ObjectContext) -> dict[str, Any]:
    """Return a mutable copy of the object's edit_status (cache-backed)."""
    key = _es_key(ctx)
    lk = _es_lock_for(key)
    with lk:
        data = _es_cache.get(key)
        if data is None:
            data = _es_read_disk(ctx)
            _es_cache[key] = data
        return copy.deepcopy(data)


def save_edit_status(ctx: ObjectContext, data: dict[str, Any]) -> None:
    """Update the in-memory cache and schedule an async disk flush.

    Returns immediately — disk I/O happens on the background writer thread, so
    a slow networked FS can never block the caller (e.g., the asyncio loop).
    """
    data["updated"] = _now()
    key = _es_key(ctx)
    lk = _es_lock_for(key)
    with lk:
        _es_cache[key] = copy.deepcopy(data)
    _es_ensure_writer()
    with _es_cv:
        _es_dirty.add(key)
        _es_cv.notify()


# --- read-modify-write ---

def update_edit_stage(
    ctx: ObjectContext,
    edit_id: str,
    edit_type: str,
    stage_key: str,
    *,
    status: str,
    reason: str | None = None,
    verdict: dict[str, Any] | None = None,
) -> None:
    """Atomically set the status for one stage of one edit.

    Merges IN PLACE into the existing stage entry rather than replacing it, so
    fields written by a different call survive. In particular a gate's
    ``verdict`` (best_view, rule/vlm payload — written by qc_io.update_edit_gate
    via this same function) is not clobbered by a later bare status write, and
    vice-versa. This is the single authoritative per-(edit, stage) record:
    ``stages.<stage> = {status, ts, reason?, verdict?}``.

    Process-safe via the shared per-object lock + write-behind cache.
    """
    with _edit_status_lock(ctx):
        es = load_edit_status(ctx)
        edit_entry = es.setdefault("edits", {}).setdefault(edit_id, {
            "edit_type": edit_type,
            "stages": {},
        })
        stage_entry = edit_entry.setdefault("stages", {}).setdefault(stage_key, {})
        stage_entry["status"] = status
        stage_entry["ts"] = _now()
        if reason is not None:
            stage_entry["reason"] = reason
        if verdict is not None:
            stage_entry["verdict"] = verdict
        save_edit_status(ctx, es)


# --- resume logic ---

def edit_needs_step(
    ctx: ObjectContext,
    edit_id: str,
    stage_key: str,
    prereq_map: dict[str, str | None],
    *,
    force: bool = False,
) -> bool:
    """Single authoritative resume function for all processing steps.

    Returns True iff the edit should be (re-)processed for *stage_key*:
      1. Prerequisite gate must have status ``"pass"`` -- always enforced,
         even when *force* is True.  Additions are exempt from ``gate_a``
         (D5: guaranteed by construction).
      2. Without *force*: own stage absent -> run; ``"error"`` -> retry;
         ``"done"``/``"pass"`` -> skip.
      3. With *force*: always run (gate permitting).
    """
    stages = (load_edit_status(ctx)
              .get("edits", {})
              .get(edit_id, {})
              .get("stages", {}))

    prereq = prereq_map.get(stage_key)
    if prereq:
        if edit_id.startswith("add_") and prereq == "gate_a":
            pass  # D5: additions exempt from gate_a
        else:
            gate_status = stages.get(prereq, {}).get("status")
            if gate_status != "pass":
                return False

    if force:
        return True

    own = stages.get(stage_key)
    if own is None:
        return True
    return own.get("status") == "error"


def gate_already_done(
    ctx: ObjectContext,
    edit_id: str,
    gate_key: str,
) -> bool:
    """True iff the gate has already been evaluated (pass or fail)."""
    stages = (load_edit_status(ctx)
              .get("edits", {})
              .get(edit_id, {})
              .get("stages", {}))
    entry = stages.get(gate_key)
    return entry is not None and entry.get("status") in ("pass", "fail")


def obj_needs_stage(
    ctx: ObjectContext,
    edit_ids: list[str],
    stage_key: str,
    prereq_map: dict[str, str | None],
    *,
    force: bool = False,
) -> bool:
    """Return True iff any edit in *edit_ids* needs *stage_key* processed.

    Loads ``edit_status.json`` exactly once for the object, making this
    efficient for per-object filtering before dispatching to a step runner.
    Replicates the gate + force + own-status logic of :func:`edit_needs_step`.
    """
    es_edits = load_edit_status(ctx).get("edits", {})
    prereq = prereq_map.get(stage_key)
    for edit_id in edit_ids:
        stages = es_edits.get(edit_id, {}).get("stages", {})
        if prereq:
            if not (edit_id.startswith("add_") and prereq == "gate_a"):
                if stages.get(prereq, {}).get("status") != "pass":
                    continue  # gate not passed → skip this edit
        if force:
            return True
        own = stages.get(stage_key)
        if own is None or own.get("status") == "error":
            return True
    return False


# --- config-derived prerequisites ---

def build_prereq_map(cfg: dict) -> dict[str, str | None]:
    """Return the prerequisite gate key for each pipeline step.

    Each entry maps a step identifier (as used in ALL_STEPS / --steps) to
    the ``edit_status.json`` stage key that must have ``status == "pass"``
    before that step is allowed to run on a given edit.

    Gate prerequisite rules (Mode E):
      * ``del_mesh`` and ``preview_del`` require gate_a (text-align gate).
      * ``gate_quality`` (Gate E) has no hard file prerequisite here;
        it is gated implicitly by the presence of preview_{0..4}.png.
      * Inactive steps (flux_2d, trellis_3d, etc.) are commented out.

    The ``active`` set is derived from ``pipeline.stages[].steps`` in the
    YAML config, so enabling a QC gate in config automatically wires it as
    a prerequisite without touching this function.
    """
    # Read active step names from config stages to decide whether each gate
    # should act as a prerequisite.  Both the functional names (new CLI names)
    # and the internal stage keys are accepted so that either config style works.
    active = {step for stage in cfg.get("pipeline", {}).get("stages", [])
              for step in stage.get("steps", [])}
    # gate_a is a prerequisite when gate_text_align (or its legacy alias sq1)
    # is declared in the active stages.
    gate_a_active = "gate_text_align" in active or "sq1" in active
    gate_c_active = "gate_2d" in active or "sq2" in active
    gate_e_active = "gate_quality" in active or "sq3" in active
    prereq_a = "gate_a" if gate_a_active else None
    prereq_c = "gate_c" if gate_c_active else None
    prereq_e = "gate_e" if gate_e_active else None

    # Keys here are the INTERNAL stage keys used by step runners when they
    # call edit_needs_step(ctx, edit_id, stage_key, prereq_map).
    # Do NOT change these to functional/CLI names — the step runner code
    # (mesh_deletion.py, preview_render.py, etc.) passes these strings directly.
    return {
        # Internal stage keys used by step runners in edit_needs_step() calls.
        # These MUST match the strings passed in s4/s5/s5b/s6/s6p/*.py code.
        "s4":  prereq_a,   # flux_2d step runner (flux_2d.py)
        # trellis_3d / trellis2_3d: require Gate C (2D edit OK) when active, so a
        # 2D edit that fails the instruction never reaches the 3D edit; falls
        # back to gate_a when Gate C is not in the pipeline.
        "s5":  prereq_c or prereq_a,

        "s5b": prereq_a,   # del_mesh step runner (mesh_deletion.py)
        "s6p": prereq_a,   # preview_del / preview_flux step runner (preview_render.py)
        "s6":  prereq_e,   # render_3d step runner (render_3d.py)
        "s6b": prereq_e,   # slat-asset linker (mesh_deletion.py link_slat_assets_batch)
    }


__all__ = [
    "load_edit_status", "save_edit_status", "obj_needs_stage",
    "update_edit_stage", "edit_needs_step", "gate_already_done",
    "build_prereq_map",
]
