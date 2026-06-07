"""Gate QC helpers stored in edit_status.json.

Legacy callers still use the qc_io API, but data is persisted in
``edit_status.json`` under each edit entry (no qc.json writes).
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ObjectContext


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def _es_lock(ctx: ObjectContext):
    import threading

    lock_path = ctx.dir / "edit_status.json.lock"
    ctx.dir.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    if not hasattr(_es_lock, "_thread_mutexes"):
        _es_lock._thread_mutexes = {}
        _es_lock._thread_mutexes_guard = threading.Lock()
    with _es_lock._thread_mutexes_guard:
        if key not in _es_lock._thread_mutexes:
            _es_lock._thread_mutexes[key] = threading.Lock()
        thread_mtx = _es_lock._thread_mutexes[key]
    with thread_mtx:
        with open(lock_path, "a") as lf:
            fcntl.lockf(lf, fcntl.LOCK_EX)
            yield


def _load_es(ctx: ObjectContext) -> dict[str, Any]:
    p = ctx.dir / "edit_status.json"
    default = {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "schema_version": 2,
        "updated": None,
        "edits": {},
        "steps": {},
    }
    # /mnt is a networked CPFS mount: is_file()/read_text() can transiently
    # raise OSError [Errno 116] Stale file handle. Retry before giving up —
    # previously only JSONDecodeError was caught, so a single ESTALE killed
    # the whole shard (cf. edit_status_io._es_read_disk). On a *persistent*
    # OSError we re-raise rather than return the default, because callers
    # load-modify-save and a default load would clobber real gate data.
    for attempt in range(5):
        try:
            if not p.is_file():
                return default
            data = json.loads(p.read_text())
            return data if isinstance(data, dict) else default
        except json.JSONDecodeError:
            return default
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))
    return default


def _save_es(ctx: ObjectContext, es: dict[str, Any]) -> None:
    es["obj_id"] = ctx.obj_id
    es["shard"] = ctx.shard
    es.setdefault("schema_version", 2)
    es["updated"] = _now()
    ctx.dir.mkdir(parents=True, exist_ok=True)
    target = ctx.dir / "edit_status.json"
    # Atomic write, retried against transient CPFS ESTALE (see _load_es).
    for attempt in range(5):
        fd, tmp = tempfile.mkstemp(prefix=".es.", suffix=".tmp", dir=str(ctx.dir))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(es, f, ensure_ascii=False, indent=2)
            os.replace(tmp, target)
            return
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise


def _gp(gd: dict | None) -> bool:
    if gd is None:
        return True
    r = gd.get("rule")
    v = gd.get("vlm")
    if r is not None and not r.get("pass", True):
        return False
    if v is not None and not v.get("pass", True):
        return False
    return True


def _sync_fail_fields(entry: dict[str, Any]) -> None:
    gates = entry.setdefault("gates", {})
    # Ensure all gate keys exist so downstream code can iterate safely.
    # Gate C (2D quality) is optional; absent gates count as passed.
    for _g in ("A", "C", "E"):
        gates.setdefault(_g, None)
    entry["final_pass"] = all(_gp(gates[g]) for g in ("A", "C", "E"))
    if not entry["final_pass"]:
        for g in ("A", "C", "E"):
            gd2 = gates.get(g)
            if gd2 is not None and not _gp(gd2):
                entry["fail_gate"] = g
                r = gd2.get("rule") if isinstance(gd2, dict) else None
                v = gd2.get("vlm") if isinstance(gd2, dict) else None
                if r and not r.get("pass", True):
                    entry["fail_reason"] = next(iter(r.get("checks") or {}), "rule_fail")
                elif v and not v.get("pass", True):
                    entry["fail_reason"] = (v.get("reason") or "vlm_fail")[:80]
                break
    else:
        entry.pop("fail_gate", None)
        entry.pop("fail_reason", None)


def _build_qc_view_from_es(ctx: ObjectContext, es: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the legacy {gates:{A,C,E}, final_pass} view from the
    authoritative per-edit ``stages`` record.

    Gate payloads now live at ``stages.<gate>.verdict``. We source from there,
    fall back to ``stages.<gate>.status`` when only a status was written, and
    finally to a pre-migration top-level ``gates`` map for un-migrated files.
    ``final_pass``/``fail_gate``/``fail_reason`` are DERIVED here (read-time),
    keeping the same lenient _gp semantics readers relied on before.
    """
    edits_out: dict[str, Any] = {}
    for edit_id, e in (es.get("edits") or {}).items():
        if not isinstance(e, dict):
            continue
        stages = e.get("stages") or {}
        legacy = e.get("gates") or {}  # only present on un-migrated files
        gates: dict[str, Any] = {}
        for k, g in (("gate_a", "A"), ("gate_c", "C"), ("gate_e", "E")):
            st = stages.get(k)
            st = st if isinstance(st, dict) else None
            if st and st.get("verdict"):
                gates[g] = st["verdict"]
            elif st and st.get("status") in ("pass", "fail"):
                gates[g] = {"vlm": {"pass": st["status"] == "pass",
                                    "reason": "from_stage_status"}}
            elif legacy.get(g) is not None:
                gates[g] = legacy[g]
            else:
                gates[g] = None

        view_entry: dict[str, Any] = {"edit_type": e.get("edit_type", ""), "gates": gates}
        _sync_fail_fields(view_entry)  # derive final_pass + fail_gate/fail_reason
        out = {
            "edit_type": view_entry["edit_type"],
            "gates": gates,
            "final_pass": view_entry["final_pass"],
        }
        if "fail_gate" in view_entry:
            out["fail_gate"] = view_entry["fail_gate"]
        if "fail_reason" in view_entry:
            out["fail_reason"] = view_entry["fail_reason"]
        edits_out[edit_id] = out

    return {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "updated": es.get("updated"),
        "edits": edits_out,
    }


def load_qc(ctx: ObjectContext) -> dict[str, Any]:
    # Read through the canonical cache so we see in-process writes (gate
    # verdicts now go through edit_status_io, not qc_io's own disk path).
    from .edit_status_io import load_edit_status
    es = load_edit_status(ctx)
    return _build_qc_view_from_es(ctx, es)


def save_qc(ctx: ObjectContext, qc: dict[str, Any]) -> None:
    with _es_lock(ctx):
        es = _load_es(ctx)
        edits = es.setdefault("edits", {})
        for edit_id, qentry in (qc.get("edits") or {}).items():
            if not isinstance(qentry, dict):
                continue
            e = edits.setdefault(edit_id, {
                "edit_type": qentry.get("edit_type", ""),
                "stages": {},
            })
            e["edit_type"] = qentry.get("edit_type", e.get("edit_type", ""))
            e["gates"] = dict(qentry.get("gates") or {"A": None, "C": None, "E": None})
            _sync_fail_fields(e)
        _save_es(ctx, es)


_GATE_TO_STAGE = {"A": "gate_a", "C": "gate_c", "E": "gate_e"}


def update_edit_gate(
    ctx: ObjectContext,
    edit_id: str,
    edit_type: str,
    gate: str,
    *,
    rule_result: dict | None = None,
    vlm_result: dict | None = None,
) -> None:
    """Persist a gate verdict into the single authoritative per-edit record.

    The verdict (rule/vlm payload incl. best_view) and its pass/fail status go
    to ``stages.<gate>`` via edit_status_io — the SAME cache + lock as s4/s5 —
    not a separate top-level ``gates`` map written out-of-band. That removes the
    second writer that raced the stage writer and silently wiped best_view.
    ``load_qc`` reconstructs the legacy {gates, final_pass} view for readers.
    """
    from .edit_status_io import update_edit_stage
    gd: dict[str, Any] = {}
    if rule_result is not None:
        gd["rule"] = rule_result
    if vlm_result is not None:
        gd["vlm"] = vlm_result
    verdict = gd if gd else None
    status = "pass" if _gp(verdict) else "fail"
    stage_key = _GATE_TO_STAGE.get(gate, gate)
    update_edit_stage(ctx, edit_id, edit_type, stage_key,
                      status=status, verdict=verdict)


def is_edit_qc_failed(ctx: ObjectContext, edit_id: str) -> bool:
    e = load_qc(ctx).get("edits", {}).get(edit_id)
    return e is not None and e.get("final_pass", True) is False


def is_gate_a_failed(ctx: ObjectContext, edit_id: str) -> bool:
    """Block only on Gate A failures (upstream part-selection errors)."""
    e = load_qc(ctx).get("edits", {}).get(edit_id)
    if e is None:
        return False
    gate_a = (e.get("gates") or {}).get("A")
    return not _gp(gate_a)


__all__ = ["load_qc", "save_qc", "update_edit_gate", "is_edit_qc_failed", "is_gate_a_failed"]
