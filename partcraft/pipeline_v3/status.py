"""Per-object step status backed by edit_status.json.

This module keeps the legacy status API (`load_status`, `update_step`,
`step_done`, `rebuild_manifest`) but stores data inside
``edit_status.json`` as top-level ``steps``. This makes ``edit_status.json``
the single source of truth for both per-edit and per-step orchestration.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import ObjectContext, PipelineRoot

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"
STATUS_PENDING = "pending"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _es_path(ctx: ObjectContext) -> Path:
    return ctx.dir / "edit_status.json"


def _load_edit_status(ctx: ObjectContext) -> dict[str, Any]:
    from .edit_status_io import load_edit_status as _canonical_load
    data = _canonical_load(ctx)
    data.setdefault("steps", {})
    return data


def _save_edit_status(ctx: ObjectContext, data: dict[str, Any]) -> None:
    # Delegate to the canonical write-behind writer so step (`steps`) and
    # edit (`edits`) writes share one cache + lock + async flush, and a slow
    # networked FS never blocks the caller (e.g. the asyncio event loop).
    from .edit_status_io import save_edit_status as _canonical_save
    data["obj_id"] = ctx.obj_id
    data["shard"] = ctx.shard
    data.setdefault("schema_version", 2)
    _canonical_save(ctx, data)


@contextmanager
def _status_lock(ctx: ObjectContext):
    """Per-object lock for edit_status read-modify-write — shared with the
    edit-stage writers via the canonical in-process lock (no fcntl)."""
    from .edit_status_io import _edit_status_lock
    with _edit_status_lock(ctx):
        yield


def load_status(ctx: ObjectContext) -> dict[str, Any]:
    """Read step-level status from edit_status.json."""
    es = _load_edit_status(ctx)
    return {
        "obj_id": ctx.obj_id,
        "shard": ctx.shard,
        "steps": es.get("steps") or {},
        "updated": es.get("updated"),
    }


def save_status(ctx: ObjectContext, status: dict[str, Any]) -> None:
    """Persist step-level status into edit_status.json."""
    es = _load_edit_status(ctx)
    es["steps"] = dict(status.get("steps") or {})
    _save_edit_status(ctx, es)


def update_step(
    ctx: ObjectContext,
    step: str,
    *,
    status: str = STATUS_OK,
    **fields: Any,
) -> dict[str, Any]:
    """Read-modify-write one step entry (process-safe)."""
    with _status_lock(ctx):
        s = load_status(ctx)
        s.setdefault("steps", {})[step] = {
            "status": status, "ts": _now(), **fields,
        }
        save_status(ctx, s)
    return s


def step_done(ctx: ObjectContext, step: str) -> bool:
    """True iff the step is marked as ok."""
    s = load_status(ctx)
    return (s.get("steps") or {}).get(step, {}).get("status") == STATUS_OK


def needs_step(ctx: ObjectContext, step: str, *, force: bool = False) -> bool:
    return force or not step_done(ctx, step)


def rebuild_manifest(root: PipelineRoot) -> Path:
    """Rebuild global manifest from edit_status-backed steps."""
    root.ensure()
    lines: list[str] = []
    if root.objects_root.is_dir():
        for shard_dir in sorted(root.objects_root.iterdir()):
            if not shard_dir.is_dir():
                continue
            shard = shard_dir.name
            for od in sorted(shard_dir.iterdir()):
                if not od.is_dir():
                    continue
                ctx = root.context(shard, od.name)
                s = load_status(ctx)
                lines.append(json.dumps({
                    "shard": shard,
                    "obj_id": od.name,
                    "steps": s.get("steps") or {},
                    "updated": s.get("updated"),
                }, ensure_ascii=False))
    fd, tmp_str = tempfile.mkstemp(
        suffix=".jsonl.tmp", dir=str(root.manifest_path.parent)
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        os.replace(tmp, root.manifest_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return root.manifest_path


def manifest_summary(root: PipelineRoot) -> dict[str, Any]:
    """Cheap aggregate over the manifest (does not rebuild)."""
    if not root.manifest_path.is_file():
        return {"objects": 0, "steps": {}}
    objs = 0
    step_counts: dict[str, dict[str, int]] = {}
    with open(root.manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            objs += 1
            for k, v in (r.get("steps") or {}).items():
                bucket = step_counts.setdefault(k, {})
                bucket[v.get("status", "?")] = bucket.get(
                    v.get("status", "?"), 0) + 1
    return {"objects": objs, "steps": step_counts}


__all__ = [
    "STATUS_OK", "STATUS_FAIL", "STATUS_SKIP", "STATUS_PENDING",
    "load_status", "save_status", "update_step",
    "step_done", "needs_step",
    "rebuild_manifest", "manifest_summary",
]
