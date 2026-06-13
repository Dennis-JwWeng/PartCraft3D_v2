"""Persistent 'bad mesh' registry + per-worker in-flight guard.

Some PartVerse-XL meshes hard-CRASH a GPU step — a process-fatal SIGSEGV that
canNOT be caught with try/except.  The two observed sites are both in native
(o_voxel / CUDA) extensions on degenerate geometry:

  * encode: ``o_voxel.convert.volumetic_attr.textured_mesh_to_volumetric_attr``
    (the core textured-mesh voxelizer in ``encode_shape_tex_ss``), and
  * render: the o-voxel / PbrMeshRenderer used for the gate views.

Because the fault kills the whole process, we detect the culprit out-of-band:
before touching the GPU a worker stamps the object it is ABOUT to process into
an 'inflight' marker file.  If the process dies, the marker survives; the next
incarnation of that worker (respawned by ``dispatch_gpus``'s supervisor, or the
next launch) reads the marker, records the object in the shared bad-mesh
registry, and skips it thereafter.  All GPU steps load the registry up front and
skip known-bad objects, so one crash quarantines a mesh everywhere, for good.

Registry: ``<root>/_global/bad_meshes.jsonl`` — one ``{obj_id, stage, reason,
ts}`` per line (human-readable, append-only, de-duplicated on obj_id).
"""
from __future__ import annotations

import json
import os
from datetime import datetime


def registry_path(root):
    return root.global_dir / "bad_meshes.jsonl"


def load_bad(root) -> set[str]:
    """Return the set of obj_ids recorded as bad (process-fatal) so far."""
    p = registry_path(root)
    out: set[str] = set()
    if not p.is_file():
        return out
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.add(json.loads(line)["obj_id"])
            except Exception:
                continue
    except OSError:
        pass
    return out


def record_bad(root, obj_id: str, stage: str, reason: str = "process_fatal") -> bool:
    """Append ``obj_id`` to the registry (no-op if already present).  Returns
    True if newly recorded."""
    if obj_id in load_bad(root):
        return False
    p = registry_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "a") as fh:
            fh.write(json.dumps({
                "obj_id": obj_id, "stage": stage, "reason": reason,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }) + "\n")
    except OSError:
        return False
    return True


class InflightGuard:
    """Per-worker in-flight marker → bad-mesh detector (+ heartbeat via mtime).

    Constructed once per worker.  ``load_bad`` is read into ``self.bad``; if a
    stale marker from THIS worker exists it means the previous incarnation died
    on that object → it is recorded bad and added to ``self.bad``.  Call
    :meth:`beat` before each object and :meth:`clear` after it completes (even on
    a caught failure); a process-fatal crash leaves the marker set so the next
    incarnation quarantines the culprit.
    """

    def __init__(self, root, shard: str, stage: str, log):
        self.root = root
        self.stage = stage
        self.log = log
        d = root.global_dir / "_inflight"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "x").replace(",", "_")
        self.marker = d / f"{stage}_{shard}_cvd{cvd}.inflight"
        self.bad: set[str] = load_bad(root)
        if self.marker.is_file():
            obj = ""
            try:
                obj = self.marker.read_text().strip()
            except OSError:
                pass
            if obj:
                if record_bad(root, obj, stage):
                    log.error("[%s/badmesh] %s crashed the previous worker "
                              "(process-fatal) — recorded bad + skipping", stage, obj)
                self.bad.add(obj)
            try:
                self.marker.unlink()
            except OSError:
                pass

    def beat(self, obj_id: str) -> None:
        try:
            self.marker.write_text(obj_id)
        except OSError:
            pass

    def clear(self) -> None:
        try:
            self.marker.unlink()
        except OSError:
            pass


def make_guard(root, shard: str, stage: str, log):
    """InflightGuard when a respawn supervisor manages this worker
    (TRELLIS2_ISOLATE_RESPAWN=1), else None (registry is still consulted via
    :func:`load_bad`)."""
    if os.environ.get("TRELLIS2_ISOLATE_RESPAWN", "0") != "1":
        return None
    return InflightGuard(root, shard, stage, log)
