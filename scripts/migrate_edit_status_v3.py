#!/usr/bin/env python
"""Migrate edit_status.json to the v3 single-authoritative-record schema.

v3 makes ``edits.<id>.stages.<stage>`` the only per-edit record. Gate verdicts
that used to live in a separate top-level ``gates.{A,C,E}`` map (written
out-of-band by qc_io) move INTO the stage entry as ``stages.<gate>.verdict``.
The derived fields ``final_pass`` / ``fail_gate`` / ``fail_reason`` are dropped
(they are now computed at read time by qc_io.load_qc).

Idempotent: a file with no top-level ``gates`` on any edit and
``schema_version == 3`` is left untouched.

Usage:
    python scripts/migrate_edit_status_v3.py <output_dir> [shard ...]
    # e.g.  python scripts/migrate_edit_status_v3.py data/Pxform_v2/prod_posthoc_no2dqc 00 01
    # no shard args -> every shard under <output_dir>/objects
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from glob import glob

GATE_MAP = (("A", "gate_a"), ("C", "gate_c"), ("E", "gate_e"))


def _gp(gd):
    """Lenient gate-pass (matches qc_io._gp): None counts as pass."""
    if gd is None:
        return True
    r = gd.get("rule") if isinstance(gd, dict) else None
    v = gd.get("vlm") if isinstance(gd, dict) else None
    if r is not None and not r.get("pass", True):
        return False
    if v is not None and not v.get("pass", True):
        return False
    return True


def migrate_one(es: dict) -> bool:
    """Mutate *es* in place to v3. Return True iff anything changed."""
    changed = False
    for e in (es.get("edits") or {}).values():
        if not isinstance(e, dict):
            continue
        gates = e.get("gates")
        if isinstance(gates, dict):
            stages = e.setdefault("stages", {})
            for G, gkey in GATE_MAP:
                gv = gates.get(G)
                if not gv:            # None / empty -> nothing to graft
                    continue
                st = stages.get(gkey)
                if not isinstance(st, dict):
                    st = stages[gkey] = {}
                if "status" not in st:
                    st["status"] = "pass" if _gp(gv) else "fail"
                st["verdict"] = gv
            e.pop("gates", None)
            changed = True
        for k in ("final_pass", "fail_gate", "fail_reason"):
            if k in e:
                e.pop(k, None)
                changed = True
    if es.get("schema_version") != 3:
        es["schema_version"] = 3
        changed = True
    return changed


def _read(path):
    for attempt in range(5):
        try:
            return json.loads(open(path).read())
        except json.JSONDecodeError:
            return None
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def _write_atomic(path, data):
    d = os.path.dirname(path)
    for attempt in range(5):
        fd, tmp = tempfile.mkstemp(prefix=".es.", suffix=".tmp", dir=d)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out = sys.argv[1]
    shards = sys.argv[2:]
    roots = ([os.path.join(out, "objects", s) for s in shards]
             if shards else [os.path.join(out, "objects", s)
                             for s in sorted(os.listdir(os.path.join(out, "objects")))])
    scanned = migrated = skipped = bad = 0
    for root in roots:
        files = glob(os.path.join(root, "*", "edit_status.json"))
        sm = 0
        for p in files:
            scanned += 1
            es = _read(p)
            if es is None:
                bad += 1
                continue
            if migrate_one(es):
                _write_atomic(p, es)
                migrated += 1
                sm += 1
            else:
                skipped += 1
        print(f"  {os.path.basename(root):>4}: {len(files):5} files, {sm:5} migrated")
    print(f"\ntotal: scanned={scanned} migrated={migrated} "
          f"already-v3/skip={skipped} bad={bad}")


if __name__ == "__main__":
    main()
