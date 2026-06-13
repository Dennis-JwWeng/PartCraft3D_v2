#!/usr/bin/env python3
"""CPU preflight for PartVerse XL raw inputs (no GPU).

Checks each object under ``data.xl_root`` before pipeline stages:

  * mesh dir exists with ``{N}.glb`` parts
  * ``captions/<uuid>/caption.json`` exists
  * part count ≤ ``max_parts`` (default 16, matches gen_edits ``MAX_PARTS``)
  * optional trimesh ``full.glb`` assembly probe (``--check-assemble``)

Writes allow/block lists for ``OBJ_IDS_FILE`` and exits non-zero when any
object is blocked (unless ``--warn-only``).

Usage
-----
    # Full shard roster from configs/partversexl_shards/shard_00.txt
    python scripts/ops/preflight_partversexl.py \\
        --config configs/pipeline_v3_trellis2_partversexl_posthoc_no2dqc.yaml \\
        --shard 00 \\
        --obj-ids-file configs/partversexl_shards/shard_00.txt \\
        --write-allow configs/partversexl_shards/shard_00_allow.txt \\
        --write-block configs/partversexl_shards/shard_00_block.tsv

    # Quick assemble probe on allow list (slower)
    python scripts/ops/preflight_partversexl.py ... --check-assemble

Then run the pipeline on the allow list only::

    OBJ_IDS_FILE=configs/partversexl_shards/shard_00_allow.txt \\
      bash run_pipeline_v3_shard_trellis2.sh ...
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.data_prep.xl_preflight import (  # noqa: E402
    inspect_batch,
    partition_reports,
    preflight_config,
    reason_kind,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--shard", default="00")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--obj-ids-file", type=Path, metavar="TXT")
    group.add_argument("--obj-ids", nargs="+", metavar="OBJ_ID")
    group.add_argument("--all", action="store_true",
                       help="all ids from RawXLSource shard partition")
    ap.add_argument("--max-parts", type=int, default=None,
                    help="override data.xl_preflight.max_parts")
    ap.add_argument("--check-assemble", action="store_true",
                    help="trimesh full.glb assembly probe (slower)")
    ap.add_argument("--write-allow", type=Path, metavar="TXT",
                    help="write passing obj_ids (one per line)")
    ap.add_argument("--write-block", type=Path, metavar="TSV",
                    help="write blocked obj_ids + reasons (tab-separated)")
    ap.add_argument("--warn-only", action="store_true",
                    help="exit 0 even when objects are blocked")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pf = preflight_config(cfg)

    if args.all:
        from scripts.data_prep.mesh_sources import get_mesh_source

        ms = get_mesh_source(cfg)
        data = cfg.get("data") or {}
        num_shards = int(data.get("num_shards", 10))
        obj_ids = ms.list_object_ids(args.shard.zfill(2), num_shards)
    elif args.obj_ids_file:
        obj_ids = [
            l.strip()
            for l in args.obj_ids_file.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
    else:
        obj_ids = list(args.obj_ids)

    if not obj_ids:
        sys.exit("No object IDs resolved")

    reports = inspect_batch(
        obj_ids,
        cfg,
        max_parts=args.max_parts,
        check_assemble=args.check_assemble or pf["check_assemble"],
    )
    allow, blocked = partition_reports(reports)

    reason_counts: Counter[str] = Counter()
    for r in blocked:
        for reason in r.reasons:
            reason_counts[reason_kind(reason)] += 1

    print(f"Preflight XL  shard={args.shard}  total={len(reports)}  "
          f"allow={len(allow)}  block={len(blocked)}")
    if reason_counts:
        print("  block reasons:")
        for k, n in reason_counts.most_common():
            print(f"    {k}: {n}")

    if args.verbose:
        for r in blocked[:20]:
            print(f"  BLOCK  {r.obj_id}\t{';'.join(r.reasons)}")
        if len(blocked) > 20:
            print(f"  ... and {len(blocked) - 20} more")

    if args.write_allow:
        args.write_allow.parent.mkdir(parents=True, exist_ok=True)
        args.write_allow.write_text(
            "\n".join(r.obj_id for r in allow) + ("\n" if allow else "")
        )
        print(f"  wrote allow → {args.write_allow}")

    if args.write_block:
        args.write_block.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{r.obj_id}\t{';'.join(r.reasons)}" for r in blocked]
        args.write_block.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  wrote block → {args.write_block}")

    if blocked and not args.warn_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
