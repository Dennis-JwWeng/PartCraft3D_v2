#!/usr/bin/env python3
"""Dump the Trellis 64³ edit mask + SLAT-aligned preserve flags for one edit.

Uses the same path as pipeline s5/s5b expectations: ``TrellisRefiner.build_part_mask``
via :class:`partcraft.pipeline_v3.mask_materialization.PartCraftRuntimeMaskBuilder`.

Output NPZ keys
---------------
``mask_edit_64`` : uint8 ``[64,64,64]`` — **1** = editable voxel (same semantics as
the mask passed to ``interweave_Trellis_TI``).

``mask_keep_64`` / ``mask_keep_ss`` — dilated-keep region for coarse SS losses
(same construction as ``mask_materialization._write_sidecar``).

``slat_coords`` : int16 ``[N,3]`` — ``(x,y,z)`` indices into the 64³ grid.

``slat_in_edit_region`` / ``slat_preserve`` : uint8 ``{0,1}`` length ``N``.

``meta.json`` (next to the NPZ) holds :class:`~partcraft.pipeline_v3.specs.EditSpec`
fields as JSON for traceability.

Examples
--------
    python scripts/tools/dump_edit_part_mask.py \\
        --config configs/pipeline_v3_shard01.yaml \\
        --shard 01 --obj-id 109d5f12e82b4bb4a081c536f39cf729 \\
        --edit-id mod_109d5f12e82b4bb4a081c536f39cf729_000 \\
        --out /tmp/mask_dump.npz

    # Ad-hoc part IDs (no pipeline output tree required beyond parsed.json):
    python scripts/tools/dump_edit_part_mask.py \\
        --config configs/pipeline_v3_shard01.yaml \\
        --shard 01 --obj-id 109d5f12e82b4bb4a081c536f39cf729 \\
        --edit-type modification --part-ids 0 1 \\
        --out /tmp/custom_mask.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from partcraft.pipeline_v3.mask_materialization import (
    PartCraftRuntimeMaskBuilder,
    downsample_keep_mask_to_ss,
    mask_keep_from_edit,
)
from partcraft.pipeline_v3.paths import DatasetRoots, PipelineRoot, normalize_shard
from partcraft.pipeline_v3.specs import EditSpec, iter_all_specs


def _load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("data", {})
    return cfg


def _find_spec(ctx_root: PipelineRoot, shard: str, obj_id: str, edit_id: str) -> EditSpec | None:
    ctx = ctx_root.context(shard, obj_id)
    for spec in iter_all_specs(ctx):
        if spec.edit_id == edit_id:
            return spec
    return None


def _make_ad_hoc_spec(
    *,
    shard: str,
    obj_id: str,
    edit_type: str,
    part_ids: list[int],
    edit_idx: int,
) -> EditSpec:
    shard = normalize_shard(shard)
    prefix = edit_type
    if edit_type == "modification":
        prefix = "mod"
    elif edit_type == "deletion":
        prefix = "del"
    elif edit_type == "scale":
        prefix = "scl"
    elif edit_type == "material":
        prefix = "mat"
    elif edit_type == "color":
        prefix = "clr"
    elif edit_type == "global":
        prefix = "glb"
    elif edit_type == "addition":
        prefix = "add"
    eid = f"{prefix}_{obj_id}_{edit_idx:03d}"
    return EditSpec(
        edit_id=eid,
        edit_type=edit_type,
        obj_id=obj_id,
        shard=shard,
        edit_idx=edit_idx,
        view_index=0,
        npz_view=-1,
        selected_part_ids=list(part_ids),
        part_labels=[],
        prompt="",
        target_part_desc="",
        new_parts_desc="",
        edit_params={},
        object_desc="",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="Pipeline YAML (data/ + ckpt_root).")
    p.add_argument("--shard", required=True)
    p.add_argument("--obj-id", required=True)
    p.add_argument("--edit-id", default=None, help="Exact edit id from parsed specs (mutually exclusive with --part-ids).")
    p.add_argument("--edit-type", default=None,
                   help="With --part-ids: canonical edit type, e.g. modification.")
    p.add_argument("--part-ids", nargs="*", type=int, default=None)
    p.add_argument("--edit-idx", type=int, default=0, help="Sequence index for ad-hoc edit_id tail.")
    p.add_argument("--out", required=True, type=Path, help="Output path ending in .npz")
    p.add_argument("--dilation-radius", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--img-enc-dir", default=None, help="Optional data.img_enc_dir override.")
    args = p.parse_args(argv)

    cfg = _load_cfg(args.config)
    roots = DatasetRoots.from_pipeline_cfg(cfg)
    out_dir = cfg["data"].get("output_dir")
    if not out_dir:
        print("[ERROR] data.output_dir is required in --config for PipelineRoot lookup", file=sys.stderr)
        return 1
    ctx_root = PipelineRoot(Path(out_dir))
    shard = normalize_shard(args.shard)

    spec: EditSpec | None = None
    if args.edit_id:
        spec = _find_spec(ctx_root, shard, args.obj_id, args.edit_id)
        if spec is None:
            print(f"[ERROR] edit_id {args.edit_id!r} not found for {shard}/{args.obj_id}", file=sys.stderr)
            return 1
    else:
        if not args.edit_type or args.part_ids is None:
            print("[ERROR] provide --edit-id OR both --edit-type and --part-ids", file=sys.stderr)
            return 1
        spec = _make_ad_hoc_spec(
            shard=shard,
            obj_id=args.obj_id,
            edit_type=args.edit_type,
            part_ids=list(args.part_ids),
            edit_idx=args.edit_idx,
        )

    ckpt_root = cfg.get("ckpt_root")
    if not ckpt_root:
        print("[ERROR] ckpt_root missing from YAML (top-level key)", file=sys.stderr)
        return 1
    if roots.slat_dir is None:
        print("[ERROR] data.slat_dir missing from YAML", file=sys.stderr)
        return 1

    from partcraft.io.partverse_dataset import PartVerseDataset
    from partcraft.trellis.refiner import TrellisRefiner

    img_enc = args.img_enc_dir or cfg["data"].get("img_enc_dir")

    dataset = PartVerseDataset(
        roots.images_root,
        roots.mesh_root,
        [shard],
        slat_dir=roots.slat_dir,
    )
    refiner = TrellisRefiner(
        cache_dir=args.out.parent / "_refiner_cache_dump_mask",
        device=args.device,
        ckpt_dir=str(ckpt_root),
        slat_dir=str(roots.slat_dir),
        image_edit_backend="local_diffusers",
        img_enc_dir=str(img_enc) if img_enc else None,
    )
    refiner.load_models()

    builder = PartCraftRuntimeMaskBuilder(dataset=dataset, refiner=refiner)
    det = builder.build_mask_detail(spec)
    mask_edit = np.asarray(det["mask_edit_64"], dtype=np.uint8)
    mask_keep = mask_keep_from_edit(mask_edit, dilation_radius=args.dilation_radius)
    mask_keep_ss = downsample_keep_mask_to_ss(mask_keep)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mask_edit_64=mask_edit,
        mask_keep_64=mask_keep,
        mask_keep_ss=mask_keep_ss,
        slat_coords=np.asarray(det["slat_coords"], dtype=np.int16),
        slat_in_edit_region=np.asarray(det["slat_in_edit_region"], dtype=np.uint8),
        slat_preserve=np.asarray(det["slat_preserve"], dtype=np.uint8),
        selected_part_ids=np.asarray(spec.selected_part_ids, dtype=np.int32),
    )
    meta = {
        "edit_id": spec.edit_id,
        "edit_type": spec.edit_type,
        "shard": spec.shard,
        "obj_id": spec.obj_id,
        "effective_edit_type": str(det.get("effective_edit_type", "")),
        "dilation_radius": args.dilation_radius,
        "preserve_loss_spaces": [
            "slat_sparse: use slat_preserve + SLAT feats",
            "ss_latent: pair mask_keep_ss with z_s [C,R,R,R] (verify R)",
            "decoded_geometry: custom render / point loss",
        ],
        "training_note": __import__(
            "partcraft.trellis.preserve_loss", fromlist=["training_integration_note"]
        ).training_integration_note(),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(out_path), "meta": str(meta_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
