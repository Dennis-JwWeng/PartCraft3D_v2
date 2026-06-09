"""Materialize TRELLIS.2 edit masks as sidecar training assets.

The mask construction mirrors the dataset export
(``scripts/export_pxform_v2_dataset.py`` — ``part_struct_grids`` +
``mask_from_ss``) so training masks are built **exactly** the way the released
del/add dataset is.  It is v2-native and CPU-only: the edit region is voxelized
straight from ``mesh_npz`` via
:func:`partcraft.pipeline_v3.trellis2_part_mask.part_edit_grid_64` — no
``TrellisRefiner``, no SLAT encode, no Open3D (the old v1 ``build_part_mask``
path is gone).

Resolution follows the active config (``trellis2_edit_res``):

* ``edit_res=512``  → 32³ slat / 16³ ss   (prod default; one ``downsample_edit_grid``)
* ``edit_res=1024`` → 64³ slat / 16³ ss   (second variant; no slat downsample)

Per-SLAT preserve flags are aligned to the object's prod-encoded SLAT coords
(``p1_encode/shape_slat_e512.npz`` for 512, ``shape_slat.npz`` for 1024); a coord
is "keep" iff it is **not** in the 32³/64³ ``edit_grid`` (the ``mask_from_ss``
rule).  Use ``FakeBoxMaskBuilder`` for dependency-free dry-run plumbing checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

from partcraft.pipeline_v3.paths import PipelineRoot
from partcraft.pipeline_v3.specs import EditSpec, iter_all_specs

# Mirror the config / dataset export (configs ... trellis2_s1_pad: 4, s1_thresh).
S1_PAD = 4
S1_THRESH = 0.1


class MaskBuilder(Protocol):
    def build(self, spec: EditSpec) -> dict | None:
        """Return the sidecar payload dict for ``spec`` (or ``None`` to skip)."""


class FakeBoxMaskBuilder:
    """Deterministic builder for tests / dry-run plumbing checks (no mesh/SLAT)."""

    def __init__(self, edit_res: int = 512):
        self.edit_res = int(edit_res)
        self.g = self.edit_res // 16  # 32 or 64

    def build(self, spec: EditSpec) -> dict:
        seed = sum(spec.selected_part_ids) if spec.selected_part_ids else spec.edit_idx
        s = 1 + (seed % max(1, self.g - 2))
        edit_grid = np.array([[s, s, s], [s + 1, s, s]], dtype=np.int16)
        keep16 = np.ones((16, 16, 16), dtype=np.uint8)
        return {
            "edit_grid": edit_grid,
            "keep16": keep16,
            "mask_keep_ss": keep16,
            "selected_part_ids": np.asarray(spec.selected_part_ids, dtype=np.int32),
            "edit_type": np.asarray(spec.edit_type),
            "s1_pad": np.int32(S1_PAD),
            "s1_thresh": np.float32(S1_THRESH),
            "edit_res": np.int32(self.edit_res),
        }


class V2StructMaskBuilder:
    """v2-native edit-mask builder — mirrors the dataset export's mask construction.

    Per spec it reproduces ``part_struct_grids`` (``part_edit_grid_64`` →
    ``edit_grid_64_to_keep16`` + ``downsample_edit_grid``) and the
    ``mask_from_ss`` per-SLAT keep, reading the object's prod-encoded SLAT coords
    from ``p1_encode/``.  CPU-only; no model load.
    """

    def __init__(
        self,
        *,
        mesh_root: str | Path,
        pipeline_root: str | Path,
        edit_res: int = 512,
        pad: int = S1_PAD,
        thresh: float = S1_THRESH,
        canonical: bool = True,
    ):
        self.mesh_root = Path(mesh_root)
        self.objects_root = Path(pipeline_root) / "objects"
        self.edit_res = int(edit_res)
        self.pad = int(pad)
        self.thresh = float(thresh)
        self.canonical = bool(canonical)
        self._slat_grid = self.edit_res // 16          # 32 (512) or 64 (1024)
        self._factor = 64 // self._slat_grid           # 2 (512) or 1 (1024)
        # 1024 keeps the main 64³ encode; 512 uses the e512 sidecar.
        self._slat_name = ("shape_slat.npz" if self.edit_res == 1024
                           else f"shape_slat_e{self.edit_res}.npz")

    def _mesh_npz(self, spec: EditSpec) -> Path:
        return self.mesh_root / spec.shard / f"{spec.obj_id}.npz"

    def _slat_coords(self, spec: EditSpec) -> np.ndarray | None:
        p = self.objects_root / spec.shard / spec.obj_id / "p1_encode" / self._slat_name
        if not p.is_file():
            return None
        with np.load(p) as z:
            return np.asarray(z["coords"]).astype(np.int16)

    def _struct_grids(self, spec: EditSpec):
        """Reproduces export part_struct_grids → (keep16 16³, edit_grid [M,3])."""
        import torch
        from partcraft.pipeline_v3.trellis2_part_mask import (
            part_edit_grid_64, edit_grid_64_to_keep16, downsample_edit_grid)
        pids = [int(x) for x in spec.selected_part_ids]
        g64 = part_edit_grid_64(self._mesh_npz(spec), pids, pad=self.pad,
                                canonical=self.canonical)
        keep16 = edit_grid_64_to_keep16(g64, thresh=self.thresh).cpu().numpy().astype(np.uint8)
        g = downsample_edit_grid(g64, self._factor) if self._factor > 1 else g64
        edit_grid = torch.nonzero(g).to(torch.int16).cpu().numpy()
        return keep16, edit_grid

    def build(self, spec: EditSpec) -> dict | None:
        if not self._mesh_npz(spec).is_file():
            return None
        keep16, edit_grid = self._struct_grids(spec)
        payload: dict[str, np.ndarray] = {
            "edit_grid": edit_grid,
            "keep16": keep16,
            "mask_keep_ss": keep16,                     # export-name alias
            "selected_part_ids": np.asarray(spec.selected_part_ids, dtype=np.int32),
            "edit_type": np.asarray(spec.edit_type),
            "s1_pad": np.int32(self.pad),
            "s1_thresh": np.float32(self.thresh),
            "edit_res": np.int32(self.edit_res),
        }
        slat = self._slat_coords(spec)
        if slat is not None:
            # mask_from_ss rule: keep iff the SLAT coord is NOT in edit_grid.
            eg = {tuple(int(x) for x in c) for c in edit_grid.tolist()}
            keep = np.fromiter(
                (tuple(int(x) for x in c) not in eg for c in slat.tolist()),
                dtype=np.uint8, count=len(slat))
            payload["slat_coords"] = slat
            payload["mask_keep_slat"] = keep
        return payload


def _pipeline_root_from_parsed_path(parsed_path: Path) -> PipelineRoot:
    # <root>/objects/<shard>/<obj_id>/phase1/parsed.json
    obj_dir = parsed_path.parent.parent
    shard_dir = obj_dir.parent
    objects_root = shard_dir.parent
    return PipelineRoot(objects_root.parent)


def iter_specs_from_parsed_path(parsed_path: str | Path) -> Iterator[EditSpec]:
    parsed_path = Path(parsed_path)
    root = _pipeline_root_from_parsed_path(parsed_path)
    shard = parsed_path.parent.parent.parent.name
    obj_id = parsed_path.parent.parent.name
    ctx = root.context(shard, obj_id)
    yield from iter_all_specs(ctx)


def iter_addition_specs_from_object_dir(obj_dir: str | Path) -> Iterator[EditSpec]:
    obj_dir = Path(obj_dir)
    edits_3d = obj_dir / "edits_3d"
    if not edits_3d.is_dir():
        return
    shard = obj_dir.parent.name.zfill(2)
    obj_id = obj_dir.name
    for add_dir in sorted(edits_3d.glob("add_*")):
        meta_path = add_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        selected = [int(x) for x in meta.get("selected_part_ids") or []]
        edit_id = meta.get("edit_id") or add_dir.name
        try:
            seq = int(edit_id.rsplit("_", 1)[-1])
        except ValueError:
            seq = 0
        yield EditSpec(
            edit_id=edit_id,
            edit_type="addition",
            obj_id=obj_id,
            shard=shard,
            edit_idx=seq,
            view_index=int(meta.get("view_index", 0) or 0),
            npz_view=-1,
            selected_part_ids=selected,
            part_labels=list(meta.get("part_labels") or []),
            prompt=meta.get("prompt") or "",
            target_part_desc=meta.get("target_part_desc") or "",
            new_parts_desc=meta.get("target_part_desc") or "",
            edit_params={},
            object_desc=meta.get("object_desc") or "",
        )


def _sidecar_path(output_root: Path, spec: EditSpec) -> Path:
    return output_root / spec.edit_type / spec.shard / spec.obj_id / f"{spec.edit_id}.npz"


def _write_sidecar(output_root: Path, spec: EditSpec, payload: dict) -> dict:
    path = _sidecar_path(output_root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)

    keep16 = np.asarray(payload["keep16"], dtype=np.uint8)
    return {
        "edit_id": spec.edit_id,
        "edit_type": spec.edit_type,
        "shard": spec.shard,
        "obj_id": spec.obj_id,
        "mask_path": str(path.relative_to(output_root)),
        "edit_res": int(np.asarray(payload["edit_res"])),
        "n_edit_voxels": int(len(payload["edit_grid"])),
        "keep_ratio_ss": float(keep16.mean()),
        "selected_part_ids": list(spec.selected_part_ids),
        "has_slat": "mask_keep_slat" in payload,
    }


def _iter_object_dirs(pipeline_root: Path, shards: Iterable[str] | None) -> Iterator[Path]:
    objects_root = pipeline_root / "objects"
    shard_names = [str(s).zfill(2) for s in shards] if shards else None
    shard_dirs = [objects_root / s for s in shard_names] if shard_names else sorted(objects_root.iterdir())
    for shard_dir in shard_dirs:
        if not shard_dir.is_dir():
            continue
        for obj_dir in sorted(shard_dir.iterdir()):
            parsed = obj_dir / "phase1" / "parsed.json"
            if parsed.is_file():
                yield obj_dir


def materialize_masks(
    *,
    pipeline_root: str | Path,
    output_root: str | Path,
    shards: Iterable[str] | None = None,
    edit_types: set[str] | None = None,
    max_edits: int | None = None,
    max_per_type: int | None = None,
    min_edit_voxels: int = 0,
    allowed_edit_ids: set[str] | None = None,
    mask_builder: MaskBuilder | None = None,
) -> list[dict]:
    pipeline_root = Path(pipeline_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if mask_builder is None:
        raise ValueError("mask_builder is required; use FakeBoxMaskBuilder for dry runs")

    rows: list[dict] = []
    counts_by_type: dict[str, int] = {}
    target_types = set(edit_types) if edit_types is not None else None
    for obj_dir in _iter_object_dirs(pipeline_root, shards):
        parsed_path = obj_dir / "phase1" / "parsed.json"
        specs = list(iter_specs_from_parsed_path(parsed_path))
        if edit_types is None or "addition" in edit_types:
            specs.extend(iter_addition_specs_from_object_dir(obj_dir))
        for spec in specs:
            if edit_types is not None and spec.edit_type not in edit_types:
                continue
            if allowed_edit_ids is not None and spec.edit_id not in allowed_edit_ids:
                continue
            if spec.edit_type == "global" or not spec.selected_part_ids:
                continue  # part-based edit region needs target parts
            if max_per_type is not None and counts_by_type.get(spec.edit_type, 0) >= max_per_type:
                continue
            payload = mask_builder.build(spec)
            if payload is None:
                continue
            if len(payload["edit_grid"]) < min_edit_voxels:
                continue
            rows.append(_write_sidecar(output_root, spec, payload))
            counts_by_type[spec.edit_type] = counts_by_type.get(spec.edit_type, 0) + 1
            if max_edits is not None and len(rows) >= max_edits:
                _write_manifest(output_root, rows)
                return rows
            if max_per_type is not None and target_types and all(
                    counts_by_type.get(t, 0) >= max_per_type for t in target_types):
                _write_manifest(output_root, rows)
                return rows

    _write_manifest(output_root, rows)
    return rows


def _write_manifest(output_root: Path, rows: list[dict]) -> None:
    manifest = output_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize v2 (TRELLIS.2) edit masks as sidecar npz, "
                    "mirroring the dataset export's part_struct_grids + mask_from_ss.")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mesh-root", default=None,
                        help="Source partverse mesh root (<mesh-root>/<shard>/<obj>.npz). Required unless --fake-builder.")
    parser.add_argument("--shards", nargs="*", default=None)
    parser.add_argument("--edit-types", nargs="*",
                        default=["deletion", "addition", "modification", "scale"])
    parser.add_argument("--edit-res", type=int, default=512, choices=[512, 1024],
                        help="512 -> 32³ slat / 16³ ss (prod config); 1024 -> 64³ slat.")
    parser.add_argument("--pad", type=int, default=S1_PAD)
    parser.add_argument("--thresh", type=float, default=S1_THRESH)
    parser.add_argument("--no-canonical", action="store_true",
                        help="Disable canonical-frame normalization (default on, matches config).")
    parser.add_argument("--max-edits", type=int, default=None)
    parser.add_argument("--max-per-type", type=int, default=None)
    parser.add_argument("--min-edit-voxels", type=int, default=0)
    parser.add_argument("--allowed-edit-ids-file", default=None)
    parser.add_argument("--fake-builder", action="store_true",
                        help="Synthetic masks for dependency-free dry-run plumbing checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fake_builder:
        builder: MaskBuilder = FakeBoxMaskBuilder(edit_res=args.edit_res)
    else:
        if not args.mesh_root:
            raise SystemExit("real mask materialization requires --mesh-root")
        builder = V2StructMaskBuilder(
            mesh_root=args.mesh_root,
            pipeline_root=args.pipeline_root,
            edit_res=args.edit_res,
            pad=args.pad,
            thresh=args.thresh,
            canonical=not args.no_canonical,
        )

    allowed_edit_ids = None
    if args.allowed_edit_ids_file:
        allowed_edit_ids = {
            line.strip()
            for line in Path(args.allowed_edit_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    rows = materialize_masks(
        pipeline_root=args.pipeline_root,
        output_root=args.output_root,
        shards=args.shards,
        edit_types=set(args.edit_types) if args.edit_types else None,
        max_edits=args.max_edits,
        max_per_type=args.max_per_type,
        min_edit_voxels=args.min_edit_voxels,
        allowed_edit_ids=allowed_edit_ids,
        mask_builder=builder,
    )
    from collections import Counter
    print(json.dumps({
        "count": len(rows),
        "by_type": dict(Counter(r["edit_type"] for r in rows)),
        "edit_res": args.edit_res,
        "output_root": str(Path(args.output_root)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
