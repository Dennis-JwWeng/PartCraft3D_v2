"""Materialize PartCraft edit masks as sidecar training assets.

This module keeps the lightweight bookkeeping and tensor transforms separate
from the heavy TRELLIS/Open3D runtime. Unit tests can use ``FakeBoxMaskBuilder``;
production callers can pass a builder that reproduces ``TrellisRefiner.build_part_mask``.

:class:`PartCraftRuntimeMaskBuilder.build_mask_detail` additionally materializes
per-SLAT preserve flags; pass ``include_slat_indices=True`` to :func:`materialize_masks`
or use ``scripts/tools/dump_edit_part_mask.py`` for one-off dumps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

from partcraft.pipeline_v3.paths import PipelineRoot
from partcraft.pipeline_v3.specs import EditSpec, iter_all_specs


class MaskBuilder(Protocol):
    def build_mask(self, spec: EditSpec) -> np.ndarray:
        """Return a bool edit mask with shape ``[64, 64, 64]``."""


class FakeBoxMaskBuilder:
    """Deterministic lightweight builder for tests and dry-run plumbing checks."""

    def build_mask(self, spec: EditSpec) -> np.ndarray:
        mask = np.zeros((64, 64, 64), dtype=bool)
        seed = sum(spec.selected_part_ids) if spec.selected_part_ids else spec.edit_idx
        start = 4 + (seed % 48)
        mask[start:start + 4, start:start + 4, start:start + 4] = True
        return mask




class _PartVerseMeshAdapter:
    """Compatibility adapter for TrellisRefiner's mesh accessor names."""

    def __init__(self, record):
        self._record = record

    def __getattr__(self, name):
        return getattr(self._record, name)

    @staticmethod
    def _load_trimesh_from_bytes(raw: bytes, key: str):
        import io
        import trimesh

        file_type = key.rsplit(".", 1)[-1]
        mesh = trimesh.load(io.BytesIO(raw), file_type=file_type, force="scene")
        if hasattr(mesh, "geometry"):
            parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not parts:
                raise ValueError(f"No triangle mesh geometry found in {key}")
            return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        return mesh

    def get_full_mesh(self, colored: bool = False):
        key = "full.glb"
        return self._load_trimesh_from_bytes(self._record.get_mesh_bytes(key), key)

    def get_part_mesh(self, part_id: int, colored: bool = False):
        key = f"part_{int(part_id)}.glb"
        return self._load_trimesh_from_bytes(self._record.get_mesh_bytes(key), key)


def _ensure_refiner_record_api(record):
    if hasattr(record, "get_full_mesh") and hasattr(record, "get_part_mesh"):
        return record
    return _PartVerseMeshAdapter(record)

class PartCraftRuntimeMaskBuilder:
    """Adapter around the real PartCraft ``TrellisRefiner.build_part_mask`` path.

    The caller owns construction of the heavy ``dataset`` and ``refiner``
    objects so this module remains importable in lightweight test contexts.
    ``dataset`` must expose ``load_object(shard, obj_id)`` and ``refiner`` must
    expose ``encode_object`` and ``build_part_mask``.
    """

    def __init__(
        self,
        *,
        dataset,
        refiner,
        large_part_threshold: float = 0.35,
        promote_scale_to_global: bool = False,
        scale_large_part_threshold: float | None = None,
    ):
        self.dataset = dataset
        self.refiner = refiner
        self.large_part_threshold = large_part_threshold
        self.promote_scale_to_global = promote_scale_to_global
        self.scale_large_part_threshold = scale_large_part_threshold
        self._ori_slat_cache = {}

    @staticmethod
    def _edit_part_ids(spec: EditSpec) -> list[int]:
        et_cap = spec.edit_type.capitalize()
        if et_cap == "Global":
            return []
        return list(spec.selected_part_ids)

    def _ori_slat(self, obj_id: str):
        if obj_id not in self._ori_slat_cache:
            slat = self.refiner.encode_object(None, obj_id)
            to_device = getattr(slat, "to", None)
            if callable(to_device):
                slat = to_device(self.refiner.device)
            else:
                slat.coords = slat.coords.to(self.refiner.device)
                slat.feats = slat.feats.to(self.refiner.device)
            self._ori_slat_cache[obj_id] = slat
        return self._ori_slat_cache[obj_id]

    def build_mask(self, spec: EditSpec) -> np.ndarray:
        obj_record = self.dataset.load_object(spec.shard, spec.obj_id)
        try:
            obj_record_for_refiner = _ensure_refiner_record_api(obj_record)
            et_cap = spec.edit_type.capitalize()
            mask, _effective_type = self.refiner.build_part_mask(
                spec.obj_id,
                obj_record_for_refiner,
                self._edit_part_ids(spec),
                self._ori_slat(spec.obj_id),
                et_cap,
                large_part_threshold=self.large_part_threshold,
                promote_scale_to_global=self.promote_scale_to_global,
                scale_large_part_threshold=self.scale_large_part_threshold,
            )
            return mask.detach().cpu().numpy().astype(bool)
        finally:
            close = getattr(obj_record, "close", None)
            if callable(close):
                close()

    def build_mask_detail(self, spec: EditSpec) -> dict[str, object]:
        """Same path as :meth:`build_mask`, plus SLAT-aligned index arrays.

        The returned ``mask_edit_64`` matches the boolean tensor passed into
        ``interweave_Trellis_TI`` — **True** means the voxel is in the *editable*
        region.  Per-SLAT-point flags are::

            slat_in_edit_region = mask_edit_64[slat_xyz[:,0], slat_xyz[:,1], slat_xyz[:,2]]
            slat_preserve = ~slat_in_edit_region
        """
        obj_record = self.dataset.load_object(spec.shard, spec.obj_id)
        try:
            obj_record_for_refiner = _ensure_refiner_record_api(obj_record)
            et_cap = spec.edit_type.capitalize()
            slat = self._ori_slat(spec.obj_id)
            mask, effective_type = self.refiner.build_part_mask(
                spec.obj_id,
                obj_record_for_refiner,
                self._edit_part_ids(spec),
                slat,
                et_cap,
                large_part_threshold=self.large_part_threshold,
                promote_scale_to_global=self.promote_scale_to_global,
                scale_large_part_threshold=self.scale_large_part_threshold,
            )
            m = mask.detach().cpu().numpy().astype(bool)
            xyz = slat.coords[:, 1:].detach().cpu().numpy().astype(np.int16)
            edit_at = m[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
            preserve_at = ~edit_at
            return {
                "mask_edit_64": m.astype(np.uint8),
                "effective_edit_type": str(effective_type),
                "slat_coords": xyz,
                "slat_in_edit_region": edit_at.astype(np.uint8),
                "slat_preserve": preserve_at.astype(np.uint8),
            }
        finally:
            close = getattr(obj_record, "close", None)
            if callable(close):
                close()


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


def dilate_edit_mask_64(mask_edit_64: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask_edit_64).astype(bool)
    if mask.shape != (64, 64, 64):
        raise ValueError(f"mask_edit_64 must have shape (64, 64, 64), got {mask.shape}")
    if radius <= 0:
        return mask.copy()

    try:
        from scipy.ndimage import binary_dilation

        structure = np.ones((3, 3, 3), dtype=bool)
        return binary_dilation(mask, structure=structure, iterations=radius).astype(bool)
    except Exception:
        # Small dependency-free fallback. Radius is expected to be 1 for smoke/tests.
        out = mask.copy()
        for _ in range(radius):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            next_out = np.zeros_like(out, dtype=bool)
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        next_out |= padded[
                            1 + dz:1 + dz + 64,
                            1 + dy:1 + dy + 64,
                            1 + dx:1 + dx + 64,
                        ]
            out = next_out
        return out


def mask_keep_from_edit(mask_edit_64: np.ndarray, dilation_radius: int = 1) -> np.ndarray:
    dilated = dilate_edit_mask_64(mask_edit_64, radius=dilation_radius)
    return (~dilated).astype(np.uint8)


def downsample_keep_mask_to_ss(mask_keep_64: np.ndarray) -> np.ndarray:
    keep = np.asarray(mask_keep_64).astype(np.uint8)
    if keep.shape != (64, 64, 64):
        raise ValueError(f"mask_keep_64 must have shape (64, 64, 64), got {keep.shape}")
    blocks = keep.reshape(16, 4, 16, 4, 16, 4)
    return blocks.min(axis=(1, 3, 5)).astype(np.uint8)


def _sidecar_path(output_root: Path, spec: EditSpec) -> Path:
    return output_root / spec.edit_type / spec.shard / spec.obj_id / f"{spec.edit_id}.npz"


def _write_sidecar(
    output_root: Path,
    spec: EditSpec,
    mask_edit_64: np.ndarray,
    dilation_radius: int,
    *,
    slat_coords: np.ndarray | None = None,
    slat_in_edit_region: np.ndarray | None = None,
    slat_preserve: np.ndarray | None = None,
) -> dict:
    mask_edit_64 = np.asarray(mask_edit_64).astype(np.uint8)
    if mask_edit_64.shape != (64, 64, 64):
        raise ValueError(f"builder returned bad mask shape for {spec.edit_id}: {mask_edit_64.shape}")
    mask_keep_64 = mask_keep_from_edit(mask_edit_64, dilation_radius=dilation_radius)
    mask_keep_ss = downsample_keep_mask_to_ss(mask_keep_64)

    path = _sidecar_path(output_root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "mask_edit_64": mask_edit_64,
        "mask_keep_64": mask_keep_64,
        "mask_keep_ss": mask_keep_ss,
        "selected_part_ids": np.asarray(spec.selected_part_ids, dtype=np.int32),
        "edit_type": np.asarray(spec.edit_type),
    }
    if slat_coords is not None:
        payload["slat_coords"] = np.asarray(slat_coords, dtype=np.int16)
        payload["slat_in_edit_region"] = np.asarray(slat_in_edit_region, dtype=np.uint8)
        payload["slat_preserve"] = np.asarray(slat_preserve, dtype=np.uint8)

    np.savez_compressed(path, **payload)

    row = {
        "edit_id": spec.edit_id,
        "edit_type": spec.edit_type,
        "shard": spec.shard,
        "obj_id": spec.obj_id,
        "mask_path": str(path.relative_to(output_root)),
        "keep_ratio_ss": float(mask_keep_ss.mean()),
        "selected_part_ids": list(spec.selected_part_ids),
        "has_slat_indices": slat_coords is not None,
    }
    return row


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
    dilation_radius: int = 1,
    include_slat_indices: bool = False,
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
            if spec.edit_type == "global":
                continue
            if max_per_type is not None and counts_by_type.get(spec.edit_type, 0) >= max_per_type:
                continue
            if include_slat_indices:
                if not isinstance(mask_builder, PartCraftRuntimeMaskBuilder):
                    raise ValueError(
                        "include_slat_indices=True requires PartCraftRuntimeMaskBuilder"
                    )
                det = mask_builder.build_mask_detail(spec)
                effective = det.get("effective_edit_type")
                mask_u8 = np.asarray(det["mask_edit_64"], dtype=np.uint8)
                sc = np.asarray(det["slat_coords"], dtype=np.int16)
                sie = np.asarray(det["slat_in_edit_region"], dtype=np.uint8)
                sp = np.asarray(det["slat_preserve"], dtype=np.uint8)
                mask = mask_u8
            else:
                effective = None
                mask = mask_builder.build_mask(spec)
                sc = sie = sp = None
            if int(np.asarray(mask).astype(bool).sum()) < min_edit_voxels:
                continue
            row = _write_sidecar(
                output_root,
                spec,
                mask,
                dilation_radius,
                slat_coords=sc,
                slat_in_edit_region=sie,
                slat_preserve=sp,
            )
            if effective is not None:
                row["effective_edit_type"] = str(effective)
            rows.append(row)
            counts_by_type[spec.edit_type] = counts_by_type.get(spec.edit_type, 0) + 1
            if max_edits is not None and len(rows) >= max_edits:
                _write_manifest(output_root, rows)
                return rows
            if max_per_type is not None and target_types and all(counts_by_type.get(t, 0) >= max_per_type for t in target_types):
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
    parser = argparse.ArgumentParser(description="Materialize PartCraft true edit masks as sidecar npz files.")
    parser.add_argument("--pipeline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--images-root", default=None)
    parser.add_argument("--mesh-root", default=None)
    parser.add_argument("--slat-root", default=None)
    parser.add_argument("--ckpt-root", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shards", nargs="*", default=None)
    parser.add_argument("--edit-types", nargs="*", default=["deletion", "addition", "modification", "scale"])
    parser.add_argument("--max-edits", type=int, default=None)
    parser.add_argument("--max-per-type", type=int, default=None)
    parser.add_argument("--min-edit-voxels", type=int, default=0)
    parser.add_argument("--allowed-edit-ids-file", default=None)
    parser.add_argument("--dilation-radius", type=int, default=1)
    parser.add_argument(
        "--include-slat-indices",
        action="store_true",
        help="Also write slat_coords / slat_preserve arrays (requires real builder).",
    )
    parser.add_argument("--fake-builder", action="store_true", help="Use deterministic fake masks for dry-run validation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fake_builder:
        builder = FakeBoxMaskBuilder()
    else:
        missing = [
            name for name, value in {
                "--images-root": args.images_root,
                "--mesh-root": args.mesh_root,
                "--slat-root": args.slat_root,
                "--ckpt-root": args.ckpt_root,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit(f"Real mask materialization requires {' '.join(missing)}")
        from partcraft.io.partverse_dataset import PartVerseDataset
        from partcraft.trellis.refiner import TrellisRefiner

        dataset = PartVerseDataset(args.images_root, args.mesh_root, args.shards, slat_dir=args.slat_root)
        refiner = TrellisRefiner(
            cache_dir=Path(args.output_root) / "_refiner_cache",
            device=args.device,
            ckpt_dir=args.ckpt_root,
            slat_dir=args.slat_root,
            image_edit_backend="local_diffusers",
        )
        builder = PartCraftRuntimeMaskBuilder(dataset=dataset, refiner=refiner)

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
        dilation_radius=args.dilation_radius,
        include_slat_indices=args.include_slat_indices,
    )
    print(json.dumps({"count": len(rows), "output_root": str(Path(args.output_root))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
