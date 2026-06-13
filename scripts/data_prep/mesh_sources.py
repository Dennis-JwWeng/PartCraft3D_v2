#!/usr/bin/env python3
"""Per-data-source mesh readers for pipeline_v3.

The pipeline core reads every mesh through one ``open_mesh(ref)`` chokepoint
that returns an object exposing the small ``numpy`` ``NpzFile`` subset the
readers actually touch (``.files``, ``in``/``__contains__``, ``__getitem__`` →
uint8 / byte arrays, ``.close()``).  Dispatch is by ref *type*, so the deep
readers stay completely source-agnostic:

  * **Packed NPZ** (legacy PartVerse + the XL pack) — ``ref`` is a ``*.npz``
    file → delegates to :func:`numpy.load` unchanged (byte-identical behaviour;
    existing trees are untouched).
  * **Raw PartVerse XL** — ``ref`` is the per-object
    ``meshes/textured_part_glbs/<uuid>`` directory → assembles ``full.glb`` /
    serves ``part_N.glb`` / ``part_captions.json`` on the fly from the raw
    layout.  No NPZ on disk: no pack pass, no ~2x mesh duplication.  ``full.glb``
    is built with the SAME assembler the pack uses, so the bytes a reader sees
    are identical to a packed NPZ.

``vd_scale`` / ``vd_offset`` are intentionally absent from the raw source: they
are read only by the legacy GLB-deletion path (``mesh_deletion``), which the XL
pipeline never takes — ``encode_after_512`` re-normalizes from the original
mesh's bounds, and ``_merge_surviving_parts_from_npz`` explicitly does NOT apply
them.  Their absence flips ``has_vd_transform`` to False, preserving behaviour.

Locating + enumerating per source is the job of the :class:`MeshSource`
subclasses (used by ``paths.py`` / ``run_trellis2``); *opening* a located ref is
:func:`open_mesh` (used by the deep readers).  This keeps the pipeline control
flow identical across sources.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import numpy as np

# The raw-XL assembler/caption converter live with the pack tool; reuse them so
# the on-the-fly ``full.glb`` / ``part_captions.json`` bytes are identical to a
# packed NPZ (no second source of truth for the assembly recipe).
from scripts.data_prep.partverse.pack_npz_xl import (
    assemble_full_glb_bytes,
    xl_caption_to_part_captions,
)
from scripts.data_prep.prerender_common import select_shard

# ─────────────────────────── raw-XL in-memory NpzFile shim ───────────────────


def _is_int_stem(p: Path) -> bool:
    try:
        int(p.stem)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=16)
def _assemble_full_glb_uint8(part_dir_str: str) -> np.ndarray:
    """Assemble ``full.glb`` bytes for a raw-XL object dir (cached per process).

    ``full.glb`` is needed by several reader stages per object; the LRU keeps the
    trimesh concat+export to once per object per worker (RAM only, no disk)."""
    raw = assemble_full_glb_bytes(Path(part_dir_str))
    if raw is None:
        raise RuntimeError(f"no part GLBs to assemble full.glb in {part_dir_str}")
    return np.frombuffer(raw, dtype=np.uint8)


def _caption_root_for(part_dir: Path) -> Path:
    """Convention: ``<root>/meshes/textured_part_glbs/<uuid>`` → ``<root>/captions``."""
    # parents[0]=textured_part_glbs, [1]=meshes, [2]=<root>
    return part_dir.parents[2] / "captions"


class RawXLMesh:
    """Lazy ``NpzFile``-compatible view over a raw PartVerse XL object directory.

    Exposes only the subset the pipeline readers use: ``.files``, ``in`` /
    ``__contains__``, ``__getitem__`` (returns ``uint8`` arrays like a real NPZ),
    and ``.close()`` (no-op).  All values are produced on demand.
    """

    def __init__(self, part_dir: Path, captions_root: Path | None = None):
        self._dir = Path(part_dir)
        self._uuid = self._dir.name
        croot = captions_root or _caption_root_for(self._dir)
        self._caption_path = Path(croot) / self._uuid / "caption.json"
        # part_<n>.glb keys, sorted by int stem.
        self._part_ids = sorted(
            int(p.stem)
            for p in self._dir.glob("*.glb")
            if _is_int_stem(p)
        )
        self._files = (
            ["full.glb", "part_captions.json"]
            + [f"part_{pid}.glb" for pid in self._part_ids]
        )
        self._cache: dict[str, np.ndarray] = {}

    # -- NpzFile-compatible surface -------------------------------------------
    @property
    def files(self) -> list[str]:
        return list(self._files)

    def __contains__(self, key: str) -> bool:
        return key in self._files

    def __iter__(self):
        return iter(self._files)

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._cache:
            return self._cache[key]
        if key == "full.glb":
            val = _assemble_full_glb_uint8(str(self._dir))
        elif key == "part_captions.json":
            caps = xl_caption_to_part_captions(self._caption_path)
            val = np.frombuffer(
                json.dumps(caps, ensure_ascii=False).encode("utf-8"),
                dtype=np.uint8,
            )
        elif key.startswith("part_") and key.endswith(".glb"):
            pid = int(key[len("part_"):-len(".glb")])
            val = np.frombuffer(
                (self._dir / f"{pid}.glb").read_bytes(), dtype=np.uint8
            )
        else:
            raise KeyError(f"{key!r} not in raw-XL mesh {self._dir}")
        self._cache[key] = val
        return val

    def get(self, key: str, default=None):
        return self[key] if key in self._files else default

    def close(self) -> None:  # NpzFile API parity (readers may call it)
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ─────────────────────────── open / availability ────────────────────────────


def open_mesh(ref: "Path | str", *, allow_pickle: bool = True):
    """Open a mesh ref → ``NpzFile`` (``*.npz``) or :class:`RawXLMesh` (raw dir).

    The single chokepoint every deep reader uses in place of ``np.load``.
    Dispatch is purely by ref type, so callers never branch on data source.
    """
    ref = Path(ref)
    if ref.suffix == ".npz":
        return np.load(ref, allow_pickle=allow_pickle)
    if ref.is_dir():
        return RawXLMesh(ref)
    raise FileNotFoundError(f"mesh ref is neither a .npz file nor a directory: {ref}")


def mesh_available(ref: "Path | str | None") -> bool:
    """True iff ``ref`` is a usable mesh (packed NPZ file or raw-XL object dir)."""
    if ref is None:
        return False
    ref = Path(ref)
    if ref.suffix == ".npz":
        return ref.is_file()
    if ref.is_dir():
        return any(_is_int_stem(p) for p in ref.glob("*.glb"))
    return False


# ─────────────────────────── per-source locating / enumeration ───────────────


class MeshSource(ABC):
    """How a data source *locates* and *enumerates* objects.

    Opening a located ref is :func:`open_mesh` (type-dispatched); this class only
    decides paths + the object list, so the pipeline core stays identical.
    """

    require_images_npz: bool = True

    @abstractmethod
    def mesh_ref(self, shard: str, obj_id: str) -> Path: ...

    @abstractmethod
    def image_ref(self, shard: str, obj_id: str) -> Path: ...

    @abstractmethod
    def list_object_ids(self, shard: str, num_shards: int) -> list[str]: ...


class PackedNpzSource(MeshSource):
    """Legacy/packed source: ``<root>/<shard>/<obj_id>.npz`` (mesh + image)."""

    def __init__(self, mesh_root: Path, images_root: Path,
                 require_images_npz: bool = True):
        self.mesh_root = Path(mesh_root)
        self.images_root = Path(images_root)
        self.require_images_npz = require_images_npz

    def mesh_ref(self, shard: str, obj_id: str) -> Path:
        return self.mesh_root / shard / f"{obj_id}.npz"

    def image_ref(self, shard: str, obj_id: str) -> Path:
        return self.images_root / shard / f"{obj_id}.npz"

    def list_object_ids(self, shard: str, num_shards: int) -> list[str]:
        d = self.mesh_root / shard
        return sorted(p.stem for p in d.glob("*.npz")) if d.is_dir() else []


class RawXLSource(MeshSource):
    """Raw PartVerse XL: read straight from ``meshes/textured_part_glbs`` +
    ``captions``.  No pack, no per-object NPZ; sharding is a pure runtime
    partition of the (textured ∩ captioned) object id list (range-based, same
    ``select_shard`` the pack used) — no data ever moves between shard dirs."""

    require_images_npz = False

    def __init__(self, xl_root: Path):
        self.xl_root = Path(xl_root)
        self.textured_root = self.xl_root / "meshes" / "textured_part_glbs"
        self.captions_root = self.xl_root / "captions"

    def mesh_ref(self, shard: str, obj_id: str) -> Path:
        # Flat layout — textured_part_glbs is not sharded on disk.
        return self.textured_root / obj_id

    def image_ref(self, shard: str, obj_id: str) -> Path:
        # No images NPZ for the mesh-only XL pack; return a (nonexistent) path so
        # ObjectContext keeps a Path. Readers guard on existence / require_images_npz.
        return self.xl_root / "inputs" / "images" / shard / f"{obj_id}.npz"

    @lru_cache(maxsize=1)
    def _all_ids(self) -> tuple[str, ...]:
        if not self.textured_root.is_dir() or not self.captions_root.is_dir():
            return ()
        part_ids = {p.name for p in self.textured_root.iterdir() if p.is_dir()}
        cap_ids = {
            p.name
            for p in self.captions_root.iterdir()
            if p.is_dir() and (p / "caption.json").is_file()
        }
        return tuple(sorted(part_ids & cap_ids))

    def list_object_ids(self, shard: str, num_shards: int) -> list[str]:
        return select_shard(list(self._all_ids()), shard, num_shards)


def get_mesh_source(cfg: dict) -> MeshSource:
    """Build the :class:`MeshSource` for a pipeline config's ``data`` block.

    ``data.source``:
      * ``"packed_npz"`` (default) — ``mesh_root`` / ``images_root`` NPZ trees.
      * ``"partversexl_raw"`` — raw XL under ``data.xl_root`` (or ``mesh_root``'s
        grandparent for back-compat), no pack.
    """
    data = cfg.get("data") or {}
    source = str(data.get("source", "packed_npz")).strip().lower()
    if source in ("partversexl_raw", "xl_raw", "raw_xl"):
        xl_root = data.get("xl_root")
        if not xl_root:
            raise ValueError(
                "data.source=partversexl_raw requires data.xl_root "
                "(e.g. /mnt/zsn/data/partversexl)"
            )
        return RawXLSource(Path(xl_root))
    return PackedNpzSource(
        mesh_root=Path(data.get("mesh_root", "data/partverse/mesh")),
        images_root=Path(data.get("images_root", "data/partverse/images")),
        require_images_npz=bool(data.get("require_images_npz", True)),
    )
