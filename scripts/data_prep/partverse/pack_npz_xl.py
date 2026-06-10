#!/usr/bin/env python3
"""Pack PartVerse XL into Trellis2-ready mesh NPZs (no img_Enc / prerender).

PartVerse XL layout (``PARTVERSE_XL_ROOT``)::

    meshes/textured_part_glbs/<uuid>/{0,1,...}.glb
    captions/<uuid>/caption.json
    anno_info/anno_infos/<uuid>/{uuid}_face2label.json  (optional, for split_mesh stub)

Writes::

    inputs/mesh/<shard>/<uuid>.npz
        full.glb            — assembled from textured parts (Y-up, raw GLB bytes)
        part_N.glb          — raw part GLB bytes (copied)
        vd_scale, vd_offset — Blender-compatible VD normalization (computed at pack)
        part_captions.json  — ``{"0": [name, description], ...}`` for VLM / gen_edits

    inputs/images/<shard>/<uuid>.npz   (optional, ``--skip-images-npz`` to omit)
        split_mesh.json     — minimal stub for legacy pre-flight / build_semantic_list

``full.glb`` is built by concatenating all ``textured_part_glbs/<uuid>/*.glb`` parts
when ``normalized_glbs`` is unavailable.  ``vd_scale`` / ``vd_offset`` are derived
from the assembled mesh using the same convention as Blender ``normalize_scene``::

    blender = (x, -z_glb, y_glb)
    vd = (blender + vd_offset) * vd_scale   # Z-up, roughly [-1, 1]^3

Trellis2 runtime does **not** read prerender views from the images NPZ; the stub
exists only for ``check_inputs`` when ``data.require_images_npz: true``.

Usage::

    # Smoke pack (5 objects, shard 00)
    python scripts/data_prep/partverse/pack_npz_xl.py \\
        --data-root /mnt/zsn/data/partversexl \\
        --shard 00 --num-shards 10 --limit 5 --workers 4

    # Full shard 00
    python scripts/data_prep/partverse/pack_npz_xl.py \\
        --data-root /mnt/zsn/data/partversexl \\
        --shard 00 --num-shards 10 --force --workers 8
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.data_prep.prerender_common import select_shard

_LOG = logging.getLogger("pack_npz_xl")
_pack_ctx: dict = {}


def _is_int_stem(p: Path) -> bool:
    try:
        int(p.stem)
        return True
    except ValueError:
        return False


def _load_trimesh_parts(part_glb_root: Path) -> list:
    import trimesh

    part_files = sorted(
        (p for p in part_glb_root.glob("*.glb") if _is_int_stem(p)),
        key=lambda p: int(p.stem),
    )
    if not part_files:
        return []
    meshes = []
    for part_path in part_files:
        scene = trimesh.load(str(part_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            for geom in scene.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
        elif isinstance(scene, trimesh.Trimesh):
            meshes.append(scene)
    return meshes


def compute_vd_params_yup_vertices(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    """VD scale/offset for Y-up GLB vertices (Blender normalize_scene convention)."""
    sv = np.asarray(vertices, dtype=np.float64)
    blender = np.empty_like(sv)
    blender[:, 0] = sv[:, 0]
    blender[:, 1] = -sv[:, 2]
    blender[:, 2] = sv[:, 1]

    mn, mx = blender.min(axis=0), blender.max(axis=0)
    extent = float((mx - mn).max())
    scale = 2.0 / extent if extent > 1e-12 else 1.0

    scaled = blender * scale
    mn2, mx2 = scaled.min(axis=0), scaled.max(axis=0)
    offset_after_scale = -(mn2 + mx2) / 2.0
    vd_offset = offset_after_scale / scale
    return scale, vd_offset


def assemble_full_glb_bytes(part_glb_root: Path) -> bytes | None:
    """Concatenate textured part GLBs → single full.glb bytes."""
    import trimesh

    meshes = _load_trimesh_parts(part_glb_root)
    if not meshes:
        return None
    full = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    buf = io.BytesIO()
    full.export(buf, file_type="glb")
    return buf.getvalue()


def xl_caption_to_part_captions(caption_path: Path) -> dict[str, list[str]]:
    """Convert XL ``caption.json`` → mesh NPZ ``part_captions.json`` layout."""
    with open(caption_path, encoding="utf-8") as f:
        data = json.load(f)
    parts = data.get("parts") or {}
    out: dict[str, list[str]] = {}
    for pid, info in parts.items():
        if not isinstance(info, dict):
            continue
        name = str(info.get("name") or "").strip()
        desc = str(info.get("description") or "").strip()
        if name and desc:
            out[str(pid)] = [name, desc]
        elif name:
            out[str(pid)] = [name]
        elif desc:
            out[str(pid)] = [desc]
    return out


def build_split_mesh_stub(part_captions: dict[str, list[str]]) -> dict:
    """Minimal split_mesh.json for legacy check_inputs / semantic fallback."""
    pids = sorted(int(k) for k in part_captions)
    pid_to_name = []
    clusters = {}
    for pid in pids:
        caps = part_captions.get(str(pid), [])
        label = caps[0] if caps else f"part_{pid}"
        pid_to_name.append(f"{label}_{pid}")
        clusters[f"part_{pid}"] = {"part_ids": [pid], "cluster_size": 0}
    return {"part_id_to_name": pid_to_name, "valid_clusters": clusters}


def _pack_mesh_xl(
    obj_id: str,
    *,
    textured_root: Path,
    caption_path: Path,
) -> dict | str:
    """Pack mesh NPZ payload. Returns dict on success, error reason string on failure."""
    part_glb_root = textured_root / obj_id
    if not part_glb_root.is_dir():
        return "no textured_part_glbs dir"

    part_files = sorted(
        (p for p in part_glb_root.glob("*.glb") if _is_int_stem(p)),
        key=lambda p: int(p.stem),
    )
    if not part_files:
        return "no part GLBs"

    if not caption_path.is_file():
        return "no caption.json"

    part_captions = xl_caption_to_part_captions(caption_path)
    if not part_captions:
        return "empty part captions"

    meshes = _load_trimesh_parts(part_glb_root)
    if not meshes:
        return "failed to load part meshes"

    import trimesh

    full = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    scale, vd_offset = compute_vd_params_yup_vertices(full.vertices)

    full_buf = io.BytesIO()
    full.export(full_buf, file_type="glb")

    result: dict[str, np.ndarray] = {
        "full.glb": np.frombuffer(full_buf.getvalue(), dtype=np.uint8),
        "vd_scale": np.array([scale], dtype=np.float64),
        "vd_offset": np.array(vd_offset, dtype=np.float64),
        "part_captions.json": np.frombuffer(
            json.dumps(part_captions, ensure_ascii=False).encode("utf-8"),
            dtype=np.uint8,
        ),
    }

    n_parts = 0
    for part_path in part_files:
        pid = int(part_path.stem)
        try:
            result[f"part_{pid}.glb"] = np.frombuffer(
                part_path.read_bytes(), dtype=np.uint8
            )
            n_parts += 1
        except OSError:
            continue

    if n_parts == 0:
        return "no part bytes copied"

    return {"mesh": result, "part_captions": part_captions, "n_parts": n_parts}


def pack_one_xl(
    obj_id: str,
    *,
    textured_root: Path,
    captions_root: Path,
    mesh_out: Path,
    images_out: Path | None,
    write_images_npz: bool,
) -> dict:
    caption_path = captions_root / obj_id / "caption.json"
    packed = _pack_mesh_xl(
        obj_id,
        textured_root=textured_root,
        caption_path=caption_path,
    )
    if isinstance(packed, str):
        return {"status": "skip", "reason": packed}

    mesh_data = packed["mesh"]
    part_captions = packed["part_captions"]
    n_parts = packed["n_parts"]

    np.savez_compressed(str(mesh_out / f"{obj_id}.npz"), **mesh_data)

    if write_images_npz and images_out is not None:
        stub = {
            "split_mesh.json": np.frombuffer(
                json.dumps(build_split_mesh_stub(part_captions)).encode("utf-8"),
                dtype=np.uint8,
            ),
        }
        np.savez_compressed(str(images_out / f"{obj_id}.npz"), **stub)

    return {"status": "ok", "parts": n_parts}


def _pack_worker(oid: str) -> dict:
    ctx = _pack_ctx
    return pack_one_xl(
        oid,
        textured_root=ctx["textured_root"],
        captions_root=ctx["captions_root"],
        mesh_out=ctx["mesh_out"],
        images_out=ctx.get("images_out"),
        write_images_npz=ctx.get("write_images_npz", True),
    )


def list_xl_object_ids(
    textured_root: Path,
    captions_root: Path,
) -> list[str]:
    """Objects with both textured parts and per-object caption.json."""
    part_ids = {p.name for p in textured_root.iterdir() if p.is_dir()}
    cap_ids = {
        p.name
        for p in captions_root.iterdir()
        if p.is_dir() and (p / "caption.json").is_file()
    }
    return sorted(part_ids & cap_ids)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pack_npz_xl")

    parser = argparse.ArgumentParser(
        description="Pack PartVerse XL (mesh-only, assemble full.glb from parts)")
    parser.add_argument(
        "--data-root", type=str,
        default=os.environ.get("PARTVERSE_XL_ROOT", "/mnt/zsn/data/partversexl"),
        help="PartVerse XL root (meshes/, captions/, anno_info/)",
    )

    sel = parser.add_argument_group("object selection")
    sel.add_argument("--obj-ids", nargs="*", default=None,
                     help="Explicit object IDs (overrides --shard)")
    sel.add_argument("--shard", type=str, default=None,
                     help="Shard to pack, e.g. '00'. Requires --num-shards.")
    sel.add_argument("--num-shards", type=int, default=10)
    sel.add_argument("--limit", type=int, default=0,
                     help="Cap to first N objects (0 = all)")

    parser.add_argument("--force", action="store_true",
                        help="Re-pack even if output NPZs already exist")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--skip-images-npz", action="store_true",
        help="Only write mesh NPZ (set data.require_images_npz: false in pipeline config)",
    )
    parser.add_argument(
        "--mesh-out-dir", type=str, default=None,
        help="Override mesh output dir (default: <data-root>/inputs/mesh/<shard>)",
    )
    parser.add_argument(
        "--images-out-dir", type=str, default=None,
        help="Override images stub dir (default: <data-root>/inputs/images/<shard>)",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root)
    textured_root = data_root / "meshes" / "textured_part_glbs"
    captions_root = data_root / "captions"

    if not textured_root.is_dir():
        raise SystemExit(f"missing textured_part_glbs: {textured_root}")
    if not captions_root.is_dir():
        raise SystemExit(f"missing captions: {captions_root}")

    if args.obj_ids:
        obj_ids = list(args.obj_ids)
        shard = "00"
        logger.info("Explicit --obj-ids: %d objects → shard %s", len(obj_ids), shard)
    else:
        all_ids = list_xl_object_ids(textured_root, captions_root)
        if args.shard is not None:
            obj_ids = select_shard(all_ids, args.shard, args.num_shards)
            shard = args.shard
            logger.info(
                "Shard %s/%d: %d/%d objects",
                shard, args.num_shards, len(obj_ids), len(all_ids),
            )
        else:
            obj_ids = all_ids
            shard = "00"
            logger.info("All objects: %d → shard %s", len(obj_ids), shard)

    if args.limit > 0:
        obj_ids = obj_ids[: args.limit]
        logger.info("--limit: capped to %d objects", len(obj_ids))

    mesh_out = (
        Path(args.mesh_out_dir)
        if args.mesh_out_dir
        else data_root / "inputs" / "mesh" / shard
    )
    images_out = None
    write_images = not args.skip_images_npz
    if write_images:
        images_out = (
            Path(args.images_out_dir)
            if args.images_out_dir
            else data_root / "inputs" / "images" / shard
        )

    mesh_out.mkdir(parents=True, exist_ok=True)
    if images_out is not None:
        images_out.mkdir(parents=True, exist_ok=True)

    logger.info("Output mesh : %s", mesh_out)
    if images_out is not None:
        logger.info("Output images (stub): %s", images_out)
    else:
        logger.info("Images NPZ: skipped (--skip-images-npz)")

    pending = []
    pre_skip = 0
    for obj_id in obj_ids:
        out_m = mesh_out / f"{obj_id}.npz"
        out_r = images_out / f"{obj_id}.npz" if images_out else None
        if (
            out_m.exists()
            and (out_r is None or out_r.exists())
            and not args.force
        ):
            pre_skip += 1
            continue
        pending.append(obj_id)

    logger.info(
        "Pack: %d pending, %d skipped / %d total (workers=%d)",
        len(pending), pre_skip, len(obj_ids), args.workers,
    )
    if not pending:
        logger.info("Nothing to pack.")
        return

    _pack_ctx.update({
        "textured_root": textured_root,
        "captions_root": captions_root,
        "mesh_out": mesh_out,
        "images_out": images_out,
        "write_images_npz": write_images,
    })

    ok = fail = 0
    if args.workers <= 1:
        for i, obj_id in enumerate(pending):
            result = _pack_worker(obj_id)
            if result["status"] == "ok":
                ok += 1
                logger.info(
                    "[%d/%d] %s: %d parts",
                    i + 1, len(pending), obj_id, result["parts"],
                )
            else:
                fail += 1
                logger.warning(
                    "[%d/%d] %s: SKIP — %s",
                    i + 1, len(pending), obj_id, result["reason"],
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_pack_worker, oid): oid for oid in pending}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                oid = futures[future]
                try:
                    result = future.result()
                    if result["status"] == "ok":
                        ok += 1
                        if done_count % 50 == 0 or done_count == len(pending):
                            logger.info("[%d/%d] packed %s", done_count, len(pending), oid)
                    else:
                        fail += 1
                        logger.warning(
                            "[%d/%d] %s: SKIP — %s",
                            done_count, len(pending), oid, result["reason"],
                        )
                except Exception as e:
                    fail += 1
                    logger.error("[%d/%d] %s: ERROR — %s", done_count, len(pending), oid, e)

    logger.info(
        "\nDone: %d packed, %d skipped, %d failed / %d total",
        ok, pre_skip, fail, len(obj_ids),
    )


if __name__ == "__main__":
    main()
