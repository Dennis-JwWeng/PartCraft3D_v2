from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy as sp
import trimesh
from tqdm import tqdm

from .protocol import case_result_dir, load_split, resolve_protocol_path, summarize_values, write_case_metrics


def _as_mesh(obj: Any) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("scene contains no meshes")
        return trimesh.util.concatenate(meshes)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if hasattr(obj, "to_mesh"):
        return obj.to_mesh()
    raise TypeError(f"unsupported mesh object: {type(obj)!r}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise FileNotFoundError(f"mesh not found: {path}")
    mesh = _as_mesh(trimesh.load(path, force=None))
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"mesh has no vertices/faces: {path}")
    return mesh


def _distance_field(source, target, normals_src=None, normals_tgt=None):
    target_kdtree = sp.spatial.cKDTree(target)
    distances, idx = target_kdtree.query(source, workers=-1)
    if normals_src is None or normals_tgt is None:
        return distances, np.full(source.shape[0], np.nan, dtype=np.float32)
    normals_src = normals_src / np.maximum(np.linalg.norm(normals_src, axis=-1, keepdims=True), 1e-8)
    normals_tgt = normals_tgt / np.maximum(np.linalg.norm(normals_tgt, axis=-1, keepdims=True), 1e-8)
    dots = np.abs((normals_tgt[idx] * normals_src).sum(axis=-1))
    return distances, dots


def compute_surface_metrics(
    mesh_pred: trimesh.Trimesh,
    mesh_gt: trimesh.Trimesh,
    *,
    eval_points: int = 100000,
    fscore_tau: float = 1e-2,
) -> dict[str, float]:
    points_gt, idx_gt = mesh_gt.sample(eval_points, return_index=True)
    normals_gt = mesh_gt.face_normals[idx_gt]
    points_pred, idx_pred = mesh_pred.sample(eval_points, return_index=True)
    normals_pred = mesh_pred.face_normals[idx_pred]

    points_gt = points_gt.astype(np.float32)
    points_pred = points_pred.astype(np.float32)
    dist_p2g, norm_p2g = _distance_field(points_pred, points_gt, normals_pred, normals_gt)
    dist_g2p, norm_g2p = _distance_field(points_gt, points_pred, normals_gt, normals_pred)

    chamfer_l1_1e3 = float(((np.mean(dist_p2g) + np.mean(dist_g2p)) / 2.0) * 1000.0)
    normal_consistency = float((np.nanmean(norm_p2g) + np.nanmean(norm_g2p)) / 2.0)
    precision = float((dist_p2g <= fscore_tau).astype(np.float32).mean() * 100.0)
    recall = float((dist_g2p <= fscore_tau).astype(np.float32).mean() * 100.0)
    fscore = float((2.0 * precision * recall) / max(precision + recall, 1e-8))
    return {
        "chamfer_l1_1e3": chamfer_l1_1e3,
        "normal_consistency": normal_consistency,
        "fscore": fscore,
        "precision_tau": precision,
        "recall_tau": recall,
    }


def compute_3d_metrics(
    *,
    protocol_root: str | Path,
    split: str,
    method: str,
    results_root: str | Path | None = None,
    eval_points: int = 100000,
    fscore_tau: float = 1e-2,
    seed: int = 0,
) -> dict:
    np.random.seed(seed)
    protocol_root = Path(protocol_root)
    results_root = Path(results_root) if results_root is not None else protocol_root / "results"
    rows = load_split(protocol_root, split)
    metric_values: dict[str, list[float]] = {
        "chamfer_l1_1e3": [],
        "normal_consistency": [],
        "fscore": [],
        "precision_tau": [],
        "recall_tau": [],
    }
    cases = []

    for row in tqdm(rows, desc="H3D_v1 3D metrics"):
        case_id = row["case_id"]
        pred_path = case_result_dir(results_root, method, split, case_id) / "pred.glb"
        target_path = resolve_protocol_path(protocol_root, row["target"]["after_glb"])
        vals = compute_surface_metrics(
            load_mesh(pred_path),
            load_mesh(target_path),
            eval_points=eval_points,
            fscore_tau=fscore_tau,
        )
        for metric, value in vals.items():
            metric_values[metric].append(value)
        write_case_metrics(
            case_result_dir(results_root, method, split, case_id) / "metrics.json",
            "3d",
            vals,
        )
        cases.append({"case_id": case_id, **vals})

    summary = {
        "protocol_version": "h3d_v1_protocol_v1",
        "split": split,
        "method": method,
        "case_count": len(rows),
        "eval_points": eval_points,
        "fscore_tau": fscore_tau,
        "metrics": {metric: summarize_values(values) for metric, values in metric_values.items()},
        "cases": cases,
    }
    out_path = results_root / method / split / "summary_3d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
