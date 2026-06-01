from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from .protocol import (
    DEFAULT_RENDER_NAMES,
    case_result_dir,
    load_split,
    resolve_protocol_path,
    summarize_values,
    write_case_metrics,
)

_SUPPORTED = {"psnr", "ssim", "lpips", "dino_i"}


def _load_rgb(path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    img = Image.open(path)
    if img.mode != "RGB":
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
        bg.alpha_composite(rgba)
        img = bg.convert("RGB")
    if image_size is not None:
        img = img.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = float(np.mean((img_a - img_b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim_global(img_a: np.ndarray, img_b: np.ndarray) -> float:
    # Lightweight SSIM over the whole image. This keeps the protocol runner usable
    # without skimage while preserving the same score range and interpretation.
    x = img_a.reshape(-1, img_a.shape[-1])
    y = img_b.reshape(-1, img_b.shape[-1])
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for c in range(x.shape[1]):
        xc = x[:, c]
        yc = y[:, c]
        mux = float(xc.mean())
        muy = float(yc.mean())
        vx = float(xc.var())
        vy = float(yc.var())
        cov = float(((xc - mux) * (yc - muy)).mean())
        denom = (mux * mux + muy * muy + c1) * (vx + vy + c2)
        val = ((2 * mux * muy + c1) * (2 * cov + c2)) / denom if denom else 1.0
        scores.append(val)
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def _optional_metric_unavailable(name: str) -> RuntimeError:
    return RuntimeError(
        f"2D metric {name!r} requires optional dependencies that are not available in this environment. "
        "Use --metrics-2d psnr,ssim or install the matching lpips/transformers stack."
    )


def compute_image_pair_metrics(
    pred_path: Path,
    target_path: Path,
    *,
    metrics: Sequence[str] = ("psnr", "ssim"),
    image_size: tuple[int, int] | None = (512, 512),
) -> dict[str, float]:
    requested = tuple(m.lower() for m in metrics)
    unknown = set(requested) - _SUPPORTED
    if unknown:
        raise ValueError(f"unsupported 2D metrics: {sorted(unknown)}")
    pred = _load_rgb(pred_path, image_size=image_size)
    target = _load_rgb(target_path, image_size=image_size)
    out: dict[str, float] = {}
    for metric in requested:
        if metric == "psnr":
            out[metric] = psnr(pred, target)
        elif metric == "ssim":
            out[metric] = ssim_global(pred, target)
        elif metric in {"lpips", "dino_i"}:
            raise _optional_metric_unavailable(metric)
    return out


def compute_2d_metrics(
    *,
    protocol_root: str | Path,
    split: str,
    method: str,
    results_root: str | Path | None = None,
    metrics: Sequence[str] = ("psnr", "ssim"),
    render_names: Iterable[str] = DEFAULT_RENDER_NAMES,
    image_size: tuple[int, int] | None = (512, 512),
) -> dict:
    protocol_root = Path(protocol_root)
    results_root = Path(results_root) if results_root is not None else protocol_root / "results"
    rows = load_split(protocol_root, split)
    names = tuple(render_names)
    metric_values: dict[str, list[float]] = {m.lower(): [] for m in metrics}
    case_summaries = []

    for row in tqdm(rows, desc="H3D_v1 2D metrics"):
        case_id = row["case_id"]
        pred_dir = case_result_dir(results_root, method, split, case_id) / "renders" / "edit"
        target_dir = resolve_protocol_path(protocol_root, row["target"]["after_render_dir"])
        per_view = []
        per_case_values: dict[str, list[float]] = {m.lower(): [] for m in metrics}
        for name in names:
            vals = compute_image_pair_metrics(
                pred_dir / name,
                target_dir / name,
                metrics=metrics,
                image_size=image_size,
            )
            per_view.append({"view": name, **vals})
            for metric, value in vals.items():
                metric_values[metric].append(value)
                per_case_values[metric].append(value)
        case_metrics = {
            "view_count": len(per_view),
            "views": per_view,
            "summary": {metric: summarize_values(values) for metric, values in per_case_values.items()},
        }
        write_case_metrics(
            case_result_dir(results_root, method, split, case_id) / "metrics.json",
            "2d",
            case_metrics,
        )
        case_summaries.append({"case_id": case_id, **case_metrics["summary"]})

    summary = {
        "protocol_version": "h3d_v1_protocol_v1",
        "split": split,
        "method": method,
        "case_count": len(rows),
        "view_count_per_case": len(names),
        "metrics": {metric: summarize_values(values) for metric, values in metric_values.items()},
        "cases": case_summaries,
    }
    out_path = results_root / method / split / "summary_2d.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
