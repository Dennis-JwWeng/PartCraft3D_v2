from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "h3d_v1_protocol_v1"
DEFAULT_RENDER_NAMES = tuple(f"{i:03d}.png" for i in range(0, 300, 30))


def load_split(protocol_root: str | Path, split: str) -> list[dict[str, Any]]:
    protocol_root = Path(protocol_root)
    path = protocol_root / "splits" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                f"{path}:{line_no} has protocol_version={row.get('protocol_version')!r}, "
                f"expected {PROTOCOL_VERSION!r}"
            )
        rows.append(row)
    return rows


def resolve_protocol_path(protocol_root: str | Path, rel_path: str | Path) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute():
        return rel
    return Path(protocol_root) / rel


def method_split_root(results_root: str | Path, method: str, split: str) -> Path:
    return Path(results_root) / method / split


def case_result_dir(results_root: str | Path, method: str, split: str, case_id: str) -> Path:
    return method_split_root(results_root, method, split) / case_id


def read_case_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_case_metrics(path: Path, section: str, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_case_metrics(path)
    data[section] = metrics
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_values(values: list[float]) -> dict[str, Any]:
    import numpy as np

    if not values:
        return {"mean": None, "std": None, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "count": int(arr.size)}
