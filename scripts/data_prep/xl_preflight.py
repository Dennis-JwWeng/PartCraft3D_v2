"""PartVerse XL raw-input preflight (CPU-only, no pipeline_v3 imports).

Validates objects under ``meshes/textured_part_glbs/<uuid>`` + ``captions/<uuid>``
before GPU stages run.  Used by :mod:`scripts.ops.preflight_partversexl` and
:func:`partcraft.pipeline_v3.run_trellis2.apply_xl_input_patch`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from scripts.data_prep.mesh_sources import RawXLSource, get_mesh_source

_PART_GLB = re.compile(r"^\d+$")

# Reasons that always block pipeline work on this object.
HARD_BLOCK_REASONS = frozenset({
    "no_mesh_dir",
    "no_parts",
    "no_caption",
    "assemble_failed",
})

# Skipped later by gen_edits anyway — safe to drop before encode to save GPU.
SOFT_BLOCK_REASONS = frozenset({
    "too_many_parts",
})


@dataclass
class XLPreflightReport:
    obj_id: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    n_parts: int = 0

    @property
    def blocked(self) -> bool:
        return not self.ok


def _part_glbs(mesh_dir: Path) -> list[Path]:
    if not mesh_dir.is_dir():
        return []
    return sorted(
        (p for p in mesh_dir.glob("*.glb") if _PART_GLB.fullmatch(p.stem)),
        key=lambda p: int(p.stem),
    )


def inspect_xl_object(
    obj_id: str,
    mesh_dir: Path,
    caption_path: Path,
    *,
    max_parts: int = 16,
    check_assemble: bool = False,
) -> XLPreflightReport:
    """Run CPU checks for one raw-XL object directory."""
    reasons: list[str] = []
    n_parts = 0

    if not mesh_dir.is_dir():
        reasons.append("no_mesh_dir")
    else:
        parts = _part_glbs(mesh_dir)
        n_parts = len(parts)
        if n_parts == 0:
            reasons.append("no_parts")
        elif max_parts > 0 and n_parts > max_parts:
            reasons.append(f"too_many_parts({n_parts}>{max_parts})")

    if not caption_path.is_file():
        reasons.append("no_caption")

    if check_assemble and mesh_dir.is_dir() and "no_parts" not in reasons:
        try:
            from scripts.data_prep.partverse.pack_npz_xl import assemble_full_glb_bytes

            if assemble_full_glb_bytes(mesh_dir) is None:
                reasons.append("assemble_failed")
        except Exception:
            reasons.append("assemble_failed")

    return XLPreflightReport(
        obj_id=obj_id,
        ok=not reasons,
        reasons=reasons,
        n_parts=n_parts,
    )


def _xl_roots(cfg: dict) -> RawXLSource | None:
    data = cfg.get("data") or {}
    source = str(data.get("source", "")).strip().lower()
    if source not in ("partversexl_raw", "xl_raw", "raw_xl"):
        return None
    ms = get_mesh_source(cfg)
    if not isinstance(ms, RawXLSource):
        return None
    return ms


def preflight_config(cfg: dict) -> dict:
    """Merge ``data.xl_preflight`` with defaults."""
    data = cfg.get("data") or {}
    raw = dict(data.get("xl_preflight") or {})
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_parts": int(raw.get("max_parts", 16)),
        "check_assemble": bool(raw.get("check_assemble", False)),
        "filter": bool(raw.get("filter", True)),
        "strict": bool(raw.get("strict", False)),
    }


def inspect_batch(
    obj_ids: Iterable[str],
    cfg: dict,
    *,
    max_parts: int | None = None,
    check_assemble: bool | None = None,
) -> list[XLPreflightReport]:
    """Inspect a list of object ids against the config's XL roots."""
    xl = _xl_roots(cfg)
    if xl is None:
        raise ValueError("config is not partversexl_raw — xl_preflight does not apply")

    pf = preflight_config(cfg)
    mp = pf["max_parts"] if max_parts is None else max_parts
    ca = pf["check_assemble"] if check_assemble is None else check_assemble

    reports: list[XLPreflightReport] = []
    for oid in obj_ids:
        mesh_dir = xl.textured_root / oid
        cap = xl.captions_root / oid / "caption.json"
        reports.append(
            inspect_xl_object(
                oid, mesh_dir, cap, max_parts=mp, check_assemble=ca,
            )
        )
    return reports


def partition_reports(
    reports: Iterable[XLPreflightReport],
) -> tuple[list[XLPreflightReport], list[XLPreflightReport]]:
    allow, blocked = [], []
    for r in reports:
        (allow if r.ok else blocked).append(r)
    return allow, blocked


def reason_kind(reason: str) -> str:
    """Map ``too_many_parts(19>16)`` → ``too_many_parts``."""
    return reason.split("(", 1)[0]


__all__ = [
    "HARD_BLOCK_REASONS",
    "SOFT_BLOCK_REASONS",
    "XLPreflightReport",
    "inspect_batch",
    "inspect_xl_object",
    "partition_reports",
    "preflight_config",
    "reason_kind",
]
