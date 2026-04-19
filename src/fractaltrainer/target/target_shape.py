"""TargetShape — the reference geometric signature the trajectory should match."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectionSpec:
    method: str = "random_proj"   # random_proj | pca | none
    n_components: int = 16
    seed: int = 0


@dataclass
class TargetShape:
    dim_target: float
    tolerance: float
    method: str = "correlation_dim"     # correlation_dim | box_counting
    min_snapshots: int = 20
    hysteresis: float = 1.2
    max_repair_iters: int = 5
    projection: ProjectionSpec = field(default_factory=ProjectionSpec)

    @property
    def band_low(self) -> float:
        return self.dim_target - self.tolerance

    @property
    def band_high(self) -> float:
        return self.dim_target + self.tolerance

    def to_dict(self) -> dict:
        return asdict(self)


def load_target(path: str | Path) -> TargetShape:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    proj_dict = data.get("projection", {}) or {}
    proj = ProjectionSpec(
        method=str(proj_dict.get("method", "random_proj")),
        n_components=int(proj_dict.get("n_components", 16)),
        seed=int(proj_dict.get("seed", 0)),
    )

    target = TargetShape(
        dim_target=float(data["dim_target"]),
        tolerance=float(data["tolerance"]),
        method=str(data.get("method", "correlation_dim")),
        min_snapshots=int(data.get("min_snapshots", 20)),
        hysteresis=float(data.get("hysteresis", 1.2)),
        max_repair_iters=int(data.get("max_repair_iters", 5)),
        projection=proj,
    )

    _validate_target(target)
    return target


def _validate_target(t: TargetShape) -> None:
    if t.tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {t.tolerance}")
    if t.dim_target <= 0:
        raise ValueError(f"dim_target must be > 0, got {t.dim_target}")
    if t.method not in {"correlation_dim", "box_counting"}:
        raise ValueError(f"unknown method: {t.method!r}")
    if t.projection.method not in {"random_proj", "pca", "none"}:
        raise ValueError(
            f"unknown projection method: {t.projection.method!r}"
        )
    if t.hysteresis < 1.0:
        raise ValueError(
            f"hysteresis must be >= 1.0 (inside tolerance band) got {t.hysteresis}"
        )
    if t.max_repair_iters < 1:
        raise ValueError(f"max_repair_iters must be >= 1, got {t.max_repair_iters}")
