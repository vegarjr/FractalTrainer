"""Box-counting fractal dimension for low-dim point clouds.

For each scale ε, count the number of non-empty unit-hypercube boxes
covering the (normalized) points:
    N(ε) ~ (1/ε)^D  =>  log N(ε) ~ D * log(1/ε)

Box-counting saturates in high-dimensional ambient space (every point
gets its own box). This implementation is intended as a SANITY CHECK on
projected-to-3d trajectories, not as the primary metric. Use
correlation_dim() for higher ambient dim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoxCountingResult:
    dim: float
    r_squared: float
    n_points: int
    box_sizes: np.ndarray = field(default_factory=lambda: np.array([]))
    box_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "n_scales": int(self.box_sizes.size),
            "error": self.error,
        }


def _count_boxes(points: np.ndarray, eps: float) -> int:
    cells = np.floor(points / eps).astype(np.int64)
    cell_tuples = set(map(tuple, cells))
    return len(cell_tuples)


def box_counting_dim(
    points: np.ndarray,
    box_sizes: np.ndarray | None = None,
    min_points: int = 20,
    n_scales: int = 12,
    trim_fraction: float = 0.2,
) -> BoxCountingResult:
    """Box-counting fractal dimension.

    Args:
        points: (N, D) array. Normalized to unit hypercube internally.
        box_sizes: optional log-spaced ε values. If None, auto-generated.
        min_points: minimum N for a meaningful estimate.
        n_scales: number of ε values if box_sizes is None.
        trim_fraction: fraction of endpoints to drop before fitting
            (mitigates known box-counting bias at the extremes).

    Returns:
        BoxCountingResult with dim, r_squared, box_sizes, box_counts.
    """
    if points.ndim != 2:
        raise ValueError(f"points must be 2-D (N, D), got shape {points.shape}")
    n_points, n_dim = points.shape

    if n_points < min_points:
        return BoxCountingResult(
            dim=float("nan"),
            r_squared=float("nan"),
            n_points=n_points,
            error=f"n_points={n_points} < min_points={min_points}",
        )

    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    p_range = p_max - p_min
    p_range[p_range < 1e-12] = 1.0
    normalized = (points - p_min) / p_range

    if box_sizes is None:
        box_sizes = np.logspace(-2.0, np.log10(0.5), n_scales)

    counts = np.array(
        [_count_boxes(normalized, eps) for eps in box_sizes],
        dtype=float,
    )

    mask = counts > 1
    if mask.sum() < 4:
        return BoxCountingResult(
            dim=float("nan"),
            r_squared=float("nan"),
            n_points=n_points,
            box_sizes=box_sizes,
            box_counts=counts,
            error="too few box-count scales with count > 1",
        )

    log_inv_eps = -np.log(box_sizes[mask])
    log_n = np.log(counts[mask])

    k = len(log_inv_eps)
    trim = max(1, int(k * trim_fraction))
    if k - 2 * trim < 3:
        start, end = 0, k
    else:
        start, end = trim, k - trim

    x = log_inv_eps[start:end]
    y = log_n[start:end]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

    return BoxCountingResult(
        dim=float(slope),
        r_squared=float(r_squared),
        n_points=n_points,
        box_sizes=box_sizes,
        box_counts=counts,
    )
