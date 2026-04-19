"""Correlation dimension (Grassberger-Procaccia) for trajectory point clouds.

The correlation sum is:
    C(r) = (2 / (N(N-1))) * sum_{i<j} H(r - ||x_i - x_j||)

where H is the Heaviside step function. In the scaling region,
C(r) ~ r^D  =>  log C(r) ~ D * log r. The slope D is the correlation
dimension.

Only pairwise distances are needed, so the method is robust to high
ambient dimension — unlike box-counting, which saturates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CorrelationDimResult:
    dim: float
    r_squared: float
    n_points: int
    radii: np.ndarray = field(default_factory=lambda: np.array([]))
    correlation_sums: np.ndarray = field(default_factory=lambda: np.array([]))
    scaling_start: int = 0
    scaling_end: int = 0
    used_max_pairs: bool = False
    n_pairs_sampled: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "n_radii": int(self.radii.size),
            "scaling_start": self.scaling_start,
            "scaling_end": self.scaling_end,
            "used_max_pairs": self.used_max_pairs,
            "n_pairs_sampled": self.n_pairs_sampled,
            "error": self.error,
        }


def _pairwise_distances(points: np.ndarray, max_pairs: int, seed: int
                        ) -> tuple[np.ndarray, int, bool]:
    """Return an array of pairwise distances. Use all pairs if N*(N-1)/2
    fits within max_pairs, otherwise sample uniformly.
    """
    n = points.shape[0]
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        diffs = points[:, None, :] - points[None, :, :]
        dists_full = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(n, k=1)
        dists = dists_full[iu]
        return dists, total_pairs, False

    rng = np.random.RandomState(seed)
    i_idx = rng.randint(0, n, size=max_pairs)
    j_idx = rng.randint(0, n, size=max_pairs)
    same = i_idx == j_idx
    while same.any():
        j_idx[same] = rng.randint(0, n, size=int(same.sum()))
        same = i_idx == j_idx
    dists = np.linalg.norm(points[i_idx] - points[j_idx], axis=-1)
    return dists, max_pairs, True


def _auto_scaling_range(log_r: np.ndarray, log_c: np.ndarray,
                        min_span: int = 6) -> tuple[int, int]:
    """Find the longest contiguous window with acceptable linearity.

    Strategy: for every contiguous window of size >= min_span, compute the
    linear-fit residual per point (RSS / n). Among windows where that value
    is within a factor of 2 of the best observed window, pick the LONGEST.
    This prefers broad scaling regions over narrow local-minima fits.
    """
    n = len(log_r)
    if n < min_span + 2:
        return 0, max(n, 1)

    best_rss_per_point = np.inf
    rss_table: list[tuple[int, int, float]] = []

    for start in range(n - min_span + 1):
        for end in range(start + min_span, n + 1):
            x = log_r[start:end]
            y = log_c[start:end]
            if np.std(x) < 1e-12:
                continue
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            residual = y - pred
            rss_pp = float(np.mean(residual ** 2))
            rss_table.append((start, end, rss_pp))
            if rss_pp < best_rss_per_point:
                best_rss_per_point = rss_pp

    if not rss_table or not np.isfinite(best_rss_per_point):
        return 0, n

    threshold = max(best_rss_per_point * 2.0, 1e-8)
    acceptable = [(s, e) for s, e, r in rss_table if r <= threshold]
    if not acceptable:
        return 0, n
    # Among acceptable windows, pick the longest (ties broken by earlier start)
    acceptable.sort(key=lambda se: (-(se[1] - se[0]), se[0]))
    return acceptable[0]


def correlation_dim(
    points: np.ndarray,
    radii: np.ndarray | None = None,
    max_pairs: int = 50_000,
    min_points: int = 20,
    seed: int = 0,
) -> CorrelationDimResult:
    """Grassberger-Procaccia correlation dimension.

    Args:
        points: (N, D) array.
        radii: optional log-spaced radii. If None, auto-generated from the
            distance distribution.
        max_pairs: cap on pairwise distance computation (random sample
            above this; exact below).
        min_points: minimum N for a meaningful estimate. Below this,
            returns NaN dim with an error message.
        seed: RNG seed for pair sampling.

    Returns:
        CorrelationDimResult with dim, r_squared, and scaling-region indices.
    """
    if points.ndim != 2:
        raise ValueError(f"points must be 2-D (N, D), got shape {points.shape}")
    n_points, _ = points.shape

    if n_points < min_points:
        return CorrelationDimResult(
            dim=float("nan"),
            r_squared=float("nan"),
            n_points=n_points,
            error=f"n_points={n_points} < min_points={min_points}",
        )

    dists, n_pairs, used_sample = _pairwise_distances(points, max_pairs, seed)
    dists = dists[dists > 0]
    if dists.size == 0:
        return CorrelationDimResult(
            dim=float("nan"),
            r_squared=float("nan"),
            n_points=n_points,
            used_max_pairs=used_sample,
            n_pairs_sampled=n_pairs,
            error="all pairwise distances were zero",
        )

    if radii is None:
        lo = float(np.quantile(dists, 0.02))
        hi = float(np.quantile(dists, 0.98))
        if lo <= 0 or hi <= lo:
            return CorrelationDimResult(
                dim=float("nan"),
                r_squared=float("nan"),
                n_points=n_points,
                used_max_pairs=used_sample,
                n_pairs_sampled=n_pairs,
                error="degenerate distance distribution",
            )
        radii = np.logspace(np.log10(lo), np.log10(hi), 24)

    counts = np.array([(dists <= r).sum() for r in radii], dtype=float)
    total = float(dists.size)
    corr_sums = counts / total
    mask = corr_sums > 0
    if mask.sum() < 4:
        return CorrelationDimResult(
            dim=float("nan"),
            r_squared=float("nan"),
            n_points=n_points,
            radii=radii,
            correlation_sums=corr_sums,
            used_max_pairs=used_sample,
            n_pairs_sampled=n_pairs,
            error="too few nonzero correlation sums",
        )

    log_r = np.log(radii[mask])
    log_c = np.log(corr_sums[mask])

    start, end = _auto_scaling_range(log_r, log_c)
    if end - start < 2:
        return CorrelationDimResult(
            dim=float("nan"),
            r_squared=float("nan"),
            n_points=n_points,
            radii=radii,
            correlation_sums=corr_sums,
            scaling_start=start,
            scaling_end=end,
            used_max_pairs=used_sample,
            n_pairs_sampled=n_pairs,
            error="scaling range collapse",
        )

    x = log_r[start:end]
    y = log_c[start:end]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

    return CorrelationDimResult(
        dim=float(slope),
        r_squared=float(r_squared),
        n_points=n_points,
        radii=radii,
        correlation_sums=corr_sums,
        scaling_start=int(start),
        scaling_end=int(end),
        used_max_pairs=used_sample,
        n_pairs_sampled=int(n_pairs),
    )
