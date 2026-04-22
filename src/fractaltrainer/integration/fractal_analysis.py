"""v3 Sprint 18 — fractal audit helpers.

Applies the existing `correlation_dim` (Grassberger-Procaccia) to
the registry's structural objects, with a scale-stability check
that tests whether the local slope of log C(r) vs log r is stable
across r-bands. A true fractal has a single slope across a scaling
window; if the slope varies significantly, the object is not
fractal in Bourke's sense even if a single-D fit yields a
non-integer value.

Three audit targets:
  1. Signature point cloud — the registry as points in R^1000.
  2. Growth trajectory — sequence of oracle signatures over a
     query stream (the sequence of tasks the registry encounters).
  3. Label-set lattice — the binary-indicator representation of
     each expert's class-1 label set in R^10 (a Boolean lattice
     restricted to the subsets actually registered).

Each audit produces:
  * correlation dimension D with R²
  * slope-stability across three r-bands (small / mid / large)
  * random-baseline D for the same N and ambient dim (controls
    for ambient-dimension saturation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from fractaltrainer.geometry.correlation_dim import (
    CorrelationDimResult,
    correlation_dim,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry


@dataclass
class ScaleStabilityResult:
    """Slope of log C(r) vs log r computed in three r-bands."""

    slope_small: float
    slope_mid: float
    slope_large: float
    slope_variance: float  # max |slope_band - mean(slopes)|
    band_r_ranges: list[tuple[float, float]] = field(default_factory=list)

    def is_stable(self, threshold: float = 0.3) -> bool:
        return (
            np.isfinite(self.slope_variance)
            and self.slope_variance < threshold
        )

    def to_dict(self) -> dict:
        return {
            "slope_small": float(self.slope_small),
            "slope_mid": float(self.slope_mid),
            "slope_large": float(self.slope_large),
            "slope_variance": float(self.slope_variance),
            "band_r_ranges": [
                [float(a), float(b)] for a, b in self.band_r_ranges
            ],
            "is_stable_threshold_0_3": self.is_stable(0.3),
        }


@dataclass
class FractalAuditResult:
    """Combined audit for one structural object."""

    name: str
    n_points: int
    ambient_dim: int
    correlation_dim: CorrelationDimResult
    random_baseline_dim: CorrelationDimResult
    scale_stability: ScaleStabilityResult
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_points": self.n_points,
            "ambient_dim": self.ambient_dim,
            "dim": self.correlation_dim.to_dict(),
            "random_baseline_dim": self.random_baseline_dim.to_dict(),
            "scale_stability": self.scale_stability.to_dict(),
            "notes": self.notes,
        }


def _compute_scale_stability(
    result: CorrelationDimResult,
) -> ScaleStabilityResult:
    """Fit slopes in three r-bands (lowest, middle, upper third of
    the scaling window) and report variance.

    The bands are defined in terms of the scaling window indices
    detected by `_auto_scaling_range` inside `correlation_dim`. Using
    the full r-range would include r-values outside the scaling region,
    where the slope is known to deviate.
    """
    if result.radii.size == 0 or result.correlation_sums.size == 0:
        return ScaleStabilityResult(
            slope_small=float("nan"),
            slope_mid=float("nan"),
            slope_large=float("nan"),
            slope_variance=float("nan"),
        )

    mask = result.correlation_sums > 0
    log_r = np.log(result.radii[mask])
    log_c = np.log(result.correlation_sums[mask])
    start, end = result.scaling_start, result.scaling_end
    x = log_r[start:end]
    y = log_c[start:end]
    if x.size < 6:
        return ScaleStabilityResult(
            slope_small=float("nan"),
            slope_mid=float("nan"),
            slope_large=float("nan"),
            slope_variance=float("nan"),
        )

    third = x.size // 3
    bands = [
        (0, third),
        (third, 2 * third),
        (2 * third, x.size),
    ]
    slopes = []
    band_ranges = []
    for (a, b) in bands:
        if b - a < 2:
            slopes.append(float("nan"))
            band_ranges.append((float(np.exp(x[a])), float(np.exp(x[b - 1]))))
            continue
        slope, _ = np.polyfit(x[a:b], y[a:b], 1)
        slopes.append(float(slope))
        band_ranges.append((float(np.exp(x[a])), float(np.exp(x[b - 1]))))

    finite_slopes = [s for s in slopes if np.isfinite(s)]
    if len(finite_slopes) < 2:
        variance = float("nan")
    else:
        mean_slope = float(np.mean(finite_slopes))
        variance = float(
            max(abs(s - mean_slope) for s in finite_slopes)
        )

    return ScaleStabilityResult(
        slope_small=slopes[0],
        slope_mid=slopes[1],
        slope_large=slopes[2],
        slope_variance=variance,
        band_r_ranges=band_ranges,
    )


def _random_baseline(
    n_points: int, ambient_dim: int, seed: int = 42,
) -> CorrelationDimResult:
    """Compute correlation dim of uniform-random points with the
    same (N, D) as the audited object. Serves as the saturation
    baseline — at high ambient dim, random points can fill all
    dimensions and correlation_dim saturates near min(N−1, D).
    """
    rng = np.random.RandomState(seed)
    points = rng.randn(n_points, ambient_dim).astype(np.float64)
    return correlation_dim(points, min_points=min(20, n_points), seed=seed)


def audit_signature_cloud(
    registry: FractalRegistry,
    *,
    name: str = "signature_cloud",
    random_seed: int = 42,
) -> FractalAuditResult:
    """Audit the registry's signatures as a point cloud in signature space."""
    entries = registry.entries()
    if not entries:
        raise ValueError("registry is empty — cannot audit signature cloud")
    sigs = np.stack([e.signature for e in entries], axis=0).astype(np.float64)
    n, d = sigs.shape
    cd = correlation_dim(sigs, min_points=min(20, n), seed=random_seed)
    stability = _compute_scale_stability(cd)
    baseline = _random_baseline(n, d, seed=random_seed)
    return FractalAuditResult(
        name=name, n_points=n, ambient_dim=d,
        correlation_dim=cd,
        random_baseline_dim=baseline,
        scale_stability=stability,
        notes=f"signatures from registry (N={n}, ambient {d}-d)",
    )


def audit_trajectory(
    signatures: Sequence[np.ndarray],
    *,
    name: str = "growth_trajectory",
    random_seed: int = 42,
) -> FractalAuditResult:
    """Audit a sequence of signatures as a trajectory.

    Args:
        signatures: list of 1-D signature vectors in order of arrival.
    """
    sigs = np.stack(list(signatures), axis=0).astype(np.float64)
    n, d = sigs.shape
    cd = correlation_dim(sigs, min_points=min(20, n), seed=random_seed)
    stability = _compute_scale_stability(cd)
    baseline = _random_baseline(n, d, seed=random_seed)
    return FractalAuditResult(
        name=name, n_points=n, ambient_dim=d,
        correlation_dim=cd,
        random_baseline_dim=baseline,
        scale_stability=stability,
        notes=f"growth trajectory (N={n} steps, ambient {d}-d)",
    )


def audit_label_lattice(
    label_sets: Iterable[Iterable[int]],
    *,
    n_classes: int = 10,
    name: str = "label_lattice",
    random_seed: int = 42,
) -> FractalAuditResult:
    """Audit task label sets as binary indicator vectors in {0,1}^n_classes.

    Args:
        label_sets: iterable of (class-1) label sets, one per expert.
        n_classes: full label space size.
    """
    vectors: list[np.ndarray] = []
    for s in label_sets:
        v = np.zeros(n_classes, dtype=np.float64)
        for lbl in s:
            if 0 <= int(lbl) < n_classes:
                v[int(lbl)] = 1.0
        vectors.append(v)
    if not vectors:
        raise ValueError("no label sets provided")
    points = np.stack(vectors, axis=0)
    n, d = points.shape
    cd = correlation_dim(points, min_points=min(20, n), seed=random_seed)
    stability = _compute_scale_stability(cd)
    baseline = _random_baseline(n, d, seed=random_seed)
    return FractalAuditResult(
        name=name, n_points=n, ambient_dim=d,
        correlation_dim=cd,
        random_baseline_dim=baseline,
        scale_stability=stability,
        notes=f"binary-indicator label vectors in R^{d}",
    )


def classify_verdict(
    results: Sequence[FractalAuditResult],
    *,
    stability_threshold: float = 0.3,
    weak_threshold: float = 0.6,
    min_non_integer_dim: float = 1.2,
    max_non_integer_dim: float = 9.0,
) -> dict:
    """Turn audit results into pass / weak-pass / fail verdict.

    Criteria (from the plan):
      - Pass:      on ≥1 object, D ∈ [min, max] AND slope_variance < 0.3
      - Weak-pass: D non-integer but variance 0.3-0.6
      - Fail:      D = integer on all objects OR variance > 0.6 everywhere

    Returns a verdict dict with per-object classification.
    """
    per_object = []
    any_pass = False
    any_weak = False
    for r in results:
        d_val = r.correlation_dim.dim
        sv = r.scale_stability.slope_variance
        non_integer = (
            np.isfinite(d_val)
            and min_non_integer_dim <= d_val <= max_non_integer_dim
            and abs(d_val - round(d_val)) > 0.15
        )
        stable = np.isfinite(sv) and sv < stability_threshold
        weakly_stable = np.isfinite(sv) and stability_threshold <= sv < weak_threshold
        classification = "fail"
        if non_integer and stable:
            classification = "pass"
            any_pass = True
        elif non_integer and weakly_stable:
            classification = "weak_pass"
            any_weak = True
        per_object.append({
            "name": r.name,
            "dim": float(d_val),
            "slope_variance": float(sv) if np.isfinite(sv) else None,
            "classification": classification,
        })
    if any_pass:
        overall = "pass"
    elif any_weak:
        overall = "weak_pass"
    else:
        overall = "fail"
    return {
        "overall": overall,
        "per_object": per_object,
    }
