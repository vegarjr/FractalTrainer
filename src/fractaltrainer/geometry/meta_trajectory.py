"""Geometric metrics on the *meta-trajectory* (the repair loop's own path
through hparam-space).

Thin wrapper: reuses v1's trajectory_metrics and correlation_dim. The only
additions are (a) convenience functions that take a MetaTrajectory directly,
and (b) a "convergence signature" metric that's specific to repair loops:
does the sequence of divergence scores decrease monotonically, or does it
bounce?

The point of measuring meta-trajectory geometry is to ask: when the AI
rewrites itself, do those rewrites *themselves* show a fractal shape?
That's the recursive self-similarity claim from the original vision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fractaltrainer.geometry.correlation_dim import (
    CorrelationDimResult,
    correlation_dim,
)
from fractaltrainer.geometry.trajectory import trajectory_metrics
from fractaltrainer.observer.repair_history import MetaTrajectory


@dataclass
class MetaTrajectorySummary:
    n_points: int
    n_transitions: int
    geometry: dict
    correlation_dim: CorrelationDimResult | None
    convergence: dict

    def to_dict(self) -> dict:
        return {
            "n_points": self.n_points,
            "n_transitions": self.n_transitions,
            "geometry": self.geometry,
            "correlation_dim": (
                self.correlation_dim.to_dict() if self.correlation_dim else None
            ),
            "convergence": self.convergence,
        }


def _convergence_signature(divergence_scores: list[float]) -> dict:
    finite = [d for d in divergence_scores if np.isfinite(d)]
    if len(finite) < 2:
        # With 0 or 1 point we can't talk about convergence — only
        # report the starting value when we have it.
        return {
            "start_divergence": finite[0] if finite else None,
            "end_divergence": None,
            "total_reduction": None,
            "monotonic_decreasing": None,
            "bounces": 0,
            "mean_step_reduction": None,
        }

    diffs = np.diff(np.asarray(finite))
    monotonic = bool(np.all(diffs <= 0))
    # A "bounce" = one direction reversal in the divergence sequence.
    sign_changes = 0
    prev = None
    for d in diffs:
        s = 0 if abs(d) < 1e-12 else (1 if d > 0 else -1)
        if prev is not None and s != 0 and prev != 0 and s != prev:
            sign_changes += 1
        if s != 0:
            prev = s

    return {
        "start_divergence": float(finite[0]),
        "end_divergence": float(finite[-1]),
        "total_reduction": float(finite[0] - finite[-1]),
        "monotonic_decreasing": monotonic,
        "bounces": int(sign_changes),
        "mean_step_reduction": float(-np.mean(diffs)),
    }


def summarize_meta_trajectory(
    mt: MetaTrajectory,
    min_points_for_dim: int = 20,
) -> MetaTrajectorySummary:
    """Compute geometry + convergence signature on a MetaTrajectory.

    If the meta-trajectory has fewer than `min_points_for_dim` points,
    correlation dim is skipped and reported as None. Trajectory metrics
    (path length, tortuosity, ...) are always computed when n_points ≥ 2.
    """
    n = mt.n_points
    if n < 2:
        return MetaTrajectorySummary(
            n_points=n,
            n_transitions=mt.n_transitions,
            geometry={},
            correlation_dim=None,
            convergence=_convergence_signature(mt.divergence_scores),
        )

    geometry = trajectory_metrics(mt.states)

    dim_res = None
    if n >= min_points_for_dim:
        dim_res = correlation_dim(mt.states, min_points=min_points_for_dim)

    convergence = _convergence_signature(mt.divergence_scores)

    return MetaTrajectorySummary(
        n_points=n,
        n_transitions=mt.n_transitions,
        geometry=geometry,
        correlation_dim=dim_res,
        convergence=convergence,
    )
