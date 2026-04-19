"""Divergence score: how far the current shape is from the target.

Two target methods supported:

(A) Scalar dimension methods (v1): method in {correlation_dim, box_counting}.
    A single measured scalar d is compared to target.dim_target:
        score(d) = |d - target_dim| / tolerance
    score(target)             == 0
    score(target ± tolerance) == 1
    score(target ± 2 * tolerance) == 2

(B) Golden-run matching (v2 Sprint 3): method == "golden_run_match".
    A full signature dict (from build_signature_from_report) is compared
    to the golden run's signature via z-normalized Euclidean distance.
    See target.golden_run.golden_run_distance.
    score == 0 means the signatures are identical.
    score == 1 means the current signature deviates by one unit-scale
      per feature on average (a moderate shape mismatch).

should_intervene returns True only when score > hysteresis. Using
hysteresis > 1 prevents flapping when the current shape sits near the
band edge — the repair loop only triggers on clear divergence.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

from fractaltrainer.target.target_shape import TargetShape


def divergence_score(
    current: Union[float, dict],
    target: TargetShape,
) -> float:
    """Unified divergence scorer.

    Args:
        current:
            For scalar methods (correlation_dim, box_counting): a float
            dim value.
            For golden_run_match: a signature dict (see
            fractaltrainer.target.golden_run.build_signature_from_report).
        target: the TargetShape describing method + parameters.

    Returns:
        Divergence score (>= 0). Non-finite input → +inf.
    """
    if target.method in ("correlation_dim", "box_counting"):
        if isinstance(current, dict):
            # Accept a signature dict; extract correlation_dim field.
            d = current.get("correlation_dim", float("nan"))
            try:
                d = float(d)
            except (TypeError, ValueError):
                d = float("nan")
        else:
            d = float(current) if current is not None else float("nan")
        if not math.isfinite(d):
            return float("inf")
        return abs(d - target.dim_target) / target.tolerance

    if target.method == "golden_run_match":
        if target.golden_run_path is None:
            raise ValueError(
                "target.method='golden_run_match' requires golden_run_path")
        if not isinstance(current, dict):
            raise TypeError(
                "golden_run_match divergence needs a signature dict, "
                "not a scalar")
        # Lazy import to avoid circular
        from fractaltrainer.target.golden_run import (
            GoldenRun,
            golden_run_distance,
        )
        golden = GoldenRun.load(Path(target.golden_run_path))
        return golden_run_distance(current, golden)

    raise ValueError(f"unknown target.method: {target.method!r}")


def should_intervene(score: float, target: TargetShape) -> bool:
    if score is None or math.isnan(score) or math.isinf(score):
        return True
    return score > target.hysteresis


def within_band(current: Union[float, dict], target: TargetShape) -> bool:
    """True iff the current measurement lies in the acceptance band.

    Scalar methods: [dim_target - tolerance, dim_target + tolerance].
    Golden-run match: divergence_score <= 1.0 (one unit-scale from golden).

    Uses a tiny epsilon (1e-9) to absorb floating-point rounding at the
    band edge — without it, `within_band(target - tolerance)` can return
    False due to IEEE-754 representation (e.g. 1.2 − 1.5 = −0.30000…04).
    """
    score = divergence_score(current, target)
    if score is None or math.isnan(score) or math.isinf(score):
        return False
    return score <= 1.0 + 1e-9
