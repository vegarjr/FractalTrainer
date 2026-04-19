"""Divergence score: how far the current shape is from the target.

A trajectory's measured fractal dimension d is mapped to a scalar score:

    score(d) = |d - target_dim| / tolerance

Properties:
    score(target) == 0
    score(target ± tolerance) == 1     (band edge)
    score(target ± 2*tolerance) == 2   (twice outside the band)

should_intervene returns True only when score > hysteresis. Using
hysteresis > 1 prevents flapping when the trajectory sits near the band
edge — the repair loop only triggers on clear-outside-band measurements.
"""

from __future__ import annotations

import math

from fractaltrainer.target.target_shape import TargetShape


def divergence_score(current_dim: float, target: TargetShape) -> float:
    if math.isnan(current_dim):
        return float("inf")
    return abs(current_dim - target.dim_target) / target.tolerance


def should_intervene(score: float, target: TargetShape) -> bool:
    if math.isnan(score) or math.isinf(score):
        return True
    return score > target.hysteresis


def within_band(current_dim: float, target: TargetShape) -> bool:
    if math.isnan(current_dim) or math.isinf(current_dim):
        return False
    return target.band_low <= current_dim <= target.band_high
