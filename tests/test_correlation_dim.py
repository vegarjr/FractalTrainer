"""Tests for correlation_dim — Grassberger-Procaccia.

Reference systems with known theoretical correlation dimensions anchor
the implementation. If these regress, something is wrong in the scaling-
range detection or the radius/quantile heuristics.
"""

import numpy as np
import pytest

from fractaltrainer.geometry.correlation_dim import correlation_dim
from tests.fixtures import (
    henon_points,
    lorenz_points,
    random_walk_high_d,
    unit_square_points,
)


def test_henon_correlation_dim_close_to_reference():
    # Theoretical D_2 ≈ 1.26
    pts = henon_points(n_points=2000)
    res = correlation_dim(pts, max_pairs=20_000, seed=0)
    assert np.isfinite(res.dim), f"non-finite dim: error={res.error!r}"
    assert res.r_squared > 0.95
    assert 1.1 <= res.dim <= 1.4, f"henon dim={res.dim:.3f} outside [1.1, 1.4]"


def test_lorenz_correlation_dim_close_to_reference():
    # Theoretical D_2 ≈ 2.05
    pts = lorenz_points(n_points=3000)
    res = correlation_dim(pts, max_pairs=30_000, seed=0)
    assert np.isfinite(res.dim)
    assert res.r_squared > 0.9
    assert 1.7 <= res.dim <= 2.3, f"lorenz dim={res.dim:.3f} outside [1.7, 2.3]"


def test_unit_square_fill_returns_approx_two():
    # Theoretical D_2 = 2. Finite-N GP on N=2000 uniform 2D points
    # commonly lands in [1.5, 2.1] due to boundary effects and scaling-
    # range dependence. We accept this band as the regression anchor.
    pts = unit_square_points(n_points=2000, seed=0)
    res = correlation_dim(pts, max_pairs=20_000, seed=0)
    assert 1.5 <= res.dim <= 2.2, f"square dim={res.dim:.3f} outside [1.5, 2.2]"


def test_high_d_random_walk_is_clearly_outside_target_band():
    # A 16-dim random walk should not satisfy the default target band
    # [1.2, 1.8]. The exact dim for N=500 is typically 1.8-2.5 — not
    # a clean saturation because a walk is a trajectory (not a point
    # cloud). The test is about divergence detection, not precise dim.
    pts = random_walk_high_d(n_points=500, n_dim=16, seed=0)
    res = correlation_dim(pts, max_pairs=20_000, seed=0)
    assert np.isfinite(res.dim)
    assert res.dim > 1.8, (
        f"high-d walk dim={res.dim:.3f} should exceed target-band upper "
        f"edge (1.8) — divergence detection would fail"
    )


def test_too_few_points_returns_nan_dim():
    pts = np.random.RandomState(0).randn(5, 2)
    res = correlation_dim(pts, min_points=20)
    assert not np.isfinite(res.dim)
    assert res.error is not None
    assert "min_points" in res.error


def test_shape_validation():
    with pytest.raises(ValueError):
        correlation_dim(np.zeros(10))  # 1-D rejected


def test_deterministic_under_seed():
    pts = henon_points(n_points=3000)  # large enough to force sampling
    a = correlation_dim(pts, max_pairs=5_000, seed=42)
    b = correlation_dim(pts, max_pairs=5_000, seed=42)
    assert a.dim == b.dim
    assert a.r_squared == b.r_squared


def test_zero_spread_handled_gracefully():
    pts = np.zeros((100, 3))
    res = correlation_dim(pts, min_points=20)
    assert not np.isfinite(res.dim)
    assert res.error is not None
