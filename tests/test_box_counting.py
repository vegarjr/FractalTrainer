"""Tests for box_counting — standard box-counting dimension."""

import numpy as np
import pytest

from fractaltrainer.geometry.box_counting import box_counting_dim
from tests.fixtures import cantor_set_points, unit_square_points


def test_unit_square_fill_returns_approx_two():
    pts = unit_square_points(n_points=2000, seed=0)
    res = box_counting_dim(pts)
    assert 1.7 <= res.dim <= 2.1, f"square dim={res.dim:.3f} outside [1.7, 2.1]"
    assert res.r_squared > 0.9


def test_cantor_set_returns_approx_0_63():
    # Theoretical log(2)/log(3) ≈ 0.6309
    pts = cantor_set_points(n_points=3000, levels=10)
    res = box_counting_dim(pts)
    assert np.isfinite(res.dim)
    assert 0.45 <= res.dim <= 0.8, f"cantor dim={res.dim:.3f} outside [0.45, 0.8]"


def test_too_few_points_returns_nan_dim():
    pts = np.random.RandomState(0).randn(5, 2)
    res = box_counting_dim(pts, min_points=20)
    assert not np.isfinite(res.dim)
    assert res.error is not None


def test_shape_validation():
    with pytest.raises(ValueError):
        box_counting_dim(np.zeros(10))


def test_degenerate_input_handled():
    pts = np.ones((100, 2))
    res = box_counting_dim(pts)
    assert not np.isfinite(res.dim) or res.dim == 0.0
