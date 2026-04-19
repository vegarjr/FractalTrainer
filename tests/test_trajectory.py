"""Tests for fractaltrainer.geometry.trajectory."""

import numpy as np
import pytest

from fractaltrainer.geometry.trajectory import trajectory_metrics


def test_straight_line_tortuosity_is_one():
    # 50 points along a straight line in 3-d
    t = np.linspace(0, 1, 50).reshape(-1, 1)
    trajectory = np.concatenate([t, 2 * t, 3 * t], axis=1)
    m = trajectory_metrics(trajectory)
    assert m["tortuosity"] == pytest.approx(1.0, abs=1e-6)
    assert m["total_path_length"] == pytest.approx(m["displacement"], rel=1e-6)


def test_circle_recurrence_rate_positive():
    # Points on a unit circle in 2-d sweep 2.5 revolutions → high recurrence
    n = 300
    theta = np.linspace(0, 5 * np.pi, n)
    trajectory = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    m = trajectory_metrics(trajectory)
    assert m["recurrence_rate"] > 0.3


def test_dispersion_of_cluster_small():
    # Tight Gaussian cluster → small dispersion
    rng = np.random.RandomState(0)
    trajectory = rng.randn(100, 4) * 0.01
    m = trajectory_metrics(trajectory)
    assert m["dispersion"] < 0.1


def test_shape_validation():
    with pytest.raises(ValueError):
        trajectory_metrics(np.zeros((10,)))  # 1-D should be rejected


def test_single_point_trajectory():
    # One point means no diffs and no movement. Should not crash.
    trajectory = np.array([[1.0, 2.0, 3.0]])
    m = trajectory_metrics(trajectory)
    assert m["total_path_length"] == 0.0
    assert m["displacement"] == 0.0


def test_zigzag_has_high_curvature():
    # Sharp zigzag → larger mean_curvature than a straight line
    n = 40
    x = np.arange(n)
    y = np.where(x % 2 == 0, 0, 1)
    trajectory = np.stack([x.astype(float), y.astype(float)], axis=1)

    t = np.linspace(0, 1, n).reshape(-1, 1)
    line = np.concatenate([t, 2 * t], axis=1)

    m_zig = trajectory_metrics(trajectory)
    m_line = trajectory_metrics(line)
    assert m_zig["mean_curvature"] > m_line["mean_curvature"]
