"""Tests for geometry.meta_trajectory — shape + convergence over repair runs."""

from __future__ import annotations

import numpy as np
import pytest

from fractaltrainer.geometry.meta_trajectory import (
    MetaTrajectorySummary,
    summarize_meta_trajectory,
)
from fractaltrainer.observer.repair_history import MetaTrajectory


def _make_meta(states: np.ndarray, div: list[float]) -> MetaTrajectory:
    return MetaTrajectory(
        states=states,
        hparams_sequence=[{} for _ in range(states.shape[0])],
        iterations=list(range(states.shape[0])),
        statuses=["accepted"] * max(0, states.shape[0] - 1),
        divergence_scores=div,
    )


def test_empty_meta_summary_all_none():
    mt = _make_meta(np.zeros((0, 6)), [])
    s = summarize_meta_trajectory(mt)
    assert s.n_points == 0
    assert s.n_transitions == 0
    assert s.geometry == {}
    assert s.correlation_dim is None
    assert s.convergence["start_divergence"] is None


def test_single_point_no_geometry():
    mt = _make_meta(np.zeros((1, 6)), [5.0])
    s = summarize_meta_trajectory(mt)
    assert s.n_points == 1
    assert s.geometry == {}
    # single-point convergence signature reports start only
    assert s.convergence["start_divergence"] == 5.0
    assert s.convergence["end_divergence"] is None


def test_two_points_geometry_present_but_dim_skipped():
    states = np.array([[0.0] * 6, [1.0] * 6])
    mt = _make_meta(states, [3.0, 1.0])
    s = summarize_meta_trajectory(mt, min_points_for_dim=20)
    assert s.n_points == 2
    assert s.geometry  # has keys like total_path_length
    assert s.correlation_dim is None  # below min_points_for_dim


def test_convergence_monotonic_decrease():
    states = np.array([[float(i)] * 6 for i in range(5)])
    mt = _make_meta(states, [5.0, 3.0, 2.0, 1.5, 1.0])
    s = summarize_meta_trajectory(mt)
    assert s.convergence["monotonic_decreasing"] is True
    assert s.convergence["bounces"] == 0
    assert s.convergence["total_reduction"] == pytest.approx(4.0)
    assert s.convergence["end_divergence"] == pytest.approx(1.0)


def test_convergence_bouncing_non_monotonic():
    states = np.array([[float(i)] * 6 for i in range(5)])
    # div goes down, up, down, up — 3 direction changes
    mt = _make_meta(states, [5.0, 3.0, 4.0, 2.0, 3.0])
    s = summarize_meta_trajectory(mt)
    assert s.convergence["monotonic_decreasing"] is False
    assert s.convergence["bounces"] >= 1


def test_dim_computed_when_enough_points():
    # 40 points on a 1-d line (dim should be ~1) in 6-d embedding
    ts = np.linspace(0, 1, 40).reshape(-1, 1)
    states = np.concatenate([ts, ts * 2, ts * 3, ts * 4, ts * 5, ts * 6],
                            axis=1)
    mt = _make_meta(states, [float(i) for i in range(40, 0, -1)])
    s = summarize_meta_trajectory(mt, min_points_for_dim=20)
    assert s.correlation_dim is not None
    # A 1-d line should report dim close to 1 (finite-N bias ok)
    assert 0.7 <= s.correlation_dim.dim <= 1.3


def test_summary_serializes_cleanly():
    states = np.array([[0.0] * 6, [1.0] * 6, [2.0] * 6])
    mt = _make_meta(states, [3.0, 2.0, 1.0])
    s = summarize_meta_trajectory(mt)
    d = s.to_dict()
    assert set(d.keys()) >= {"n_points", "n_transitions", "geometry",
                              "correlation_dim", "convergence"}
