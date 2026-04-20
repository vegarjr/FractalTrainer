"""Tests for FractalRegistry.calibrate_thresholds (v3 Sprint 6)."""

from __future__ import annotations

import numpy as np
import pytest

from fractaltrainer.registry import (
    CalibrationResult,
    FractalEntry,
    FractalRegistry,
)


def _sig(*vals) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64)


def _make_registry(layout: dict[str, list[np.ndarray]]) -> FractalRegistry:
    """Build a registry from {task_name: [signature, ...]}."""
    r = FractalRegistry()
    for task, sigs in layout.items():
        for i, s in enumerate(sigs):
            r.add(FractalEntry(
                name=f"{task}_{i}", signature=s,
                metadata={"task": task, "seed": i},
            ))
    return r


def test_calibrate_basic_two_task_clean_gap():
    # Task A clustered near (0,0); task B clustered near (10,10).
    # Within-task distances ≈ 0.3-0.7; cross ≈ 13-15.
    r = _make_registry({
        "A": [_sig(0.0, 0.0), _sig(0.3, 0.0), _sig(0.0, 0.5)],
        "B": [_sig(10.0, 10.0), _sig(10.2, 10.3), _sig(9.8, 10.5)],
    })
    cal = r.calibrate_thresholds()
    assert isinstance(cal, CalibrationResult)
    assert cal.n_tasks == 2
    assert cal.overlap is False
    assert cal.match_threshold < cal.spawn_threshold
    # Within-task distances should all be < 1.0
    assert cal.within_task_distances.max() < 1.0
    # Cross-task distances should all be > 13
    assert cal.cross_task_distances.min() > 13.0


def test_calibrate_overlap_collapses_band():
    # Overlapping tasks: within-A goes up to 5, cross A-B starts at 3.
    r = _make_registry({
        "A": [_sig(0.0, 0.0), _sig(5.0, 0.0), _sig(0.0, 5.0)],
        "B": [_sig(3.0, 0.0), _sig(4.0, 1.0), _sig(3.5, 0.5)],
    })
    cal = r.calibrate_thresholds()
    # Raw percentiles likely cross — but even if they don't here, the
    # invariant to check is match ≤ spawn always.
    assert cal.match_threshold <= cal.spawn_threshold


def test_calibrate_overlap_detected():
    # Force clear overlap: all pairs ~similar scale.
    r = _make_registry({
        "A": [_sig(0.0), _sig(1.0), _sig(2.0), _sig(3.0)],
        "B": [_sig(1.5), _sig(2.5), _sig(0.5), _sig(3.5)],
    })
    cal = r.calibrate_thresholds(within_percentile=95.0, cross_percentile=5.0)
    # Within-A distances: {1,2,3,1,2,1} = up to 3. Cross A-B min ~= 0.5.
    # So within_p95 ~= 3, cross_p05 ~= 0.5 → overlap.
    assert cal.overlap is True
    # Clamp invariant holds
    assert cal.match_threshold == cal.spawn_threshold


def test_calibrate_custom_percentiles():
    r = _make_registry({
        "A": [_sig(0.0, 0.0), _sig(0.5, 0.0), _sig(0.0, 0.5)],
        "B": [_sig(10.0, 0.0), _sig(10.5, 0.0), _sig(10.0, 0.5)],
    })
    # More permissive match (75) and stricter spawn (25) both move
    # inward from defaults (95 / 5).
    cal_loose = r.calibrate_thresholds(
        within_percentile=75.0, cross_percentile=25.0)
    cal_strict = r.calibrate_thresholds(
        within_percentile=95.0, cross_percentile=5.0)
    # Stricter-within → lower match threshold
    assert cal_loose.match_threshold <= cal_strict.match_threshold
    # Stricter-cross (=5) → lower spawn threshold than 25th
    assert cal_strict.spawn_threshold <= cal_loose.spawn_threshold


def test_calibrate_single_task_rejected():
    r = _make_registry({
        "A": [_sig(0.0, 0.0), _sig(0.3, 0.0), _sig(0.0, 0.5)],
    })
    with pytest.raises(ValueError, match="at least 2 distinct tasks"):
        r.calibrate_thresholds()


def test_calibrate_missing_metadata_raises():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(0.0, 0.0), metadata={"task": "A"}))
    r.add(FractalEntry("b", _sig(0.3, 0.0), metadata={}))  # no task
    with pytest.raises(ValueError, match="missing metadata"):
        r.calibrate_thresholds()


def test_calibrate_empty_registry_rejected():
    r = FractalRegistry()
    with pytest.raises(ValueError, match="at least 2 registered entries"):
        r.calibrate_thresholds()


def test_calibrate_single_entry_rejected():
    r = _make_registry({"A": [_sig(0.0, 0.0)]})
    with pytest.raises(ValueError, match="at least 2 registered entries"):
        r.calibrate_thresholds()


def test_calibrate_custom_task_key():
    r = FractalRegistry()
    r.add(FractalEntry("a1", _sig(0.0, 0.0), metadata={"dataset": "X"}))
    r.add(FractalEntry("a2", _sig(0.3, 0.0), metadata={"dataset": "X"}))
    r.add(FractalEntry("b1", _sig(10.0, 0.0), metadata={"dataset": "Y"}))
    r.add(FractalEntry("b2", _sig(10.3, 0.0), metadata={"dataset": "Y"}))
    cal = r.calibrate_thresholds(task_key="dataset")
    assert cal.n_tasks == 2
    assert cal.overlap is False


def test_calibrate_result_to_dict_serializes():
    r = _make_registry({
        "A": [_sig(0.0), _sig(0.5), _sig(1.0)],
        "B": [_sig(10.0), _sig(10.5), _sig(11.0)],
    })
    cal = r.calibrate_thresholds()
    d = cal.to_dict()
    assert set(d.keys()) >= {
        "match_threshold", "spawn_threshold",
        "within_percentile", "cross_percentile",
        "overlap", "n_tasks",
        "within_distance_stats", "cross_distance_stats",
    }
    assert d["n_tasks"] == 2
    assert d["within_distance_stats"]["n"] == 6  # 2 tasks × C(3,2) = 6
    assert d["cross_distance_stats"]["n"] == 9   # 3 × 3


def test_calibrate_all_singleton_tasks_rejected():
    # 3 tasks, 1 entry each — no within-task pairs possible.
    r = _make_registry({
        "A": [_sig(0.0, 0.0)],
        "B": [_sig(10.0, 0.0)],
        "C": [_sig(20.0, 0.0)],
    })
    with pytest.raises(ValueError, match="No within-task pairs"):
        r.calibrate_thresholds()
