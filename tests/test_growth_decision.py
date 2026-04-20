"""Tests for FractalRegistry.decide — the growth/routing decision."""

from __future__ import annotations

import numpy as np
import pytest

from fractaltrainer.registry import FractalEntry, FractalRegistry


def _sig(*vals) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64)


def _populate() -> FractalRegistry:
    r = FractalRegistry()
    # Place four clearly-separated entries in 2-d
    r.add(FractalEntry("a", _sig(0.0, 0.0), metadata={"task": "A"}))
    r.add(FractalEntry("b", _sig(10.0, 0.0), metadata={"task": "B"}))
    r.add(FractalEntry("c", _sig(0.0, 10.0), metadata={"task": "C"}))
    r.add(FractalEntry("d", _sig(10.0, 10.0), metadata={"task": "D"}))
    return r


def test_match_verdict():
    r = _populate()
    # Query very close to "a"
    d = r.decide(_sig(0.1, 0.1),
                  match_threshold=1.0, spawn_threshold=5.0)
    assert d.verdict == "match"
    assert d.retrieval is not None
    assert d.retrieval.nearest.name == "a"
    assert d.min_distance < 1.0


def test_compose_verdict_in_gap():
    r = _populate()
    # Query between match and spawn thresholds
    d = r.decide(_sig(3.0, 0.0),
                  match_threshold=1.0, spawn_threshold=5.0, compose_k=2)
    assert d.verdict == "compose"
    # Nearest should still be "a"; but because it's > match_threshold,
    # we're asked to blend
    assert d.retrieval.nearest.name == "a"
    assert d.composite_entries is not None
    assert len(d.composite_entries) == 2
    # Weights sum to 1
    w = d.composite_weights
    assert abs(sum(w) - 1.0) < 1e-9
    # Closest entry should get the largest weight
    assert w[0] >= w[1]


def test_spawn_verdict():
    r = _populate()
    # Query far from all entries
    d = r.decide(_sig(100.0, 100.0),
                  match_threshold=1.0, spawn_threshold=5.0)
    assert d.verdict == "spawn"
    assert d.min_distance > 5.0
    # Even on spawn, retrieval is populated (for context/logging)
    assert d.retrieval is not None


def test_empty_verdict():
    r = FractalRegistry()
    d = r.decide(_sig(0.0, 0.0),
                  match_threshold=1.0, spawn_threshold=5.0)
    assert d.verdict == "empty"
    assert d.min_distance is None


def test_decide_rejects_misordered_thresholds():
    r = _populate()
    with pytest.raises(ValueError):
        r.decide(_sig(0.0, 0.0),
                  match_threshold=5.0, spawn_threshold=1.0)


def test_decide_respects_exclude():
    r = _populate()
    # Exclude "a"; nearest becomes "b" at distance 10
    d = r.decide(_sig(0.1, 0.0),
                  match_threshold=1.0, spawn_threshold=5.0,
                  exclude=["a"])
    assert d.verdict == "spawn"
    assert d.retrieval.nearest.name == "b"


def test_decision_to_dict_roundtrip_shape():
    r = _populate()
    d = r.decide(_sig(3.0, 0.0),
                  match_threshold=1.0, spawn_threshold=5.0, compose_k=2)
    j = d.to_dict()
    assert j["verdict"] == "compose"
    assert j["min_distance"] is not None
    assert j["retrieval"] is not None
    assert len(j["composite_entries"]) == 2
    assert len(j["composite_weights"]) == 2


def test_match_beats_all_other_thresholds():
    r = _populate()
    # Exact hit — distance 0
    r.add(FractalEntry("exact", _sig(5.0, 5.0), metadata={"task": "E"}))
    d = r.decide(_sig(5.0, 5.0),
                  match_threshold=0.001, spawn_threshold=5.0)
    assert d.verdict == "match"
    assert d.retrieval.nearest.name == "exact"
    assert d.min_distance == 0.0


def test_compose_weights_higher_for_closer():
    r = _populate()
    d = r.decide(_sig(1.0, 0.0),
                  match_threshold=0.5, spawn_threshold=20.0, compose_k=4)
    assert d.verdict == "compose"
    w = d.composite_weights
    # "a" at distance 1 is closest, should get highest weight
    assert d.composite_entries[0].name == "a"
    assert w[0] == max(w)


def test_empty_with_only_excluded_entries():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(0.0, 0.0)))
    d = r.decide(_sig(0.0, 0.0),
                  match_threshold=1.0, spawn_threshold=5.0,
                  exclude=["a"])
    assert d.verdict == "empty"
