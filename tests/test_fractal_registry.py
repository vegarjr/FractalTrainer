"""Tests for FractalRegistry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fractaltrainer.registry import FractalEntry, FractalRegistry


def _sig(*vals) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64)


def test_add_and_len():
    r = FractalRegistry()
    assert len(r) == 0
    r.add(FractalEntry("a", _sig(1.0, 0.0, 0.0)))
    r.add(FractalEntry("b", _sig(0.0, 1.0, 0.0)))
    assert len(r) == 2
    assert "a" in r
    assert "b" in r
    assert "c" not in r


def test_add_overwrites_duplicate():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(1.0, 0.0)))
    r.add(FractalEntry("a", _sig(0.0, 1.0)))
    assert len(r) == 1
    # Second add wins
    got = r.get("a")
    assert got is not None
    assert got.signature[1] == 1.0


def test_remove():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(1.0)))
    removed = r.remove("a")
    assert removed is not None
    assert removed.name == "a"
    assert len(r) == 0
    assert r.remove("nonexistent") is None


def test_find_nearest_basic():
    r = FractalRegistry()
    r.add(FractalEntry("close", _sig(0.0, 0.0, 1.0)))
    r.add(FractalEntry("far", _sig(100.0, 100.0, 100.0)))
    res = r.find_nearest(_sig(0.0, 0.0, 0.9), k=1)
    assert res.nearest.name == "close"
    assert res.distances[0] < 1.0


def test_find_nearest_returns_k():
    r = FractalRegistry()
    for i in range(5):
        r.add(FractalEntry(f"e{i}", _sig(float(i), 0.0)))
    res = r.find_nearest(_sig(0.0, 0.0), k=3)
    assert len(res.entries) == 3
    assert res.entries[0].name == "e0"
    assert res.entries[1].name == "e1"
    assert res.entries[2].name == "e2"
    # Distances must be non-decreasing
    for i in range(len(res.distances) - 1):
        assert res.distances[i] <= res.distances[i + 1]


def test_find_nearest_exclude():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(0.0, 0.0)))
    r.add(FractalEntry("b", _sig(1.0, 1.0)))
    r.add(FractalEntry("c", _sig(2.0, 2.0)))
    res = r.find_nearest(_sig(0.0, 0.0), k=1, exclude=["a"])
    assert res.nearest.name == "b"


def test_find_nearest_empty_registry():
    r = FractalRegistry()
    res = r.find_nearest(_sig(0.0, 0.0), k=1)
    assert res.entries == []
    assert res.distances == []
    assert res.nearest is None


def test_find_nearest_k_clamped_to_registry_size():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(0.0)))
    res = r.find_nearest(_sig(0.0), k=10)
    assert len(res.entries) == 1


def test_incompatible_shape_silently_skipped():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(0.0, 0.0)))
    r.add(FractalEntry("b", _sig(0.0, 0.0, 0.0)))  # different dim
    res = r.find_nearest(_sig(0.0, 0.0), k=2)
    # Only the 2-d entry should have produced a distance
    assert len(res.entries) == 1
    assert res.entries[0].name == "a"


def test_composite_weights_inverse_distance():
    r = FractalRegistry()
    r.add(FractalEntry("close", _sig(0.0, 0.0)))
    r.add(FractalEntry("mid", _sig(1.0, 0.0)))
    r.add(FractalEntry("far", _sig(10.0, 0.0)))
    res, w = r.composite_weights(_sig(0.0, 0.0), k=3, temperature=1.0)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6
    # Closest should get the highest weight
    assert w[0] > w[1] > w[2]


def test_composite_weights_respects_k_and_exclude():
    r = FractalRegistry()
    for i in range(5):
        r.add(FractalEntry(f"e{i}", _sig(float(i), 0.0)))
    res, w = r.composite_weights(_sig(0.0, 0.0), k=2, exclude=["e0"])
    assert len(res.entries) == 2
    assert res.entries[0].name == "e1"
    assert res.entries[1].name == "e2"


def test_composite_weights_empty_registry():
    r = FractalRegistry()
    res, w = r.composite_weights(_sig(0.0, 0.0), k=3)
    assert w.size == 0
    assert res.entries == []


def test_save_load_roundtrip():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(1.0, 2.0, 3.0),
                        metadata={"task": "digit_class", "seed": 42}))
    r.add(FractalEntry("b", _sig(4.0, 5.0, 6.0),
                        metadata={"task": "parity", "seed": 101}))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "registry.json"
        r.save(p)
        loaded = FractalRegistry.load(p)
        assert len(loaded) == 2
        assert loaded.get("a").metadata["task"] == "digit_class"
        np.testing.assert_array_almost_equal(
            loaded.get("b").signature, _sig(4.0, 5.0, 6.0),
        )


def test_retrieval_result_to_dict():
    r = FractalRegistry()
    r.add(FractalEntry("x", _sig(0.0, 0.0),
                        metadata={"task": "t1"}))
    res = r.find_nearest(_sig(0.0, 0.1), k=1, query_name="probe")
    d = res.to_dict()
    assert d["query_name"] == "probe"
    assert d["ranked"][0]["name"] == "x"
    assert d["ranked"][0]["metadata"]["task"] == "t1"
    assert "distance" in d["ranked"][0]


def test_names_and_entries_return_list():
    r = FractalRegistry()
    r.add(FractalEntry("a", _sig(0.0)))
    r.add(FractalEntry("b", _sig(1.0)))
    names = r.names()
    assert isinstance(names, list)
    assert sorted(names) == ["a", "b"]
    entries = r.entries()
    assert isinstance(entries, list)
    assert len(entries) == 2
