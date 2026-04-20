"""Tests for HierarchicalRegistry (v3 Sprint 8)."""

from __future__ import annotations

import numpy as np
import pytest

from fractaltrainer.registry import (
    FractalEntry,
    FractalRegistry,
    HierarchicalDecision,
    HierarchicalRegistry,
)


def _sig(*vals) -> np.ndarray:
    return np.asarray(vals, dtype=np.float64)


def _entry(name: str, sig: np.ndarray, dataset: str,
           **extra) -> FractalEntry:
    meta = {"dataset": dataset}
    meta.update(extra)
    return FractalEntry(name=name, signature=sig, metadata=meta)


def test_add_two_datasets_routes_to_nearest_cluster():
    h = HierarchicalRegistry()
    # Cluster A near (0, 0)
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("a2", _sig(0.2, 0.1), dataset="A"))
    # Cluster B near (10, 10)
    h.add(_entry("b1", _sig(10.0, 10.0), dataset="B"))
    h.add(_entry("b2", _sig(10.2, 9.9), dataset="B"))

    # Query near cluster A
    d = h.find_nearest_hierarchical(_sig(0.1, 0.1))
    assert isinstance(d, HierarchicalDecision)
    assert d.chosen_dataset == "A"
    assert d.retrieval.nearest is not None
    assert d.retrieval.nearest.name in {"a1", "a2"}

    # Query near cluster B
    d = h.find_nearest_hierarchical(_sig(10.1, 10.1))
    assert d.chosen_dataset == "B"
    assert d.retrieval.nearest.name in {"b1", "b2"}


def test_missing_dataset_metadata_raises():
    h = HierarchicalRegistry()
    bad_entry = FractalEntry(
        name="x", signature=_sig(0.0, 0.0),
        metadata={},  # no dataset key
    )
    with pytest.raises(ValueError, match="missing metadata"):
        h.add(bad_entry)


def test_custom_dataset_key():
    h = HierarchicalRegistry(dataset_key="modality")
    h.add(FractalEntry(
        name="a", signature=_sig(0.0, 0.0),
        metadata={"modality": "vision"},
    ))
    h.add(FractalEntry(
        name="b", signature=_sig(10.0, 10.0),
        metadata={"modality": "audio"},
    ))
    assert set(h.datasets()) == {"vision", "audio"}
    d = h.find_nearest_hierarchical(_sig(0.1, 0.0))
    assert d.chosen_dataset == "vision"


def test_empty_registry_raises_on_query():
    h = HierarchicalRegistry()
    with pytest.raises(ValueError, match="empty"):
        h.find_nearest_hierarchical(_sig(0.0, 0.0))


def test_len_sums_across_datasets():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("a2", _sig(0.1, 0.0), dataset="A"))
    h.add(_entry("b1", _sig(10.0, 10.0), dataset="B"))
    assert len(h) == 3
    assert len(h.sub_registry("A")) == 2
    assert len(h.sub_registry("B")) == 1


def test_centroid_updates_on_add():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    c1 = h.centroid("A")
    np.testing.assert_allclose(c1, [0.0, 0.0])

    h.add(_entry("a2", _sig(2.0, 0.0), dataset="A"))
    c2 = h.centroid("A")
    np.testing.assert_allclose(c2, [1.0, 0.0])

    h.add(_entry("a3", _sig(4.0, 0.0), dataset="A"))
    c3 = h.centroid("A")
    np.testing.assert_allclose(c3, [2.0, 0.0])


def test_single_dataset_always_routes_there():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="only"))
    h.add(_entry("a2", _sig(1.0, 0.0), dataset="only"))
    # Query far from centroid — still routes to the only dataset
    d = h.find_nearest_hierarchical(_sig(100.0, 100.0))
    assert d.chosen_dataset == "only"
    assert d.retrieval.nearest is not None


def test_exclude_propagates_to_subregistry():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("a2", _sig(0.1, 0.0), dataset="A"))
    h.add(_entry("a3", _sig(0.3, 0.0), dataset="A"))
    # Exclude a1 and a2 — only a3 should be available
    d = h.find_nearest_hierarchical(_sig(0.05, 0.0), exclude=["a1", "a2"])
    assert d.chosen_dataset == "A"
    assert d.retrieval.nearest.name == "a3"


def test_remove_by_name_cleans_up_empty_dataset():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("a2", _sig(1.0, 0.0), dataset="A"))
    h.add(_entry("b1", _sig(10.0, 0.0), dataset="B"))
    assert set(h.datasets()) == {"A", "B"}
    # Remove b1 — dataset B should be cleaned up
    removed = h.remove("b1")
    assert removed is not None
    assert removed.name == "b1"
    assert set(h.datasets()) == {"A"}
    # Removing a non-existent name returns None
    assert h.remove("nonexistent") is None


def test_remove_preserves_non_empty_dataset_and_recomputes_centroid():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("a2", _sig(2.0, 0.0), dataset="A"))
    h.add(_entry("a3", _sig(4.0, 0.0), dataset="A"))
    np.testing.assert_allclose(h.centroid("A"), [2.0, 0.0])
    h.remove("a3")
    assert len(h.sub_registry("A")) == 2
    np.testing.assert_allclose(h.centroid("A"), [1.0, 0.0])


def test_all_centroid_distances_reported():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("b1", _sig(10.0, 0.0), dataset="B"))
    h.add(_entry("c1", _sig(20.0, 0.0), dataset="C"))
    d = h.find_nearest_hierarchical(_sig(0.0, 0.0))
    assert d.chosen_dataset == "A"
    assert set(d.all_centroid_distances.keys()) == {"A", "B", "C"}
    # A has smallest centroid distance, C largest
    assert d.all_centroid_distances["A"] < d.all_centroid_distances["B"]
    assert d.all_centroid_distances["B"] < d.all_centroid_distances["C"]


def test_decision_to_dict_serializes():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("b1", _sig(10.0, 0.0), dataset="B"))
    d = h.find_nearest_hierarchical(_sig(0.1, 0.0))
    obj = d.to_dict()
    assert obj["chosen_dataset"] == "A"
    assert obj["centroid_distance"] >= 0
    assert set(obj["all_centroid_distances"].keys()) == {"A", "B"}
    assert obj["retrieval"]["ranked"][0]["name"] == "a1"


def test_contains_across_sub_registries():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("b1", _sig(10.0, 0.0), dataset="B"))
    assert "a1" in h
    assert "b1" in h
    assert "c1" not in h


def test_all_entries_flattens():
    h = HierarchicalRegistry()
    h.add(_entry("a1", _sig(0.0, 0.0), dataset="A"))
    h.add(_entry("a2", _sig(1.0, 0.0), dataset="A"))
    h.add(_entry("b1", _sig(10.0, 0.0), dataset="B"))
    names = sorted(e.name for e in h.all_entries())
    assert names == ["a1", "a2", "b1"]
