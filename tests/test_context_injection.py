"""Unit tests for context injection — gather_context, ContextSpec, random_context."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fractaltrainer.integration.context_injection import (
    ContextSpec,
    gather_context,
    random_context,
)
from fractaltrainer.integration.context_mlp import (
    ContextAwareMLP,
    PENULTIMATE_DIM,
)


class _ConstPenultimateModel(torch.nn.Module):
    """Test stub whose penultimate() always returns a fixed tensor."""

    def __init__(self, value: float, batch: int = 8):
        super().__init__()
        self.value = float(value)
        self.batch = batch

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0) if x.ndim >= 1 else self.batch
        return torch.full((b, PENULTIMATE_DIM), self.value)


def _probe(batch: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, 28, 28, generator=g)


def test_gather_context_shape_k3_weighted_mean():
    models = [ContextAwareMLP() for _ in range(3)]
    probe = _probe(batch=8)
    out = gather_context(models, probe, ContextSpec(k=3), distances=[1.0, 2.0, 3.0])
    assert out.shape == (8, PENULTIMATE_DIM)


def test_gather_context_shape_uniform_mean():
    models = [ContextAwareMLP() for _ in range(4)]
    probe = _probe(batch=5)
    out = gather_context(
        models, probe, ContextSpec(k=4, aggregation="uniform_mean")
    )
    assert out.shape == (5, PENULTIMATE_DIM)


def test_gather_context_k_clamped_to_available():
    models = [ContextAwareMLP() for _ in range(2)]
    probe = _probe()
    out = gather_context(models, probe, ContextSpec(k=10))
    assert out.shape == (8, PENULTIMATE_DIM)


def test_gather_context_empty_neighbors_returns_zeros():
    out = gather_context([], _probe(batch=4), ContextSpec(k=3))
    assert out.shape == (4, PENULTIMATE_DIM)
    assert torch.all(out == 0)


def test_weighted_mean_zero_distance_dominates():
    """With distances [0, 10, 10], weights after softmax(-d) concentrate
    almost entirely on the first neighbor. Output should be ≈ first
    model's activation value.
    """
    m_a = _ConstPenultimateModel(value=1.0)
    m_b = _ConstPenultimateModel(value=0.0)
    m_c = _ConstPenultimateModel(value=0.0)
    probe = _probe(batch=4)
    out = gather_context(
        [m_a, m_b, m_c], probe,
        ContextSpec(k=3, aggregation="weighted_mean", temperature=1.0),
        distances=[0.0, 10.0, 10.0],
    )
    assert torch.allclose(out, torch.ones_like(out), atol=1e-4)


def test_weighted_mean_equal_distances_equals_uniform():
    """Equal distances → uniform weights → weighted_mean == uniform_mean."""
    models = [_ConstPenultimateModel(v) for v in (1.0, 2.0, 3.0)]
    probe = _probe()
    w = gather_context(models, probe, ContextSpec(k=3, aggregation="weighted_mean"),
                       distances=[5.0, 5.0, 5.0])
    u = gather_context(models, probe, ContextSpec(k=3, aggregation="uniform_mean"))
    assert torch.allclose(w, u)
    # And both should equal the mean of [1.0, 2.0, 3.0] = 2.0 everywhere
    assert torch.allclose(w, torch.full_like(w, 2.0))


def test_weighted_mean_without_distances_falls_back_to_uniform():
    models = [_ConstPenultimateModel(v) for v in (1.0, 3.0)]
    probe = _probe()
    out = gather_context(models, probe, ContextSpec(k=2, aggregation="weighted_mean"),
                         distances=None)
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_penultimate_fn_used_when_no_method():
    """A plain Sequential without .penultimate() should work via penultimate_fn."""
    seq = torch.nn.Sequential(
        torch.nn.Linear(784, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )

    def extract(m, x):
        x = x.view(x.size(0), -1) if x.ndim > 2 else x
        return m[:-1](x)

    probe = _probe(batch=3)
    out = gather_context([seq], probe, ContextSpec(k=1), penultimate_fn=extract)
    assert out.shape == (3, PENULTIMATE_DIM)


def test_random_context_shape_and_determinism():
    a = random_context(batch_size=16, seed=42)
    b = random_context(batch_size=16, seed=42)
    c = random_context(batch_size=16, seed=43)
    assert a.shape == (16, PENULTIMATE_DIM)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_wrong_shape_penultimate_raises():
    class Bad(torch.nn.Module):
        def penultimate(self, x):
            return torch.zeros(x.size(0), 7)  # wrong dim

    with pytest.raises(ValueError, match="expected penultimate"):
        gather_context([Bad()], _probe(), ContextSpec(k=1))
