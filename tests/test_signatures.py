"""Unit tests for signature functions — softmax vs penultimate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fractaltrainer.integration.context_mlp import (
    ContextAwareMLP,
    N_CLASSES,
    PENULTIMATE_DIM,
)
from fractaltrainer.integration.signatures import (
    get_signature_fn,
    penultimate_normalized_signature,
    penultimate_signature,
    penultimate_softmax_signature,
    softmax_signature,
)


def _probe(batch: int = 100, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, 28, 28, generator=g)


def test_softmax_signature_shape():
    m = ContextAwareMLP()
    probe = _probe(batch=100)
    sig = softmax_signature(m, probe)
    assert sig.shape == (100 * N_CLASSES,)
    # softmax rows sum to 1
    assert np.allclose(sig.reshape(100, N_CLASSES).sum(axis=1), 1.0, atol=1e-5)


def test_penultimate_signature_shape():
    m = ContextAwareMLP()
    probe = _probe(batch=100)
    sig = penultimate_signature(m, probe)
    assert sig.shape == (100 * PENULTIMATE_DIM,)
    # penultimate is post-ReLU, so non-negative
    assert (sig >= 0).all()


def test_penultimate_signature_raises_without_method():
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(784, 10)
        def forward(self, x):
            if x.ndim > 2: x = x.view(x.size(0), -1)
            return self.fc(x)

    with pytest.raises(TypeError, match="has no .penultimate"):
        penultimate_signature(Plain(), _probe(batch=4))


def test_get_signature_fn_dispatch():
    assert get_signature_fn("softmax") is softmax_signature
    assert get_signature_fn("penultimate") is penultimate_signature
    assert get_signature_fn("penultimate_normalized") is penultimate_normalized_signature
    assert get_signature_fn("penultimate_softmax") is penultimate_softmax_signature
    with pytest.raises(ValueError, match="unknown signature mode"):
        get_signature_fn("bogus")


def test_penultimate_normalized_unit_norm():
    m = ContextAwareMLP()
    probe = _probe(batch=50)
    sig = penultimate_normalized_signature(m, probe)
    assert sig.shape == (50 * PENULTIMATE_DIM,)
    # Unit norm (within tiny numerical tolerance)
    n = float(np.linalg.norm(sig))
    assert abs(n - 1.0) < 1e-5


def test_penultimate_softmax_simplex_bounded():
    m = ContextAwareMLP()
    probe = _probe(batch=50)
    sig = penultimate_softmax_signature(m, probe)
    # Per-row sums to 1
    rows = sig.reshape(50, PENULTIMATE_DIM)
    assert np.allclose(rows.sum(axis=1), 1.0, atol=1e-5)
    # All non-negative
    assert (sig >= 0).all()


def test_signatures_are_deterministic():
    torch.manual_seed(7)
    m = ContextAwareMLP()
    probe = _probe(batch=16, seed=99)
    s1 = softmax_signature(m, probe)
    s2 = softmax_signature(m, probe)
    p1 = penultimate_signature(m, probe)
    p2 = penultimate_signature(m, probe)
    assert np.allclose(s1, s2)
    assert np.allclose(p1, p2)


def test_signatures_differ_across_models():
    torch.manual_seed(1)
    m1 = ContextAwareMLP()
    torch.manual_seed(2)
    m2 = ContextAwareMLP()
    probe = _probe(batch=50, seed=42)
    s1 = softmax_signature(m1, probe)
    s2 = softmax_signature(m2, probe)
    p1 = penultimate_signature(m1, probe)
    p2 = penultimate_signature(m2, probe)
    # Differently-initialised models should have distinct signatures
    assert np.linalg.norm(s1 - s2) > 0.1
    assert np.linalg.norm(p1 - p2) > 0.1


def test_signatures_stable_under_context_none():
    """penultimate_signature calls model.penultimate(x, context=None);
    softmax_signature calls model(x, context=None). Both should work
    on ContextAwareMLP even when context_scale is non-zero — the
    probe-signature invariant requires this.
    """
    torch.manual_seed(3)
    m = ContextAwareMLP(context_scale=2.5)
    probe = _probe(batch=20)
    # Should not raise
    s = softmax_signature(m, probe)
    p = penultimate_signature(m, probe)
    assert np.isfinite(s).all()
    assert np.isfinite(p).all()
