"""Unit tests for ContextAwareMLP."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fractaltrainer.integration.context_mlp import (
    ContextAwareMLP,
    HIDDEN_DIM,
    N_CLASSES,
    PENULTIMATE_DIM,
    baseline_mlp_forward,
)


def _seeded_input(batch: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, 28, 28, generator=g)


def test_forward_shape_without_context():
    m = ContextAwareMLP()
    x = _seeded_input()
    out = m(x)
    assert out.shape == (8, N_CLASSES)


def test_forward_shape_with_context():
    m = ContextAwareMLP(context_dim=PENULTIMATE_DIM)
    x = _seeded_input()
    ctx = torch.randn(8, PENULTIMATE_DIM)
    out = m(x, context=ctx)
    assert out.shape == (8, N_CLASSES)


def test_penultimate_shape():
    m = ContextAwareMLP()
    x = _seeded_input()
    p = m.penultimate(x)
    assert p.shape == (8, PENULTIMATE_DIM)


def test_context_scale_zero_matches_baseline():
    torch.manual_seed(42)
    m = ContextAwareMLP(context_scale=0.0)
    x = _seeded_input()
    ctx = torch.randn(8, PENULTIMATE_DIM) * 10  # any context, should be ignored
    out_ctx = m(x, context=ctx)
    out_none = m(x, context=None)
    out_baseline = baseline_mlp_forward(m, x)

    assert torch.allclose(out_ctx, out_none)
    assert torch.allclose(out_ctx, out_baseline)


def test_context_none_equals_baseline_even_when_scale_nonzero():
    torch.manual_seed(1)
    m = ContextAwareMLP(context_scale=1.0)
    x = _seeded_input()
    out_none = m(x, context=None)
    out_baseline = baseline_mlp_forward(m, x)
    # When context=None, the context lane is skipped entirely —
    # forward must match the baseline MLP exactly.
    assert torch.allclose(out_none, out_baseline)


def test_context_changes_output_when_scale_nonzero():
    torch.manual_seed(7)
    m = ContextAwareMLP(context_scale=1.0)
    x = _seeded_input()
    ctx = torch.randn(8, PENULTIMATE_DIM) * 3.0
    out_with = m(x, context=ctx)
    out_none = m(x, context=None)
    # Non-zero context with non-zero scale must shift the output.
    assert not torch.allclose(out_with, out_none, atol=1e-4)


def test_gradient_flows_through_context_lane():
    torch.manual_seed(0)
    m = ContextAwareMLP(context_scale=1.0)
    x = _seeded_input()
    ctx = torch.randn(8, PENULTIMATE_DIM, requires_grad=False)
    y = torch.zeros(8, dtype=torch.long)
    logits = m(x, context=ctx)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    assert m.ctx_proj.weight.grad is not None
    # Expect some non-zero gradient magnitude on the context lane.
    assert m.ctx_proj.weight.grad.abs().sum().item() > 0


def test_context_dim_zero_disables_lane():
    m = ContextAwareMLP(context_dim=0)
    assert m.ctx_proj is None
    assert m.ctx_norm is None
    x = _seeded_input()
    out = m(x, context=None)
    assert out.shape == (8, N_CLASSES)


def test_probe_signature_invariant_under_context_zero():
    """Signature is defined as softmax on a probe batch with context=None
    (see registry docstring). A context-trained expert probed without
    context must produce a signature in the same space as a legacy
    expert, or routing breaks.
    """
    torch.manual_seed(99)
    m = ContextAwareMLP(context_scale=1.0)
    probe = _seeded_input(batch=100, seed=12345)

    import torch.nn.functional as F

    with torch.no_grad():
        sig_no_ctx = F.softmax(m(probe), dim=1).flatten().cpu().numpy()
    assert sig_no_ctx.shape == (100 * N_CLASSES,)
    assert np.isfinite(sig_no_ctx).all()
    # Softmax rows sum to 1
    assert np.allclose(sig_no_ctx.reshape(100, N_CLASSES).sum(axis=1), 1.0, atol=1e-5)


def test_penultimate_consistent_with_forward():
    """penultimate(x) + fc3 must equal forward(x)."""
    torch.manual_seed(3)
    m = ContextAwareMLP()
    x = _seeded_input()
    with torch.no_grad():
        p = m.penultimate(x)
        out_via_pen = m.fc3(p)
        out_direct = m(x)
    assert torch.allclose(out_via_pen, out_direct)
