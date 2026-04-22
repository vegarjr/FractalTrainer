"""Unit tests for Direction G — consolidation / distillation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fractaltrainer.integration.context_mlp import ContextAwareMLP
from fractaltrainer.integration.consolidation import (
    ConsolidatedDecision,
    ConsolidatedRouter,
    teacher_output,
    train_generalist,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry


def _dummy_loader(n: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = (x.flatten(1).sum(1) > 0).long()
    class L:
        def __iter__(self):
            for i in range(0, n, 8):
                yield x[i:i+8], y[i:i+8]
    return L()


def test_teacher_output_shape_and_mean():
    m1 = ContextAwareMLP()
    m2 = ContextAwareMLP()
    m3 = ContextAwareMLP()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(4, 1, 28, 28, generator=g)
    t = teacher_output([m1, m2, m3], x)
    assert t.shape == (4, 10)
    # Rows sum to 1 (averaged softmax distributions)
    assert torch.allclose(t.sum(dim=-1), torch.ones(4), atol=1e-4)


def test_teacher_output_weights_override():
    """Weights dominated by one model → teacher ≈ that model."""
    m_confident = ContextAwareMLP()
    m_other = ContextAwareMLP()
    g = torch.Generator().manual_seed(11)
    x = torch.randn(4, 1, 28, 28, generator=g)
    t_weighted = teacher_output([m_confident, m_other], x, weights=[1.0, 1e-9])
    t_solo = teacher_output([m_confident], x)
    assert torch.allclose(t_weighted, t_solo, atol=1e-4)


def test_train_generalist_reduces_loss():
    m1 = ContextAwareMLP()
    m2 = ContextAwareMLP()
    loader = _dummy_loader()
    g, stats = train_generalist([m1, m2], loader, n_steps=30, lr=0.01, seed=1)
    assert stats.n_steps == 30
    assert stats.loss_history[0] > stats.loss_history[-1]
    assert stats.final_loss >= 0


def test_consolidated_router_high_confidence_uses_generalist():
    reg = FractalRegistry()
    g_model = ContextAwareMLP()
    cr = ConsolidatedRouter(
        g_model, reg, model_by_entry={},
        confidence_threshold=0.0,  # always confident
    )
    x = torch.randn(1, 1, 28, 28)
    decision = cr.predict(x)
    assert decision.used_generalist is True
    assert decision.specialist_used is None


def test_consolidated_router_low_confidence_falls_through():
    """With threshold=1.0, generalist is never confident enough;
    specialist path should answer when a signature is provided and a
    matching entry exists."""
    reg = FractalRegistry()
    specialist = ContextAwareMLP()
    entry = FractalEntry(
        name="spec_a",
        signature=np.array([0.0] + [0.0] * 99),
        metadata={"task": "a", "task_labels": [0, 1]},
    )
    reg.add(entry)
    g_model = ContextAwareMLP()
    cr = ConsolidatedRouter(
        g_model, reg,
        model_by_entry={"spec_a": specialist},
        confidence_threshold=1.01,  # impossible to reach → always falls through
    )
    x = torch.randn(1, 1, 28, 28)
    sig = np.array([0.01] + [0.0] * 99)
    decision = cr.predict(x, query_signature=sig)
    assert decision.used_generalist is False
    assert decision.specialist_used == "spec_a"


def test_consolidated_router_no_signature_uses_generalist_even_low_conf():
    reg = FractalRegistry()
    g_model = ContextAwareMLP()
    cr = ConsolidatedRouter(
        g_model, reg, model_by_entry={},
        confidence_threshold=1.01,
    )
    x = torch.randn(1, 1, 28, 28)
    decision = cr.predict(x, query_signature=None)
    assert decision.used_generalist is True
    assert decision.specialist_used is None


def test_confidence_is_finite_and_in_unit_interval():
    g_model = ContextAwareMLP()
    reg = FractalRegistry()
    cr = ConsolidatedRouter(g_model, reg, {})
    x = torch.randn(1, 1, 28, 28)
    decision = cr.predict(x)
    assert 0.0 <= decision.confidence <= 1.0


def test_predict_batch_returns_decision_per_sample():
    g_model = ContextAwareMLP()
    reg = FractalRegistry()
    cr = ConsolidatedRouter(g_model, reg, {}, confidence_threshold=0.0)
    x = torch.randn(5, 1, 28, 28)
    decisions = cr.predict_batch(x)
    assert len(decisions) == 5
    assert all(isinstance(d, ConsolidatedDecision) for d in decisions)
