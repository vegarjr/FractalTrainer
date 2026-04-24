"""Tests for HybridMLP: meta-trained frozen encoder + trainable head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fractaltrainer.integration.context_mlp import ContextAwareMLP
from fractaltrainer.integration.hybrid_head import (
    MLPEncoderTrunk,
    build_hybrid_expert,
    load_pretrained_encoder,
    trainable_parameters,
)


def test_encoder_trunk_forward_shape():
    enc = MLPEncoderTrunk()
    x = torch.randn(5, 784)
    h = enc(x)
    assert h.shape == (5, 32)


def test_encoder_trunk_accepts_4d_input():
    """Auto-flattens 4-d image inputs."""
    enc = MLPEncoderTrunk()
    x = torch.randn(3, 1, 28, 28)
    h = enc(x)
    assert h.shape == (3, 32)


def test_load_pretrained_encoder_copies_weights():
    enc = MLPEncoderTrunk()
    # Set encoder to known non-zero weights
    with torch.no_grad():
        enc.fc1.weight.fill_(0.123)
        enc.fc1.bias.fill_(-0.456)
        enc.fc2.weight.fill_(0.789)
        enc.fc2.bias.fill_(1.234)

    model = ContextAwareMLP(n_classes=5)
    load_pretrained_encoder(model, enc, freeze=True)

    assert torch.allclose(model.fc1.weight, torch.full_like(model.fc1.weight, 0.123))
    assert torch.allclose(model.fc1.bias, torch.full_like(model.fc1.bias, -0.456))
    assert torch.allclose(model.fc2.weight, torch.full_like(model.fc2.weight, 0.789))
    assert torch.allclose(model.fc2.bias, torch.full_like(model.fc2.bias, 1.234))


def test_load_pretrained_encoder_freezes_fc1_fc2():
    enc = MLPEncoderTrunk()
    model = ContextAwareMLP(n_classes=5)
    load_pretrained_encoder(model, enc, freeze=True)

    # fc1, fc2 are frozen
    assert all(not p.requires_grad for p in model.fc1.parameters())
    assert all(not p.requires_grad for p in model.fc2.parameters())
    # fc3 (head) is trainable
    assert all(p.requires_grad for p in model.fc3.parameters())
    # Context lane is trainable
    assert all(p.requires_grad for p in model.ctx_proj.parameters())
    assert all(p.requires_grad for p in model.ctx_norm.parameters())


def test_build_hybrid_expert_default():
    enc = MLPEncoderTrunk()
    model = build_hybrid_expert(enc, n_classes=5)
    assert isinstance(model, ContextAwareMLP)
    assert model.n_classes == 5
    assert not any(p.requires_grad for p in model.fc1.parameters())
    assert any(p.requires_grad for p in model.fc3.parameters())


def test_trainable_parameters_excludes_frozen():
    enc = MLPEncoderTrunk()
    model = build_hybrid_expert(enc, n_classes=5)
    tps = trainable_parameters(model)
    # Should not include any tensor that shares storage with fc1/fc2
    frozen_ids = {id(p) for p in list(model.fc1.parameters()) + list(model.fc2.parameters())}
    assert not any(id(p) in frozen_ids for p in tps)
    # Should include fc3, ctx_proj, ctx_norm
    expected_ids = {id(p)
                    for p in list(model.fc3.parameters())
                    + list(model.ctx_proj.parameters())
                    + list(model.ctx_norm.parameters())}
    assert set(id(p) for p in tps) == expected_ids


def test_head_trains_while_encoder_stays_frozen():
    """After a few gradient steps, fc1/fc2 weights must be unchanged;
    fc3 must have changed."""
    enc = MLPEncoderTrunk()
    torch.manual_seed(0)
    model = build_hybrid_expert(enc, n_classes=3)
    fc1_before = model.fc1.weight.detach().clone()
    fc2_before = model.fc2.weight.detach().clone()
    fc3_before = model.fc3.weight.detach().clone()

    opt = torch.optim.Adam(trainable_parameters(model), lr=0.05)
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((20, 784))).float()
    y = torch.from_numpy(rng.integers(0, 3, size=20)).long()
    for _ in range(30):
        opt.zero_grad()
        loss = F.cross_entropy(model(X, context=None), y)
        loss.backward(); opt.step()

    assert torch.allclose(model.fc1.weight, fc1_before), "fc1 should not have changed"
    assert torch.allclose(model.fc2.weight, fc2_before), "fc2 should not have changed"
    assert not torch.allclose(model.fc3.weight, fc3_before), "fc3 should have changed"


def test_forward_matches_baseline_with_no_context():
    enc = MLPEncoderTrunk()
    model = build_hybrid_expert(enc, n_classes=5)
    from fractaltrainer.integration.context_mlp import baseline_mlp_forward
    x = torch.randn(4, 784)
    out1 = model(x, context=None)
    out2 = baseline_mlp_forward(model, x)
    assert torch.allclose(out1, out2, atol=1e-5)
