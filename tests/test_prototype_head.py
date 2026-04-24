"""Tests for PrototypeExpert — frozen encoder + per-class prototypes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fractaltrainer.integration.hybrid_head import MLPEncoderTrunk
from fractaltrainer.integration.prototype_head import (
    PrototypeExpert, compose_prototypes,
)
from fractaltrainer.integration.signatures import softmax_signature


def _make_encoder(seed=0):
    torch.manual_seed(seed)
    return MLPEncoderTrunk()


def test_fit_returns_prototype_expert():
    enc = _make_encoder()
    torch.manual_seed(0)
    sx = torch.randn(15, 784)
    sy = torch.tensor([0]*5 + [1]*5 + [2]*5)
    pe = PrototypeExpert.fit(enc, sx, sy, n_way=3)
    assert isinstance(pe, PrototypeExpert)
    assert pe.prototypes.shape == (3, 32)


def test_forward_shape_and_logits_negative_distance():
    enc = _make_encoder()
    sx = torch.randn(10, 784)
    sy = torch.tensor([0]*5 + [1]*5)
    pe = PrototypeExpert.fit(enc, sx, sy, n_way=2, temperature=1.0)
    qx = torch.randn(7, 784)
    logits = pe.forward(qx)
    assert logits.shape == (7, 2)
    # Logits are negative distances; check sign
    assert (logits <= 0).all() or logits.max() <= 1e-4


def test_penultimate_matches_encoder_output():
    enc = _make_encoder()
    sx = torch.randn(10, 784)
    sy = torch.tensor([0]*5 + [1]*5)
    pe = PrototypeExpert.fit(enc, sx, sy, n_way=2)
    x = torch.randn(4, 784)
    pen = pe.penultimate(x)
    enc.eval()
    with torch.no_grad():
        expected = enc(x)
    assert torch.allclose(pen, expected, atol=1e-6)


def test_predict_matches_argmin_distance():
    """Sanity check: predict == argmax of forward == argmin of distance."""
    enc = _make_encoder()
    sx = torch.randn(10, 784)
    sy = torch.tensor([0]*5 + [1]*5)
    pe = PrototypeExpert.fit(enc, sx, sy, n_way=2)
    qx = torch.randn(5, 784)
    enc.eval()
    with torch.no_grad():
        qe = enc(qx)
    expected = torch.cdist(qe, pe.prototypes).argmin(dim=1)
    pred = pe.predict(qx)
    assert torch.equal(pred, expected)


def test_signature_compatible_with_softmax_signature():
    enc = _make_encoder()
    sx = torch.randn(10, 784)
    sy = torch.tensor([0]*5 + [1]*5)
    pe = PrototypeExpert.fit(enc, sx, sy, n_way=2)
    probe = torch.randn(20, 784)
    sig = softmax_signature(pe, probe)
    assert sig.shape == (20 * 2,)
    assert np.isfinite(sig).all()


def test_fit_requires_every_class():
    enc = _make_encoder()
    sx = torch.randn(10, 784)
    sy = torch.tensor([0]*5 + [1]*5)  # missing class 2 for 3-way
    try:
        PrototypeExpert.fit(enc, sx, sy, n_way=3)
    except ValueError as e:
        assert "has no support" in str(e)
    else:
        raise AssertionError("expected ValueError for missing class")


def test_high_accuracy_on_well_separated_synthetic():
    """Strong test: when support comes from very separated Gaussians,
    the prototype classifier should be near-perfect."""
    enc = _make_encoder()
    rng = np.random.default_rng(0)
    # 3 Gaussian clusters well separated
    centers = rng.standard_normal((3, 784)) * 3
    sup_x, sup_y, qry_x, qry_y = [], [], [], []
    for k in range(3):
        for _ in range(5):
            sup_x.append(centers[k] + rng.standard_normal(784) * 0.1); sup_y.append(k)
        for _ in range(15):
            qry_x.append(centers[k] + rng.standard_normal(784) * 0.1); qry_y.append(k)
    sx = torch.from_numpy(np.stack(sup_x)).float()
    sy = torch.tensor(sup_y, dtype=torch.long)
    qx = torch.from_numpy(np.stack(qry_x)).float()
    qy = torch.tensor(qry_y, dtype=torch.long)
    pe = PrototypeExpert.fit(enc, sx, sy, n_way=3)
    acc = pe.score(qx, qy)
    assert acc > 0.90, f"expected >0.9 on separated clusters, got {acc}"


def test_compose_prototypes_uniform():
    enc = _make_encoder()
    torch.manual_seed(0)
    experts = []
    for i in range(3):
        sx = torch.randn(10, 784)
        sy = torch.tensor([0]*5 + [1]*5)
        experts.append(PrototypeExpert.fit(enc, sx, sy, n_way=2))
    blended = compose_prototypes(experts)
    assert blended.shape == (2, 32)
    expected = torch.stack([e.prototypes for e in experts], dim=0).mean(0)
    assert torch.allclose(blended, expected, atol=1e-6)


def test_compose_prototypes_weighted():
    enc = _make_encoder()
    sx = torch.randn(10, 784)
    sy = torch.tensor([0]*5 + [1]*5)
    e1 = PrototypeExpert.fit(enc, sx, sy, n_way=2)
    sx2 = sx + 5.0
    e2 = PrototypeExpert.fit(enc, sx2, sy, n_way=2)
    w = np.array([1.0, 3.0])  # strongly weights e2
    blended = compose_prototypes([e1, e2], weights=w)
    expected = (1 * e1.prototypes + 3 * e2.prototypes) / 4
    assert torch.allclose(blended, expected, atol=1e-6)
