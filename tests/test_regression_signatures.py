"""Unit tests for Direction H — regression-probe signatures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fractaltrainer.integration.regression_signatures import (
    SineRegressor,
    make_probe_inputs,
    make_sine_task,
    regression_probe_signature,
    train_sine_regressor,
)


def test_make_probe_shape():
    p = make_probe_inputs(n=50)
    assert p.shape == (50, 1)
    # End-points
    assert abs(float(p[0, 0]) - (-np.pi)) < 1e-5
    assert abs(float(p[-1, 0]) - np.pi) < 1e-5


def test_sine_regressor_forward_shape():
    m = SineRegressor()
    x = torch.randn(8, 1)
    out = m(x)
    assert out.shape == (8,)


def test_train_sine_regressor_reduces_loss():
    """A trained freq=1 regressor should approximate sin(x) well."""
    m = train_sine_regressor(freq=1.0, n_steps=500, seed=0)
    x = make_probe_inputs(100)
    y_true = make_sine_task(1.0)(x)
    with torch.no_grad():
        y_pred = m(x)
    mse = float(((y_pred - y_true) ** 2).mean())
    assert mse < 0.05  # reasonable fit


def test_regression_signature_l2_unit_norm():
    m = train_sine_regressor(freq=1.0, n_steps=200, seed=0)
    probe = make_probe_inputs(50)
    sig = regression_probe_signature(m, probe, normalize="l2")
    assert sig.shape == (50,)
    assert abs(float(np.linalg.norm(sig)) - 1.0) < 1e-5


def test_regression_signature_zscore_mean0_std1():
    m = train_sine_regressor(freq=1.0, n_steps=200, seed=0)
    probe = make_probe_inputs(50)
    sig = regression_probe_signature(m, probe, normalize="zscore")
    assert abs(float(sig.mean())) < 1e-5
    assert abs(float(sig.std()) - 1.0) < 1e-4


def test_regression_signature_deterministic():
    m = train_sine_regressor(freq=2.0, n_steps=200, seed=0)
    probe = make_probe_inputs(30, seed=12)
    s1 = regression_probe_signature(m, probe, normalize="l2")
    s2 = regression_probe_signature(m, probe, normalize="l2")
    assert np.allclose(s1, s2)


def test_signatures_cluster_by_task_zscore():
    """Within-task z-score signatures are tighter than cross-task."""
    probe = make_probe_inputs(100)
    within: list[float] = []
    cross: list[float] = []
    # Two tasks, 3 seeds each
    sig_group1 = [
        regression_probe_signature(
            train_sine_regressor(freq=1.0, n_steps=300, seed=s), probe,
            normalize="zscore",
        )
        for s in (1, 2, 3)
    ]
    sig_group2 = [
        regression_probe_signature(
            train_sine_regressor(freq=3.0, n_steps=300, seed=s), probe,
            normalize="zscore",
        )
        for s in (1, 2, 3)
    ]
    for group in [sig_group1, sig_group2]:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                within.append(float(np.linalg.norm(group[i] - group[j])))
    for a in sig_group1:
        for b in sig_group2:
            cross.append(float(np.linalg.norm(a - b)))
    assert max(within) < min(cross)


def test_unknown_normalize_raises():
    m = train_sine_regressor(freq=1.0, n_steps=50, seed=0)
    probe = make_probe_inputs(10)
    with pytest.raises(ValueError, match="unknown normalize"):
        regression_probe_signature(m, probe, normalize="bogus")
