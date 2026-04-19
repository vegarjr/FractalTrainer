"""Tests for the vendored+adapted fractal_summary module (baseline metrics).

These aren't the primary metric; they complement correlation_dim. But they
should still return sane values on known inputs.
"""

import numpy as np

from fractaltrainer.geometry.fractal_summary import (
    compression_ratio,
    fractal_metrics,
    motif_analysis,
    self_similarity_score,
)


def test_compression_ratio_lower_for_repetitive_trajectory():
    n = 400
    repetitive = np.tile(np.array([[0.0, 0.0], [1.0, 0.0]]), (n // 2, 1))
    rng = np.random.RandomState(0)
    noisy = rng.randn(n, 2)
    ratio_rep = compression_ratio(repetitive)
    ratio_noise = compression_ratio(noisy)
    assert ratio_rep < ratio_noise, (
        f"repetitive ratio {ratio_rep:.3f} should be lower than noisy {ratio_noise:.3f}"
    )


def test_motif_analysis_repetitive_has_lower_unique_ratio():
    n = 200
    repetitive = np.tile(np.arange(4).reshape(-1, 1).astype(float), (n // 4, 1))
    rng = np.random.RandomState(1)
    noisy = rng.randn(n, 1)
    m_rep = motif_analysis(repetitive)
    m_noise = motif_analysis(noisy)
    assert m_rep["motif_5_unique_ratio"] < m_noise["motif_5_unique_ratio"]


def test_self_similarity_score_higher_for_exactly_repeating_signal():
    # A signal that exactly repeats 8 times in 320 points should score
    # HIGH self-similarity at scale 8 (all segments identical). Random
    # Gaussian noise should score LOW.
    unit = np.sin(np.linspace(0, 2 * np.pi, 40)).reshape(-1, 1)
    periodic = np.tile(unit, (8, 1))
    assert periodic.shape[0] == 320
    rng = np.random.RandomState(2)
    noisy = rng.randn(320, 1)
    s_per = self_similarity_score(periodic)
    s_noise = self_similarity_score(noisy)
    assert s_per["self_similarity_mean"] > s_noise["self_similarity_mean"], (
        f"periodic {s_per['self_similarity_mean']:.3f} should exceed "
        f"noise {s_noise['self_similarity_mean']:.3f}"
    )


def test_fractal_metrics_returns_all_keys():
    rng = np.random.RandomState(3)
    traj = rng.randn(200, 4)
    m = fractal_metrics(traj)
    assert "compression_ratio" in m
    assert "motif_3_unique_ratio" in m
    assert "motif_5_unique_ratio" in m
    assert "motif_8_unique_ratio" in m
    assert "self_similarity_mean" in m
    assert "self_similarity_std" in m


def test_fractal_metrics_handles_short_trajectory():
    traj = np.zeros((5, 2))
    m = fractal_metrics(traj)
    assert m["self_similarity_mean"] == 0.0
