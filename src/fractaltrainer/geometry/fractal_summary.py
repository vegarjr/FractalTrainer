"""VENDORED and ADAPTED from /home/vegar/Documents/Code_Geometry/representations/fractal_summary.py
Source commit: e8a00f05e821f0636d8ac422f49d98c7bb1324ca (Initial commit: Code Geometry Lab MVP)
Original author: Vegar Ratdal
License: MIT
Modifications:
  - Input type changed from trace-event records to numeric (N, D) trajectories.
  - Quantization step added: each trajectory dim is binned into K symbols,
    then concatenated to a per-timestep multi-dim symbol.
  - Dropped branching_profile (trace-specific: depends on call-stack depth).
  - Kept: compression_ratio, motif_analysis, self_similarity_score — concept
    preserved, input adapted.

These are *baseline* fractal-proxy metrics. Primary metric is
correlation_dim.correlation_dim() (Grassberger-Procaccia). These exist for
comparison and to allow alternative scoring later.
"""

from __future__ import annotations

import zlib
from collections import Counter

import numpy as np


def _quantize(trajectory: np.ndarray, n_bins: int = 16) -> list[str]:
    """Convert (N, D) trajectory into a list of N compact string symbols.

    Each dim is independently binned into n_bins; per timestep, the resulting
    bin indices are joined into one symbol like "b3|b11|b7|..." .
    """
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must be 2-D, got {trajectory.shape}")
    n, d = trajectory.shape
    bins = np.zeros_like(trajectory, dtype=np.int32)
    for k in range(d):
        col = trajectory[:, k]
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-12:
            bins[:, k] = 0
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        bins[:, k] = np.clip(np.digitize(col, edges) - 1, 0, n_bins - 1)
    symbols = ["|".join(f"b{b}" for b in bins[i]) for i in range(n)]
    return symbols


def compression_ratio(trajectory: np.ndarray, n_bins: int = 16) -> float:
    """Encode quantized trajectory as a string and zlib-compress.

    Lower ratio = more compressible = more repetitive / self-similar.
    """
    symbols = _quantize(trajectory, n_bins=n_bins)
    if not symbols:
        return 0.0
    joined = "\n".join(symbols)
    raw = joined.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / max(len(raw), 1)


def _extract_ngrams(sequence: list[str], n: int) -> list[tuple]:
    if len(sequence) < n:
        return []
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def motif_analysis(
    trajectory: np.ndarray,
    ngram_sizes: tuple[int, ...] = (3, 5, 8),
    n_bins: int = 16,
) -> dict:
    """Repetition structure of the quantized symbol sequence."""
    symbols = _quantize(trajectory, n_bins=n_bins)
    results: dict = {}
    for n in ngram_sizes:
        if len(symbols) < n:
            results[f"motif_{n}_unique_ratio"] = 1.0
            results[f"motif_{n}_top_frequency"] = 0
            results[f"motif_{n}_entropy"] = 0.0
            continue

        ngrams = _extract_ngrams(symbols, n)
        counts = Counter(ngrams)
        total = len(ngrams)
        unique = len(counts)

        results[f"motif_{n}_unique_ratio"] = unique / total if total > 0 else 1.0
        results[f"motif_{n}_top_frequency"] = (
            counts.most_common(1)[0][1] if counts else 0
        )

        probs = np.array(list(counts.values()), dtype=float) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
        results[f"motif_{n}_entropy"] = float(entropy)

    return results


def self_similarity_score(
    trajectory: np.ndarray,
    scales: tuple[int, ...] = (2, 4, 8),
) -> dict:
    """Compare trajectory segments at multiple scales using RMSE similarity.

    For each scale s, slice the trajectory into s contiguous equal-length
    segments (after length normalization by resampling). Compare every pair
    of segments; similarity = 1 - normalized RMSE. Mean + std across all
    pair comparisons.
    """
    n, d = trajectory.shape
    if n < 16:
        return {"self_similarity_mean": 0.0, "self_similarity_std": 0.0}

    sims_all: list[float] = []
    for scale in scales:
        seg_len = n // scale
        if seg_len < 4:
            continue
        segments = [trajectory[i * seg_len:(i + 1) * seg_len] for i in range(scale)]
        # Normalize each segment to [0, 1] per-dim for fair comparison
        normed = []
        for seg in segments:
            lo = seg.min(axis=0)
            hi = seg.max(axis=0)
            rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
            normed.append((seg - lo) / rng)
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                rmse = float(np.sqrt(np.mean((normed[i] - normed[j]) ** 2)))
                sims_all.append(1.0 - rmse)

    if not sims_all:
        return {"self_similarity_mean": 0.0, "self_similarity_std": 0.0}

    return {
        "self_similarity_mean": float(np.mean(sims_all)),
        "self_similarity_std": float(np.std(sims_all)),
    }


def fractal_metrics(trajectory: np.ndarray) -> dict:
    """All fractal-proxy metrics for a trajectory (baseline, not primary)."""
    metrics: dict = {}
    metrics["compression_ratio"] = compression_ratio(trajectory)
    metrics.update(motif_analysis(trajectory))
    metrics.update(self_similarity_score(trajectory))
    return metrics
