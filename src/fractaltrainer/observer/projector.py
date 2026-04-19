"""Dimensionality reduction for high-dim trajectories.

Weight-space is typically 10^3–10^5 dimensions. Raw correlation dimension
on that saturates. Projecting to ~16 dims via Johnson-Lindenstrauss random
projection preserves pairwise distances within ε ≈ 0.3 for N ≤ 500 points,
and is deterministic per seed.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def project_random(
    trajectory: np.ndarray,
    n_components: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Random Gaussian projection to n_components.

    Args:
        trajectory: (N, D) array.
        n_components: target dim. Default 16 for Johnson-Lindenstrauss margin.
        seed: RNG seed. Same seed → identical projection matrix.

    Returns:
        (N, n_components) array.
    """
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must be 2-D, got {trajectory.shape}")
    n_points, n_features = trajectory.shape
    if n_components >= n_features:
        return trajectory.copy()
    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    return rp.fit_transform(trajectory)


def project_pca(trajectory: np.ndarray, n_components: int = 8) -> np.ndarray:
    """PCA projection to n_components.

    Args:
        trajectory: (N, D) array.
        n_components: target dim. Default 8.

    Returns:
        (N, n_components) array.

    Note: PCA is fit on the trajectory itself (no external training data), so
    results depend on the trajectory. Use project_random for deterministic,
    trajectory-independent projection.
    """
    if trajectory.ndim != 2:
        raise ValueError(f"trajectory must be 2-D, got {trajectory.shape}")
    n_points, n_features = trajectory.shape
    max_components = min(n_points, n_features)
    if n_components >= max_components:
        n_components = max_components
    pca = PCA(n_components=n_components)
    return pca.fit_transform(trajectory)
