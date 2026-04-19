"""Reference fixtures with known fractal dimensions.

Used by test_correlation_dim.py and test_box_counting.py as regression
anchors. These are the "well-behaved systems" the geometry modules must
handle correctly before we trust them on real training trajectories.
"""

from __future__ import annotations

import numpy as np


def cantor_set_points(n_points: int = 2000, levels: int = 8, seed: int = 0) -> np.ndarray:
    """Points sampled uniformly from the middle-thirds Cantor set.

    Theoretical box-counting dim = log(2) / log(3) ≈ 0.6309.
    Returned as (n_points, 1).
    """
    rng = np.random.RandomState(seed)
    bits = rng.randint(0, 2, size=(n_points, levels))
    powers = (1.0 / 3.0) ** np.arange(1, levels + 1)
    values = 2.0 * bits * powers  # each level contributes 0 or 2/3^k
    x = values.sum(axis=1)
    return x.reshape(-1, 1)


def henon_points(n_points: int = 2000, burn_in: int = 500, a: float = 1.4,
                 b: float = 0.3, x0: float = 0.1, y0: float = 0.1) -> np.ndarray:
    """Classical Henon attractor. Correlation dim ≈ 1.26.

    Returned as (n_points, 2).
    """
    total = n_points + burn_in
    xs = np.zeros(total)
    ys = np.zeros(total)
    xs[0], ys[0] = x0, y0
    for i in range(1, total):
        xs[i] = 1.0 - a * xs[i - 1] ** 2 + ys[i - 1]
        ys[i] = b * xs[i - 1]
    return np.stack([xs[burn_in:], ys[burn_in:]], axis=1)


def lorenz_points(n_points: int = 3000, dt: float = 0.01, burn_in: int = 500,
                  sigma: float = 10.0, rho: float = 28.0,
                  beta: float = 8.0 / 3.0,
                  x0: float = 1.0, y0: float = 1.0, z0: float = 1.0) -> np.ndarray:
    """Lorenz attractor sampled via Euler integration. Correlation dim ≈ 2.05.

    Returned as (n_points, 3).
    """
    total = n_points + burn_in
    x = np.zeros(total)
    y = np.zeros(total)
    z = np.zeros(total)
    x[0], y[0], z[0] = x0, y0, z0
    for i in range(1, total):
        dx = sigma * (y[i - 1] - x[i - 1])
        dy = x[i - 1] * (rho - z[i - 1]) - y[i - 1]
        dz = x[i - 1] * y[i - 1] - beta * z[i - 1]
        x[i] = x[i - 1] + dt * dx
        y[i] = y[i - 1] + dt * dy
        z[i] = z[i - 1] + dt * dz
    return np.stack([x[burn_in:], y[burn_in:], z[burn_in:]], axis=1)


def unit_square_points(n_points: int = 2000, seed: int = 0) -> np.ndarray:
    """Uniform fill of the unit square. Box-counting dim = 2.0 exactly."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, 1.0, size=(n_points, 2))


def random_walk_high_d(n_points: int = 500, n_dim: int = 16,
                       step_scale: float = 0.1, seed: int = 0) -> np.ndarray:
    """High-dim Gaussian random walk. Correlation dim saturates ≈ n_dim."""
    rng = np.random.RandomState(seed)
    steps = rng.randn(n_points, n_dim) * step_scale
    return np.cumsum(steps, axis=0)
