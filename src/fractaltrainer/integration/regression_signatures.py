"""Direction H — signatures for non-classification regimes.

Sprint 17's Direction-B negative (Review 34) ruled out penultimate-
activation signatures as a universal drop-in: the within- vs cross-
task L2 gap collapses because latent representations encode
seed/trajectory nuisance more than task identity. Direction B's
conclusion was that signatures must live in an "output-collapse"
space — for classification, softmax.

This module proposes the analog for **regression**:

    regression_probe_signature(model, probe) =
        L2_normalize(model(probe).flatten())

For a model that predicts a scalar y from an input x, the signature
is the vector of predictions on a canonical probe batch, then
normalized. Two regression models that implement similar functions
should produce similar prediction vectors (in direction, after
L2-norm) even if their absolute scale differs.

If this separates tasks cleanly (within-task L2 < cross-task L2),
Direction H is viable and the signature machinery extends to
regression. If it collapses like penultimate did, regression needs
more careful treatment.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def regression_probe_signature(
    model: torch.nn.Module,
    probe: torch.Tensor,
    normalize: str = "l2",
) -> np.ndarray:
    """Signature of a regression model: its predictions on `probe`.

    Args:
        model: any `nn.Module` where `model(probe)` yields a tensor of
            shape `(B,)` or `(B, 1)` — scalar predictions per sample.
        probe: fixed probe batch, typically `(B, input_dim)` or an
            image shape.
        normalize: "l2" (unit L2 norm), "zscore" (mean-0 std-1), or
            "none". L2 removes absolute scale; zscore removes both
            mean offset and scale; none keeps raw.

    Returns:
        1-D float numpy array of length B.
    """
    model.eval()
    with torch.no_grad():
        pred = model(probe)
        if pred.ndim > 1:
            pred = pred.squeeze(-1)
    arr = pred.float().cpu().numpy()

    if normalize == "l2":
        n = float(np.linalg.norm(arr))
        return arr / n if n > 1e-12 else arr
    if normalize == "zscore":
        mu = float(arr.mean())
        sigma = float(arr.std())
        return (arr - mu) / sigma if sigma > 1e-12 else arr - mu
    if normalize == "none":
        return arr
    raise ValueError(f"unknown normalize: {normalize!r}")


class SineRegressor(torch.nn.Module):
    """Minimal scalar-output regression MLP, deliberately tiny so
    we can train many of them quickly.

    input: (B, 1) — x in [-pi, pi]
    output: (B,) — scalar prediction
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.fc3 = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


def make_sine_task(freq: float, phase: float = 0.0) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a target function for a sine task: y = sin(freq·x + phase)."""
    def target(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(freq * x.squeeze(-1) + phase)
    return target


def train_sine_regressor(
    freq: float, phase: float = 0.0,
    *,
    n_steps: int = 500, lr: float = 0.01,
    seed: int = 0, n_samples: int = 1000,
) -> SineRegressor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SineRegressor()
    # Training data: uniform x in [-pi, pi]
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n_samples, 1, generator=g) * 2 - 1) * np.pi
    y = make_sine_task(freq, phase)(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 64
    for step in range(n_steps):
        idx = torch.randint(0, n_samples, (batch_size,), generator=g)
        xb, yb = x[idx], y[idx]
        pred = model(xb)
        loss = torch.nn.functional.mse_loss(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def make_probe_inputs(n: int = 100, seed: int = 12345) -> torch.Tensor:
    """Canonical probe: n evenly-spaced x in [-pi, pi]."""
    return torch.linspace(-np.pi, np.pi, n).unsqueeze(-1)
