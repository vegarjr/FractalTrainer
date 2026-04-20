"""Behavior cloning — train a policy MLP from demonstrations, then
compute its FractalRegistry signature.

Policy architecture: state_dim → 64 → 32 → n_actions.
Loss: cross-entropy over 4 actions.
Optimizer: Adam, lr=1e-3, 200 epochs, batch=64.

Signature: softmax outputs on the canonical probe batch from env.py,
flattened to a (n_probes * n_actions,)-dim vector.

Evaluation helpers: run a trained policy in SnakeEnv, measure average
survival + food eaten over N games. Used by the sprint driver to
compare single-policy vs ensemble performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fractaltrainer.snake.env import ACTIONS, SnakeEnv


class PolicyMLP(nn.Module):
    """Small 3-layer MLP mapping flat state → action logits."""

    def __init__(self, input_dim: int = 100, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        return self.net(x)


@dataclass
class TrainedPolicy:
    name: str
    model: PolicyMLP
    signature: np.ndarray    # (n_probes * n_actions,)
    train_loss_curve: list[float]
    n_demos: int


def train_policy(
    demos_states: np.ndarray, demos_actions: np.ndarray, *,
    name: str, probe: np.ndarray,
    input_dim: Optional[int] = None, n_actions: int = 4,
    n_epochs: int = 200, batch_size: int = 64, lr: float = 1e-3,
    seed: int = 42, device: str = "cpu",
) -> TrainedPolicy:
    """Train a PolicyMLP via supervised behavior cloning.

    Args:
        demos_states: (N, D) int array of board states.
        demos_actions: (N,) int array of action labels in 0..n_actions.
        name: label for the resulting TrainedPolicy.
        probe: (P, D) int array — canonical probe states for signature.
        input_dim: inferred from demos_states if None.
        n_epochs, batch_size, lr: standard SGD-with-Adam knobs.
        seed: for reproducibility.
        device: 'cpu' or 'cuda:0'. Default cpu — these MLPs are tiny.

    Returns a TrainedPolicy with model + action-distribution signature.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if input_dim is None:
        input_dim = demos_states.shape[1]

    model = PolicyMLP(input_dim=input_dim, n_actions=n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.tensor(demos_states, dtype=torch.float32, device=device)
    y = torch.tensor(demos_actions, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size,
                         shuffle=True, drop_last=False)

    loss_curve: list[float] = []
    for epoch in range(n_epochs):
        ep_loss = 0.0
        n = 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
            n += len(xb)
        loss_curve.append(ep_loss / max(n, 1))

    model.eval()
    with torch.no_grad():
        probe_t = torch.tensor(probe, dtype=torch.float32, device=device)
        probe_probs = F.softmax(model(probe_t), dim=1)
    signature = probe_probs.cpu().numpy().flatten().astype(np.float64)
    return TrainedPolicy(
        name=name, model=model, signature=signature,
        train_loss_curve=loss_curve, n_demos=len(demos_states),
    )


# ── Running policies in-env (for evaluation) ─────────────────────────

def _policy_choose(model: PolicyMLP, state: np.ndarray) -> int:
    """argmax over logits; deterministic tiebreak."""
    with torch.no_grad():
        logits = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    return int(torch.argmax(logits, dim=1).item())


def _ensemble_choose(models: list[PolicyMLP], weights: np.ndarray,
                      state: np.ndarray) -> int:
    """Weighted softmax-averaged argmax across an ensemble."""
    probs = None
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for w, m in zip(weights, models):
            p = F.softmax(m(x), dim=1).cpu().numpy().flatten()
            probs = p * w if probs is None else probs + p * w
    return int(np.argmax(probs))


def play_games(policy: Callable[[np.ndarray], int], *,
                n_games: int = 50, height: int = 10, width: int = 10,
                seed_base: int = 5000) -> dict:
    """Play n_games using a policy that takes (state_vector,) → action.

    Returns aggregate stats: mean/median survival steps, food eaten, etc.
    """
    survivals, scores = [], []
    for g in range(n_games):
        env = SnakeEnv(height=height, width=width, seed=seed_base + g)
        state = env.reset()
        while not env.done:
            action = policy(state)
            result = env.step(action)
            state = result.state
        survivals.append(env.steps)
        scores.append(env.score)
    return {
        "n_games": n_games,
        "mean_survival": float(np.mean(survivals)),
        "mean_score": float(np.mean(scores)),
        "median_survival": float(np.median(survivals)),
        "max_score": int(np.max(scores)),
    }


def evaluate_single_policy(model: PolicyMLP, **kwargs) -> dict:
    return play_games(lambda s: _policy_choose(model, s), **kwargs)


def evaluate_ensemble(models: list[PolicyMLP], weights: np.ndarray,
                       **kwargs) -> dict:
    return play_games(lambda s: _ensemble_choose(models, weights, s),
                       **kwargs)
