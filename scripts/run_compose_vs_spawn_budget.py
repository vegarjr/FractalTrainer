"""v3 Sprint 9b — Data-efficient growth: compose-vs-spawn crossover.

A new task T arrives. The user has N labeled examples. Two paths:

  COMPOSE: use all N as a selection subset. Pre-filter top-10 nearest
           by signature. Greedy-select K=3 that maximize ensemble
           accuracy on the N-example subset. Evaluate on held-out test.

  SPAWN:   use all N as training data. Train a new MLP for 500 steps.
           Evaluate on held-out test.

Both paths evaluate on the SAME held-out 1000-example MNIST test set,
so the compose/spawn accuracies are directly comparable.

Expected shape: at small N, compose wins (can exploit the registry's
existing experts); at large N, spawn wins (can overfit-then-converge
to a true task expert). The crossover N* is the budget at which
spawning starts to pay off.

Experimental setup (leave-one-task-out, 27 binary tasks):
  Registry        = 26 other binary tasks × 3 seeds = 78 entries.
  Candidate pool  = top-10 nearest by signature (from leave-one-out).
  Budget sweep    = N ∈ {50, 100, 300, 1000, 5000}.
  Budget data     = first N examples of seeded-shuffled MNIST train.
  Eval set        = 1000 fixed MNIST test examples (disjoint from
                    budget data by construction: train vs test split).

Cost:
  Spawn runs      = 27 tasks × 5 budgets × 1 seed = 135 training runs
                    @ ~7s each = ~16 min.
  Compose runs    = negligible (candidate probs cached per task).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]
SEEDS = [42, 101, 2024]

EXISTING_BINARY: dict[str, tuple[int, ...]] = {
    "parity":          (1, 3, 5, 7, 9),
    "high_vs_low":     (5, 6, 7, 8, 9),
    "primes_vs_rest":  (2, 3, 5, 7),
    "ones_vs_teens":   (0, 1, 2, 3, 4),
    "triangular":      (1, 3, 6),
    "fibonacci":       (1, 2, 3, 5, 8),
    "middle_456":      (4, 5, 6),
}


def _sprint7_new_tasks() -> dict[str, tuple[int, ...]]:
    existing = [frozenset(s) for s in EXISTING_BINARY.values()]

    def _is_novel(s: frozenset[int]) -> bool:
        full = frozenset(range(10))
        return all(s != ex and s != (full - ex) for ex in existing)

    candidates = []
    for k in (3, 4, 5, 6):
        for c in itertools.combinations(range(10), k):
            if _is_novel(frozenset(c)):
                candidates.append(tuple(sorted(c)))
    rng = random.Random(42)
    rng.shuffle(candidates)
    return {
        "subset_" + "".join(str(d) for d in s): s
        for s in candidates[:20]
    }


def all_binary_tasks() -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    out.update(EXISTING_BINARY)
    out.update(_sprint7_new_tasks())
    return out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def _load_model(traj_path: Path) -> nn.Module:
    trajectory = np.load(traj_path)
    final_flat = trajectory[-1]
    model = MLP()
    offset = 0
    state_dict = {}
    for shape, name in LAYER_SHAPES:
        size = int(np.prod(shape))
        chunk = final_flat[offset:offset + size]
        state_dict[name] = torch.tensor(chunk.reshape(shape), dtype=torch.float32)
        offset += size
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _mnist_probe(data_dir: str, n: int, seed: int) -> torch.Tensor:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n, replace=False)
    return torch.stack([base[i][0] for i in idx.tolist()], dim=0)


def _signature(model: nn.Module, probe: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        probs = F.softmax(model(probe), dim=1)
    return probs.flatten().cpu().numpy()


class RelabeledMNIST(Dataset):
    def __init__(self, base, target_subset: set[int]):
        self.base, self.target = base, set(target_subset)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _make_budget_loader(task_subset: tuple[int, ...], N: int,
                         data_dir: str, seed: int) -> DataLoader:
    """First N examples of a seeded shuffle of MNIST train."""
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, set(task_subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=N, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                       batch_size=min(64, N),
                       shuffle=False, drop_last=False)


def _make_eval_loader(task_subset: tuple[int, ...], n_eval: int,
                       data_dir: str, seed: int) -> DataLoader:
    """n_eval examples from MNIST test (disjoint from train by construction)."""
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    ds = RelabeledMNIST(base, set(task_subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n_eval, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64,
                       shuffle=False)


def _compute_probs_and_labels(
    model: nn.Module, loader: DataLoader, n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)[:, :n_classes]
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            labels_list.append(y.cpu().numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def _greedy_coverage_select(
    cand_probs_sel: list[np.ndarray],
    sel_labels: np.ndarray, k: int,
) -> list[int]:
    n = len(cand_probs_sel)
    if k >= n:
        return list(range(n))
    selected: list[int] = []
    remaining = set(range(n))
    running_sum = np.zeros_like(cand_probs_sel[0])
    for _ in range(k):
        best_acc, best_i = -1.0, None
        for i in remaining:
            trial = (running_sum + cand_probs_sel[i]) / (len(selected) + 1)
            acc = float((trial.argmax(axis=1) == sel_labels).mean())
            if acc > best_acc:
                best_acc, best_i = acc, i
        assert best_i is not None
        selected.append(best_i)
        remaining.discard(best_i)
        running_sum = running_sum + cand_probs_sel[best_i]
    return selected


def _ensemble_accuracy(probs_list: list[np.ndarray],
                        labels: np.ndarray) -> float:
    if not probs_list:
        return 0.0
    blend = np.mean(probs_list, axis=0)
    return float((blend.argmax(axis=1) == labels).mean())


def _spawn_with_budget(
    task_subset: tuple[int, ...], N: int, seed: int, data_dir: str,
    eval_loader: DataLoader, n_steps: int = 500,
) -> tuple[float, float]:
    """Train a fresh MLP on N labeled examples, evaluate on eval set."""
    budget_loader = _make_budget_loader(task_subset, N, data_dir, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = F.cross_entropy

    t0 = time.time()
    step = 0
    while step < n_steps:
        for x, y in budget_loader:
            if step >= n_steps:
                break
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            step += 1
    elapsed = time.time() - t0

    eval_probs, eval_labels = _compute_probs_and_labels(
        model, eval_loader, n_classes=2)
    acc = float((eval_probs.argmax(axis=1) == eval_labels).mean())
    return acc, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--sprint7-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--candidate-pool-size", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=7777)
    parser.add_argument("--budget-seed", type=int, default=2024)
    parser.add_argument("--spawn-seed", type=int, default=2024)
    parser.add_argument("--spawn-steps", type=int, default=500)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[50, 100, 300, 1000, 5000])
    parser.add_argument("--results-out", type=str,
                        default="results/compose_vs_spawn_budget.json")
    args = parser.parse_args(argv)

    tasks = all_binary_tasks()
    task_names = sorted(tasks.keys())
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)

    # Build all entries
    all_entries: dict[tuple[str, int], FractalEntry] = {}
    for name in task_names:
        for seed in SEEDS:
            if name in EXISTING_BINARY:
                p = Path(args.cached_traj_dir) / f"ext_{name}_seed{seed}_trajectory.npy"
            else:
                p = Path(args.sprint7_dir) / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            sig = _signature(_load_model(p), probe)
            all_entries[(name, seed)] = FractalEntry(
                name=f"{name}_seed{seed}", signature=sig,
                metadata={"task": name, "seed": seed,
                          "trajectory_path": str(p)},
            )
    print(f"Built {len(all_entries)} entries / {len(task_names)} tasks.")
    print(f"Budget sweep: {args.budgets}")
    print(f"Candidate pool size: {args.candidate_pool_size}, K={args.k}")
    print()

    rows: list[dict] = []
    t_start = time.time()

    for task_idx, held_task in enumerate(task_names, start=1):
        # Leave-one-task-out registry
        registry = FractalRegistry()
        for (t, s), e in all_entries.items():
            if t != held_task:
                registry.add(e)

        query_entry = all_entries[(held_task, 2024)]

        # Candidate pool + models (loaded once per task)
        res = registry.find_nearest(
            query_entry.signature,
            k=min(args.candidate_pool_size, len(registry)),
        )
        candidates = res.entries
        cand_models = [_load_model(Path(e.metadata["trajectory_path"]))
                       for e in candidates]

        # Eval set (same across budgets)
        eval_loader = _make_eval_loader(
            tasks[held_task], args.n_eval, args.data_dir, args.eval_seed)

        # Precompute candidate probs on eval (shared across budgets)
        cand_probs_eval = []
        eval_labels = None
        for m in cand_models:
            p, l = _compute_probs_and_labels(m, eval_loader, n_classes=2)
            cand_probs_eval.append(p)
            if eval_labels is None:
                eval_labels = l

        # Top-1 baseline (budget-independent)
        acc_top1 = _ensemble_accuracy([cand_probs_eval[0]], eval_labels)

        for budget in args.budgets:
            # Budget data (selection subset for compose, train data for spawn)
            budget_loader = _make_budget_loader(
                tasks[held_task], budget, args.data_dir, args.budget_seed)

            # ── Compose path ──
            cand_probs_budget = []
            budget_labels = None
            for m in cand_models:
                p, l = _compute_probs_and_labels(m, budget_loader, n_classes=2)
                cand_probs_budget.append(p)
                if budget_labels is None:
                    budget_labels = l
            selected = _greedy_coverage_select(
                cand_probs_budget, budget_labels, k=args.k)
            acc_compose = _ensemble_accuracy(
                [cand_probs_eval[i] for i in selected], eval_labels)

            # ── Spawn path ──
            acc_spawn, spawn_s = _spawn_with_budget(
                tasks[held_task], budget,
                seed=args.spawn_seed,
                data_dir=args.data_dir,
                eval_loader=eval_loader,
                n_steps=args.spawn_steps,
            )

            row = {
                "task": held_task,
                "budget_N": budget,
                "acc_top1": acc_top1,
                "acc_compose": acc_compose,
                "acc_spawn": acc_spawn,
                "compose_selected": [candidates[i].name for i in selected],
                "spawn_train_s": spawn_s,
            }
            rows.append(row)
            print(f"  [{task_idx:>2}/{len(task_names)}] "
                  f"{held_task:<18s} N={budget:>5d}  "
                  f"top1={acc_top1:.3f}  compose={acc_compose:.3f}  "
                  f"spawn={acc_spawn:.3f}  (spawn {spawn_s:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\nTotal wall clock: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ── Per-budget aggregation ──
    print("\n" + "=" * 72)
    print("  PER-BUDGET MEANS (across 27 tasks)")
    print("=" * 72)
    print(f"  {'N':>6s}  {'top1':>8s}  {'compose':>8s}  {'spawn':>8s}  "
          f"{'Δ(spawn−compose)':>17s}  {'compose>spawn':>14s}")
    per_budget: dict[int, dict] = {}
    for budget in args.budgets:
        subset = [r for r in rows if r["budget_N"] == budget]
        t1 = np.array([r["acc_top1"] for r in subset])
        cp = np.array([r["acc_compose"] for r in subset])
        sp = np.array([r["acc_spawn"] for r in subset])
        delta = sp - cp
        n_compose_wins = int((cp > sp).sum())
        n_tied = int((cp == sp).sum())
        n_spawn_wins = int((sp > cp).sum())
        per_budget[budget] = {
            "mean_top1": float(t1.mean()),
            "mean_compose": float(cp.mean()),
            "mean_spawn": float(sp.mean()),
            "mean_delta_spawn_minus_compose": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)),
            "n_compose_wins": n_compose_wins,
            "n_tied": n_tied,
            "n_spawn_wins": n_spawn_wins,
        }
        print(f"  {budget:>6d}  {t1.mean():>8.4f}  {cp.mean():>8.4f}  "
              f"{sp.mean():>8.4f}  "
              f"{delta.mean():>+17.4f}  "
              f"{n_compose_wins:>5d}/{len(subset):<4d}")

    # ── Per-task crossover ──
    print()
    print("  Per-task crossover budget (smallest N where spawn > compose):")
    crossovers: dict[str, int | None] = {}
    for task in task_names:
        task_rows = sorted([r for r in rows if r["task"] == task],
                            key=lambda r: r["budget_N"])
        crossover = None
        for r in task_rows:
            if r["acc_spawn"] > r["acc_compose"]:
                crossover = r["budget_N"]
                break
        crossovers[task] = crossover
        tag = f"{crossover}" if crossover is not None else "never in sweep"
        print(f"    {task:<22s} crossover N = {tag}")

    # Sort crossovers
    never = [t for t, n in crossovers.items() if n is None]
    finite = sorted([n for n in crossovers.values() if n is not None])
    print()
    print(f"  Tasks where spawn never beat compose in [{min(args.budgets)}, "
          f"{max(args.budgets)}]: {len(never)}")
    if finite:
        print(f"  Median crossover (among tasks where it occurs): {int(np.median(finite))}")

    # ── Verdict ──
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    for budget in args.budgets:
        row = per_budget[budget]
        if row["n_compose_wins"] > row["n_spawn_wins"] + 2:
            regime = "COMPOSE WINS"
        elif row["n_spawn_wins"] > row["n_compose_wins"] + 2:
            regime = "SPAWN WINS"
        else:
            regime = "TIED"
        print(f"  N={budget:>5d}: mean Δ={row['mean_delta_spawn_minus_compose']:+.4f}  "
              f"({row['n_compose_wins']}C/{row['n_tied']}T/"
              f"{row['n_spawn_wins']}S) → {regime}")

    out = {
        "n_tasks": len(task_names),
        "budgets": args.budgets,
        "candidate_pool_size": args.candidate_pool_size,
        "k": args.k,
        "n_eval": args.n_eval,
        "rows": rows,
        "per_budget": {str(k): v for k, v in per_budget.items()},
        "crossovers": {t: n for t, n in crossovers.items()},
        "tasks_never_crossover": never,
        "median_crossover": (int(np.median(finite)) if finite else None),
        "total_wall_clock_s": elapsed,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
