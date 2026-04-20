"""v3 Sprint 9a — Coverage-based compose: does greedy ensemble selection
beat nearest-by-signature?

Sprint 5's null result (blend ~= top-1, paired t=0.63) used top-K nearest
by signature as the compose ensemble. Sprint 7b explained why that's
redundant: signature-nearest neighbors are label-redundant (Spearman
rho=-0.848 between Jaccard overlap and signature distance), so blending
K label-aligned experts adds no independent information.

Coverage-based compose tests a different selection rule:

  Pre-filter: top-10 nearest by signature (candidate pool).
  Greedy selection on a held-out selection subset of the query's test
  data: start empty, at each step pick the candidate whose addition
  most improves the running ensemble's accuracy. Stop at K=3.
  Evaluate the chosen K=3 ensemble on a disjoint eval subset.

The signature pre-filter ensures candidates are at least plausibly
related; the greedy selection then picks the 3 that COVER the query
together, which may or may not be the 3 closest by signature.

Experimental design (leave-one-task-out, 27 binary tasks):
  Registry      = 26 other binary tasks x 3 seeds = 78 entries.
  Query         = T seed=2024 signature.
  Test split    = T's 1000-example test set -> 300 selection / 700 eval.
  Oracle        = T seed=42's own expert (never in registry).

Compared methods (all evaluated on the disjoint 700-example eval subset):
  top_1              : the single nearest expert by signature.
  nearest_k3_equal   : equal-weighted blend of top-3 nearest signatures.
  coverage_k3_equal  : equal-weighted blend of greedy-selected K=3 from
                       top-10 pool, selection on selection subset.
  random_k3_equal    : equal-weighted blend of 3 randomly chosen from
                       the top-10 pool (lower-bound control).
  oracle             : T's own seed=42 expert (upper bound).

Key question: does coverage_k3_equal > nearest_k3_equal on the paired
t-test across 27 tasks? If yes, Sprint 5's null was a selection-rule
problem, not a fundamental limit of composition.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
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


def _split_loaders(task_subset: tuple[int, ...], data_dir: str,
                    n_sel: int, n_eval: int, seed: int
                    ) -> tuple[DataLoader, DataLoader]:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    ds = RelabeledMNIST(base, set(task_subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n_sel + n_eval, replace=False)
    sel_idx, eval_idx = idx[:n_sel].tolist(), idx[n_sel:].tolist()
    sel_loader = DataLoader(Subset(ds, sel_idx), batch_size=64, shuffle=False)
    eval_loader = DataLoader(Subset(ds, eval_idx),
                              batch_size=64, shuffle=False)
    return sel_loader, eval_loader


def _compute_probs_and_labels(
    model: nn.Module, loader: DataLoader, n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)[:, :n_classes]
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(y.cpu().numpy())
    return np.concatenate(probs_list, axis=0), np.concatenate(labels_list)


def _ensemble_accuracy(probs_list: list[np.ndarray],
                        weights: list[float] | None,
                        labels: np.ndarray) -> float:
    """Equal-weighted or user-weighted mean of per-expert probs → argmax."""
    if not probs_list:
        return 0.0
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    blend = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        blend = blend + w * p
    blend = blend / sum(weights)
    preds = blend.argmax(axis=1)
    return float((preds == labels).mean())


def _greedy_coverage_select(
    candidate_probs_sel: list[np.ndarray],
    sel_labels: np.ndarray, k: int,
) -> list[int]:
    """Greedy: at each step, pick the candidate whose addition to the
    running (equal-weighted) ensemble most improves selection-subset
    accuracy. Ties broken by lowest index."""
    n = len(candidate_probs_sel)
    if k >= n:
        return list(range(n))
    selected: list[int] = []
    remaining = set(range(n))
    running_sum = np.zeros_like(candidate_probs_sel[0])

    for step in range(k):
        best_acc = -1.0
        best_i = None
        for i in remaining:
            trial_sum = running_sum + candidate_probs_sel[i]
            trial_blend = trial_sum / (len(selected) + 1)
            preds = trial_blend.argmax(axis=1)
            acc = float((preds == sel_labels).mean())
            if acc > best_acc:
                best_acc = acc
                best_i = i
        assert best_i is not None
        selected.append(best_i)
        remaining.discard(best_i)
        running_sum = running_sum + candidate_probs_sel[best_i]
    return selected


def _paired_t(deltas: np.ndarray) -> tuple[float, float]:
    n = deltas.size
    if n < 2:
        return 0.0, 0.0
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return 0.0, mean
    return mean / se, mean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--sprint7-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--n-selection", type=int, default=300)
    parser.add_argument("--n-eval", type=int, default=700)
    parser.add_argument("--split-seed", type=int, default=7777)
    parser.add_argument("--candidate-pool-size", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=2024)
    parser.add_argument("--results-out", type=str,
                        default="results/coverage_compose.json")
    args = parser.parse_args(argv)

    tasks = all_binary_tasks()
    task_names = sorted(tasks.keys())
    print(f"Binary tasks: {len(task_names)}")

    # Build all entries (27 × 3 = 81)
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    all_entries: dict[tuple[str, int], FractalEntry] = {}
    for name in task_names:
        for seed in SEEDS:
            if name in EXISTING_BINARY:
                p = Path(args.cached_traj_dir) / f"ext_{name}_seed{seed}_trajectory.npy"
            else:
                p = Path(args.sprint7_dir) / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file():
                print(f"  MISSING  {p}")
                continue
            model = _load_model(p)
            sig = _signature(model, probe)
            all_entries[(name, seed)] = FractalEntry(
                name=f"{name}_seed{seed}", signature=sig,
                metadata={"task": name, "seed": seed,
                          "trajectory_path": str(p)},
            )
    print(f"Built {len(all_entries)} entries.\n")

    rng = np.random.RandomState(args.random_seed)
    rows: list[dict] = []

    print("=" * 72)
    print("  COVERAGE COMPOSE — leave-one-task-out, 27 binary tasks")
    print("=" * 72)
    print(f"  Candidate pool: top-{args.candidate_pool_size} by signature.")
    print(f"  Test split: {args.n_selection} selection / {args.n_eval} eval.")
    print(f"  Ensemble K: {args.k} (equal weights).")
    print()

    for held_task in task_names:
        # Registry = all entries except held_task's
        registry = FractalRegistry()
        for (t, s), e in all_entries.items():
            if t != held_task:
                registry.add(e)

        query_entry = all_entries[(held_task, 2024)]
        oracle_entry = all_entries[(held_task, 42)]

        # Candidate pool: top-M nearest by signature
        res = registry.find_nearest(
            query_entry.signature,
            k=min(args.candidate_pool_size, len(registry)),
            query_name=held_task,
        )
        candidates = res.entries
        distances = list(res.distances)

        # Load candidate models
        cand_models = [_load_model(Path(e.metadata["trajectory_path"]))
                       for e in candidates]
        oracle_model = _load_model(Path(oracle_entry.metadata["trajectory_path"]))

        # Test split
        sel_loader, eval_loader = _split_loaders(
            tasks[held_task], args.data_dir,
            args.n_selection, args.n_eval, args.split_seed,
        )

        # Precompute per-candidate probs on sel + eval
        cand_probs_sel, sel_labels = [], None
        cand_probs_eval, eval_labels = [], None
        for m in cand_models:
            ps, ls = _compute_probs_and_labels(m, sel_loader, n_classes=2)
            pe, le = _compute_probs_and_labels(m, eval_loader, n_classes=2)
            cand_probs_sel.append(ps)
            cand_probs_eval.append(pe)
            if sel_labels is None:
                sel_labels = ls
            if eval_labels is None:
                eval_labels = le

        # Top-1: just the nearest-signature expert, on eval set.
        acc_top1 = _ensemble_accuracy([cand_probs_eval[0]], None, eval_labels)

        # Nearest-K=3 equal-weighted blend (the top 3 by signature)
        acc_nearest_k3 = _ensemble_accuracy(
            cand_probs_eval[:args.k], None, eval_labels)

        # Coverage K=3: greedy select on selection subset, evaluate on eval.
        sel_indices = _greedy_coverage_select(
            cand_probs_sel, sel_labels, k=args.k)
        acc_coverage_k3 = _ensemble_accuracy(
            [cand_probs_eval[i] for i in sel_indices],
            None, eval_labels,
        )

        # Random K=3 from the candidate pool
        random_indices = rng.choice(
            len(candidates), size=args.k, replace=False).tolist()
        acc_random_k3 = _ensemble_accuracy(
            [cand_probs_eval[i] for i in random_indices],
            None, eval_labels,
        )

        # Oracle: T seed=42 (never in registry during leave-one-task-out).
        oracle_probs, _ = _compute_probs_and_labels(
            oracle_model, eval_loader, n_classes=2)
        acc_oracle = _ensemble_accuracy([oracle_probs], None, eval_labels)

        row = {
            "held_task": held_task,
            "top1_name": candidates[0].name,
            "nearest_k3_names": [candidates[i].name for i in range(args.k)],
            "coverage_k3_names": [candidates[i].name for i in sel_indices],
            "random_k3_names": [candidates[i].name for i in random_indices],
            "acc_top1": acc_top1,
            "acc_nearest_k3": acc_nearest_k3,
            "acc_coverage_k3": acc_coverage_k3,
            "acc_random_k3": acc_random_k3,
            "acc_oracle": acc_oracle,
            "coverage_differs_from_nearest": (
                sorted(sel_indices) != list(range(args.k))),
            "sel_indices": sel_indices,
            "candidate_distances": distances,
        }
        rows.append(row)
        print(f"  {held_task:<20s}  top1={acc_top1:.3f}  "
              f"near_k3={acc_nearest_k3:.3f}  "
              f"cov_k3={acc_coverage_k3:.3f}  "
              f"rand_k3={acc_random_k3:.3f}  "
              f"oracle={acc_oracle:.3f}  "
              f"diff={row['coverage_differs_from_nearest']}")

    top1 = np.array([r["acc_top1"] for r in rows])
    near_k3 = np.array([r["acc_nearest_k3"] for r in rows])
    cov_k3 = np.array([r["acc_coverage_k3"] for r in rows])
    rand_k3 = np.array([r["acc_random_k3"] for r in rows])
    oracle = np.array([r["acc_oracle"] for r in rows])

    print()
    print(f"  Mean top1     : {top1.mean():.4f}")
    print(f"  Mean near_k3  : {near_k3.mean():.4f}")
    print(f"  Mean cov_k3   : {cov_k3.mean():.4f}")
    print(f"  Mean rand_k3  : {rand_k3.mean():.4f}")
    print(f"  Mean oracle   : {oracle.mean():.4f}")

    # Paired t-tests
    t_cov_vs_top1, d_cov_vs_top1 = _paired_t(cov_k3 - top1)
    t_cov_vs_near, d_cov_vs_near = _paired_t(cov_k3 - near_k3)
    t_near_vs_top1, d_near_vs_top1 = _paired_t(near_k3 - top1)

    wins_cov_vs_near = int((cov_k3 > near_k3).sum())
    ties_cov_vs_near = int((cov_k3 == near_k3).sum())
    losses_cov_vs_near = int((cov_k3 < near_k3).sum())
    wins_cov_vs_top1 = int((cov_k3 > top1).sum())
    ties_cov_vs_top1 = int((cov_k3 == top1).sum())
    losses_cov_vs_top1 = int((cov_k3 < top1).sum())
    n_differs = sum(r["coverage_differs_from_nearest"] for r in rows)

    print()
    print(f"  Paired Δ coverage − top1     : {d_cov_vs_top1:+.4f}  "
          f"t={t_cov_vs_top1:.2f}  "
          f"({wins_cov_vs_top1}W/{ties_cov_vs_top1}T/{losses_cov_vs_top1}L)")
    print(f"  Paired Δ coverage − nearest_k3: {d_cov_vs_near:+.4f}  "
          f"t={t_cov_vs_near:.2f}  "
          f"({wins_cov_vs_near}W/{ties_cov_vs_near}T/{losses_cov_vs_near}L)")
    print(f"  Paired Δ nearest_k3 − top1    : {d_near_vs_top1:+.4f}  "
          f"t={t_near_vs_top1:.2f}")
    print(f"  Coverage picked non-trivially (≠ top-3 nearest): "
          f"{n_differs}/{len(rows)}")

    # Verdict
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if t_cov_vs_near > 2.0 and wins_cov_vs_near > len(rows) / 2:
        v_vs_near = "COVERAGE BEATS NEAREST"
    elif abs(t_cov_vs_near) <= 2.0:
        v_vs_near = "COVERAGE TIES NEAREST"
    else:
        v_vs_near = "COVERAGE LOSES TO NEAREST"
    if t_cov_vs_top1 > 2.0 and wins_cov_vs_top1 > len(rows) / 2:
        v_vs_top1 = "COVERAGE BEATS TOP-1"
    elif abs(t_cov_vs_top1) <= 2.0:
        v_vs_top1 = "COVERAGE TIES TOP-1"
    else:
        v_vs_top1 = "COVERAGE LOSES TO TOP-1"
    print(f"  vs nearest-K=3: {v_vs_near} (t={t_cov_vs_near:+.2f}, "
          f"Δ={d_cov_vs_near:+.4f})")
    print(f"  vs top-1      : {v_vs_top1} (t={t_cov_vs_top1:+.2f}, "
          f"Δ={d_cov_vs_top1:+.4f})")

    out = {
        "n_tasks": len(task_names),
        "candidate_pool_size": args.candidate_pool_size,
        "k": args.k,
        "n_selection": args.n_selection,
        "n_eval": args.n_eval,
        "rows": rows,
        "summary": {
            "mean_top1": float(top1.mean()),
            "mean_nearest_k3": float(near_k3.mean()),
            "mean_coverage_k3": float(cov_k3.mean()),
            "mean_random_k3": float(rand_k3.mean()),
            "mean_oracle": float(oracle.mean()),
            "delta_cov_vs_top1": float(d_cov_vs_top1),
            "t_cov_vs_top1": float(t_cov_vs_top1),
            "delta_cov_vs_near": float(d_cov_vs_near),
            "t_cov_vs_near": float(t_cov_vs_near),
            "delta_near_vs_top1": float(d_near_vs_top1),
            "t_near_vs_top1": float(t_near_vs_top1),
            "wins_cov_vs_near":
                {"w": wins_cov_vs_near, "t": ties_cov_vs_near,
                 "l": losses_cov_vs_near},
            "wins_cov_vs_top1":
                {"w": wins_cov_vs_top1, "t": ties_cov_vs_top1,
                 "l": losses_cov_vs_top1},
            "coverage_picks_differ_from_nearest": n_differs,
            "verdict_vs_near": v_vs_near,
            "verdict_vs_top1": v_vs_top1,
        },
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
