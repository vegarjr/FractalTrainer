"""v3 Sprint 9d — Test routing rules for the triage decision.

Sprint 9c found min_element_cover and jaccard_to_target correlate
strongly with compose-minus-spawn advantage (ρ = +0.52 and +0.48).
Sprint 9d asks the next question: do these correlations convert into
a *routing rule* that actually achieves better mean accuracy than the
simple N<100 rule from Sprint 9b?

For each (task, budget) combo in Sprint 9b's stored data we know
BOTH compose and spawn accuracy. A routing rule picks one of the two;
its achieved accuracy is the accuracy of the picked path. We compare
six rules:

  R0 oracle              — always pick max(compose, spawn). Ceiling
                           that no deployable rule can beat.
  R1 always_spawn        — ignore the registry; spawn every time.
  R2 always_compose      — trust the registry; compose every time.
  R3 simple_N            — compose if N<100, else spawn. Sprint 9b.
  R4 metadata            — compose if N<100 OR top-10 pool has
                           redundant coverage of target. Uses ONLY
                           candidate label metadata — no model
                           forwards, no labeled data needed.
  R5 post_hoc            — compose if N<100 OR the already-selected
                           3 picks have redundant coverage metrics.
                           Requires running coverage selection first
                           (so uses labeled data), but is a cheaper
                           signal than running the evaluation.

If R4 or R5 beat R3, redundant-coverage is a useful triage signal;
if not, Sprint 9c's correlation was real but didn't convert.

Rule parameters (kept simple — no hyperparameter search):
  redundant_metadata     : min_e(count of top-10 claiming e) ≥ 3
                           AND mean jaccard of best 3 ≥ 0.40
  redundant_post_hoc     : min_element_cover over picks ≥ 2
                           AND mean jaccard to target over picks ≥ 0.40
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


def all_binary_tasks() -> dict[str, frozenset[int]]:
    out: dict[str, frozenset[int]] = {}
    for k, v in EXISTING_BINARY.items():
        out[k] = frozenset(v)
    for k, v in _sprint7_new_tasks().items():
        out[k] = frozenset(v)
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


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _strip_seed(entry_name: str) -> str:
    if "_seed" in entry_name:
        return entry_name.rsplit("_seed", 1)[0]
    return entry_name


def _build_top10_per_task(
    tasks: dict[str, frozenset[int]],
    cached_traj_dir: Path, sprint7_dir: Path, data_dir: str,
    n_probe: int, probe_seed: int,
) -> dict[str, list[str]]:
    """For each task T, compute top-10 nearest-by-signature in the
    leave-T-out 78-entry registry. Returns task_name → list of
    candidate task names (one per top-10 slot; may duplicate task
    across seeds)."""
    probe = _mnist_probe(data_dir, n_probe, probe_seed)
    all_entries: dict[tuple[str, int], FractalEntry] = {}
    for name in tasks.keys():
        for seed in SEEDS:
            if name in EXISTING_BINARY:
                p = cached_traj_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            else:
                p = sprint7_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            sig = _signature(_load_model(p), probe)
            all_entries[(name, seed)] = FractalEntry(
                name=f"{name}_seed{seed}", signature=sig,
                metadata={"task": name, "seed": seed},
            )

    out: dict[str, list[str]] = {}
    for held_task in tasks:
        registry = FractalRegistry()
        for (t, s), e in all_entries.items():
            if t != held_task:
                registry.add(e)
        query = all_entries.get((held_task, 2024))
        if query is None:
            continue
        res = registry.find_nearest(query.signature, k=10)
        out[held_task] = [_strip_seed(e.name) for e in res.entries]
    return out


# ── Redundancy tests ──

def redundant_metadata(target: frozenset[int],
                        top_10_tasks: list[str],
                        tasks: dict[str, frozenset[int]],
                        min_elem_claims: int = 3,
                        mean_jaccard_thresh: float = 0.40,
                        ) -> bool:
    """Test applied to the top-10 candidate pool BEFORE running
    coverage. True iff every target element is claimed by ≥
    min_elem_claims candidates in top-10 AND the top-3 by Jaccard-to-
    target have mean jaccard ≥ mean_jaccard_thresh."""
    pool_labels = [tasks[t] for t in top_10_tasks if t in tasks]
    if not pool_labels:
        return False
    counts = [sum(1 for s in pool_labels if e in s) for e in target]
    if not counts or min(counts) < min_elem_claims:
        return False
    jaccards = sorted([_jaccard(s, target) for s in pool_labels],
                       reverse=True)
    if len(jaccards) < 3:
        return False
    return float(np.mean(jaccards[:3])) >= mean_jaccard_thresh


def redundant_post_hoc(target: frozenset[int],
                        picks_tasks: list[str],
                        tasks: dict[str, frozenset[int]],
                        min_element_cover_thresh: int = 2,
                        mean_jaccard_thresh: float = 0.40,
                        ) -> bool:
    """Test applied to the 3 already-selected picks. True iff every
    target element is claimed by ≥ min_element_cover_thresh of the
    picks AND mean Jaccard to target ≥ mean_jaccard_thresh."""
    picks_labels = [tasks[t] for t in picks_tasks if t in tasks]
    if len(picks_labels) < 1:
        return False
    counts = [sum(1 for s in picks_labels if e in s) for e in target]
    if not counts or min(counts) < min_element_cover_thresh:
        return False
    mean_jac = float(np.mean([_jaccard(s, target) for s in picks_labels]))
    return mean_jac >= mean_jaccard_thresh


# ── Rules ──

def rule_oracle(acc_compose: float, acc_spawn: float) -> str:
    return "compose" if acc_compose > acc_spawn else "spawn"


def rule_always_spawn(*_a, **_k) -> str:
    return "spawn"


def rule_always_compose(*_a, **_k) -> str:
    return "compose"


def rule_simple_N(N: int) -> str:
    return "compose" if N < 100 else "spawn"


def rule_metadata(N: int, target: frozenset[int],
                   top_10_tasks: list[str],
                   tasks: dict[str, frozenset[int]]) -> str:
    if N < 100:
        return "compose"
    if redundant_metadata(target, top_10_tasks, tasks):
        return "compose"
    return "spawn"


def rule_post_hoc(N: int, target: frozenset[int],
                   picks_tasks: list[str],
                   tasks: dict[str, frozenset[int]]) -> str:
    if N < 100:
        return "compose"
    if redundant_post_hoc(target, picks_tasks, tasks):
        return "compose"
    return "spawn"


def _apply_and_score(rule_choice: str, acc_compose: float,
                      acc_spawn: float) -> float:
    return acc_compose if rule_choice == "compose" else acc_spawn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprint9b", type=str,
                        default="results/compose_vs_spawn_budget.json")
    parser.add_argument("--cached-traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--sprint7-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--results-out", type=str,
                        default="results/routing_rules.json")
    args = parser.parse_args(argv)

    with open(args.sprint9b) as f:
        sprint9b = json.load(f)
    tasks = all_binary_tasks()

    print("Computing top-10 per task...")
    top_10 = _build_top10_per_task(
        tasks, Path(args.cached_traj_dir), Path(args.sprint7_dir),
        args.data_dir, args.n_probe, args.probe_seed,
    )
    print(f"Top-10 computed for {len(top_10)} tasks.\n")

    rule_names = ["oracle", "always_compose", "always_spawn",
                   "simple_N", "metadata", "post_hoc"]
    per_rule_totals: dict[str, list[float]] = {r: [] for r in rule_names}
    per_rule_choices: dict[str, list[str]] = {r: [] for r in rule_names}
    per_row: list[dict] = []

    for r in sprint9b["rows"]:
        task = r["task"]
        budget = r["budget_N"]
        target = tasks[task]
        acc_compose = r["acc_compose"]
        acc_spawn = r["acc_spawn"]
        picks_tasks = [_strip_seed(p) for p in r["compose_selected"]]
        top_10_tasks = top_10.get(task, [])

        choices = {
            "oracle": rule_oracle(acc_compose, acc_spawn),
            "always_compose": rule_always_compose(),
            "always_spawn": rule_always_spawn(),
            "simple_N": rule_simple_N(budget),
            "metadata": rule_metadata(budget, target, top_10_tasks, tasks),
            "post_hoc": rule_post_hoc(budget, target, picks_tasks, tasks),
        }
        scores = {name: _apply_and_score(c, acc_compose, acc_spawn)
                  for name, c in choices.items()}
        for name, score in scores.items():
            per_rule_totals[name].append(score)
            per_rule_choices[name].append(choices[name])
        per_row.append({
            "task": task, "budget_N": budget,
            "acc_compose": acc_compose, "acc_spawn": acc_spawn,
            "choices": choices, "scores": scores,
        })

    # ── Overall ──
    print("=" * 72)
    print("  ACHIEVED MEAN ACCURACY BY ROUTING RULE")
    print("=" * 72)
    print(f"  Aggregated over 27 tasks × 5 budgets = 135 decisions.\n")
    print(f"  {'rule':<18s}  {'mean_acc':>10s}  {'compose%':>9s}  "
          f"{'spawn%':>7s}")
    summary: dict[str, dict] = {}
    for name in rule_names:
        accs = np.array(per_rule_totals[name])
        choices = per_rule_choices[name]
        n_compose = sum(1 for c in choices if c == "compose")
        n_spawn = sum(1 for c in choices if c == "spawn")
        mean_acc = float(accs.mean())
        summary[name] = {
            "mean_accuracy": mean_acc,
            "n_compose": n_compose,
            "n_spawn": n_spawn,
            "total_decisions": len(accs),
        }
        print(f"  {name:<18s}  {mean_acc:>10.4f}  "
              f"{100 * n_compose / len(accs):>8.1f}%  "
              f"{100 * n_spawn / len(accs):>6.1f}%")

    # ── Per-budget breakdown ──
    print()
    print("=" * 72)
    print("  PER-BUDGET MEAN ACCURACY")
    print("=" * 72)
    budgets = sorted({r["budget_N"] for r in per_row})
    print(f"  {'N':>6s}  " +
          "  ".join(f"{n:>8s}" for n in rule_names))
    per_budget: dict[int, dict] = {}
    for budget in budgets:
        rows_B = [r for r in per_row if r["budget_N"] == budget]
        row_line = [f"{budget:>6d}"]
        per_budget[budget] = {}
        for name in rule_names:
            mean_acc = float(np.mean([r["scores"][name] for r in rows_B]))
            per_budget[budget][name] = mean_acc
            row_line.append(f"{mean_acc:>8.4f}")
        print("  " + "  ".join(row_line))

    # ── Rule vs rule comparisons ──
    print()
    print("=" * 72)
    print("  PAIRWISE RULE COMPARISONS (mean Δ vs simple_N)")
    print("=" * 72)
    baseline_accs = np.array(per_rule_totals["simple_N"])
    for name in rule_names:
        if name == "simple_N":
            continue
        these = np.array(per_rule_totals[name])
        delta = these - baseline_accs
        n = delta.size
        wins = int((delta > 0).sum())
        ties = int((delta == 0).sum())
        losses = int((delta < 0).sum())
        se = float(delta.std(ddof=1)) / float(np.sqrt(n))
        t = float(delta.mean()) / se if se > 0 else 0.0
        print(f"  {name:<18s} Δ = {delta.mean():+.4f}  "
              f"t = {t:+.2f}  "
              f"({wins}W/{ties}T/{losses}L)")

    # ── Verdict ──
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    simple_acc = summary["simple_N"]["mean_accuracy"]
    metadata_acc = summary["metadata"]["mean_accuracy"]
    post_hoc_acc = summary["post_hoc"]["mean_accuracy"]
    oracle_acc = summary["oracle"]["mean_accuracy"]
    always_spawn_acc = summary["always_spawn"]["mean_accuracy"]
    always_compose_acc = summary["always_compose"]["mean_accuracy"]

    print(f"  Oracle ceiling     : {oracle_acc:.4f}")
    print(f"  Always-spawn floor : {always_spawn_acc:.4f}")
    print(f"  Always-compose     : {always_compose_acc:.4f}")
    print(f"  Simple-N baseline  : {simple_acc:.4f}")
    print(f"  Metadata rule      : {metadata_acc:.4f}   "
          f"Δ vs simple = {metadata_acc - simple_acc:+.4f}")
    print(f"  Post-hoc rule      : {post_hoc_acc:.4f}   "
          f"Δ vs simple = {post_hoc_acc - simple_acc:+.4f}")

    def pct_of_gap(rule_acc: float) -> float:
        gap = oracle_acc - simple_acc
        return (rule_acc - simple_acc) / gap if gap > 0 else 0.0

    print()
    print(f"  Simple-N closes {100 * (simple_acc - always_spawn_acc) / max(oracle_acc - always_spawn_acc, 1e-9):.1f}% "
          f"of always-spawn → oracle gap.")
    print(f"  Metadata rule    : {100 * pct_of_gap(metadata_acc):+.1f}% of simple → oracle gap.")
    print(f"  Post-hoc rule    : {100 * pct_of_gap(post_hoc_acc):+.1f}% of simple → oracle gap.")

    out = {
        "summary": summary,
        "per_budget": {str(k): v for k, v in per_budget.items()},
        "per_row": per_row,
        "oracle": oracle_acc,
        "always_spawn": always_spawn_acc,
        "always_compose": always_compose_acc,
        "simple_N": simple_acc,
        "metadata": metadata_acc,
        "post_hoc": post_hoc_acc,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
