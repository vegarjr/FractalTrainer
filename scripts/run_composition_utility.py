"""v3 Sprint 5 — Composition utility test.

The growth-decision hierarchy has three verdicts:
    match   — route to nearest expert
    compose — blend top-K nearest
    spawn   — train a new expert

Sprint 4 showed match and spawn fire end-to-end. The compose verdict was
proven in unit tests but never shown to BUY anything — does blending
top-K experts actually beat the single nearest one?

Experimental design (leave-one-task-out, binary tasks only):

  Cached set: 7 binary MNIST tasks × 3 seeds = 21 trained experts.
  For each binary task T (held out):
      registry = 6 other binary tasks × 3 seeds = 18 entries
      query    = T seed=2024 signature
      top_K    = 3 nearest (inverse-distance weights via softmax of -d)
      evaluate on T's relabeled test set:
          acc_top1   — single nearest expert
          acc_blend  — weighted softmax of top-K
          acc_oracle — T seed=42 (upper bound; never in registry here)

  Summary: does acc_blend > acc_top1, and by how much?

We also verify the compose verdict fires end-to-end under narrow bands
(match_threshold=5.0, spawn_threshold=15.0) — cross-task-within-MNIST
distances (7.3-11) land inside [5, 15] and fire compose, which unit
tests already cover but we show the full path here.

Binary tasks use only the first 2 logits from each expert (logits [2..9]
are ignored). This is semantically honest: the expert was trained to
push mass onto its own 2-class label set, and we interpret "class 0
probability" as the expert's vote for the query task's class 0. Errors
come from label-semantic mismatch (e.g. primes ≠ odds); the question is
whether weighted averaging cancels those errors or amplifies them.
"""

from __future__ import annotations

import argparse
import json
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


BINARY_TASKS = [
    "parity", "high_vs_low", "primes_vs_rest", "ones_vs_teens",
    "triangular", "fibonacci", "middle_456",
]
SEEDS = [42, 101, 2024]


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


def _relabel(y: int, task: str) -> int:
    if task == "parity":             return int(y % 2)
    if task == "high_vs_low":        return int(y >= 5)
    if task == "primes_vs_rest":     return int(y in (2, 3, 5, 7))
    if task == "ones_vs_teens":      return int(y <= 4)
    if task == "triangular":         return int(y in (1, 3, 6))
    if task == "fibonacci":          return int(y in (1, 2, 3, 5, 8))
    if task == "middle_456":         return int(y in (4, 5, 6))
    raise ValueError(task)


class RelabeledMNIST(Dataset):
    def __init__(self, base, task): self.base, self.task = base, task
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, _relabel(int(y), self.task)


def _test_loader(task: str, data_dir: str, test_size: int,
                 seed: int) -> DataLoader:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    ds = RelabeledMNIST(base, task)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=test_size, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=False)


def _probe(data_dir: str, n: int, seed: int) -> torch.Tensor:
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
        logits = model(probe)
        probs = F.softmax(logits, dim=1)
    return probs.flatten().cpu().numpy()


def _eval_single(model: nn.Module, loader: DataLoader,
                 n_classes: int) -> float:
    """Accuracy using only the first n_classes logits."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)[:, :n_classes]
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _eval_blend(models: list[nn.Module], weights: np.ndarray,
                loader: DataLoader, n_classes: int) -> float:
    """Weighted softmax blend of top-K experts over first n_classes logits.

    Each expert contributes softmax(logits[:, :n_classes]); the stack is
    averaged under the inverse-distance weights and argmax gives the
    blended prediction.
    """
    for m in models:
        m.eval()
    w_t = torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            probs = []
            for m in models:
                logits = m(x)[:, :n_classes]
                probs.append(F.softmax(logits, dim=1))
            stacked = torch.stack(probs, dim=0)  # (K, B, C)
            blend = (stacked * w_t).sum(dim=0)   # (B, C)
            pred = blend.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def _build_entries(
    traj_dir: Path, tasks: list[str], seeds: list[int],
    probe: torch.Tensor,
) -> dict[tuple[str, int], FractalEntry]:
    out: dict[tuple[str, int], FractalEntry] = {}
    for task in tasks:
        for seed in seeds:
            p = traj_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                print(f"  MISSING {p.name} — skipping")
                continue
            model = _load_model(p)
            sig = _signature(model, probe)
            out[(task, seed)] = FractalEntry(
                name=f"{task}_seed{seed}",
                signature=sig,
                metadata={"task": task, "seed": seed,
                          "trajectory_path": str(p)},
            )
    return out


def _leave_one_task_out(
    all_entries: dict[tuple[str, int], FractalEntry],
    held_task: str, probe: torch.Tensor,
    data_dir: str, test_size: int, compose_k: int,
    compose_temperature: float,
) -> dict:
    # Build registry with all binary tasks except held_task
    registry = FractalRegistry()
    for (t, s), e in all_entries.items():
        if t != held_task:
            registry.add(e)

    query_entry = all_entries[(held_task, 2024)]
    oracle_entry = all_entries[(held_task, 42)]

    # Compose weights: top-K nearest, softmax over -distance
    retrieval, weights = registry.composite_weights(
        query_entry.signature, k=compose_k,
        temperature=compose_temperature,
    )
    top_k_entries = retrieval.entries
    top_k_distances = retrieval.distances

    # Load models
    top_k_models = [_load_model(Path(e.metadata["trajectory_path"]))
                    for e in top_k_entries]
    oracle_model = _load_model(Path(oracle_entry.metadata["trajectory_path"]))

    loader = _test_loader(held_task, data_dir, test_size, seed=7777)

    acc_top1 = _eval_single(top_k_models[0], loader, n_classes=2)
    acc_blend = _eval_blend(top_k_models, weights, loader, n_classes=2)
    acc_oracle = _eval_single(oracle_model, loader, n_classes=2)

    return {
        "held_task": held_task,
        "query": query_entry.name,
        "nearest": top_k_entries[0].name,
        "top_k_names": [e.name for e in top_k_entries],
        "top_k_distances": [float(d) for d in top_k_distances],
        "top_k_weights": [float(w) for w in weights],
        "acc_top1": acc_top1,
        "acc_blend": acc_blend,
        "acc_oracle": acc_oracle,
        "delta_blend_minus_top1": acc_blend - acc_top1,
        "gap_to_oracle": acc_oracle - acc_blend,
    }


def _verify_compose_verdict_fires(
    all_entries: dict[tuple[str, int], FractalEntry],
    match_threshold: float, spawn_threshold: float,
) -> dict:
    """Show the compose verdict actually fires end-to-end.

    Use a leave-one-task-out registry (so query is cross-task), with
    bands wide enough that cross-task distances land in compose.
    """
    held_task = "primes_vs_rest"  # known cross-task nearest ~7-10
    registry = FractalRegistry()
    for (t, s), e in all_entries.items():
        if t != held_task:
            registry.add(e)
    query_entry = all_entries[(held_task, 2024)]
    decision = registry.decide(
        query_entry.signature,
        match_threshold=match_threshold,
        spawn_threshold=spawn_threshold,
        compose_k=3,
    )
    return {
        "query": query_entry.name,
        "match_threshold": match_threshold,
        "spawn_threshold": spawn_threshold,
        "verdict": decision.verdict,
        "min_distance": decision.min_distance,
        "composite_entries": (
            [e.name for e in decision.composite_entries]
            if decision.composite_entries else None),
        "composite_weights": (
            [float(w) for w in decision.composite_weights]
            if decision.composite_weights is not None else None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--compose-k", type=int, default=3)
    parser.add_argument("--compose-temperature", type=float, default=1.0)
    parser.add_argument("--match-threshold", type=float, default=5.0,
                        help="For the compose-verdict verification.")
    parser.add_argument("--spawn-threshold", type=float, default=15.0,
                        help="For the compose-verdict verification.")
    parser.add_argument("--results-out", type=str,
                        default="results/composition_utility.json")
    args = parser.parse_args(argv)

    print(f"Binary tasks: {BINARY_TASKS}")
    print(f"Seeds: {SEEDS}  →  {len(BINARY_TASKS) * len(SEEDS)} entries")
    print(f"compose_k={args.compose_k}, temperature={args.compose_temperature}")
    print()

    probe = _probe(args.data_dir, args.n_probe, args.probe_seed)
    all_entries = _build_entries(Path(args.traj_dir), BINARY_TASKS, SEEDS, probe)
    print(f"Built {len(all_entries)} binary-task entries.\n")

    # ── Main experiment: leave-one-task-out utility comparison ────────
    print("=" * 72)
    print("  LEAVE-ONE-TASK-OUT UTILITY COMPARISON")
    print("=" * 72)
    print("  Registry = 6 binary tasks × 3 seeds = 18 entries (T held out).")
    print("  Query    = T seed=2024. Top-3 blend vs top-1 vs oracle on T test.")
    print()

    rows: list[dict] = []
    for held_task in BINARY_TASKS:
        r = _leave_one_task_out(
            all_entries, held_task, probe,
            args.data_dir, args.test_size,
            args.compose_k, args.compose_temperature,
        )
        rows.append(r)
        print(f"  {held_task:<16s}  top1={r['acc_top1']:.3f}  "
              f"blend={r['acc_blend']:.3f}  oracle={r['acc_oracle']:.3f}  "
              f"(Δ blend−top1 = {r['delta_blend_minus_top1']:+.3f})")
        print(f"    nearest: {r['nearest']}  "
              f"(d={r['top_k_distances'][0]:.3f}, w={r['top_k_weights'][0]:.3f})")
        for name, d, w in zip(r["top_k_names"][1:],
                               r["top_k_distances"][1:],
                               r["top_k_weights"][1:]):
            print(f"    next   : {name}  (d={d:.3f}, w={w:.3f})")

    # Summary stats
    top1s = np.array([r["acc_top1"] for r in rows])
    blends = np.array([r["acc_blend"] for r in rows])
    oracles = np.array([r["acc_oracle"] for r in rows])
    deltas = blends - top1s
    wins = int((deltas > 0).sum())
    ties = int((deltas == 0).sum())
    losses = int((deltas < 0).sum())

    print()
    print(f"  Mean top1    : {top1s.mean():.4f}")
    print(f"  Mean blend   : {blends.mean():.4f}")
    print(f"  Mean oracle  : {oracles.mean():.4f}  (upper bound)")
    print(f"  Mean Δ (blend − top1): {deltas.mean():+.4f}   "
          f"std={deltas.std(ddof=1):.4f}")
    print(f"  Sign test: blend beats top-1 in {wins}/{len(rows)}  "
          f"(ties: {ties}, losses: {losses})")

    # ── Side test: compose verdict fires end-to-end ───────────────────
    print()
    print("=" * 72)
    print("  COMPOSE VERDICT END-TO-END (narrow-band thresholds)")
    print("=" * 72)
    compose_check = _verify_compose_verdict_fires(
        all_entries,
        match_threshold=args.match_threshold,
        spawn_threshold=args.spawn_threshold,
    )
    print(f"  query: {compose_check['query']}")
    print(f"  bands: match ≤ {compose_check['match_threshold']}, "
          f"compose ≤ {compose_check['spawn_threshold']}")
    print(f"  verdict: {compose_check['verdict']}   "
          f"min_distance={compose_check['min_distance']:.3f}")
    if compose_check["composite_entries"]:
        print(f"  composite:")
        for n, w in zip(compose_check["composite_entries"],
                         compose_check["composite_weights"]):
            print(f"    {n:<30s}  weight={w:.3f}")

    # ── Verdict ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std(ddof=1))
    se_delta = std_delta / np.sqrt(len(deltas))
    t_stat = mean_delta / se_delta if se_delta > 0 else 0.0
    # Two-sigma paired threshold (n=7, so this is a rough proxy; real
    # statistical power at n=7 is low, so we want a strong effect).
    if t_stat > 2.0 and wins > len(rows) / 2:
        verdict = "COMPOSE HAS UTILITY"
        reason = (f"blend beats top-1 on {wins}/{len(rows)} tasks, "
                  f"Δ=+{mean_delta:.3f} (t={t_stat:.2f}, se={se_delta:.3f})")
    elif t_stat < -2.0 and losses > len(rows) / 2:
        verdict = "COMPOSE HURTS"
        reason = (f"blend worse than top-1, Δ={mean_delta:+.3f} "
                  f"(t={t_stat:.2f}, se={se_delta:.3f})")
    else:
        verdict = "NULL"
        reason = (f"Δ={mean_delta:+.3f} within ±2·se (se={se_delta:.3f}, "
                  f"t={t_stat:.2f}) — no measurable lift or harm from "
                  f"compose over top-1 at n={len(rows)}")
    print(f"  {verdict}: {reason}")
    print(f"  (paired t-stat = {t_stat:.2f}, se = {se_delta:.4f})")

    results = {
        "binary_tasks": BINARY_TASKS,
        "seeds": SEEDS,
        "compose_k": args.compose_k,
        "compose_temperature": args.compose_temperature,
        "test_size": args.test_size,
        "rows": rows,
        "summary": {
            "mean_top1": float(top1s.mean()),
            "mean_blend": float(blends.mean()),
            "mean_oracle": float(oracles.mean()),
            "mean_delta_blend_minus_top1": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)),
            "n_wins_for_blend": wins,
            "n_ties": ties,
            "n_losses": losses,
            "verdict": verdict,
            "reason": reason,
            "paired_t_stat": float(t_stat),
            "paired_se": float(se_delta),
        },
        "compose_verdict_check": compose_check,
    }

    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
