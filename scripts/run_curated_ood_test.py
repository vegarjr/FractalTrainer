"""v3 Sprint 10b — Honest OOD test for the curated registry.

Sprint 10 tested probes INSIDE the anchor region {0, 1, 2, 3, 4} and
found the curated registry saturates compose at oracle accuracy.
Fair criticism: the probe set was chosen to match the registry's
built-in coverage — a best-case scenario.

Sprint 10b stress-tests that claim. We reuse the same two registries:

  Registry A (curated): 45 entries covering subsets of {0, 1, 2, 3, 4}.
  Registry B (uniform): 60 entries (Sprint 7's 20 subset_* tasks).

But now the probe queries are OUTSIDE the anchor region — five
2-subsets drawn from {5, 6, 7, 8, 9}:
    probe_oob_56 = {5, 6}
    probe_oob_78 = {7, 8}
    probe_oob_59 = {5, 9}
    probe_oob_67 = {6, 7}
    probe_oob_89 = {8, 9}

NONE of the curated registry's 15 tasks has a class-1 digit ≥ 5. The
curated registry's best overlap with {5, 6} is zero (on digits 5, 6).
Uniform registry has varied class-1 sets over {0..9} and likely
contains experts with genuine overlap.

Prediction (falsifiable):
  Compose_A on OOD probes: crashes — no redundant coverage available.
  Compose_B on OOD probes: works — usual uniform-registry baseline.
  Spawn: scales the same as Sprint 9b.

If prediction holds, Sprint 10's curated-beats-uniform finding must
be restated: curation helps for IN-DISTRIBUTION queries and hurts
for OUT-OF-DISTRIBUTION ones. Curation is not a free lunch.
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

from fractaltrainer.observer.trainer import InstrumentedTrainer  # noqa: E402
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]
SEEDS = [42, 101, 2024]
ANCHOR = (0, 1, 2, 3, 4)

# Same curated registry as Sprint 10
CURATED_3_SUBSETS = list(itertools.combinations(ANCHOR, 3))
CURATED_4_SUBSETS = list(itertools.combinations(ANCHOR, 4))
CURATED_TASKS: dict[str, tuple[int, ...]] = {}
for s in CURATED_3_SUBSETS:
    CURATED_TASKS[f"anchor3_{''.join(str(d) for d in s)}"] = tuple(s)
for s in CURATED_4_SUBSETS:
    CURATED_TASKS[f"anchor4_{''.join(str(d) for d in s)}"] = tuple(s)

# OOD probes — 2-subsets of {5, 6, 7, 8, 9}
OOD_PROBES: dict[str, tuple[int, ...]] = {
    "probe_oob_56": (5, 6),
    "probe_oob_78": (7, 8),
    "probe_oob_59": (5, 9),
    "probe_oob_67": (6, 7),
    "probe_oob_89": (8, 9),
}

# Sprint 7's uniform registry (same as Sprint 10's Registry B)
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

    cands = []
    for k in (3, 4, 5, 6):
        for c in itertools.combinations(range(10), k):
            if _is_novel(frozenset(c)):
                cands.append(tuple(sorted(c)))
    rng = random.Random(42)
    rng.shuffle(cands)
    return {
        "subset_" + "".join(str(d) for d in s): s
        for s in cands[:20]
    }


UNIFORM_TASKS = _sprint7_new_tasks()


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
    def __init__(self, base, target: set[int]):
        self.base, self.target = base, set(target)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _train_expert(
    name: str, subset: tuple[int, ...], seed: int,
    data_dir: str, out_dir: str,
    n_steps: int = 500, train_size: int = 5000,
    batch_size: int = 64, snapshot_every: int = 10,
) -> tuple[Path, float]:
    from torchvision import datasets, transforms
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=train_size, replace=False)
    loader = DataLoader(Subset(ds, idx.tolist()),
                         batch_size=batch_size, shuffle=True, drop_last=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP()
    hparams = {"learning_rate": 0.01, "batch_size": batch_size,
               "weight_decay": 0.0, "dropout": 0.0,
               "init_seed": seed, "optimizer": "adam"}
    run_id = f"ext_{name}_seed{seed}"
    trainer = InstrumentedTrainer(
        model=model, dataloader=loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=snapshot_every,
        out_dir=out_dir, run_id=run_id,
    )
    t0 = time.time()
    run = trainer.train(n_steps)
    return Path(run.snapshot_path), time.time() - t0


def _make_budget_loader(subset, N, data_dir, seed):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=N, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                       batch_size=min(64, N), shuffle=False)


def _make_eval_loader(subset, n_eval, data_dir, seed):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    ds = RelabeledMNIST(base, set(subset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n_eval, replace=False)
    return DataLoader(Subset(ds, idx.tolist()), batch_size=64, shuffle=False)


def _compute_probs_and_labels(model, loader, n_classes):
    model.eval()
    pl, ll = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)[:, :n_classes]
            pl.append(F.softmax(logits, dim=1).cpu().numpy())
            ll.append(y.cpu().numpy())
    return np.concatenate(pl), np.concatenate(ll)


def _greedy_coverage_select(cand_probs, labels, k):
    n = len(cand_probs)
    if k >= n: return list(range(n))
    selected: list[int] = []
    remaining = set(range(n))
    running = np.zeros_like(cand_probs[0])
    for _ in range(k):
        best_acc, best_i = -1.0, None
        for i in remaining:
            trial = (running + cand_probs[i]) / (len(selected) + 1)
            acc = float((trial.argmax(axis=1) == labels).mean())
            if acc > best_acc: best_acc, best_i = acc, i
        selected.append(best_i); remaining.discard(best_i)
        running = running + cand_probs[best_i]
    return selected


def _ensemble_accuracy(probs_list, labels):
    if not probs_list: return 0.0
    blend = np.mean(probs_list, axis=0)
    return float((blend.argmax(axis=1) == labels).mean())


def _spawn_with_budget(subset, N, seed, data_dir, eval_loader, n_steps=500):
    bl = _make_budget_loader(subset, N, data_dir, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    step = 0
    while step < n_steps:
        for x, y in bl:
            if step >= n_steps: break
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
            step += 1
    p, l = _compute_probs_and_labels(m, eval_loader, n_classes=2)
    return float((p.argmax(axis=1) == l).mean())


def build_registry(task_dict, traj_dir, probe):
    reg = FractalRegistry()
    entries = {}
    for name in task_dict:
        for seed in SEEDS:
            p = traj_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if not p.is_file(): continue
            sig = _signature(_load_model(p), probe)
            e = FractalEntry(name=f"{name}_seed{seed}", signature=sig,
                              metadata={"task": name, "seed": seed,
                                        "trajectory_path": str(p)})
            reg.add(e); entries[(name, seed)] = e
    return reg, entries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", type=str,
                        default="results/sprint10_curated_trajectories")
    parser.add_argument("--sprint7-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--candidate-pool", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[50, 100, 300, 1000, 5000])
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=7777)
    parser.add_argument("--budget-seed", type=int, default=2024)
    parser.add_argument("--spawn-seed", type=int, default=2024)
    parser.add_argument("--results-out", type=str,
                        default="results/curated_ood_test.json")
    args = parser.parse_args(argv)

    curated_dir = Path(args.curated_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)

    # Train OOD probe experts (15)
    print(f"Training {len(OOD_PROBES) * len(SEEDS)} OOD probe experts.")
    for name, subset in OOD_PROBES.items():
        for seed in SEEDS:
            p = curated_dir / f"ext_{name}_seed{seed}_trajectory.npy"
            if p.is_file():
                print(f"  {name} seed={seed}  (cached)")
                continue
            _, elapsed = _train_expert(
                name, subset, seed, args.data_dir, str(curated_dir))
            print(f"  {name} seed={seed}  trained in {elapsed:.1f}s")

    # Build registries (same as Sprint 10)
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    curated_reg, _ = build_registry(CURATED_TASKS, curated_dir, probe)
    uniform_reg, _ = build_registry(UNIFORM_TASKS,
                                      Path(args.sprint7_dir), probe)
    _, ood_entries = build_registry(OOD_PROBES, curated_dir, probe)
    print(f"\nRegistry A (curated): {len(curated_reg)} entries")
    print(f"Registry B (uniform): {len(uniform_reg)} entries")
    print(f"OOD probe entries:   {len(ood_entries)}\n")

    all_rows = []
    spawn_rows = []
    print("=" * 72)
    print("  OOD PROBE BUDGET SWEEP")
    print("=" * 72)

    for probe_name, probe_subset in OOD_PROBES.items():
        q = ood_entries.get((probe_name, 2024))
        if q is None: continue

        eval_loader = _make_eval_loader(
            probe_subset, args.n_eval, args.data_dir, args.eval_seed)
        ll = []
        for _, y in eval_loader: ll.append(y.numpy())
        eval_labels = np.concatenate(ll)

        oracle_entry = ood_entries.get((probe_name, 42))
        oracle_acc = None
        if oracle_entry:
            m = _load_model(Path(oracle_entry.metadata["trajectory_path"]))
            op, _ = _compute_probs_and_labels(m, eval_loader, n_classes=2)
            oracle_acc = float((op.argmax(axis=1) == eval_labels).mean())

        # Compose on both registries
        for reg_name, reg in [("curated", curated_reg),
                               ("uniform", uniform_reg)]:
            res = reg.find_nearest(
                q.signature, k=min(args.candidate_pool, len(reg)))
            cand_names = [e.name for e in res.entries]
            cand_models = [_load_model(Path(e.metadata["trajectory_path"]))
                           for e in res.entries]
            # Eval probs per candidate
            cand_probs_eval = []
            for m in cand_models:
                p, _ = _compute_probs_and_labels(m, eval_loader, n_classes=2)
                cand_probs_eval.append(p)
            acc_top1 = _ensemble_accuracy([cand_probs_eval[0]], eval_labels)
            for budget in args.budgets:
                bl = _make_budget_loader(probe_subset, budget, args.data_dir,
                                           args.budget_seed)
                cand_probs_budget, budget_labels = [], None
                for m in cand_models:
                    p, l = _compute_probs_and_labels(m, bl, n_classes=2)
                    cand_probs_budget.append(p)
                    if budget_labels is None: budget_labels = l
                selected = _greedy_coverage_select(
                    cand_probs_budget, budget_labels, k=args.k)
                acc_compose = _ensemble_accuracy(
                    [cand_probs_eval[i] for i in selected], eval_labels)
                all_rows.append({
                    "registry": reg_name,
                    "query_task": probe_name,
                    "budget_N": budget,
                    "acc_top1": acc_top1,
                    "acc_compose": acc_compose,
                    "compose_selected": [cand_names[i] for i in selected],
                })

        # Spawn (registry-independent)
        for budget in args.budgets:
            acc_spawn = _spawn_with_budget(
                probe_subset, budget, args.spawn_seed, args.data_dir,
                eval_loader)
            spawn_rows.append({
                "query_task": probe_name,
                "budget_N": budget,
                "acc_spawn": acc_spawn,
            })

        # Print per-query
        print(f"\n  query = {probe_name}  (y ∈ {set(probe_subset)})  "
              f"oracle={oracle_acc:.3f}")
        print(f"  {'N':>6s}  {'comp_A':>7s}  {'comp_B':>7s}  {'spawn':>7s}  "
              f"{'top1_A':>7s}  {'top1_B':>7s}")
        spawn_by_N = {r["budget_N"]: r["acc_spawn"]
                      for r in spawn_rows if r["query_task"] == probe_name}
        rows_A = [r for r in all_rows
                  if r["query_task"] == probe_name
                  and r["registry"] == "curated"]
        rows_B = [r for r in all_rows
                  if r["query_task"] == probe_name
                  and r["registry"] == "uniform"]
        for budget in args.budgets:
            cA = next(r for r in rows_A if r["budget_N"] == budget)
            cB = next(r for r in rows_B if r["budget_N"] == budget)
            print(f"  {budget:>6d}  {cA['acc_compose']:>7.3f}  "
                  f"{cB['acc_compose']:>7.3f}  {spawn_by_N[budget]:>7.3f}  "
                  f"{cA['acc_top1']:>7.3f}  {cB['acc_top1']:>7.3f}")

    # Aggregate
    print()
    print("=" * 72)
    print("  AGGREGATE (OOD): compose_curated vs compose_uniform vs spawn")
    print("=" * 72)
    agg = {b: {"cA": [], "cB": [], "spawn": []} for b in args.budgets}
    for r in all_rows:
        agg[r["budget_N"]]["cA" if r["registry"] == "curated" else "cB"
                           ].append(r["acc_compose"])
    for r in spawn_rows:
        agg[r["budget_N"]]["spawn"].append(r["acc_spawn"])
    print(f"  {'N':>6s}  {'comp_A_mean':>12s}  {'comp_B_mean':>12s}  "
          f"{'spawn_mean':>11s}  {'ΔA−B':>7s}  {'ΔA−spawn':>9s}")
    summary = {}
    for budget in args.budgets:
        a = np.array(agg[budget]["cA"])
        b = np.array(agg[budget]["cB"])
        s = np.array(agg[budget]["spawn"])
        summary[budget] = {
            "compose_curated_mean": float(a.mean()),
            "compose_uniform_mean": float(b.mean()),
            "spawn_mean": float(s.mean()),
            "delta_curated_minus_uniform": float(a.mean() - b.mean()),
            "delta_curated_minus_spawn": float(a.mean() - s.mean()),
        }
        print(f"  {budget:>6d}  {a.mean():>12.4f}  {b.mean():>12.4f}  "
              f"{s.mean():>11.4f}  {a.mean() - b.mean():>+7.4f}  "
              f"{a.mean() - s.mean():>+9.4f}")

    # Verdict
    print()
    print("=" * 72)
    print("  VERDICT — does Sprint 10's claim survive OOD?")
    print("=" * 72)
    compose_A_beats_B = all(
        summary[b]["delta_curated_minus_uniform"] > 0
        for b in args.budgets)
    compose_B_beats_A = all(
        summary[b]["delta_curated_minus_uniform"] < 0
        for b in args.budgets)
    if compose_A_beats_B:
        verdict = ("CURATED STILL WINS ON OOD — Sprint 10's claim "
                   "generalizes.")
    elif compose_B_beats_A:
        verdict = ("UNIFORM BEATS CURATED ON OOD — Sprint 10's claim "
                   "was in-distribution-specific. Honest restatement: "
                   "curation helps for known query regions, hurts for "
                   "unknown ones.")
    else:
        verdict = "MIXED — curated wins at some budgets, loses at others."
    print(f"  {verdict}")

    out = {
        "anchor": list(ANCHOR),
        "ood_probes": {k: list(v) for k, v in OOD_PROBES.items()},
        "all_rows": all_rows,
        "spawn_rows": spawn_rows,
        "per_budget_summary": {str(k): v for k, v in summary.items()},
        "verdict": verdict,
    }
    Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {args.results_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
