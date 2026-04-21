"""v3 Sprint 17 follow-up — Direction B validation.

Compare softmax signatures (Sprint 3+ default) vs penultimate-layer
signatures on the same 15-expert registry used by run_fractal_demo.py.

For each signature mode, report:
  1. Within-task pairwise L2 distance stats (smaller is better —
     same-task experts should cluster).
  2. Cross-task pairwise L2 distance stats (larger is better —
     different-task experts should separate).
  3. Gap = cross_mean - within_mean (bigger gap = cleaner separation).
  4. Spearman ρ between Jaccard(class-1-sets) and signature distance.
     Sprint 7b baseline: ρ = -0.85 for softmax. Penultimate should be
     at least as negatively correlated for classification to claim
     "drop-in replacement".

Go/no-go gate for adopting penultimate signatures:
  - Gap must be positive (otherwise separation fails).
  - Gap should be within 2× of softmax gap (otherwise penultimate is
    substantially worse at classification — tolerable only if the
    non-classification use case justifies it).
  - Spearman ρ must be negative and |ρ| ≥ 0.7.

Run:
    python scripts/run_latent_signatures_validation.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.integration import (  # noqa: E402
    ContextAwareMLP,
    penultimate_signature,
    softmax_signature,
)
from fractaltrainer.integration.signatures import (  # noqa: E402
    penultimate_normalized_signature,
    penultimate_softmax_signature,
)


# Same seed tasks as run_fractal_demo.py
SEED_TASKS: dict[str, tuple[int, ...]] = {
    "subset_01234": (0, 1, 2, 3, 4),
    "subset_56789": (5, 6, 7, 8, 9),
    "subset_024":   (0, 2, 4),
    "subset_13579": (1, 3, 5, 7, 9),
    "subset_357":   (3, 5, 7),
}
SEEDS = [42, 101, 2024]


class RelabeledMNIST(Dataset):
    def __init__(self, base, target):
        self.base = base
        self.target = set(int(d) for d in target)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(int(y) in self.target)


def _mnist_probe(data_dir: str, n: int, seed: int) -> torch.Tensor:
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=False, download=True, transform=t)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(base), size=n, replace=False)
    return torch.stack([base[i][0] for i in idx.tolist()], dim=0)


def _train_loader(target, n, batch_size, data_dir, seed):
    from torchvision import datasets, transforms
    t = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    base = datasets.MNIST(data_dir, train=True, download=True, transform=t)
    ds = RelabeledMNIST(base, target)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    return DataLoader(Subset(ds, idx.tolist()),
                       batch_size=batch_size, shuffle=True, drop_last=True)


def _train_expert(target, seed, n_steps, train_size, batch_size, data_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ContextAwareMLP(context_scale=0.0)
    loader = _train_loader(target, train_size, batch_size, data_dir, seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    it = iter(loader); step = 0
    while step < n_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        logits = model(x, context=None)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
    return model


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return 1.0 if not u else len(a & b) / len(u)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Simple rank-correlation — no scipy dependency."""
    def rank(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r
    ra, rb = rank(a), rank(b)
    ra_c = ra - ra.mean()
    rb_c = rb - rb.mean()
    num = float((ra_c * rb_c).sum())
    den = float(np.sqrt((ra_c ** 2).sum() * (rb_c ** 2).sum()))
    return num / den if den > 0 else 0.0


def _distance_stats(entries_by_task: dict[str, list[np.ndarray]],
                     task_labels: dict[str, set]):
    """Return (within_arr, cross_arr, jaccard_arr, dist_arr).

    within_arr: within-task pairwise L2s
    cross_arr:  cross-task pairwise L2s
    jaccard_arr, dist_arr: aligned arrays for correlation — one entry
       per (task_i, task_j, seed_i, seed_j) pair with i≠j.
    """
    within: list[float] = []
    for sigs in entries_by_task.values():
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                within.append(float(np.linalg.norm(sigs[i] - sigs[j])))

    cross: list[float] = []
    jac: list[float] = []
    dst: list[float] = []
    tasks = list(entries_by_task.keys())
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t_i, t_j = tasks[i], tasks[j]
            l_i, l_j = task_labels[t_i], task_labels[t_j]
            j_score = _jaccard(l_i, l_j)
            for a in entries_by_task[t_i]:
                for b in entries_by_task[t_j]:
                    d = float(np.linalg.norm(a - b))
                    cross.append(d)
                    jac.append(j_score)
                    dst.append(d)

    return (np.array(within), np.array(cross),
            np.array(jac), np.array(dst))


def _format_stats(name, within, cross, jac, dst):
    w_mean, w_std = float(within.mean()), float(within.std())
    c_mean, c_std = float(cross.mean()), float(cross.std())
    gap = c_mean - w_mean
    # Spearman ρ between jaccard and cross-task distance
    rho = _spearman(jac, dst)
    return {
        "mode": name,
        "within_task": {
            "n": int(within.size), "min": float(within.min()),
            "mean": w_mean, "max": float(within.max()), "std": w_std,
        },
        "cross_task": {
            "n": int(cross.size), "min": float(cross.min()),
            "mean": c_mean, "max": float(cross.max()), "std": c_std,
        },
        "gap_cross_minus_within": gap,
        "spearman_jaccard_vs_distance": rho,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--results-out",
                        default="results/latent_signatures_validation.json")
    args = parser.parse_args(argv)

    print("=" * 72)
    print("  LATENT SIGNATURES VALIDATION — softmax vs penultimate")
    print("=" * 72)

    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    models_by_task: dict[str, list[ContextAwareMLP]] = {}
    task_labels: dict[str, set] = {name: set(tgt) for name, tgt in SEED_TASKS.items()}

    print(f"\n[1/3] training {len(SEED_TASKS) * len(SEEDS)} seed experts "
          f"(n_steps={args.n_steps}, train_size={args.train_size})...")
    t0 = time.time()
    i = 0
    total = len(SEED_TASKS) * len(SEEDS)
    for task_name, target in SEED_TASKS.items():
        models_by_task[task_name] = []
        for seed in SEEDS:
            i += 1
            print(f"  [{i}/{total}] {task_name} seed={seed}")
            m = _train_expert(target, seed, args.n_steps,
                               args.train_size, args.batch_size,
                               args.data_dir)
            models_by_task[task_name].append(m)
    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.1f}s")

    print("\n[2/3] computing signatures under four modes...")
    modes = {
        "softmax":                softmax_signature,
        "penultimate":            penultimate_signature,
        "penultimate_normalized": penultimate_normalized_signature,
        "penultimate_softmax":    penultimate_softmax_signature,
    }
    sigs_by_mode_by_task: dict[str, dict[str, list[np.ndarray]]] = {
        mode: {} for mode in modes
    }
    for task_name, models in models_by_task.items():
        for mode, fn in modes.items():
            sigs_by_mode_by_task[mode][task_name] = [fn(m, probe) for m in models]
    for mode in modes:
        any_sig = next(iter(sigs_by_mode_by_task[mode].values()))[0]
        print(f"  {mode:<24s} signature dim: {any_sig.size}")

    print("\n[3/3] distance stats + Jaccard correlation...")
    stats_by_mode: dict[str, dict] = {}
    for mode in modes:
        stats_tuple = _distance_stats(sigs_by_mode_by_task[mode], task_labels)
        stats_by_mode[mode] = _format_stats(mode, *stats_tuple)

    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    def _fmt(stats):
        w = stats["within_task"]; c = stats["cross_task"]
        return (f"  {stats['mode']:<24s}  "
                f"within μ={w['mean']:>9.3f}±{w['std']:>6.3f}  "
                f"cross μ={c['mean']:>9.3f}±{c['std']:>6.3f}  "
                f"gap={stats['gap_cross_minus_within']:+7.3f}  "
                f"ρ={stats['spearman_jaccard_vs_distance']:+.3f}")

    print()
    for mode in modes:
        print(_fmt(stats_by_mode[mode]))

    sm_stats = stats_by_mode["softmax"]
    sm_gap = sm_stats["gap_cross_minus_within"]

    verdicts: dict[str, dict] = {}
    any_pass = False
    print("\n" + "-" * 72)
    print("  GO/NO-GO gate per mode (gap>0, gap≥0.5×softmax_gap, ρ<0, |ρ|≥0.7)")
    for mode, stats in stats_by_mode.items():
        if mode == "softmax":
            continue
        gap = stats["gap_cross_minus_within"]
        rho = stats["spearman_jaccard_vs_distance"]
        gap_ratio = gap / sm_gap if sm_gap > 0 else None
        v = {
            "gap": gap,
            "gap_ratio_vs_softmax": gap_ratio,
            "spearman_rho": rho,
            "gap_positive": gap > 0,
            "gap_ratio_ge_0_5": (gap_ratio is not None and gap_ratio >= 0.5),
            "rho_negative": rho < 0,
            "rho_magnitude_ge_0_7": abs(rho) >= 0.7,
        }
        v["gate_pass"] = (v["gap_positive"] and v["gap_ratio_ge_0_5"]
                          and v["rho_negative"] and v["rho_magnitude_ge_0_7"])
        verdicts[mode] = v
        any_pass = any_pass or v["gate_pass"]
        print(f"  {mode:<24s}  gap={gap:+.3f}  ratio={gap_ratio:+.3f}  "
              f"ρ={rho:+.3f}  {'PASS' if v['gate_pass'] else 'FAIL'}")

    print("\n" + ("Any non-softmax mode passes." if any_pass
                   else "No non-softmax mode passes — softmax remains default."))

    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "signatures": stats_by_mode,
        "verdicts": verdicts,
        "any_pass": bool(any_pass),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
