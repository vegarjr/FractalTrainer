"""v3 Sprint 7 supplement — Jaccard overlap vs signature distance.

The Sprint 7 review claimed "signature space encodes task-function
kinship": cross-task nearest neighbors in signature space reflect
digit-subset overlap. E.g., parity → subset_123679 (4 of 5 odd
digits in common).

A rigorous test: compute pairwise **Jaccard overlap** between every
pair of binary tasks' class-1 digit sets, and pairwise **signature
distance** between the corresponding trained experts. If the claim
holds, these should correlate negatively (high overlap → low
distance).

Tasks: 27 binary MNIST labelings
  - 7 from the Sprint 3 set: parity, high_vs_low, primes_vs_rest,
    ones_vs_teens, triangular, fibonacci, middle_456
  - 20 from Sprint 7: subset_{...}

For each task T, the "class 1" digit set is a subset of {0..9}.
Jaccard(T1, T2) = |S1 ∩ S2| / |S1 ∪ S2|.

Signature distance: L2 between seed=42 signatures on the standard
probe batch (100 MNIST images). Restrict to one seed per task to
keep this a 27×27 matrix rather than 81×81 (avoids averaging over
within-task variance).

Spearman ρ (rank correlation) is the headline statistic — robust to
the nonlinear relationship between Jaccard and L2.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]


# Binary-task label sets (class-1 digits for each)
EXISTING_BINARY: dict[str, frozenset[int]] = {
    "parity":          frozenset({1, 3, 5, 7, 9}),
    "high_vs_low":     frozenset({5, 6, 7, 8, 9}),
    "primes_vs_rest":  frozenset({2, 3, 5, 7}),
    "ones_vs_teens":   frozenset({0, 1, 2, 3, 4}),
    "triangular":      frozenset({1, 3, 6}),
    "fibonacci":       frozenset({1, 2, 3, 5, 8}),
    "middle_456":      frozenset({4, 5, 6}),
}


def _sprint7_new_tasks() -> dict[str, frozenset[int]]:
    """Reproduce the Sprint 7 task generator exactly."""
    import random as _r

    EXISTING_BINARY_SUBSETS = list(EXISTING_BINARY.values())

    def _is_novel(s: frozenset[int]) -> bool:
        full = frozenset(range(10))
        for ex in EXISTING_BINARY_SUBSETS:
            if s == ex or s == (full - ex):
                return False
        return True

    candidates = []
    for k in (3, 4, 5, 6):
        for c in itertools.combinations(range(10), k):
            if _is_novel(frozenset(c)):
                candidates.append(tuple(sorted(c)))
    rng = _r.Random(42)
    rng.shuffle(candidates)
    chosen = candidates[:20]
    out: dict[str, frozenset[int]] = {}
    for subset in chosen:
        name = "subset_" + "".join(str(d) for d in subset)
        out[name] = frozenset(subset)
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
        return 1.0  # both empty — treat as identical
    return len(a & b) / len(u)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Spearman rank correlation via Pearson on ranks."""
    assert x.shape == y.shape
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denom == 0:
        return 0.0, int(x.size)
    return float((rx * ry).sum() / denom), int(x.size)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xa = x - x.mean(); ya = y - y.mean()
    denom = float(np.sqrt((xa ** 2).sum() * (ya ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((xa * ya).sum() / denom)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--new-traj-dir", type=str,
                        default="results/sprint7_v3_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--signature-seed", type=int, default=42,
                        help="Which training seed's signature to use")
    parser.add_argument("--results-out", type=str,
                        default="results/jaccard_vs_distance.json")
    args = parser.parse_args(argv)

    # Build task → label-set mapping
    tasks: dict[str, frozenset[int]] = {}
    tasks.update(EXISTING_BINARY)
    tasks.update(_sprint7_new_tasks())
    names = sorted(tasks.keys())
    print(f"Binary tasks: {len(names)}")

    # Load signatures (seed=42 per task)
    probe = _mnist_probe(args.data_dir, args.n_probe, args.probe_seed)
    sigs: dict[str, np.ndarray] = {}
    for name in names:
        # Find the trajectory file
        if name in EXISTING_BINARY:
            p = Path(args.cached_traj_dir) / f"ext_{name}_seed{args.signature_seed}_trajectory.npy"
        else:
            p = Path(args.new_traj_dir) / f"ext_{name}_seed{args.signature_seed}_trajectory.npy"
        if not p.is_file():
            print(f"  MISSING  {p}")
            continue
        sigs[name] = _signature(_load_model(p), probe)
    print(f"Signatures loaded: {len(sigs)}/{len(names)}")
    names = [n for n in names if n in sigs]

    # Compute pairwise Jaccard and signature distance
    pairs: list[dict] = []
    for n1, n2 in itertools.combinations(names, 2):
        j = _jaccard(tasks[n1], tasks[n2])
        d = float(np.linalg.norm(sigs[n1] - sigs[n2]))
        pairs.append({
            "task_a": n1, "task_b": n2,
            "labels_a": sorted(tasks[n1]),
            "labels_b": sorted(tasks[n2]),
            "jaccard": j,
            "distance": d,
        })
    print(f"Pairs: {len(pairs)}")

    jaccards = np.array([p["jaccard"] for p in pairs], dtype=np.float64)
    distances = np.array([p["distance"] for p in pairs], dtype=np.float64)

    rho, n = _spearman_rho(jaccards, distances)
    r = _pearson_r(jaccards, distances)

    print()
    print("=" * 72)
    print("  JACCARD OVERLAP vs SIGNATURE DISTANCE")
    print("=" * 72)
    print(f"  Pairs: {n}")
    print(f"  Jaccard range:  [{jaccards.min():.3f}, {jaccards.max():.3f}]  "
          f"mean={jaccards.mean():.3f}")
    print(f"  Distance range: [{distances.min():.3f}, {distances.max():.3f}]  "
          f"mean={distances.mean():.3f}")
    print(f"  Spearman ρ (jaccard vs distance): {rho:+.3f}")
    print(f"  Pearson  r (jaccard vs distance): {r:+.3f}")

    # Quartile analysis: mean distance in top-25% Jaccard vs bottom-25%
    sorted_by_j = sorted(pairs, key=lambda p: p["jaccard"])
    q = len(sorted_by_j) // 4
    low_j = sorted_by_j[:q]           # lowest-overlap
    high_j = sorted_by_j[-q:]          # highest-overlap
    mean_d_low = float(np.mean([p["distance"] for p in low_j]))
    mean_d_high = float(np.mean([p["distance"] for p in high_j]))
    mean_j_low = float(np.mean([p["jaccard"] for p in low_j]))
    mean_j_high = float(np.mean([p["jaccard"] for p in high_j]))
    print()
    print(f"  Bottom quartile Jaccard ({q} pairs, mean J={mean_j_low:.3f}): "
          f"mean signature distance = {mean_d_low:.3f}")
    print(f"  Top quartile    Jaccard ({q} pairs, mean J={mean_j_high:.3f}): "
          f"mean signature distance = {mean_d_high:.3f}")
    print(f"  Distance lift at high-Jaccard: -{mean_d_low - mean_d_high:.3f}")

    # Top-5 most overlapping pairs and their distances
    print()
    print("  Top 5 most-overlapping pairs:")
    for p in sorted_by_j[-5:][::-1]:
        print(f"    J={p['jaccard']:.3f}  d={p['distance']:.3f}  "
              f"{p['task_a']} ({p['labels_a']}) ↔ "
              f"{p['task_b']} ({p['labels_b']})")
    print()
    print("  Top 5 most-distant (lowest-overlap) pairs:")
    for p in sorted_by_j[:5]:
        print(f"    J={p['jaccard']:.3f}  d={p['distance']:.3f}  "
              f"{p['task_a']} ({p['labels_a']}) ↔ "
              f"{p['task_b']} ({p['labels_b']})")

    # Verdict
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if rho < -0.4:
        verdict = "CLAIM SUPPORTED"
        reason = (f"Spearman ρ={rho:.3f} — strong negative correlation. "
                  "Signature space encodes task-function kinship.")
    elif rho < -0.15:
        verdict = "CLAIM WEAKLY SUPPORTED"
        reason = (f"Spearman ρ={rho:.3f} — small but real negative "
                  "correlation. Kinship exists but is noisy.")
    elif abs(rho) <= 0.15:
        verdict = "CLAIM UNSUPPORTED"
        reason = (f"Spearman ρ={rho:.3f} — no meaningful correlation. "
                  "Sprint 7 semantic-kinship story was post-hoc.")
    else:
        verdict = "UNEXPECTED"
        reason = (f"Spearman ρ={rho:+.3f} — positive correlation means "
                  "similar labels are DISTANT in signature space. "
                  "Something is backwards.")
    print(f"  {verdict}: {reason}")

    out = {
        "n_tasks": len(names),
        "task_labels": {k: sorted(v) for k, v in tasks.items() if k in sigs},
        "signature_seed": args.signature_seed,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "stats": {
            "jaccard_mean": float(jaccards.mean()),
            "jaccard_range": [float(jaccards.min()), float(jaccards.max())],
            "distance_mean": float(distances.mean()),
            "distance_range": [float(distances.min()), float(distances.max())],
            "spearman_rho": float(rho),
            "pearson_r": float(r),
            "quartile_distance_lift":
                float(mean_d_low - mean_d_high),
            "bottom_quartile_jaccard_mean_distance": mean_d_low,
            "top_quartile_jaccard_mean_distance": mean_d_high,
        },
        "verdict": verdict,
        "reason": reason,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
