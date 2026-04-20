"""v3 Sprint 6 — Adaptive threshold calibration demo.

Replaces the hand-tuned Sprint 4 defaults (match=5.0, spawn=7.0) with
thresholds computed from the registry's own within-task and cross-task
distance distributions.

Runs calibration under three configurations:

  1. Sprint 3 MVP registry  (22 entries: 11 tasks × 2 seeds).
  2. Full registry          (33 entries: 11 tasks × 3 seeds).
  3. Binary-task-only       (21 entries: 7 binary tasks × 3 seeds).

For each: report (match_threshold, spawn_threshold), compare to the
hand-tuned (5.0, 7.0), and re-run a leave-one-seed-out retrieval check
with calibrated thresholds to confirm same-task queries still land in
match (high-percentile-catching).

Expected outcome: calibrated thresholds should land roughly in the
Sprint 3-observed (3.43, 7.32) gap — likely ~(3.5-4.5, 6.0-7.5) —
reproducing the hand-tuned values without tuning.
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


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402


LAYER_SHAPES = [
    ((64, 784), "net.0.weight"), ((64,), "net.0.bias"),
    ((32, 64),  "net.2.weight"), ((32,), "net.2.bias"),
    ((10, 32),  "net.4.weight"), ((10,), "net.4.bias"),
]

MNIST_TASKS = [
    "digit_class", "parity", "high_vs_low", "mod3", "mod5",
    "primes_vs_rest", "ones_vs_teens", "triangular",
    "fibonacci", "middle_456",
]
FASHION_TASKS = ["fashion_class"]
ALL_TASKS = MNIST_TASKS + FASHION_TASKS
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


def _probe_batch(data_dir: str, n: int, seed: int) -> torch.Tensor:
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


def _build_entries(traj_dir: Path, tasks: list[str], seeds: list[int],
                   probe: torch.Tensor) -> list[FractalEntry]:
    out: list[FractalEntry] = []
    for task in tasks:
        for seed in seeds:
            p = traj_dir / f"ext_{task}_seed{seed}_trajectory.npy"
            if not p.is_file():
                continue
            sig = _signature(_load_model(p), probe)
            out.append(FractalEntry(
                name=f"{task}_seed{seed}", signature=sig,
                metadata={"task": task, "seed": seed,
                          "trajectory_path": str(p)},
            ))
    return out


def _calibrate_and_report(registry: FractalRegistry, label: str) -> dict:
    print()
    print("=" * 72)
    print(f"  CALIBRATION — {label}")
    print("=" * 72)
    print(f"  Registry size: {len(registry)}")
    cal = registry.calibrate_thresholds()
    d = cal.to_dict()
    print(f"  Tasks: {cal.n_tasks}")
    print(f"  Within-task pairs:  n={d['within_distance_stats']['n']}, "
          f"min={d['within_distance_stats']['min']:.3f}, "
          f"mean={d['within_distance_stats']['mean']:.3f}, "
          f"max={d['within_distance_stats']['max']:.3f}")
    print(f"  Cross-task pairs:   n={d['cross_distance_stats']['n']}, "
          f"min={d['cross_distance_stats']['min']:.3f}, "
          f"mean={d['cross_distance_stats']['mean']:.3f}, "
          f"max={d['cross_distance_stats']['max']:.3f}")
    print(f"  Calibrated match_threshold = {cal.match_threshold:.3f} "
          f"(within p{cal.within_percentile:.0f})")
    print(f"  Calibrated spawn_threshold = {cal.spawn_threshold:.3f} "
          f"(cross p{cal.cross_percentile:.0f})")
    print(f"  Overlap: {cal.overlap}")
    print(f"  Gap width: {cal.spawn_threshold - cal.match_threshold:.3f}")
    print(f"  (Sprint 4 hand-tuned defaults: match=5.0, spawn=7.0, gap=2.0)")
    return {"label": label, "registry_size": len(registry), **d}


def _retrieval_check(registry: FractalRegistry,
                     held_out: list[FractalEntry],
                     match_threshold: float, spawn_threshold: float,
                     ) -> dict:
    """Leave-one-seed-out: does calibrated match_threshold catch
    same-task queries?"""
    rows: list[dict] = []
    for e in held_out:
        d = registry.decide(
            e.signature,
            match_threshold=match_threshold,
            spawn_threshold=spawn_threshold,
            compose_k=3,
        )
        nearest = d.retrieval.nearest if d.retrieval else None
        same_task = (nearest is not None
                     and nearest.metadata.get("task")
                     == e.metadata.get("task"))
        rows.append({
            "held": e.name, "verdict": d.verdict,
            "min_distance": d.min_distance,
            "nearest": nearest.name if nearest else None,
            "same_task": same_task,
        })
    n_match = sum(r["verdict"] == "match" for r in rows)
    n_compose = sum(r["verdict"] == "compose" for r in rows)
    n_spawn = sum(r["verdict"] == "spawn" for r in rows)
    n_correct_task = sum(r["same_task"] for r in rows)
    return {
        "rows": rows,
        "n_total": len(rows),
        "n_match_verdict": n_match,
        "n_compose_verdict": n_compose,
        "n_spawn_verdict": n_spawn,
        "n_correct_nearest_task": n_correct_task,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str,
                        default="results/discriminability_extended_trajectories")
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--results-out", type=str,
                        default="results/calibration_demo.json")
    args = parser.parse_args(argv)

    probe = _probe_batch(args.data_dir, args.n_probe, args.probe_seed)

    # Build all 33 entries
    all_entries = _build_entries(
        Path(args.traj_dir), ALL_TASKS, SEEDS, probe,
    )
    print(f"Loaded {len(all_entries)} entries from cached trajectories.\n")

    # ── 1. Sprint 3 MVP layout: seeds 42+101, 11 tasks (22 entries) ──
    mvp_reg = FractalRegistry()
    mvp_held: list[FractalEntry] = []
    for e in all_entries:
        if e.metadata["seed"] == 2024:
            mvp_held.append(e)
        else:
            mvp_reg.add(e)
    mvp_cal = _calibrate_and_report(mvp_reg, "Sprint 3 MVP (22 entries)")
    mvp_retrieval = _retrieval_check(
        mvp_reg, mvp_held,
        mvp_cal["match_threshold"], mvp_cal["spawn_threshold"],
    )
    print(f"  Leave-one-seed-out retrieval ({mvp_retrieval['n_total']} queries):")
    print(f"    verdicts: match={mvp_retrieval['n_match_verdict']}, "
          f"compose={mvp_retrieval['n_compose_verdict']}, "
          f"spawn={mvp_retrieval['n_spawn_verdict']}")
    print(f"    nearest-task correct: "
          f"{mvp_retrieval['n_correct_nearest_task']}"
          f"/{mvp_retrieval['n_total']}")

    # ── 2. Full registry: all 33 entries ──────────────────────────
    full_reg = FractalRegistry()
    for e in all_entries:
        full_reg.add(e)
    full_cal = _calibrate_and_report(full_reg, "Full (33 entries)")

    # ── 3. Binary tasks only: 21 entries ──────────────────────────
    bin_reg = FractalRegistry()
    for e in all_entries:
        if e.metadata["task"] in BINARY_TASKS:
            bin_reg.add(e)
    bin_cal = _calibrate_and_report(bin_reg, "Binary tasks only (21 entries)")

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'layout':<35s} {'match':>8s} {'spawn':>8s} {'gap':>8s}")
    print(f"  {'hand-tuned (Sprint 4)':<35s} "
          f"{'5.000':>8s} {'7.000':>8s} {'2.000':>8s}")
    for cal in (mvp_cal, full_cal, bin_cal):
        gap = cal["spawn_threshold"] - cal["match_threshold"]
        print(f"  {cal['label']:<35s} "
              f"{cal['match_threshold']:>8.3f} "
              f"{cal['spawn_threshold']:>8.3f} "
              f"{gap:>8.3f}")

    out = {
        "mvp_calibration": mvp_cal,
        "mvp_retrieval_check": mvp_retrieval,
        "full_calibration": full_cal,
        "binary_calibration": bin_cal,
        "hand_tuned_defaults": {"match": 5.0, "spawn": 7.0},
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
