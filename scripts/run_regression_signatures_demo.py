"""v3 Sprint 17 — Direction H demo: regression-probe signatures.

Sprint 17's Direction B negative said penultimate-activation
signatures collapse the within-cross task distance gap for
classification (Review 34). This doesn't generalize naturally to
non-classification regimes because there's no softmax to probe.

Direction H tests the simplest alternative: for a regression task,
signature = L2-normalized prediction vector on a fixed probe input.

Setup:
  - 5 "tasks" × 3 seeds each = 15 sine regressors, varying in
    frequency (0.5, 1.0, 2.0, 3.0, 5.0). Same task = same frequency,
    different seeds = different random init.
  - Probe: 100 evenly-spaced x ∈ [-π, π]
  - Signature: model(probe), L2-normalized → (100,) vector
  - Metric: within-task vs cross-task L2 distance in signature space

Goal: confirm within-task mean < cross-task mean (the positive case
of Direction B on regression). If yes, regression-probe signatures
are a viable non-classification signature design.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.integration.regression_signatures import (  # noqa: E402
    make_probe_inputs, make_sine_task,
    regression_probe_signature, train_sine_regressor,
)


SINE_TASKS = [
    # (task_name, frequency, phase)
    ("sine_0.5", 0.5, 0.0),
    ("sine_1.0", 1.0, 0.0),
    ("sine_2.0", 2.0, 0.0),
    ("sine_3.0", 3.0, 0.0),
    ("sine_5.0", 5.0, 0.0),
]
SEEDS = [42, 101, 2024]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Simple rank-correlation coefficient."""
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--normalize", choices=("l2", "zscore", "none"),
                        default="l2")
    parser.add_argument("--results-out",
                        default="results/regression_signatures_demo.json")
    args = parser.parse_args(argv)

    print("=" * 72)
    print(f"  REGRESSION SIGNATURES DEMO — {len(SINE_TASKS)} tasks × "
          f"{len(SEEDS)} seeds, normalize={args.normalize}")
    print("=" * 72)

    probe = make_probe_inputs(n=args.n_probe)

    # ── Train N models ──
    print(f"\n[1/3] training {len(SINE_TASKS) * len(SEEDS)} sine regressors...")
    t0 = time.time()
    models_by_task: dict[str, list] = {n: [] for n, _, _ in SINE_TASKS}
    signatures_by_task: dict[str, list[np.ndarray]] = {n: [] for n, _, _ in SINE_TASKS}
    for task_name, freq, phase in SINE_TASKS:
        for seed in SEEDS:
            model = train_sine_regressor(
                freq, phase, n_steps=args.n_steps, seed=seed,
            )
            sig = regression_probe_signature(
                model, probe, normalize=args.normalize,
            )
            models_by_task[task_name].append(model)
            signatures_by_task[task_name].append(sig)
    print(f"  trained in {time.time()-t0:.1f}s")
    print(f"  signature dim: {sig.size}")
    print(f"  signature norm (first): {np.linalg.norm(sig):.4f}")

    # ── Compute within- vs cross-task distances ──
    print("\n[2/3] computing pairwise distances...")
    within: list[float] = []
    cross: list[float] = []
    # Also: freq-distance (|freq_i - freq_j|) vs signature distance for
    # Spearman correlation — tests whether "closer frequency → closer
    # signature" holds.
    task_freqs = {name: freq for name, freq, _ in SINE_TASKS}
    freq_d: list[float] = []
    sig_d: list[float] = []

    tasks = list(signatures_by_task.keys())
    for i, t_i in enumerate(tasks):
        sigs_i = signatures_by_task[t_i]
        # Within-task
        for a in range(len(sigs_i)):
            for b in range(a + 1, len(sigs_i)):
                within.append(float(np.linalg.norm(sigs_i[a] - sigs_i[b])))
        # Cross-task
        for j in range(i + 1, len(tasks)):
            t_j = tasks[j]
            sigs_j = signatures_by_task[t_j]
            for a in sigs_i:
                for b in sigs_j:
                    d = float(np.linalg.norm(a - b))
                    cross.append(d)
                    freq_d.append(abs(task_freqs[t_i] - task_freqs[t_j]))
                    sig_d.append(d)

    within_arr = np.array(within)
    cross_arr = np.array(cross)

    print(f"\n  within-task n={within_arr.size}  "
          f"mean={within_arr.mean():.4f}  std={within_arr.std():.4f}  "
          f"max={within_arr.max():.4f}")
    print(f"  cross-task  n={cross_arr.size}  "
          f"mean={cross_arr.mean():.4f}  std={cross_arr.std():.4f}  "
          f"min={cross_arr.min():.4f}")
    gap = float(cross_arr.mean() - within_arr.mean())
    print(f"  gap (cross_mean − within_mean): {gap:+.4f}")

    # Spearman rho between |freq_i - freq_j| and signature distance
    rho = _spearman(np.array(freq_d), np.array(sig_d))
    print(f"  Spearman ρ(|Δfreq|, signature distance): {rho:+.3f}")
    print(f"    (Sprint 7b baseline for classification: ρ ≈ −0.85)")

    # ── Verdict ──
    verdict_pass = (
        gap > 0
        and within_arr.max() < cross_arr.min()  # clean separation
        and rho > 0.5  # closer-freq → closer-sig
    )
    clean_sep = bool(within_arr.max() < cross_arr.min())

    print("\n[3/3] VERDICT")
    print(f"  gap > 0:                          {'✓' if gap > 0 else '✗'}")
    print(f"  clean separation (within.max <= cross.min): "
          f"{'✓' if clean_sep else '✗'}")
    print(f"  ρ > 0.5 (freq-distance correlates): "
          f"{'✓' if rho > 0.5 else '✗'}")
    print(f"  → regression-probe signatures: "
          f"{'VIABLE' if verdict_pass else 'NOT VIABLE'}")

    out = {
        "config": vars(args),
        "n_tasks": len(SINE_TASKS),
        "n_seeds_per_task": len(SEEDS),
        "within_task": {
            "n": int(within_arr.size),
            "mean": float(within_arr.mean()),
            "std": float(within_arr.std()),
            "min": float(within_arr.min()),
            "max": float(within_arr.max()),
        },
        "cross_task": {
            "n": int(cross_arr.size),
            "mean": float(cross_arr.mean()),
            "std": float(cross_arr.std()),
            "min": float(cross_arr.min()),
            "max": float(cross_arr.max()),
        },
        "gap": gap,
        "clean_separation": clean_sep,
        "spearman_freq_vs_sig_distance": rho,
        "verdict_pass": verdict_pass,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
