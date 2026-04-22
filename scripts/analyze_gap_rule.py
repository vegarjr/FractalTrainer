"""Sprint 17 follow-up — quantitative gap rule analysis.

Reads all results/fractal_demo_*.json files, extracts per-budget
(A, B, C) accuracies from each ablation, and fits

    (B − A)  ≈  α × (ceiling_A − A)

where ceiling_A is the maximum accuracy that arm A reaches across
the budget sweep for that same ablation (a proxy for "how much
headroom was available to any method training on this data").

If α is tight across all ablations, "context injection recovers
~α fraction of the available gap" becomes a predictive claim.

Run:
    python scripts/analyze_gap_rule.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def _dataset_from_filename(name: str) -> str:
    """Infer dataset for older JSONs that pre-dated --dataset."""
    n = name.lower()
    if "cifar" in n:
        return "cifar10"
    if "fashion" in n:
        return "fashion"
    # Default: all Sprint 17 MNIST runs
    return "mnist"


def _extract_datapoints(json_path: Path) -> list[dict]:
    """Return a list of (ablation_name, budget, A, B, C) rows.

    Ceiling is filled in by the caller after seeing all rows —
    the true full-data ceiling for a (dataset, query) pair is the
    max A across every ablation, not within a single ablation.
    """
    with open(json_path) as f:
        payload = json.load(f)

    rows: list[dict] = []
    ablation_by_q = payload.get("ablation") or {}
    config = payload.get("config", {}) or {}
    dataset = config.get("dataset") or _dataset_from_filename(json_path.name)
    train_size = config.get("train_size", 5000)
    seed_train_size = config.get("seed_train_size") or train_size

    # Heuristic regime tag
    is_data_starved = (train_size < 1000) if train_size is not None else False
    regime = "data_starved" if is_data_starved else "full_data"

    for q_name, arms in ablation_by_q.items():
        A = arms.get("A_no_context")
        B = arms.get("B_nearest_context")
        C = arms.get("C_random_context")
        if not (A and B and C):
            continue

        budgets = sorted(int(b) for b in A["mean_by_budget"].keys())
        for b in budgets:
            a = float(A["mean_by_budget"][str(b)])
            bb = float(B["mean_by_budget"][str(b)])
            cc = float(C["mean_by_budget"][str(b)])
            a_std = float(A["stdev_by_budget"][str(b)])
            b_std = float(B["stdev_by_budget"][str(b)])

            rows.append({
                "source": json_path.name,
                "regime": regime,
                "dataset": dataset,
                "query": q_name,
                "train_size": train_size,
                "budget": b,
                "A": a,
                "B": bb,
                "C": cc,
                "A_stdev": a_std,
                "B_stdev": b_std,
                "delta_B": bb - a,
                "delta_C": cc - a,
            })
    return rows


TASK_CEILINGS = {
    # Full-data ceilings derived from Sprints 17-37 full-data runs.
    # MNIST binary subset_019 reaches ~0.96 @ full data; evens-related
    # subsets are similarly learnable on MNIST.
    ("mnist",   "Q_match"):   0.97,
    ("mnist",   "Q_compose"): 0.96,
    ("mnist",   "Q_spawn"):   0.96,
    # Fashion subset_019 = {T-shirt, Trouser, Ankle boot} tops out ~0.95;
    # evens-related partitions cross visually-mixed clothing → lower.
    ("fashion", "Q_match"):   0.95,
    ("fashion", "Q_compose"): 0.80,
    ("fashion", "Q_spawn"):   0.95,
    # CIFAR: all tasks are harder — 10-way visual categories flatten
    # binary accuracy. Values from the CIFAR full-data Q_match/compose/
    # spawn outputs (~0.77 / 0.80 / 0.83).
    ("cifar10", "Q_match"):   0.78,
    ("cifar10", "Q_compose"): 0.80,
    ("cifar10", "Q_spawn"):   0.85,
}


def _attach_ceilings(rows: list[dict]) -> None:
    """Fill in ceiling_A and gap using TASK_CEILINGS (per-task full-data
    ceilings derived from observed full-data runs), so data-starved
    ablations see the true gap, not just the small-sample max.
    """
    for r in rows:
        key = (r["dataset"], r["query"])
        ceiling = TASK_CEILINGS.get(key)
        if ceiling is None:
            # Fallback: max A observed for this task across all rows
            ceiling = max((rr["A"] for rr in rows
                            if (rr["dataset"], rr["query"]) == key),
                           default=r["A"])
        r["ceiling_A"] = ceiling
        r["gap"] = max(0.0, ceiling - r["A"])


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (alpha, beta, r_squared) for y = alpha*x + beta."""
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    (alpha, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = alpha * x + beta
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(alpha), float(beta), r2


def _print_table(rows: list[dict]) -> None:
    header = (f"{'source':<42} {'query':<11} {'budget':>6} "
             f"{'A':>6} {'B':>6} {'C':>6} {'gap':>6} {'B−A':>7} {'C−A':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['source']:<42} {r['query']:<11} {r['budget']:>6} "
            f"{r['A']:>6.3f} {r['B']:>6.3f} {r['C']:>6.3f} "
            f"{r['gap']:>6.3f} {r['delta_B']:>+7.3f} {r['delta_C']:>+7.3f}"
        )


def _plot(rows: list[dict], alpha_b: float, beta_b: float, r2_b: float,
           alpha_c: float, beta_c: float, r2_c: float, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plot")
        return

    gaps = np.array([r["gap"] for r in rows])
    delta_b = np.array([r["delta_B"] for r in rows])
    delta_c = np.array([r["delta_C"] for r in rows])
    datasets = [r["dataset"] for r in rows]
    colors = {"mnist": "tab:blue", "fashion": "tab:orange",
              "cifar10": "tab:green", "unknown": "gray"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, deltas, arm, alpha, beta, r2 in [
        (axes[0], delta_b, "B (nearest)", alpha_b, beta_b, r2_b),
        (axes[1], delta_c, "C (random)",  alpha_c, beta_c, r2_c),
    ]:
        for ds in set(datasets):
            mask = np.array([d == ds for d in datasets])
            ax.scatter(gaps[mask], deltas[mask], label=ds,
                        color=colors.get(ds, "gray"), alpha=0.7)
        xs = np.linspace(0, max(0.02, float(gaps.max())), 100)
        ax.plot(xs, alpha * xs + beta, "k--",
                 label=f"fit: α={alpha:.3f}, β={beta:+.3f}, R²={r2:.3f}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("gap = ceiling_A − A_achieved")
        ax.set_ylabel("arm − A")
        ax.set_title(f"arm {arm}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Gap rule — context injection recovers α × (ceiling − A)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nplot saved: {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-glob", default="results/fractal_demo_*.json")
    parser.add_argument("--output-json", default="results/gap_rule_analysis.json")
    parser.add_argument("--output-plot", default="results/gap_rule_analysis.png")
    parser.add_argument("--exclude", nargs="*",
                        default=["fractal_demo_smoke_test.json"],
                        help="filenames (basename) to skip")
    args = parser.parse_args(argv)

    paths = sorted(REPO_ROOT.glob(args.results_glob))
    paths = [p for p in paths if p.name not in set(args.exclude)]
    print(f"Analyzing {len(paths)} result files:")
    for p in paths:
        print(f"  {p.name}")

    all_rows: list[dict] = []
    for p in paths:
        rows = _extract_datapoints(p)
        all_rows.extend(rows)
    _attach_ceilings(all_rows)
    print(f"\nExtracted {len(all_rows)} (ablation × budget) datapoints\n")

    _print_table(all_rows)

    # Fit linear regression for arm B
    gaps = np.array([r["gap"] for r in all_rows])
    delta_b = np.array([r["delta_B"] for r in all_rows])
    delta_c = np.array([r["delta_C"] for r in all_rows])

    alpha_b, beta_b, r2_b = _linear_fit(gaps, delta_b)
    alpha_c, beta_c, r2_c = _linear_fit(gaps, delta_c)

    print("\n" + "=" * 72)
    print("  GAP RULE FIT: arm − A = α × (ceiling_A − A_achieved) + β")
    print("=" * 72)
    print(f"  Arm B (nearest context): α={alpha_b:+.4f}  β={beta_b:+.4f}  R²={r2_b:.4f}")
    print(f"  Arm C (random context):  α={alpha_c:+.4f}  β={beta_c:+.4f}  R²={r2_c:.4f}")

    # Per-dataset fit
    print("\n  Per-dataset fit (arm B):")
    for ds in sorted(set(r["dataset"] for r in all_rows)):
        sub = [r for r in all_rows if r["dataset"] == ds]
        xs = np.array([r["gap"] for r in sub])
        ys = np.array([r["delta_B"] for r in sub])
        a, b, r2 = _linear_fit(xs, ys)
        print(f"    {ds:<10s}  α={a:+.4f}  β={b:+.4f}  R²={r2:.4f}  "
              f"(n={len(sub)})")

    # Acceptance: R² ≥ 0.5 means the gap rule is meaningfully predictive
    verdict = "predictive" if r2_b >= 0.5 else "weak/task-specific"
    print(f"\n  VERDICT: gap rule is {verdict} (R²={r2_b:.3f})")

    # Save
    out = {
        "n_files": len(paths),
        "n_datapoints": len(all_rows),
        "fit_arm_B": {"alpha": alpha_b, "beta": beta_b, "r_squared": r2_b},
        "fit_arm_C": {"alpha": alpha_c, "beta": beta_c, "r_squared": r2_c},
        "per_dataset_arm_B": {
            ds: {"alpha": (f := _linear_fit(
                    np.array([r["gap"] for r in all_rows if r["dataset"] == ds]),
                    np.array([r["delta_B"] for r in all_rows if r["dataset"] == ds]),
                 ))[0], "beta": f[1], "r_squared": f[2],
                 "n": sum(1 for r in all_rows if r["dataset"] == ds)}
            for ds in sorted(set(r["dataset"] for r in all_rows))
        },
        "rows": all_rows,
    }
    out_path = REPO_ROOT / args.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")

    _plot(all_rows, alpha_b, beta_b, r2_b, alpha_c, beta_c, r2_c,
           REPO_ROOT / args.output_plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
