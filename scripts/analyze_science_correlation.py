"""Analyze the science-correlation sweep: dim vs test_accuracy.

Loads results/science_correlation_raw.json, computes Spearman rank
correlations between each trajectory metric and both test_accuracy and
test_loss, prints a table, and saves a summary + optional scatter plot
of (correlation_dim, test_accuracy).

Usage:
    python scripts/analyze_science_correlation.py \\
        [--raw results/science_correlation_raw.json] \\
        [--plot results/science_correlation_scatter.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def spearman(a: list[float], b: list[float]) -> tuple[float, float]:
    """Spearman rank correlation + two-sided p-value approximation.

    Returns (rho, p_value). p is approximated via the t-distribution with
    n-2 df; for small N this is conservative.
    """
    if len(a) != len(b) or len(a) < 4:
        return float("nan"), float("nan")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    a = a[mask]
    b = b[mask]
    n = len(a)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra_centered = ra - ra.mean()
    rb_centered = rb - rb.mean()
    denom = np.sqrt(np.sum(ra_centered ** 2) * np.sum(rb_centered ** 2))
    if denom < 1e-12:
        return 0.0, float("nan")
    rho = float(np.sum(ra_centered * rb_centered) / denom)
    if abs(rho) >= 1.0 or n <= 2:
        return rho, float("nan")
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    # Approximate two-sided p via normal approximation of t distribution.
    from math import erf, sqrt
    z = abs(float(t))
    p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    return rho, float(p)


def _ok(rec: dict) -> bool:
    if rec.get("error"):
        return False
    if rec.get("correlation_dim") is None:
        return False
    if rec.get("test_accuracy") is None:
        return False
    return True


def _print_table(rows: list[tuple[str, float, float, int]]) -> None:
    print(f"{'metric':<30s}  {'rho':>8s}  {'p':>10s}  {'n':>5s}")
    print("-" * 60)
    for name, rho, p, n in rows:
        rho_s = f"{rho:+.4f}" if np.isfinite(rho) else "  n/a  "
        p_s = f"{p:.4g}" if np.isfinite(p) else "  n/a"
        print(f"{name:<30s}  {rho_s:>8s}  {p_s:>10s}  {n:>5d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str,
                        default="results/science_correlation_raw.json")
    parser.add_argument("--out", type=str,
                        default="results/science_correlation_analysis.json")
    parser.add_argument("--plot", type=str,
                        default="results/science_correlation_scatter.png",
                        help="scatter plot output path; empty string disables")
    parser.add_argument("--target-dim", type=float, default=1.5)
    parser.add_argument("--tolerance", type=float, default=0.3)
    args = parser.parse_args(argv)

    raw = json.load(open(args.raw))
    records = raw["records"]
    n_total = len(records)
    good = [r for r in records if _ok(r)]
    n_good = len(good)
    print(f"[analyze] {n_good}/{n_total} runs usable "
          f"(rest failed or had undefined correlation_dim)")
    if n_good < 4:
        print("[analyze] not enough data — need at least 4 usable runs")
        return 2

    dims = [r["correlation_dim"] for r in good]
    test_acc = [r["test_accuracy"] for r in good]
    test_loss = [r["test_loss"] for r in good]
    path_length = [r["trajectory_metrics"]["total_path_length"] for r in good]
    tortuosity = [r["trajectory_metrics"]["tortuosity"] for r in good]
    recurrence = [r["trajectory_metrics"]["recurrence_rate"] for r in good]
    displacement = [r["trajectory_metrics"]["displacement"] for r in good]

    # Primary hypothesis: correlation_dim ↔ test_accuracy
    primary_rho, primary_p = spearman(dims, test_acc)
    # Secondary: dim ↔ test_loss (should be anti-correlated with above)
    loss_rho, loss_p = spearman(dims, test_loss)

    # Other metrics for cross-reference
    rows = [
        ("correlation_dim vs test_acc", primary_rho, primary_p, n_good),
        ("correlation_dim vs test_loss", loss_rho, loss_p, n_good),
        ("path_length vs test_acc",
            *spearman(path_length, test_acc), n_good),
        ("tortuosity vs test_acc",
            *spearman(tortuosity, test_acc), n_good),
        ("recurrence_rate vs test_acc",
            *spearman(recurrence, test_acc), n_good),
        ("displacement vs test_acc",
            *spearman(displacement, test_acc), n_good),
    ]

    print()
    _print_table(rows)

    # Check whether target band contains best-accuracy runs
    band_lo = args.target_dim - args.tolerance
    band_hi = args.target_dim + args.tolerance
    in_band = [(d, a) for d, a in zip(dims, test_acc) if band_lo <= d <= band_hi]
    out_of_band = [(d, a) for d, a in zip(dims, test_acc)
                   if not (band_lo <= d <= band_hi)]

    band_mean_acc = float(np.mean([a for _, a in in_band])) if in_band else float("nan")
    out_mean_acc = float(np.mean([a for _, a in out_of_band])) if out_of_band else float("nan")

    print()
    print(f"[analyze] target band: dim ∈ [{band_lo:.2f}, {band_hi:.2f}] "
          f"(target_dim={args.target_dim}, tolerance={args.tolerance})")
    print(f"  runs in band:  {len(in_band):>3d}   mean test_acc = {band_mean_acc:.4f}")
    print(f"  runs out:      {len(out_of_band):>3d}   mean test_acc = {out_mean_acc:.4f}")
    band_diff = band_mean_acc - out_mean_acc if not np.isnan(band_mean_acc) and not np.isnan(out_mean_acc) else float("nan")
    print(f"  band ∆ acc:    {band_diff:+.4f}")

    # Interpretation summary (for the review doc)
    verdict = _interpret(primary_rho, primary_p, band_diff, n_good)
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    for line in verdict:
        print(f"  {line}")

    # Scatter plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))

            # Color by optimizer
            opt_colors = {"sgd": "tab:blue", "adam": "tab:orange",
                          "adamw": "tab:green"}
            for opt in sorted({r["hparams"].get("optimizer", "?") for r in good}):
                xs = [r["correlation_dim"] for r in good
                      if r["hparams"].get("optimizer") == opt]
                ys = [r["test_accuracy"] for r in good
                      if r["hparams"].get("optimizer") == opt]
                ax.scatter(xs, ys, label=opt,
                           color=opt_colors.get(opt, "gray"),
                           alpha=0.7, s=45, edgecolors="black", linewidths=0.5)

            ax.axvspan(band_lo, band_hi, color="green", alpha=0.10,
                       label=f"target band [{band_lo:.1f}, {band_hi:.1f}]")
            ax.axvline(args.target_dim, linestyle=":", color="green",
                       alpha=0.5)

            ax.set_xlabel("Correlation dimension (of 16-d projected trajectory)")
            ax.set_ylabel("Test accuracy on MNIST held-out")
            title = (f"Does fractal dim correlate with generalization?\n"
                     f"Spearman ρ = {primary_rho:+.3f}  "
                     f"(p ≈ {primary_p:.3g},  n = {n_good})")
            ax.set_title(title)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()

            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"\n[analyze] scatter plot saved: {plot_path}")
        except Exception as e:
            print(f"[analyze] plot generation failed: {type(e).__name__}: {e}")

    summary = {
        "n_total": n_total,
        "n_good": n_good,
        "target_band": [band_lo, band_hi],
        "target_dim": args.target_dim,
        "tolerance": args.tolerance,
        "primary": {
            "correlation_dim_vs_test_accuracy": {
                "spearman_rho": primary_rho, "p_value": primary_p,
            },
            "correlation_dim_vs_test_loss": {
                "spearman_rho": loss_rho, "p_value": loss_p,
            },
        },
        "secondary": {
            "path_length_vs_test_acc": dict(zip(
                ("spearman_rho", "p_value"),
                spearman(path_length, test_acc))),
            "tortuosity_vs_test_acc": dict(zip(
                ("spearman_rho", "p_value"),
                spearman(tortuosity, test_acc))),
            "recurrence_rate_vs_test_acc": dict(zip(
                ("spearman_rho", "p_value"),
                spearman(recurrence, test_acc))),
            "displacement_vs_test_acc": dict(zip(
                ("spearman_rho", "p_value"),
                spearman(displacement, test_acc))),
        },
        "band_analysis": {
            "n_in_band": len(in_band),
            "n_out_of_band": len(out_of_band),
            "mean_test_acc_in_band": band_mean_acc,
            "mean_test_acc_out_of_band": out_mean_acc,
            "band_minus_out": band_diff,
        },
        "verdict": verdict,
        "data_summary": {
            "dim_min": float(min(dims)),
            "dim_max": float(max(dims)),
            "dim_mean": float(np.mean(dims)),
            "test_acc_min": float(min(test_acc)),
            "test_acc_max": float(max(test_acc)),
            "test_acc_mean": float(np.mean(test_acc)),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[analyze] analysis summary saved: {out_path}")

    return 0


def _interpret(rho: float, p: float, band_diff: float, n: int) -> list[str]:
    lines: list[str] = []
    if not np.isfinite(rho):
        lines.append("Primary Spearman undefined — insufficient usable data.")
        return lines
    if not np.isfinite(p) or p > 0.1:
        lines.append(f"NULL RESULT: rho = {rho:+.3f}, p = {p:.3g}, n = {n}.")
        lines.append("No statistically significant correlation between")
        lines.append("correlation dimension and test accuracy.")
        lines.append("The target-dim-1.5 choice remains heuristic.")
    elif p <= 0.05:
        if abs(rho) < 0.3:
            lines.append(f"WEAK significant correlation (rho = {rho:+.3f}, "
                         f"p = {p:.3g}, n = {n}).")
            lines.append("Statistically present but effect size small.")
        else:
            sign = "positive" if rho > 0 else "negative"
            lines.append(f"REAL {sign} correlation (rho = {rho:+.3f}, "
                         f"p = {p:.3g}, n = {n}).")
            lines.append("Fractal dim does track generalization in this grid.")
    else:  # 0.05 < p <= 0.1
        lines.append(f"MARGINAL trend (rho = {rho:+.3f}, p = {p:.3g}, n = {n}).")
        lines.append("Worth a larger replication before making claims.")

    if np.isfinite(band_diff):
        if band_diff > 0.02:
            lines.append(
                f"Target band has +{band_diff:.3f} higher mean test_acc than "
                f"out-of-band runs.")
        elif band_diff < -0.02:
            lines.append(
                f"Target band has {band_diff:+.3f} LOWER mean test_acc than "
                f"out-of-band runs. The target range may be miscalibrated.")
        else:
            lines.append(
                f"Target band and out-of-band have comparable test_acc "
                f"(Δ = {band_diff:+.3f}).")

    return lines


if __name__ == "__main__":
    raise SystemExit(main())
