"""v3 Sprint 9c — Diagnose why subset_459 (and others) beat spawn
under coverage compose.

Sprint 9b found one outlier: subset_459 where coverage (0.959) never
lost to spawn even at N=5000 (0.931). We hypothesized this is because
the candidate pool happened to contain a 3-expert combination that
PARTITIONS the target's class-1 digit set — each picked expert
contributes a distinct slice, their union covers the target, and
majority-vote recovers the full task function.

This sprint tests the hypothesis by computing per-task coverage
metrics on the Sprint 9a/9b stored picks:

  union_completeness : |T ∩ (S1 ∪ S2 ∪ S3)| / |T|
                       (how much of the target's class-1 set is
                       claimed by at least one picked expert)
  over_coverage      : |(S1 ∪ S2 ∪ S3) \\ T| / |(S1 ∪ S2 ∪ S3)|
                       (how much of the picks' union is OUTSIDE
                       the target)
  jaccard_to_target  : mean Jaccard(Si, T) across picks
                       (how similar each pick is to the target)
  pairwise_jaccard   : mean pairwise Jaccard among picks
                       (low = diverse picks, high = redundant picks)
  min_element_cover  : for each element in T, how many picks claim it?
                       (partition score — ≥1 means every target
                       digit is covered by at least one expert)

And correlate these with the compose-vs-spawn gap at N=5000:
  advantage = acc_compose(N=5000) - acc_spawn(N=5000)

Predicts: tasks where compose genuinely wins at full data (e.g.
subset_459) have high union_completeness, high min_element_cover,
and LOW pairwise_jaccard (diverse picks that partition the target).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


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


def _strip_seed(entry_name: str) -> str:
    """ext_subset_0459_seed42 -> subset_0459; parity_seed42 -> parity."""
    if "_seed" in entry_name:
        return entry_name.rsplit("_seed", 1)[0]
    return entry_name


def _jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    xa = x - x.mean(); ya = y - y.mean()
    denom = float(np.sqrt((xa ** 2).sum() * (ya ** 2).sum()))
    return float((xa * ya).sum() / denom) if denom > 0 else 0.0


def coverage_metrics(
    target: frozenset[int], picked_labels: list[frozenset[int]],
) -> dict:
    picks_union = frozenset().union(*picked_labels) if picked_labels else frozenset()
    if target:
        union_completeness = len(target & picks_union) / len(target)
    else:
        union_completeness = 1.0
    if picks_union:
        over_coverage = len(picks_union - target) / len(picks_union)
    else:
        over_coverage = 0.0
    jaccard_to_target = np.mean([_jaccard(s, target) for s in picked_labels])
    pairwise = [_jaccard(a, b) for a, b in
                itertools.combinations(picked_labels, 2)]
    pairwise_jaccard = float(np.mean(pairwise)) if pairwise else 0.0
    element_claims = {e: sum(1 for s in picked_labels if e in s)
                      for e in target}
    min_element_cover = (min(element_claims.values())
                          if element_claims else 0)
    max_element_cover = (max(element_claims.values())
                          if element_claims else 0)
    return {
        "union_completeness": float(union_completeness),
        "over_coverage": float(over_coverage),
        "jaccard_to_target": float(jaccard_to_target),
        "pairwise_jaccard": pairwise_jaccard,
        "min_element_cover": int(min_element_cover),
        "max_element_cover": int(max_element_cover),
        "picks_union": sorted(picks_union),
        "element_claims": {str(k): v for k, v in element_claims.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprint9a", type=str,
                        default="results/coverage_compose.json")
    parser.add_argument("--sprint9b", type=str,
                        default="results/compose_vs_spawn_budget.json")
    parser.add_argument("--results-out", type=str,
                        default="results/coverage_mechanism.json")
    parser.add_argument("--focus-budget", type=int, default=5000,
                        help="Budget at which to compute compose/spawn gap")
    args = parser.parse_args(argv)

    with open(args.sprint9a) as f:
        sprint9a = json.load(f)
    with open(args.sprint9b) as f:
        sprint9b = json.load(f)

    tasks = all_binary_tasks()

    # Build per-task picks at Sprint 9a's n_selection=300 level
    sprint9a_picks: dict[str, list[str]] = {}
    for r in sprint9a["rows"]:
        sprint9a_picks[r["held_task"]] = r["coverage_k3_names"]

    # Build per-(task, budget) picks from Sprint 9b
    sprint9b_picks: dict[tuple[str, int], list[str]] = {}
    for r in sprint9b["rows"]:
        sprint9b_picks[(r["task"], r["budget_N"])] = r["compose_selected"]

    # Build per-task compose/spawn accuracies at focus_budget
    acc_at_focus: dict[str, dict] = {}
    for r in sprint9b["rows"]:
        if r["budget_N"] == args.focus_budget:
            acc_at_focus[r["task"]] = {
                "compose": r["acc_compose"],
                "spawn": r["acc_spawn"],
                "top1": r["acc_top1"],
            }

    print(f"Analyzing {len(sprint9a_picks)} tasks at budget "
          f"N={args.focus_budget}.\n")

    # Per-task analysis
    analysis: list[dict] = []
    for task_name in sorted(tasks.keys()):
        if task_name not in sprint9a_picks:
            continue
        target = tasks[task_name]

        # Picks from Sprint 9a (300-example selection)
        picks_9a_tasks = [_strip_seed(n) for n in sprint9a_picks[task_name]]
        picks_9a_labels = [tasks[t] for t in picks_9a_tasks if t in tasks]
        metrics_9a = coverage_metrics(target, picks_9a_labels)

        # Picks from Sprint 9b at focus_budget
        picks_key = (task_name, args.focus_budget)
        picks_9b_tasks = [_strip_seed(n)
                           for n in sprint9b_picks.get(picks_key, [])]
        picks_9b_labels = [tasks[t] for t in picks_9b_tasks if t in tasks]
        metrics_9b = coverage_metrics(target, picks_9b_labels)

        acc = acc_at_focus.get(task_name, {})
        advantage = (acc.get("compose", 0.0)
                     - acc.get("spawn", 0.0))

        analysis.append({
            "task": task_name,
            "target_labels": sorted(target),
            "target_size": len(target),
            "picks_9a_tasks": picks_9a_tasks,
            "picks_9a_labels": [sorted(l) for l in picks_9a_labels],
            "metrics_9a": metrics_9a,
            "picks_9b_tasks": picks_9b_tasks,
            "picks_9b_labels": [sorted(l) for l in picks_9b_labels],
            "metrics_9b": metrics_9b,
            "acc_compose": acc.get("compose"),
            "acc_spawn": acc.get("spawn"),
            "acc_top1": acc.get("top1"),
            "advantage_compose_minus_spawn": advantage,
        })

    # ── Focused look at the "never crosses" outliers ──
    print("=" * 72)
    print(f"  PER-TASK COVERAGE METRICS at N={args.focus_budget}")
    print("=" * 72)
    print(f"  {'task':<18s}  {'target':<18s}  {'union_compl':>11s}  "
          f"{'over_cov':>9s}  {'pair_jac':>9s}  {'min_elem':>8s}  "
          f"{'Δ(comp−spawn)':>13s}")
    analysis_sorted = sorted(analysis,
                              key=lambda a: -(a["advantage_compose_minus_spawn"]
                                              or 0))
    for a in analysis_sorted:
        m = a["metrics_9b"]
        target_str = str(a["target_labels"])[:18]
        adv = a["advantage_compose_minus_spawn"]
        adv_str = f"{adv:+.3f}" if adv is not None else "  —  "
        print(f"  {a['task']:<18s}  {target_str:<18s}  "
              f"{m['union_completeness']:>11.3f}  "
              f"{m['over_coverage']:>9.3f}  "
              f"{m['pairwise_jaccard']:>9.3f}  "
              f"{m['min_element_cover']:>8d}  "
              f"{adv_str:>13s}")

    # ── Deep-dive on subset_459 ──
    print()
    print("=" * 72)
    print("  DEEP DIVE — subset_459 ({4, 5, 9})")
    print("=" * 72)
    sr = next((a for a in analysis if a["task"] == "subset_459"), None)
    if sr:
        print(f"  Target: y ∈ {set(sr['target_labels'])}")
        print()
        print(f"  Picks at N={args.focus_budget}:")
        for pt, pl in zip(sr["picks_9b_tasks"], sr["picks_9b_labels"]):
            overlap = set(sr["target_labels"]) & set(pl)
            extra = set(pl) - set(sr["target_labels"])
            print(f"    {pt:<24s}  labels={pl}  "
                  f"∩target={sorted(overlap)}  \\target={sorted(extra)}")
        m = sr["metrics_9b"]
        print()
        print(f"  Union of picks: {m['picks_union']}")
        print(f"  Target elements' cover-count: {m['element_claims']}")
        print(f"  union_completeness = {m['union_completeness']:.3f}")
        print(f"    (fraction of {{4, 5, 9}} covered by ≥1 pick)")
        print(f"  over_coverage    = {m['over_coverage']:.3f}")
        print(f"    (fraction of pick-union outside target)")
        print(f"  pairwise_jaccard = {m['pairwise_jaccard']:.3f}")
        print(f"    (low = picks are diverse among each other)")
        print(f"  min_element_cover= {m['min_element_cover']}")
        print(f"    (weakest-covered element in target)")

    # ── Correlation analysis across tasks ──
    advantages = np.array([a["advantage_compose_minus_spawn"] or 0
                            for a in analysis])
    union_c = np.array([a["metrics_9b"]["union_completeness"]
                         for a in analysis])
    over_c = np.array([a["metrics_9b"]["over_coverage"]
                        for a in analysis])
    pair_j = np.array([a["metrics_9b"]["pairwise_jaccard"]
                        for a in analysis])
    min_e = np.array([a["metrics_9b"]["min_element_cover"]
                       for a in analysis])
    jac_to_t = np.array([a["metrics_9b"]["jaccard_to_target"]
                          for a in analysis])

    print()
    print("=" * 72)
    print(f"  CORRELATION WITH compose−spawn ADVANTAGE at N={args.focus_budget}")
    print("=" * 72)
    rows: list[tuple[str, float, float]] = [
        ("union_completeness", _pearson_r(union_c, advantages),
         _spearman_rho(union_c, advantages)),
        ("over_coverage",      _pearson_r(over_c, advantages),
         _spearman_rho(over_c, advantages)),
        ("pairwise_jaccard",   _pearson_r(pair_j, advantages),
         _spearman_rho(pair_j, advantages)),
        ("min_element_cover",  _pearson_r(min_e.astype(float), advantages),
         _spearman_rho(min_e.astype(float), advantages)),
        ("jaccard_to_target",  _pearson_r(jac_to_t, advantages),
         _spearman_rho(jac_to_t, advantages)),
    ]
    print(f"  {'metric':<22s}  {'Pearson r':>11s}  {'Spearman ρ':>11s}")
    for name, r, rho in rows:
        print(f"  {name:<22s}  {r:>+11.3f}  {rho:>+11.3f}")

    # ── Verdict ──
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    # Hypothesis: higher union_completeness → larger advantage.
    uc_rho = _spearman_rho(union_c, advantages)
    pj_rho = _spearman_rho(pair_j, advantages)
    me_rho = _spearman_rho(min_e.astype(float), advantages)

    # Rank the signal
    if abs(uc_rho) > 0.3:
        signal_union = (f"Union completeness has rho={uc_rho:+.3f} with "
                        "advantage — "
                        f"{'hypothesis supported' if uc_rho > 0 else 'opposite sign!'}")
    else:
        signal_union = (f"Union completeness has weak correlation "
                        f"(rho={uc_rho:+.3f}).")
    if abs(pj_rho) > 0.3:
        signal_pair = (f"Pairwise Jaccard has rho={pj_rho:+.3f} — "
                        f"{'picks are diverse when compose wins' if pj_rho < 0 else 'picks are REDUNDANT when compose wins (surprising)'}.")
    else:
        signal_pair = (f"Pairwise Jaccard has weak correlation "
                        f"(rho={pj_rho:+.3f}).")
    if abs(me_rho) > 0.3:
        signal_me = (f"Min-element-cover has rho={me_rho:+.3f} — "
                     f"{'every target element claimed by ≥1 pick when compose wins (partition hypothesis)' if me_rho > 0 else 'opposite sign'}.")
    else:
        signal_me = (f"Min-element-cover has weak correlation "
                     f"(rho={me_rho:+.3f}).")
    print(f"  {signal_union}")
    print(f"  {signal_pair}")
    print(f"  {signal_me}")

    out = {
        "focus_budget": args.focus_budget,
        "per_task_analysis": analysis,
        "correlations": {
            name: {"pearson_r": r, "spearman_rho": rho}
            for name, r, rho in rows
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
