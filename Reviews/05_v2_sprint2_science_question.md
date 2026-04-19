# v2 Sprint 2 — Science question (candidate D)

**Date:** 2026-04-19
**Reviewer:** Claude Opus 4.7 (1M context)

## The question

The v1 system targets correlation dimension ≈ 1.5 on a 16-d random-projected weight trajectory. This target was chosen heuristically — the README documents this openly: "Whether this dim corresponds to any meaningful property of trained models (generalization, loss plateau, mode collapse) is an open research question."

Sprint 2 asks that question with an experiment: **does correlation dimension actually correlate with test accuracy on MNIST?**

## Experiment

Grid sweep, 48 runs:
- learning_rate: [0.005, 0.01, 0.03, 0.1]
- weight_decay: [0.0, 0.01]
- optimizer: [sgd, adam]
- seed: [42, 101, 2024]

Per run: MLP 784→64→32→10, 500 SGD/Adam steps on MNIST-5000 subset, 25 snapshots, 16-d random projection, correlation dim via Grassberger-Procaccia. Then evaluate on held-out MNIST-1000. Record `(hparams, final_correlation_dim, test_accuracy, test_loss, trajectory_metrics)`.

Compute: 8.5 minutes total, 10.7 s/run average.

Code: `scripts/science_correlation_sweep.py` + `scripts/analyze_science_correlation.py`. Raw + analysis JSONs in `results/science_correlation_*.json`. Scatter plot in `results/science_correlation_scatter.png`.

## Finding

**Null correlation.** ρ = +0.122, p = 0.406, n = 48. No significant monotonic relationship between correlation dim and test accuracy.

Secondary trajectory metrics (path length, tortuosity, recurrence rate, displacement) also show no significant correlation. Tortuosity had the strongest directional signal (ρ = +0.209, p = 0.146) — suggestive but not significant.

## Worse: the v1 target band is anti-correlated

Split the runs by whether their `correlation_dim` lies in the v1 target band [1.2, 1.8]:

| Bucket | n | Mean test_acc |
|---|---|---|
| In target band [1.2, 1.8] | 4 | **0.5690** |
| Out of band | 44 | **0.8204** |
| Δ | | **−0.2514** |

**Runs whose trajectories sit in the v1 target band have substantially WORSE test accuracy than runs outside it.** Not by a little — by 25 percentage points.

## Why?

Split by optimizer:

| Optimizer | n | Mean test_acc ± σ | Mean dim ± σ |
|---|---|---|---|
| SGD | 24 | 0.869 ± 0.060 | **0.693 ± 0.131** |
| Adam | 24 | 0.730 ± 0.305 | **1.134 ± 0.960** |

The SGD runs form a **narrow** dim cluster around 0.7 with tight test accuracy around 0.87. Adam spans 0.003 to 3.66 in dim with test accuracy ranging 0.12 to 0.93 — the σ on Adam's test_acc (0.305) tells the story: **many Adam runs at high learning rates catastrophically diverged** (lr=0.1 + Adam: test_acc = 0.115–0.281).

The trajectories of diverged Adam runs are the ones that land in the target band [1.2, 1.8]. A chaotic/diverging trajectory exhibits higher correlation dimension than a steady-descent trajectory — which is exactly what SGD produces on this problem.

In short: **the v1 target pushed the repair loop TOWARD the dim signature of divergent training, not away from it.**

## Implications for v1

The v1 repair loop in `Reviews/03_sprint3_closed_loop.md` converged a seeded-divergent SGD run (lr=0.1, dim≈0.36, test acc high) to dim≈1.27 (inside target band) by switching to AdamW + lr=0.01 + weight_decay + dropout. The loop did its advertised job (shape moved into the band). But what it actually did in MNIST-test-accuracy terms was take a decent SGD run and replace it with a more-regularized Adam variant that happens to have a higher correlation dim. Whether that's *better training* is unknown and, per this experiment, not predicted by dim.

The system works at the engineering level — observer, geometry, target, repair all wire up cleanly. But **the target value (1.5) has no empirical basis on MNIST**, and actually tracks divergent training more than healthy training.

## Implications for v1's architecture (not the target)

Nothing about this finding invalidates:
- The observer (weight snapshots, trajectory construction)
- The geometry plane (correlation dim, box-counting, trajectory metrics)
- The divergence + repair pipeline (scope/schema/outcome gates)
- The self-modification observer from v2 Sprint 1

Those are all correct given *some* target. The finding is that *scalar correlation dim* is not the right target on MNIST with this training setup.

## What the data suggests instead

Looking at top-25% runs by test accuracy (n=12, test_acc 0.919–0.949):
- dim median: 0.830
- dim range: 0.343 to 3.202

The distribution is wide. You can't clearly gate "good" runs by dim alone. But you can observe: **SGD with moderate lr** (0.03, 0.1) and weight_decay≈0.01 is very reliably in test_acc 0.9+. These runs have dim in [0.47, 0.84] — a narrower sub-band.

If we wanted to pick a new scalar target, it would be roughly `dim ≈ 0.7 ± 0.2` — but this would really just be a proxy for "use SGD with lr 0.03–0.1." A dim target isn't doing any work here that the hparam schema isn't already doing.

## Recommendations

### For v1 as shipped
Document the finding in README: **target-dim=1.5 is an arbitrary numerical value and has been empirically shown not to correlate with test accuracy in this training setup**. The system's success depends on the target being well-chosen; on MNIST with correlation dim, it isn't.

Do not claim the v1 closed-loop demo produced "better training" — only that it produced a trajectory matching the target.

### For v2+
**Candidate A (golden-run matching) is now the strongest next step**, not because it's more sophisticated, but because it sidesteps the problem this experiment revealed:

- A golden-run target is *learned from an empirically good run*, not picked a priori. If you pick your golden run by test accuracy, the target shape is guaranteed to correspond to good training.
- This inverts the epistemology: v1 picks a shape and asks the LLM to match it; golden-run v2 picks a good run and asks what shape it has, then asks the LLM to match *that*.

This is a more honest version of the fractal-AI vision. The training trajectory still has geometry, and we still observe/measure it, but we no longer pretend to know in advance what "fractal-shaped" means.

### Other candidates
- **B (training-code rewrites):** Same scalar-dim-target problem. Widening the rewrite scope doesn't fix the target problem. Should not proceed before A.
- **C (meta-controller recursion):** Would recursively optimize toward a bad target. Strongly not recommended until A establishes a better target.

## Decision

**v2 Sprint 2 candidate D: GATE PASSED in the scientific sense** — the question was well-posed and decisively answered. The answer happens to be inconvenient for v1's specific target choice, but that's science working correctly, not failure.

**Next step (with user approval):** v2 Sprint 3 = candidate A (golden-run trajectory matching). Candidates B and C are parked pending A's completion.

## Files delivered

- `scripts/science_correlation_sweep.py` — 48-run grid driver
- `scripts/analyze_science_correlation.py` — Spearman + band analysis + scatter plot
- `results/science_correlation_raw.json` — 48-run raw records
- `results/science_correlation_analysis.json` — Spearman table + verdict
- `results/science_correlation_scatter.png` — scatter plot
- `results/science_trajectories/` — per-run trajectory npys + meta jsons (48 × 2 files; gitignored)

## Paired ChatGPT review

Per the dual-AI workflow, this is Claude's verdict. Given the significance of the finding (v1's target is wrong), a ChatGPT-side independent read before acting on the "switch to golden-run matching" recommendation would be especially valuable here. Ideally run the same analysis independently — the raw JSON is committed.
