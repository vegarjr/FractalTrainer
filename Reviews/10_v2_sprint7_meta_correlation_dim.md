# v2 Sprint 7 — Meta-trajectory correlation dimension (N=22)

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Purpose

Sprint 6 produced a 9-point meta-trajectory and found Claude's fixes cluster in hparam-space — but N=9 was too small for an honest correlation dimension measurement. The Grassberger-Procaccia detector returned a degenerate "flat-plateau" result (dim=0, r²=1) — correctly refusing to hallucinate a number from 36 pairwise distances.

Sprint 7 adds 12 more multi-runs spanning a wider design space, pushing the combined meta-trajectory past the detector's N=20 threshold. The question Sprint 7 answers: **does Claude's sequence of repair fixes, across 17 different starting configurations, have a genuine fractal dimension?**

## Setup

- `configs/multi_loop_starting_points_sprint7.yaml`: 12 new diverse divergent starts:
  - `divergent_sgd_lr1` (too-large SGD)
  - `frozen_adam_tiny_lr` (under-trained)
  - `plain_adamw_baseline` (reasonable-but-not-great)
  - `heavy_reg_small_lr_sgd` (over-regularized)
  - `adam_lr_0p5` (exploding Adam)
  - `sgd_lr05_seed314` (different seed)
  - `small_batch_adam` (bs=8, noisy gradients)
  - `large_batch_adam` (bs=512, smooth gradients)
  - `medium_adamw`
  - `sgd_with_dropout`
  - `divergent_adamw_lr02` (AdamW version of divergent Adam)
  - `very_conservative_adam` (barely any training)

Golden: same opaque golden as Sprint 5/6 (`golden_runs/golden_alpha.json` — signature only, no hparam leak).

## Execution

`scripts/run_multi_loop.py` with the 12 new starts, `--llm cli`, max_iters=3.

- 12/12 runs completed successfully
- 26.4 minutes wall clock total (1584.6 s)
- 27 repair-log entries

Combined with Sprint 6's 11 entries → **38 total aggregated log entries**.

After filtering for accepted patches with actual state change (the meta-trajectory contract): **22 points, 21 transitions**.

## The result — genuine fractal dimension measurement

```
Correlation dimension = 1.4908
r_squared             = 0.9890
n_points              = 22
scaling_range         = [16, 24] out of ~24 log-radii (wide window)
n_pairs_sampled       = 231  (full pairwise, no truncation)
```

This is a clean measurement. The high r² (0.989) and wide scaling range mean the log(C(r)) vs log(r) plot has a real linear region. Not a plateau-fitting artifact.

**Claude's meta-trajectory has a correlation dimension of ~1.5.**

## What 1.5 means

A correlation dimension between 1 (straight line) and 2 (filled plane) says: the path through hparam-space is locally one-dimensional with some 2-d-ish wobble. Concretely: at a given scale, new points mostly add to the existing line rather than filling out volume, but the line itself is non-straight.

This is the natural dimensionality for a sequence of LLM fixes that bounces around a preferred attractor: you keep returning to similar regions (1-d structure along the attractor spine) with modest deviations (the extra ~0.5 dims of wobble).

## Amusing coincidence

The v1 target dim was **1.5** (arbitrarily chosen). We now have an observed meta-trajectory dim of **1.49**. Completely coincidental — these are unrelated quantities — but worth noting. The v1 scalar target turned out to be empirically wrong (Sprint 2). The meta-trajectory of the repair loop trying to fix the consequences of that wrong target happens to sit at the same number. It's not a meaningful coincidence; it's just a funny one.

## Attractor structure — 3-way clustering

Partitioning the 22 meta-trajectory points by the hparam regime Claude landed on:

| Region | Criterion | Count |
|---|---|---|
| adam, low lr | `optimizer=adam, lr<0.01` | 7 / 22 |
| adamw, low lr | `optimizer=adamw, lr<0.01` | 6 / 22 |
| sgd | `optimizer=sgd` | 7 / 22 |
| other | higher-lr adam/adamw | 2 / 22 |

Three roughly equal-sized clusters (adam-like 32%, adamw-like 27%, sgd 32%), plus a small "other" residual. Sprint 6 suggested Claude prefers `lr=0.001 adam-family`; at larger N, we see the picture is more nuanced — **Claude has three attractor regions**, not one, and splits its fixes between them depending on starting configuration.

## Recurrence: 58%

```
recurrence_rate = 0.5833
```

58% of sampled trajectory points return close to an earlier-visited point in the 6-d embedding (within 0.5× the median step size). This is extremely high — by contrast, a random-walk-in-6-d typical recurrence rate is near zero. The repair loop is *not* exploring new territory on each run; it's revisiting the same clusters.

Combined with the 3-way clustering above: Claude's repair policy is effectively a finite-state attractor — it returns to a small number of canonical "healthy" hparam regions, regardless of starting point. The meta-trajectory's fractal dim reflects the geometry of moving between those attractors.

## Convergence signature across 17 runs

```
start_divergence       = 312500.0   (first run's exploding Adam)
end_divergence         =      1.10  (last run — slightly outside band)
total_reduction        = 312498.9
monotonic_decreasing   = False       (as expected — sawtooth across runs)
bounces                = 13          (one per run-start, approximately)
mean_step_reduction    = 14881       (dominated by the first giant drop)
```

13 bounces across 17 runs is the concatenation sawtooth — each run starts high (because it resets to a fresh divergent config), Claude pulls it down, next run restarts high. The per-run convergence is mostly monotonic; the aggregated sequence isn't, by construction.

## Geometry

```
total_path_length = 30.3    (total distance traveled in 6-d)
displacement      =  2.26   (end-to-end Euclidean)
tortuosity        = 13.4    (path winds 13× its straight-line)
mean_curvature    =  1.92   (sharp direction changes)
dispersion        =  1.30   (moderate spread around centroid)
```

Tortuosity 13.4 is high but expected given the multi-run structure. The real finding is that this path has r²=0.989 linearity in its GP plot despite the apparent chaos — the meta-trajectory is fractally coherent, not noisy.

## What's validated

At N=22 we can now make a first-of-its-kind claim: **across 17 independent repair runs from diverse divergent starting points, the Claude-LLM-driven sequence of hparam patches forms a point cloud in hparam-space with correlation dimension ~1.5 and strong attractor structure (58% recurrence).**

This is genuinely new data. Nobody has (to my knowledge) previously measured the fractal dimension of an LLM's sequence of code fixes. It's mildly interesting in itself, and it validates the machinery built across Sprints 1–6.

## Limits of the claim

- **Local, not universal.** This dim is for this specific setup (MNIST MLP, this specific opaque golden, Sonnet model, our specific 17 starting configs). Different training task, different golden, or different LLM would give different numbers. The methodology is general; the number isn't.
- **N=22 is still modest.** GP at N=22 is at the low end of "reliable." r²=0.989 is encouraging, but a more rigorous study would use N=100+. Sprint 8 could push further if desired.
- **The meta-trajectory is multi-run-concatenated.** The within-run dynamics (Claude's patches within a single closed-loop run) are a different, smaller, related object. Per-run analysis would give finer-grained dimensionality, potentially different from the aggregated 1.5.
- **Aggregation bias.** Some runs contributed more transitions than others (e.g., `adam_lr_0p5` hit the iter cap without converging — its contributions skew high-divergence). The aggregate shape is weighted by how hard each fix was.

## Candidate C (meta-controller recursion) is now unblocked

With N=22 meta-trajectory data and a real dim estimate, Candidate C's "a second repair loop watches the meta-trajectory and patches the target when the outer loop bounces" experiment has something concrete to regress against. Specifically:
- The meta-trajectory's dim is ~1.5; if the outer loop were bouncing (poorly converging), its meta-trajectory dim would be >1.5 (closer to random). A meta-controller could watch for that excess.
- The convergence signature's `bounces` metric (13 across 17 runs here) is a simpler alternative signal — a meta-controller could patch `target.tolerance` higher when bounces are high relative to points.

But C is a separate feature and needs its own design.

## Candidate B (training-code rewrites) — status unchanged

B is still independently available, still has the no-schema-gate risk. Sprint 7's finding doesn't particularly affect B's viability either way.

## Decision

**v2 Sprint 7 gate: PASSED.**

We have a genuine correlation-dim measurement on Claude's repair meta-trajectory, with high r² and wide scaling range. The attractor structure from Sprint 6 is confirmed at 3-way (not 1-way) and at higher recurrence (58%). The v2 campaign has produced actual scientific data about the geometric shape of an LLM's sequence of self-modifications.

The core vision — "an AI looks at the geometric shape of its own evolution and rewrites when the shape drifts from fractal" — is now validated at four nested levels:

1. The AI watches its training trajectory (v1).
2. The training trajectory has a measurable geometric shape (v1, Sprint 2).
3. When the shape drifts, the AI rewrites hparams to move back (Sprints 3, 4, 5).
4. **The AI's OWN sequence of rewrites has a measurable geometric shape too** — dim ~1.5, 3-attractor structure (Sprint 7).

That's the recursive self-similarity proposition in the original vision, now empirically characterized.

## Paired ChatGPT review

Per the dual-AI workflow: ChatGPT should review (a) whether the 3-way attractor finding is robust or driven by the specific starting configs I picked (someone else might have picked 12 starts that all converged to one region), and (b) whether Sprint 8 (more data, potentially N=100) is worthwhile vs moving on to Candidate C.
