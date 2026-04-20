# v3 Sprint 6 — Adaptive Threshold Calibration

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Problem

Sprint 4 picked `match_threshold=5.0, spawn_threshold=7.0` by eyeballing
the Sprint 3 MVP observation that same-task MNIST distances ran
0.93–3.43 and cross-task distances started at 7.32. That worked for the
22-entry MVP registry but has two problems:

1. As the registry grows, the same-task and cross-task distance
   distributions drift. Constants don't adapt.
2. The hand-tuned values have no justification beyond "looks right".
   A reader reviewing the code cannot tell whether 5.0/7.0 was
   carefully chosen or chosen by accident.

This sprint replaces constants with a **percentile-based calibration**
that computes thresholds from the registry's own pairwise distance
distribution, grouped by `metadata['task']`.

## Machinery

### `FractalRegistry.calibrate_thresholds(task_key, within_percentile, cross_percentile)`

Returns a `CalibrationResult`:

- **match_threshold** = the `within_percentile` (default 95) of
  within-task pairwise L2 distances. Meaning: ≥within_percentile of
  same-task queries will land in the `match` verdict.
- **spawn_threshold** = the `cross_percentile` (default 5) of
  cross-task pairwise distances. Meaning: ≥(100 − cross_percentile)%
  of cross-task queries will land in the `spawn` verdict.
- **overlap** = True when raw within_p95 > raw cross_p05 (the two
  distributions don't have a clean separation at those percentiles);
  spawn_threshold is clamped to ≥ match_threshold so the decide()
  invariant holds.

Errors informatively on: single-task registry, missing task metadata,
all-singleton tasks (no within-task pairs), <2 entries.

### 11 new unit tests (`tests/test_calibration.py`)

Covers: basic clean-gap, overlap detection, single-task rejection,
missing-metadata rejection, custom percentiles, custom task_key,
serialization, all-singleton rejection, empty-registry rejection.

Full suite: **138/138 pass** (127 prior + 11 new).

## Results (`scripts/run_calibration_demo.py`)

| Registry                            | N   | tasks | match | spawn | gap    |
|-------------------------------------|----:|------:|------:|------:|-------:|
| Hand-tuned (Sprint 4)               |  -  |   -   | 5.000 | 7.000 | 2.000  |
| Sprint 3 MVP (22 entries, 11 tasks) |  22 |  11   | 5.723 | 7.557 | 1.834  |
| Full (33 entries, 11 tasks)         |  33 |  11   | 7.141 | 7.593 | 0.452  |
| Binary-only (21 entries, 7 tasks)   |  21 |   7   | 3.235 | 7.385 | 4.150  |

### What these numbers say

1. **MVP (22 entries) ≈ hand-tuned.** Calibrated (5.72, 7.56) is within
   15% of hand-tuned (5.0, 7.0). The method reproduces the original
   intuition **without tuning**.
2. **Full registry (33 entries) compresses to a 0.45 gap.** The p95 of
   within-task jumped from 5.72 → 7.14 because one same-task pair has
   a distance of 7.89 (max within-task) — the 95th percentile is
   sensitive to a single outlier when you have only 33 within-task
   pairs. Cross-task p5 barely moved (7.56 → 7.59). Net: the compose
   band narrows sharply, which is consistent with Sprint 5's finding
   that compose buys nothing anyway.
3. **Binary-only registry has a wide 4.15 gap.** Within-task p95 drops
   to 3.24 because binary tasks have tighter within-task variance than
   digit_class or fashion_class (which max out at 7.89 within-task).
   Homogeneous-complexity tasks → cleaner discrimination.
4. **Retrieval under calibrated MVP thresholds:** 10/11 match, 1
   compose, 0 spawn on leave-one-seed-out queries. Nearest-task is
   correct on 11/11. The 1 compose is a boundary case where the
   held-out seed=2024 entry sat just above 5.72 from its siblings.

## Key insight

Adaptive thresholds work, but **they inherit the within-task variance
of the most-variable task in the registry**. If one task's same-seed
runs disagree more (e.g., digit_class 10-class vs parity 2-class), its
within-task distances dominate the p95 and push `match_threshold` up
for everyone.

Sprint 5 already showed the compose verdict gives no lift on this
registry, so the right response to a narrow calibrated gap isn't to
widen it — it's to accept that same-task and cross-task signatures
are adequately but not perfectly separated, and the few boundary
queries end up in compose (which degrades gracefully to top-1 anyway).

## What ships

- `FractalRegistry.calibrate_thresholds()` method and `CalibrationResult`
  dataclass, both exported from `fractaltrainer.registry`.
- 11 unit tests (all pass, full suite 138/138).
- `scripts/run_calibration_demo.py` runs calibration on three registry
  layouts and reports per-verdict retrieval.
- `results/calibration_demo.json` with full distance stats per layout.

No change to `decide()` signature — it still takes thresholds
explicitly. The calibration is a **pre-processing step** that produces
values to pass into `decide()`, not a new decision mode. This
preserves the ability to pass hand-tuned values for regression tests
or domain-specific overrides.

## Limits

- **Percentile defaults (95, 5) are a design choice, not a theorem.**
  Lowering `within_percentile` to 90 or 75 gives more permissive match
  (more queries land in match, fewer in compose/spawn). Sprint 6 did
  not sweep this — `custom_percentiles` unit test demonstrates the
  parameter works but doesn't argue for a particular choice.
- **Outliers dominate.** One anomalous same-task pair at distance 7.89
  shifts p95 from 5.7 to 7.1 in the full registry. A robustified
  alternative would use median + MAD instead of percentiles, or
  trimmed percentiles. Not implemented.
- **Single-pair-of-thresholds assumes homogeneous-complexity tasks.**
  Mixing a 10-class image task with 2-class relabelings in the same
  registry produces a within-task distribution with heavy tails. A
  per-task-complexity-bin calibration (e.g., separate thresholds for
  binary vs multiclass) would be a natural extension — effectively a
  two-level hierarchy, which is Sprint 8's territory.
- **Calibration uses all registered entries.** As the registry grows,
  recomputing every time is O(N²). Fine for N ≤ a few thousand; past
  that, an online-updating running percentile estimator would be
  better. Deferred.
- **Overlap clamp destroys information.** When raw within-p95 > raw
  cross-p5, we clamp spawn to match (compose band = 0). A more
  informative response would be to surface the overlap in the decide
  path (e.g., return `verdict="ambiguous"` when the distance falls in
  a statistical overlap region). Not implemented; for the current
  registry `overlap=False` in every layout.

## Natural follow-ups

- **v3 Sprint 7 — Scale stress.** Push past 30 tasks to see whether
  calibration stays well-behaved when the within/cross distance
  distributions get richer. Concrete: add 20+ MNIST relabelings
  (randomized binary splits over digits) → 50-task × 3 seeds = 150
  entries. See whether calibrated thresholds track what the hand-tuned
  (5, 7) would predict, or diverge.
- **v3 Sprint 8 — Hierarchical / per-complexity calibration.** Split
  the registry by task complexity (binary / 3-class / 10-class) and
  calibrate separately. Query hits the appropriate subset first, then
  the subset's thresholds apply. This is the natural response to the
  "single-pair-of-thresholds is too rigid" limit above.
- **Update `run_growth_demo.py` to call `calibrate_thresholds()` by
  default** instead of hardcoded 5.0/7.0. Deferred to avoid changing
  the demo's baseline numbers; Sprint 7 or 8 can fold this in.

## Paired ChatGPT review prompts

A ChatGPT read should validate (a) whether (95, 5) are defensible
defaults or whether a stricter (99, 1) would be more conservative
against false-match errors; (b) whether the "clamp spawn to match on
overlap" is the right graceful-degradation path or whether raising on
overlap would force the user to think before proceeding; (c) whether
the within-task outlier problem (1 pair at 7.89 compressing the full
registry's gap) deserves a robustified statistic (median+MAD) or is a
signal that the registry has a real quality issue worth investigating.
