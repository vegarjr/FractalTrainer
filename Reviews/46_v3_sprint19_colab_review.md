# v3 Sprint 19 Colab integration review — before the paid run

**Date:** 2026-04-24
**Reviewer:** Claude Opus 4.7 (1M context)
**Decision:** Ship fixed notebook; first run `SMOKE=True` (~2 min, free), then `SMOKE=False` (~60 min on T4, ~3 hr on CPU).

## Why a review

The first-draft notebook (commit `de3d30e`) had four issues that
would either **crash the run, produce bad data, or lose results** —
all bad outcomes when Colab compute is metered. This review
documents the fixes so the paid run is trustworthy.

## Issues found

### 1. Calibration at N=15 would crash or silently produce garbage

First draft: `reg15.calibrate_thresholds(...)` on a uniform-random
subsample of 15 entries from 1000.

Math: with 500 subsets × 2 seeds = 1000 entries, a random
15-subsample has expected *within-task pair count* of

    C(15, 2) × (1 / 500) = 105 × 0.002 ≈ 0.21

So roughly 80% of random 15-subsamples have **zero within-task
pairs**. `calibrate_thresholds` then has no within-task
percentile to compute — either crashes, returns NaN, or returns a
meaningless value depending on code path. All three are bad for a
paid headless run.

**Fix:** Drop empirical per-N calibration. Use the paper's own
published MNIST-MLP thresholds (`match_t=5.0, spawn_t=7.0` from
§2). This is actually more faithful — it tests *"will the
paper's deployment thresholds still work at N=1000?"* which is
the real operational question.

### 2. Within-task distance stats degenerate at small N

Same root cause. At N=50 random, expected within-pairs ≈ 2.4 —
too few for a meaningful mean ± std. The plot error bars would be
bogus and the "within/cross gap" at N=50 would be noise.

**Fix:** Restructure task grid to **250 subsets × 4 seeds = 1000**
and **subsample *by subset*** (keep all 4 seeds of each chosen
subset). At every N the registry has exactly `N/4` subsets ×
`C(4,2)=6` pairs = clean within-task statistics.

At the smoke scale of N=40 this gives **60 within-task pairs**;
verified end-to-end locally.

### 3. Runtime estimate was wrong (cost-relevant)

First draft estimated 35–50 min on T4 but **forgot the held-out
training loop** — one expert per unique subset, needed for routing
tests. That added 500 more trainings.

**Fixed estimate** (subsets × seeds × ~2.5s per T4 training):
- `SMOKE=True`: 10 × 4 registry + 10 held-out = 50 trainings →
  **~2 min T4, ~8 min CPU**
- `SMOKE=False`: 250 × 4 + 250 = 1250 trainings →
  **~60 min T4, ~3 hr CPU**

On free Colab the 60 min fits but is risky — 90 min inactivity
timeout can kill it. Colab Pro removes that. The SMOKE path gives
2-minute full-pipeline validation before committing to the paid
run; if SMOKE passes, the full run will too.

### 4. Results would be lost on a runtime disconnect

First draft saved only to `/content/`, which is ephemeral. If the
browser tab loses focus during the 60-minute run (Colab
aggressively reclaims idle sessions) the JSON + PNG are wiped —
compute spent, nothing to show.

**Fix:** Three persistence channels:
1. **Drive mount** (`USE_DRIVE=True` by default) — results copied
   to `/content/drive/MyDrive/FractalTrainer_results/` which
   survives the runtime.
2. **Inline JSON dump** — Cell 12 prints the full payload. If the
   notebook is saved (`Ctrl+S`) after Cell 12 runs, the output
   persists in Colab's saved `.ipynb` regardless of runtime
   state.
3. **Browser download** (Cell 13) — final fallback, requires user
   to be present.

### Minor improvements

- `probe_X.to(device)` hoisted out of the training loop (saved
  redundant memcopies).
- `non_blocking=True` on per-batch `.to()` calls (minor T4
  speedup).
- Explicit `torch.cuda.empty_cache()` between experts (keeps GPU
  memory bounded).
- Added header table showing cost/runtime per mode.
- Added "what to do after Colab finishes" section at the bottom.

## How the user reviews results

### Best case (everything works)

1. Cell 12 outputs a formatted table + pass/fail gates + full
   JSON inline. `Ctrl+S` to persist this view.
2. Drive folder `FractalTrainer_results/` has the JSON + PNG.
3. Browser downloads the same files.

### If the run completes but Drive failed

- Cell 12's inline JSON is still visible — copy-paste into a
  `results/sprint19_scale_n1000.json` file locally and commit.

### If the run partially ran (e.g., timeout during Cell 6)

- The notebook itself is saved in Colab; output state through
  the last completed cell is visible. Re-open in a fresh runtime,
  re-run from Cell 6 (no caching — experts are re-trained from
  scratch). Consider shrinking `N_SUBSETS` to fit.

### Interpretation

- **Pass** (top-1 ≥ 0.9 at N=1000, p95 < 50 ms, gap > 0): paper
  §5 "scale untested" → "validated to N=1000." Adds one
  sentence to paper.
- **Fail on routing accuracy** (top-1 drops below 0.9 as N grows):
  this IS the finding. Linear-scan nearest-neighbor degrades
  because similar-but-not-identical subsets produce similar
  signatures. Motivates switching to a per-signature cluster
  structure, a learned re-ranker, or an ANN index (FAISS/HNSW)
  for scale. Publishable as a scale-limit result.
- **Fail on gap** (within-task mean catches up to cross-task as
  N grows): signature space is too narrow — would require
  higher-dim probe or learned signatures. Also publishable.

## Smoke-test evidence (local, synthetic data)

Revised logic smoke-tested end-to-end at `SMOKE=True` scale
(10 subsets × 4 seeds + 10 held-outs = 50 trainings, CPU):

    N  entries  within_pairs  cross_pairs  within  cross  gap   top1  top3  p95_lat
    8       8            12           16   10.02  10.00  -0.02  1.00  1.00  0.07ms
    16     16            24           96    9.95  11.34  +1.40  1.00  1.00  0.09ms
    32     32            48          448   10.15  10.89  +0.74  1.00  1.00  0.16ms
    40     40            60          720   10.17  10.86  +0.68  1.00  1.00  0.23ms

Within-task pair counts exactly match prediction
(`M × C(4,2) = 6M`). Latency p95 grows linearly from 0.07 to
0.23 ms across 5× registry growth — extrapolates to **~6–10 ms
at N=1000**, well under the 50 ms gate.

On synthetic data within/cross are crushed (both ≈ 10) so
verdicts collapse to 'spawn'; on real MNIST the paper observed
within ≈ 3, cross ≈ 10, which will produce `match` verdicts at
the paper's fixed thresholds.

## Deliverables

```
notebooks/scale_n1000_colab.ipynb   (replaced, 13 cells)
Reviews/46_v3_sprint19_colab_review.md   (this)
```

## Push requirement

Local `master` is 15 commits ahead of `origin/master`. The
notebook clones `master` from GitHub, so **master must be pushed
first** for Colab to see this notebook's dependencies.

Recommended:

    cd /home/vegar/Documents/FractalTrainer
    git push origin master
