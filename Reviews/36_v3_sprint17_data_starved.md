# v3 Sprint 17 — Data-starved ablation

**Date:** 2026-04-21 (same-day follow-up to Review 35)
**Reviewer:** Claude Opus 4.7 (1M context)

## Purpose

Previous ablations (Review 34) varied **training budget N** (number of
Adam steps on a 5000-sample training set). Review 35 showed C helps on
MNIST but not on Fashion-MNIST because Fashion saturates too quickly
at any budget N ≥ 50 — the "cold-start" regime there is absent.

But that ablation conflated two things:
- **Training-starved:** few gradient steps on plenty of data
- **Data-starved:** plenty of gradient steps on few labeled samples

The practically interesting regime is the second — a user comes with
50 labeled examples and wants to spawn a working model. Does context
from K=3 registry neighbors help them reach useful accuracy faster?

**Change:** `--train-size` now controls Q_spawn's sample count;
`--seed-train-size` keeps the seed registry at its default 5000.
Ran both MNIST and Fashion with `--train-size 50`.

## Results

### MNIST subset_019, train_size=50, seed registry at train_size=5000

**Pipeline side note:** with the Q_spawn task's oracle signature now
computed from just 50 samples, both Q_compose and Q_spawn routed to
the spawn verdict (oracle signatures drift when training data is
scarce). The F pipeline gracefully handles this by spawning extra
experts rather than forcing a bad compose/match. The ablation table
below is for the designated Q_spawn task (subset_019, labels={0,1,9}).

| Arm | N=50 | N=100 | N=300 | N=500 | N=1000 |
|---|---|---|---|---|---|
| A — no context             | 0.797±0.004 | 0.797±0.003 | 0.798±0.002 | 0.798±0.002 | 0.799±0.002 |
| **B — K=3 nearest ctx**    | **0.847**±0.005 | **0.849**±0.002 | **0.850**±0.003 | **0.850**±0.003 | **0.848**±0.003 |
| C — K=3 random ctx         | 0.799±0.002 | 0.798±0.002 | 0.800±0.004 | 0.802±0.003 | 0.804±0.002 |

**Acceptance PASS at every budget.** B beats A by **+5.0 pp absolute
at N=1000**, and the gap holds across all budgets. Both arms are flat
across N — the model converges to its ceiling quickly on 50 samples,
and C's lift is a *ceiling shift*, not a convergence acceleration.

Arm C (random) sits exactly at A, confirming the lift comes from
routing — the signal is from actual nearest-neighbor feature
detectors, not from any auxiliary input noise.

### Fashion subset_019, train_size=50, seed registry at train_size=5000

| Arm | N=50 | N=100 | N=300 | N=500 | N=1000 |
|---|---|---|---|---|---|
| A — no context             | 0.866±0.004 | 0.869±0.008 | 0.870±0.005 | 0.870±0.005 | 0.869±0.005 |
| B — K=3 nearest ctx        | 0.863±0.006 | 0.865±0.003 | 0.865±0.003 | 0.866±0.003 | 0.865±0.003 |
| C — K=3 random ctx         | 0.861±0.006 | 0.866±0.008 | 0.864±0.009 | 0.863±0.011 | 0.867±0.010 |

**Acceptance FAIL.** All arms are tied at ~0.865 across all budgets.
Context injection provides no measurable benefit on Fashion
subset_019 even in the data-starved regime.

## Interpretation

The two datasets tell a consistent story once you notice the A
ceilings:

- **MNIST A-ceiling (50 samples):** 0.798. MNIST binary
  subset_019 ({0,1,9} vs others) genuinely needs more data than 50
  samples to separate cleanly — digits 0 and 6, 8, and 9 share
  visual features that a 50-sample model can't disambiguate.
  Context from neighbors lifts this ceiling to 0.85 (+5.0 pp).
- **Fashion A-ceiling (50 samples):** 0.869. Fashion binary
  subset_019 ({T-shirt=0, Trouser=1, Ankle boot=9} vs others) is
  visually easier — three distinct silhouettes. A 50-sample model
  already hits ~0.87 because the feature distinctions are coarse
  enough to generalize from few examples.
- **The gap rule:** Context injection lifts the ceiling in
  proportion to how much is missing from A. MNIST has 20 pp of room
  between A (0.80) and a full-data ceiling (~0.96). Fashion has 8 pp
  (0.87 to 0.95). MNIST's extra headroom is where C pays; Fashion's
  narrower gap doesn't leave enough room.

This unifies the Review 34/35/36 story:
- **Training-starved MNIST (Rev 34):** A at N=25 is 0.842, ceiling ~0.96. C lifts to 0.884 (+4.2 pp).
- **Training-starved Fashion (Rev 35):** A at N=25 is 0.916, ceiling ~0.95. C lifts to 0.925 (+0.9 pp, within noise).
- **Data-starved MNIST (this):** A is 0.799, ceiling ~0.96. C lifts to 0.848 (+5.0 pp).
- **Data-starved Fashion (this):** A is 0.869, ceiling ~0.95. C makes no difference.

**Rule:** C's benefit ≈ (some fraction of) the gap between A's
achieved accuracy and the task's full-data ceiling. When the gap is
small (Fashion), C contributes nothing. When the gap is large (MNIST
at reduced data or reduced training), C pays proportionally.

The 50-sample MNIST regime is the **most practically interesting**
case we've tested: +5.0 pp absolute lift on a realistic few-shot
scenario, with training stable across a wide budget range. This is
the scenario a deployed fractal registry would handle often — a new
user with 50 labels — and C delivers meaningfully there.

## Cross-ablation summary

| Regime | Dataset | Base task | C benefit? | Notes |
|---|---|---|---|---|
| Training-starved (Rev 34) | MNIST | binary subset_019 | **+4.2pp @ N=25** | cold-start regime exists |
| Training-starved full (Rev 35) | MNIST | binary subset_019 | +2.7pp @ N=100 | converges by N=1000 |
| Training-starved full (Rev 35) | Fashion | binary subset_019 | none | task too easy — all arms tied |
| Training-starved cold (Rev 35) | Fashion | binary subset_019 | none | no starved phase to fill |
| **Data-starved (this review)** | MNIST | binary subset_019 | **+5.0pp across all budgets** | 50 samples — biggest C effect recorded |
| **Data-starved (this review)** | Fashion | binary subset_019 | none | A already at ~0.87 ceiling |

## What ships

- `scripts/run_fractal_demo.py` — `--seed-train-size` + `--seed-batch-size`
  flags added; `_train_loader` clamps batch_size to ≤ train_size for
  tiny sample counts
- `results/fractal_demo_mnist_starved_data.{json,png}`
- `results/fractal_demo_fashion_starved_data.{json,png}`
- `Reviews/36_v3_sprint17_data_starved.md` — this doc

## Paired ChatGPT review prompts

1. **On the "gap rule".** "C's benefit scales with the gap between
   A's achieved accuracy and the task's full-data ceiling" is the
   emerging interpretation after four ablations. Is that a correct
   reading, or is the Fashion null result due to something else
   specific to Fashion-MNIST (label geometry, pixel statistics, low-
   dimensionality of the task)? A third dataset (CIFAR, Omniglot)
   would disambiguate.
2. **On routing drift in data-starved mode.** The F pipeline still
   works — routing gracefully falls back to spawn when oracle
   signatures drift. But Q_compose was *supposed* to compose and
   got routed to spawn instead. Is that degradation mode a bug, or
   a feature? A separate match/compose/spawn threshold calibration
   based on the oracle's training size might fix it; or it might be
   correct that "uncertain signatures should spawn" is the right
   default.
3. **On C as a deployable primitive.** The 50-sample MNIST case
   shows +5.0 pp absolute from a primitive that costs ~0 extra
   training compute. That's a clean practical claim. Is it enough
   evidence to ship C as a default-on feature (if context_mode is
   available), or does it need cross-modal validation first
   (CIFAR, tabular, NLP) to rule out MNIST-specific artifacts?
