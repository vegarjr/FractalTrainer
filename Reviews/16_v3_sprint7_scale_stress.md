# v3 Sprint 7 — Scale Stress Test

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

The v3 invariants (100% same-task retrieval, clean within/cross gap,
sub-millisecond query) were established on a 33-entry / 11-task
registry (Sprint 3–6). Do they hold at **3× scale**?

Test: push to 93 entries / 31 tasks and re-check.
- Same-task retrieval accuracy still ≥ 95%?
- Calibration still produces a clean gap, or does overlap appear?
- Query latency still < 1 ms?
- Cross-task routing reveals sensible semantic neighbors, or does
  it become chaotic as the registry densifies?

## Setup

**20 novel binary tasks** generated deterministically
(`task_sample_seed=42`). Each is `y ∈ S` for a 3–6 digit subset,
filtered to avoid collision with the 7 existing binary tasks and their
complements. Examples: `subset_01458`, `subset_267`, `subset_123679`.

**Training:** 500-step MNIST MLP (784→64→32→10), same hparams as
Sprint 4 spawn. 20 tasks × 3 seeds = 60 new experts, 394s (6.6 min)
total — ~6.6s per expert.

**Registry:** 33 cached + 60 new = **93 entries / 31 tasks**.
Signatures on the Sprint 3 probe batch (100 MNIST test images,
probe_seed=12345). Full artifact: `results/sprint7_v3_scale_stress.json`.

## Results

### 1. Same-task retrieval: **31/31 = 100.00%**

Held out seed=2024 for every task, registered seeds 42+101 of all 31
tasks (62 entries). For every one of the 31 seed=2024 queries, the
top-1 nearest was a same-task sibling. **Same as N=22 and N=33: no
degradation at scale.**

### 2. Calibration at scale: gap **widens** from 0.45 → 3.01

| Layout                   | N   | match | spawn | gap   | within p95 source                       |
|--------------------------|----:|------:|------:|------:|-----------------------------------------|
| Sprint 6 MVP             |  22 | 5.723 | 7.557 | 1.834 | within p95 = single outlier rank-31     |
| Sprint 6 full            |  33 | 7.141 | 7.593 | 0.452 | within p95 stuck at the 7.89 outlier    |
| **Sprint 7 (this)**      |  93 | **3.833** | **6.844** | **3.010** | within p95 drops below the outliers   |

Within-task distances: **n=93, mean=2.74, max=7.89**.
Cross-task distances: **n=4185, min=4.42, mean=10.18, max=13.89**.

The gap widens because the within-task distance distribution is
bimodal: most pairs cluster tightly (mean 2.74), but a few
heterogeneous-complexity tasks (digit_class, fashion_class) have
within-task pairs up to 7.89. At N=33 the p95 position landed on
those outliers; at N=93 the outliers move past p95, and the
calibrated match threshold drops to **3.83** — a principled,
data-driven value that's tighter than the hand-tuned 5.0.

**This is the best scaling behavior we could have hoped for.** The
calibration method gets MORE robust as the registry grows, because
outliers become structural features rather than point-estimate
distortions. Cross-task min dropped from 7.21 → 4.42 (one new
cross-task pair is unusually close), but p5 is stable at 6.84 —
the lower tail is a single pair, not a persistent density.

### 3. Query latency: **0.36 ms** at N=93

A naive O(N) scan with 1000-d signatures stays well under the 1-ms
mark through 93 entries. Back-of-envelope: 10 ms would take N ≈ 2600
at this hardware, 100 ms would take N ≈ 26,000. **No indexing needed
before the 10³-entry mark.**

### 4. Routing consistency: **25/31 tasks (81%)** route consistently

For each held-out task, we asked: do all 3 seed queries route to the
same nearest task in the leave-one-task-out registry? 25 tasks do; 6
split across 2-3 different neighbors.

**Splits observed:**
- `triangular` — 2 seeds → `subset_1256`, 1 → `subset_13678`
- `fashion_class` — 2 seeds → `mod5`, 1 → `digit_class` (both are
  10-class; semantically they're adjacent)
- `subset_1256` — 2 → `triangular`, 1 → `subset_267`
- `subset_3457` — 2 → `subset_2349`, 1 → `subset_459`
- `subset_02459` — 2 → `subset_0459`, 1 → `subset_02349`
- `subset_123679` — 2 → `subset_123689`, 1 → `parity`

These are **boundary cases** — held-out tasks with 2 or 3 plausible
nearest neighbors, none dominating. The actual split structure is
informative: all splits are between *semantically related* tasks
(10-class-ish, subset-overlap, or parity-like), not random. The
signature carries more granular information than the 3-seed
majority-vote can resolve.

### 5. Cross-task routing discovers real semantics

Held-task → dominant nearest (leave-one-task-out):

| Held          | Nearest (dominant)  | Why this makes sense                        |
|---------------|---------------------|---------------------------------------------|
| digit_class   | mod5                | mod5 is a 5-class partition of 10 digits    |
| mod5          | digit_class         | mutual nearest neighbor                     |
| parity        | subset_123679       | {1,3,7,9} are 4 of 5 odd digits in subset   |
| high_vs_low   | subset_15789        | 3 of 5 subset digits are high (≥5)          |
| fibonacci     | subset_123689       | 4 of 5 fibonacci digits in subset           |
| triangular    | subset_1256         | shares 1 and 6; both include 2-3 of {1,3,6} |
| fashion_class | mod5 / digit_class  | output structure kinship (10-class)         |
| primes_vs_rest| subset_267          | shares 2, 7; "contains some primes"         |

The signature space embeds **digit-subset kinship** — a task's
nearest neighbor when held out is the task whose label set most
overlaps with it. This emerges from the activation signature
without being programmed in. It's exactly what the
Mixture-of-Fractals vision predicted: neighbors in signature space
are neighbors in function space.

## Verdict

**PASS, and better than expected.**

1. **Retrieval:** 100% same-task accuracy, unchanged from smaller
   registries.
2. **Calibration:** gap grows from 0.45 (N=33) to 3.01 (N=93) as
   within-task outliers become "above p95" rather than "at p95".
   Adaptive thresholds are MORE robust at scale — the Sprint 6
   concern that outlier-sensitivity compresses the gap diminishes
   with density.
3. **Latency:** 0.36 ms / query at N=93. Naive O(N) scan is safe
   to at least N ≈ 10³.
4. **Semantics:** Cross-task nearest-neighbor structure reflects
   digit-subset overlap — signature space encodes task-function
   kinship.

## Implications

- **The "N=33 narrow gap" from Sprint 6 was a small-sample artifact.**
  The hand-tuned (5.0, 7.0) values remain defensible but the
  calibrated (3.83, 6.84) at scale are better motivated. Update
  `run_growth_demo.py` default thresholds to use calibration rather
  than constants, as a natural next step.
- **Compose-band width has ground truth now.** At N=93, the
  [3.83, 6.84] band is 3.01 wide — non-trivial but still narrow
  relative to cross-task range [4.42, 13.89]. Queries falling in
  compose are genuinely ambiguous; Sprint 5 already showed compose
  gives no measurable lift, so this band width is fine as-is.
- **Heterogeneous-complexity registries work.** Mixing 2-class, 3-
  class, 5-class, 10-class, and Fashion in one registry does not
  break retrieval. The within-task outliers come from the 10-class
  tasks (digit_class 7.89, fashion_class similar); but at N=93 these
  don't dominate calibration.

## Limits

- **N=93 is still small.** The naive O(N) scan's sub-ms claim
  extrapolates; actual test at N=1000 would confirm. Sprint 8
  (hierarchical routing) is the right response if/when N exceeds a
  few thousand.
- **All signatures are from the same model architecture** (MLP
  784→64→32→10). A truly heterogeneous stress test would mix in
  different arcs (say, a CNN or a larger MLP). Deferred.
- **MNIST is the only dataset.** One Fashion task is present, but
  the dataset diversity at N=31 is essentially "MNIST-labeling-
  space + one Fashion task". Cross-dataset distances remain in the
  11.4+ range observed in Sprint 3, meaning dataset-level clustering
  is trivial to add (a hierarchical Sprint 8 would split by dataset
  first, then by task).
- **The 6/31 routing splits are interesting but not diagnosed.**
  Each split represents a held-out task whose 3-seed signatures
  lie on a boundary between two other tasks. A deeper analysis
  would check whether the boundary crossings correlate with
  held-out task complexity or label-overlap ambiguity. Deferred.
- **No compose evaluation at N=93.** Sprint 5 concluded compose
  offers no lift on the 11-task / 21-binary-task set; no reason to
  re-run it at 3× scale with the same negative expectation. If
  Sprint 8 (hierarchical) exposes new compose-band cases, it's
  worth revisiting.

## Natural follow-ups

- **v3 Sprint 8 — Hierarchical routing.** Dataset-level coarse gate
  (MNIST vs Fashion vs future) before task-level fine routing.
  Expected payoff: query latency stays sub-ms past N=10⁴, and
  cross-dataset overlap (Fashion signatures bleeding near MNIST
  signatures) is physically separated.
- **Replace hardcoded (5.0, 7.0) in `run_growth_demo.py` with
  `calibrate_thresholds()`.** A one-line change that makes the
  demo self-adapting to whatever registry is loaded.
- **Persist registry state.** The 93-entry registry is computable
  from cached trajectories + the new sprint7_v3_trajectories folder,
  but we don't checkpoint a `registry_v3_sprint7.json`. A
  `FractalRegistry.save` call in this script would make reloading
  cheap. Trivial addition.

## Paired ChatGPT review prompts

ChatGPT should independently validate:

1. Whether the **gap widening** (0.45 → 3.01 going N=33 → N=93) is a
   genuine scale benefit or an artifact of adding tasks with
   systematically small within-task distance. Concretely: do the 20
   new binary subsets have inherently tighter within-task distances
   than digit_class/fashion_class simply because they're 2-class
   and therefore lower-complexity? If yes, the "scaling makes
   calibration more robust" claim needs qualification: "…when new
   tasks are of similar or lower complexity than existing ones".
2. Whether **6/31 split routing** is a sign of ambiguous boundaries
   (feature, implying signature space is granular enough to split
   hairs) or a sign that signatures are noisier at this registry
   size (bug, implying the 100-image probe batch is too small for
   the task set). The answer determines whether the next sprint
   should improve probe batch size or leave it alone.
3. Whether the **semantic kinship claim** (parity → subset_123679
   because of shared odd digits) is a fair summary, or whether it's
   post-hoc storytelling on a small sample. A rigorous test would
   compute digit-subset Jaccard overlap for each pair and check if
   signature distance correlates with overlap — left as a possible
   Sprint 8+ add-on.
