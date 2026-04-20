# v3 Sprint 8 — Hierarchical Routing: honest negative on the mechanism

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Motivation

Sprint 7's review proposed hierarchical routing as Sprint 8:

> Dataset-level coarse gate (MNIST vs Fashion vs future) before
> task-level fine. Expected payoff: query latency stays sub-ms past
> N=10⁴, and cross-dataset overlap (Fashion signatures bleeding near
> MNIST signatures) is physically separated.

Two claims to check:
1. **Contamination protection**: does flat routing actually
   cross-pollinate between datasets? If yes, hierarchical fixes it.
2. **Latency at scale**: is the two-phase (coarse + fine) scan
   measurably faster than flat O(N)?

## Setup

### Two-dataset registry

- **MNIST kingdom** (30 tasks × 3 seeds = 90 entries):
  11 cached tasks from Sprint 3–6 + 20 `subset_*` binary tasks from
  Sprint 7.
- **Fashion kingdom** (11 tasks × 3 seeds = 33 entries):
  `fashion_class` (10-class) + 10 new Fashion binary tasks trained
  for this sprint (`fashion_upperbody, fashion_footwear,
  fashion_even_idx, fashion_first_half, fashion_corners,
  fashion_middle, fashion_no_bag, fashion_warm, fashion_athletic,
  fashion_casual`).
- **Total:** 123 entries / 41 tasks / 2 datasets.

Sprint 8 added: `src/fractaltrainer/registry/hierarchical_registry.py`
(`HierarchicalRegistry` + `HierarchicalDecision`, 14 unit tests) and
30 new Fashion experts (~3.5 min training, 7s/expert).

Test suite: **152/152 pass** (138 prior + 14 new).

### Probe batch

**Critical design choice:** signatures for all 123 experts (including
Fashion experts) use the **same MNIST probe batch** (100 MNIST test
images, `probe_seed=12345`). This is consistent with Sprint 3–7, so
all signatures live in the same space and are directly comparable.
This choice is the root cause of the negative result below.

## Results

### Inter-centroid distance: **1.32** (!)

```
Datasets: ['mnist', 'fashion']
  mnist        90 entries / 30 tasks
  fashion      33 entries / 11 tasks
  mnist ↔ fashion centroid distance: 1.320
```

**1.32 is tiny.** Compare to:
- Sprint 3 within-task same-seed MNIST distances: 0.93–3.43.
- Sprint 7 within-task max: 7.89.
- Sprint 7 cross-task min: 4.42.

The two dataset centroids are closer together than typical
within-task same-dataset pairs. This is the first warning.

### Test A — Same-task retrieval: **HIERARCHICAL LOSES** (0.976 → 0.683)

Leave seed=2024 out, register seeds 42+101 of all 41 tasks (82-entry
registry), query with each of the 41 held-out signatures:

| Method       | same-task top-1 | dataset-correct |
|--------------|----------------:|----------------:|
| Flat         | **40/41 (97.6%)** | —             |
| Hierarchical | 28/41 (68.3%)   | 29/41 (70.7%)   |

**12 disagreements** — all traceable to the coarse gate picking the
wrong dataset:

- 7 MNIST `subset_*` queries routed to Fashion (including
  `subset_01458`, `subset_0459`, `subset_01789`, `subset_014567`,
  `subset_0379`, `subset_01589`, `subset_025678`).
- 3 Fashion queries routed to MNIST (`fashion_even_idx`,
  `fashion_corners`, `fashion_no_bag`).

Every disagreement is of the form: flat found the correct
same-task sibling directly; hierarchical routed to the wrong
dataset and then couldn't find it. The 2 remaining "correct"
hierarchical misses inside the right dataset are simply task-level
retrieval failures in the sub-registry.

### Test B — Cross-dataset contamination: **DOES NOT OCCUR**

The central premise of hierarchical routing was that flat routing
could contaminate across datasets. The flat-registry measurement:

- **Flat top-1 same-dataset: 41/41 = 100%**
- Flat top-3 has ≥1 cross-dataset entry: 4/41 = 9.8% (the 2nd or
  3rd nearest sometimes crosses — the top-1 never does).

Flat routing already lands the same-dataset target every time at
top-1. The signatures are task-specific enough that within-task
tightness dominates over cross-dataset adjacency — consistent with
Sprint 7b's finding that signatures cluster by label-set overlap
rather than by dataset membership per se.

**Hierarchical routing was solving a problem that does not exist at
this scale.**

### Test C — Latency: hierarchical is **1.63× faster** but it doesn't matter

| Method       | ms/query | note                                 |
|--------------|---------:|--------------------------------------|
| Flat         | 0.363    | N=82 linear scan                     |
| Hierarchical | 0.223    | 2-centroid gate + ≤90-entry subscan  |

A 0.14 ms saving on a sub-millisecond operation is a rounding error
in any realistic application. Latency is not the bottleneck.

## Verdict: **HIERARCHICAL ROUTING LOSES AT THIS SCALE**

For the 123-entry / 2-dataset registry:
- Flat routing: **97.6% same-task accuracy, 100% same-dataset at
  top-1, 0.36 ms/query.**
- Hierarchical routing: **68.3% same-task accuracy** because the
  coarse gate is wrong 29% of the time.

The coarse gate fails because **the inter-centroid distance (1.32)
is smaller than typical within-task variance**. Centroid-based
dataset assignment is geometrically unsound when dataset clusters
overlap in the signature space they're computed from.

## Diagnosis — why the centroids are so close

The probe batch is 100 MNIST test images. For every expert
(MNIST-trained or Fashion-trained), the signature is the softmax
output of the expert on these MNIST images. The two dataset kingdoms
therefore produce signatures in the same 1000-d simplex space —
asked "what do these MNIST digits look like to you?". The dataset
identity of the training data doesn't directly live in the signature
unless the expert's response function differs on MNIST images.

Specifically: a Fashion-trained expert, asked about MNIST digits,
produces a softmax distribution that reflects its **label-binning
structure** (which digits it groups together as "class 1"), not its
training modality. Since Sprint 7b showed that **signature
similarity tracks class-1 Jaccard overlap**, two experts with similar
class-1 digit sets — even if one was trained on MNIST and the other
on Fashion — will produce similar signatures.

The "modality" signal is washed out because the probe is a fixed
MNIST batch.

## What this means for the vision

This is a **useful negative result**, not a failure of the
Mixture-of-Fractals idea. Three things it teaches:

1. **Flat routing is enough at this scale.** Sprint 7 already showed
   100% retrieval at N=93. Sprint 8 confirms 97.6% at N=82 with a
   mixed-dataset registry. Routing accuracy is not the bottleneck;
   *coverage* and *growth* are.

2. **The probe batch is part of the representation and should be
   designed thoughtfully.** For single-dataset routing, MNIST-only
   probe is fine. For cross-dataset routing, the probe should
   include all modalities the registry will host — otherwise the
   "dataset identity" gets filtered out of the signature.

3. **Centroid-based coarse routing is the wrong primitive.** Even
   with a better probe, a centroid summary statistic will fail when
   within-cluster variance is comparable to inter-cluster distance.
   A nearest-neighbor dataset classifier (or a metadata filter +
   per-dataset sub-registries used purely for bookkeeping) is more
   robust.

## What ships anyway

Despite the negative verdict on the hierarchical mechanism, the
Sprint 8 code is useful and clean:

- **`HierarchicalRegistry`** composes sub-registries by a
  configurable `dataset_key`. Mechanically works and is tested.
  The per-dataset sub-registries (and their centroids) are valuable
  as bookkeeping even when not used for routing — e.g., for
  computing dataset-specific calibration thresholds.
- **14 unit tests pass** including add/remove, centroid recomputation,
  single-dataset degenerate case, empty-registry handling,
  exclude propagation, and JSON serialization.
- **10 new Fashion binary experts × 3 seeds** cached in
  `results/sprint8_v3_fashion_trajectories/` — immediately usable
  for future cross-modality experiments.
- The comparison script `scripts/run_sprint8_hierarchical_demo.py`
  is the canonical template for any future "does X routing mechanism
  help?" question.

## Implications for Sprint 9+

1. **Don't ship `find_nearest_hierarchical` as the default.** Keep
   it as an opt-in (it works, but the coarse gate is unreliable
   given the current probe design). Sub-registries for bookkeeping
   are fine to keep.
2. **Sprint 9 direction A — fix the probe.** Mixed-modality probe
   (50 MNIST + 50 Fashion + …) would likely push inter-centroid
   distance well above within-task variance. Test: rebuild all 123
   signatures under a 50/50 probe and rerun Sprint 8 Test A/B/C.
3. **Sprint 9 direction B — drop coarse routing, keep the
   bookkeeping.** Use HierarchicalRegistry's sub-registries as
   storage-only containers. A `find_nearest` at the HierarchicalRegistry
   level could simply concatenate sub-registry results rather than
   picking one dataset. This is flat routing with hierarchical
   storage — a useful data-organization improvement with no
   retrieval regression.
4. **Sprint 9 direction C — task-function-kinship-based hierarchy.**
   Use the Sprint 7b finding: cluster the registry by Jaccard on
   label sets (not by dataset), yielding groups of
   mutually-composable experts. This is a more principled hierarchy
   that also dovetails with the compose-band redesign.

My recommendation: **Sprint 9 direction B (drop coarse routing,
keep bookkeeping)** as the first cleanup, because it's a ~20-line
change and a clear win. **Direction A (mixed-modality probe)** as
a follow-up experiment before committing to the Jaccard-based
hierarchy of direction C.

## Paired ChatGPT review prompts

A ChatGPT read should validate:

1. Whether **"flat routing is 100% same-dataset at top-1"** is a
   robust claim or an artifact of using only 41 held-out queries.
   A stronger test would use an out-of-sample held-out query set
   (e.g., train a fresh Fashion task and query it) to see whether
   new Fashion queries also land cleanly in Fashion.
2. Whether the **inter-centroid distance of 1.32** is diagnostic on
   its own or whether a Mantel-like test (comparing inter-dataset
   distance distributions to within-dataset distributions) would
   give a sharper conclusion.
3. Whether my recommended **Sprint 9 direction B (drop coarse
   routing)** is the right call, or whether the **mixed-modality
   probe fix (direction A)** should come first since it addresses
   the root cause rather than working around it.
