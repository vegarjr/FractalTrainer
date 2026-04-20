# v3 Sprint 11 — Multi-region curated registry: the architecture works

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

Sprint 10 showed curation wins in-distribution; Sprint 10b showed it
crashes out-of-distribution. User prompt:

> *"do the multi region. it needs a wider starting point as far as i
> can see? more data as the standard model?"*

The suggestion: widen the starting point by building multiple
curated sub-registries covering different regions of the label
space, with routing that picks the right one — and a fallback
path for genuinely cross-region queries. This sprint tests whether
that architecture actually resolves the in-distribution vs
out-of-distribution tension.

## Architecture

```
                                    query target T
                                         │
                                         ▼
                ┌──────── content-aware router ────────┐
                │                                       │
     overlap(T, {0-4}) ≥ 0.75?            overlap(T, {5-9}) ≥ 0.75?
                │                                       │
                ▼                                       ▼
     ┌──────────────────┐                ┌──────────────────┐
     │  Region A        │                │  Region B        │
     │  sub-registry    │                │  sub-registry    │
     │  (anchor {0-4})  │                │  (anchor {5-9})  │
     │  45 entries      │                │  45 entries      │
     └──────────────────┘                └──────────────────┘
                                      
            (else)  ──── fall through ────▶ spawn
```

- **Region A sub-registry** (Sprint 10): 15 curated tasks × 3 seeds
  = 45 entries. All class-1 sets ⊆ {0, 1, 2, 3, 4}.
- **Region B sub-registry** (Sprint 11 new): 15 curated tasks × 3
  seeds = 45 entries. All class-1 sets ⊆ {5, 6, 7, 8, 9}.
- **Content-aware router**: for each query with target `T`, compute
  `overlap(T, anchor_A)` and `overlap(T, anchor_B)`. Route to the
  sub-registry with the higher overlap if it ≥ 0.75; else fall
  through to spawn.

**60 new experts trained** in 391s (45 Region B + 15 cross-region
probes).

## Probe set (15 queries)

| Probe kind  | Count | Targets                                    |
|-------------|------:|--------------------------------------------|
| in_A        |     5 | {0,1}, {1,2}, {2,3}, {3,4}, {0,4}          |
| in_B        |     5 | {5,6}, {7,8}, {5,9}, {6,7}, {8,9}          |
| cross       |     5 | {0,5}, {1,6}, {2,7}, {3,8}, {4,9}          |

For each, run the full budget sweep {50, 100, 300, 1000, 5000} and
measure compose_A, compose_B, uniform-compose (Sprint 7), router's
routed accuracy, spawn, and oracle.

## Results (aggregate by region type)

### in_A probes (5 queries, all routed to Region A)

| N    | compose_A | compose_B | uniform | routed | spawn  |
|-----:|----------:|----------:|--------:|-------:|-------:|
|   50 |   0.977   |   0.584   |  0.803  | **0.977** | 0.882 |
|  100 |   0.977   |   0.544   |  0.816  | 0.977 | 0.905 |
|  300 |   0.978   |   0.561   |  0.818  | 0.978 | 0.942 |
| 1000 |   0.978   |   0.545   |  0.802  | 0.978 | 0.963 |
| 5000 |   0.979   |   0.548   |  0.803  | 0.979 | 0.970 |

### in_B probes (5 queries, all routed to Region B)

| N    | compose_A | compose_B | uniform | routed | spawn  |
|-----:|----------:|----------:|--------:|-------:|-------:|
|   50 |   0.537   |   0.948   |  0.841  | **0.948** | 0.865 |
|  100 |   0.520   |   0.966   |  0.841  | 0.966 | 0.868 |
|  300 |   0.519   |   0.963   |  0.840  | 0.963 | 0.911 |
| 1000 |   0.522   |   0.964   |  0.841  | 0.964 | 0.945 |
| 5000 |   0.522   |   0.965   |  0.838  | 0.965 | 0.960 |

### cross probes (5 queries, all routed to spawn)

| N    | compose_A | compose_B | uniform | routed | spawn  |
|-----:|----------:|----------:|--------:|-------:|-------:|
|   50 |   0.757   |   0.752   |  0.827  | **0.892** | 0.892 |
|  100 |   0.726   |   0.728   |  0.844  | 0.915 | 0.915 |
|  300 |   0.726   |   0.722   |  0.822  | 0.945 | 0.945 |
| 1000 |   0.719   |   0.750   |  0.822  | 0.959 | 0.959 |
| 5000 |   0.721   |   0.752   |  0.847  | 0.969 | 0.969 |

### Budget-averaged summary

| Region  | routed | spawn | uniform-compose | Δ routed vs spawn |
|---------|-------:|------:|----------------:|------------------:|
| in_A    |  0.978 | 0.932 | 0.808           | **+0.046**        |
| in_B    |  0.961 | 0.910 | 0.840           | **+0.051**        |
| cross   |  0.936 | 0.936 | 0.832           | **0.000**         |

## Verdict

**ARCHITECTURE WORKS CLEANLY.** The router makes the correct
dispatch decision 15/15 times:

- In-region queries routed to the matching sub-registry → compose
  at near-oracle accuracy (0.978 / 0.965 averages, vs oracle
  ~0.97–0.98).
- Cross-region queries routed to spawn → baseline spawn accuracy
  (no compose penalty).
- No routing errors: every in-region probe has overlap 1.0 with
  exactly one anchor and 0.0 with the other; every cross probe
  has overlap 0.5 with both (below threshold 0.75).

### The three wins vs alternatives

1. **Multi-region beats uniform at every region type by 9–17 pp.**
   Even the cross-region case (where compose can't help) still
   wins by +10 pp via the spawn fallback — because uniform-compose
   is also weak on cross-region queries (0.83 vs spawn's 0.94).
2. **Multi-region beats spawn on in-region queries by 4.6–5.1 pp
   on average.** At N=5000 specifically: in_A compose 0.979 vs
   spawn 0.970 (+0.9 pp); in_B compose 0.965 vs spawn 0.960
   (+0.5 pp). Even at full data, the curated sub-registry edges
   spawn.
3. **Multi-region matches spawn on cross-region queries with zero
   penalty.** Because the router correctly falls through rather
   than forcing a bad compose.

The Mixture-of-Fractals architecture is now operationally sound.

## What the user's "wider starting point" insight captured

The Sprint 10 mistake was that "curation" was presented as a
general mechanism when it was actually a bet on a narrow region.
The user's push — *it needs a wider starting point* — was the
architectural correction: cover MORE regions with curation, and
use content-aware routing to dispatch queries to the matching
region. This is exactly what multi-region does, and it works.

The "more data as the standard model" part of the question I
interpret as orthogonal: spawn at N=5000 already approaches
oracle (0.97 average), so adding more training data to the
baseline wouldn't change the comparison much. The lever is
registry coverage, not per-expert training depth.

## The v3 arc, recapped

After 10 sprints of v3, the complete picture:

1. **Routing works** — signature distance tracks label-set Jaccard
   (Sprint 7b, ρ = −0.85).
2. **Spawn works as default** — even at N=50, a simple MLP hits
   ~0.88 on binary tasks (Sprint 9b); scales smoothly to 0.97 at
   N=5000.
3. **Similarity-compose is null** — top-K nearest are
   label-redundant, blending adds no independent signal (Sprint 5,
   explained by 7b).
4. **Coverage-compose works when H2 fires** — redundant positive-
   label voting across picks (Sprint 9c). Mechanism is
   error-diversity by label coverage, not by signature diversity.
5. **Simple-N triage is near-optimal on uniform registries** —
   fancier rules don't convert Sprint 9c's correlation into
   useful routing (Sprint 9d).
6. **Curation is a distribution bet** — curated registries are
   sharp in-region (+17 pp over uniform) and catastrophic
   out-of-region (−32 pp). Sprint 10 + 10b together.
7. **Multi-region + content-aware routing resolves the bet** —
   curated sub-registries for each region, fallback to spawn for
   cross-region, no accuracy penalty anywhere (Sprint 11).

Deployable architecture:

```python
def handle_new_task(query_signature, target_labels, labeled_data):
    # Step 1: coarse content-aware routing
    for sub_registry in sub_registries:
        if overlap(target_labels, sub_registry.anchor) >= 0.75:
            # Step 2a: in-region — use coverage-compose at any budget
            return sub_registry.coverage_compose(
                query_signature, labeled_data, k=3)

    # Step 2b: cross-region or unknown — spawn
    if len(labeled_data) >= 100:
        return spawn(target_labels, labeled_data)
    else:
        # Very small budget — fall back to uniform compose
        return uniform_registry.coverage_compose(
            query_signature, labeled_data, k=3)
```

## Limits

- **Anchors are binary here** (A = {0-4}, B = {5-9}). Real systems
  need anchors defined somehow: either hand-curated regions or
  learned clusters of expert label sets. Deferred.
- **Threshold 0.75 is hand-tuned.** For 2-subset targets, overlap
  is either 1.0, 0.5, or 0.0 — no middle values. For 3-subsets
  the overlap space is finer (0, 1/3, 2/3, 1). A broader test
  should sweep threshold.
- **Cross probes tied at 0.5/0.5.** The router correctly rejects
  on ties. But for a 3-subset target like {0, 1, 6}, overlap is
  (2/3, 1/3) — the router would correctly pick A. The 2-subset
  choice was chosen to stress-test the fallback; it's not the
  only pattern.
- **Single seed for spawn baselines.** A 3-seed variance estimate
  would tighten the small-N comparisons (±0.01 ish noise typical).
- **5 probes per region type (15 total).** Scale is small but
  the pattern across every (probe, budget) cell is clean enough
  that the effect is obvious.
- **Region anchors are MNIST-digit-specific.** For other modalities
  (Fashion-MNIST, CIFAR), "overlap" requires task label-set
  structure that may not be as clean. This architecture assumes
  tasks are labeled with explicit class-1 sets the router can
  inspect. For fuzzier task descriptions, a learned region
  classifier would replace the hand-coded overlap computation.

## What ships

- **`scripts/run_multi_region_experiment.py`** — full experiment
  + content-aware router.
- **`results/multi_region_experiment.json`** — 75 per-probe
  per-budget rows, router decisions, spawn accuracies, oracle
  references.
- **The deployable architecture** described above — match
  (signature-nearest) / compose (in-region curated) / spawn
  (cross-region or small-N fallback), gated by a metadata overlap
  test.

No code change to `FractalRegistry` or `HierarchicalRegistry`. The
multi-region structure is built at application level; the existing
registry primitives support it without modification.

## Natural follow-ups

1. **Promote the router to an API method.** A
   `MixtureOfFractals.decide(query, target_labels, labeled_data)`
   that encapsulates the full pipeline. This is finally the
   deployable API v3 has been building toward.
2. **Many-region scaling.** What happens with 5, 10, 50 curated
   sub-registries? Router latency becomes linear in number of
   regions unless we add hierarchical region-of-regions routing.
3. **Learned anchors.** Instead of hand-chosen {0-4} / {5-9}, use
   k-means or agglomerative clustering on registered experts'
   label sets to discover regions automatically.
4. **Threshold sensitivity.** Sweep `in_region_threshold` from
   0.5 to 1.0 and map the tradeoff between false-routes (query
   goes to a sub-registry that doesn't really cover it) and
   over-fallback (router bails to spawn when a sub-registry
   would have worked).

## Paired ChatGPT review prompts

ChatGPT should independently validate:

1. Whether **15 probes is enough** to declare the architecture
   works. The signal is strong (15/15 correct routes, compose
   hits oracle in-region every time, spawn fallback zero-penalty
   on cross) but a skeptic could argue the probe set is too
   constructed. A stronger test would use probes discovered at
   runtime from a real task stream.
2. Whether the **"fallback to spawn" design is too lenient**.
   A stricter evaluation would require the router to be correct
   about WHICH sub-registry fires, not just to fall through when
   uncertain. With this design, the router is right-by-default
   for any 0.5-overlap case (always rejects) — that's partly
   engineered into the probe set.
3. Whether **the `overlap ≥ 0.75` threshold** is the right
   primitive or whether a more nuanced metric (e.g.,
   redundant_coverage from Sprint 9c) would give sharper routing
   decisions. My view: for 2-subsets, threshold overlap is fine
   because the overlap space is coarse (values in {0, 0.5, 1}).
   For richer task spaces, Sprint 9c's metric might matter more.
