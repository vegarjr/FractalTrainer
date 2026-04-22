# v3 Sprint 17 — self-spawn fix + Direction E (hierarchical)

**Date:** 2026-04-22
**Reviewer:** Claude Opus 4.7 (1M context)

Two follow-ups that close out the Sprint 17 cluster:

1. **Self-spawn redundancy fix.** Review 39 identified that
   AutoSpawnPolicy's union-of-K-neighbors heuristic duplicates
   existing tasks when compose clusters land on a single
   well-represented seed. One-line suppression added:
   `if union ⊆ nearest.task_labels: return None`.
2. **Direction E — hierarchical fractals.** The registry was named
   "fractal" but is flat. A 2-level view (tasks as branches, seeds as
   leaves) already works because Sprint 8's `HierarchicalRegistry`
   supports any metadata field as the branch key. Demo compares flat
   vs hierarchical routing on 15-expert MNIST registry.

## 1. AutoSpawnPolicy redundancy suppression

### Change

`AutoSpawnPolicy.__init__` gains a `suppress_redundant: bool = True`
parameter. Inside `propose()`:

```python
if self.suppress_redundant and retrieval.entries:
    nearest_labels = set(retrieval.entries[0].metadata["task_labels"])
    if nearest_labels and union_labels <= nearest_labels:
        self._suppressed_redundant_count += 1
        return None
```

Three new unit tests cover the path (redundant→suppressed,
non-redundant→passes, suppress-off→redundant-allowed). All 12
self-spawn tests pass.

### V2 re-run: bridge stream spanning different seeds

Redesigned `COMPOSE_BAND_TASKS` to use labels like `{0,5}`, `{2,8}`,
`{4,9}` — pairing one low-digit with one high-digit so the K=3
nearest neighbors in signature space would come from multiple seed
tasks.

**Result:** 7 of 10 queries routed to **spawn** (min_distance > 7.0)
and only 3 to compose. Threshold (5 compose verdicts) not met → no
proposal fired.

The lesson: these bridge pairs were *too novel* (signature distances
7-9 = spawn band, not compose band 5-7). The suppression fix didn't
get exercised because compose didn't accumulate.

### V3 re-run: compose-band tasks

Replaced the bridge stream with `subset_02468` (evens-all) repeats
plus a few variants. These are known from earlier sprints to land in
the compose band (distance ~6).

**V3 results:** 9 compose + 1 spawn out of 10 queries. Threshold
(5 compose) easily met → `propose()` called.

**Proposal: None (suppressed).** For every compose verdict, the K=3
nearest neighbors were all from a *single* seed task — subset_024
(for evens-heavy queries) or subset_56789 (for one high-mixing
query). Union of K=3 neighbors' labels was always a subset of the
nearest neighbor's own labels → redundancy check fired.

This is the suppression working **as designed**: compose clusters in
this registry indicate query load on existing experts, not coverage
gaps. A genuine bridging task requires compose queries whose
signatures span *different* seed clusters — a narrow band that
didn't materialize on the evens-heavy stream either.

### Honest conclusion on self-spawn

AutoSpawnPolicy with `suppress_redundant=True` is correctly
conservative: it identifies compose clusters, computes their K-
neighbor union, and suppresses proposals that would duplicate an
existing task. On MNIST + 15 seed experts:
- V1 (bridge tasks overlap-heavy with one seed): compose cluster
  on subset_01234 → proposal suppressed (would have been
  redundant).
- V2 (bridge tasks genuinely cross-seed): mostly spawn-verdict,
  threshold not met → no proposal.
- V3 (evens-heavy compose-band tasks): compose on subset_024 →
  suppressed (redundant).

The primitive works. The missing piece is a *query stream* whose
compose-band signatures actually span multiple seed clusters, which
is either (a) rare in practice for this registry structure, or (b)
requires a different stream-generation approach (e.g., adversarial
labels chosen to maximize coverage-gap signatures). Follow-up work.

## 2. Direction E — Hierarchical fractals

### Existing infrastructure

`src/fractaltrainer/registry/hierarchical_registry.py` (Sprint 8)
implements a 2-level registry: one `FractalRegistry` per distinct
value of `metadata[dataset_key]`, coarse-routes by centroid distance,
then flat-routes within the chosen branch. 14 unit tests, not used
by the integration pipeline.

Using `dataset_key="task"` with our 15-expert seed registry
automatically partitions into 5 task-branches × 3 seed-leaves each —
no new code needed.

### Demo

`scripts/run_hierarchical_demo.py` builds a flat registry and a
matching `HierarchicalRegistry(dataset_key="task")`, computes
signatures for 8 test queries, and compares:

1. **Decision consistency**: does hierarchical pick the same entry?
2. **Latency**: microbench over N trials
3. **Distance-op count**: 15 (flat) vs 5+3=8 (hier)

### Results (MNIST, 15 experts, 5 task-branches of 3 seeds each)

**Decision consistency: 8/8 queries agreed.** Both flat and
hierarchical picked the same seed expert for every query. Every
decision sat cleanly in one branch — the hierarchical router never
had to re-route because of a cross-branch close second place.

| Query | Flat pick | Hier pick (branch) | Agree |
|---|---|---|---|
| Q_match_odds        | subset_13579_seed2024 | subset_13579/seed2024 | ✓ |
| Q_match_lows        | subset_01234_seed42   | subset_01234/seed42   | ✓ |
| Q_match_evens_low   | subset_024_seed2024   | subset_024/seed2024   | ✓ |
| Q_match_highs       | subset_56789_seed2024 | subset_56789/seed2024 | ✓ |
| Q_match_midodds     | subset_357_seed42     | subset_357/seed42     | ✓ |
| Q_bridge_lowhigh_05 | subset_357_seed101    | subset_357/seed101    | ✓ |
| Q_bridge_mixodd_357 | subset_357_seed42     | subset_357/seed42     | ✓ |
| Q_bridge_evens_048  | subset_024_seed101    | subset_024/seed101    | ✓ |

**Latency over 100 trials:**

| Mode | ms/query | Distance ops |
|---|---|---|
| Flat                | 0.103 | 15 |
| Hierarchical        | 0.074 | 8 (5 centroids + 3 within-branch) |
| **Speedup**         | **1.39×** | — |

The speedup at N=15 is modest (distance ops drop 15→8 but each
centroid distance is itself an L2 computation). The asymptotic
saving grows with N: at N=100 with 10 task-branches of 10 seeds,
the ratio is 100 vs 10+10=20 → expected 5× speedup.

**Cross-branch protection.** The real value isn't latency at
N=15; it's preventing cross-task routing errors at N≫100 where flat
nearest-neighbor can misroute on signature noise. Hierarchical's
coarse gate commits to a task-branch before the fine search runs,
so a close-but-wrong-task match gets filtered by the centroid
comparison.

### Direction E in the vision

The hierarchical demo confirms the "fractal" naming is meaningful
at depth 2. True self-similarity would require:

- Sub-registries at depth ≥ 3 (e.g., task → style → seed).
- Recursive match/compose/spawn decisions at each level.
- A spawn that can *insert at any depth* (not just as a top-level
  peer), which requires the registry to know when to create a new
  level vs a new leaf.

Those are real architectural extensions — this sprint establishes
the depth-2 base case works and routes correctly. Going deeper is a
next-sprint move.

## 3. What ships

- `src/fractaltrainer/integration/self_spawn.py` — `suppress_redundant`
  parameter and redundancy check
- `tests/test_self_spawn.py` — 3 new tests (12 total)
- `scripts/run_self_spawn_demo.py` — v3 stream with compose-band
  tasks
- `scripts/run_hierarchical_demo.py` — new; flat vs hierarchical
  comparison
- `results/self_spawn_demo_v3.json`, `results/hierarchical_demo.json`
- `Reviews/40_v3_sprint17_self_spawn_fix_hierarchical.md` — this doc

**All tests passing** (self-spawn 12, total 233).

## Paired ChatGPT review prompts

1. **On the self-spawn suppression being "too conservative."** The
   redundancy check fires correctly but ends up suppressing every
   proposal that the demo can generate. Is the primitive actually
   conservative in production (which is desirable) or is the
   redundancy check masking a signature-geometry limitation? A
   stronger test would be a registry where compose clusters
   *genuinely* bridge two tasks — if the fix still suppresses
   those, the heuristic is broken; if it allows them, it's just
   that MNIST + 5 seeds doesn't naturally produce them.
2. **On the hierarchical 1.39× at N=15.** Speedup is real but
   modest. At what N does the asymptotic advantage start to
   matter in practice? And is "correctness protection against
   cross-task misrouting" the more important claim for the vision
   than raw latency?
3. **On true recursive fractals.** The depth-2 hierarchy is a
   fractal in name but not really in structure. Going to depth
   ≥3 needs a spawn policy that decides *where to insert* a new
   expert (new peer, new sub-leaf, or new level). Does the
   signature-distance framework extend to that decision naturally,
   or does it need a new "depth-aware routing" primitive?
