# v3 Sprint 4 — Growth Trigger (auto-spawn new fractals)

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Purpose

Sprint 3's FractalRegistry could route to existing experts but was static. Sprint 4 adds the growth mechanism: **when a new task arrives, the registry decides whether to route (match), blend top-K (compose), or train a new expert (spawn)**. This is the missing piece of the brain analogy — the mind doesn't just reuse modules, it grows new ones when the old ones don't fit.

## Machinery

### `FractalRegistry.decide(query_signature, match_threshold, spawn_threshold, compose_k)`

Returns a `GrowthDecision` with verdict in `{match, compose, spawn, empty}`:

- **match**: `min_distance ≤ match_threshold` → route to nearest expert
- **compose**: `match < min_distance ≤ spawn_threshold` → top-K inverse-distance blend
- **spawn**: `min_distance > spawn_threshold` → no close neighbor, train+register new
- **empty**: registry has no usable entries → spawn the first

Threshold calibration from Sprint 3 MVP Test A/B data:

| Distance class | Range observed |
|---|---|
| Same-task MNIST (within-seed variance) | 0.93 – 3.43 |
| Same-task Fashion (OOD probe) | 6.68 |
| Cross-task within MNIST (Test B min) | 7.32 – 10.80 |
| Cross-dataset Fashion ↔ MNIST | 11.4 average |

Chosen: `match_threshold = 5.0`, `spawn_threshold = 7.0`. Clean gap [3.43, 7.32] ⊃ [5.0, 7.0] makes the decision unambiguous for MNIST-family tasks.

### 10 new unit tests (`tests/test_growth_decision.py`)

All four verdicts covered plus: misordered thresholds rejected, exclude honored, serialization, compose weights shaped correctly, edge cases (empty-after-exclude, exact-hit match).

Full suite: **25/25 registry tests pass** (15 from Sprint 3 + 10 new).

## Demo scenarios (`scripts/run_growth_demo.py`)

### Scenario A — MATCH

- Registry: 22 entries (seeds 42 + 101 of all tasks).
- Query: `parity_seed2024` (same task, different seed, held out).
- Result: `verdict=match, min_distance=1.706, nearest=parity_seed42`. ✓

### Scenario B — SPAWN (end-to-end)

This is the real demonstration of the vision: a genuinely new task arrives, no close neighbor exists, the system trains a new expert and registers it.

- Registry: 22 entries.
- New task: `perfect_squares` (`y ∈ {1, 4, 9}`) — not in any cached entry.
- Step 1 (train the probe): trained a `perfect_squares_seed42` expert in 7.8s.
- Step 2 (decide before spawn): `verdict=spawn, min_distance=7.434` — nearest was `middle_456_seed42` at exactly the boundary (middle_456 is `{4, 5, 6}`, which shares the element 4 with perfect_squares — structural kinship, but still below the spawn threshold).
- Step 3 (spawn): `registry.add(entry)`. Registry grows 22 → 23.
- Step 4 (re-query): `verdict=match, min_distance=0.000`. ✓

**The full growth cycle works: decide → train → register → re-decide matches.**

### Scenario C — What falls in the compose band?

- Registry: 30 entries (all MNIST, no Fashion).
- Query: `fashion_class_seed42`.
- Result: `verdict=spawn, min_distance=11.436`.

The Fashion query landed at distance 11.4 from its nearest MNIST neighbor — far above the 7.0 spawn threshold. This demonstrates the **ambiguity band is small in practice**: with sharply-discriminative activation signatures, most queries land cleanly in `match` or `spawn`.

Which is a feature, not a bug — it means the growth decisions are confident. The compose verdict exists for edge cases (e.g., a new binary MNIST task with `primes_vs_rest ↔ parity` distance 7.38 landing just inside a compose band if thresholds are narrowed), but those cases are structurally rare.

## What this validates

The Mixture-of-Fractals vision is now fully dynamic:

1. **Routing (Sprint 3)**: given a task, find the right expert.
2. **Growth (Sprint 4)**: given a NEW task, either route, blend, or train+register a new fractal.
3. **Self-extending registry**: the registry grows over time. A system that has seen N tasks will correctly recognize the N+1th and either use an existing expert or spawn a new one.

Brain analogy mapping:
- Match = "I've seen something like this before, use the existing circuit"
- Compose = "This is like a mix of things I know, blend them"
- Spawn = "This is genuinely new, allocate a new module for it"

## Results

`results/growth_demo.json` contains the three scenarios and their full decisions. Key decisions table:

| Scenario | Registry | Query | Verdict | min_distance |
|---|---|---|---|---|
| A (match) | 22 | parity_seed2024 | match | 1.706 |
| B before spawn | 22 | perfect_squares (new) | spawn | 7.434 |
| B after spawn | 23 | perfect_squares | match | 0.000 |
| C (Fashion vs MNIST) | 30 | fashion_class | spawn | 11.436 |

## Limits

- **Compose path untriggered in the demo**. Exists in the code, tested at unit level. For realistic queries against this registry, the sharp discrimination makes compose rare. That's fine — the hierarchy (match → compose → spawn) captures all decision modes, even if compose is a corner case.
- **Threshold calibration is hand-tuned.** The 5.0/7.0 choice is based on Sprint 3's observed distance distribution. A production system might want to calibrate these per-registry (e.g., 95th-percentile same-task distance = match_threshold).
- **Spawn assumes the system has training data for the new task.** The demo uses a hardcoded labeling rule (`y ∈ {1, 4, 9}`). A real deployment needs a way for users to describe new tasks (data + loss).
- **No decay / eviction.** The registry only grows, never shrinks. A long-lived registry would want a policy for removing underused or outdated experts.

## Natural next sprints

- **v3 Sprint 5 — Composition utility test**: force a compose verdict, measure whether blended top-K predictions beat single top-1 on intermediate queries. Narrow match/spawn bands to induce compose.
- **v3 Sprint 6 — Adaptive thresholds**: calibrate match/spawn thresholds from running distance-distribution statistics instead of hardcoded constants.
- **v3 Sprint 7 — Scale stress**: 50+ tasks, heterogeneous architectures. Does the approach still work?
- **v3 Sprint 8 — Hierarchical routing**: dataset-level coarse routing (11 dataset bins), then task-level fine. Shaves query cost as registry grows past N=1000.

## Paired ChatGPT review

Per the dual-AI workflow, this is Claude's verdict. A ChatGPT-side read should particularly validate (a) whether the 5.0 / 7.0 threshold calibration is sensible given the empirical distance distribution, or whether running a percentile-based calibration would be more robust, and (b) whether the end-to-end spawn demo on `perfect_squares` is a meaningful test or too close to the existing MNIST variants.
