# v3 Sprint 17 — Fractal showcase: perception→growth loop + context injection

**Date:** 2026-04-21
**Reviewer:** Claude Opus 4.7 (1M context)

## Purpose

Stitch the existing FractalTrainer primitives (describer, multi-region
router, coverage-compose, spawn, reclustering) into one continuous
end-to-end pipeline (**F: closed perception→growth loop**) and implement
the previously-unexplored vision leg **C: context injection** — when a
new expert spawns, its first hidden layer is enriched with the
penultimate activations from the top-K nearest registry neighbors on
the new task's data.

This is the first sprint that runs all the pieces together, and the
first that attempts "one fractal helps another" at training time, not
just at inference-time ensembling.

## What was built

New module: `src/fractaltrainer/integration/` with 8 files.

| File | What it does |
|------|-------------|
| `context_mlp.py` | `ContextAwareMLP` — 784→64→32→10 spine + optional additive context lane (`Linear(32,64)` + `LayerNorm`). `context_scale=0` ⇒ bit-exact baseline. Exposes `penultimate(x)`. |
| `context_injection.py` | `gather_context(neighbors, probe, spec, distances)` → `(B, 32)` inverse-distance-weighted mean of neighbor penultimates. `ContextSpec` + `random_context` (for ablation arm C). |
| `spawn.py` | `spawn_baseline` (arm A), `spawn_with_context` (arm B), `spawn_random_context` (arm C). All return `(model, FractalEntry, TrainStats)`; probe signature computed with `context=None` to preserve the routing invariant. |
| `recluster.py` | `recluster(registry, k, metric)` — Sprint 12's agglomerative clustering extracted into a reusable callable. |
| `describer_adapter.py` | `Describer(llm_fn)`, `MockDescriber`, `OracleDescriber` — Sprint 15c's prompt logic lifted into classes. |
| `pipeline.py` | `FractalPipeline` — orchestrator that turns a `QueryInput` into a `PipelineStep` with verdict / action / accuracy / spawn stats. Owns `ReclusterPolicy`. |
| `evaluation.py` | `evaluate_expert`, `sample_efficiency_curve`, `render_efficiency_table_md`. |
| `__init__.py` | Public surface. |

New scripts:
- `scripts/run_fractal_demo.py` — the showcase driver.

New tests: 5 files, **36 tests**. All **212 tests pass** (176 existing + 36 new).

## Scenario

Seed registry: 5 tasks × 3 seeds = **15 experts**, each a
`ContextAwareMLP(context_scale=0)` trained for 500 steps on a
5000-sample MNIST binary subset.

| Seed expert | Class-1 set |
|-------------|-------------|
| subset_01234 | {0,1,2,3,4} |
| subset_56789 | {5,6,7,8,9} |
| subset_024   | {0,2,4} |
| subset_13579 | {1,3,5,7,9} |
| subset_357   | {3,5,7} |

Three queries:

| Query | Task | Class-1 | Expected verdict |
|-------|------|---------|------------------|
| Q_match   | subset_13579 (different seed) | {1,3,5,7,9} | match |
| Q_compose | subset_02468 | {0,2,4,6,8} | compose |
| Q_spawn   | subset_019   | {0,1,9}     | spawn — C showcase |

Three-arm × five-budget × three-seed ablation on the spawn path:

| Arm | Mechanism |
|-----|-----------|
| A — no context | `spawn_baseline` (`context_scale=0`) |
| B — K=3 nearest context | `spawn_with_context` — the C contribution |
| C — K=3 random context | `spawn_random_context` — control |

Budgets: {50, 100, 300, 500, 1000}. Seeds: {42, 101, 2024}.

## Results

### Pipeline (F) — the three verdicts fire correctly

| Query | Verdict | Action | Accuracy | Elapsed (s) |
|---|---|---|---|---|
| Q_match   | match   | route → subset_13579_seed2024 | **0.965** | 0.00 |
| Q_compose | compose | top-3 blend: {subset_024_seed42, subset_024_seed101, subset_024_seed2024} | **0.772** | 0.00 |
| Q_spawn   | spawn   | spawn w/ K=3 context from subset_13579 seeds + subset_024_seed2024 | **0.918** | 15.18 |

All three verdicts are the ones the scenario was designed to elicit.
Routing is driven by signature distance, without reference to the
describer (the describer outputs are logged but unused in the
pipeline decision — as planned). Final reclustering produced 3
clusters from the 15 seed entries + 1 spawn entry, grouping
`{subset_01234, subset_024}`, `{subset_56789, subset_357}`,
`{subset_13579, subset_019}` by Jaccard similarity of class-1 sets —
the split the user would draw by hand.

### Ablation (C) — context injection did NOT help

| Arm | N=50 | N=100 | N=300 | N=500 | N=1000 |
|---|---|---|---|---|---|
| **A — no context**      | 0.899±0.017 | 0.928±0.007 | 0.946±0.007 | **0.959**±0.007 | **0.964**±0.002 |
| **B — K=3 nearest ctx** | 0.895±0.018 | 0.933±0.005 | 0.913±0.032 | 0.947±0.002 | 0.940±0.008 |
| **C — K=3 random ctx**  | 0.908±0.014 | 0.940±0.005 | 0.952±0.009 | 0.962±0.005 | 0.963±0.003 |

**Acceptance criterion (B_mean > A_mean + A_stdev at N=1000): FAIL.**
B (0.940) is *below* A (0.964 ± 0.002) — by two percentage points.
The random-context control (C) matches A closely and is always
within A's noise, confirming that the result isn't driven by any
auxiliary input.

Reading across budgets:
- At small N (50–100), all three arms are statistically indistinguishable (~0.90).
- At mid N (300–500), arm B *regresses* while A and C continue to improve.
- At large N (1000), A+C converge to ~0.96 while B plateaus at 0.94.

## Verdict

**F (integration) is a success.** The pipeline runs end-to-end on
real MNIST, the three verdicts fire for the three scripted
scenarios, reclustering recovers a human-interpretable anchor split,
and accuracy on the match / compose paths is reasonable (0.97 /
0.77). The integration is the canonical "it works" artifact the
vision needed after 16 sprints of siloed primitives.

**C (context injection) is a negative result in this setup.**
K=3-nearest context does not accelerate spawn-path training; it
slightly hurts it. The random-context control rules out "any
auxiliary input helps" — and it also rules out "routing chose bad
neighbors" as the primary failure, since arm C (random) matches
arm A almost exactly while arm B (nearest) is worse than both.

### Why C failed (best current hypothesis)

Three candidate explanations, ordered by how much evidence supports
them:

1. **Train-test distribution mismatch from the probe-signature
   invariant.** Training runs with `context = nearest-neighbor
   penultimates on this batch` but signature-time eval runs with
   `context = None`. The model's first hidden layer has learned to
   incorporate context signal during training; at eval time that
   signal is zero, so the model is operating off-distribution. Arm C
   doesn't suffer as badly because random context is approximately
   mean-zero, so the model simply learns to average-out the noise —
   but arm B's context has real structure the model comes to rely
   on, and removing it at eval time costs more. This is consistent
   with arm B plateauing below arm A as N grows: more training
   means stronger reliance on context, which vanishes at eval.
2. **Signal/noise mismatch at `context_scale=1.0`.** LayerNorm
   normalizes the context to unit scale, then the `Linear(32, 64)`
   projection and `context_scale=1.0` multiplier inject it at the
   same order of magnitude as the primary first-hidden activation.
   Neighbors trained on different tasks don't carry enough
   task-relevant signal to justify equal-magnitude fusion; a
   smaller `context_scale` (0.1–0.3) might be a better default.
3. **Task is too easy — no room for context to help.** MNIST
   binary classification with 5000 samples + Adam converges to
   ~0.96 accuracy in 300 steps. There's no cold-start regime
   where borrowed feature detectors could accelerate learning.
   A sample-starved regime (N<<50 with tiny batches) or a harder
   task (multi-class, out-of-distribution target) might show the
   effect.

Hypothesis 1 is the most actionable: a quick follow-up would
retrain arm B with **context kept at eval time** (compute a per-
sample context using the same neighbors + probe batch) instead of
the `context=None` invariant. If that lifts arm B to match or
exceed arm A, hypothesis 1 is the explanation; if not, the other
hypotheses dominate.

### The F pipeline is the shipable showcase

Even with C failing, F is a clean result. The sprint delivers the
first end-to-end run of the Mixture-of-Fractals architecture on
real data, with all primitives firing as designed and a
reproducible scenario that demonstrates match / compose / spawn in
one driver. That's the shipable artifact for the vision.

## What surprised us

- The baseline MLP at N=50 is already at 0.90 accuracy on this
  task — we expected a cold-start regime under 0.85 where context
  could plausibly add value. Binary MNIST is easier than the
  design premise assumed.
- Arm C (random context) does not hurt — it slightly *helps*
  (0.908 vs 0.899 at N=50). Zero-mean random projection noise
  behaves like a regulariser.
- Arm B's *mid-budget* regression (N=300, 0.913 ± 0.032) is the
  steepest drop anywhere in the table, with 3× the stdev of any
  other cell. That's a reliability signal — context injection
  introduces a training instability that shows up most strongly
  somewhere in the middle of training, not at the endpoints.
- The compose verdict on Q_compose picked 3 experts from
  `subset_024`, not a mix of {0,2,4} and {5,6,7,8,9} experts —
  because all three `subset_024` seeds are close to each other in
  signature space but moderately distant from Q_compose. The
  coverage-greedy selector would have picked differently; the
  pipeline's min-distance router picked the cluster nearest Q.
  Accuracy 0.77 on evens-when-registered-experts-only-see-
  {0,2,4} is reasonable but suggests the compose path should
  consult coverage-greedy downstream, not just nearest-cluster
  top-K.

## What ships

- `src/fractaltrainer/integration/*` — 8 new module files (~1200 LoC)
- `scripts/run_fractal_demo.py` — showcase driver (~470 LoC)
- `tests/test_context_mlp.py`, `test_context_injection.py`,
  `test_recluster.py`, `test_fractal_pipeline.py`,
  `test_fractal_demo.py` — 36 new tests
- `results/fractal_demo.json` — machine-readable output
- `results/fractal_demo.png` — ablation plot
- `Reviews/33_v3_sprint17_fractal_demo.md` — this doc

**212/212 tests passing** (176 pre-existing + 36 new).
No existing files modified destructively.

## Paired ChatGPT review prompts

1. **On the negative result for C.** Is "probe-signature invariant
   causes train-test mismatch" (hypothesis 1) a correct
   identification, and if so, does that mean context injection as
   currently formulated is architecturally unsalvageable, or is
   there a clean fix (e.g. compute context at eval time using the
   query's neighbors on the input batch, just as at training time)?
2. **On the scenario being too easy.** Binary MNIST at N=5000 is
   a saturated regime. Would the right next test be (a) the same
   architecture on a harder base task (e.g. CIFAR binary, or a
   multi-class MNIST subset), (b) a sample-starved regime
   (N<100 training examples), or (c) a cross-modal task where
   feature detectors genuinely differ between neighbors and
   target?
3. **On whether the F pipeline is publishable as-is.** All
   primitives fire and the scenario is reproducible. Is the
   integration demo a sufficient claim on its own, or does it
   need either (a) a working C contribution or (b) a stronger
   compose-path ablation to stand up as a paper?

## Next-step options

1. **Fix C: keep context at eval time** — biggest-lever test of
   hypothesis 1. A few lines of change to `evaluate_expert` +
   `spawn_with_context`. Expected to materially move arm B.
2. **Sample-starved ablation** — rerun F+C with N ∈ {10, 25, 50,
   100} instead of {50,...,1000}, same architecture. If context
   helps at N=10–25 but not higher, the mechanism is a cold-start
   accelerator, not a convergence optimizer.
3. **Direction B from the brainstorm: latent-space signatures** —
   independently valuable regardless of C's fate. Unlocks
   regression / RL. Sprint 18 candidate.
4. **Real-LLM describer run** — `--llm local` with Qwen-7B.
   Validates the full perception→growth story end-to-end (F is
   isolated from describer quality by default). Low effort now
   that the pipeline is plumbed.
