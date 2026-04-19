# v2 Sprint 3 — Golden-run trajectory matching (candidate A)

**Date:** 2026-04-19
**Reviewer:** Claude Opus 4.7 (1M context)

## Why

Sprint 2's science experiment (Reviews/05) established that the v1 target `correlation_dim ≈ 1.5 ± 0.3` is not just arbitrary — it's actively **anti-correlated** with good training on MNIST (runs in-band: mean test_acc 0.57; out-of-band: 0.82). The target band happens to land where Adam's divergent-at-high-lr trajectories sit.

The fix **isn't a better scalar value**. A scalar is not the right type of thing to target. Picking any number prospectively — even 0.7, even "the median of top-25% runs" — bakes in the assumption that one real number captures geometric "goodness." It doesn't.

Golden-run matching inverts the epistemology:

- v1: pick a target shape, measure if the current shape matches it.
- v2 Sprint 3: pick a run that demonstrably produced good training, measure if the current shape matches *that run's* shape.

The target becomes **observed, not prescribed**.

## What

A 9-d geometric signature per trajectory:
```
correlation_dim, total_path_length, mean_step_norm, step_norm_std,
dispersion, displacement, tortuosity, mean_curvature, recurrence_rate
```

Distance between two signatures: z-normalized Euclidean on the per-feature differences, where the scale per feature is `|golden_value| + ε`. Self-match = 0; unit-magnitude means "off by about one scale-unit per feature on average."

## Code delivered

| Path | Role | LOC | Status |
|---|---|---|---|
| `src/fractaltrainer/target/golden_run.py` | `GoldenRun` dataclass, `build_signature_from_report`, `golden_run_distance`, `golden_run_per_feature_deltas` | 170 | New |
| `src/fractaltrainer/target/target_shape.py` | Added `method="golden_run_match"` + `golden_run_path` field + validator | +12 | Edit |
| `src/fractaltrainer/target/divergence.py` | Unified `divergence_score(current, target)` — dispatches on `target.method`. Accepts scalar *or* signature dict. `within_band` now uses score-threshold semantics with a floating-point epsilon | +40 / -20 | Edit |
| `scripts/record_golden_run.py` | CLI: given an existing trajectory .npy + test_acc, write `golden_runs/<name>.json` | 90 | New |
| `scripts/run_comparison.py` | Dispatches on target.method; emits per-feature deltas in the report when using golden_run_match | +35 | Edit |
| `src/fractaltrainer/repair/repair_loop.py` | New `_current_measurement()` helper; packs signature dict for golden_run_match, scalar dim for v1 methods. Within-band check now uses `divergence_after <= 1.0` uniformly | +15 | Edit |
| `src/fractaltrainer/repair/context.py` | Context now includes `golden_run` block from the comparator report when present | +2 | Edit |
| `src/fractaltrainer/repair/prompt_builder.py` | Golden-run-aware target section; adds per-feature z-score delta block to the prompt | +20 | Edit |
| `configs/target_shape_golden.yaml` | Example config using golden-run matching | 14 | New |
| `golden_runs/sgd_lr0p1_seed101.json` | First golden reference (from Sprint 2 top-test_acc run) | — | New |
| `tests/test_golden_run.py` | 18 tests: signature shape, save/load, distance metrics, NaN handling, target YAML loading, divergence dispatch, within_band | — | New |

**102/102 tests pass.** (65 v1 + 19 v2 Sprint 1 + 18 v2 Sprint 3 = 102. One v1 edge-case test was updated for floating-point tolerance in `within_band`.)

## Acceptance gate — does the metric discriminate good from bad training?

Using the Sprint 2 science-sweep data (no new compute needed). Golden run chosen: the top-test_acc run from the 48-run sweep:

- **Golden:** SGD, lr=0.1, wd=0.0, seed=101. Test accuracy = **0.944**. Signature recorded to `golden_runs/sgd_lr0p1_seed101.json`.

Comparing other runs' signatures to this golden:

| Run | hparams | Test acc | Scalar dim | Golden-distance |
|---|---|---|---|---|
| **Golden itself** | lr=0.1 sgd seed=101 | 0.944 | 0.744 | **0.000** ✓ |
| Other healthy SGD | lr=0.1 sgd seed=42 | 0.920 | 0.699 | **0.552** (in band) |
| Mid-acc Adam | lr=0.01 adam seed=42 | 0.898 | 0.779 | **11.45** |
| Divergent Adam | lr=0.1 adam seed=42 | 0.235 | 0.326 | **91.45** |

**Key observation — contrast with scalar-dim:**

The four runs' scalar correlation dims lie in `[0.326, 0.779]` — a tight range. V1's scalar target **cannot** discriminate these runs. The 9-d signature **can**. Per-feature deltas on the divergent Adam vs golden:

```
total_path_length  : +23.0 z   (training moved 331 units vs golden's 14)
mean_step_norm     : +23.0 z   (each step 24× larger — gradient explosion)
step_norm_std      : +80.1 z   (step sizes all over the place)
dispersion         : +16.9 z   (points spread far from trajectory centroid)
displacement       : +24.4 z   (end state far from start — large, volatile)
correlation_dim    :  -0.56 z   (small)
tortuosity         :  -0.06 z   (small)
mean_curvature     :  -0.10 z   (small)
recurrence_rate    :   0.00 z   (identical — both zero)
```

The divergence signal is dominated by **step-size statistics and total movement**, not by correlation dim. That's exactly the right signature for a training run that's exploding — and it's **invisible** in the scalar-dim-only view because dim can be small for both healthy-converging trajectories (SGD, short steady path) and exploding-early-then-flat trajectories (Adam-lr-0.1, huge early steps that saturate).

## What this changes

- The v1 system's architecture (observer → geometry → target → repair) is unchanged. Same three gates (scope, schema, outcome).
- The target plane has a new mode (`golden_run_match`) that's a drop-in replacement for the scalar methods — same scoring contract (`score <= 1.0` = in band; `> hysteresis` = intervene), same interface to the repair loop.
- The LLM gets richer information: per-feature z-score deltas instead of a single dim number. This should lead to more interpretable patches (the LLM can reason about which features are off, not just "move dim toward 1.5").

## What is deliberately NOT tested in this sprint

- **End-to-end closed-loop with golden-run matching + real Claude LLM.** That would run ~60–90s per iteration × 2–3 iterations = ~3–5 minutes of compute and one LLM call per iteration. It's achievable but burns Claude quota, which is already a constraint in this session (user hit "out of extra usage, resets 11 pm"). Deferred as v2 Sprint 4 or a post-quota-reset verification step.
- **Multi-golden ensemble** (record N golden runs, use per-feature stds for normalization). Current single-golden uses `|value| + ε` as a unit-scale, which is reasonable but less principled. Ensemble would be a v2 Sprint 4 addition.
- **Across-domain transfer**: golden from MNIST-SGD, apply to CIFAR training, etc. Out of scope.

## Recommendation for next steps

**If you want to prove the closed loop works with golden-run matching end-to-end:**

```bash
# 1. Copy configs/target_shape_golden.yaml as the active target:
cp configs/target_shape_golden.yaml configs/target_shape.yaml

# 2. Seed divergent hparams:
# Set learning_rate: 0.1, optimizer: adam, weight_decay: 0.0 in configs/hparams.yaml
# (these produced test_acc ≈ 0.24 in the science sweep)

# 3. Run the closed loop:
PYTHONPATH=src <python> scripts/run_closed_loop.py \
  --llm cli --max-iters 3 --verbose \
  --python-bin <python>
```

Expected: the loop starts with divergence >> 10, after one or two patches it should drop toward 1.0. If the LLM proposes moving from Adam-lr=0.1 to something like SGD-lr=0.1, the distance should drop dramatically (it's now trying to match the golden shape, and the golden itself is SGD-lr=0.1).

That's the natural Sprint 4 acceptance gate. It requires compute + quota — not run in this session.

## Compared to candidates B and C

The Sprint 2 finding made it clear that B and C should not proceed before A was in place. Now A is in place:

- **B (training-code rewrites):** Still has the "widen rewrite scope" + "prompt gets more complex" concerns independently, but those are now grounded in a working golden-run target — the LLM would be proposing code changes aimed at matching a known-good shape, not at hitting an arbitrary dim. Viable next step. Risk: model/loss rewrites are much harder to validate than YAML patches, and the schema gate doesn't protect against bad Python.

- **C (meta-controller recursion):** Still blocked on accumulating enough meta-trajectory data to meaningfully compute correlation dim of the repair loop's own output. v2 Sprint 1 showed the machinery but explicitly flagged N=2 as too small. v2 Sprint 4 could be "run golden-run-guided closed loop N times across a grid of starting hparams, record the meta-trajectory across all of them, then check if the meta-trajectory itself is fractal." That would be the true recursive-self-similarity experiment.

## Decision

**v2 Sprint 3 gate: ACCEPTED.** The golden-run-matching machinery is implemented, unit-tested (18 tests), and validated on real science-sweep data — it discriminates healthy from divergent training where scalar-dim could not.

Candidate A is demonstrably viable as a replacement for the scalar-dim target exposed by Sprint 2 as empirically wrong.

## Paired ChatGPT review

Per the dual-AI workflow, this is Claude's verdict. Given the user's explicit request to compare candidates, an independent ChatGPT read on whether golden-run matching is actually better than scalar-dim (or whether some third approach, e.g., learning an ensemble of top-K golden runs and targeting the convex hull of their signatures, would be better still) is the natural parallel review.
