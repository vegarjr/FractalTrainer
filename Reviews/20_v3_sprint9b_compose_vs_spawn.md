# v3 Sprint 9b — Compose vs Spawn: the crossover lands at N ≈ 100

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

Sprint 9a showed coverage-compose beats single top-1 by +8.1 pp when
given 300 labeled examples for selection. Sprint 9b asks the
*deployment* question: given a budget of N labeled examples for a new
task, which path is better — compose them into a selection subset,
or use them to train a fresh expert?

This is the triage policy a Mixture-of-Fractals would need in
production. If compose wins below some budget N*, the growth decision
becomes: spawn if N ≥ N*, compose otherwise.

## Design

For each of 27 binary MNIST tasks, leave-one-task-out. For each
budget `N ∈ {50, 100, 300, 1000, 5000}`:

- **Compose path:** N examples from MNIST train become the selection
  subset. Pre-filter top-10 nearest by signature from the 78-entry
  registry. Greedy-pick K=3 that maximize ensemble accuracy on the
  N-example subset. Evaluate the K=3 ensemble on a disjoint
  1000-example test set.
- **Spawn path:** Train a fresh MLP (784→64→32→10) from scratch on
  the same N labeled examples, 500 Adam steps, lr=0.01, equal
  hparams to Sprint 4/7 spawn. Evaluate on the same test set.

Both methods get the same budget and are evaluated on the same
disjoint held-out set, making the per-task compose-vs-spawn delta
directly comparable.

**Cost:** 27 × 5 = 135 spawn trainings @ 5.5–7s = 21 min total wall
clock. Compose is effectively free per query (candidate probs
cached once per task).

## Results

### Per-budget means (across 27 tasks)

| N    | Compose | Spawn | Δ(spawn−compose) | Compose wins | Verdict       |
|-----:|--------:|------:|-----------------:|-------------:|---------------|
|  50  |  0.8190 | 0.7883 | **−0.0307**      | **16/27**    | **COMPOSE WINS** |
| 100  |  0.8190 | 0.8301 | +0.0111          | 11/27        | spawn wins    |
| 300  |  0.8198 | 0.8879 | +0.0681          | 6/27         | spawn wins    |
| 1000 |  0.8244 | 0.9292 | +0.1048          | 3/27         | spawn wins    |
| 5000 |  0.8151 | 0.9443 | +0.1291          | 1/27         | spawn wins    |

For reference, the fixed top-1 baseline across all budgets: **0.7452**.

### Per-task crossover (smallest N where spawn first beats compose)

| Crossover N | # tasks | Tasks                                                              |
|------------:|--------:|--------------------------------------------------------------------|
|          50 | 11      | fibonacci, high_vs_low, middle_456, ones_vs_teens, parity, subset_014567, subset_025678, subset_0379, subset_1256, subset_267, triangular |
|         100 | 7       | primes_vs_rest, subset_01458, subset_123679, subset_123689, subset_13678, subset_3457 (+1) |
|         300 | 6       | subset_01789, subset_02349, subset_0489, subset_123468, subset_2349 (+1) |
|        1000 | 2       | subset_01589, subset_02459                                         |
|        5000 | 1       | subset_0459                                                        |
|   never(≤5k)| 1       | **subset_459**                                                     |

**Median crossover: N = 100.**

### Two surprising findings

1. **Compose is saturated around ≈ 0.82 for *any* N from 50 upward.**
   The ensemble accuracy barely moves between N=50 (0.819) and N=5000
   (0.815). Coverage selection extracts essentially all the signal
   it can from ~50 examples — larger selection subsets don't help.
   The ceiling comes from the candidate pool's best 3-combination,
   not from selection-subset size.

2. **Spawn overfits far less than I expected at small N.**
   At N=50, spawn hits 0.79 — only 6 pp behind compose's 0.82.
   500 Adam-at-lr=0.01 steps on a 53k-parameter MLP with 50
   well-separated binary labels converges to a reasonable model,
   not to a catastrophically-overfit one. Binary MNIST-subset tasks
   are just *easy*. This is the main reason the crossover is at
   N=100 rather than at N=1000 as I predicted.

### The single "never crosses" task: subset_459 ({4, 5, 9})

Spawn at N=5000 hits 0.931 but coverage hits 0.959 — still higher.
Registry-composition reaches near-oracle on this task, and no amount
of from-scratch training (within 500 steps / 5000 examples) matches
it. The registry's candidate pool for `subset_459` evidently
contained a K=3 combination that captures the task almost perfectly.

This is one instance but it's worth noting: some tasks have
"registry-native" structure where composition genuinely outperforms
training. It's the minority case, but it exists.

## Verdict: the deployment triage policy

Given a new task with N labeled examples:

```
if N < 100:    compose  (coverage-greedy K=3 from top-10 pool)
else:          spawn    (train 500-step MLP on the N labels)
```

This is the data-efficient growth policy. The crossover is sharp
(compose 16/27 at N=50, spawn 16/27 at N=100) and close to the
smallest budget tested — **compose's useful window is very narrow.**

## What this means for the vision

Two legs of the Mixture-of-Fractals story now have crisp boundaries:

1. **Growth (spawn) is surprisingly data-efficient.** Sprint 4's
   ~5000-example spawn recipe was overkill. 100 examples is enough
   to beat composition; 300 examples reaches 0.89; 1000 reaches
   0.93; 5000 reaches 0.94 (oracle 0.96). The dominant path for
   new tasks is: if you have ≥100 labels, spawn.
2. **Compose has a niche, not a mandate.** Sprint 9a's big +8.1 pp
   lift was real, but it was measured against top-1 and nearest-K3.
   Against actually-training-a-new-expert, compose only wins when
   you have fewer than ~100 labels. The "fractals help each other"
   claim is true but the window where it's the *best* choice is
   narrow. For most real deployments, spawn is better.

The refined vision: **a Mixture of Label-Function Fractals where
growth (spawn) is the default mechanism, composition is a
label-scarce fallback, and match is free when available.** The
registry's main job is routing + growing; composition is a rare
optimization, not a central feature.

## Alternative reading: compose is a feature of the *cold-start* regime

At N < 100 labels, training a useful new model is genuinely hard —
any training recipe will overfit. Compose's win at N=50 isn't a
novel mechanism as much as a principled way to exploit existing
experts when you literally can't train a new one.

If you reframe it that way, compose isn't an alternative to spawn —
it's a bridge for the budget range where spawn is
data-insufficient. Once spawn is feasible, compose's role diminishes.
This is consistent with how ensembles are used in practice: small-
data regimes, or when you can't retrain.

## Limits

- **Single seed for spawn.** Each spawn used seed=2024. Variance
  estimates would need 3+ seeds per (task, budget) cell → 405
  trainings instead of 135. Deferred. The headline crossover at
  N=100 is likely robust to this because the effect size at larger
  budgets is so clear (spawn beats compose by 10+ pp), but the
  N=50 edge case might be noisier than reported.
- **Spawn training recipe is fixed.** 500 steps, Adam, lr=0.01.
  Early-stopping or a budget-adaptive step count would probably
  help compose more at small N (save spawn from overfitting when
  labels are very scarce). A sweep over spawn hparams is a Sprint
  10 question.
- **Compose uses the same K=3 from the same top-10 pool.** Sprint
  9a's K=3 default might be sub-optimal at different N. And K=3
  might be wrong for tasks that need more or fewer complementary
  neighbors. A K sweep is deferred.
- **Task universe is 27 binary MNIST subsets.** As in every
  previous v3 sprint, results live inside a tight sandbox.
  Generalization to 3-class, 10-class, or CIFAR-style tasks is
  unmeasured. Spawn probably scales worse on harder tasks, which
  would push the crossover to larger N and restore more niche for
  compose. This is the main extrapolation caveat.
- **Labels-at-query-time is still an operational cost.** Compose
  needs labeled examples to do its selection; spawn needs them to
  train. Both are "not free". The comparison is fair *given that
  the user has labels to spend*; neither path works without them.

## What ships

- `scripts/run_compose_vs_spawn_budget.py` — the full sweep
  script. Budgets, K, candidate pool size, and spawn step count
  are all CLI args.
- `results/compose_vs_spawn_budget.json` — per-(task, budget) rows
  with compose/spawn/top1 accuracies, greedy-selected ensembles,
  and per-task crossover budgets.
- The triage policy itself (compose if N<100 else spawn) as a
  concrete decision rule — not yet promoted to a method on
  `FractalRegistry`, by the same logic as Sprint 9a: the decide
  API needs to know whether labels are available before it can
  pick the right mechanism, and that API question is deferred.

No new tests this sprint; 152/152 suite still passes.

## Natural follow-ups

- **Promote the full triage policy to `FractalRegistry.decide()`.**
  A new signature: `decide(query_sig, label_budget=0, labeled_data=None)`.
  If `label_budget < 100` and `labeled_data` is provided: coverage
  compose. Else: spawn signal. If no budget: match-or-spawn (legacy).
  This is the concrete API Sprint 10 would ship.
- **10-class / 3-class task sweep.** Predict: spawn's sample
  complexity rises with n_classes, so the crossover moves right.
  For digit_class (10-class), spawn at N=100 would likely be bad;
  compose might dominate up to N=500 or more. A Sprint 10
  candidate.
- **Harder task universe.** CIFAR, Omniglot, or hierarchical
  multi-label tasks where spawn is genuinely hard even at N=1000.
  That's where compose should have the widest niche — testing
  there would clarify whether the Mixture-of-Fractals vision is
  most useful for hard tasks (novel, intuitive) or for
  easy-task-with-no-data regimes (what we measured).
- **The subset_459 outlier.** Why does coverage nearly saturate on
  this task but not others? A mechanistic look at which candidate
  experts get picked for subset_459 (and their Jaccard overlaps
  with the target set) would sharpen the "coverage wins when
  complementary experts exist in the pool" claim.

## Paired ChatGPT review prompts

ChatGPT should validate:

1. Whether **compose saturating at ~0.82 regardless of N** is a
   genuine phenomenon or an artifact of the K=3-from-top-10 design.
   Widening K or increasing the candidate pool size might lift the
   ceiling; a sweep would settle this. If the ceiling is intrinsic
   to the registry's candidate quality rather than the selection
   rule, it has real implications for whether scaling the registry
   helps composition.
2. Whether **median crossover N=100** is too optimistic given
   single-seed spawn. A 3-seed variance estimate on the border
   budgets (N=50, N=100) would tell us whether the crossover is
   sharp or a noisy band. If it's a band spanning [50, 300], the
   triage rule should be "compose if N < 300" to be conservative.
3. Whether the **single outlier (subset_459) that never crossed**
   is a worthwhile direction to chase. If we can characterize what
   makes a task "registry-native" vs "train-from-scratch-friendly",
   the compose-vs-spawn decision could be made more intelligently
   per-task, not just per-budget.
