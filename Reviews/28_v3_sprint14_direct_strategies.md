# v3 Sprint 14 — Direct LLM strategies as experts: refined negative

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

Sprint 13 (Review 27) showed behavior-cloning LLM strategies into MLPs
destroys their generalization: cloned `greedy_food` dropped from 19.13
food/game (at source) to 0.42 food/game (as MLP). The diagnosis was
clear: MLPs memorize (state, action) pairs; the underlying algorithm
doesn't survive the compression.

Sprint 14 tests the obvious fix: **skip the MLP entirely**. Use the
LLM-generated Python functions directly as registry entries. Compute
signatures from their action distributions on a probe batch. Majority-
vote among K strategies for ensemble play.

## Hypothesis

If Sprint 13's failure was generalization loss in the MLP, then:

1. Direct LLM strategies should achieve at-source quality at test time.
2. Ensembles should beat the best single strategy (Sprint 9a
   translation), because the strategies' errors are genuinely
   structurally diverse rather than chaotic.

## Result: (1) is true, (2) is false.

### Direct-strategy single evaluation (50 held-out games, seed 5000+)

| Strategy        | Cloned-MLP (Sprint 13) | Direct (Sprint 14) | Recovery |
|-----------------|-----------------------:|-------------------:|:--------:|
| greedy_food     | 0.42 food              | **16.74 food**      | **40×** |
| bfs_safe        | 0.12                   | 0.66               | 5.5×    |
| wall_hugger     | 0.26                   | 0.48               | 1.8×    |
| center_stayer   | 0.08                   | 0.20               | 2.5×    |
| survival_first  | 0.04                   | 0.04               | —       |

**Direct-strategy use recovers nearly all of the source LLM's
performance.** `greedy_food` now scores 16.74 food per game (vs
0.42 when cloned into an MLP), very close to the 19.13 it hit during
demo collection. Sprint 13's diagnosis is confirmed.

Survival:
- **greedy_food: 131.82 steps** (aggressive; dies eventually)
- **wall_hugger: 294.26 steps** (max-cap hugger; ~max of 300)
- the other three: 4–10 steps (buggy strategies, as before)

### Ensemble evaluation (same 50 games, majority-vote)

| Condition                                 | Survival | Score |
|-------------------------------------------|---------:|------:|
| Best single — wall_hugger                 | **294.26** | 0.48 |
| Best single by score — greedy_food        | 131.82   | **16.74** |
| all_K (5) majority vote                   | 166.52   | 2.84  |
| Random-3 (bfs + center + survival)        | 7.60     | 0.30  |
| Nearest-3 to centroid (wall + greedy + surv) | 13.40 | 0.52  |
| **Coverage-greedy-3 (greedy + wall + center)** | **240.12** | **8.40** |

**No ensemble Pareto-dominates any single strategy on its own metric.**

Looking at survival (max at 300): wall_hugger alone hits 294. Best
ensemble (coverage-greedy-3) hits 240 — a 54-step drop. On score:
greedy_food hits 16.74. Best ensemble hits 8.40 — half the food.

The coverage-greedy-3 ensemble is a **balanced middle**: it survives
longer than greedy_food (240 vs 132) and scores more than wall_hugger
(8.40 vs 0.48), but dominates neither.

### Verdict: BEST SINGLE WINS (on its respective metric)

Δ(all-K ensemble − best single survival) = **−127.74**.
Δ(coverage-greedy-3 − best single survival) = **−54.14**.

## Why majority-vote fails on Snake

Sprint 9a (MNIST classification) showed ensemble > top-1 by +8.1 pp.
Sprint 14 (Snake sequential decisions) shows ensemble < best single.
The mechanism difference is informative:

- **Classification with softmax blending:** each expert outputs a
  distribution over labels. When experts AGREE on correct answers
  and DISAGREE on different errors, averaging distributions sharpens
  the correct answer. Ensemble is a Bayesian-ish marginalization.
- **Sequential decisions with majority-vote over actions:** each
  step, if experts vote `{up, up, left}`, you pick `up`. This
  breaks the internal coherence of ANY expert's strategy.
  wall_hugger's "always move along the wall" is a multi-step
  consistent behavior; forcing it to pick `up` at some step and
  `right` at the next (because greedy_food wanted food) creates a
  trajectory neither expert would have chosen alone — and is often
  worse than either alone.

Put differently: **in classification, different experts can each
contribute a piece of evidence for the same answer. In sequential
control, different experts often want different answers, and the
"compromise" action is a coherent plan of neither.**

## The real finding

Sprint 13 showed: **MLPs don't preserve algorithmic generalization.**
Sprint 14 adds: **majority-vote compose doesn't work on sequential
decision tasks, even when individual experts DO generalize.**

The Mixture-of-Fractals architecture has two conditions for the
compose path to Pareto-improve:

1. **Experts must generalize** (Sprint 13 reified this).
2. **The compose operator must respect the task's structure**.
   Discrete classification → softmax averaging. Sequential control
   → this is the open problem.

What would likely work on Snake:

- **Contextual (per-state) routing.** At each game step, pick which
  single expert to follow based on the current state. E.g., "if food
  is within Manhattan distance 3, use greedy_food; otherwise use
  wall_hugger." This is a **hierarchical policy**: a meta-router
  over a set of base policies, not an ensemble over their joint
  actions. This would retain the coherence of the chosen expert at
  each step.

- **Confidence-weighted voting.** If `greedy_food` is very confident
  (close food, clear path) and `wall_hugger` is ambivalent (no wall
  nearby), weight the former more. Strategies in Sprint 14 have
  binary confidences (they return a single action); a softened
  policy (action distribution) would enable this.

- **Sequential composition.** Commit to one expert for a "phase" of
  the game (e.g., opening: wall_hugger for N steps to set up; then
  greedy_food once the snake has a stable shape). This is how humans
  play complex games — regime switching.

These are all meaningful research directions that the current v3
primitives (match/compose/spawn) don't directly support — they'd need
a new primitive.

## Coverage-greedy-3 is the interesting intermediate

The coverage-greedy-3 ensemble (selected from a held-out mini-eval of
10 games at seeds 4000–4009) picked `{greedy_food, wall_hugger,
center_stayer}`. On the 50-game eval it hit 240 survival + 8.40
score. This is:

- Worse than wall_hugger alone on survival (240 vs 294).
- Worse than greedy_food alone on score (8.40 vs 16.74).
- Better on survival × score product than either (240×8.40 = 2016
  vs 294×0.48 = 141 and 132×17 = 2243).

**So if the metric is "survival × score" (approximating "useful play"),
coverage-greedy-3 nearly matches greedy_food at 2016 vs 2243 — and
does so with much higher survival.** That's a Pareto-intermediate
point: a safer version of the aggressive strategy.

For a risk-averse agent (game with penalty for early death),
coverage-greedy-3 might be preferred. For a reward-maximizer,
greedy_food is best. The choice depends on the objective; neither
strategy is universally dominant.

## What ships

- `scripts/run_snake_sprint14.py` — direct-strategy evaluation driver.
  Reuses Sprint 13's cached `raw_code` from `snake_demos.npz`, so no
  additional LLM calls required.
- `results/snake_sprint14.json` — per-strategy, per-ensemble metrics,
  signature distances, coverage-selection trace.

176/176 test suite still passes (no new tests; Sprint 14 reuses
existing infrastructure).

## The v3 arc reading now

Sprints 9–12 built up to "compose works on classification, enabled by
coverage-greedy selection." Sprints 13–14 showed "compose doesn't
naïvely transfer to sequential control: either the experts don't
generalize (13) or the operator doesn't respect task structure (14)."

That's a real scope-boundary finding for the Mixture-of-Fractals
vision: it's **a classification-regime architecture**, and extending
to sequential control requires either a per-step router (hierarchical
policy) or a confidence-weighted softening that current LLM-generated
strategies don't naturally provide.

Snake isn't the right demonstration domain for the current v3
primitives. The architecture's natural home remains classification-
shaped tasks with label-function variation — exactly what Sprints 3–12
proved it handles well.

## Limits

- **n=5 strategies, 50 eval games.** Conclusions are directional; a
  20-strategy pool would test more ensemble compositions.
- **No hyperparameter tuning on the LLM prompts.** Other strategy
  descriptions might produce more synergistic experts (e.g., explicit
  "this strategy complements X" prompting).
- **Coverage selection used survival as the selection metric.** Using
  score instead would probably have picked `{greedy_food, ?, ?}`
  ensembles — different Pareto point.
- **Majority-vote only.** Didn't explore weighted voting or contextual
  per-step routing. Those are the next-sprint questions if the user
  wants to push on Snake.

## Natural follow-ups

1. **Sprint 15 — Per-state contextual routing.** Meta-policy that
   picks ONE expert per state. Train on labeled (state, best-expert)
   via behavior cloning or RL. Prediction: Pareto-dominates both
   wall_hugger (survival) and greedy_food (score) because it can be
   aggressive when safe and defensive when crowded.
2. **Sprint 16 — Pivot away from Snake.** Find a demonstration domain
   where compose naturally works (multi-label image classification,
   tabular regression with diverse feature-importance per expert).
   The v3 architecture deserves a showcase where its strengths
   align with the task structure.
3. **Write-up.** After 14 sprints and an explicit scope-boundary
   finding, the v3 arc has a coherent story worth documenting:
   "Mixture of Label-Function Fractals for classification-regime
   tasks — architecturally self-organizing, empirically validated
   at 100% same-task retrieval, coverage-compose beats top-1 by 8
   pp when experts generalize, honest limits on sequential control."

## Paired ChatGPT review prompts

ChatGPT should validate:

1. Whether **majority-vote failure on sequential tasks** is a
   well-known phenomenon (I suspect yes — this is why hierarchical
   RL uses options and why behavior fusion in robotics uses smoothed
   policies). If novel, Sprint 15 should explore; if known, point
   to the literature and pick a different problem.
2. Whether the **"classification-regime architecture" scope statement**
   is too narrow. Arguably the Mixture-of-Fractals works for any
   problem where (a) experts' outputs can be meaningfully blended,
   (b) blending produces a coherent aggregate decision. Tabular
   regression qualifies; NLP classification qualifies; multi-label
   qualifies. Sequential control doesn't. Is there a cleaner framing?
3. Whether **Sprint 15 (per-state contextual routing) is the right
   next step** or whether the cleaner finding is to freeze v3's scope
   and pursue a classification-regime showcase.
