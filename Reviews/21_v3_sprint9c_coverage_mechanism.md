# v3 Sprint 9c — Why coverage wins when it wins: redundant label coverage

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

Sprint 9b found one task (`subset_459`) where coverage-compose
beat spawn even at N=5000 labels. I called this "registry-native"
and promised a mechanistic diagnosis. This sprint provides it.

Two hypotheses to test against each other:

**H1 — Partition:** compose wins when the picked experts PARTITION
the target's class-1 digit set — i.e. pick 1 claims digit A, pick 2
claims digit B, pick 3 claims digit C, and the union covers the
target exactly.

**H2 — Redundant coverage:** compose wins when EVERY target
digit is claimed by multiple picks, giving ensemble voting strong
agreement on positive examples. Picks may overlap heavily among
themselves, as long as they collectively hit every element.

Under H1: low pairwise Jaccard among picks (diverse/orthogonal).
Under H2: high `min_element_cover` (every digit doubly/triply
claimed).

## Method

Reuse Sprint 9a and 9b stored picks — no new training. For each of
the 27 binary MNIST tasks at N=5000, compute:

- `union_completeness` = |target ∩ (S₁ ∪ S₂ ∪ S₃)| / |target|
- `over_coverage`      = |(S₁ ∪ S₂ ∪ S₃) \ target| / |union|
- `pairwise_jaccard`   = mean pairwise Jaccard among picks (diversity)
- `min_element_cover`  = smallest number of picks claiming any target
  digit (partition quality — 0 means at least one target digit is
  unclaimed; ≥2 means every target digit is claimed by multiple picks)
- `jaccard_to_target`  = mean Jaccard(Sᵢ, target)

And correlate each with the **compose − spawn advantage at N=5000**.

## Deep dive — subset_459

Target: `y ∈ {4, 5, 9}`.

Coverage picked:

| Pick | class-1 set | ∩ target | \ target |
|------|-------------|----------|----------|
| subset_0459  | {0, 4, 5, 9} | {4, 5, 9} | {0}    |
| middle_456   | {4, 5, 6}    | {4, 5}    | {6}    |
| subset_2349  | {2, 3, 4, 9} | {4, 9}    | {2, 3} |

**Per-element claim count:**

| target element | # picks claiming it |
|:--------------:|:-------------------:|
| 4              | **3**               |
| 5              | 2                   |
| 9              | 2                   |

Union = {0, 2, 3, 4, 5, 6, 9}. Every target element is claimed by
≥2 picks. `min_element_cover = 2`. And `pairwise_jaccard = 0.300`
— the picks aren't identical, but they're **not** an orthogonal
partition either. This falsifies H1 immediately: subset_459 wins
not because its picks split the target but because they
**redundantly** agree that 4, 5, and 9 are positive.

## Correlation analysis across 27 tasks

Compose-minus-spawn advantage at N=5000 (all values negative except
subset_459's +0.002) vs each coverage metric:

| Metric               | Pearson r | Spearman ρ |
|----------------------|:---------:|:----------:|
| `min_element_cover`  | +0.506    | **+0.516** |
| `jaccard_to_target`  | +0.421    | +0.479     |
| `union_completeness` | +0.273    | +0.377     |
| `over_coverage`      | +0.017    | −0.053     |
| `pairwise_jaccard`   | −0.051    | −0.088     |

**The story is H2, not H1.** `min_element_cover` is the strongest
predictor (ρ=+0.52). `pairwise_jaccard` (which H1 would require to
be strongly negative) is essentially zero — picking diverse experts
does NOT make coverage compose better.

The second-strongest predictor is `jaccard_to_target` (ρ=+0.48):
picks should be individually label-similar to the target, not
orthogonal to it. This also contradicts H1 (which predicts
low-overlap picks).

## Interpretation

Compose wins — when it wins — because **every target-class digit
gets multiple positive votes from the ensemble**, not because the
picks occupy complementary niches. This is classical ensemble
wisdom in disguise: voting redundancy on correct positive labels
beats a single near-correct expert, *provided* every correct label
is covered.

The practical test for "this task is registry-native":

> Given target T and candidate pool C, the ensemble will likely
> beat spawn if there exist ≥3 candidates whose class-1 sets each
> contain ≥60% of T AND whose union covers T completely, with
> every target element claimed by ≥2 of the chosen candidates.

This turns the Sprint 9b crossover rule from:
```
if N < 100:   compose
else:         spawn
```
into a smarter per-task rule:
```
if N < 100:                     compose
elif redundant_coverage(T, C):  compose            # per-task test
else:                           spawn              # default
```

where `redundant_coverage(T, C)` can be evaluated using ONLY the
candidates' label metadata — no model forwards, no labeled data
needed. It's a free, up-front test.

## Why H1 fails

"Partition" would require finding 3 experts whose class-1 sets
*cleanly slice* the target. In practice, the binary tasks in this
registry have class-1 sets of size 2–6 over 10 MNIST digits, so a
true partition of a 3-digit target would need experts each
claiming a single digit — but the smallest existing class-1 sets
are still size 3 (`triangular`={1,3,6}, `fashion_footwear` is in a
different kingdom). The registry doesn't contain granular-enough
experts to partition most targets.

Redundant coverage, by contrast, is abundant: for any 3-digit
target, there are many 3-to-5-digit experts that each share most
of the target. Their ensemble builds confidence through agreement.

## What this means for the vision

The Mixture-of-Fractals "1+1 fractal and 1×1 fractal help each
other" story needs one refinement: **helping each other means
AGREEING on correct positive labels**, not occupying orthogonal
niches. The ensemble value comes from voting, not from
orthogonality.

This also sharpens what "registry density" should mean. A registry
with 10 experts whose class-1 sets are orthogonal (each claiming
one digit) would be bad for compose — you'd always need 3 picks
and the picks would each be distant (low Jaccard to target). A
registry with 30 experts each claiming 3–5 digits heavily
overlapping with common digits is actually better for compose —
more redundancy means more voting confidence.

This is the first time the registry-design question becomes
concrete. Future expert-generation strategies should prioritize
**high-overlap coverage** over **orthogonal coverage**.

## Ripple effects

1. **Coverage greedy selection isn't finding partitions** — it's
   finding redundantly-voting ensembles. The algorithm works as
   advertised but the *reason* it works is different from my
   Sprint 9a intuition.

2. **Sprint 5's null is doubly explained.** Signature-nearest
   neighbors are label-redundant (Sprint 7b's Jaccard result).
   Blending them equal-weighted gives you label-redundant votes
   — which, if the redundant votes are all on the target, would
   AGREE strongly. Why did Sprint 5 still null? Because the top-3
   nearest by signature aren't necessarily the top-3 by
   `min_element_cover`. Sprint 9a/9b's greedy selection optimizes
   accuracy directly; it turns out that selection tends to
   surface the redundant-coverage triples.

3. **Cheap up-front triage is now possible.** Before even loading
   candidate models, the metadata-only `redundant_coverage(T, C)`
   test tells you whether compose will likely beat spawn. This
   could make the Sprint 10 triage API nearly free:
   - If `N ≥ 100` AND redundant coverage is weak: spawn directly.
   - Else: load candidates, run coverage selection, compose.

## Limits

- **27 tasks, single seed.** The per-task advantages at N=5000
  are mostly negative (spawn usually wins). The ranking is clean
  but the effect size variance is unestimated.
- **ρ=+0.52 is strong but not decisive.** `min_element_cover`
  explains about ¼ of the variance in advantage on ranks; there's
  substantial residual noise from compose/spawn stochastics.
- **H1 and H2 aren't exhaustive.** Other mechanisms
  (e.g. calibration mismatch, ensemble entropy minimization, or
  expert confidence on specific digit classes) aren't tested.
  H2's win over H1 is robust, but some H3 might explain more.
- **"60% overlap" and "claimed by ≥2 picks" are illustrative
  thresholds.** A proper decision rule would need to learn these
  thresholds from held-out data.
- **Redundant-coverage test assumes you can enumerate candidate
  label sets.** This works when tasks are labeled with explicit
  class-1 digit sets; for fuzzier task descriptions (natural
  language, learned embeddings), you'd need a separate mapping
  from task → label proxy. Not trivial.

## Natural follow-ups

- **Sprint 10 — deploy-the-rule.** Implement the `decide()` API
  with the full triage: match (free, signature) / compose
  (signature pre-filter + redundant-coverage test + labeled
  selection) / spawn (default). Falsifiable end-to-end test: on
  27 tasks × budgets, does the routed decision match the best
  available path?
- **Generalize to multi-class targets.** The `min_element_cover`
  metric extends naturally: for a 3-class task, require every
  class's positive set claimed by ≥2 picks. Worth testing on
  `mod3` and `mod5` from the original 11-task set.
- **Expert generation that targets redundant-coverage.** Instead
  of generating experts uniformly over the label-set space,
  explicitly train experts whose class-1 sets are 3-to-5-digit
  "anchor" sets commonly overlapping with many realistic targets.
  This is a Sprint-sized research question: does curated registry
  density outperform uniform?

## Paired ChatGPT review prompts

ChatGPT should independently validate:

1. Whether **H2 (redundant coverage) truly rules out H1
   (partition)** — ρ(`pairwise_jaccard`, advantage) = −0.088 is
   weak evidence. A stronger falsification would be to SYNTHESIZE
   a pure-partition ensemble (pick 3 experts whose class-1 sets
   are disjoint and cover target) and measure. If partition
   ensembles perform badly despite perfect union coverage, H2's
   dominance is settled.
2. Whether **ρ=+0.52 on `min_element_cover` is impressive** given
   the small n (27 tasks) and the constrained label-set-size
   distribution. A Monte Carlo test with randomly reshuffled
   advantages would give a proper null.
3. Whether **the "cheap up-front triage" claim** is realistic —
   `redundant_coverage(T, C)` requires knowing each candidate's
   class-1 set exactly. In any real deployment where tasks are
   described by data rather than explicit label sets, this might
   be harder than it looks.
