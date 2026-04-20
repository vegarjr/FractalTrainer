# v3 Sprint 9d — Testing the redundant-coverage rule: correlation ≠ decision signal

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

Sprint 9c found that `min_element_cover` and `jaccard_to_target`
correlate with compose-minus-spawn advantage at Spearman ρ ≈ +0.5.
I proposed a triage rule upgrade:

```
if N < 100:                             compose
elif redundant_coverage(T, candidates): compose     # new
else:                                   spawn
```

**Sprint 9d tests whether this rule actually routes better than the
simple N-threshold rule.** This is the critical falsification: does
Sprint 9c's correlation translate into a useful decision?

## Method

Reuse Sprint 9b's 135 (task, budget) rows — each has `acc_compose`,
`acc_spawn`, and `compose_selected` (the 3 experts coverage actually
picked). For each rule, apply it and score with the chosen path's
accuracy. Six rules:

| Rule             | Decision logic                                   |
|------------------|--------------------------------------------------|
| `oracle`         | pick max(compose, spawn) — ceiling, not deployable |
| `always_compose` | always compose                                    |
| `always_spawn`   | always spawn                                      |
| `simple_N`       | compose if N<100 else spawn (Sprint 9b)          |
| `metadata`       | simple_N, OR compose if top-10 pool has redundant coverage (metadata-only) |
| `post_hoc`       | simple_N, OR compose if already-selected 3 picks have min_element_cover ≥ 2 AND mean Jaccard ≥ 0.40 |

`metadata` requires nothing beyond label metadata — free up-front.
`post_hoc` requires the greedy coverage selection to have already
run (so the picks are known), but is cheaper than the eval forward pass.

## Results

### Aggregate over 27 tasks × 5 budgets = 135 decisions

| Rule             | Mean accuracy | compose % | spawn % | Δ vs simple_N | paired t |
|------------------|--------------:|----------:|--------:|--------------:|---------:|
| **oracle**       |  0.8934       | 27.4      | 72.6    | +0.0113       | +4.91    |
| **simple_N**     |  0.8821       | 20.0      | 80.0    | —             | —        |
| post_hoc         |  0.8782       | 43.7      | 56.3    | −0.0039       | −1.22    |
| always_spawn     |  0.8759       |  0.0      | 100.0   | −0.0061       | −1.83    |
| metadata         |  0.8440       | 82.2      | 17.8    | −0.0381       | −6.54    |
| always_compose   |  0.8195       | 100.0     |  0.0    | −0.0626       | −9.26    |

### Per-budget

| N    | oracle | always_compose | always_spawn | simple_N | metadata | post_hoc |
|-----:|-------:|---------------:|-------------:|---------:|---------:|---------:|
|   50 | 0.8397 | 0.8190         | 0.7883       | 0.8190   | 0.8190   | 0.8190   |
|  100 | 0.8550 | 0.8190         | 0.8301       | 0.8301   | 0.8347   | 0.8467   |
|  300 | 0.8964 | 0.8198         | 0.8879       | 0.8879   | 0.8498   | 0.8853   |
| 1000 | 0.9313 | 0.8244         | 0.9292       | 0.9292   | 0.8614   | 0.9173   |
| 5000 | 0.9443 | 0.8151         | 0.9443       | 0.9443   | 0.8553   | 0.9228   |

## Verdict: **NEITHER PROPOSED RULE BEATS simple_N.**

- **Metadata rule loses by 3.8 pp** (t=−6.54). It fires compose 82%
  of the time — the "redundant pool exists" signal is not selective
  enough. Many tasks have a top-10 pool with multiply-claimed target
  elements, but compose still loses to spawn at N≥300.
- **Post-hoc rule ties simple_N** (−0.004, t=−1.22, within noise).
  It fires compose 44% of the time and improves over simple_N at
  N=100 (+0.017) but costs accuracy at N=1000 (−0.012) and N=5000
  (−0.021). The marginal gains and losses cancel.
- **The oracle beats simple_N by only +1.1 pp.** Even with perfect
  per-decision routing knowledge, the maximum achievable lift over
  simple_N is small. **The routing problem is nearly saturated.**

Sprint 9c's correlation (ρ=+0.52 between `min_element_cover` and
advantage) was real. But the correlation did not convert into a
decision rule because:

1. The advantage distribution is extremely asymmetric. 26/27 tasks
   have negative advantage at N=5000; only subset_459 has +0.002.
   A correlation on a distribution dominated by negative values is
   informative about which tasks lose *less* badly, not about
   which tasks win. A rule that tilts toward compose tilts toward
   losing.
2. The signal is noisy at individual-task decisions. Even tasks
   that satisfy `min_element_cover ≥ 2` and high Jaccard-to-target
   still lose to spawn most of the time at N=300+, because spawn
   scales steeply (0.79 → 0.94 from N=50 to N=5000) and compose
   saturates flat (0.82 at all N).

## What this means for the vision

The triage story is settled at three levels:

1. **Simple-N is near-optimal.** The `if N<100: compose else: spawn`
   rule closes 35% of the always-spawn → oracle gap, and the oracle
   is only +1.1 pp above it. No deployable rule we tried adds more.
2. **Compose's niche is small and operationally hard to identify
   in advance.** The tasks where compose genuinely wins at large N
   (subset_459 alone) can be described post-hoc but not predicted
   from available signals with enough precision to override the
   spawn default. Unless a sharper signal appears, compose should
   stay as a cold-start fallback only.
3. **Sprint 9c's correlation is still the right mechanistic
   explanation for WHY coverage works when it works.** Sprint 9d
   only rules out using it as a triage predictor. The mechanistic
   claim ("compose wins through redundant voting on positive
   labels") remains useful for understanding and for future
   registry design choices (e.g., expert-generation strategy).

## Honest reframing of what we've learned

Composition's story, from Sprint 5 through 9d:

- **Sprint 5:** top-K-nearest-by-signature compose = null vs top-1.
- **Sprint 7b:** signatures encode Jaccard kinship → neighbors are
  label-redundant → explains why similarity-compose is null.
- **Sprint 9a:** greedy coverage selection beats both top-1 and
  nearest-K3 by +8.1 pp when given 300 labels.
- **Sprint 9b:** beating spawn requires N < 100 — compose is a
  cold-start fallback.
- **Sprint 9c:** when compose DOES win, it wins through redundant
  positive-label agreement, not orthogonal partition.
- **Sprint 9d:** the redundancy signal is real but too noisy to
  improve triage beyond the simple-N rule.

The final, empirically-grounded version of the Mixture-of-Fractals
triage:

```
if labeled_data is None:       match-or-spawn  (legacy)
elif N_labels < 100:            compose         (coverage-greedy K=3)
else:                           spawn           (train fresh MLP)
```

This is simple, empirically near-optimal, and implementable. The
"smart per-task rule" remains an open research question — possibly
solvable with a better-designed registry (Sprint 9c's
high-overlap-curation recommendation) or a sharper per-task signal.

## Limits

- **27 tasks, all binary MNIST subsets.** Sprint 9d's verdict that
  "simple_N is near-optimal" is MNIST-binary-specific. For harder
  tasks where spawn is less effective at small N, compose's niche
  widens and a smarter rule might pay off.
- **Thresholds in the proposed rules are not tuned.** Sprint 9d
  used `min_element_cover ≥ 2`, `mean_jaccard ≥ 0.40`, `metadata
  min_elem_claims ≥ 3`. A hyperparameter sweep might find better
  thresholds; I didn't sweep because the effect size gap to
  simple_N (−3.8 pp on metadata) is too large to be rescued by
  threshold tuning.
- **The metadata rule fires too often.** A stricter condition (e.g.,
  require `mean_jaccard ≥ 0.60` AND `min_elem_claims ≥ 4`) would
  compose less. But the whole point of redundant coverage is that
  it's supposed to detect "compose-wins territory", and if we
  stricten it to near-zero trigger rate we haven't gained anything
  over simple_N.
- **Oracle ceiling is low.** +1.1 pp improvement over simple_N is
  all that's available. Even a perfect rule wouldn't change the
  story much. This strongly suggests the next frontier isn't
  routing refinement, it's **changing the registry contents** — by
  Sprint 9c's advice, curating for redundant-overlap density.

## What ships

- `scripts/test_routing_rules.py` — applies 6 routing rules to
  Sprint 9b's stored (task, budget, acc_compose, acc_spawn,
  compose_selected) rows. Fully reproducible with existing data.
- `results/routing_rules.json` — per-rule aggregates, per-budget
  breakdown, paired-t statistics.

No new models, no registry changes. Pure meta-analysis of existing
data.

## Natural follow-ups

- **Ship simple_N in `FractalRegistry.decide()`.** The rule works
  and is near-optimal. An API that takes `labeled_data=None`,
  `label_budget=N` and routes according to simple_N is deployable
  today.
- **Curated-registry experiment.** Sprint 9c suggested that
  high-overlap coverage beats orthogonal coverage for compose.
  A real test: generate 20 new expert pairs where one cohort is
  deliberately high-overlap, another deliberately orthogonal, then
  repeat 9b's budget sweep. Does the high-overlap registry widen
  compose's niche?
- **Harder task universe.** On 10-class classification (where
  spawn struggles at small N), simple_N's "compose if N<100"
  might not be enough — maybe "compose if N<500". A direct test on
  `mod3`, `mod5`, `digit_class` would recalibrate.

## Paired ChatGPT review prompts

ChatGPT should independently validate:

1. Whether the verdict "simple_N is near-optimal" would survive a
   **threshold sweep on the redundant-coverage rule**. If a
   threshold search can find a setting where metadata beats
   simple_N by ≥1 pp, that changes the verdict. I claim not, but
   a proper search would settle it.
2. Whether the **oracle-vs-simple_N gap of +1.1 pp** is small
   because the dataset is easy (binary MNIST) or because routing
   fundamentally can't do much more than the label-budget rule on
   this sort of problem. A harder task universe would distinguish.
3. Whether the **failure of correlation-to-rule translation** here
   undermines the Sprint 9c mechanistic story, or whether the two
   are independent claims (correlation explains mechanism;
   decision rule requires sharper per-task signal). My read is the
   latter — Sprint 9c's story stands, Sprint 9d just rules out
   using it as a deployable predictor at 27-task scale.
