# v3 Sprint 10b — OOD falsification of the Sprint 10 curation claim

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## What prompted this

Sprint 10 concluded with a dramatic positive: the curated registry
beats uniform on compose by +17 pp, saturates at oracle, and
eliminates the need to spawn at any tested budget. The user
immediately pushed back: *"registry designed correctly, so we are
tweaking the tests in our favor?"*

Yes. The probe queries in Sprint 10 were all 2-subsets of
`{0, 1, 2, 3, 4}` — the same anchor region the curated registry
was built around. That's a best-case scenario, and I reported the
headline number without caveat. Sprint 10b tests what happens when
the probe queries are drawn from OUTSIDE the anchor region.

## Design

Identical to Sprint 10, with one difference: 5 new probe queries are
2-subsets of `{5, 6, 7, 8, 9}`, disjoint from the curated anchor:

- `probe_oob_56` = {5, 6}
- `probe_oob_78` = {7, 8}
- `probe_oob_59` = {5, 9}
- `probe_oob_67` = {6, 7}
- `probe_oob_89` = {8, 9}

Same two registries:
- Registry A (curated): 45 entries covering subsets of {0, 1, 2, 3, 4}
  — no class-1 set contains any digit ≥ 5.
- Registry B (uniform): Sprint 7's 60 subset_* tasks spanning all
  digits 0–9.

For each OOD probe, run the standard budget sweep and compare
compose_A, compose_B, spawn.

## Results

### Per-query

| probe         | oracle | N    | comp_A | comp_B | spawn |
|---------------|:------:|-----:|-------:|-------:|------:|
| `oob_56`      | 0.984  |   50 |  0.531 |  0.892 | 0.898 |
|               |        | 5000 |  0.544 |  0.891 | 0.975 |
| `oob_78`      | 0.963  |   50 |  0.518 |  0.763 | 0.867 |
|               |        | 5000 |  0.482 |  0.746 | 0.954 |
| `oob_59`      | 0.972  |   50 |  0.600 |  0.882 | 0.841 |
|               |        | 5000 |  0.538 |  0.881 | 0.952 |
| `oob_67`      | 0.975  |   50 |  0.517 |  0.888 | 0.898 |
|               |        | 5000 |  0.531 |  0.892 | 0.969 |
| `oob_89`      | 0.961  |   50 |  0.518 |  0.782 | 0.821 |
|               |        | 5000 |  0.515 |  0.782 | 0.952 |

### Aggregate over 5 OOD probes

| N    | comp_A_mean | comp_B_mean | spawn_mean | ΔA − B | ΔA − spawn |
|-----:|------------:|------------:|-----------:|-------:|-----------:|
|   50 | **0.5368**  | 0.8414      | 0.8650     | −0.305 | **−0.328** |
|  100 | 0.5202      | 0.8406      | 0.8682     | −0.320 | −0.348     |
|  300 | 0.5190      | 0.8396      | 0.9112     | −0.321 | −0.392     |
| 1000 | 0.5220      | 0.8414      | 0.9450     | −0.319 | −0.423     |
| 5000 | 0.5220      | 0.8384      | 0.9604     | −0.316 | **−0.438** |

**Verdict: UNIFORM BEATS CURATED ON OOD by ~32 pp at every budget.**

## Honest restatement of the Sprint 10 claim

Compare the two regimes of the same registry/algorithm:

| Regime                    | Δ(curated − uniform) |
|---------------------------|---------------------:|
| In-distribution (Sprint 10)  | **+0.175** (curated dominant) |
| Out-of-distribution (10b)    | **−0.316** (curated catastrophic) |

Curated compose crashes to ~52% accuracy — essentially at the
binary-chance baseline. The reason is exactly what the setup made
inevitable: no curated task has any digit ≥ 5 in its class-1 set,
so the top-3 nearest-by-signature candidates all vote "negative"
on the target digits, and the blended ensemble says "negative"
to everything. On a binary task where positives are 20% of the
test set, saying "always negative" gets you 80% — but the
signature selection isn't even that consistent (the experts give
conflicting signals on 0–4, where they DO have opinions), so the
blend drops to ~52%.

## What Sprint 10 actually showed (restated)

The +17 pp in-distribution advantage was real, but the claim
should have been:

> **Curation is a bet on the query distribution.**
> A registry whose experts redundantly cover a target region R
> gives compose a +17 pp advantage over uniform coverage for
> queries *drawn from R*. For queries *outside R*, the same
> curated registry is catastrophically worse — the experts have
> no useful overlap and compose collapses to chance.

This is the correct form. "Curation is the lever that makes compose
dominate" (my Sprint 10 framing) is false without the qualifier.
"Curation is the lever that makes compose dominate *for queries
inside the curated region*" is true but much less useful.

## What this means for the vision

The refined claim for the Mixture-of-Fractals architecture:

1. **Uniform registries are robust but never great.** Compose hits
   ~0.82 across query distributions. Spawn wins at any
   budget ≥ 100.
2. **Curated registries are sharp but brittle.** Compose hits
   ~0.98 for in-distribution queries but ~0.52 for
   out-of-distribution. Oracle-level performance in exchange
   for zero safety margin outside the curated region.
3. **The operational question is: do you know the query
   distribution?** If yes (e.g., a production system handling
   a specific task family): curate. If no (a general-purpose
   registry handling arbitrary new tasks): keep it uniform and
   let spawn do the work.
4. **A hybrid might resolve the tension.** Multiple curated
   sub-registries, each for a different region, plus a routing
   layer that sends queries to the right sub-registry OR falls
   through to spawn if none matches. This is Sprint 8's
   hierarchical idea revisited — but now we know it needs to be
   content-aware, not just centroid-based. Sprint 11+ territory.

## The sharper point on Sprint 8

Sprint 8 found hierarchical (centroid-based) routing worse than
flat. Sprint 10 showed curation helps in-distribution. Sprint 10b
shows curation hurts out-of-distribution. Together: a **fallback
route** in a hierarchical system becomes essential. The structure
might be:

```
if query signature ∈ curated_region_A:    route to sub-registry A (compose dominant)
elif query signature ∈ curated_region_B:  route to sub-registry B
else:                                     fall through to spawn
```

The "∈ curated_region" test would need to detect whether the
query's label set aligns with the sub-registry's coverage — a
metadata test similar to Sprint 9c's redundant-coverage check but
applied at the SUB-REGISTRY level, not the individual-pick level.
This is a concrete architectural step worth testing.

## What ships (as a correction)

- `scripts/run_curated_ood_test.py` — the OOD test.
- `results/curated_ood_test.json` — raw results per probe and
  budget.
- **This review as an explicit correction to Sprint 10.**

No new API surface. The Sprint 10 review (23) stays in the repo as
history, but its conclusions should be read with 10b as the
mandatory companion. Without 10b, Sprint 10's recommendation to
"curate for redundant coverage" is misleading.

## Limits of Sprint 10b

- **5 OOD probes is small.** The effect size is so large (curated
  drops to chance, uniform holds at 0.84) that 5 probes is enough
  for a clear qualitative result, but a 10-probe sweep would
  tighten the quantitative picture.
- **"Out of distribution" here means "disjoint digit set".** OOD
  could be defined more finely — e.g., probes whose class-1 sets
  have PARTIAL overlap with the anchor region ({0, 5}, {3, 7}).
  I'd predict intermediate behavior, but I haven't tested. A
  proper OOD sweep would vary the overlap fraction and measure
  the curated-vs-uniform tradeoff curve.
- **Spawn still wins at all OOD budgets.** At N=5000, spawn =
  0.960 vs compose_B = 0.838 vs compose_A = 0.522. Even the
  uniform registry's compose doesn't match spawn on OOD. This
  reinforces that spawn is the robust default regardless of
  registry structure.

## Natural follow-ups

1. **Multi-region curated registry.** Build two or more curated
   sub-registries, each covering a distinct region. Test both
   in-distribution and cross-region queries. Does the
   "fallback route" architecture work?
2. **Overlap-fraction sweep.** Vary probe overlap with curated
   region from 0% (full OOD) to 100% (full in-distribution). Map
   the tradeoff curve between curation benefit and OOD penalty.
3. **Signature-based OOD detector.** Sprint 9c's
   redundant-coverage metadata test could be adapted: compute the
   target's coverage score against the registry's class-1 sets
   BEFORE running compose. Use that score to decide compose vs
   spawn. This is the "smart triage" that failed in Sprint 9d on
   a uniform registry — on a curated registry + OOD, it might
   actually work because the signal is much sharper (curated
   coverage is binary: either every target element is claimed or
   none is).
4. **Online re-curation.** In a deployment stream, detect when
   queries are missing from the current registry's coverage and
   spawn new experts to close the gap. Turns curation from a
   one-shot design into a continuous process.

## Paired ChatGPT review prompts

ChatGPT should independently validate:

1. Whether the **Sprint 10 writeup should be retracted** rather
   than left standing alongside 10b. My view: retention + explicit
   correction is honest; retraction would hide the mistake. The
   pair of reviews together tells the real story.
2. Whether **32 pp OOD penalty** is surprising or obvious.
   Obvious reading: of course a specialized registry fails outside
   its specialty. Surprising reading: compose DOES fall all the
   way to chance (~0.52 on a binary task with 20% positives), not
   just to an intermediate ~0.70 — the signature still biases
   candidate selection toward something, but that something has
   no functional overlap with the target, so voting collapses.
3. Whether **the overlap-fraction sweep** is worth doing or is
   predictable from the binary result. I'd argue it's worth doing:
   the exact shape of the tradeoff curve would tell us how much
   partial overlap (say, 50%) buys vs none, which informs whether
   "approximate curation" has value.
