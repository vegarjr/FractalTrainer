# v3 Sprint 5 — Composition Utility Test

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

Sprint 4 exposed the growth decision (match / compose / spawn) and showed
match and spawn fire end-to-end. The compose verdict was validated in
unit tests but never exercised on real data in a way that answers the
obvious question: **does blending the top-K nearest experts actually
buy you anything over the single nearest expert?**

If compose just ties top-1, the verdict is dead weight — we should
collapse the band to zero and let queries fall into either match or
spawn. If compose meaningfully beats top-1 on intermediate queries,
the hierarchy earns its third tier.

## Design

Leave-one-task-out over the 7 binary MNIST tasks in the cached set:
`parity, high_vs_low, primes_vs_rest, ones_vs_teens, triangular,
fibonacci, middle_456`. 21 trained experts total (3 seeds each).

For each binary task T:
- **registry** = 6 other binary tasks × 3 seeds = 18 entries
- **query** = T seed=2024 signature (T itself is never in the registry)
- **top-K** = 3 nearest, inverse-distance weights `softmax(−d/τ)` at τ=1.0
- **evaluate on T's relabeled test set (N=1000)**:
  - `acc_top1`: argmax of nearest expert's first-2 logits
  - `acc_blend`: argmax of weighted softmax across top-3 experts
  - `acc_oracle`: T seed=42's own expert (upper bound, never in registry)

Binary tasks use only logits [0, 1] from every expert; logits [2..9]
are ignored. This is semantically honest: each expert was trained to
push mass onto its own 2-class label set, and "class 0 probability"
from expert A is interpreted as A's vote for the query's class 0 —
with the obvious caveat that A's class 0 might mean "even" while
the query's class 0 means "not prime".

## Results

| Held-out task  | top-1 | blend | oracle | Δ blend−top1 |
|----------------|------:|------:|-------:|-------------:|
| parity         | 0.587 | 0.667 |  0.964 | **+0.080**   |
| high_vs_low    | 0.608 | 0.612 |  0.955 | +0.004       |
| primes_vs_rest | 0.667 | 0.674 |  0.961 | +0.007       |
| ones_vs_teens  | 0.611 | 0.609 |  0.956 | −0.002       |
| triangular     | 0.647 | 0.609 |  0.969 | **−0.038**   |
| fibonacci      | 0.666 | 0.674 |  0.975 | +0.008       |
| middle_456     | 0.613 | 0.613 |  0.971 | +0.000       |
| **mean**       | **0.628** | **0.637** | **0.964** | **+0.008** (std 0.035) |

**Paired t-statistic: 0.63.** Wins: 4 / ties: 1 / losses: 2 out of 7.

The signed Δ mean (+0.008) is well inside ±2·SE (SE = 0.013). On a
sign-test basis, P(≥4 wins | 6 non-ties, H₀=0.5) ≈ 0.34. No evidence
the blend differs from top-1.

### Verdict: **NULL** — compose does not beat top-1 on this registry.

## What this means

1. **Compose as implemented gives no lift.** The inverse-distance blend
   of cross-task binary experts averages to ≈ the nearest one. Label-
   semantic mismatch (parity's "odd" ≠ primes's "prime") dominates.
   The experts agree often enough to reinforce each other but not in a
   way that corrects nearest-expert errors.
2. **Both top-1 and blend are catastrophically below oracle** (0.63 vs
   0.96). Cross-task routing buys ~12 pp above the ~50% class-prior
   floor; the oracle captures the remaining 34 pp. The interesting
   question isn't "blend vs top-1" — it's "when should we spawn rather
   than route at all?" Sprint 4's answer (distance > 7) seems right:
   all these leave-one-task-out queries sat at distances 7.3-9.4, and
   they all belong in spawn territory, not compose.
3. **The one real win (parity, +0.080) is structurally explainable.**
   Top-3 for parity was {triangular_seed101, primes_vs_rest_seed2024,
   primes_vs_rest_seed101}. Triangular labels `y ∈ {1, 3, 6}` (two odd,
   one even) and primes labels `y ∈ {2, 3, 5, 7}` (three odd, one
   even) — both biased toward odd. Averaging two odd-biased signals
   gets closer to parity's true "odd" label than any single expert
   alone. The rest of the tasks don't share such a structural accident.
4. **End-to-end compose verdict does fire.** With `match=5.0,
   spawn=15.0` and primes_vs_rest held out, the decision is
   `verdict=compose, min_distance=7.375, top-3 = {parity×2,
   fibonacci}`. The hierarchy plumbing is correct; the question was
   purely about utility.

## Why NULL (not an indictment of the vision)

The Mixture-of-Fractals idea — "1+1 expert and 1×1 expert help each
other through blending" — requires tasks whose label semantics share
structure in ways that inverse-distance weighting can exploit. The 7
binary MNIST tasks don't share enough structure for this: parity, 
primality, and is-member-of-{4,5,6} are independently-defined label
functions over the same input space. A nearest-expert lookup picks the
task whose input-feature distribution is most similar, but its labels
are largely orthogonal to the query's.

Where compose should work (not demonstrated here):
- Tasks that are **refinements** of each other: "primes vs rest" and
  "small primes vs rest" — an expert blend could interpolate between
  coarser and finer granularities.
- Tasks that share a **hierarchical label structure**: multi-label
  tasks where the query is a projection of a larger multi-label task
  in the registry.
- **Multi-modal queries**: an input that's part-numeric-part-fashion
  where blending a digit expert and a fashion expert might genuinely
  be useful.

None of these are in the current cached set. A meaningful compose
test requires either building such a test set or finding it in a
real-world stream.

## What ships

- `scripts/run_composition_utility.py` — the full experiment, reusable
  for a different binary-task set or larger K.
- `results/composition_utility.json` — raw per-task accuracies, top-K
  distances and weights, verdict + paired-t stats.
- Compose-verdict end-to-end check (primes_vs_rest leave-one-out with
  match=5.0, spawn=15.0 → verdict=compose). Plumbing confirmed.

No changes to `FractalRegistry` itself — the existing
`composite_weights` method was exactly what this experiment needed.
The implication: the registry interface is stable; what's open is the
empirical question of when compose is worth invoking.

## Implications for Sprint 6+

1. **Shrink the compose band** — default thresholds should put
   cross-task-within-family queries in spawn, not compose. The Sprint 4
   defaults (match=5.0, spawn=7.0) already do this for the current
   MNIST signature distribution. That was the right call.
2. **Compose is a future feature, not a current one.** Don't advertise
   blending as a utility win on this registry. It's mechanically
   implemented and verifiably fires; real usefulness is contingent on
   a registry whose nearby-but-not-same-task neighbors are
   semantically composable.
3. **Sprint 6 (adaptive thresholds) should calibrate `spawn_threshold`
   from running percentiles of cross-task distances, not from
   compose-band width.** The compose band can stay narrow by default
   until there's demonstrated utility in widening it.

## Limits

- **n=7 is small.** The paired-t and sign test are underpowered.
  Adding a 3-class and 10-class leave-one-task-out would triple the
  sample but complicates the n_classes slicing. Skipped for focus.
- **Temperature τ=1.0 is arbitrary.** A lower τ (sharper weights,
  closer to top-1) or higher τ (more equal blend) might move the
  numbers. Did not sweep — the point estimate at τ=1.0 is null, and
  a sweep is more likely to confirm null than overturn it.
- **Slicing logits to query-task n_classes is a modeling choice.**
  An alternative is to blend the full 10-class softmax and then sum
  class-groups per the query's relabel function. I expect that to
  perform identically for binary tasks (logits 2..9 are suppressed in
  training), but it would be a cleaner abstraction for mixed-n_classes
  scenarios.
- **Oracle is seed=42, not the held-out seed=2024.** That's the
  nearest non-registry same-task expert. Using seed=2024 as oracle
  would be definitionally perfect (it IS the query) and trivialize
  the upper bound. Seed=42 is a fair "other-good-expert" upper bound.

## Paired ChatGPT review prompts

Per the dual-AI workflow, independent reads from ChatGPT should
validate:

1. Whether the **null verdict is the right call** given the small n
   and the single +0.080 outlier (parity). A skeptical reader might
   argue for "underpowered" rather than "null", which would suggest a
   different Sprint 6 direction (run more tasks before concluding).
2. Whether the **label-semantic-mismatch explanation** for the null
   is convincing or just post-hoc narrative. Stronger would be a
   prediction: "on tasks T1 and T2 whose labels ARE a composable
   function, compose would win". The trilogy of binary tasks where
   this should be true (if the theory holds) would be: parity ∧
   primes, parity ∧ high_vs_low, with a target like "parity ∧ prime
   ∧ high". This is a concrete Sprint-6-alternative test.
