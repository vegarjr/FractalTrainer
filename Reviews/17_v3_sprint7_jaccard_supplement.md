# v3 Sprint 7 supplement — Jaccard overlap vs signature distance

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Question

The Sprint 7 review ([16](16_v3_sprint7_scale_stress.md)) claimed
"signature space encodes task-function kinship": cross-task nearest
neighbors in signature space reflect digit-subset overlap. The
ChatGPT-review prompt flagged this as possibly post-hoc storytelling
on a small sample.

**Falsifiable prediction:** if the claim holds, pairwise Jaccard
overlap between tasks' class-1 digit sets should correlate negatively
with pairwise signature distance. If the claim is post-hoc, the
correlation should be weak or absent.

## Test

27 binary MNIST tasks (7 existing + 20 Sprint 7). For each task T,
class-1 digit set S_T ⊆ {0..9}. For each pair (T1, T2):
- `Jaccard(T1, T2) = |S1 ∩ S2| / |S1 ∪ S2|`
- `distance(T1, T2) = L2(sig(T1 seed=42), sig(T2 seed=42))`

One seed per task (seed=42). 351 pairs total.

Spearman rank correlation (robust to nonlinear Jaccard↔L2 mapping) is
the headline. Pearson correlation reported for comparison.

## Result

### **Spearman ρ = −0.848, Pearson r = −0.852 (n=351)**

This is a **very strong negative correlation**. At ρ=−0.85, rank
positions of Jaccard and distance explain ≈ 72% of each other's
variance on ranks. For context: Sprint 2 (scalar dim vs test
accuracy) was ρ=+0.12, null.

### Quartile analysis

| Quartile              | mean Jaccard | mean signature distance |
|-----------------------|-------------:|------------------------:|
| Bottom (87 pairs)     | 0.104        | 11.354                  |
| Top    (87 pairs)     | 0.513        | 7.688                   |
| **Δ**                 |              | **−3.666**              |

Top-quartile Jaccard pairs are **3.7 L2 units closer** than
bottom-quartile — a massive effect relative to the 4.6–13.6 overall
distance range.

### Extremes are clean

**Highest-overlap pairs (all low-distance):**
- J=0.800, d=4.63 : subset_02459 {0,2,4,5,9} ↔ subset_0459 {0,4,5,9}
- J=0.800, d=5.57 : subset_02349 {0,2,3,4,9} ↔ subset_2349 {2,3,4,9}
- J=0.750, d=5.38 : subset_0459 {0,4,5,9} ↔ subset_459 {4,5,9}
- J=0.714, d=6.71 : subset_123679 ↔ subset_123689
- J=0.714, d=7.04 : subset_123468 ↔ subset_123689

**Zero-overlap pairs (all high-distance):**
- J=0.000, d=13.61 : high_vs_low {5,6,7,8,9} ↔ ones_vs_teens {0..4}
- J=0.000, d=12.70 : primes_vs_rest ↔ subset_0489
- J=0.000, d=12.42 : subset_01458 ↔ subset_267
- J=0.000, d=11.98 : middle_456 ↔ subset_01789
- J=0.000, d=11.51 : middle_456 ↔ subset_0379

The high_vs_low ↔ ones_vs_teens pair is particularly clean:
**perfectly complementary** class-1 sets, maximum signature distance
of the entire 351-pair set.

## Verdict: **CLAIM SUPPORTED**

Signatures do not just happen to cluster by task — they cluster by
**label-set overlap**. A task whose class-1 digits are {1,3,5,7,9}
produces an activation signature that is close to other tasks whose
class-1 digits include {1,3,5,7,9} (or a subset thereof).

This has a concrete operational implication: **a new binary task's
nearest-registry neighbor by signature is (approximately) the
nearest registry task by Jaccard overlap of class-1 digits**. Without
ever being told what "class 1" means, the registry routes by
label-set semantics.

## Why this matters for the Mixture-of-Fractals vision

The original user vision was "the fractal geometry expands as it
learns… every new fractal aspect will help any other fractal where
the task fits best". This supplement provides the first quantitative
evidence that "fits best" has a concrete meaning:

- Signature distance is a **proxy for label-set overlap**.
- Routing by signature distance is **routing by task-function
  kinship**.
- The Sprint 5 null on composition utility (blend ≈ top-1) is
  consistent with this: when your nearest neighbors are already
  label-aligned, blending two almost-same-label experts adds no
  information beyond the single nearest. Composition would help if
  the nearest neighbors were *orthogonally* useful — but the
  signature space, by this measurement, tends to produce aligned
  rather than complementary neighbors.

This reframes compose: its utility would be highest on queries whose
top-K spread out in label space (low Jaccard between the K-nearest).
When the K-nearest are all high-Jaccard to the query and to each
other, compose is redundant with top-1.

## Limits

- **Pearson r and Spearman ρ nearly identical (both ≈ −0.85)**
  suggests the Jaccard → distance mapping is close to linear, not
  just monotonic. A more sophisticated analysis (Mantel test on
  distance matrices, partial correlation controlling for set size)
  would tighten the claim but probably not change the sign or rough
  magnitude.
- **Only one seed per task** (seed=42). A full study would use all
  3 seeds and either average or report the between-seed variance
  in the correlation. Deferred — the effect is strong enough
  (−0.85) that between-seed noise is unlikely to flip the sign.
- **27 binary tasks only.** Excludes digit_class (10-class),
  fashion_class, mod3, mod5 — the heterogeneous-complexity
  members. Extending Jaccard to partition-based distance (e.g.,
  Adjusted Rand Index) would include those. Deferred.
- **The 20 Sprint 7 tasks are deterministic subset samples** —
  they densely tile a region of label-set space that the original
  7 tasks only sparsely covered. A fresh random seed for task
  selection might shift the correlation magnitude modestly.
- **Correlation is not causation.** The experts all share the
  same MLP architecture trained on the same MNIST distribution. A
  mechanistic explanation would be: "the last layer's softmax over
  training concentrates mass on the class-1 digits, so signatures
  on unseen-digit probe batches reflect which digits the expert
  flags as class 1". This is consistent with everything observed
  but not formally demonstrated. An ablation: shuffle the probe
  batch's known vs held-out digits and see if the correlation
  persists. Deferred.

## Artifacts

- `scripts/analyze_jaccard_vs_distance.py` — the full test.
- `results/jaccard_vs_distance.json` — all 351 pairs, per-pair
  Jaccard + distance, summary stats, verdict.

## Implications for Sprint 8 / future

The signature-distance-as-label-overlap-proxy finding **supports**
hierarchical routing: dataset-level coarse routing is essentially
asking "does this query's label set overlap with ANY registered
task at all?" before drilling down into the finer task-level
distance. Sprint 8's hierarchical routing is now better motivated:
- Dataset-level: partitions the signature space by modality
  (MNIST vs Fashion vs whatever).
- Task-level within a dataset: partitions by label-set overlap, as
  demonstrated here.

It also suggests a principled **compose band redesign**: instead of
"match < d ≤ spawn", compose should fire when the top-K nearest have
high pairwise Jaccard with each other but low Jaccard with the
query — a signature of complementary neighbors, not redundant ones.
This is a research-grade refinement, not an immediate sprint.

## Paired ChatGPT review prompts

1. Whether ρ = −0.85 at n=351 is **strong enough** to accept the
   kinship claim, or whether the distribution-of-pairs structure
   (different Jaccard distributions for different set-size pairs)
   would inflate the correlation artefactually. A Mantel test with
   a size-matched null would be the formal check.
2. Whether the **one-seed-per-task** choice materially understates
   or overstates the effect. If the within-task variance in
   signatures is small relative to between-task, the effect should
   be robust; if within-task is large, the −0.85 is an
   underestimate of the underlying kinship.
3. Whether the **signature-distance-as-label-overlap-proxy** has
   second-order implications for training: if we KNOW signature
   distance tracks Jaccard, we could synthesize a "target
   signature" for a new task from its label set alone, without
   training — and use it to pre-route the query before any
   expert exists. That's a concrete research direction if the
   correlation holds up.
