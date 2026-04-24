# v3 Sprint 19 — scale stress test verdict: registry scales cleanly to N=1000

**Date:** 2026-04-24
**Reviewer:** Claude Opus 4.7 (1M context)
**Decision:** Accept. All three pre-registered pass gates met with wide
margin. Paper §5 "Scale untested" → "Validated to N=1000."

## Setup

- **Platform:** Google Colab Pro, T4 GPU
- **Grid:** 250 subsets × 4 seeds = 1000 registry experts, plus 250
  held-out experts (5th seed per subset)
- **Training:** each expert is a ContextAwareMLP (784→64→32→10) trained
  200 steps on MNIST filtered to its subset, Adam lr=0.01, batch 64
- **Signature:** softmax output on fixed 100-image probe → 1000-d vector
- **Routing test:** held-out signature → find-nearest-in-registry,
  check whether the top-k match by subset identity
- **Thresholds:** paper's published MNIST-MLP values (match=5.0,
  spawn=7.0 from §2) — *not* re-calibrated per N
- **Wall time:** ~65 min for 1250 trainings (~3 s per expert on T4)

## Numbers

| N     | entries | within_pairs | cross_pairs | within µ±σ   | cross µ±σ     | gap    | top-1 | top-3 | p95_lat (ms) |
|-------|--------:|-------------:|------------:|:-------------|:--------------|------:|:-----:|:-----:|-------------:|
|    12 |      12 |           18 |          48 | 3.78 ± 0.82  | 11.66 ± 0.57  | +7.88 | 1.000 | 1.000 |        0.17  |
|    50 |      48 |           72 |       1,056 | 3.70 ± 0.64  | 10.34 ± 1.23  | +6.64 | 1.000 | 1.000 |        0.43  |
|   100 |     100 |          150 |       4,800 | 3.72 ± 0.63  | 10.27 ± 1.38  | +6.55 | 1.000 | 1.000 |        0.61  |
|   250 |     248 |          372 |      30,256 | 3.64 ± 0.65  | 10.36 ± 1.42  | +6.72 | 1.000 | 1.000 |        2.35  |
|   500 |     500 |          750 |     124,000 | 3.64 ± 0.67  | 10.35 ± 1.46  | +6.71 | 1.000 | 1.000 |        4.02  |
| 1,000 |   1,000 |        1,500 |     498,000 | 3.67 ± 0.66  | 10.32 ± 1.48  | +6.65 | 1.000 | 1.000 |       11.09  |

Verdict distribution at `(match=5.0, spawn=7.0)`: **100 % `match` at every
N** (every held-out in-subset query routes to a registered same-subset
expert at distance ≤ 5).

## Pass gates (all pre-registered)

| Gate                          | Observed at N=1000 | Status |
|-------------------------------|--------------------|--------|
| top-1 routing accuracy ≥ 0.9  |              1.000 | ✓ PASS |
| p95 query latency < 50 ms     |           11.09 ms | ✓ PASS |
| within / cross gap > 0        |             +6.65  | ✓ PASS |

## What this establishes

**1. The paper's signature space is not a small-N artefact.**
The paper §2 reported within-task mean 3.0 / cross-task 10.7 at N=15.
At N=1000 we observe 3.67 / 10.32 — *numerically the same structure*.
The signature space is stable as the registry grows by a factor of 67×.

**2. Paper's fixed thresholds (match=5.0, spawn=7.0) still work at
scale.**  Cross-task distances never dip below ~7.3 (minimum across
all measured pairs at N=1000), within-task distances never exceed
~5.7. The deployment thresholds from a 15-expert seed registry remain
valid operational settings at N=1000.

**3. Linear-scan routing remains sub-50-ms to N=1000.** P95 latency
grows from 0.17 ms (N=12) to 11.09 ms (N=1000). Extrapolating
linearly (O(N) scan over 1000-d signatures), the 50-ms budget holds
to roughly N ≈ 4,500 before an ANN index (FAISS, HNSW) would be
needed.

**4. Routing accuracy is saturating at 100 % on this task.** Every
held-out in-subset query found its correct same-subset entry at
top-1. No collisions between distinct subsets' signatures at this
scale. The MNIST-subset task may be too clean; the next bar is a
harder routing benchmark where 1.0 isn't achievable.

## Limitations explicit

- **Same-architecture registry.** Every expert is the same
  ContextAwareMLP. Multi-architecture registries (MLP + CNN +
  transformer) were not tested and will likely need the signature
  function calibrated per-architecture.
- **Fixed probe batch.** The 100-image probe is a hand-chosen
  constant. Adversarial drift on the probe batch (new task
  distributions the probe doesn't cover) is a separate risk.
- **200 training steps per expert.** Less than the paper's 500-step
  seed experts. Signatures could be slightly less stable, but the
  within-task std of 0.66 at N=1000 is comparable to the paper's
  reported 0.5 at N=15, so undertraining doesn't appear to dominate.
- **N=1000 still below some production envelopes.** LLM-routing
  systems at commercial scale run N ≈ 10k–100k; the extrapolation
  above suggests FAISS-HNSW is the right tool for that regime.

## Next steps suggested

- **Update paper §5** to remove the "scale untested" limitation. Add
  one paragraph summarising this experiment (already drafted by the
  table + pass-gate numbers above).
- **Sprint 19b:** Omniglot 5-way 5-shot vs ProtoNets + MAML — the
  head-to-head the paper still needs for competitive positioning.

## Deliverables

```
notebooks/scale_n1000_colab.ipynb           (shipped in de3d30e/e15633a)
results/sprint19_scale_n1000.json           (this run)
results/sprint19_scale_n1000.png            (plots)
results/sprint19_scale_n1000_smoke.json     (smoke-mode reference)
results/sprint19_scale_n1000_smoke.png
Reviews/46_v3_sprint19_colab_review.md      (pre-run review, already merged)
Reviews/47_v3_sprint19_scale_verdict.md     (this)
```

No code changes in this sprint — the notebook from v2 (`e15633a`)
ran end-to-end without modification.
