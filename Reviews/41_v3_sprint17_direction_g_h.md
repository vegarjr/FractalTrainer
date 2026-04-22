# v3 Sprint 17 — Directions G + H

**Date:** 2026-04-22
**Reviewer:** Claude Opus 4.7 (1M context)

Two final primitive extensions before closing out the Sprint 17
cluster:

- **G — cross-task distillation/consolidation.** Train a generalist
  model that mimics the specialist registry's outputs on a broad
  probe set; route queries to the generalist first, fall through to
  specialists when the generalist is uncertain. Analog to how human
  memory consolidates specific experiences into general patterns.
- **H — non-classification signatures.** Sprint 17 Direction B
  ruled out penultimate-activation signatures for classification.
  H tests the next-obvious alternative for regression: signature =
  L2- or z-score-normalized prediction vector on a fixed probe.

Both primitives are implemented, unit-tested, and demoed. Both
produce partially-positive results with honest scope boundaries.

## 1. Direction G — Consolidation

### What ships

- `src/fractaltrainer/integration/consolidation.py`:
  - `teacher_output(specialists, x, weights)` — aggregate softmax
    outputs into a teacher distribution (uniform or weighted).
  - `train_generalist(specialists, loader, n_steps, temperature)` —
    KL-divergence distillation (Hinton et al., 2015) with
    temperature T=2 default.
  - `ConsolidatedRouter(generalist, registry, models, threshold)` —
    generalist-first with specialist fall-through on low confidence.
  - `ConsolidatedDecision` dataclass.
- `scripts/run_consolidation_demo.py` — demo on 15-expert MNIST.
- `tests/test_consolidation.py` — 8 unit tests (teacher aggregation,
  distillation loss reduction, confidence thresholding, fall-through
  behavior, batched prediction).

### Results (MNIST, 15 specialists, confidence_threshold=0.75)

| Task | Consolidated acc | Generalist fires (correct rate) | Fallthrough |
|---|---|---|---|
| subset_01234 | 0.956 | 13.8%  | 85.6% |
| subset_56789 | 0.830 | 0.6%   | 86.6% |
| subset_024   | 0.960 | 12.8%  | 87.0% |
| subset_13579 | 0.954 | 11.2%  | 88.6% |
| subset_357   | 0.968 | 15.4%  | 84.6% |

Mean consolidated accuracy **0.934**; mean fallthrough **86.5%**.

**Latency: 0.195 ms/query (consolidated) vs 0.074 ms/query
(always-specialist) → 2.65× SLOWER.**

### Honest interpretation

- **Accuracy neutral.** Generalist handles the 14% of queries it's
  confident about at ~95% accuracy; specialists handle the rest.
  Net accuracy is within noise of always-specialist.
- **Latency regression, not improvement.** At MNIST scale (N=15
  specialists, single-layer MLPs), the cost of (generalist forward +
  confidence check) exceeds the cost of a single specialist forward
  on fallthrough cases. Consolidation is 2.65× slower.
- **The primitive works architecturally.** Distillation converges
  (KL loss 0.028); `ConsolidatedRouter` correctly gates between
  paths; tests all pass.
- **The headroom is at scale.** At N=100+ specialists where routing
  a query touches many models (or at K=10+ compose blending),
  generalist-first starts to pay. The demo doesn't realize this —
  N=15 is too small for the overhead to amortize.

**Revised claim:** Direction G ships as a correct primitive with
measured overhead; the conditions under which it pays off (large
N, wide compose blending, or a very-confident generalist) are not
present in the MNIST demo and remain empirical follow-ups.

## 2. Direction H — Regression-probe signatures

### What ships

- `src/fractaltrainer/integration/regression_signatures.py`:
  - `regression_probe_signature(model, probe, normalize)` — L2, z-
    score, or no normalization of `model(probe).flatten()`.
  - `SineRegressor`, `make_sine_task`, `train_sine_regressor`,
    `make_probe_inputs` — toy regression infrastructure.
- `scripts/run_regression_signatures_demo.py` — 5 sine tasks × 3
  seeds, within/cross-task distance stats, Spearman ρ on
  |Δfreq| vs signature distance.
- `tests/test_regression_signatures.py` — 8 unit tests.

### Results (5 sine tasks × 3 seeds)

| Normalize | within μ | cross μ | gap | Clean sep? | Spearman ρ |
|---|---|---|---|---|---|
| L2     | 0.196 | 1.297 | +1.101 | ✗ (within-max 0.68 > cross-min 0.56) | −0.03 |
| zscore | 1.615 | 12.95 | **+11.34** | ✓ (within-max 5.05 < cross-min 5.64) | −0.03 |
| none   | 1.250 | 8.725 | +7.475 | ✗ | −0.11 |

**z-score normalization gives clean within/cross separation** across
all 15 model pairs. Sprint 7b analog on classification had ρ ≈
−0.85 between Jaccard(label sets) and signature distance; here the
analog (|Δfreq| vs signature distance) has ρ ≈ 0. Clustering works,
function-space similarity does not.

### Interpretation

- **Regression-probe signatures cluster by task.** z-score
  normalization produces a clean gap (cross-task distances
  consistently larger than within-task). Direction H's basic viability
  claim holds.
- **But the signature distance doesn't encode function similarity.**
  For sine tasks, freq=1 and freq=3 produce signatures that are no
  closer to each other than to freq=0.5. The softmax classification
  analog (Jaccard similarity → signature proximity) is specific to
  how softmax outputs compose under label-set union.
- **Practical implication.** Regression-probe signatures support
  `match` verdicts (is this new task the same as a registered
  regressor?) but not `compose` verdicts with meaningful inverse-
  distance weighting (nearest-regressor distance doesn't predict how
  much that regressor helps). The primitive is half of the
  classification story.

Possible fixes (future work):
- Use per-task probe inputs (e.g., task-dependent x-range) rather
  than a fixed probe — may recover some function similarity.
- Encode frequency explicitly via spectral signature (FFT of
  predictions) rather than raw predictions.
- Use a kernel on prediction functions (e.g., RKHS distance) rather
  than L2 in prediction space.

**Revised claim:** Regression-probe signatures enable the match
verdict in non-classification regimes; compose/spawn require
additional design. The Direction B negative (latent signatures
fail) is only partially repaired; a full fix is task-family-specific.

## 3. Paired prompts for ChatGPT review

1. **On consolidation's 2.65× slowdown at N=15.** Is there a clean
   theoretical crossover point? At what N does generalist-first
   become faster than always-specialist, assuming both run on the
   same hardware? Should the primitive advertise a minimum-N
   recommendation before shipping?
2. **On regression ρ = 0.** The clustering works but the similarity
   structure doesn't. Is "match-only" a respectable primitive, or
   does the compose-verdict gap disqualify the design for the full
   vision? If compose requires a different distance metric per
   task-type, is the signature's role architecturally the same or
   fundamentally different?
3. **On the paper.** Directions G and H are now implemented. Should
   the paper claim them (and its scope becomes "F + 6 primitives")
   or should they ship as follow-up Reviews outside the paper's
   narrative?

## 4. What ships (combined commit)

- Consolidation: `consolidation.py`, demo, 8 tests
- Regression signatures: `regression_signatures.py`, demo, 8 tests
- Results: `consolidation_demo.json`, `regression_signatures_demo_*.json`
- Review 41 (this)
- **Total tests: 249/249 passing** (233 + 16 new).
