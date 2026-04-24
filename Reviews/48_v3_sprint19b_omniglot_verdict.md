# v3 Sprint 19b — Omniglot verdict: FractalTrainer loses to ProtoNets by 18.7 pp (but context injection works)

**Date:** 2026-04-24
**Platform:** Google Colab Pro, T4 GPU
**Result file:** `results/sprint19b_omniglot.json` (100 test episodes)
**Decision:** Accept negative; **update paper §5 / §1.5 to report** the
meta-learning gap honestly. The sub-finding — context injection
independently validates at p=0.018 — survives and strengthens §3.

## Headline numbers (Omniglot 5-way 5-shot, 100 test episodes)

| Method                 | Mean accuracy | Std    | n   |
|------------------------|--------------:|-------:|----:|
| pixel-kNN (baseline)   | 0.666         | 0.094  | 100 |
| FractalTrainer no-ctx  | 0.662         | 0.104  | 100 |
| **FractalTrainer +ctx**| **0.698**     | 0.108  | 100 |
| **MLP-ProtoNets**      | **0.886**     | 0.063  | 100 |

Shared backbone: `784 → 64 → 32` MLP across all three learned
methods (pixel-kNN is backbone-free). Only difference is what
happens on top of the 32-d embedding.

## Welch t-tests

| Pair                                          | Δ          | t       | p           | Sig |
|-----------------------------------------------|-----------:|--------:|------------:|-----|
| fractal vs protonets                          | **−0.187** | −14.86  | 6.7×10⁻³²   | **—loss** |
| fractal vs pixel-kNN                          | +0.033     |  +2.27  | 0.024       | *   |
| **fractal vs fractal_no_ctx (context alone)** | **+0.036** |  +2.39  | **0.018**   | **—context works** |
| protonets vs pixel-kNN                        | +0.220     | +19.32  | 4.1×10⁻⁴⁵   | **—sanity** |

## Pre-registered verdicts

- **FractalTrainer vs ProtoNets**: *LOSS*. Pre-registered rule
  "LOSS if Δ < −0.03" is met decisively (Δ = −0.187, p < 10⁻³¹).
- **Fractal > pixel-kNN by > 0.2**: **FAIL** (only +3.3 pp).
  The MLP backbone without meta-learning is barely better than
  naive pixel k-NN on Omniglot — backbone is too weak for this
  task without meta-training.
- **Context injection works independently**: *PASS* (+3.6 pp,
  p=0.018). Per-episode the context-enriched expert beats the
  same expert trained cold on the same 25 support examples.

## What this means

### 1. Registry-based spawn is *not* a drop-in replacement for meta-learning in few-shot regimes

ProtoNets reaches 0.886 on Omniglot 5-way 5-shot with the same 32-d
MLP embedding. FractalTrainer reaches 0.698 with that same embedding
architecture but no meta-training. **The 18.7 pp gap is the cost of
not meta-training.** Registry spawning supervises each expert on its
own 25 support examples; ProtoNets meta-learns an encoder that makes
*any* 5-way 25-sample task trivially separable by prototype distance.

This is consistent with the paper's §4 framing — FractalTrainer
targets a different regime (independent classifiers, monotonic
growth, no retraining). Few-shot with tiny support sets is the
regime meta-learning owns.

### 2. Context injection itself is externally validated

The context primitive's sub-finding is the cleanest external
result the paper has: **+3.6 pp over its own no-context
ablation at p=0.018 on 100 held-out Omniglot episodes.**

This corroborates the paper's MNIST/CIFAR ablations (α ≈ 0.15 gap
recovery) in a second, independent dataset. The mechanism is
real; it just isn't *sufficient* to close the meta-learning gap.

### 3. The paper needs a §1.5 limitation paragraph

Honest framing:
> *"We tested MoSR's spawn+context primitive against MLP-ProtoNets
> (Snell et al. 2017) on Omniglot 5-way 5-shot over 100 held-out
> episodes with matched 784→64→32 embedding backbones.  MoSR's
> per-episode spawn reaches 0.698 ± 0.108; MLP-ProtoNets reaches
> 0.886 ± 0.063 (Δ=−18.7 pp, p < 10⁻³¹). The spawn-time context
> primitive provides a reliable +3.6 pp lift over its no-context
> ablation (p=0.018), reproducing the α≈0.15 gap rule on a second
> dataset, but cannot compensate for the absence of episodic
> meta-training.  MoSR's intended regime is continual expert
> expansion under drifting task distributions (Sprint 19 validated
> routing to N=1000 with 100% top-1 retrieval); few-shot meta-
> learning is a complementary, not competing, paradigm."*

## What this rules in and out for future sprints

- **Ruled in**: a *hybrid* sprint where the registry stores a single
  meta-trained encoder (as a shared backbone) + per-episode heads.
  This matches ProtoNets for few-shot and adds the registry's
  expansion/audit/swap properties. Might close most of the gap.
- **Ruled out**: trying to tune FractalTrainer's few-shot accuracy
  by knob-turning (more spawn steps, different context K, etc.).
  The gap is structural — the backbone is untrained on episodes.
- **Still open**: head-to-head on a non-few-shot continual-learning
  benchmark (Split-CIFAR-100 with iCaRL / A-GEM), where
  FractalTrainer's natural advantages (monotonic expansion, per-
  task specialization) matter. This is Option B from the sprint-
  planning, still viable.

## Infrastructure finding from the run

One bug caught and fixed mid-run: `gather_context()` hardcoded
`.cpu()` on returned tensors; on Colab GPU this crashed
ContextAwareMLP's LayerNorm with a device mismatch. Commit
`7fbb322` preserves `probe.device` throughout. Backward-compatible
(CPU no-op). The test suite didn't catch this because all context-
injection tests run on CPU.

**Lesson for future paper submissions**: anything tested only on
CPU can hide device assumptions. Worth adding a CI job that runs
a subset of tests on CUDA when available.

## Deliverables

```
notebooks/sprint19b_omniglot_colab.ipynb       (shipped c9d5ce6)
src/fractaltrainer/integration/context_mlp.py  (n_classes param, c9d5ce6)
src/fractaltrainer/integration/context_injection.py  (device fix, 7fbb322)
results/sprint19b_omniglot.json                (this run)
results/sprint19b_omniglot.png                 (plot)
results/sprint19b_omniglot_smoke.json          (smoke reference)
results/sprint19b_omniglot_smoke.png
Reviews/48_v3_sprint19b_omniglot_verdict.md    (this)
```

## Closing arc summary

- Sprint 19: validated scale to N=1000 — routing and signature
  space hold up. Pre-registered pass.
- Sprint 19b: attempted head-to-head vs meta-learning; lost by
  18.7 pp on Omniglot but context injection independently
  replicated at p=0.018 in a new dataset.

**The paper now has two external-benchmark validations: one
positive (scale), one negative (few-shot). Both publishable.**
