# v3 Sprint 19d — prototype head verification (tautological, but useful)

**Date:** 2026-04-24
**Platform:** local CPU
**Wall time:** 133 s (~2 min) full run
**Decision:** Accept with honest framing. Sprint 19d is a **verification
+ tooling sprint**, not a research sprint. The compose-prototype
primitive is now in place for a future in-distribution benchmark.

## Numbers (100 Omniglot 5-way 5-shot episodes)

| Method             | Accuracy      | Δ vs ProtoNets |
|--------------------|--------------:|---------------:|
| hybrid-softmax (19c) | 0.837 ± 0.079 | −0.063 |
| **hybrid-prototype (new)** | **0.900 ± 0.065** | **0.000 exactly** |
| MLP-ProtoNets | 0.900 ± 0.065 | — |

Welch t-test hybrid-prototype vs ProtoNets: Δ=0.0000, t=0.00, p=1.0000.

## Why Δ is exactly zero

This is not noise; it's structural. `PrototypeExpert.forward(x) =
−cdist(encoder(x), prototypes) / temperature` while
`protonet_eval_episode` does `−cdist(encoder(qx), prototypes)`. Both
use:

- the same meta-trained encoder weights
- prototypes computed from the same support examples the same way
  (per-class mean of support embeddings)
- argmax of the same logits (the `/temperature` scaling is invariant
  to argmax)

Predictions are identical on every episode → accuracies are
identical to numerical precision.

## What this confirms (useful)

1. **Sprint 19c's diagnosis was exactly right.** The 5.6 pp residual
   gap (hybrid-softmax 0.844 vs ProtoNets 0.900) was 100 % the softmax-
   head-vs-prototype-classification mismatch. Not a registry artifact.
   Swap the classifier and the gap closes to zero.

2. **The registry layer is zero-cost.** Wrapping ProtoNets'
   prototype classification inside a FractalRegistry-compatible
   `PrototypeExpert` produces identical numbers. The registry adds
   *operational* capabilities (signatures, routing, storage, audit,
   swap) without any accuracy penalty.

3. **Recommendation for paper §deployment section:** the production-
   ready MoSR configuration is meta-trained encoder + prototype-expert
   registry entries. Matches ProtoNets on few-shot while keeping the
   registry's expansion/audit properties.

## What this does NOT demonstrate (honest)

1. **Compose-prototype blending** — the new `compose_prototypes()`
   primitive that blends K nearest experts' prototype tensors is
   implemented but **not tested** by this sprint. Few-shot eval on
   Omniglot uses novel characters; no registry entry can be for the
   same task, so compose never fires during evaluation.
2. **Variance reduction from registry** — the proposed benefit
   ("same-task registry entries provide additional samples, averaging
   them reduces prototype-estimation noise") is intuitive but
   unverified. A proper test needs multi-episode same-character-set
   queries.
3. **Novel research contribution** — Sprint 19d is tooling +
   verification. The numerical result (Δ=0.000) is a tautology because
   I implemented the same classifier in two ways and compared them.

## Assessment: accept with caveat

This sprint verifies the 19c diagnosis but adds no new empirical
claim. Framing it as a "research win" would be dishonest. Framing it
as "tooling + verification" is correct — the `PrototypeExpert` class
and `compose_prototypes()` helper are building blocks for future
in-distribution compose experiments, and the verification result is a
useful paper sentence ("ProtoNets-equivalent accuracy when using
prototype classification inside the registry").

## Deliverables

```
src/fractaltrainer/integration/prototype_head.py  (~160 LoC)
tests/test_prototype_head.py                       (9 tests)
scripts/run_sprint19d_prototype.py
results/sprint19d_prototype.json                   (this run)
results/sprint19d_prototype.png
results/sprint19d_prototype_smoke.json
results/sprint19d_prototype_smoke.png
Reviews/50_v3_sprint19d_prototype_verdict.md       (this)
```

Test suite: 46/46 pass (37 pre-existing + 9 new prototype tests).

## What's next

- **Sprint 19e**: encoder fine-tuning. Does low-lr unfreeze of fc2
  during per-episode training help? Tests whether strict freeze is
  too restrictive.
- **Sprint 19f**: Split-CIFAR-100 continual-learning benchmark vs
  iCaRL / A-GEM. FractalTrainer's *native* regime.
- **Sprint 19g (optional)**: compose-prototype in-distribution benchmark.
  Multi-episode same-character-set setup. Only worth doing if 19e/19f
  don't yield a stronger finding.
