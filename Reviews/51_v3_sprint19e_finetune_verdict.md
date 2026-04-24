# v3 Sprint 19e — encoder fine-tuning verdict: strict freeze is correct

**Date:** 2026-04-24
**Wall time:** 206 s (~3.5 min) on CPU
**Decision:** Accept clean negative. Strict freeze (19c) is the right
choice. Fine-tuning adds nothing.

## Numbers (100 Omniglot 5-way 5-shot episodes)

| Condition       | fc1 frozen | fc2 frozen | encoder lr | Accuracy        |
|-----------------|:----------:|:----------:|:----------:|----------------:|
| strict_freeze   | ✓          | ✓          | —          | 0.832 ± 0.083   |
| ft_fc2_low      | ✓          |            | 1e-4       | 0.833 ± 0.083   |
| ft_fc2_mid      | ✓          |            | 1e-3       | 0.834 ± 0.081   |
| ft_all_low      |            |            | 1e-4       | 0.835 ± 0.083   |
| ProtoNets (ref) | n/a        | n/a        | end-to-end | 0.900 ± 0.065   |

## Welch's t-tests vs strict_freeze

| Condition | Δ | t | p |
|---|---:|---:|---:|
| ft_fc2_low   | +0.0007 | +0.06 | 0.955 (NS) |
| ft_fc2_mid   | +0.0020 | +0.17 | 0.865 (NS) |
| ft_all_low   | +0.0028 | +0.24 | 0.813 (NS) |
| **ProtoNets** | **+0.0675** | **+6.37** | **< 10⁻⁸** |

## What this tells us

Fine-tuning the meta-trained encoder on 25 support examples per episode
does **nothing** — all three unfreeze conditions are within 0.3 pp of
strict freeze at p > 0.8. The ~6.7 pp gap to ProtoNets is unchanged.

Two reasons this is expected in retrospect:

1. **ProtoNets already optimized the encoder for few-shot.** It was
   meta-trained across thousands of episodes to produce embeddings
   tuned for prototype-distance classification. Any 25-example
   adjustment is either too small (low-lr) to matter or wrong
   direction (over-fits the support batch and hurts query
   generalization).

2. **The encoder is not the bottleneck.** Sprint 19c's hybrid-softmax
   at 0.832 and all four Sprint 19e conditions at 0.832–0.835 sit on
   the same plateau. Sprint 19d showed that swapping the softmax head
   for prototype classification instantly gives 0.900 (matches
   ProtoNets). The gap is the classifier choice, not the encoder.

## Implication for paper

Strict freeze (hybrid encoder + trainable head) is correct. When the
head is softmax, accuracy plateaus at 0.832. When the head is
prototype-distance, accuracy matches ProtoNets at 0.900.

**Revised paper recommendation** (supersedes 19c's):

> *For production MoSR deployment with a meta-trained shared encoder,
> use prototype-distance classification (via `PrototypeExpert`) rather
> than a trainable softmax head. Strict encoder freeze is the correct
> setting — fine-tuning the encoder on support-set examples does not
> help (Sprint 19e: all unfreeze conditions within 0.3 pp, p > 0.8)
> and a softmax head plateaus 6.7 pp below the prototype-head
> alternative.*

## Relation to Sprints 19 / 19b / 19c / 19d

Five external Omniglot validations now in place, all consistent:

| Sprint | Finding |
|---|---|
| 19 | Scale validated to N=1000. |
| 19b | Vanilla spawn loses 18.7 pp to ProtoNets (encoder from scratch). |
| 19c | Meta-trained frozen encoder closes 12.6 pp of that gap. Context injection helps vanilla (+3.6 pp) but not hybrid. |
| 19d | Swap softmax head for prototype-distance → closes remaining 5.6 pp. |
| **19e** | **Fine-tuning the frozen encoder does not help (all conditions NS vs strict freeze). Strict freeze was the correct choice.** |

## Deliverables

```
scripts/run_sprint19e_finetune.py          (new)
results/sprint19e_finetune.json            (this run)
results/sprint19e_finetune.png             (plot)
results/sprint19e_finetune_smoke.json      (smoke ref)
results/sprint19e_finetune_smoke.png
Reviews/51_v3_sprint19e_finetune_verdict.md  (this)
```

No new source module. Tests: 46/46 still passing.

## What's next

- **Sprint 19f**: Split-CIFAR-100 continual learning vs iCaRL / A-GEM.
  FractalTrainer's *native* regime — the registry's monotonic
  expansion and per-task specialization are the main selling points
  for continual learning, not few-shot.
