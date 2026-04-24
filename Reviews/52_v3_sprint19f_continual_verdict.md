# v3 Sprint 19f — Split-CIFAR-50 continual: FractalTrainer underperforms all baselines

**Date:** 2026-04-24
**Wall time:** 331 s full, 135 s smoke (local CPU)
**Decision:** Accept clean negative, with proper scope limitation.
**One restart executed** — first pass exposed argmax-vote routing bug
(task 0 wins all queries due to better calibration). Fixed to
nearest-centroid routing + added held-out-class pretraining to give
FractalTrainer a fair shot. Second restart confirmed honest finding.

## Setup

- **Benchmark:** Split-CIFAR-50 continual. First 50 classes of CIFAR-100
  permuted into 5 sequential tasks of 10 classes each. Standard CL
  eval protocol.
- **Pretraining:** CNN jointly trained on held-out CIFAR-100 classes
  50–99 for `n_epochs` (analog to Omniglot background meta-training,
  but cross-domain — different class set than CL targets).
- **Shared CNN:** 3-conv-block small net, 128-d features, used as
  starting point by *all* methods.
- **FractalTrainer routing:** nearest-centroid task id (iCaRL-style) →
  then per-task head for within-task classification. Replaced naive
  argmax-vote after it failed catastrophically in the first pass.
- **300 samples/class × 5 epochs/task** (full); CPU ~5 min total.

## Numbers (Split-CIFAR-50, 5 tasks × 10 classes)

| Method              | AA    | F (forgetting)  | per-task final accs                |
|---------------------|------:|---------------:|:-----------------------------------|
| naive_sequential    | 0.189 | +0.437 (huge) | 0.12 0.08 0.10 0.06 0.58           |
| experience_replay   | 0.241 | +0.361        | 0.19 0.15 0.16 0.12 0.58           |
| **fractaltrainer**  | **0.131** | +0.304*  | 0.20 0.12 0.11 0.04 0.17           |
| joint_train (upper) | 0.328 | +0.000 (N/A)  | 0.39 0.29 0.27 0.25 0.44           |

\* FractalTrainer's "F" conflates two things: parameter forgetting
(which is structurally zero — heads never update after spawn) and
routing-quality drift as more registry entries come online. The
observed 0.304 is routing noise, not real forgetting.

## Ranking vs pre-registered verdicts

| Prediction | Observed | Met? |
|---|---|---|
| naive has F > 0.3 | +0.437 | ✓ |
| FractalTrainer has "F" ≈ 0 | +0.304 | ✗ (metric pollution, see above) |
| all methods < joint_train | ✓ | ✓ |
| **FractalTrainer AA > replay AA** | **-0.110 worse** | ✗ |

## Why FractalTrainer underperforms

**Cross-domain pretraining is not enough.** The CNN features learned
from classes 50–99 don't discriminate classes 0–49 well. This breaks:

1. **Nearest-centroid routing**: task centroids (means of task samples
   in feature space) are poorly separated because the encoder's
   manifold wasn't trained to separate these particular classes.
2. **Per-task heads**: training a 10-way linear classifier on top of
   bad features converges near chance; the worst task (task 3, acc
   0.04) shows the frailty.

By contrast:
- **naive** keeps updating the CNN on each new task, so features
  drift toward whatever task was trained last — that's why task 4
  (last-trained) gets 0.58 while earlier tasks collapse.
- **experience_replay** updates the CNN with both new-task and K=20
  exemplars per old task — trades forgetting for worse single-task
  accuracy but higher average.
- **joint_train** sees all 50 classes at once — its features are
  by construction good for the target distribution.

## Contrast with Sprint 19c (Omniglot)

Sprint 19c's hybrid matched ProtoNets within 5.6 pp because:
- Omniglot has 1,200 background characters → ample meta-training
  variety.
- Meta-training used ProtoNets-style *episodic* supervision, which
  produces embeddings that generalize to *novel* N-way tasks.
- Target (evaluation) and pretrain characters come from the same
  low-level distribution (Omniglot binary character images).

Sprint 19f's pretraining had none of these advantages:
- Only 50 held-out classes (small variety).
- Joint supervision on a closed 50-class problem, not meta-learned
  for generalization.
- Target and pretrain classes are disjoint semantic sets (e.g.,
  vehicles vs animals) within CIFAR-100 — more domain shift than
  Omniglot's within-script characters.

**Conclusion**: FractalTrainer's registry architecture works when the
encoder is *meta-trained for generalization* on the target domain.
Cross-domain pretraining is insufficient for continual learning when
significant semantic shift exists between pretrain and target class
sets.

## Restart decision

**One restart executed** (out of two problems in first pass):

1. **Routing bug fix** (first pass → restart): the first implementation
   used argmax-of-concatenated-softmax to pick a head. Task 0's head
   was consistently more confident → every query routed there. Per-
   task accs were `(0.15, 0, 0, …, 0)`. Fixed to nearest-feature-
   centroid routing (iCaRL's standard).
2. **Encoder pretraining** (also in restart): added held-out-class
   pretraining so FractalTrainer doesn't rely on task-0-only features.
   Improved fractal AA from 0.089 → 0.131 at FULL, but still below
   all baselines.

**No further restart**: the finding is clean and matches the Sprint
19c narrative. Another iteration would require implementing
ProtoNets-style meta-learning on CIFAR episodes — a separate sprint
(19g) if worth pursuing.

## What this tells the paper

Honest paper-level statement:

> *"On Split-CIFAR-50 continual learning (5 sequential tasks of 10
> classes, pretrained CNN from held-out 50 classes, 300 samples/class,
> 5 epochs/task), MoSR's frozen-encoder-plus-per-task-heads registry
> achieves 0.131 average accuracy — below naive fine-tune (0.189) and
> experience replay (0.241). Joint training reaches 0.328.  The
> mechanistic gap: cross-domain pretraining does not produce
> embeddings that discriminate target-task classes well enough for
> frozen-encoder classification. MoSR's zero-parameter-forgetting
> (architecturally guaranteed by registry-immutability) does not
> translate to competitive accuracy without in-domain meta-training
> of the encoder, consistent with the Sprint 19c Omniglot result."*

Paper-level consequence: remove any claim that MoSR is a drop-in
replacement for iCaRL / A-GEM in continual learning. It's not. The
architecture is valuable for its *operational properties* (expert
audit, swap, compose, monotonic growth) in domains where a good
encoder is already available.

## Five-sprint Omniglot arc + CIFAR check

| Sprint | Finding |
|---|---|
| 19 | Scale to N=1000 with 100 % top-1 routing (no competitor). |
| 19b | Plain MoSR loses 18.7 pp to ProtoNets (encoder from scratch). |
| 19c | Hybrid (meta-trained encoder) recovers 12.6 pp. |
| 19d | Prototype head closes the remaining 5.6 pp (matches ProtoNets). |
| 19e | Encoder fine-tuning doesn't help; strict freeze correct. |
| **19f** | **Continual learning on CIFAR: MoSR loses to naive/replay without in-domain meta-training.** Confirms encoder-primacy diagnosis. |

Consistent thread across six sprints: **the encoder is the whole
game.** Give MoSR a good meta-trained encoder → ProtoNets-level
accuracy. Give it anything less → struggles.

## Deliverables

```
scripts/run_sprint19f_continual.py          (new)
results/sprint19f_continual.json            (this run)
results/sprint19f_continual.png             (plot)
results/sprint19f_continual_smoke.json
results/sprint19f_continual_smoke.png
Reviews/52_v3_sprint19f_continual_verdict.md (this)
```

## What's next

Options ranked by expected payoff vs effort:

1. **Stop and consolidate paper.** Five Omniglot sprints + one CIFAR
   CL check is already a compelling paper-level story with both
   positive and negative results. Writing time is higher-value than
   more experiments.
2. **Sprint 19g (optional)**: ProtoNets-style meta-learning on CIFAR
   sub-episodes for FractalTrainer's encoder. If it recovers the gap,
   strengthens the "meta-training matters" claim. ~3 hours of work.
3. **Skip further sprints**: the paper can honestly state the scope
   (few-shot and scale work, CL requires in-domain meta-training)
   without further validation.

Recommend option 1 unless the user explicitly wants the CL story
strengthened.
