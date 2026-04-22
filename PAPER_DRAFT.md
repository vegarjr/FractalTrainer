# Mixture-of-Fractals: a routed registry of task-specialist experts with context-injection

**Draft, 2026-04-22.** Primary author: Vegar Ratdal · vegarjr@users.noreply.github.com
Co-author: Claude Opus 4.7 (1M context, Anthropic).

## Abstract

We describe FractalTrainer, a routed mixture-of-experts registry in
which each entry is a standalone classifier identified by a 1000-d
softmax-probe signature vector. A query is routed by three-way
nearest-neighbor decision: **match** to a single nearest entry,
**compose** as an inverse-distance-weighted blend of the K nearest,
or **spawn** a new expert when no existing signature is close
enough. We introduce *context injection*: on spawn, the new expert's
first hidden layer is additively fused with the weighted mean of
its routing-nearest neighbors' penultimate activations on the
training batch. Across seven controlled ablations (MNIST,
Fashion-MNIST, CIFAR-10; MLP and CNN; full-data and data-starved
regimes; 54 `(ablation × budget)` datapoints), context injection
fits a linear gap rule:

```
  (B − A) ≈ 0.139 × (ceiling − A),   R² = 0.37,
          α ≈ 0.15 on MNIST + CIFAR; α ≈ 0.00 on Fashion.
```

Context injection recovers about 15% of the available accuracy gap
on datasets where it works, with peak lift +5 pp in a 50-sample
MNIST regime. On Fashion-MNIST the primitive contributes nothing —
a clean null that we trace to the task's feature geometry. We also
report a decisive negative: replacing softmax signatures with
penultimate-activation signatures collapses the within-task /
cross-task distance gap, so signatures must live in an *output-
collapse* space rather than a generic latent space. Two follow-on
primitives extend the architecture: a hierarchical depth-2 registry
(task-branches × seed-leaves) that matches flat routing decisions
on 8/8 test queries at 1.39× lower latency, and an auto-spawn
policy that observes compose-verdict streams and proposes bridging
experts. All primitives run end-to-end with an LLM-based data
describer (local Qwen2.5-Coder-7B) that produces correct routing
decisions despite partially incorrect describer outputs. Code and
nine result artifacts are released at vegarjr/FractalTrainer.

## 1. Motivation

Mixture-of-experts (MoE) is typically formulated as a single model
with multiple sub-networks plus a learned gating function, trained
end-to-end. This paper addresses a different regime: a *registry*
of independently-trained experts, each specialized by supervised
pre-training on a task, with routing done by measuring distance in
a fixed-dimensional signature space computed post-hoc at expert
registration time. Each expert is a standalone classifier that can
be used independently; the registry supplies a dispatcher that
picks which one(s) to use for a given query, and a rule for when
to train a new one.

Three design requirements motivate the architecture:

1. **Expandable**: new experts are added without retraining existing
   ones. The registry grows monotonically.
2. **Cheap routing**: O(N) linear scan over signatures is fine at
   N ≤ 1000; no gating network needs training.
3. **Compositional at inference**: when no single expert is clearly
   the right one, multiple experts can be blended.

A longer-standing vision motivated the project ("each fractal is
specialized, together they help each other"). The paper tests
whether that vision can be operationalized.

## 2. Architecture

**Signature**: an expert's signature is the softmax output of its
final layer on a fixed 100-image probe batch, flattened to a
1000-dimensional vector. Distances between experts are L2 distances
in this space. Sprint 7b earlier established Spearman ρ = −0.85
between label-set Jaccard and signature distance on 27 MNIST binary
tasks, meaning the signature space already approximates label-
functional kinship.

**Router**: given a query signature `s`, find the K=3 nearest
entries by L2 distance. Let `min_d` be the distance to the nearest.
Two thresholds split the verdict space:
- `min_d ≤ match_threshold` → **match** (route to the single nearest)
- `match_threshold < min_d ≤ spawn_threshold` → **compose**
  (inverse-distance-weighted blend of K entries' outputs)
- `min_d > spawn_threshold` → **spawn** (train a new expert on the
  query data; register it)

Thresholds are calibrated per dataset from the within-task / cross-
task distance distribution. For MNIST MLP the gap between within-task
(mean 3.0) and cross-task (mean 10.7) distances is clean; we use
match=5.0, spawn=7.0 by default. For CIFAR CNN the gap is narrower
(2-3 range) and requires match=2.0, spawn=3.3.

**Context injection (the novel primitive)**: when the spawn verdict
fires, the new expert trains on the query data with an auxiliary
input lane. For each training batch `x`, context `c` is computed as
the inverse-distance-weighted mean of the K=3 nearest neighbors'
penultimate activations on `x`:
```
c = Σᵢ wᵢ · neighborᵢ.penultimate(x),   wᵢ ∝ softmax(−distᵢ)
c_norm = LayerNorm(c)
h₀ = Linear(in, 64)(x) + context_scale × Linear(32, 64)(c_norm)
```
Context is supplied at both training *and* evaluation time
(Sprint-17 follow-up "eval-time context fix"). Signatures for the
resulting entry are computed with `context=None` so the registry's
routing invariant is preserved.

**Closed loop**: a `FractalPipeline` orchestrator chains describer →
signature → router → action, with periodic reclustering of entries
by label-set Jaccard. The describer is an LLM-based perception layer
that turns raw `(data, label)` pairs into a task-identity prediction;
the router uses its own signature, not the describer's output, so
describer noise cannot misroute. End-to-end validation with a local
Qwen2.5-Coder-7B describer confirmed correct routing decisions on
three scripted queries despite the describer being wrong on one.

## 3. Ablations

### 3.1 Setup

Every ablation uses the same structure: a seed registry of 5 MNIST-
subset binary tasks × 3 seeds = 15 experts, plus three scripted
queries (Q_match, Q_compose, Q_spawn). The query designed to trigger
the spawn verdict gets a three-arm ablation:
- **A**: baseline spawn, no context (`context_scale=0`)
- **B**: K=3 nearest-neighbor context injection (the primitive)
- **C**: K=3 *random* context (control that isolates "context from
  routing" from "context from any source")

Each arm at budgets N ∈ {50, 100, 300, 500, 1000} and seeds
{42, 101, 2024}. Budgets count training steps; train_size is the
sample count.

Seven ablations were run:

| # | Dataset | Architecture | Regime | Notes |
|---|---|---|---|---|
| 1 | MNIST   | MLP | train-starved full (5k samples) | |
| 2 | MNIST   | MLP | train-starved cold (budgets 10-100) | |
| 3 | MNIST   | MLP | data-starved (50 samples, budgets 50-1000) | |
| 4 | Fashion | MLP | train-starved full | |
| 5 | Fashion | MLP | train-starved cold | |
| 6 | Fashion | MLP | data-starved (50 samples) | |
| 7 | CIFAR   | CNN | train-starved full | |
| 8 | CIFAR   | CNN | data-starved (50 samples, 3 queries) | |

Running the demo also validated Qwen describer end-to-end on MNIST.

### 3.2 Headline results

Peak context-injection lifts by ablation:

| Ablation | Peak (B−A) | Budget at peak |
|---|---:|---|
| MNIST train-starved cold   | **+4.2 pp @ N=25**  | mid-training |
| MNIST data-starved         | **+5.0 pp @ every N** | ceiling shift |
| Fashion train-starved      | ≤ noise (+0.9 pp) | n/a |
| Fashion data-starved       | 0.0 pp | n/a |
| CIFAR train-starved        | **+3.6 pp @ N=100** | mid-training |
| CIFAR data-starved Q_compose | **+4.6 pp @ N=50** | ceiling shift |
| CIFAR data-starved Q_spawn | 0.0 pp (task saturates at 0.72) | n/a |

**Arm C control**: across all MNIST ablations, random context tracks
the baseline (+0 to +1 pp), confirming the lift is routing-specific.
On CIFAR (CNN architecture), random context tracks closer to nearest
context (~1-2 pp gap), consistent with CNNs benefiting from any
non-zero first-hidden-layer perturbation as a mild regularizer.

### 3.3 The gap rule

Defining `gap = ceiling_A − A_achieved` where ceiling is the task's
empirical full-data accuracy, linear regression over 54
`(ablation × budget)` datapoints yields:

```
(B − A) = 0.139 × gap + 0.002,      R² = 0.37
```

Per-dataset α: MNIST 0.157, CIFAR 0.150, Fashion 0.000. MNIST and
CIFAR converge to α ≈ 0.15 (context recovers ~15% of the gap);
Fashion is a clean null. R² = 0.37 is meaningful but not tight —
about 2/3 of variance is dataset- and architecture-specific.

### 3.4 Latent-space signatures — a cautionary negative

We tested replacing softmax-probe signatures with penultimate-layer
activations to support non-classification regimes. Four variants
(raw, L2-normalized, row-softmax, and a baseline) were run on the
same 15-seed MNIST registry. Results:

| Signature mode | Within μ | Cross μ | Gap | Spearman ρ (Jaccard, dist) |
|---|---|---|---|---|
| Softmax (default)           | 2.50 | 9.92 | **+7.42** | **−0.945** |
| Penultimate (raw)           | 334  | 325  | −9.40 | −0.137 |
| Penultimate (L2-normalized) | 1.21 | 1.17 | −0.04 | −0.269 |
| Penultimate (row-softmax)   | 10.9 | 10.8 | −0.10 | −0.052 |

All three penultimate variants have **cross-task distance lower
than within-task distance**. Routing by L2 nearest-neighbor in these
spaces would misroute. Softmax does essential *task-identity
collapse* work that penultimate representations do not.

Implication: for non-classification regimes (regression, RL,
generation), signatures must be designed per task-type — there is no
universal latent-space drop-in. This is a cautionary finding against
"universal latent representations as routing objects".

## 4. Related work

**Sparse mixture-of-experts.** Shazeer et al. (2017) and the Switch
Transformer line (Fedus et al., 2021) route inputs to one of N
expert sub-networks via a learned gating network trained jointly
end-to-end. Experts are sub-blocks of a single monolithic model and
cannot be trained independently. Our setting differs on three
dimensions: (i) experts are standalone classifiers with their own
optimizers and objectives, trained per-task at registration time;
(ii) the "gate" is not learned — it is a fixed L2 nearest-neighbor
search over signature vectors, so adding an expert requires no
gradient step at the router; (iii) the registry is monotonically
expandable, with no parameter-sharing constraint between experts.
The trade-off is capacity density: a 1B-parameter MoE packs more
tasks per parameter than a 15-expert registry. The payoff is
operational: experts can be independently audited, swapped, or
retrained.

**Continual learning and expansion.** A-GEM (Chaudhry et al., 2019),
iCaRL (Rebuffi et al., 2017), and experience-replay methods all
address the forgetting problem: a model that learns task T₂ after
T₁ usually loses T₁'s competence. We side-step forgetting by
never overwriting: each task gets its own model, registered under
a stable signature. Progressive Networks (Rusu et al., 2016) also
add per-task columns, but with lateral weight-sharing between
columns; our registry is flat and has no weight-sharing. The spawn
verdict is conceptually close to Progressive Networks' "add a
column when the new task warrants one"; but where Progressive
Networks determines warrant manually, our router computes it from
signature distance.

**Model soups and ensembles.** Model soups (Wortsman et al., 2022)
average the parameters of multiple independently-trained models
into one to improve robustness. Our compose verdict is a similar
idea at inference rather than weight-merge time, gated by signature
distance. Where a model soup assumes the averaged models agree,
our compose assumes they disagree in useful ways and that their
inverse-distance-weighted blend over their softmax outputs
recovers task-appropriate behavior.

**k-NN language models and retrieval-augmented generation.** kNN-LM
(Khandelwal et al., 2020) and RETRO (Borgeaud et al., 2022) retrieve
cached *examples* at inference and mix them with a parametric
model's prediction. Our registry retrieves cached *experts* instead:
the "item" stored per entry is a trained classifier, not a
(context, next-token) pair. The routing structure (nearest-neighbor
in a signature space) is shared; what is retrieved is not.

**Meta-learning and episodic memory.** Prototypical Networks (Snell
et al., 2017) and MAML (Finn et al., 2017) operate under the
assumption that a new task is "close" to a meta-training distribution
and learn how to adapt quickly. Our registry does not meta-train;
it simply accumulates experts and treats each new task as independent
supervised learning, with routing doing the
"which-past-task-is-this-close-to?" work that meta-learning's inner
loop tries to compile into initialization. Context injection
(§2) is the closest analog to episodic memory — a new expert's
training input is *enriched* with activations from nearby past
experts, which is functionally similar to retrieving a few support
examples and concatenating them.

**Signatures as task identifiers.** The signature mechanism —
softmax output on a fixed probe batch — is related to dataset
cartography (Swayamdipta et al., 2020) in that it characterizes
what a model has learned through its outputs on a reference set.
The key difference is that cartography studies training dynamics
over a single model, while we use the final softmax as a persistent
task-identity vector. Our Direction-B negative finding (§3.4) is
specific to our registry setting and should not be read as a
comment on representation-learning methods more broadly.

**Novel contribution.** The context-injection primitive (§2) — *a
newly-spawning expert's first hidden layer is additively fused with
the inverse-distance-weighted mean of its routing-nearest
neighbors' penultimate activations on the query batch* — has, to
our knowledge, no direct precedent. The closest related idea is the
retrieval-augmented-training literature (Guu et al., 2020), where a
language model retrieves passages during pretraining; we retrieve
*classifier representations* during spawn, and we do so from
experts selected by the same router that handles inference-time
queries.

## 5. Discussion and limitations

- **C is conditional, not universal.** Fashion's α=0 is a worked
  example of when the primitive contributes nothing. We hypothesize
  this happens when (a) the task's small-sample ceiling binds before
  the full-data ceiling and (b) neighbors' feature detectors do not
  transfer usefully — but we have not isolated which of the two
  dominates.

- **Signatures are classification-specific.** The Direction B
  negative closes off the "drop in penultimate activations" path.
  Supporting regression, RL, or LM tasks requires probe-response
  signatures designed per task type, not a universal latent swap.

- **K=3 is not tuned.** Experiments fixed K=3 throughout (matching
  Sprint 11's compose default). A principled K-selection study is
  out of scope here.

- **Scale untested.** The registry tops out at N=15 in the main
  experiments; an earlier sprint reached N=93 but with a different
  signature pipeline. Behavior at N=1000+ (realistic deployment
  scale) is not validated.

## 6. Conclusion

We present a registry-based mixture-of-experts with three routing
primitives (match / compose / spawn), one novel training-time
extension (context injection), and two follow-on primitives (depth-2
hierarchical routing and autonomous compose-stream spawn proposals).
The central empirical finding is that context injection obeys a
quantified gap rule on the datasets where it works: across 54
(ablation × budget) datapoints on MNIST + CIFAR-10, context recovers
approximately 15% of the accuracy gap between the no-context
baseline and the task's full-data ceiling. A paired negative result
— latent-space signatures collapse the within-cross task distance
gap — carves out a principled limit on the router's design space:
signatures must live in an output-collapse space, not a generic
latent one.

The architecture is a reproducible framework for expandable expert
registries. Growth, specialization, composition, routing, and
context-enrichment — the five primitives the vision called for —
are each implemented and measured. Remaining scope includes
non-classification signatures (§5), recursive hierarchical depth
≥3, and scale beyond N=100 entries. The code and all ablation
artifacts are released; subsequent work can extend the primitives
without re-implementing the infrastructure.

## 7. Reproducibility

All code at vegarjr/FractalTrainer. Specifically:
- Integration module: `src/fractaltrainer/integration/`
- Demo driver: `scripts/run_fractal_demo.py`
- Gap-rule analysis: `scripts/analyze_gap_rule.py`
- Results artifacts: `results/fractal_demo_*.json` (9 files), `.png` plots
- Sprint logs: `Reviews/33_v3_sprint17_*.md` through `Reviews/38_*.md`

221/221 tests pass. Demo runs to completion on CPU within ~10 minutes
per ablation (MNIST/MLP) or ~25 minutes (CIFAR/CNN).

## Appendix A: Sprint cluster chronology

Sprints 17 main + follow-ups, in order of commit:

| # | Commit | Contribution |
|---|---|---|
| 17 main | 53d95ba | F + C + B; first end-to-end |
| 17a | da4a24a | Eval-time context fix; C cold-start +5pp confirmed |
| 17b | d638f07 | Sample-starved ablation + Direction B decisive negative |
| 17c | 39daa7a | Fashion-MNIST port + Qwen describer end-to-end |
| 17d | 8123eb8 | Data-starved ablation + gap rule proposed |
| 17e | 3dff930 | CIFAR-10 binary — gap rule cross-modal validation |
| 17f | d552bc9 | Quantitative gap rule fit (α ≈ 0.15) |
| 17g | 6f4441a | First paper draft |
| 17h | d0fe7cc | Direction D — AutoSpawnPolicy (self-spawning detector) |
| 17i | 3dcbfba | Self-spawn redundancy fix + Direction E hierarchical depth-2 |
