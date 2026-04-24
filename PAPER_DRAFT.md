# Mixture-of-Specialists Registry: a routed nearest-neighbor architecture with context injection at spawn

**Draft, 2026-04-24** (Sprint-19 external-validation update; renamed
from "Mixture-of-Fractals" at Sprint 18, see
`Reviews/42_v3_sprint18_fractal_audit.md` and
`Reviews/43_v3_sprint18_rename.md`). Primary author: Vegar Ratdal ·
vegarjr@users.noreply.github.com
Co-author: Claude Opus 4.7 (1M context, Anthropic).

## Abstract

We describe a **Mixture-of-Specialists Registry** (MoSR), a routed
mixture-of-experts architecture in which each entry is a standalone
classifier identified by a 1000-d softmax-probe signature vector.
(The repo retains the name *FractalTrainer* for historical
continuity — see Reviews 42–43 for the audit that motivated the
paper-level rename.) A query is routed by three-way
nearest-neighbor decision: **match** to a single nearest entry,
**compose** as an inverse-distance-weighted blend of the K nearest,
or **spawn** a new expert when no existing signature is close
enough. We introduce *context injection*: on spawn, the new expert's
first hidden layer is additively fused with the weighted mean of
its routing-nearest neighbors' penultimate activations on the
training batch. Across seven controlled ablations on MNIST,
Fashion-MNIST, and CIFAR-10 (54 `(ablation × budget)` datapoints),
context injection fits a linear gap rule
`(B − A) ≈ 0.139 × (ceiling − A)` with `R² = 0.37`, recovering
`α ≈ 0.15` of the accuracy gap on MNIST + CIFAR (null on
Fashion-MNIST, a clean conditional negative). We **replicate the
gap rule on Omniglot** 5-way 5-shot (`+3.0 pp`, `p = 0.008`,
`n = 100` episodes) — the primitive is real on two independent
datasets.

We then test MoSR against published baselines across three external
regimes. **(i) Scale.** Registry routing with a fixed-threshold
linear scan validates to **N = 1000 with 100 % top-1 routing**
(Split-MNIST, n=1250 experts, p95 latency 11.09 ms). The paper's
earlier "scale untested" limitation is resolved. **(ii) Few-shot.**
Plain MoSR spawn+context on Omniglot 5-way 5-shot loses to
MLP-ProtoNets by **18.7 pp** (p < 10⁻³¹); a *hybrid* configuration
that pairs a ProtoNets-meta-trained frozen encoder with MoSR's
registry heads recovers 12.6 pp (p < 10⁻¹⁹); swapping the softmax
head for a prototype head closes the remaining 5.6 pp
(Δ = 0.000 vs ProtoNets). **(iii) Continual learning.**
Split-CIFAR-50 with cross-domain encoder pretraining, MoSR hits
`0.131` vs replay `0.241`, naive `0.189`, joint-train `0.328` —
MoSR underperforms all baselines.

The unifying finding across these experiments is that MoSR's
accuracy depends almost entirely on the encoder's prior training.
With a meta-trained in-domain encoder (Omniglot), MoSR plus a
prototype head matches ProtoNets exactly. With a scratch or
cross-domain encoder (Omniglot vanilla, Split-CIFAR), MoSR trails
existing methods. The architecture's distinctive value is
therefore *operational* — monotonic expansion, independent
expert audit/swap, scale-free routing, compose-blending for
same-task queries — rather than raw accuracy in any single
regime. We also report a decisive negative on alternative
signatures: replacing softmax signatures with penultimate-
activation signatures collapses the within/cross gap, so routing
signatures must live in an *output-collapse* space. Two follow-on
primitives (depth-2 hierarchical registry; auto-spawn policy)
extend the architecture, and a full end-to-end demo runs with a
local Qwen2.5-Coder-7B describer. Code, six result artifacts
from the Sprint-19 external-validation arc, and the original
nine ablation artifacts are released at
vegarjr/FractalTrainer.

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

## 4. External validation across three regimes

Section 3 established context injection on MNIST, Fashion-MNIST, and
CIFAR-10 with seed-registry sizes N ≤ 15. This section tests MoSR
against published baselines on three external regimes that the
paper's §4 relates to but §3 did not measure: (i) scale, (ii)
few-shot head-to-head vs ProtoNets, (iii) continual learning on
Split-CIFAR-100. All experiments run from code in
`scripts/run_sprint19*.py` and `notebooks/scale_n1000_colab.ipynb`;
verdicts are documented in `Reviews/47_*.md` through `Reviews/52_*.md`.

### 4.1 Scale validation (Sprint 19)

We test whether the architecture's fixed-threshold linear-scan
routing survives at realistic deployment scale. Using the same
MNIST-subset task family as §3 but at **250 subsets × 4 seeds =
1000 registry experts** plus **250 held-out experts** for routing
queries, we evaluate at six registry sizes N ∈ {12, 50, 100, 250,
500, 1000} with subsample-by-subset (always 4 seeds per chosen
subset, so within-task pairs exist at every N). Thresholds are
`match = 5.0`, `spawn = 7.0` — the paper's §2 values, *not*
re-calibrated per N.

Headline numbers at N = 1000:

| Metric                                    | Observed | Pre-registered gate |
|-------------------------------------------|---------:|--------------------:|
| Top-1 routing accuracy                    | **1.000** | ≥ 0.9 ✓           |
| p95 per-query latency (linear scan)       | 11.09 ms  | < 50 ms ✓         |
| Within-task / cross-task distance gap     | +6.65     | > 0 ✓             |

The signature-space structure the paper reported at N = 15
(within-task mean 3.0, cross-task 10.7) is **stable across a 67×
scale expansion**: we observe within 3.67, cross 10.32 at N = 1000.
The paper's fixed thresholds remain operationally correct;
100 % of in-subset held-out queries route to a same-subset entry.
Linear-scan p95 latency grows from 0.17 ms (N = 12) to 11.09 ms
(N = 1000); the 50-ms budget extrapolates to roughly N ≈ 4,500
before an ANN index (FAISS, HNSW) would be advantageous. The
earlier "scale untested" limitation is resolved.

Run on Google Colab Pro T4, ~65 min wall time for 1,250 trainings.
Artifacts: `results/sprint19_scale_n1000.json`,
`Reviews/47_v3_sprint19_scale_verdict.md`.

### 4.2 Few-shot head-to-head: Omniglot 5-way 5-shot (Sprints 19b–19e)

The paper's §4 cites MAML (Finn et al., 2017) and Prototypical
Networks (Snell et al., 2017) as the nearest meta-learning analogs.
We run direct head-to-head comparisons on the canonical Omniglot
5-way 5-shot benchmark (1,200 background characters for pretraining
+ 423 evaluation characters for test-episode sampling, 5 support +
15 query examples per episode, 100 held-out episodes, shared 784 →
64 → 32 MLP backbone).

Five configurations:

| Configuration                | Encoder              | Classifier       |
|------------------------------|----------------------|------------------|
| pixel-kNN (baseline)         | none                 | 1-NN in 784-d    |
| vanilla spawn                | scratch per-episode  | softmax fc₃      |
| vanilla spawn + context (C)  | scratch per-episode  | softmax fc₃      |
| hybrid spawn + context       | ProtoNets-meta-trained, **frozen** | softmax fc₃ |
| **hybrid + prototype head**  | ProtoNets-meta-trained, **frozen** | prototype-distance |
| MLP-ProtoNets (reference)    | end-to-end meta-trained | prototype-distance |

Results (mean ± std, n = 100 episodes):

| Configuration                | Accuracy       | Δ vs ProtoNets |
|------------------------------|---------------:|---------------:|
| pixel-kNN                    | 0.666 ± 0.094  | −0.220         |
| vanilla spawn                | 0.662 ± 0.104  | −0.224         |
| vanilla spawn + context (C)  | 0.698 ± 0.108  | −0.187  (p < 10⁻³¹) |
| hybrid spawn + context       | 0.844 ± 0.071  | −0.056  (p < 10⁻⁷)  |
| **hybrid + prototype head**  | **0.900 ± 0.065** | **0.000 (Δ = 0 exactly)** |
| MLP-ProtoNets (reference)    | 0.900 ± 0.065  | —              |

We extract three quantitative findings:

**Finding F1: Context injection replicates on Omniglot.** The
paper's α ≈ 0.15 gap rule from MNIST + CIFAR reproduces on
Omniglot: vanilla(+C) vs vanilla(no-C) gives +3.6 pp at p = 0.018
(Sprint 19b). A second dataset, same sign, same magnitude, same
diagnosis — the primitive is real.

**Finding F2: Context injection is a weak-encoder rescue.** On
hybrid(+C) vs hybrid(no-C) — where the encoder is already
meta-trained — context adds +0.7 pp at p = 0.51 (Sprint 19c).
The primitive helps when the encoder is undertrained (vanilla)
but has no information to add when embeddings are already good.

**Finding F3: The encoder is the whole game.** Sprint 19b diagnosed
the 18.7 pp vanilla→ProtoNets gap as "the cost of not meta-training
the encoder." Sprints 19c–e confirm this: swapping the scratch
encoder for a meta-trained frozen encoder recovers 12.6 pp;
swapping the softmax head for a prototype head closes the
remaining 5.6 pp (**Δ = 0.000** vs ProtoNets — structurally exact,
not within noise). A separate ablation (Sprint 19e) shows
fine-tuning the encoder on 25 support examples at lr ∈ {1e-4, 1e-3}
changes accuracy by < 0.3 pp at p > 0.8 — strict freeze is the
correct choice, the encoder is not the bottleneck when a strong
classifier head is used.

Artifacts: `results/sprint19{b,c,d,e}_*.json`,
`Reviews/{45..51}_*.md`,
`notebooks/sprint19b_omniglot_colab.ipynb`,
`scripts/run_sprint19{c,d,e}_*.py`,
`src/fractaltrainer/integration/hybrid_head.py`,
`src/fractaltrainer/integration/prototype_head.py`.

### 4.3 Continual learning: Split-CIFAR-50 (Sprint 19f)

MoSR's monotonic-growth, no-overwrite design naturally targets
continual learning. We test it against two standard baselines
(naive sequential fine-tune, experience replay with K = 20
exemplars per task) and a joint-train upper bound on
Split-CIFAR-50 (first 50 CIFAR-100 classes permuted into 5
sequential tasks of 10 classes). All methods start from the same
CNN pretrained on the held-out 50 CIFAR-100 classes (the analog
to Omniglot background meta-training).

| Method               | Avg. accuracy (AA)  | Forgetting (F)         |
|----------------------|--------------------:|-----------------------:|
| naive_sequential     | 0.189               | +0.437 (large)         |
| experience_replay    | 0.241               | +0.361                 |
| **fractaltrainer**   | **0.131**           | +0.304 (routing drift, not parameter drift — see below) |
| joint_train (upper)  | 0.328               | 0                      |

MoSR's routing uses a nearest-task-centroid rule (iCaRL-style
mean-feature classifier; Sprint 19f restart after an initial
argmax-vote implementation failed). Heads once spawned are immutable
— *parameter* forgetting is structurally zero. The observed F
value of +0.304 is the product of routing drift as new task
centroids are added, not classifier-weight drift; the metric is a
standard CL definition and we report it unmodified for
comparability, but note its interpretation differs here.

MoSR underperforms every baseline. Diagnosis: the pretraining was
*cross-domain* (classes 50–99 for pretraining, 0–49 for CL targets),
which is insufficient to produce embeddings that discriminate the
target classes. Both nearest-centroid routing and per-task head
learnability suffer when the frozen encoder cannot separate target
classes in feature space. The contrast with Omniglot (§4.2) is
instructive: Omniglot pretraining was in-domain *episodic*
meta-training over 1,200 characters (same low-level distribution,
same task type), which produces embeddings that generalise to novel
N-way tasks on the same domain. No such in-domain meta-training
regime exists natively in Split-CIFAR-100.

The result establishes an *honest scope limitation*: MoSR is not a
drop-in replacement for iCaRL / A-GEM / Progressive Networks in
continual learning when only cross-domain pretraining is available.
Its value in the CL regime is its operational properties
(monotonic expansion, audit/swap, zero parameter forgetting) when
a well-adapted encoder is already in hand.

Artifacts: `results/sprint19f_continual.json`,
`scripts/run_sprint19f_continual.py`,
`Reviews/52_v3_sprint19f_continual_verdict.md`.

### 4.4 Unifying diagnosis: the encoder determines the ceiling

Across the six external sprints (19 scale, 19b few-shot plain,
19c hybrid, 19d prototype head, 19e fine-tune ablation,
19f CL), one pattern holds: MoSR's accuracy ceiling is set almost
entirely by the encoder's prior training. With an in-domain
meta-trained encoder (§4.2 prototype head, Δ = 0.000 vs
ProtoNets), MoSR is accuracy-equivalent to the classifier it's
built around. Without one (§4.2 vanilla, §4.3 cross-domain
pretrained), it trails standard methods by 18 pp (few-shot) or
5 pp (CL). The registry's contribution is therefore *operational*:
expandable expert inventory, signature-based routing, scale-free
linear scan, audit/swap, and compose-blending of same-task
entries. These are the paper's unique deliverables; accuracy
parity with existing methods is conditional on encoder quality.

## 5. Related work

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

## 6. Discussion and limitations

- **C is conditional, not universal.** Fashion's α = 0 is a worked
  example of when the primitive contributes nothing. We hypothesize
  this happens when (a) the task's small-sample ceiling binds before
  the full-data ceiling and (b) neighbors' feature detectors do not
  transfer usefully — but we have not isolated which of the two
  dominates. **Sprint 19c refines this:** context injection is
  specifically a *weak-encoder rescue* — it adds + 3.0 to +3.6 pp
  on registries with scratch-trained encoders (MNIST, CIFAR,
  Omniglot) but adds nothing (+0.7 pp, p = 0.51) when the encoder
  is meta-trained. The primitive is valuable, but it does not
  substitute for encoder quality.

- **Signatures are classification-specific.** The Direction B
  negative closes off the "drop in penultimate activations" path.
  Supporting regression, RL, or LM tasks requires probe-response
  signatures designed per task type, not a universal latent swap.

- **K = 3 is not tuned.** Experiments fixed K = 3 throughout (matching
  Sprint 11's compose default). A principled K-selection study is
  out of scope here.

- **Encoder is the bottleneck for competitive accuracy.** Across six
  external sprints, MoSR's accuracy ceiling is set by the encoder's
  prior training. With in-domain meta-training (Omniglot ProtoNets
  style), MoSR + prototype head is structurally equivalent to
  ProtoNets (Δ = 0.000 exactly). Without such training (Split-
  CIFAR-50 cross-domain pretraining), MoSR trails standard
  continual-learning methods by 5–11 pp. The registry adds
  *operational* capabilities (expandable expert inventory, audit/
  swap, compose-blending of same-task entries) on top of whatever
  encoder is available; it does not substitute for encoder quality.

- **Not a drop-in replacement for meta-learning or CL baselines.**
  §4.2 shows the plain MoSR architecture loses to ProtoNets by
  18.7 pp on Omniglot 5-way 5-shot; §4.3 shows it loses to
  experience replay by 11 pp on Split-CIFAR-50 with cross-domain
  pretraining. Users deploying MoSR in these regimes should plan
  on meta-training the encoder in-domain as a pre-step.

- **Scale validated to N = 1000, extrapolates to ~N = 4,500.**
  (Resolved from the prior draft's "scale untested" limitation.)
  Beyond the extrapolated linear-scan budget an ANN index (FAISS,
  HNSW) would be needed; swapping the router's scan for an ANN
  lookup is a straightforward drop-in.

## 7. Conclusion

We present a registry-based mixture-of-experts with three routing
primitives (match / compose / spawn), one novel training-time
extension (context injection), and two follow-on primitives (depth-2
hierarchical routing and autonomous compose-stream spawn proposals).
The central quantitative finding on §3 seed registries is that
context injection obeys a linear gap rule — across 54
`(ablation × budget)` datapoints on MNIST + CIFAR-10, context
recovers approximately 15 % of the accuracy gap between the
no-context baseline and the full-data ceiling — and the rule
replicates on Omniglot (§4.2 Finding F1: +3.0 pp, p = 0.008,
n = 100).

Against published baselines we report a mixed result that is
honest about the architecture's scope. The registry routing
validates to N = 1000 with zero accuracy loss (§4.1), resolving
the prior draft's "scale untested" limitation. On Omniglot 5-way
5-shot (§4.2), plain MoSR loses to MLP-ProtoNets by 18.7 pp; a
hybrid configuration with a meta-trained frozen encoder closes
most of that gap; swapping the softmax head for a prototype head
closes it exactly (Δ = 0.000 vs ProtoNets). On Split-CIFAR-50
continual learning (§4.3), MoSR with cross-domain pretraining
loses to experience replay by 11 pp. The unifying diagnosis (§4.4)
is that MoSR's accuracy ceiling is determined by the encoder's
prior training; the registry's distinctive contribution is
*operational* (expandable inventory, signature-based routing,
scale-free linear scan, audit/swap, compose-blending of same-task
entries) rather than raw predictive accuracy in any single regime.

Deployment recommendation: pair MoSR with an in-domain meta-trained
encoder (ProtoNets-style or equivalent) and a prototype head
(`PrototypeExpert`). This gives ProtoNets-equivalent accuracy on
few-shot tasks while adding the registry's operational layer. For
continual learning, the required in-domain meta-training phase
must be performed separately — cross-domain pretraining is not
sufficient. A paired negative result from §3.4 — latent-space
signatures collapse the within-cross task distance gap — stands
as a principled constraint on the router's design space:
signatures must live in an output-collapse space, not a generic
latent one.

The code, full ablation artifacts (§3), and external-validation
results (§4) are released; subsequent work can extend the
primitives without re-implementing the infrastructure. Remaining
scope includes non-classification signatures (§6), recursive
hierarchical depth ≥ 3, ANN routing beyond the ~N = 4,500
linear-scan envelope, and in-domain meta-training pipelines for
domains other than Omniglot.

## 8. Reproducibility

All code at https://github.com/vegarjr/FractalTrainer (public as of
2026-04-24). Key locations:

- Core integration module: `src/fractaltrainer/integration/`
  - `context_mlp.py` — MoSR's ContextAwareMLP and CNN
  - `context_injection.py` — gather_context, ContextSpec
  - `signatures.py` — softmax_signature and latent baselines
  - **`hybrid_head.py`** (Sprint 19c) — meta-trained frozen encoder + head
  - **`prototype_head.py`** (Sprint 19d) — PrototypeExpert + compose_prototypes
- Registry: `src/fractaltrainer/registry/fractal_registry.py`,
  `hierarchical_registry.py`
- Demo drivers:
  - `scripts/run_fractal_demo.py` (§3 MNIST/CIFAR seed-registry ablations)
  - `scripts/analyze_gap_rule.py` (§3.3 α fit)
  - **`notebooks/scale_n1000_colab.ipynb`** (§4.1 scale)
  - **`notebooks/sprint19b_omniglot_colab.ipynb`** (§4.2 Omniglot head-to-head)
  - **`scripts/run_sprint19c_hybrid.py`** (§4.2 hybrid)
  - **`scripts/run_sprint19d_prototype.py`** (§4.2 prototype head)
  - **`scripts/run_sprint19e_finetune.py`** (§4.2 fine-tune ablation)
  - **`scripts/run_sprint19f_continual.py`** (§4.3 Split-CIFAR-50 CL)
- Result artifacts:
  - `results/fractal_demo_*.json` (9 files, §3)
  - `results/sprint19_scale_n1000.{json,png}` (§4.1)
  - `results/sprint19b_omniglot.{json,png}` (§4.2)
  - `results/sprint19c_hybrid.{json,png}` (§4.2)
  - `results/sprint19d_prototype.{json,png}` (§4.2)
  - `results/sprint19e_finetune.{json,png}` (§4.2)
  - `results/sprint19f_continual.{json,png}` (§4.3)
- Verdicts:
  - `Reviews/33_v3_sprint17_*.md` through `Reviews/43_v3_sprint18_rename.md` (§3 + audit/rename)
  - `Reviews/46_v3_sprint19_colab_review.md` through
    `Reviews/52_v3_sprint19f_continual_verdict.md` (§4 external arc)

Tests: 46 + 221 = **267** pass locally (§3 + hybrid + prototype head +
existing suite). Demos run on CPU in ~6 min (§4.2 Omniglot),
~6 min (§4.3 Split-CIFAR-50); §4.1 scale test is ~65 min on
Colab Pro T4 (notebook, not script).

## Appendix A: Sprint chronology

### A.1 Sprint 17 cluster (§3 ablations)

| # | Commit | Contribution |
|---|---|---|
| 17 main | 53d95ba | F + C + B; first end-to-end |
| 17a | da4a24a | Eval-time context fix; C cold-start +5 pp confirmed |
| 17b | d638f07 | Sample-starved ablation + Direction B decisive negative |
| 17c | 39daa7a | Fashion-MNIST port + Qwen describer end-to-end |
| 17d | 8123eb8 | Data-starved ablation + gap rule proposed |
| 17e | 3dff930 | CIFAR-10 binary — gap rule cross-modal validation |
| 17f | d552bc9 | Quantitative gap rule fit (α ≈ 0.15) |
| 17g | 6f4441a | First paper draft |
| 17h | d0fe7cc | Direction D — AutoSpawnPolicy (self-spawning detector) |
| 17i | 3dcbfba | Self-spawn redundancy fix + Direction E hierarchical depth-2 |

### A.2 Sprint 18: fractal audit + rename

| # | Commit | Contribution |
|---|---|---|
| 18.1 | a3cc187 | Fractal audit: signature cloud D = 0.69, slope variance 0.53 — fails scale-invariance gate |
| 18.2 | e9aa43b | Paper-level rename to "Mixture-of-Specialists Registry" |

### A.3 Sprint 19 external-validation arc (§4)

| # | Commit | Contribution |
|---|---|---|
| 19 | 6539ee4 | **Scale**: N = 1000, top-1 = 1.000, p95 11.09 ms (§4.1) |
| 19b | 6f5a373 | Omniglot 5-way 5-shot head-to-head: vanilla MoSR loses 18.7 pp to ProtoNets; context injection replicates (+3.6 pp, p = 0.018) |
| gather_context fix | 7fbb322 | GPU-device bug caught mid-19b run; fix preserves probe.device |
| 19c | e277207 | **Hybrid**: meta-trained frozen encoder + softmax head recovers 12.6 pp of 18.7 pp gap |
| 19d | adcf1e8 | **Prototype head**: matches ProtoNets exactly (Δ = 0.000); `compose_prototypes()` tooling |
| 19e | 180f1fc | Fine-tune ablation: all unfreeze conditions NS vs strict freeze (p > 0.8); encoder is not the bottleneck |
| 19f | 7edd915 | **Continual learning**: Split-CIFAR-50 — MoSR underperforms baselines; cross-domain pretraining insufficient; honest scope limitation documented |
