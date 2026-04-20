# v3 Sprint 13 — Snake Teacher: honest negative on behavior-clone ensembles

**Date:** 2026-04-20
**Reviewer:** Claude Opus 4.7 (1M context)

## Purpose

First non-MNIST application of the v3 Mixture-of-Fractals architecture.
Use the local Qwen2.5-Coder-7B LLM as a teacher: prompt it for K
diverse Snake strategies, behavior-clone each into an MLP expert,
register in FractalRegistry, then test whether ensembling these
cloned experts beats the best single expert — the Sprint 9a coverage-
compose claim reapplied to a sequential RL-adjacent domain.

**Headline finding:** **behavior-clone ensembles lose badly to the best
single clone.** The Mixture-of-Fractals architecture is not broken —
the individual cloned experts don't generalize, and ensembling weak
clones compounds rather than cancels their errors.

## Experimental setup

- **Environment:** custom deterministic 10×10 SnakeEnv with wall-death
  and self-death detection, max 300 steps per episode. 15/15 unit tests
  pass (`tests/test_snake_env.py`).
- **Teacher:** local Qwen2.5-Coder-7B-Instruct Q4_K_M via llama-server
  on 127.0.0.1:8080 (GTX 1050 / Vulkan / ~8 tok/s).
- **Strategies prompted (K=5):** `greedy_food`, `bfs_safe`, `wall_hugger`,
  `center_stayer`, `survival_first` — described in natural language,
  LLM replies with a Python `next_action(board, snake, food)` function.
- **Validation:** ast.parse → exec in sandboxed namespace → 30 training
  games each to collect (state, action) demos.
- **Behavior cloning:** 100→64→32→4 MLP, cross-entropy on (state,
  action), 200 epochs, Adam lr=1e-3.
- **Signature:** 50-probe canonical game-state batch, MLP softmax flat,
  200-dim signature → registered in FractalRegistry.
- **Evaluation:** 50 held-out games per condition (seed 5000+).

## LLM-strategy quality (demo-time)

Behavior of the raw LLM-generated functions during demonstration
collection:

| Strategy         | Avg survival | Avg score | Errors | Demos collected |
|------------------|-------------:|----------:|-------:|----------------:|
| greedy_food      |  159.2 steps |   19.13   |   0    |  4,776          |
| bfs_safe         |    4.8       |    0.80   |   0    |    144          |
| wall_hugger      |  300.0 (max) |    0.40   |   0    |  9,000          |
| center_stayer    |   10.0       |    0.13   |   0    |    300          |
| survival_first   |    6.0       |    0.07   |   0    |    181          |

Real behavioral diversity: `greedy_food` actually plays well (scores
19.13 food per 160-step game); `wall_hugger` survives the max 300
steps but barely eats (0.40 avg); the other three have buggy logic
and die quickly. All 5 strategies compiled + executed without runtime
errors — the LLM produced syntactically valid code every time.

## Behavior-cloned expert performance (50 held-out games)

The cloned MLPs evaluated individually:

| Clone              | Mean survival | Mean score | Max score |
|--------------------|--------------:|-----------:|----------:|
| greedy_food        |       7.18    |    0.42    |    2      |
| bfs_safe           |       6.38    |    0.12    |    1      |
| **wall_hugger**    |    **122.46** | **0.26**   |    2      |
| center_stayer      |       9.36    |    0.08    |    1      |
| survival_first     |       6.00    |    0.04    |    1      |

**Key observation:** the cloned `greedy_food` MLP survives only 7.18
steps and scores 0.42 food on average — DESPITE its source LLM
strategy scoring 19.13 food over 159 steps on the same board. A 27×
drop in food-eating. Same for `wall_hugger`: source survived 300 on
training games, clone drops to 122.46 on held-out games — still the
best clone, but well below source.

The MLP achieves loss 0.000–0.003 on the training demos (it memorizes
them perfectly), but **fails catastrophically on unseen board
configurations**. Classic behavior-cloning failure mode:
distribution-shift at test time and no inductive bias about 2D
spatial reasoning.

## Ensemble evaluation

| Condition                      | Survival | Score | Max |
|--------------------------------|---------:|------:|----:|
| Best single (wall_hugger)      | **122.46** | 0.26 | 2  |
| All-K ensemble (K=5 average)   |    7.96  | 0.04  | 1  |
| Random-3 ensemble              |    8.74  | 0.08  | 1  |

**Verdict: BEST SINGLE WINS. Δ(ensemble − best) = −114.50 survival
steps.** Adding the 4 weak clones to wall_hugger completely destroys
the one behavior that worked. The ensemble's action distribution
"averages out" wall_hugger's consistent wall-oriented moves with 4
chaotic signals, yielding incoherent play.

## Why this contradicts (and illuminates) Sprint 9a's result

Sprint 9a showed **coverage-compose beats top-1 by +8.1 pp on 27 binary
MNIST tasks.** Why doesn't it work here?

The key difference is **generalization of the individual experts**.

- **MNIST classifiers generalize within their class space.** Train on
  5,000 images, test on 1,000 held-out; the MLP has learned something
  real about the pixel patterns → ensembling genuinely complementary
  errors cancels out.
- **Snake behavior-cloning doesn't generalize.** The cloned MLP
  memorizes 30 games' worth of (state, action) pairs. On a 10×10 board
  with a snake of varying length and food in varying positions, the
  state space is essentially infinite. Behavior cloning captures
  surface-level patterns ("at these exact pixels, do this exact
  action") but not the underlying algorithm ("compute Manhattan
  distance to food"). When test games produce unseen states, the MLP
  outputs noise.

Sprint 9a's compose worked because experts made *correlated correct
predictions and uncorrelated errors* on the class-1 targets. Sprint
13's clones fail because experts make *essentially random predictions
on unseen states* — their errors aren't diverse-but-correctable, they
are *chaotic*. Averaging chaos is still chaos.

## The real finding: LLM strategies should be used directly, not cloned

The LLM-generated Python functions ALREADY generalize (they're
algorithms, not pattern-matchers). The cloned MLPs lose this property.

The fix isn't architectural — it's **skipping the behavior-cloning
step entirely**. The registry should hold:

- **Direct LLM strategies** (compiled Python functions) as the
  "experts," not behavior-cloned MLPs.
- **Signatures computed from strategy action-distributions** on a
  canonical probe batch — same methodology, just skip the training
  step.
- **Compose = majority-vote or weighted-vote across K strategies'
  action recommendations** for each game step.

This is a cleaner system: no training, no distribution shift, no
generalization gap. Each strategy is as good as the LLM that wrote it.

Proposed Sprint 14 (Snake Teacher v2):

```python
@dataclass
class LLMExpert:
    name: str
    strategy_fn: Callable        # compiled Python from LLM
    signature: np.ndarray        # action dist on probe batch

def decide(state, expert_triple):
    votes = [e.strategy_fn(board, snake, food) for e in expert_triple]
    # majority-vote action
    return Counter(votes).most_common(1)[0][0]
```

This would validate: can LLM strategies compose into stronger
ensembles when you skip the MLP intermediate? The prediction (from
Sprint 9a's mechanism) is yes — greedy_food + wall_hugger + a
third-way should score higher food than greedy alone AND survive
longer than wall_hugger alone.

## What the sprint legitimately proved

1. **End-to-end pipeline works.** Local LLM → strategies → demos →
   training → registry → eval. The whole stack exercises correctly,
   including `make_local_llm_client` wired in Sprint 13's prerequisites.
2. **LLM produces diverse strategies when prompted.** 5 distinct
   behaviors emerged: aggressive (greedy_food), defensive (wall_hugger),
   and three buggy variants. The LLM IS a useful teacher for
   generating a diverse expert pool.
3. **Behavior cloning of LLM strategies is a weak approach.** The
   MLP intermediate is the limiting factor. Sprint 14 should drop it.
4. **FractalTrainer's architecture isn't broken — it just needs the
   right experts.** The registry, routing, signature, and compose
   primitives all worked as designed. They will produce meaningful
   outcomes as soon as the experts themselves generalize.

## Limits

- **n=1 run, 5 strategies, 50 test games, one board size (10×10).**
  Conclusions are directional, not statistically definitive. A 3-seed
  repeat with different LLM temperatures would tighten confidence.
- **No hyperparameter tuning on the MLPs.** Possibly a different
  architecture (CNN, or larger MLP with positional encoding) would
  learn to generalize better. This sprint didn't explore that direction.
- **LLM temperature 0.5 produced fairly similar outputs.** Higher
  temperature (0.8-1.0) might produce more diverse strategies. Didn't
  sweep.
- **Behavior-cloning only saw ~3K-9K demos per strategy.** More games
  wouldn't help since the MLP has already memorized its training set
  (loss 0.000). The failure is structural, not data-starved.

## What ships

- `src/fractaltrainer/snake/env.py` — SnakeEnv, deterministic,
  seeded, 15 unit tests. Usable for any future Snake experiment.
- `src/fractaltrainer/snake/teacher.py` — LLM-driven strategy
  generation + demonstration collection. Reusable for Sprint 14's
  direct-LLM-expert approach.
- `src/fractaltrainer/snake/behavior_clone.py` — MLP policy training
  + ensemble evaluation. Still useful as a baseline to compare Sprint
  14's direct-strategy approach against.
- `scripts/run_snake_sprint.py` — end-to-end driver with `--llm
  local/cli/api` support.
- `results/snake_sprint.json` — full run artifacts.
- `results/snake_demos.npz` — cached LLM demonstrations (gitignored,
  regenerable via `--skip-llm` flag).

Full test suite: **176/176 passing** (was 161 + 15 new Snake env tests).

## Natural follow-ups

1. **Sprint 14 — LLM strategies as direct experts (no behavior cloning).**
   Keep the LLM-generated Python functions as the registry entries.
   Compute signatures from their action distributions directly. Ensemble
   via majority-vote or weighted-vote. Prediction: ensemble ≥ best single
   because the individual strategies now preserve their generalization.
2. **Sprint 15 — CNN policies for behavior cloning.** If the user wants
   to keep the behavior-cloning approach, swap the flat MLP for a small
   CNN that respects 2D board structure. Might close the train-test gap.
3. **Sprint 16 — Direct RL.** Policy-gradient training (REINFORCE or
   PPO) on Snake with the LLM strategies as initialization. This is
   how you'd actually surpass the LLM teacher — start with its
   knowledge, improve via reward signal.

## Paired ChatGPT review prompts

ChatGPT should validate:

1. Whether the **behavior-cloning generalization failure** is a
   well-known phenomenon or something specific to this setup. My
   read: it's the canonical covariate-shift / DAgger-style failure;
   the literature has known this since 2011 and it's why real RL
   uses either on-policy exploration or interactive demonstration
   collection. This sprint re-derived the phenomenon on a Snake
   toy but the finding isn't novel; it's just context for deciding
   whether the Mixture-of-Fractals architecture fits Snake.
2. Whether my **"skip behavior cloning, use LLM strategies directly"**
   proposal is the cleanest next step, or whether a proper RL
   fine-tune on top of the clones would be more productive. RL is
   more work but can exceed the teacher's ceiling; direct-strategy
   compose is cheaper but caps at max(individual LLM strategy quality).
3. Whether the **ensemble-loses-to-best** result should be reported as
   "compose doesn't transfer to Snake" or "compose of bad clones
   doesn't work, but compose of good experts (sprint 14) will." The
   two framings differ in their implications for the Mixture-of-
   Fractals vision.
