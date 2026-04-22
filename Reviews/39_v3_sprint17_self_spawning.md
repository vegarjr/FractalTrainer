# v3 Sprint 17 — Direction D: self-spawning fractals

**Date:** 2026-04-22
**Reviewer:** Claude Opus 4.7 (1M context)

## Purpose

The match/compose/spawn primitive handles each query in isolation,
but the **query stream** carries signal the single-query router
doesn't exploit: if many queries in a row land in the compose band
(same K neighbors blended), there's a gap in the registry that the
stream is asking to be filled. This sprint implements **Direction D**
from the brainstorm — `AutoSpawnPolicy`, which observes compose
verdicts and proposes a bridging expert whose class-1 labels union
the neighbors' label sets, without requiring external supervision
for the proposed task.

## Mechanism

```python
policy = AutoSpawnPolicy(trigger_threshold=5, k_neighbors=3,
                         max_cluster_radius=None)
for query_signature, decision in stream:
    policy.observe(query_signature, decision)   # only recorded when compose
    if policy.should_propose():
        proposal = policy.propose(registry)
        if proposal is not None:
            # caller decides: train it, or reject it
            ...
        policy.reset()
```

The proposal consists of:
- **centroid_signature** — mean of observed compose signatures
- **neighbor_entries** — K entries nearest the centroid
- **proposed_task_labels** — union of neighbors' class-1 label sets
- **mean/max_distance_to_centroid** — cluster tightness

No new labeled data is required: the registry proposes its own next
training task from the gap structure.

## Demo

`scripts/run_self_spawn_demo.py` runs a seed registry of 15 experts,
then streams 10 "bridge" tasks designed to land in the compose band
(labels that span multiple existing seeds' class-1 sets but match
none exactly — e.g. `bridge_0124` is {0,1,2,4} vs seeds {0,1,2,3,4},
{0,2,4}, {1,3,5,7,9}, etc.).

### Results

**First stream** (10 bridge tasks with labels spanning multiple seeds):

| Verdict | Count |
|---|---|
| match   | 3 |
| compose | 5 |
| spawn   | 2 |

Trigger threshold (5 compose) met → proposal generated.

**Proposal:**
- K=3 neighbors: `[subset_01234_seed2024, subset_01234_seed42, subset_01234_seed101]`
  — all three seeds of the **same** existing task
- Proposed class-1 label union: `{0, 1, 2, 3, 4}` — identical to `subset_01234`
- Mean distance to centroid: 3.954

**Executed** the proposal (trained a new expert on {0,1,2,3,4} with
context-injection from the three subset_01234 neighbors). The expert
registered as `autospawn_0_1_2_3_4`, trained in 7.7s to final loss
0.052.

**Re-stream after autospawn:**

| Verdict | Before | After |
|---|---|---|
| match   | 3 | 3 |
| compose | 5 | **6** |
| spawn   | 2 | 1 |

The autospawn **did not reduce** the compose rate. It actually
increased by one (5 → 6), because a query that previously spawned
now falls in the compose band — the new expert moved one verdict,
but not in the direction the policy intended.

## Interpretation

**The policy fires correctly.** Detection is working: 5 compose
verdicts clustered in a tight region (mean distance to centroid
3.954), the K=3 nearest entries to the centroid were identified
correctly, and the label-set union was computed.

**The union heuristic failed on this stream.** All 5 compose
queries' signatures clustered near `subset_01234`, so the K=3
nearest were three *seeds of that same task*. Their label-set
union equals `subset_01234`'s own class-1 set — the proposal
duplicates an existing task.

**Why the cluster landed on one task, not between tasks.** The
bridge_* tasks were designed with labels spanning multiple seeds
(e.g., `bridge_0125` = {0,1,2,5} spans subset_01234 and subset_56789).
But their *signatures* — computed on the fixed MNIST probe batch
after 100 oracle training steps — cluster nearest to whichever seed
task shares the most class-1 digits. Tasks with overlap-heavy label
sets ({0,1,2,5} has 3 digits of 5 in subset_01234, 1 in subset_56789)
look closer to the dominant parent. The signature geometry doesn't
honor the "bridging" intent the label sets encode.

**What the policy actually detects.** It detects **query-load
clusters** around existing experts, not novel bridging regions.
That's useful information — it tells you which existing experts are
most-queried via compose — but it's **not autonomous novel-task
discovery** without further filtering.

**Refinement (future work):** before emitting a proposal, the policy
should check that the union label set is *distinct* from any single
nearest neighbor's label set. If `union = nearest_entry.task_labels`
(or a subset thereof), the cluster is query-load not gap-discovery
and the proposal should be suppressed. Pseudo-code:

```python
if proposal.proposed_task_labels <= proposal.neighbor_entries[0].task_labels:
    # union is already covered by the nearest neighbor — redundant
    return None
```

This would have suppressed the demo's proposal and avoided training
a redundant expert.

**What this means for the vision.** Self-spawning as an idea works —
the plumbing is validated, the tests cover the edge cases, and the
demo runs end-to-end. But the specific **union heuristic for
proposal labels is too eager**. True bridging-task discovery needs
either (a) a diversity check like the one above, (b) signatures that
encode label-set geometry more literally (an open problem from
Review 34 Direction B), or (c) an explicit "spawn when K nearest
are from different tasks" trigger rather than "spawn when K
compose-verdicts accumulate".

This is an honest partial win: AutoSpawnPolicy is useful as a
**detector** for registry-coverage gaps but not yet as an
**autonomous grower**. The paper should treat it as an observed
primitive with documented limitations, not a solved problem.

## What ships

- `src/fractaltrainer/integration/self_spawn.py` — `AutoSpawnPolicy` +
  `SpawnProposal` dataclass
- `src/fractaltrainer/integration/__init__.py` — exports extended
- `tests/test_self_spawn.py` — 9 unit tests (threshold, label-union,
  cluster-radius rejection, reset, `observe` delegation, empty
  registry, JSON serialization)
- `scripts/run_self_spawn_demo.py` — end-to-end demo with optional
  `--train-proposal` flag that actually trains the proposed expert
  and re-streams to measure compose-rate reduction
- `results/self_spawn_demo.json`
- `Reviews/39_v3_sprint17_self_spawning.md` — this doc

**230/230 tests passing** (221 pre-existing + 9 new).

## Paired ChatGPT review prompts

1. **On the "proposed_task_labels = union" heuristic.** Taking the
   union of K neighbors' class-1 sets is a natural bridging
   definition but it assumes the query stream's true labels are
   *inside* that union. In practice some compose queries have labels
   outside the union (e.g. `bridge_0345` has label 5 which is in
   subset_56789 but may not be one of the top-K neighbors). Is a
   union-based bridging task the right formulation, or should the
   policy also consider intersection, symmetric difference, or a
   learned aggregation?
2. **On reset-after-propose.** The policy clears its buffer after
   proposing. This means sequential compose clusters are treated
   independently. In a production stream that's correct (each
   proposal executes or is rejected, then the counter starts fresh),
   but it loses information — e.g., a cluster that keeps regenerating
   over time could mean a task deserves multiple specialists. Worth
   tracking, or overkill?
3. **On whether this closes the "autonomous growth" vision leg.**
   AutoSpawnPolicy proposes tasks; the caller still has to execute.
   A fully-autonomous system would also decide to execute, train, and
   integrate without human intervention — but then we need an
   acceptance criterion (what prevents runaway spawning?). Is "always
   execute the proposal" fine, or does autonomous growth need a
   cost-benefit gate?
