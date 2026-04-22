"""v3 Sprint 17 — Direction D demo: self-spawning from compose-heavy stream.

Scenario: the seed registry (15 experts, 5 tasks × 3 seeds) receives
a stream of 20 queries, half of which are designed to land in the
compose band (labels that span multiple existing experts' class-1
sets). We run AutoSpawnPolicy alongside the pipeline and observe:

  1. How many compose verdicts accumulate
  2. When the policy triggers a proposal
  3. What task labels the proposal nominates
  4. Whether training on the proposal actually reduces future
     compose verdicts (the "did autonomous growth help?" question)

Smoke mode runs with a subset; full mode runs 20 queries and trains
one proposed expert.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fractaltrainer.integration import (  # noqa: E402
    AutoSpawnPolicy, ContextAwareMLP, ContextSpec,
    FractalPipeline, MockDescriber, OracleDescriber,
    QueryInput, ReclusterPolicy,
    evaluate_expert, spawn_baseline, spawn_with_context,
)
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402

from run_fractal_demo import (  # noqa: E402
    _build_seed_registry, _probe, _train_loader, _eval_loader, _sample_pairs,
    _probe_signature, _make_untrained_model, _model_factory_for,
)


# Tasks DESIGNED to land in the compose band between seed experts.
# Seed tasks: subset_01234 ({0-4}), subset_56789 ({5-9}),
#             subset_024 ({0,2,4}), subset_13579 ({1,3,5,7,9}),
#             subset_357 ({3,5,7}).
# Revised stream (Review 40): each bridge task has HALF its class-1
# labels from the low-digit seeds and HALF from the high-digit seeds,
# so the signature K=3 neighbors genuinely span DIFFERENT seed tasks.
# Design goal: force the label-set union to be a genuinely novel task
# (not a subset of any single seed's labels).
COMPOSE_BAND_TASKS = [
    # Tasks empirically known to land in the compose band (distance
    # 5-7 from seed experts). Each shares moderate overlap with
    # multiple seeds — enough to avoid match, not novel enough to
    # cleanly spawn. subset_02468 is the canonical compose-band task
    # from earlier sprints.
    ("bridge_02468_a", (0, 2, 4, 6, 8)),    # evens-all
    ("bridge_02468_b", (0, 2, 4, 6, 8)),    # repeated (different oracle seed)
    ("bridge_02468_c", (0, 2, 4, 6, 8)),    # repeated
    ("bridge_02468_d", (0, 2, 4, 6, 8)),    # repeated
    ("bridge_02468_e", (0, 2, 4, 6, 8)),    # repeated
    ("bridge_24678",   (2, 4, 6, 7, 8)),    # mid-even mix
    ("bridge_12678",   (1, 2, 6, 7, 8)),    # low+high mix
    ("bridge_02467",   (0, 2, 4, 6, 7)),    # evens+one-odd
    ("bridge_02468_f", (0, 2, 4, 6, 8)),    # repeated
    ("bridge_02468_g", (0, 2, 4, 6, 8)),    # repeated
]


def _oracle_signature(target, train_loader, probe, arch, n_oracle_steps=100):
    """Train a quick oracle classifier on the task and signature it."""
    torch.manual_seed(13)
    oracle = _make_untrained_model(arch)
    opt = torch.optim.Adam(oracle.parameters(), lr=0.01)
    it = iter(train_loader)
    for _ in range(n_oracle_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        logits = oracle(x, context=None)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return _probe_signature(oracle, probe)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--data-dir", default="results/data")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--arch", default="mlp")
    parser.add_argument("--seed-steps", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--match-threshold", type=float, default=5.0)
    parser.add_argument("--spawn-threshold", type=float, default=7.0)
    parser.add_argument("--n-probe", type=int, default=100)
    parser.add_argument("--probe-seed", type=int, default=12345)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--stream-size", type=int, default=10,
                        help="number of compose-band queries")
    parser.add_argument("--trigger-threshold", type=int, default=5)
    parser.add_argument("--train-proposal", action="store_true",
                        help="actually train the proposed expert + "
                             "re-measure compose rate on a second stream")
    parser.add_argument("--proposal-n-steps", type=int, default=500)
    parser.add_argument("--results-out", default="results/self_spawn_demo.json")
    args = parser.parse_args(argv)

    is_smoke = args.mode == "smoke"
    if is_smoke:
        args.seed_steps = min(args.seed_steps, 40)
        args.train_size = min(args.train_size, 512)
        args.stream_size = min(args.stream_size, 4)
        args.trigger_threshold = min(args.trigger_threshold, 2)
        args.proposal_n_steps = min(args.proposal_n_steps, 40)

    print("=" * 72)
    print(f"  SELF-SPAWN DEMO — mode={args.mode}  dataset={args.dataset}  "
          f"stream_size={args.stream_size}")
    print("=" * 72)

    print("\n[1/5] loading probe + seed registry...")
    probe = _probe(args.data_dir, args.n_probe, args.probe_seed, dataset=args.dataset)
    registry, models = _build_seed_registry(
        probe, args.data_dir, args.seed_steps,
        args.train_size, args.batch_size, verbose=True,
        dataset=args.dataset, arch=args.arch,
    )
    print(f"  registered {len(registry)} seed experts")

    describer = OracleDescriber()
    context_spec = ContextSpec(k=args.K, aggregation="weighted_mean")
    pipeline = FractalPipeline(
        registry, describer,
        match_threshold=args.match_threshold,
        spawn_threshold=args.spawn_threshold,
        compose_k=args.K, context_spec=context_spec,
        model_by_entry=models,
        model_factory=_model_factory_for(args.arch),
        recluster_policy=ReclusterPolicy(interval_spawns=99, trigger_at_end=True),
    )
    policy = AutoSpawnPolicy(
        trigger_threshold=args.trigger_threshold,
        k_neighbors=args.K,
        max_cluster_radius=None,
    )

    # ── Stream phase ──
    print(f"\n[2/5] streaming {args.stream_size} compose-band queries...")
    stream_records: list[dict] = []
    verdict_counts = {"match": 0, "compose": 0, "spawn": 0}

    for i, (name, labels) in enumerate(COMPOSE_BAND_TASKS[: args.stream_size]):
        print(f"\n  [{i+1}/{args.stream_size}] {name} (labels={labels})")
        train_ldr = _train_loader(
            labels, args.train_size, args.batch_size,
            args.data_dir, 9000 + i, dataset=args.dataset,
        )
        # Compute query signature via oracle
        q_sig = _oracle_signature(labels, train_ldr, probe, args.arch)

        # Run pipeline step without executing spawns (just classify verdict)
        pairs = _sample_pairs(labels, 20, args.data_dir, 9000 + i,
                                dataset=args.dataset)
        q = QueryInput(
            name=name, pairs=pairs, truth_labels=frozenset(labels),
            train_loader=train_ldr, probe=probe,
        )
        # Just decide — don't actually spawn, just observe
        decision = pipeline._decide(q_sig)
        verdict_counts[decision.verdict] = verdict_counts.get(decision.verdict, 0) + 1
        print(f"    verdict: {decision.verdict}  min_distance={decision.min_distance:.3f}")

        policy.observe(q_sig, decision)

        stream_records.append({
            "query_name": name, "labels": list(labels),
            "verdict": decision.verdict,
            "min_distance": float(decision.min_distance),
            "nearest_names": (
                [e.name for e in decision.retrieval.entries[:3]]
                if decision.retrieval else []
            ),
        })

    print(f"\n  verdict counts: {verdict_counts}")
    print(f"  compose verdicts accumulated: {policy.compose_count}")

    # ── Proposal phase ──
    print(f"\n[3/5] proposal check (threshold={args.trigger_threshold})...")
    if not policy.should_propose():
        print("  not enough compose verdicts — no proposal generated.")
        proposal = None
    else:
        proposal = policy.propose(registry)
        if proposal is None:
            print("  threshold met but proposal rejected (e.g., cluster too wide).")
        else:
            print(f"  PROPOSAL:")
            print(f"    neighbors: {[e.name for e in proposal.neighbor_entries]}")
            print(f"    proposed class-1 label union: "
                  f"{sorted(proposal.proposed_task_labels)}")
            print(f"    n_queries: {proposal.n_queries_in_region}")
            print(f"    mean distance to centroid: "
                  f"{proposal.mean_distance_to_centroid:.3f}")

    # ── Execute proposal (optional) ──
    second_stream_verdicts = None
    new_entry_name = None
    if proposal is not None and args.train_proposal:
        print(f"\n[4/5] training the proposed expert...")
        target = tuple(sorted(proposal.proposed_task_labels))
        train_ldr = _train_loader(
            target, args.train_size, args.batch_size,
            args.data_dir, 1234, dataset=args.dataset,
        )
        t0 = time.time()
        # Use the weighted-mean context from the neighbors for the new expert
        neighbor_models = [models[e.name] for e in proposal.neighbor_entries
                            if e.name in models]
        neighbor_dists = [
            float(np.linalg.norm(proposal.centroid_signature - e.signature))
            for e in proposal.neighbor_entries
        ]
        model, entry, stats = spawn_with_context(
            train_ldr, probe,
            neighbors=neighbor_models,
            neighbor_distances=neighbor_dists,
            spec=context_spec,
            context_scale=1.0,
            n_steps=args.proposal_n_steps,
            lr=0.01, seed=42,
            entry_name=f"autospawn_{'_'.join(str(d) for d in target)}",
            task=f"autospawn_union_of_{proposal.n_queries_in_region}_compose_queries",
            metadata_extra={
                "task_labels": list(sorted(target)),
                "proposed_by": "AutoSpawnPolicy",
            },
            model_factory=_model_factory_for(args.arch),
        )
        registry.add(entry)
        pipeline.model_by_entry[entry.name] = model
        models[entry.name] = model
        new_entry_name = entry.name
        print(f"    trained in {time.time() - t0:.1f}s, "
              f"final_loss={stats.final_loss:.4f}")
        print(f"    registered as: {entry.name}")

        # Re-stream the same compose-band queries and count verdicts
        print(f"\n[5/5] re-streaming to measure compose-rate reduction...")
        second_stream_verdicts = {"match": 0, "compose": 0, "spawn": 0}
        second_stream_records = []
        for i, (name, labels) in enumerate(COMPOSE_BAND_TASKS[: args.stream_size]):
            train_ldr = _train_loader(
                labels, args.train_size, args.batch_size,
                args.data_dir, 9000 + i, dataset=args.dataset,
            )
            q_sig = _oracle_signature(labels, train_ldr, probe, args.arch)
            decision = pipeline._decide(q_sig)
            second_stream_verdicts[decision.verdict] = (
                second_stream_verdicts.get(decision.verdict, 0) + 1)
            second_stream_records.append({
                "query_name": name, "labels": list(labels),
                "verdict": decision.verdict,
                "min_distance": float(decision.min_distance),
                "nearest": (
                    decision.retrieval.nearest.name
                    if decision.retrieval and decision.retrieval.nearest
                    else None
                ),
            })
        print(f"    verdict counts (after autospawn): {second_stream_verdicts}")

    # ── Save ──
    out = {
        "mode": args.mode,
        "config": vars(args),
        "seed_registry_size": len(registry),
        "stream_verdicts_before": verdict_counts,
        "stream_records_before": stream_records,
        "compose_count": policy.compose_count,
        "proposal": proposal.to_dict() if proposal else None,
        "executed_autospawn": new_entry_name,
        "stream_verdicts_after": second_stream_verdicts,
    }
    out_path = Path(args.results_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
