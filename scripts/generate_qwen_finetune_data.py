"""v3 Sprint 16c — Generate fine-tune data for Qwen repair-loop training.

Produces JSONL in OpenAI chat-format for supervised fine-tuning (SFT):
    {"messages": [
        {"role": "system", "content": <system prompt>},
        {"role": "user", "content": <user prompt>},
        {"role": "assistant", "content": <Claude's accepted patch>}
    ]}

Mechanism: run Claude (via CLI) on N different starting hparam configs
(mix of pathological + healthy). Record (system, user, response) for
each Claude call that produced an ACCEPTED patch. Skip iterations
where Claude returned NO_FIX_FOUND or the outcome gate rejected the
patch.

The resulting JSONL is a small seed dataset. Real fine-tune quality
would want hundreds-to-thousands of examples; this produces ~5-20
demonstrations in a few minutes of wall clock. That's enough for a
proof-of-concept LoRA fine-tune — the point here is to have the
dataset pipeline ready, not to run the fine-tune itself (needs more
VRAM than this GTX 1050 has).

Usage:
    python scripts/generate_qwen_finetune_data.py --n-configs 5

Output:
    results/qwen_finetune_data.jsonl   (SFT dataset)
    results/qwen_finetune_trace.json   (full run trace per config)
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.repair.context import GeometricContextGatherer  # noqa: E402
from fractaltrainer.repair.hparam_config import save_hparams  # noqa: E402
from fractaltrainer.repair.llm_client import make_claude_cli_client  # noqa: E402
from fractaltrainer.repair.prompt_builder import PromptBuilder  # noqa: E402
from fractaltrainer.repair.repair_loop import RepairLoop  # noqa: E402
from fractaltrainer.target.target_shape import load_target  # noqa: E402


PATHOLOGICAL_CONFIGS = [
    # name → hparams dict
    ("lr_too_high_sgd", {
        "learning_rate": 0.1, "batch_size": 64, "weight_decay": 0.0,
        "dropout": 0.0, "init_seed": 42, "optimizer": "sgd",
    }),
    ("lr_too_low", {
        "learning_rate": 1e-5, "batch_size": 128, "weight_decay": 0.0,
        "dropout": 0.0, "init_seed": 42, "optimizer": "adam",
    }),
    ("no_regularization_large_batch", {
        "learning_rate": 0.01, "batch_size": 512, "weight_decay": 0.0,
        "dropout": 0.0, "init_seed": 42, "optimizer": "adam",
    }),
    ("tiny_batch_high_lr", {
        "learning_rate": 0.05, "batch_size": 4, "weight_decay": 0.0,
        "dropout": 0.0, "init_seed": 42, "optimizer": "sgd",
    }),
    ("heavy_dropout_no_wd", {
        "learning_rate": 0.001, "batch_size": 32, "weight_decay": 0.0,
        "dropout": 0.85, "init_seed": 42, "optimizer": "adam",
    }),
    ("adamw_zero_wd", {
        "learning_rate": 0.005, "batch_size": 64, "weight_decay": 0.0,
        "dropout": 0.0, "init_seed": 42, "optimizer": "adamw",
    }),
]


def run_one_config(
    name: str, hparams: dict, target_path: Path,
    experiment_path: Path, hparams_path: Path,
    max_iters: int = 3, verbose: bool = True,
) -> dict:
    """Reset hparams to the given config, run Claude closed-loop,
    return all successful (prompt, patch) pairs from accepted iters."""
    # Seed the pathological config
    save_hparams(hparams, hparams_path)

    llm_fn = make_claude_cli_client()
    target = load_target(target_path)
    loop = RepairLoop(
        project_root=str(REPO_ROOT), target=target,
        experiment_config=str(experiment_path),
        hparams_path=str(hparams_path.relative_to(REPO_ROOT)),
        llm_fn=llm_fn,
    )

    # We need the prompts the loop USED — capture them by wrapping llm_fn.
    captured_calls: list[dict] = []

    def capture_llm(system: str, user: str) -> str:
        t0 = time.time()
        response = llm_fn(system, user)
        captured_calls.append({
            "system": system, "user": user,
            "response": response, "elapsed_s": time.time() - t0,
        })
        return response

    loop.llm_fn = capture_llm

    attempts = loop.repair(max_iters=max_iters, verbose=verbose)

    # Match captured calls to attempts by ordering (the loop calls
    # the LLM once per iteration before running trial probes).
    sft_examples: list[dict] = []
    for i, (attempt, call) in enumerate(zip(attempts, captured_calls)):
        if attempt.status == "accepted":
            sft_examples.append({
                "config_name": name,
                "iteration": attempt.iteration,
                "system": call["system"],
                "user": call["user"],
                "assistant": call["response"],
                "dim_before": attempt.dim_before,
                "dim_after": attempt.dim_after,
                "divergence_before": attempt.divergence_before,
                "divergence_after": attempt.divergence_after,
            })

    return {
        "config_name": name,
        "initial_hparams": hparams,
        "attempts": [{
            "iteration": a.iteration, "status": a.status,
            "dim_before": a.dim_before, "dim_after": a.dim_after,
        } for a in attempts],
        "sft_examples": sft_examples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiment.yaml")
    parser.add_argument("--target", default="configs/target_shape.yaml")
    parser.add_argument("--hparams", default="configs/hparams.yaml")
    parser.add_argument("--n-configs", type=int, default=6)
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--sft-out",
                        default="results/qwen_finetune_data.jsonl")
    parser.add_argument("--trace-out",
                        default="results/qwen_finetune_trace.json")
    parser.add_argument("--skip-names", nargs="*", default=[])
    args = parser.parse_args(argv)

    experiment_path = REPO_ROOT / args.experiment
    target_path = REPO_ROOT / args.target
    hparams_path = REPO_ROOT / args.hparams

    # Back up user's current hparams.yaml
    hparams_backup = hparams_path.with_suffix(".yaml.pre_finetune_bak")
    shutil.copy2(hparams_path, hparams_backup)
    print(f"[sft] backed up hparams.yaml → {hparams_backup}")

    configs_to_run = [(n, h) for n, h in PATHOLOGICAL_CONFIGS
                      if n not in args.skip_names][: args.n_configs]
    print(f"[sft] running Claude CLI on {len(configs_to_run)} configs\n")

    all_traces: list[dict] = []
    all_sft: list[dict] = []
    for i, (name, hp) in enumerate(configs_to_run, 1):
        print(f"\n[sft] ({i}/{len(configs_to_run)}) config={name}")
        print(f"       starting hparams: {hp}")
        try:
            trace = run_one_config(
                name, hp, target_path, experiment_path, hparams_path,
                max_iters=args.max_iters, verbose=True,
            )
            all_traces.append(trace)
            all_sft.extend(trace["sft_examples"])
            print(f"[sft] accepted examples from {name}: "
                  f"{len(trace['sft_examples'])}")
        except Exception as e:
            print(f"[sft] config {name} failed: {e}")
            all_traces.append({"config_name": name, "error": str(e)})

    # Restore original hparams
    shutil.copy2(hparams_backup, hparams_path)
    hparams_backup.unlink()
    print(f"\n[sft] restored original hparams.yaml")

    # Write JSONL in chat-format
    sft_out = Path(args.sft_out)
    sft_out.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_out, "w") as f:
        for ex in all_sft:
            record = {
                "messages": [
                    {"role": "system", "content": ex["system"]},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]},
                ],
                "metadata": {
                    "config_name": ex["config_name"],
                    "iteration": ex["iteration"],
                    "dim_before": ex["dim_before"],
                    "dim_after": ex["dim_after"],
                    "divergence_before": ex["divergence_before"],
                    "divergence_after": ex["divergence_after"],
                },
            }
            f.write(json.dumps(record, default=str) + "\n")
    print(f"[sft] wrote {len(all_sft)} SFT examples → {sft_out}")

    # Full trace
    trace_out = Path(args.trace_out)
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_out, "w") as f:
        json.dump({
            "n_configs_attempted": len(configs_to_run),
            "n_sft_examples": len(all_sft),
            "traces": all_traces,
        }, f, indent=2, default=str)
    print(f"[sft] full trace → {trace_out}")

    # Summary
    print(f"\n====== SFT DATA GENERATION SUMMARY ======")
    print(f"Configs attempted: {len(configs_to_run)}")
    successful = sum(1 for t in all_traces if "error" not in t)
    print(f"Configs successful: {successful}")
    print(f"Total SFT examples (accepted iterations): {len(all_sft)}")
    print(f"\nDataset ready at {sft_out}. Example format:")
    print("  {'messages': [system, user, assistant], 'metadata': {...}}")
    print(f"\nTo fine-tune Qwen-7B on this data on a machine with >= 16 GB VRAM:")
    print(f"  pip install unsloth peft transformers")
    print(f"  python -c \"")
    print(f"  from unsloth import FastLanguageModel, apply_chat_template")
    print(f"  model, tok = FastLanguageModel.from_pretrained(")
    print(f"    'Qwen/Qwen2.5-Coder-7B-Instruct', load_in_4bit=True)")
    print(f"  # ... LoRA adapter + trainer, ~10 minutes on A100\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
