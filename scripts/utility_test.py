"""v2 Sprint 8 utility test: is the golden-run repair loop actually useful?

Four conditions, same divergent starting point, same MNIST train/test
split. Measure held-out test accuracy on all four.

Condition 1 — Golden-run repair loop (the system under test)
    Run scripts/run_closed_loop.py --llm cli against the opaque
    golden, then train from scratch with the final hparams and
    evaluate on test. Uses Claude CLI quota.

Condition 2 — Naive first-aid
    Halve the starting learning rate, keep everything else. No LLM.

Condition 3 — Textbook "fix diverging Adam" recipe
    AdamW, lr=1e-3, wd=1e-2, do=0.1. No LLM, no search.

Condition 4 — Random search
    Sample 5 random hparam configurations from a sensible prior,
    train each, pick the one with highest test accuracy. No LLM.

Usage:
    python scripts/utility_test.py \\
        [--out results/utility_test.json] \\
        [--python-bin <fractal_env python>]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fractaltrainer.observer.trainer import InstrumentedTrainer  # noqa: E402


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(64, 32), output_dim=10,
                 dropout=0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def _mnist_loaders(train_size: int, test_size: int, batch_size: int,
                   data_dir: str, seed: int):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    train_full = datasets.MNIST(data_dir, train=True, download=True,
                                transform=transform)
    test_full = datasets.MNIST(data_dir, train=False, download=True,
                               transform=transform)
    rng = np.random.RandomState(seed)
    train_idx = rng.choice(len(train_full), size=train_size, replace=False)
    test_idx = rng.choice(len(test_full), size=test_size, replace=False)
    train_loader = DataLoader(Subset(train_full, train_idx.tolist()),
                              batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(Subset(test_full, test_idx.tolist()),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _evaluate(model: nn.Module, loader) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            loss_sum += float(F.cross_entropy(logits, y).item())
            n_batches += 1
    return correct / max(total, 1), loss_sum / max(n_batches, 1)


def _train_and_eval(hparams: dict, n_steps: int, train_size: int,
                    test_size: int, data_dir: str, data_seed: int
                    ) -> dict:
    """Train an MLP from scratch with these hparams. Return test metrics + wall time."""
    t0 = time.time()
    model = MLP(input_dim=784, hidden_dims=(64, 32), output_dim=10,
                dropout=float(hparams.get("dropout", 0.0)))
    train_loader, test_loader = _mnist_loaders(
        train_size=train_size, test_size=test_size,
        batch_size=int(hparams["batch_size"]),
        data_dir=data_dir, seed=data_seed,
    )
    trainer = InstrumentedTrainer(
        model=model, dataloader=train_loader, loss_fn=F.cross_entropy,
        hparams=hparams, snapshot_every=max(n_steps // 5, 1),
        out_dir=str(REPO_ROOT / "results" / "utility_trajectories"),
        run_id=f"utility_lr{hparams['learning_rate']}_opt{hparams['optimizer']}_seed{hparams['init_seed']}_t{int(t0)}",
    )
    trainer.train(n_steps=n_steps)
    test_acc, test_loss = _evaluate(model, test_loader)
    return {
        "hparams": hparams,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "wall_clock_s": time.time() - t0,
    }


def _read_hparams(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _write_hparams(hparams: dict, path: Path) -> None:
    ordered = ["learning_rate", "batch_size", "weight_decay",
               "dropout", "init_seed", "optimizer"]
    lines = ["# Utility test starting configuration.", ""]
    for k in ordered:
        v = hparams[k]
        if isinstance(v, str):
            lines.append(f'{k}: "{v}"')
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n")


def condition_1_golden_run_loop(start: dict, python_bin: str,
                                 n_steps: int, train_size: int,
                                 test_size: int, data_dir: str,
                                 data_seed: int) -> dict:
    """Run the golden-run closed loop, then evaluate the final model."""
    print("\n[C1] Golden-run repair loop (opaque golden)")
    hparams_path = REPO_ROOT / "configs" / "hparams.yaml"
    target_path = REPO_ROOT / "configs" / "target_shape_golden_opaque.yaml"
    repair_log_path = REPO_ROOT / "results" / "repair_log.jsonl"
    closed_loop_script = REPO_ROOT / "scripts" / "run_closed_loop.py"

    _write_hparams(start, hparams_path)
    if repair_log_path.exists():
        repair_log_path.unlink()

    t0 = time.time()
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [python_bin, str(closed_loop_script),
         "--llm", "cli",
         "--target", str(target_path.relative_to(REPO_ROOT)),
         "--max-iters", "3",
         "--python-bin", python_bin],
        capture_output=True, text=True, timeout=1800,
        cwd=str(REPO_ROOT), env=env,
    )
    loop_wall = time.time() - t0
    if result.returncode != 0:
        return {"condition": "golden_run_loop", "error": result.stderr[-500:],
                "wall_clock_s": loop_wall}

    final_hparams = _read_hparams(hparams_path)
    print(f"    loop finished in {loop_wall:.1f}s")
    print(f"    final hparams: {final_hparams}")

    eval_out = _train_and_eval(final_hparams, n_steps, train_size,
                                test_size, data_dir, data_seed)
    eval_out["condition"] = "golden_run_loop"
    eval_out["starting_hparams"] = start
    eval_out["loop_wall_s"] = loop_wall
    eval_out["total_wall_s"] = loop_wall + eval_out["wall_clock_s"]
    print(f"    test_acc (from-scratch with final hparams): "
          f"{eval_out['test_accuracy']:.4f}")
    return eval_out


def condition_2_first_aid(start: dict, n_steps: int, train_size: int,
                           test_size: int, data_dir: str,
                           data_seed: int) -> dict:
    print("\n[C2] Naive first-aid (halve lr, keep everything else)")
    hp = dict(start)
    hp["learning_rate"] = float(start["learning_rate"]) / 2.0
    print(f"    hparams: {hp}")
    r = _train_and_eval(hp, n_steps, train_size, test_size,
                         data_dir, data_seed)
    r["condition"] = "first_aid_halve_lr"
    r["starting_hparams"] = start
    r["total_wall_s"] = r["wall_clock_s"]
    print(f"    test_acc: {r['test_accuracy']:.4f}")
    return r


def condition_3_textbook(start: dict, n_steps: int, train_size: int,
                          test_size: int, data_dir: str,
                          data_seed: int) -> dict:
    print("\n[C3] Textbook 'fix diverging Adam' recipe")
    hp = {
        "learning_rate": 0.001,
        "batch_size": int(start.get("batch_size", 64)),
        "weight_decay": 0.01,
        "dropout": 0.1,
        "init_seed": int(start.get("init_seed", 42)),
        "optimizer": "adamw",
    }
    print(f"    hparams: {hp}")
    r = _train_and_eval(hp, n_steps, train_size, test_size,
                         data_dir, data_seed)
    r["condition"] = "textbook_adamw_small_lr"
    r["starting_hparams"] = start
    r["total_wall_s"] = r["wall_clock_s"]
    print(f"    test_acc: {r['test_accuracy']:.4f}")
    return r


def condition_4_random_search(start: dict, n_candidates: int, n_steps: int,
                               train_size: int, test_size: int,
                               data_dir: str, data_seed: int,
                               search_seed: int = 42) -> dict:
    print(f"\n[C4] Random search ({n_candidates} candidates)")
    rng = np.random.RandomState(search_seed)
    OPTS = ["sgd", "adam", "adamw"]
    candidates: list[dict] = []
    for i in range(n_candidates):
        hp = {
            "learning_rate": float(10 ** rng.uniform(-4, -1)),  # [1e-4, 1e-1]
            "batch_size": int(start.get("batch_size", 64)),
            "weight_decay": float(rng.uniform(0.0, 0.05)),
            "dropout": float(rng.uniform(0.0, 0.3)),
            "init_seed": int(start.get("init_seed", 42)),
            "optimizer": OPTS[rng.randint(0, 3)],
        }
        candidates.append(hp)

    results = []
    t_start = time.time()
    for i, hp in enumerate(candidates, 1):
        print(f"    candidate {i}/{n_candidates}: {hp}")
        r = _train_and_eval(hp, n_steps, train_size, test_size,
                             data_dir, data_seed)
        print(f"        test_acc = {r['test_accuracy']:.4f}")
        results.append(r)
    total_wall = time.time() - t_start

    best = max(results, key=lambda r: r["test_accuracy"])
    return {
        "condition": "random_search",
        "starting_hparams": start,
        "candidates": candidates,
        "all_results": results,
        "best_hparams": best["hparams"],
        "test_accuracy": best["test_accuracy"],
        "test_loss": best["test_loss"],
        "total_wall_s": total_wall,
        "n_candidates": n_candidates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 Sprint 8 utility test")
    parser.add_argument("--out", type=str,
                        default="results/utility_test.json")
    parser.add_argument("--python-bin", type=str,
                        default=sys.executable)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default="results/data")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--n-random-candidates", type=int, default=5)
    args = parser.parse_args(argv)

    START = {
        "learning_rate": 0.1,
        "batch_size": 64,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "init_seed": 42,
        "optimizer": "adam",
    }

    print("=" * 70)
    print("v2 Sprint 8 Utility Test")
    print("=" * 70)
    print(f"Starting divergent config: {START}")
    print(f"n_steps={args.n_steps}  train={args.train_size}  test={args.test_size}")
    print(f"data_seed={args.data_seed} (determines same MNIST splits for all conditions)")

    # Also record a baseline-zero: the divergent config itself, no fix.
    print("\n[C0] No fix (just run the divergent config as-is)")
    baseline_zero = _train_and_eval(
        START, args.n_steps, args.train_size, args.test_size,
        args.data_dir, args.data_seed,
    )
    baseline_zero["condition"] = "no_fix"
    baseline_zero["starting_hparams"] = START
    baseline_zero["total_wall_s"] = baseline_zero["wall_clock_s"]
    print(f"    test_acc (no fix applied): {baseline_zero['test_accuracy']:.4f}")

    c1 = condition_1_golden_run_loop(
        START, args.python_bin, args.n_steps, args.train_size,
        args.test_size, args.data_dir, args.data_seed,
    )
    c2 = condition_2_first_aid(
        START, args.n_steps, args.train_size, args.test_size,
        args.data_dir, args.data_seed,
    )
    c3 = condition_3_textbook(
        START, args.n_steps, args.train_size, args.test_size,
        args.data_dir, args.data_seed,
    )
    c4 = condition_4_random_search(
        START, args.n_random_candidates, args.n_steps,
        args.train_size, args.test_size, args.data_dir, args.data_seed,
    )

    results = [baseline_zero, c1, c2, c3, c4]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'condition':<30s}  {'test_acc':>9s}  {'wall_s':>8s}")
    print("-" * 60)
    for r in results:
        cond = r.get("condition", "?")
        acc = r.get("test_accuracy", float("nan"))
        wall = r.get("total_wall_s", r.get("wall_clock_s", 0.0))
        acc_s = f"{acc:.4f}" if isinstance(acc, float) and math.isfinite(acc) \
            else "n/a"
        print(f"{cond:<30s}  {acc_s:>9s}  {wall:>8.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "starting_config": START,
            "n_steps": args.n_steps,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "data_seed": args.data_seed,
            "conditions": results,
        }, f, indent=2, default=str)
    print(f"\n[utility] results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
