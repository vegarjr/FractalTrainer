"""Sprint 19e: encoder fine-tuning vs strict freeze.

Sprint 19c froze the meta-trained encoder completely; only the softmax
head `fc3` (and the context lane) trained per episode. That reached
0.844 / 0.900 ProtoNets — 5.6 pp below ProtoNets. Sprint 19d confirmed
the gap was the softmax-vs-prototype classifier choice, not the freeze.

This sprint asks a different question: **was the strict freeze leaving
capacity on the table?** Specifically, with a softmax head — can a
low-lr unfreeze of the encoder's last layer (fc2) recover some accuracy
without overfitting on just 25 support examples?

Conditions compared (100 Omniglot 5-way 5-shot episodes):

  1. hybrid-softmax strict       (19c baseline)         0.837 expected
  2. hybrid-softmax ft_fc2_low   (fc2 @ lr=1e-4)        novel
  3. hybrid-softmax ft_all_low   (fc1+fc2 @ lr=1e-4)    novel
  4. ProtoNets (upper)                                   0.900 expected

If any finetune condition matches or beats ProtoNets while keeping the
registry's expansion/audit/swap properties, we have the best of both
worlds: meta-learning accuracy + registry operational benefits without
the prototype-head constraint.

If all finetune conditions are at or below strict-freeze, the 19c
choice (strict freeze) is validated and 19d (prototype head) is the
right path to close the gap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from fractaltrainer.integration.context_mlp import ContextAwareMLP  # noqa: E402
from fractaltrainer.integration.hybrid_head import (  # noqa: E402
    MLPEncoderTrunk, load_pretrained_encoder,
)
from run_sprint19c_hybrid import (  # noqa: E402
    build_probe, load_omniglot, meta_train_protonets,
    protonet_eval_episode, sample_episode,
)

RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def build_model_with_layerwise_lrs(encoder: MLPEncoderTrunk, *, n_classes: int,
                                     device, seed,
                                     fc1_frozen: bool, fc2_frozen: bool):
    """Build a ContextAwareMLP with fc1/fc2 initialised from encoder.
    Freeze flags are independent (not necessarily the same)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ContextAwareMLP(n_classes=n_classes, context_scale=0.0).to(device)
    load_pretrained_encoder(model, encoder, freeze=False)  # copy weights
    # Apply independent freeze
    for p in model.fc1.parameters():
        p.requires_grad = not fc1_frozen
    for p in model.fc2.parameters():
        p.requires_grad = not fc2_frozen
    # fc3 is always trainable
    return model


def train_finetune(model, sx, sy, *, n_steps, head_lr, encoder_lr, device, seed):
    """Train with layerwise lrs: head (fc3) at head_lr, unfrozen encoder
    parts at encoder_lr."""
    torch.manual_seed(seed); np.random.seed(seed)
    head_params = list(model.fc3.parameters())
    # Context lane ignored here (context_scale=0), but include for optimizer tracking
    head_params += list(model.ctx_proj.parameters()) + list(model.ctx_norm.parameters())
    encoder_params = []
    if model.fc1.weight.requires_grad:
        encoder_params += list(model.fc1.parameters())
    if model.fc2.weight.requires_grad:
        encoder_params += list(model.fc2.parameters())
    param_groups = [{"params": head_params, "lr": head_lr}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    opt = torch.optim.Adam(param_groups)
    model.train()
    for step in range(n_steps):
        perm = torch.randperm(sx.size(0), device=sx.device)
        xb, yb = sx[perm], sy[perm]
        opt.zero_grad()
        loss = F.cross_entropy(model(xb, context=None), yb)
        loss.backward(); opt.step()
    model.eval()
    return model


def eval_model(model, qx, qy):
    model.eval()
    with torch.no_grad():
        pred = model(qx).argmax(dim=1)
    return float((pred == qy).float().mean().item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--omniglot-root", default="/tmp/omniglot")
    p.add_argument("--out-dir", default=RESULTS)
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.smoke or args.full):
        print("Pick --smoke or --full"); sys.exit(2)
    if args.smoke:
        mode, n_test_eps, protonet_iters, spawn_steps = "SMOKE", 10, 500, 100
    else:
        mode, n_test_eps, protonet_iters, spawn_steps = "FULL", 100, 3000, 200

    n_way, k_shot, n_query = 5, 5, 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sprint 19e — {mode}  device={device}  eps={n_test_eps}")

    t_all = time.time()

    print("\n[1/3] Loading Omniglot...")
    bg_ds, ev_ds, bg_idx, ev_idx = load_omniglot(args.omniglot_root, device)
    probe_X = build_probe(bg_ds, bg_idx, size=100, device=device)

    print(f"\n[2/3] Meta-training ProtoNets encoder ({protonet_iters} iters)...")
    encoder = meta_train_protonets(bg_ds, bg_idx, device=device, n_iters=protonet_iters,
                                    n_way=n_way, k_shot=k_shot, n_query=n_query)

    conditions = [
        # (name, fc1_frozen, fc2_frozen, encoder_lr)
        ("strict_freeze",       True,  True,  0.0),
        ("ft_fc2_low",          True,  False, 1e-4),
        ("ft_fc2_mid",          True,  False, 1e-3),
        ("ft_all_low",          False, False, 1e-4),
    ]

    print(f"\n[3/3] Evaluating {n_test_eps} test episodes × {len(conditions)+1} methods...")
    rng = np.random.default_rng(2024)
    accs = {name: [] for name, *_ in conditions}
    accs["protonets"] = []
    t_eval = time.time()
    for ep in range(n_test_eps):
        sx, sy, qx, qy, chars = sample_episode(ev_ds, ev_idx, rng, n_way, k_shot, n_query, device=device)

        # ProtoNets upper bound
        accs["protonets"].append(protonet_eval_episode(encoder, sx, sy, qx, qy, n_way))

        # Each hybrid finetune condition
        for name, fc1_f, fc2_f, enc_lr in conditions:
            model = build_model_with_layerwise_lrs(
                encoder, n_classes=n_way, device=device,
                seed=ep + 40000,
                fc1_frozen=fc1_f, fc2_frozen=fc2_f,
            )
            train_finetune(model, sx, sy, n_steps=spawn_steps,
                           head_lr=0.01, encoder_lr=enc_lr,
                           device=device, seed=ep + 40000)
            accs[name].append(eval_model(model, qx, qy))

        if (ep + 1) % max(1, n_test_eps // 10) == 0:
            means = {k: np.mean(v) for k, v in accs.items()}
            msg = "  ep %3d/%d " % (ep + 1, n_test_eps) + "  ".join(
                f"{k}={means[k]:.3f}" for k in ["strict_freeze", "ft_fc2_low",
                                                  "ft_fc2_mid", "ft_all_low", "protonets"]
            ) + f"  [{time.time()-t_eval:.0f}s]"
            print(msg)

    # Summarize + tests
    summary = {name: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                       "n": len(v), "accs": list(v)} for name, v in accs.items()}
    pairs = [
        ("ft_fc2_low", "strict_freeze"),
        ("ft_fc2_mid", "strict_freeze"),
        ("ft_all_low", "strict_freeze"),
        ("ft_fc2_low", "protonets"),
        ("strict_freeze", "protonets"),
    ]
    tests = {}
    for a, b in pairs:
        aa, bb = np.asarray(accs[a]), np.asarray(accs[b])
        t, p = stats.ttest_ind(aa, bb, equal_var=False)
        tests[f"{a}_vs_{b}"] = {"delta": float(aa.mean() - bb.mean()),
                                  "welch_t": float(t), "welch_p": float(p)}

    print("\n" + "=" * 60)
    print(f"SPRINT 19e FINETUNE — {mode} ({n_test_eps} eps)")
    print("=" * 60)
    order = ["strict_freeze", "ft_fc2_low", "ft_fc2_mid", "ft_all_low", "protonets"]
    for name in order:
        s = summary[name]
        print(f"  {name:>18s}  {s['mean']:.4f} ± {s['std']:.4f}")
    print("\nWelch's t-tests:")
    for k, v in tests.items():
        sig = "**" if v["welch_p"] < 0.01 else ("*" if v["welch_p"] < 0.05 else "(NS)")
        print(f"  {k:>38s}  Δ={v['delta']:+.4f}  t={v['welch_t']:+.2f}  p={v['welch_p']:.4f}  {sig}")

    # Pre-registered verdict
    print("\n=== Pre-registered verdicts ===")
    proto = summary["protonets"]["mean"]; strict = summary["strict_freeze"]["mean"]
    best_ft = max(summary[n]["mean"] for n in ["ft_fc2_low", "ft_fc2_mid", "ft_all_low"])
    best_name = max(["ft_fc2_low", "ft_fc2_mid", "ft_all_low"], key=lambda n: summary[n]["mean"])
    if best_ft >= proto - 0.01:
        print(f"  Best finetune ({best_name}) ≥ ProtoNets: {best_ft:.4f} vs {proto:.4f} — closes the gap!")
    elif best_ft > strict + 0.02:
        print(f"  Best finetune ({best_name}) beats strict by {best_ft - strict:+.4f}  (partial)")
    else:
        print(f"  Best finetune ({best_name}) = {best_ft:.4f}; no improvement over strict {strict:.4f}")
        print("  → Strict freeze was the right 19c choice; softmax head is the bottleneck not the freeze")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    stem = "sprint19e_finetune" + ("_smoke" if args.smoke else "")
    payload = {
        "config": {"mode": mode, "n_test_episodes": n_test_eps,
                    "protonet_iters": protonet_iters, "spawn_steps": spawn_steps,
                    "conditions": [{"name": n, "fc1_frozen": f1, "fc2_frozen": f2, "encoder_lr": lr}
                                    for n, f1, f2, lr in conditions],
                    "device": str(device)},
        "summary": summary, "tests": tests,
    }
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"-> {stem}.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        means = [summary[n]["mean"] for n in order]
        stds = [summary[n]["std"] for n in order]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(order)), means, yerr=stds, capsize=5,
               color=["steelblue", "peachpuff", "orange", "darkorange", "seagreen"])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([n.replace("_", "\n") for n in order], fontsize=9)
        ax.axhline(0.2, color="gray", ls=":", alpha=0.5, label="chance")
        ax.set_ylabel("Accuracy"); ax.set_ylim(0.4, 1.02)
        ax.set_title("Sprint 19e: encoder fine-tune vs strict freeze")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print(f"\nTotal: {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
