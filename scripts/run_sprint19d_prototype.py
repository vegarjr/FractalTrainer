"""Sprint 19d: prototype head closes the residual 5.6 pp gap?

Sprint 19c's hybrid (meta-trained frozen encoder + trainable softmax
fc3 head) reached 0.844 on Omniglot 5-way 5-shot — 12.6 pp above
vanilla but 5.6 pp below ProtoNets (0.900). Hypothesis: the residual
gap is the softmax-head vs prototype-classification mismatch. Meta-
training was done with prototype-distance loss; test time used a
softmax head on `fc3`. That's a training/inference mismatch the 25
support examples can't fully close.

This sprint tests whether using **prototype classification** at test
time (same encoder, no trainable head) matches ProtoNets exactly while
preserving the registry interface.

Methods compared (100 Omniglot 5-way 5-shot episodes):

  1. ProtoNets                — upper bound (0.900 expected)
  2. hybrid-softmax (19c)     — 0.844 expected
  3. hybrid-prototype (new)   — should match ProtoNets within noise

Wrapped through FractalRegistry → routing still works (for seeding +
nearest-neighbor signature lookup), just with zero-trainable experts.

Runs locally on CPU in ~6 min.
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

from fractaltrainer.integration.hybrid_head import (  # noqa: E402
    MLPEncoderTrunk, build_hybrid_expert, trainable_parameters,
)
from fractaltrainer.integration.prototype_head import PrototypeExpert  # noqa: E402
from fractaltrainer.integration.signatures import softmax_signature  # noqa: E402
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402

# Reuse data loading from 19c
sys.path.insert(0, os.path.dirname(__file__))
from run_sprint19c_hybrid import (  # noqa: E402
    build_probe, load_omniglot, meta_train_protonets,
    protonet_eval_episode, sample_episode,
)

RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


def train_hybrid_softmax_expert(encoder, sx, sy, *, n_way, n_steps, lr, device, seed):
    """Train the hybrid-softmax expert from 19c (frozen encoder + trainable fc3)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_hybrid_expert(encoder, n_classes=n_way, context_scale=0.0,
                                  freeze_encoder=True).to(device)
    opt = torch.optim.Adam(trainable_parameters(model), lr=lr)
    model.train()
    for step in range(n_steps):
        perm = torch.randperm(sx.size(0), device=sx.device)
        xb, yb = sx[perm], sy[perm]
        opt.zero_grad()
        loss = F.cross_entropy(model(xb, context=None), yb)
        loss.backward(); opt.step()
    model.eval()
    return model


def fit_prototype_expert(encoder, sx, sy, *, n_way, device):
    return PrototypeExpert.fit(encoder, sx.to(device), sy.to(device), n_way=n_way)


def eval_hybrid_softmax(model, qx, qy):
    model.eval()
    with torch.no_grad():
        pred = model(qx).argmax(dim=1)
    return float((pred == qy).float().mean().item())


def eval_prototype(expert, qx, qy):
    return expert.score(qx, qy)


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
        print("Pick one: --smoke or --full"); sys.exit(2)
    if args.smoke:
        mode = "SMOKE"; n_test_eps = 10; protonet_iters = 500; spawn_steps = 100
    else:
        mode = "FULL"; n_test_eps = 100; protonet_iters = 3000; spawn_steps = 200

    n_way, k_shot, n_query = 5, 5, 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sprint 19d — {mode}  device={device}  test_eps={n_test_eps}")

    t_all = time.time()

    print("\n[1/3] Loading Omniglot...")
    bg_ds, ev_ds, bg_idx, ev_idx = load_omniglot(args.omniglot_root, device)
    probe_X = build_probe(bg_ds, bg_idx, size=100, device=device)

    print(f"\n[2/3] Meta-training ProtoNets encoder ({protonet_iters} iters)...")
    encoder = meta_train_protonets(bg_ds, bg_idx, device=device, n_iters=protonet_iters,
                                    n_way=n_way, k_shot=k_shot, n_query=n_query)

    print(f"\n[3/3] Evaluating {n_test_eps} test episodes...")
    rng = np.random.default_rng(2024)
    accs = {"protonets": [], "hybrid_softmax": [], "hybrid_prototype": []}
    t_eval = time.time()
    for ep in range(n_test_eps):
        sx, sy, qx, qy, chars = sample_episode(ev_ds, ev_idx, rng, n_way, k_shot, n_query, device=device)

        # ProtoNets (upper bound)
        accs["protonets"].append(protonet_eval_episode(encoder, sx, sy, qx, qy, n_way))

        # Hybrid + softmax (19c)
        hs_model = train_hybrid_softmax_expert(encoder, sx, sy, n_way=n_way,
                                                n_steps=spawn_steps, lr=0.01,
                                                device=device, seed=ep + 30000)
        accs["hybrid_softmax"].append(eval_hybrid_softmax(hs_model, qx, qy))

        # Hybrid + prototype (new)
        hp_expert = fit_prototype_expert(encoder, sx, sy, n_way=n_way, device=device)
        accs["hybrid_prototype"].append(eval_prototype(hp_expert, qx, qy))

        if (ep + 1) % max(1, n_test_eps // 10) == 0:
            means = {k: np.mean(v) for k, v in accs.items()}
            print(f"  ep {ep+1:>3d}/{n_test_eps}  "
                  f"proto={means['protonets']:.3f}  "
                  f"hyb_sm={means['hybrid_softmax']:.3f}  "
                  f"hyb_pr={means['hybrid_prototype']:.3f}  "
                  f"[{time.time()-t_eval:.0f}s]")

    # Summarize
    summary = {name: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                       "n": len(v), "accs": list(v)} for name, v in accs.items()}
    pairs = [
        ("hybrid_prototype", "protonets"),
        ("hybrid_prototype", "hybrid_softmax"),
        ("protonets", "hybrid_softmax"),
    ]
    tests = {}
    for a, b in pairs:
        aa, bb = np.asarray(accs[a]), np.asarray(accs[b])
        t, p = stats.ttest_ind(aa, bb, equal_var=False)
        tests[f"{a}_vs_{b}"] = {"delta": float(aa.mean() - bb.mean()),
                                  "welch_t": float(t), "welch_p": float(p)}

    print("\n" + "=" * 60)
    print(f"SPRINT 19d PROTOTYPE — {mode} ({n_test_eps} eps)")
    print("=" * 60)
    for name in ["hybrid_softmax", "hybrid_prototype", "protonets"]:
        s = summary[name]
        print(f"  {name:>18s}  {s['mean']:.4f} ± {s['std']:.4f}")
    print("\nWelch t-tests:")
    for k, v in tests.items():
        sig = "**" if v["welch_p"] < 0.01 else ("*" if v["welch_p"] < 0.05 else "(NS)")
        print(f"  {k:>40s}  Δ={v['delta']:+.4f}  t={v['welch_t']:+.2f}  p={v['welch_p']:.4f}  {sig}")

    delta_pp = summary["hybrid_prototype"]["mean"] - summary["protonets"]["mean"]
    delta_ps = summary["hybrid_prototype"]["mean"] - summary["hybrid_softmax"]["mean"]
    print("\n=== Pre-registered verdicts ===")
    if abs(delta_pp) <= 0.01:
        print(f"  hybrid_prototype ≈ ProtoNets (|Δ|={abs(delta_pp):.3f} ≤ 0.01) → CLOSED THE GAP")
    elif abs(delta_pp) <= 0.03:
        print(f"  hybrid_prototype ≈ ProtoNets (|Δ|={abs(delta_pp):.3f} ≤ 0.03) — near-tie")
    else:
        print(f"  hybrid_prototype vs ProtoNets: Δ={delta_pp:+.4f} — unexpected deviation")
    if delta_ps > 0.03:
        print(f"  hybrid_prototype > hybrid_softmax by {delta_ps:+.4f} → prototype head is the right choice")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = "sprint19d_prototype" + ("_smoke" if args.smoke else "")
    payload = {
        "config": {
            "mode": mode, "n_way": n_way, "k_shot": k_shot, "n_query": n_query,
            "n_test_episodes": n_test_eps, "protonet_iters": protonet_iters,
            "spawn_steps": spawn_steps, "device": str(device),
        },
        "summary": summary, "tests": tests,
    }
    out = os.path.join(args.out_dir, f"{stem}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n-> {out}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["hybrid_softmax", "hybrid_prototype", "protonets"]
        labels = ["hybrid\n+softmax (19c)", "hybrid\n+prototype (19d)", "ProtoNets"]
        colors = ["steelblue", "darkorange", "seagreen"]
        means = [summary[n]["mean"] for n in order]
        stds = [summary[n]["std"] for n in order]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(3), means, yerr=stds, capsize=5, color=colors)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels)
        ax.axhline(0.2, color="gray", ls=":", alpha=0.5, label="chance")
        ax.set_ylabel("Accuracy"); ax.set_ylim(0.4, 1.02)
        ax.set_title("Omniglot 5-way 5-shot — does prototype head close the residual 5.6 pp gap?")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        png = os.path.join(args.out_dir, f"{stem}.png")
        fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"-> {png}")
    except Exception as e:
        print("plot skipped:", e)

    print(f"\nTotal: {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
