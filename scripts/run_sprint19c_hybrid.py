"""Sprint 19c: hybrid registry (meta-trained encoder + FractalTrainer head).

Sprint 19b showed FractalTrainer's plain spawn loses to ProtoNets by
18.7 pp on Omniglot 5-way 5-shot. The 18.7 pp gap is the cost of
supervising the encoder from scratch on 25 support examples per
episode, while ProtoNets meta-learns the encoder across thousands of
background episodes.

This sprint tests a hybrid: a *single* meta-trained encoder (frozen
after training) shared by every registry entry, and a trainable head +
context lane that's spawned per episode. Registry expansion, signature
routing, and context injection mechanics are preserved; the "each
expert re-learns vision from 25 examples" overhead is eliminated.

Methods compared (100 test episodes, shared 784→64→32 backbone):

  1. pixel-kNN                   — sanity floor
  2. MLP-ProtoNets               — upper bound (meta-trained, end-to-end)
  3. FractalTrainer vanilla      — Sprint 19b reference (encoder from scratch)
  4. **hybrid +ctx**             — meta-trained frozen encoder + spawn +
                                    K=3 context injection (the novel method)
  5. hybrid no-ctx               — ablation (encoder frozen, no context)

Statistical tests: Welch's t-test between method pairs + per-episode
win/loss count.

This script runs **locally** (not Colab). Expected wall time on a
typical CPU-only laptop: ~45 min. On a local GPU: ~15 min.

Setup:
    cd /path/to/FractalTrainer
    python -m venv .venv && source .venv/bin/activate
    pip install -e .          # installs torch, torchvision, numpy, sklearn
    pip install matplotlib scipy

Run:
    python scripts/run_sprint19c_hybrid.py --smoke        # ~3 min
    python scripts/run_sprint19c_hybrid.py --full         # ~45 min CPU
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fractaltrainer.integration.context_injection import ContextSpec, gather_context  # noqa: E402
from fractaltrainer.integration.context_mlp import ContextAwareMLP  # noqa: E402
from fractaltrainer.integration.hybrid_head import (  # noqa: E402
    MLPEncoderTrunk, build_hybrid_expert, trainable_parameters,
)
from fractaltrainer.integration.signatures import softmax_signature  # noqa: E402
from fractaltrainer.registry import FractalEntry, FractalRegistry  # noqa: E402

RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))


# --------------------------------------------------------------
# Config
# --------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Fast validation run (20 seed / 10 test episodes, ~3 min)")
    p.add_argument("--full", action="store_true",
                   help="Full benchmark (200 seed / 100 test episodes, ~45 min)")
    p.add_argument("--omniglot-root", default="/tmp/omniglot",
                   help="Directory for Omniglot data download")
    p.add_argument("--out-dir", default=RESULTS,
                   help="Directory to save result JSON + PNG")
    return p.parse_args()


# --------------------------------------------------------------
# Data
# --------------------------------------------------------------
def load_omniglot(root: str, device: torch.device):
    tfm = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
    ])
    bg = datasets.Omniglot(root=root, background=True,  download=True, transform=tfm)
    ev = datasets.Omniglot(root=root, background=False, download=True, transform=tfm)

    def _class_index(ds):
        idx_by_class = {}
        for i, (_, y) in enumerate(ds._flat_character_images):
            idx_by_class.setdefault(y, []).append(i)
        return idx_by_class

    bg_idx = _class_index(bg)
    ev_idx = _class_index(ev)
    return bg, ev, bg_idx, ev_idx


def sample_episode(ds, idx_by_class, rng, n_way=5, k_shot=5, n_query=15,
                   device=None, exclude=None):
    classes = sorted(idx_by_class.keys())
    exclude = set(exclude or ())
    picks = []
    while len(picks) < n_way:
        c = int(rng.choice(classes))
        if c in picks or c in exclude:
            continue
        if len(idx_by_class[c]) < k_shot + n_query:
            continue
        picks.append(c)
    sup_x, sup_y, qry_x, qry_y = [], [], [], []
    for lab, c in enumerate(picks):
        idx = list(idx_by_class[c])
        rng.shuffle(idx)
        for j in idx[:k_shot]:
            sup_x.append(ds[j][0].view(-1))
            sup_y.append(lab)
        for j in idx[k_shot:k_shot + n_query]:
            qry_x.append(ds[j][0].view(-1))
            qry_y.append(lab)
    kwargs = {"device": device} if device is not None else {}
    return (
        torch.stack(sup_x).to(**kwargs),
        torch.tensor(sup_y, dtype=torch.long, **kwargs),
        torch.stack(qry_x).to(**kwargs),
        torch.tensor(qry_y, dtype=torch.long, **kwargs),
        tuple(picks),
    )


def build_probe(bg_ds, bg_idx, size: int, device, seed=0):
    rng = np.random.default_rng(seed)
    classes = sorted(bg_idx.keys())
    items = []
    for _ in range(size):
        c = rng.choice(classes)
        i = rng.choice(bg_idx[int(c)])
        items.append(int(i))
    X = torch.stack([bg_ds[i][0] for i in items]).view(size, -1).to(device)
    return X


# --------------------------------------------------------------
# ProtoNets meta-training (shared encoder baseline)
# --------------------------------------------------------------
def protonet_episode_loss(encoder, sx, sy, qx, qy, n_way):
    sup = encoder(sx)
    qry = encoder(qx)
    protos = torch.stack([sup[sy == k].mean(0) for k in range(n_way)])
    logits = -torch.cdist(qry, protos)
    return F.cross_entropy(logits, qy), float((logits.argmax(1) == qy).float().mean().item())


def meta_train_protonets(bg_ds, bg_idx, *, device, n_iters, n_way=5,
                          k_shot=5, n_query=15, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    encoder = MLPEncoderTrunk().to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    rng = np.random.default_rng(seed + 7)
    t0 = time.time()
    for it in range(n_iters):
        sx, sy, qx, qy, _ = sample_episode(bg_ds, bg_idx, rng, n_way, k_shot, n_query, device=device)
        loss, acc = protonet_episode_loss(encoder, sx, sy, qx, qy, n_way)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % max(1, n_iters // 10) == 0:
            print(f"  protonet iter {it+1:>4d}/{n_iters}  loss={loss.item():.4f}  acc={acc:.3f}"
                  f"  [{time.time()-t0:.0f}s]")
    encoder.eval()
    return encoder


def protonet_eval_episode(encoder, sx, sy, qx, qy, n_way):
    encoder.eval()
    with torch.no_grad():
        sup = encoder(sx); qry = encoder(qx)
        protos = torch.stack([sup[sy == k].mean(0) for k in range(n_way)])
        logits = -torch.cdist(qry, protos)
        pred = logits.argmax(1)
    return float((pred == qy).float().mean().item())


# --------------------------------------------------------------
# FractalTrainer spawn (vanilla and hybrid)
# --------------------------------------------------------------
def train_episode_expert(sx, sy, *, n_way, n_steps, encoder=None, freeze_encoder=False,
                          neighbors=None, neighbor_distances=None,
                          context_scale=1.0, lr=0.01, device=None, seed=0):
    """Train a fresh ContextAwareMLP on the 25 support examples.

    If ``encoder`` is given, its fc1/fc2 are loaded into the model (and
    frozen if ``freeze_encoder=True``). This is the hybrid path.

    Returns: (model, signature)  — signature is the 500-d softmax probe.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    cs = float(context_scale) if neighbors else 0.0
    if encoder is None:
        model = ContextAwareMLP(n_classes=n_way, context_scale=cs).to(device)
        params = model.parameters()
    else:
        model = build_hybrid_expert(encoder, n_classes=n_way, context_scale=cs,
                                     freeze_encoder=freeze_encoder).to(device)
        params = trainable_parameters(model)
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for step in range(n_steps):
        perm = torch.randperm(sx.size(0), device=sx.device)
        xb, yb = sx[perm], sy[perm]
        ctx = None
        if neighbors and cs > 0.0:
            ctx = gather_context(neighbors, xb, ContextSpec(), distances=neighbor_distances)
        opt.zero_grad()
        loss = F.cross_entropy(model(xb, context=ctx), yb)
        loss.backward(); opt.step()
    model.eval()
    return model


def compute_signature(model, probe_X):
    return softmax_signature(model, probe_X)


def eval_query(model, qx, qy, *, neighbors=None, neighbor_distances=None,
               context_scale=1.0):
    model.eval()
    with torch.no_grad():
        ctx = None
        if neighbors and context_scale > 0.0:
            ctx = gather_context(neighbors, qx, ContextSpec(), distances=neighbor_distances)
        pred = model(qx, context=ctx).argmax(1)
    return float((pred == qy).float().mean().item())


# --------------------------------------------------------------
# Registry seeding
# --------------------------------------------------------------
def seed_registry(bg_ds, bg_idx, probe_X, *, device, n_episodes, encoder=None,
                   n_way=5, k_shot=5, n_query=15, n_steps=200, seed=11):
    reg = FractalRegistry()
    models = {}
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(n_episodes):
        sx, sy, _, _, chars = sample_episode(bg_ds, bg_idx, rng, n_way, k_shot, n_query, device=device)
        # Seed episodes use no context (paper arm A / cold start)
        model = train_episode_expert(sx, sy, n_way=n_way, n_steps=n_steps,
                                      encoder=encoder, freeze_encoder=(encoder is not None),
                                      neighbors=None, context_scale=0.0,
                                      device=device, seed=ep)
        sig = compute_signature(model, probe_X)
        name = f"seed_{ep:04d}"
        entry = FractalEntry(name=name, signature=sig,
                              metadata={"task": "_".join(str(c) for c in chars),
                                        "chars": list(chars), "ep": ep})
        reg.add(entry)
        models[name] = model
        if (ep + 1) % max(1, n_episodes // 10) == 0:
            print(f"  seed ep {ep+1:>4d}/{n_episodes}  reg_size={len(reg)}  [{time.time()-t0:.0f}s]")
    return reg, models


# --------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------
def pixel_knn(sx, sy, qx, qy):
    d = torch.cdist(qx, sx)
    pred = sy[d.argmin(1)]
    return float((pred == qy).float().mean().item())


def run_evaluation(*, ev_ds, ev_idx, probe_X, device, n_test_episodes, encoder,
                    vanilla_registry, vanilla_models,
                    hybrid_registry, hybrid_models,
                    n_way=5, k_shot=5, n_query=15, spawn_steps=200, context_k=3):
    rng_test = np.random.default_rng(2024)
    accs = {k: [] for k in ("pixel_knn", "protonets",
                             "fractal_vanilla", "fractal_vanilla_noctx",
                             "fractal_hybrid", "fractal_hybrid_noctx")}
    t0 = time.time()
    for ep in range(n_test_episodes):
        sx, sy, qx, qy, chars = sample_episode(ev_ds, ev_idx, rng_test, n_way, k_shot, n_query, device=device)

        # 1. pixel kNN
        accs["pixel_knn"].append(pixel_knn(sx, sy, qx, qy))
        # 2. ProtoNets
        accs["protonets"].append(protonet_eval_episode(encoder, sx, sy, qx, qy, n_way))

        # 3. Vanilla fractal (no hybrid): cold-spawn then find neighbors then warm-spawn
        cold_v = train_episode_expert(sx, sy, n_way=n_way, n_steps=spawn_steps,
                                       encoder=None, neighbors=None, context_scale=0.0,
                                       device=device, seed=ep + 10000)
        cold_v_sig = compute_signature(cold_v, probe_X)
        accs["fractal_vanilla_noctx"].append(eval_query(cold_v, qx, qy))
        res_v = vanilla_registry.find_nearest(cold_v_sig, k=context_k, query_name=f"v_{ep}")
        neighbors_v = [vanilla_models[e.name] for e in res_v.entries]
        distances_v = list(res_v.distances)
        warm_v = train_episode_expert(sx, sy, n_way=n_way, n_steps=spawn_steps,
                                       encoder=None,
                                       neighbors=neighbors_v, neighbor_distances=distances_v,
                                       context_scale=1.0, device=device, seed=ep + 10000)
        accs["fractal_vanilla"].append(eval_query(warm_v, qx, qy,
                                                    neighbors=neighbors_v,
                                                    neighbor_distances=distances_v,
                                                    context_scale=1.0))

        # 4. Hybrid fractal: same protocol but encoder=pre-trained ProtoNets encoder
        cold_h = train_episode_expert(sx, sy, n_way=n_way, n_steps=spawn_steps,
                                       encoder=encoder, freeze_encoder=True,
                                       neighbors=None, context_scale=0.0,
                                       device=device, seed=ep + 20000)
        cold_h_sig = compute_signature(cold_h, probe_X)
        accs["fractal_hybrid_noctx"].append(eval_query(cold_h, qx, qy))
        res_h = hybrid_registry.find_nearest(cold_h_sig, k=context_k, query_name=f"h_{ep}")
        neighbors_h = [hybrid_models[e.name] for e in res_h.entries]
        distances_h = list(res_h.distances)
        warm_h = train_episode_expert(sx, sy, n_way=n_way, n_steps=spawn_steps,
                                       encoder=encoder, freeze_encoder=True,
                                       neighbors=neighbors_h, neighbor_distances=distances_h,
                                       context_scale=1.0, device=device, seed=ep + 20000)
        accs["fractal_hybrid"].append(eval_query(warm_h, qx, qy,
                                                   neighbors=neighbors_h,
                                                   neighbor_distances=distances_h,
                                                   context_scale=1.0))

        if (ep + 1) % max(1, n_test_episodes // 10) == 0:
            means = {k: np.mean(v) for k, v in accs.items()}
            print(f"  test ep {ep+1:>3d}/{n_test_episodes}  "
                  f"proto={means['protonets']:.3f}  "
                  f"vanilla={means['fractal_vanilla']:.3f}  "
                  f"hybrid={means['fractal_hybrid']:.3f}  "
                  f"[{time.time()-t0:.0f}s]")
    return accs


# --------------------------------------------------------------
# Reporting
# --------------------------------------------------------------
def summarize(accs):
    summary = {}
    for name, arr in accs.items():
        a = np.asarray(arr, dtype=np.float64)
        summary[name] = {"mean": float(a.mean()), "std": float(a.std()),
                         "n": int(a.size), "accs": a.tolist()}
    return summary


def pairwise_welch(accs, pairs):
    tests = {}
    for a, b in pairs:
        aa, bb = np.asarray(accs[a]), np.asarray(accs[b])
        t, p = stats.ttest_ind(aa, bb, equal_var=False)
        tests[f"{a}_vs_{b}"] = {
            "delta": float(aa.mean() - bb.mean()),
            "welch_t": float(t),
            "welch_p": float(p),
        }
    return tests


def maybe_make_plot(summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return None

    order = ["pixel_knn", "fractal_vanilla_noctx", "fractal_vanilla",
             "fractal_hybrid_noctx", "fractal_hybrid", "protonets"]
    labels = ["pixel-kNN", "fractal\nno ctx", "fractal\n+ctx",
              "hybrid\nno ctx", "hybrid\n+ctx", "ProtoNets"]
    colors = ["gray", "lightblue", "steelblue", "peachpuff", "darkorange", "seagreen"]
    means = [summary[n]["mean"] for n in order]
    stds = [summary[n]["std"] for n in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(6), means, yerr=stds, capsize=5, color=colors)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0.2, color="gray", ls=":", alpha=0.5, label="chance (5-way)")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Omniglot 5-way 5-shot — hybrid vs vanilla vs ProtoNets")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    args = parse_args()
    if not (args.smoke or args.full):
        print("Pick one: --smoke or --full")
        sys.exit(2)
    if args.smoke:
        mode = "SMOKE"
        n_seed_episodes, n_test_episodes = 20, 10
        protonet_iters = 500
        spawn_steps = 100
    else:
        mode = "FULL"
        n_seed_episodes, n_test_episodes = 200, 100
        protonet_iters = 3000
        spawn_steps = 200

    n_way, k_shot, n_query = 5, 5, 15
    context_k = 3
    probe_size = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sprint 19c hybrid — {mode}  device={device}  seed/test eps={n_seed_episodes}/{n_test_episodes}")

    t_all = time.time()

    # --- Load Omniglot ---
    print("\n[1/5] Loading Omniglot...")
    bg_ds, ev_ds, bg_idx, ev_idx = load_omniglot(args.omniglot_root, device)
    print(f"  background classes: {len(bg_idx)}  evaluation: {len(ev_idx)}")

    # --- Probe ---
    probe_X = build_probe(bg_ds, bg_idx, probe_size, device)

    # --- ProtoNets meta-training ---
    print(f"\n[2/5] Meta-training ProtoNets encoder ({protonet_iters} iters)...")
    encoder = meta_train_protonets(bg_ds, bg_idx, device=device, n_iters=protonet_iters,
                                    n_way=n_way, k_shot=k_shot, n_query=n_query)

    # --- Seed two registries (vanilla + hybrid) ---
    print(f"\n[3/5] Seeding vanilla registry ({n_seed_episodes} episodes)...")
    vanilla_registry, vanilla_models = seed_registry(
        bg_ds, bg_idx, probe_X, device=device, n_episodes=n_seed_episodes,
        encoder=None, n_way=n_way, k_shot=k_shot, n_query=n_query,
        n_steps=spawn_steps, seed=11,
    )
    print(f"\n[4/5] Seeding hybrid registry ({n_seed_episodes} episodes)...")
    hybrid_registry, hybrid_models = seed_registry(
        bg_ds, bg_idx, probe_X, device=device, n_episodes=n_seed_episodes,
        encoder=encoder, n_way=n_way, k_shot=k_shot, n_query=n_query,
        n_steps=spawn_steps, seed=11,
    )

    # --- Test episode evaluation ---
    print(f"\n[5/5] Evaluating {n_test_episodes} test episodes across 6 methods...")
    accs = run_evaluation(
        ev_ds=ev_ds, ev_idx=ev_idx, probe_X=probe_X, device=device,
        n_test_episodes=n_test_episodes, encoder=encoder,
        vanilla_registry=vanilla_registry, vanilla_models=vanilla_models,
        hybrid_registry=hybrid_registry, hybrid_models=hybrid_models,
        n_way=n_way, k_shot=k_shot, n_query=n_query,
        spawn_steps=spawn_steps, context_k=context_k,
    )

    # --- Summarize + test ---
    summary = summarize(accs)
    pairs = [
        ("fractal_hybrid", "protonets"),
        ("fractal_hybrid", "fractal_vanilla"),
        ("fractal_hybrid", "fractal_hybrid_noctx"),
        ("fractal_vanilla", "fractal_vanilla_noctx"),
        ("protonets", "pixel_knn"),
    ]
    tests = pairwise_welch(accs, pairs)

    print("\n" + "=" * 70)
    print(f"SPRINT 19c hybrid — {mode} ({n_test_episodes} eps)")
    print("=" * 70)
    for n in ["pixel_knn", "fractal_vanilla_noctx", "fractal_vanilla",
              "fractal_hybrid_noctx", "fractal_hybrid", "protonets"]:
        s = summary[n]
        print(f"  {n:>25s}  {s['mean']:.4f} ± {s['std']:.4f}   (n={s['n']})")
    print("\nPairwise Welch's t-tests:")
    for k, v in tests.items():
        sig = "**" if v["welch_p"] < 0.01 else ("*" if v["welch_p"] < 0.05 else "(NS)")
        print(f"  {k:>40s}   Δ={v['delta']:+.4f}  t={v['welch_t']:+.2f}  p={v['welch_p']:.4f}  {sig}")

    # Pre-registered verdicts
    print("\n=== Pre-registered verdicts ===")
    delta_hp = summary["fractal_hybrid"]["mean"] - summary["protonets"]["mean"]
    delta_hv = summary["fractal_hybrid"]["mean"] - summary["fractal_vanilla"]["mean"]
    if abs(delta_hp) <= 0.03:
        print(f"  Hybrid ≈ ProtoNets (|Δ|={abs(delta_hp):.3f} ≤ 0.03)  ← BIG WIN: hybrid closes the 18.7 pp gap")
    elif delta_hp < -0.03:
        print(f"  Hybrid < ProtoNets  (Δ={delta_hp:+.3f})  ← partial recovery vs vanilla's -0.187")
    else:
        print(f"  Hybrid > ProtoNets  (Δ={delta_hp:+.3f})  ← extraordinary result, double-check for leakage")
    if delta_hv > 0.10:
        print(f"  Hybrid ≫ vanilla by {delta_hv:+.3f} — meta-trained encoder matters")
    elif delta_hv > 0:
        print(f"  Hybrid > vanilla by {delta_hv:+.3f} — minor benefit")
    else:
        print(f"  Hybrid ≤ vanilla (Δ={delta_hv:+.3f}) — meta-trained encoder did NOT help (unexpected)")

    # --- Save artifacts ---
    os.makedirs(args.out_dir, exist_ok=True)
    stem = "sprint19c_hybrid" + ("_smoke" if args.smoke else "")
    payload = {
        "config": {
            "mode": mode,
            "n_way": n_way, "k_shot": k_shot, "n_query": n_query,
            "n_seed_episodes": n_seed_episodes,
            "n_test_episodes": n_test_episodes,
            "spawn_steps": spawn_steps,
            "protonet_iters": protonet_iters,
            "context_k": context_k,
            "probe_size": probe_size,
            "device": str(device),
        },
        "summary": summary,
        "tests": tests,
    }
    out_json = os.path.join(args.out_dir, f"{stem}.json")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\n-> JSON: {out_json}")

    out_png = maybe_make_plot(summary, os.path.join(args.out_dir, f"{stem}.png"))
    if out_png: print(f"-> PNG:  {out_png}")

    print(f"\nTotal wall time: {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
