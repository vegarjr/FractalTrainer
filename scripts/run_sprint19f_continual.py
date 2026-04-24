"""Sprint 19f: Split-CIFAR-100 continual learning — FractalTrainer's native regime.

After Sprints 19/19b/19c/19d/19e established FractalTrainer's behavior
on Omniglot few-shot (a regime where meta-learning dominates), this
sprint tests its *native* regime: continual learning where tasks arrive
sequentially, each with full training data, and the question is
forgetting-free accumulation.

Split-CIFAR-100: split 100 classes into 10 sequential tasks of 10
classes each. Train on task 0, then task 1, ..., then task 9. At the
end evaluate on all 10 tasks' test sets.

Methods (shared small CNN backbone, matched capacity):
  1. naive_sequential   — train CNN+head sequentially, catastrophic
                          forgetting expected (lower bound)
  2. experience_replay  — naive + K=20 exemplars per task replayed
                          on each new task (standard CL baseline)
  3. **fractaltrainer** — task 0: train CNN. task 1..9: freeze CNN,
                          spawn 10-way head per task, register.
                          Inference: max-softmax-across-experts vote.
  4. joint_train        — train on all 100 classes at once (upper bound)

Metrics:
  - Average Accuracy (AA): mean accuracy across all 10 test sets after
    training on all 10 tasks (higher = better)
  - Forgetting F: accuracy on task 0 after seeing only task 0 MINUS
    accuracy on task 0 after seeing all 10 tasks (higher = more
    forgetting, 0 = no forgetting)

Pre-registered verdicts:
  - naive should have high forgetting (F > 0.3)
  - fractaltrainer should have F = 0 exactly (no param changes to
    earlier experts)
  - fractaltrainer AA vs replay AA is the key comparison
  - joint_train is the ceiling; all methods should be < joint_train
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

N_CLASSES_TOTAL = 50     # first 50 CIFAR-100 classes used for CL eval (5 tasks × 10)
N_CLASSES_PER_TASK = 10
N_TASKS = N_CLASSES_TOTAL // N_CLASSES_PER_TASK
N_PRETRAIN_CLASSES = 50  # classes 50..99 held out for CNN pretraining (analog to Omniglot meta-training)


# --------------------------------------------------------------
# Data
# --------------------------------------------------------------
class CIFAR100Subset(Dataset):
    """Wrap a torchvision dataset, filter to a class subset, remap class
    labels to 0..n-1 within the subset."""
    def __init__(self, base, class_subset):
        self.base = base
        labels = np.array(base.targets) if hasattr(base, "targets") else np.array([y for _, y in base])
        self.indices = np.where(np.isin(labels, list(class_subset)))[0]
        self._class_map = {c: i for i, c in enumerate(class_subset)}
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        x, y = self.base[int(self.indices[i])]
        return x, self._class_map[int(y)]


def load_cifar100_splits(root: str, *, samples_per_class: int | None = None,
                          device=None):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441),
                              (0.267, 0.256, 0.276)),
    ])
    train_ds = datasets.CIFAR100(root=root, train=True,  download=True, transform=tfm)
    test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)

    # Build class-assignment: deterministic split of 100 classes into:
    # [50..99] → CNN pretraining (analog to Omniglot meta-training)
    # [0..49]  → CL eval, randomly permuted into 5 tasks
    pretrain_classes = list(range(N_CLASSES_TOTAL, 100))
    rng = np.random.default_rng(42)
    cl_classes = list(rng.permutation(N_CLASSES_TOTAL))
    task_classes = [cl_classes[i * N_CLASSES_PER_TASK:(i + 1) * N_CLASSES_PER_TASK]
                    for i in range(N_TASKS)]

    train_subs, test_subs = [], []
    for classes in task_classes:
        tr = CIFAR100Subset(train_ds, classes)
        te = CIFAR100Subset(test_ds, classes)
        if samples_per_class is not None:
            idx_by_class = {c: [] for c in range(N_CLASSES_PER_TASK)}
            for i, (_, y) in enumerate(tr):
                if len(idx_by_class[y]) < samples_per_class:
                    idx_by_class[y].append(i)
            flat = sum(idx_by_class.values(), [])
            tr = Subset(tr, flat)
        train_subs.append(tr); test_subs.append(te)

    # Pretraining dataset: all samples of classes 50..99, remapped to 0..49
    pretrain_ds = CIFAR100Subset(train_ds, pretrain_classes)
    if samples_per_class is not None:
        idx_by_class = {c: [] for c in range(len(pretrain_classes))}
        for i, (_, y) in enumerate(pretrain_ds):
            if len(idx_by_class[y]) < samples_per_class:
                idx_by_class[y].append(i)
        flat = sum(idx_by_class.values(), [])
        pretrain_ds = Subset(pretrain_ds, flat)
    return task_classes, train_subs, test_subs, pretrain_ds


# --------------------------------------------------------------
# Model
# --------------------------------------------------------------
class SmallCNN(nn.Module):
    """Small CIFAR CNN: 3 conv blocks → global pool → FC."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.fc(h)

    def features_only(self, x):
        return self.features(x).flatten(1)


class FrozenBackboneHead(nn.Module):
    """Frozen SmallCNN features + fresh linear head."""
    def __init__(self, frozen_features: nn.Module, n_classes: int):
        super().__init__()
        self.features = frozen_features  # assumed already .eval() + frozen
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        with torch.no_grad():
            h = self.features(x).flatten(1)
        return self.fc(h)


# --------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------
def train_loop(model, loader, *, n_epochs, lr, device, freeze_features=False):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(n_epochs):
        model.train()
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward(); opt.step()
    return model


def eval_loader(model, loader, device):
    model.eval(); n_correct = 0; n_total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x).argmax(dim=1)
            n_correct += int((pred == y).sum().item())
            n_total += y.size(0)
    return n_correct / max(1, n_total)


# --------------------------------------------------------------
# Methods
# --------------------------------------------------------------
def pretrain_cnn(pretrain_ds, *, n_epochs, lr, batch_size, device, n_classes):
    """Jointly train a SmallCNN on the held-out pretraining dataset.
    Returns the trained CNN (features in good shape, head will be replaced)."""
    loader = DataLoader(pretrain_ds, batch_size=batch_size, shuffle=True)
    model = SmallCNN(n_classes).to(device)
    train_loop(model, loader, n_epochs=n_epochs, lr=lr, device=device)
    return model


def _fresh_cnn_from_pretrain(pretrained: SmallCNN, *, n_classes: int, device):
    """Copy pretrained features into a fresh SmallCNN with `n_classes`
    output head. Features are trainable (caller freezes if needed)."""
    model = SmallCNN(n_classes).to(device)
    model.features.load_state_dict(pretrained.features.state_dict())
    return model


def method_naive_sequential(task_train, task_test, *, n_epochs, lr, batch_size, device,
                              pretrained=None):
    """Train a single CNN+head(10) sequentially on each task. Optionally
    starts from pretrained features."""
    if pretrained is not None:
        model = _fresh_cnn_from_pretrain(pretrained, n_classes=N_CLASSES_PER_TASK, device=device)
    else:
        model = SmallCNN(N_CLASSES_PER_TASK).to(device)
    per_task_at_each_stage = []
    for i in range(len(task_train)):
        loader = DataLoader(task_train[i], batch_size=batch_size, shuffle=True)
        train_loop(model, loader, n_epochs=n_epochs, lr=lr, device=device)
        accs_row = []
        for j in range(len(task_test)):
            accs_row.append(eval_loader(model, DataLoader(task_test[j], batch_size=256), device))
        per_task_at_each_stage.append(accs_row)
    return per_task_at_each_stage


def method_experience_replay(task_train, task_test, *, n_epochs, lr, batch_size,
                              device, replay_k=20, pretrained=None):
    """Naive + K replay exemplars per task. Replay is concatenated with
    new-task data each training round."""
    if pretrained is not None:
        model = _fresh_cnn_from_pretrain(pretrained, n_classes=N_CLASSES_PER_TASK, device=device)
    else:
        model = SmallCNN(N_CLASSES_PER_TASK).to(device)
    replay_data = []  # list of (x, y) tuples with REMAPPED y
    per_task_at_each_stage = []
    rng = random.Random(0)
    for i in range(len(task_train)):
        # Build combined loader with task_train[i] + replay
        task_items = [task_train[i][j] for j in range(len(task_train[i]))]
        all_items = task_items + replay_data
        xs = torch.stack([it[0] for it in all_items])
        ys = torch.tensor([it[1] for it in all_items], dtype=torch.long)
        loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=True)
        train_loop(model, loader, n_epochs=n_epochs, lr=lr, device=device)
        # Add K exemplars from task i to replay
        task_idx = list(range(len(task_train[i])))
        rng.shuffle(task_idx)
        for j in task_idx[:replay_k * N_CLASSES_PER_TASK]:
            replay_data.append(task_train[i][j])
        accs_row = []
        for j in range(len(task_test)):
            accs_row.append(eval_loader(model, DataLoader(task_test[j], batch_size=256), device))
        per_task_at_each_stage.append(accs_row)
    return per_task_at_each_stage


def _compute_task_centroid(features_fn, task_dataset, device):
    """Mean feature embedding of all task training samples — used for
    nearest-centroid task-id routing. Standard iCaRL-style trick."""
    loader = DataLoader(task_dataset, batch_size=256, shuffle=False)
    feats = []
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            feats.append(features_fn(x))
    F_ = torch.cat(feats, dim=0)
    return F_.mean(dim=0)  # (feat_dim,)


def method_fractaltrainer(task_train, task_test, *, n_epochs, lr, batch_size, device,
                            pretrained=None):
    """Freeze pretrained CNN features → spawn a 10-way head per task.
    With pretrained=None, falls back to the old 'train task 0 then freeze'
    behavior (poor encoder for later tasks). With pretrained provided,
    features come from held-out pretraining → fair CL evaluation.

    Inference uses nearest-centroid task routing (iCaRL-style).
    """
    if pretrained is not None:
        cnn0 = _fresh_cnn_from_pretrain(pretrained, n_classes=N_CLASSES_PER_TASK, device=device)
        # Still train the task-0 head to get a sensible first-task accuracy
        task0_loader = DataLoader(task_train[0], batch_size=batch_size, shuffle=True)
        # Only the fc trains; features frozen
        for p in cnn0.features.parameters():
            p.requires_grad = False
        train_loop(cnn0, task0_loader, n_epochs=n_epochs, lr=lr, device=device)
    else:
        task0_loader = DataLoader(task_train[0], batch_size=batch_size, shuffle=True)
        cnn0 = SmallCNN(N_CLASSES_PER_TASK).to(device)
        train_loop(cnn0, task0_loader, n_epochs=n_epochs, lr=lr, device=device)
    # Freeze features (no-op if already frozen)
    frozen_features = cnn0.features
    for p in frozen_features.parameters():
        p.requires_grad = False
    frozen_features.eval()

    def features_fn(x):
        return frozen_features(x).flatten(1)

    # Registry of (head model, centroid) pairs
    heads = [cnn0]
    centroids = [_compute_task_centroid(features_fn, task_train[0], device)]

    # Stage 0: after training task 0 only, only task 0 can be predicted
    stage0_row = []
    for j in range(len(task_test)):
        if j == 0:
            stage0_row.append(eval_loader(cnn0, DataLoader(task_test[j], batch_size=256), device))
        else:
            stage0_row.append(_chance_accuracy())
    per_task_at_each_stage = [stage0_row]

    # Tasks 1..N
    for i in range(1, len(task_train)):
        new_head = FrozenBackboneHead(frozen_features, N_CLASSES_PER_TASK).to(device)
        loader = DataLoader(task_train[i], batch_size=batch_size, shuffle=True)
        train_loop(new_head, loader, n_epochs=n_epochs, lr=lr, device=device)
        heads.append(new_head)
        centroids.append(_compute_task_centroid(features_fn, task_train[i], device))
        accs_row = fractal_eval(heads, centroids, features_fn, task_test, device)
        per_task_at_each_stage.append(accs_row)
    return per_task_at_each_stage


def fractal_eval(heads, centroids, features_fn, task_test, device):
    """Two-stage eval:
       1) Route by nearest feature centroid → predicted task id.
       2) Use that task's head to classify within-task.
    A query is correct iff (predicted_task == true_task) AND
    (predicted_class_within == true_class_within).
    """
    C = torch.stack(centroids, dim=0)  # (n_tasks_seen, feat_dim)
    accs_row = []
    for j in range(len(task_test)):
        loader = DataLoader(task_test[j], batch_size=256)
        n_correct, n_total = 0, 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            with torch.no_grad():
                # Feature embedding for routing
                f = features_fn(x)              # (B, feat_dim)
                d = torch.cdist(f, C)           # (B, n_tasks)
                pred_head_id = d.argmin(dim=1)  # (B,)
                # Forward through EACH head once; we'll pick the right row
                per_head_preds = []
                for h in heads:
                    h.eval()
                    per_head_preds.append(h(x).argmax(dim=1))  # (B,)
                # Gather the predicted class from the routed head per sample
                per_head_preds = torch.stack(per_head_preds, dim=1)  # (B, n_heads)
                pred_class = per_head_preds.gather(1, pred_head_id.unsqueeze(1)).squeeze(1)
                correct = (pred_head_id == j) & (pred_class == y)
                n_correct += int(correct.sum().item())
                n_total += y.size(0)
        accs_row.append(n_correct / max(1, n_total))
    return accs_row


def _chance_accuracy():
    return 1.0 / N_CLASSES_PER_TASK  # 0.1 for 10-way


def method_joint_train(task_train, task_test, *, n_epochs, lr, batch_size,
                        device, task_classes, pretrained=None):
    """Joint training on all N_CLASSES_TOTAL classes. Upper bound.
    Labels in each task_train subset are 0..9 (task-local); we remap them
    to 0..N_CLASSES_TOTAL-1 for joint training."""
    merged_items = []
    for i, ds in enumerate(task_train):
        for j in range(len(ds)):
            x, y_local = ds[j]
            y_global = i * N_CLASSES_PER_TASK + y_local
            merged_items.append((x, y_global))
    xs = torch.stack([it[0] for it in merged_items])
    ys = torch.tensor([it[1] for it in merged_items], dtype=torch.long)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=True)
    if pretrained is not None:
        model = _fresh_cnn_from_pretrain(pretrained, n_classes=N_CLASSES_TOTAL, device=device)
    else:
        model = SmallCNN(N_CLASSES_TOTAL).to(device)
    train_loop(model, loader, n_epochs=n_epochs, lr=lr, device=device)
    # Eval: for test set of task j, the correct class in global label-space is j * 10 + y_local
    accs_row = []
    for j in range(len(task_test)):
        model.eval()
        n_correct, n_total = 0, 0
        for x, y_local in DataLoader(task_test[j], batch_size=256):
            x = x.to(device); y_local = y_local.to(device)
            with torch.no_grad():
                pred = model(x).argmax(dim=1)
                correct = (pred == (j * N_CLASSES_PER_TASK + y_local))
            n_correct += int(correct.sum().item())
            n_total += y_local.size(0)
        accs_row.append(n_correct / max(1, n_total))
    return [accs_row]  # single-stage: just after full training


# --------------------------------------------------------------
# Metrics
# --------------------------------------------------------------
def compute_metrics(per_task_at_each_stage, method_name):
    """Returns {'AA': average accuracy across tasks after final training,
               'F':  forgetting (acc on task 0 after task 0 minus acc
                     on task 0 after all tasks).}"""
    if method_name == "joint_train":
        # Single stage
        final_accs = per_task_at_each_stage[0]
        return {
            "AA": float(np.mean(final_accs)),
            "F":  0.0,
            "final_accs_per_task": [float(a) for a in final_accs],
            "n_stages": 1,
        }
    final_accs = per_task_at_each_stage[-1]  # after training on final task
    task0_when_learned = per_task_at_each_stage[0][0]  # accuracy on task 0 immediately after training task 0
    task0_now = final_accs[0]
    return {
        "AA": float(np.mean(final_accs)),
        "F": float(task0_when_learned - task0_now),
        "final_accs_per_task": [float(a) for a in final_accs],
        "task0_when_learned": float(task0_when_learned),
        "task0_after_all": float(task0_now),
        "n_stages": len(per_task_at_each_stage),
    }


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--cifar-root", default="/tmp/cifar100")
    p.add_argument("--out-dir", default=RESULTS)
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.smoke or args.full):
        print("Pick --smoke or --full"); sys.exit(2)
    if args.smoke:
        mode, spc, n_epochs = "SMOKE", 100, 3
    else:
        mode, spc, n_epochs = "FULL", 300, 5

    batch_size = 64
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sprint 19f CONTINUAL — {mode}  device={device}  samples/class={spc}  epochs/task={n_epochs}")

    t_all = time.time()

    print("\n[1/6] Loading CIFAR-100 Split...")
    task_classes, task_train, task_test, pretrain_ds = load_cifar100_splits(
        args.cifar_root, samples_per_class=spc, device=device)
    print(f"  CL tasks: {len(task_train)} × {N_CLASSES_PER_TASK} classes  "
          f"(first {N_CLASSES_TOTAL}); pretrain on held-out {N_PRETRAIN_CLASSES} classes "
          f"({len(pretrain_ds)} samples)")

    print(f"\n[2/6] Pretraining CNN on held-out 50 CIFAR-100 classes...")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    pretrained = pretrain_cnn(pretrain_ds, n_epochs=n_epochs, lr=lr,
                                batch_size=batch_size, device=device,
                                n_classes=N_PRETRAIN_CLASSES)
    print(f"  pretraining done  [{time.time()-t_all:.0f}s]")

    results = {}

    print(f"\n[3/6] naive_sequential (from pretrained)...")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    per_stage = method_naive_sequential(task_train, task_test, n_epochs=n_epochs,
                                          lr=lr, batch_size=batch_size, device=device,
                                          pretrained=pretrained)
    results["naive_sequential"] = {"per_stage": per_stage, "metrics": compute_metrics(per_stage, "naive")}
    print(f"  AA={results['naive_sequential']['metrics']['AA']:.3f}  "
          f"F={results['naive_sequential']['metrics']['F']:.3f}  "
          f"[{time.time()-t_all:.0f}s]")

    print(f"\n[4/6] experience_replay (from pretrained, K=20 per task)...")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    per_stage = method_experience_replay(task_train, task_test, n_epochs=n_epochs,
                                           lr=lr, batch_size=batch_size, device=device,
                                           replay_k=20, pretrained=pretrained)
    results["experience_replay"] = {"per_stage": per_stage, "metrics": compute_metrics(per_stage, "replay")}
    print(f"  AA={results['experience_replay']['metrics']['AA']:.3f}  "
          f"F={results['experience_replay']['metrics']['F']:.3f}  "
          f"[{time.time()-t_all:.0f}s]")

    print(f"\n[5/6] fractaltrainer (frozen pretrained, per-task heads)...")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    per_stage = method_fractaltrainer(task_train, task_test, n_epochs=n_epochs,
                                        lr=lr, batch_size=batch_size, device=device,
                                        pretrained=pretrained)
    results["fractaltrainer"] = {"per_stage": per_stage, "metrics": compute_metrics(per_stage, "fractal")}
    print(f"  AA={results['fractaltrainer']['metrics']['AA']:.3f}  "
          f"F={results['fractaltrainer']['metrics']['F']:.3f}  "
          f"[{time.time()-t_all:.0f}s]")

    print(f"\n[6/6] joint_train (upper bound, from pretrained)...")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    per_stage = method_joint_train(task_train, task_test, n_epochs=max(n_epochs, 5),
                                     lr=lr, batch_size=batch_size, device=device,
                                     task_classes=task_classes, pretrained=pretrained)
    results["joint_train"] = {"per_stage": per_stage, "metrics": compute_metrics(per_stage, "joint_train")}
    print(f"  AA={results['joint_train']['metrics']['AA']:.3f}  "
          f"[{time.time()-t_all:.0f}s]")

    # Report
    print("\n" + "=" * 70)
    print(f"SPRINT 19f — Split-CIFAR-100 continual ({mode})")
    print("=" * 70)
    print(f"{'method':>22s}  {'AA':>6s}  {'F':>6s}  per-task-accs")
    print("-" * 70)
    for name in ["naive_sequential", "experience_replay", "fractaltrainer", "joint_train"]:
        m = results[name]["metrics"]
        per = " ".join(f"{a:.2f}" for a in m["final_accs_per_task"])
        print(f"{name:>22s}  {m['AA']:.3f}  {m['F']:+.3f}  {per}")

    # Pre-registered verdicts
    print("\n=== Pre-registered verdicts ===")
    ft = results["fractaltrainer"]["metrics"]
    naive = results["naive_sequential"]["metrics"]
    replay = results["experience_replay"]["metrics"]
    joint = results["joint_train"]["metrics"]
    print(f"  naive has high forgetting (F={naive['F']:+.3f}, expected F > 0.3): "
          f"{'✓' if naive['F'] > 0.3 else '✗'}")
    print(f"  fractaltrainer has zero forgetting (F={ft['F']:+.3f}, expected ~0): "
          f"{'✓' if abs(ft['F']) < 0.02 else '✗'}")
    print(f"  fractaltrainer vs replay AA: Δ={ft['AA']-replay['AA']:+.3f}")
    print(f"  all methods below joint_train ({joint['AA']:.3f}): "
          f"naive={naive['AA']<joint['AA']}, replay={replay['AA']<joint['AA']}, "
          f"fractal={ft['AA']<joint['AA']}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    stem = "sprint19f_continual" + ("_smoke" if args.smoke else "")
    payload = {
        "config": {"mode": mode, "n_tasks": N_TASKS, "classes_per_task": N_CLASSES_PER_TASK,
                    "samples_per_class": spc, "n_epochs_per_task": n_epochs,
                    "batch_size": batch_size, "lr": lr, "device": str(device)},
        "results": results,
    }
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"-> {stem}.json")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # (a) AA comparison
        ax = axes[0]
        order = ["naive_sequential", "experience_replay", "fractaltrainer", "joint_train"]
        means = [results[n]["metrics"]["AA"] for n in order]
        ax.bar(range(len(order)), means, color=["tomato", "peachpuff", "darkorange", "seagreen"])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([n.replace("_", "\n") for n in order], fontsize=9)
        ax.set_ylabel("Average accuracy (AA)"); ax.set_ylim(0, 1.02)
        ax.set_title("Split-CIFAR-100 — final average accuracy")
        ax.grid(axis="y", alpha=0.3)
        # (b) Task-0 accuracy trajectory
        ax = axes[1]
        for name in ["naive_sequential", "experience_replay", "fractaltrainer"]:
            per_stage = results[name]["per_stage"]
            task0_traj = [stage[0] for stage in per_stage]
            ax.plot(range(len(task0_traj)), task0_traj, "o-", label=name)
        ax.set_xlabel("Training stage (task)"); ax.set_ylabel("Task-0 test accuracy")
        ax.set_title("Task-0 accuracy as later tasks are learned (forgetting)")
        ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print(f"\nTotal: {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
