"""Tests for fractaltrainer.observer — snapshot, projector, trainer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fractaltrainer.observer.projector import project_pca, project_random
from fractaltrainer.observer.snapshot import StreamWriter, WeightSnapshot
from fractaltrainer.observer.trainer import InstrumentedTrainer


def test_weight_snapshot_meta_roundtrip():
    snap = WeightSnapshot(step=5, flat_weights=np.arange(10, dtype=np.float32),
                          loss=0.25)
    meta = snap.to_meta()
    assert meta["step"] == 5
    assert meta["n_params"] == 10
    assert meta["loss"] == 0.25


def test_stream_writer_finalize_shape():
    with tempfile.TemporaryDirectory() as tmp:
        writer = StreamWriter(tmp, run_id="test_run", buffer_size=2)
        for step in [0, 10, 20, 30]:
            writer.append(
                WeightSnapshot(step=step, flat_weights=np.arange(5) + step,
                               loss=float(step))
            )
        meta = writer.finalize()
        assert meta["n_snapshots"] == 4

        traj = np.load(writer.traj_path)
        assert traj.shape == (4, 5)
        # First snapshot is step=0 → flat_weights == [0, 1, 2, 3, 4]
        assert np.array_equal(traj[0], np.arange(5))

        with open(writer.meta_path) as f:
            loaded = json.load(f)
        assert loaded["n_snapshots"] == 4


def test_stream_writer_double_finalize_raises():
    with tempfile.TemporaryDirectory() as tmp:
        writer = StreamWriter(tmp, run_id="dup")
        writer.append(WeightSnapshot(step=0, flat_weights=np.zeros(3), loss=0.0))
        writer.finalize()
        with pytest.raises(RuntimeError):
            writer.finalize()


def test_project_random_is_deterministic_and_preserves_shape():
    rng = np.random.RandomState(42)
    trajectory = rng.randn(50, 1000).astype(np.float32)
    a = project_random(trajectory, n_components=16, seed=7)
    b = project_random(trajectory, n_components=16, seed=7)
    c = project_random(trajectory, n_components=16, seed=8)
    assert a.shape == (50, 16)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_project_random_passthrough_when_n_components_exceeds_features():
    t = np.arange(30).reshape(6, 5).astype(np.float32)
    out = project_random(t, n_components=100, seed=0)
    assert out.shape == t.shape


def test_project_pca_shape():
    rng = np.random.RandomState(0)
    t = rng.randn(30, 100).astype(np.float32)
    out = project_pca(t, n_components=5)
    assert out.shape == (30, 5)


class _Tiny(nn.Module):
    def __init__(self, n_in=4, n_out=2):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


def _toy_loader(seed: int = 0, n: int = 256, n_in: int = 4):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, n_in).astype(np.float32)
    y = rng.randint(0, 2, size=n)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)


def test_trainer_produces_expected_snapshot_count():
    with tempfile.TemporaryDirectory() as tmp:
        model = _Tiny()
        hparams = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "init_seed": 0,
            "optimizer": "sgd",
        }
        trainer = InstrumentedTrainer(
            model=model,
            dataloader=_toy_loader(seed=0),
            loss_fn=F.cross_entropy,
            hparams=hparams,
            snapshot_every=10,
            out_dir=tmp,
            run_id="unit",
        )
        run = trainer.train(n_steps=50)
        # Expected: step 0 + 10, 20, 30, 40, 50 = 6 snapshots
        assert run.n_snapshots == 6
        traj = np.load(run.snapshot_path)
        assert traj.shape[0] == 6
        # n_params = 4*2 + 2 = 10
        assert traj.shape[1] == 10


def test_trainer_different_seeds_produce_different_trajectories():
    with tempfile.TemporaryDirectory() as tmp:
        hparams_a = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "init_seed": 0,
            "optimizer": "sgd",
        }
        hparams_b = dict(hparams_a)
        hparams_b["init_seed"] = 1

        run_a = InstrumentedTrainer(
            model=_Tiny(),
            dataloader=_toy_loader(seed=0),
            loss_fn=F.cross_entropy,
            hparams=hparams_a,
            snapshot_every=5,
            out_dir=tmp,
            run_id="a",
        ).train(n_steps=20)

        run_b = InstrumentedTrainer(
            model=_Tiny(),
            dataloader=_toy_loader(seed=1),
            loss_fn=F.cross_entropy,
            hparams=hparams_b,
            snapshot_every=5,
            out_dir=tmp,
            run_id="b",
        ).train(n_steps=20)

        ta = np.load(run_a.snapshot_path)
        tb = np.load(run_b.snapshot_path)
        assert ta.shape == tb.shape
        assert not np.allclose(ta, tb)


def test_trainer_records_snapshot_overhead():
    with tempfile.TemporaryDirectory() as tmp:
        hparams = {
            "learning_rate": 0.01,
            "batch_size": 16,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "init_seed": 0,
            "optimizer": "sgd",
        }
        run = InstrumentedTrainer(
            model=_Tiny(),
            dataloader=_toy_loader(seed=0),
            loss_fn=F.cross_entropy,
            hparams=hparams,
            snapshot_every=5,
            out_dir=tmp,
            run_id="overhead",
        ).train(n_steps=20)
        assert run.wall_clock_s > 0
        assert run.snapshot_overhead_s >= 0
        assert run.snapshot_overhead_s <= run.wall_clock_s
