from __future__ import annotations

import numpy as np
import torch
from scipy.io import savemat

from craic_pipeline.world_model_mamba import (
    BatteryWorldModel,
    WorldModelConfig,
    build_world_tensors,
    evaluate_rollout_drift,
    evaluate_randomized_directory,
    load_world_tensors,
    save_world_tensors,
    split_bundle_by_cell,
)


def test_gru_world_model_forward_shape():
    """GRU fallback predicts the four W2 next-step targets."""
    cfg = WorldModelConfig(hidden_dim=16, n_layers=1, seq_len=8, use_mamba=False)
    model = BatteryWorldModel(cfg)
    x = torch.randn(3, cfg.seq_len, cfg.state_dim + cfg.action_dim)

    y = model(x)

    assert y.shape == (3, 4)


def test_build_world_tensors_saves_schema_package(tmp_path):
    """W2 tensor builder emits `(N,L,6)` windows plus source metadata."""
    pcoe = tmp_path / "B000x"
    randomized = tmp_path / "Randomized"
    pcoe.mkdir()
    randomized.mkdir()
    cycle = {
        "data": {
            "Voltage_measured": np.linspace(4.2, 3.2, 12),
            "Current_measured": -np.ones(12),
            "Temperature_measured": np.linspace(25.0, 26.0, 12),
            "Time": np.arange(12, dtype=float),
            "Capacity": 1.0,
        }
    }
    savemat(pcoe / "B0005.mat", {"B0005": {"cycle": np.array([cycle], dtype=object)}})

    bundle = build_world_tensors(
        pcoe,
        randomized,
        seq_len=4,
        stride=2,
        max_windows=10,
        use_soc_model=False,
    )
    out = tmp_path / "world_data.pt"
    save_world_tensors(bundle, out)
    loaded = load_world_tensors(out)

    assert loaded["X"].shape[1:] == (4, 6)
    assert loaded["y"].shape[1] == 4
    assert loaded["schema"] == ["SOC", "SOH", "V", "I", "T", "action_current"]
    assert loaded["meta"]["soc_source"].startswith("strict_coulomb")
    assert loaded["traces"]


def test_split_bundle_by_cell_uses_holdout_metadata():
    """Cell split keeps requested holdout cell out of the train dataset."""
    bundle = {
        "X": np.zeros((4, 3, 6), dtype=np.float32),
        "y": np.zeros((4, 4), dtype=np.float32),
        "cell": np.array(["B0005", "B0005", "B0018", "B0018"], dtype=object),
    }

    train_ds, val_ds = split_bundle_by_cell(bundle, ["B0018"])

    assert len(train_ds) == 2
    assert len(val_ds) == 2


def test_evaluate_rollout_drift_uses_saved_traces():
    """Rollout evaluator computes multi-step V drift from bundle traces."""
    cfg = WorldModelConfig(hidden_dim=8, n_layers=1, seq_len=3, use_mamba=False)
    model = BatteryWorldModel(cfg)
    features = np.column_stack(
        [
            np.linspace(1.0, 0.9, 12),
            np.ones(12),
            np.full(12, 4.0),
            -np.ones(12),
            np.full(12, 25.0),
            -np.ones(12),
        ]
    ).astype(np.float32)
    bundle = {
        "meta": {"seq_len": 3},
        "traces": [{"subset": "pcoe", "cell": "B0018", "cycle_id": 0, "features": features}],
    }

    metrics = evaluate_rollout_drift(
        model,
        bundle,
        horizon=2,
        seq_len=3,
        stride=2,
        max_rollouts=2,
        cells=["B0018"],
        device="cpu",
    )

    assert metrics["rollouts"] == 2
    assert metrics["voltage_mae_mV"] == 0.0


def test_soc_soft_labels_falls_back_fast_without_capacity():
    """Randomized-style unlabeled data gets finite SOC labels without strict grouping."""
    from craic_pipeline.world_model_mamba import _soc_soft_labels

    import pandas as pd

    df = pd.DataFrame(
        {
            "current": -np.ones(8),
            "capacity": np.full(8, np.nan),
            "cycle_id": np.arange(8),
        }
    )

    labels = _soc_soft_labels(df, fast_when_unlabeled=True)

    assert np.isfinite(labels).all()
    assert labels[0] >= labels[-1]


def test_build_world_tensors_reuses_file_shard_cache(tmp_path):
    """W2 builder writes and reuses per-file cache shards."""
    pcoe = tmp_path / "B000x"
    randomized = tmp_path / "Randomized"
    cache = tmp_path / "cache"
    pcoe.mkdir()
    randomized.mkdir()
    cycle = {
        "data": {
            "Voltage_measured": np.linspace(4.2, 3.8, 8),
            "Current_measured": -np.ones(8),
            "Temperature_measured": np.linspace(25.0, 25.2, 8),
            "Time": np.arange(8, dtype=float),
            "Capacity": 1.0,
        }
    }
    savemat(pcoe / "B0005.mat", {"B0005": {"cycle": np.array([cycle], dtype=object)}})

    first = build_world_tensors(pcoe, randomized, seq_len=3, stride=1, cache_dir=cache, use_soc_model=False)
    second = build_world_tensors(pcoe, randomized, seq_len=3, stride=1, cache_dir=cache, use_soc_model=False)

    assert first["X"].shape == second["X"].shape
    assert list(cache.rglob("*.pt"))


def test_evaluate_randomized_directory_streams_file_shards(tmp_path):
    """Full Randomized evaluator streams one file shard without concatenating all data."""
    randomized = tmp_path / "Randomized"
    cache = tmp_path / "cache"
    randomized.mkdir()
    cycle = {
        "data": {
            "Voltage_measured": np.linspace(3.8, 3.9, 14),
            "Current_measured": np.full(14, 1.0),
            "Temperature_measured": np.linspace(25.0, 25.2, 14),
            "Time": np.arange(14, dtype=float),
        }
    }
    savemat(randomized / "RW1.mat", {"RW1": {"cycle": np.array([cycle], dtype=object)}})
    model = BatteryWorldModel(WorldModelConfig(hidden_dim=8, n_layers=1, seq_len=4, use_mamba=False))

    report = evaluate_randomized_directory(
        model,
        randomized,
        cache_dir=cache,
        seq_len=4,
        stride=2,
        rollout_horizon=2,
        rollout_stride=3,
        device="cpu",
    )

    assert report["files_total"] == 1
    assert report["files_evaluated"] == 1
    assert report["one_step_samples"] > 0
    assert np.isfinite(report["one_step"]["voltage_mae_mV"])
    assert list(cache.rglob("*.pt"))
