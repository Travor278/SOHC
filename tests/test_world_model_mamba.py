from __future__ import annotations

import numpy as np
import torch
from scipy.io import savemat

from craic_pipeline.world_model_mamba import (
    BatteryWorldModel,
    WorldModelConfig,
    build_world_tensors,
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
