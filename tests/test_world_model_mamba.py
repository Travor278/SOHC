from __future__ import annotations

import torch

from craic_pipeline.world_model_mamba import BatteryWorldModel, WorldModelConfig


def test_gru_world_model_forward_shape():
    """GRU fallback predicts the four W2 next-step targets."""
    cfg = WorldModelConfig(hidden_dim=16, n_layers=1, seq_len=8, use_mamba=False)
    model = BatteryWorldModel(cfg)
    x = torch.randn(3, cfg.seq_len, cfg.state_dim + cfg.action_dim)

    y = model(x)

    assert y.shape == (3, 4)
