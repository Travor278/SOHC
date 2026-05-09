from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from craic_pipeline.eval_compare import (
    _current_to_action,
    _paired_against_cc_cv,
    _parse_mfcc_stages,
    compute_metrics,
    measure_closed_loop_replay_error,
)


def test_current_to_action_maps_amp_range():
    """W4 current helper maps physical amps to normalized RL actions."""
    assert _current_to_action(0.0, 5.0) == -1.0
    assert _current_to_action(2.5, 5.0) == 0.0
    assert _current_to_action(5.0, 5.0) == 1.0
    assert _current_to_action(7.5, 5.0) == 1.0


def test_parse_mfcc_stages_validates_nonempty_list():
    """MFCC CLI parser accepts `I:soc` stage strings."""
    assert _parse_mfcc_stages("4.0:0.5,2.0:0.8") == [(4.0, 0.5), (2.0, 0.8)]
    with pytest.raises(ValueError):
        _parse_mfcc_stages("")


def test_compute_metrics_tracks_w4_targets():
    """W4 metrics include target time, degradation, safety, and temperature."""
    df = pd.DataFrame(
        {
            "time_s": [1.0, 2.0, 3.0],
            "soc_before": [0.70, 0.75, 0.79],
            "soc": [0.75, 0.79, 0.81],
            "soh_before": [1.0, 0.999, 0.998],
            "soh": [0.999, 0.998, 0.997],
            "voltage": [4.0, 4.1, 4.2],
            "world_model_voltage_raw": [4.0, 4.1, 4.21],
            "temperature": [25.0, 26.0, 27.0],
            "current_A": [3.0, 2.0, 1.0],
            "model_delta_soh": [1e-6, 3e-6, 2e-6],
            "aging_proxy_delta_soh": [2e-6, 2e-6, 4e-6],
            "reward": [1.0, 1.0, 1.0],
        }
    )

    metrics = compute_metrics(df, soc_target=0.8, v_min=2.5, v_max=4.2)

    assert metrics["time_to_80_s"] == 3.0
    assert metrics["hit_target"] is True
    assert np.isclose(metrics["delta_soh"], 0.003)
    assert metrics["overvoltage_count"] == 0
    assert metrics["raw_overvoltage_count"] == 1
    assert metrics["mean_T"] == 26.0
    assert metrics["model_dominated_steps"] == 1
    assert metrics["proxy_dominated_steps"] == 2
    assert metrics["aging_proxy_delta_soh_sum"] > metrics["model_delta_soh_sum"]


def test_paired_against_cc_cv_reports_relative_w4_targets():
    """Paired W4 summary compares only episodes where both strategies hit 80% SOC."""
    metrics = pd.DataFrame(
        [
            {"strategy": "cc_cv", "episode": 0, "hit_target": True, "time_to_80_s": 600.0, "delta_soh": 0.002, "overvoltage_count": 0},
            {"strategy": "ours", "episode": 0, "hit_target": True, "time_to_80_s": 400.0, "delta_soh": 0.0015, "overvoltage_count": 0},
            {"strategy": "cc_cv", "episode": 1, "hit_target": False, "time_to_80_s": np.nan, "delta_soh": 0.002, "overvoltage_count": 0},
            {"strategy": "ours", "episode": 1, "hit_target": True, "time_to_80_s": 500.0, "delta_soh": 0.0015, "overvoltage_count": 0},
        ]
    )

    paired = _paired_against_cc_cv(metrics)

    assert paired.loc[0, "paired_episodes"] == 1
    assert paired.loc[0, "speed_improvement_pct"] > 30.0
    assert paired.loc[0, "delta_soh_reduction_pct"] > 20.0


def test_measure_closed_loop_replay_error_reports_horizon_metrics():
    """Closed-loop replay metric rolls the model on predicted states."""

    class PersistenceModel(torch.nn.Module):
        """Tiny model that keeps SOC/V/T unchanged and reports no aging."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            last = x[:, -1]
            return torch.stack([last[:, 0], last[:, 2], last[:, 4], torch.zeros_like(last[:, 0])], dim=-1)

    trajectory = pd.DataFrame(
        {
            "soc": [0.2] * 8,
            "soh": [1.0] * 8,
            "voltage": [3.7] * 8,
            "current_A": [1.0] * 8,
            "temperature": [25.0] * 8,
        }
    )

    metrics = measure_closed_loop_replay_error(PersistenceModel(), trajectory, seq_len=3, horizons=(2, 5))

    assert metrics["closed_loop_replay_available_steps"] == 5
    assert np.isclose(metrics["closed_loop_2step_V_MAE_mV"], 0.0, atol=1e-3)
    assert metrics["closed_loop_5step_V_samples"] == 5
