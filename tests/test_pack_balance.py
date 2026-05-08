from __future__ import annotations

import numpy as np
import pandas as pd

from craic_pipeline.pack_balance import (
    PackConfig,
    apply_soc_balancer,
    compute_pack_metrics,
    initial_pack_states,
    make_cc_cv_controller,
    make_mfcc_controller,
)


def test_initial_pack_states_are_reproducible_and_cell_shaped():
    """Pack initialization returns one W3-compatible state per cell."""
    cfg = PackConfig(n_series=6, n_parallel=1)

    first = initial_pack_states(cfg, seed=2026)
    second = initial_pack_states(cfg, seed=2026)

    assert first.shape == (6, 5)
    assert np.allclose(first, second)
    assert np.all((first[:, 0] >= 0.0) & (first[:, 0] <= 1.0))
    assert np.all((first[:, 1] >= 0.0) & (first[:, 1] <= 1.0))


def test_soc_balancer_adds_current_to_lower_soc_cells():
    """SOC-spread coordinator charges low-SOC cells harder and high-SOC cells softer."""
    cfg = PackConfig(n_series=4, n_parallel=1, balance_gain_A_per_soc=10.0, max_balance_current_A=0.5)
    currents = np.full(cfg.n_cells, 3.0)
    soc = np.array([0.20, 0.24, 0.30, 0.34])

    balanced = apply_soc_balancer(currents, soc, cfg)

    assert balanced[0] > currents[0]
    assert balanced[-1] < currents[-1]
    assert np.all(balanced >= 0.0)
    assert np.all(balanced <= cfg.i_max_amps)


def test_pack_current_limit_scales_mean_current():
    """Pack-level charger current limit is enforced after balancing."""
    cfg = PackConfig(n_series=3, n_parallel=1, pack_current_limit_A=2.0)
    currents = np.full(cfg.n_cells, 4.0)
    soc = np.array([0.20, 0.21, 0.22])

    balanced = apply_soc_balancer(currents, soc, cfg)

    assert np.mean(balanced) <= 2.0 + 1e-8


def test_replicated_baseline_controllers_return_per_cell_currents():
    """CC-CV and MFCC controllers replicate single-cell logic across the pack."""
    obs = np.array(
        [
            [0.30, 1.0, 3.7, 0.0, 25.0],
            [0.55, 1.0, 3.8, 0.0, 25.0],
            [0.82, 1.0, 4.0, 0.0, 25.0],
        ],
        dtype=np.float32,
    )

    cc = make_cc_cv_controller(3.0)(obs, 0)
    mfcc = make_mfcc_controller([(4.0, 0.5), (2.0, 0.8)])(obs, 0)

    assert cc.tolist() == [3.0, 3.0, 0.0]
    assert mfcc.tolist() == [4.0, 2.0, 0.0]


def test_compute_pack_metrics_tracks_target_spread_and_safety():
    """Pack metrics summarize min-SOC target, spread, SOH, and voltage safety."""
    rows = []
    for step, socs in [(1, [0.75, 0.78]), (2, [0.81, 0.83])]:
        for cell_id, soc in enumerate(socs):
            rows.append(
                {
                    "strategy": "ours",
                    "step": step,
                    "time_s": float(step),
                    "cell_id": cell_id,
                    "soc_before": soc - 0.02,
                    "soh_before": 1.0 - 0.001 * (step - 1),
                    "soc": soc,
                    "soh": 1.0 - 0.001 * step,
                    "voltage": 4.0 + 0.01 * cell_id,
                    "temperature": 25.0 + cell_id,
                    "current_A": 2.0,
                    "balance_current_A": 0.1 * (-1 if cell_id else 1),
                    "world_model_voltage_raw": 4.0,
                    "reward": 1.0,
                    "pack_voltage": 8.0,
                }
            )
    traj = pd.DataFrame(rows)

    metrics = compute_pack_metrics(traj, soc_target=0.8)

    assert metrics["hit_target"] is True
    assert metrics["time_to_target_s"] == 2.0
    assert np.isclose(metrics["soc_spread_end"], 0.02)
    assert metrics["overvoltage_count"] == 0
    assert metrics["balance_throughput_A_s"] > 0.0
