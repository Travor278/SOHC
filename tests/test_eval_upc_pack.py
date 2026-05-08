from __future__ import annotations

import numpy as np

from craic_pipeline.eval_upc_pack import (
    BalancerConfig,
    compute_balancing_metrics,
    measured_balancing_metrics,
    simulate_voltage_balancer,
)


def test_simulate_voltage_balancer_reduces_spread_for_active_case():
    """Active balancing digital twin reduces voltage spread from an imbalanced sample."""
    initial = np.array([4.12, 4.10, 4.05, 4.00, 3.96, 3.92])
    sim = simulate_voltage_balancer(initial, BalancerConfig(duration_s=1200.0, dt_s=5.0))
    metrics = compute_balancing_metrics(sim).set_index("case")

    assert metrics.loc["active_buck_boost", "final_spread_mV"] < metrics.loc["active_buck_boost", "initial_spread_mV"]
    assert metrics.loc["balancing_off", "final_spread_mV"] == metrics.loc["balancing_off", "initial_spread_mV"]
    assert metrics.loc["active_buck_boost", "max_balance_current_A"] <= 0.8 + 1e-8


def test_compute_balancing_metrics_reports_both_cases():
    """Balancing metrics include off and active cases for paper tables."""
    sim = simulate_voltage_balancer(np.array([4.1, 4.0, 3.9]), BalancerConfig(duration_s=20.0, dt_s=10.0))
    metrics = compute_balancing_metrics(sim)

    assert set(metrics["case"]) == {"balancing_off", "active_buck_boost"}
    assert {"initial_spread_mV", "final_spread_mV", "spread_reduction_pct"} <= set(metrics.columns)


def test_measured_balancing_metrics_handles_no_balancing_cycle():
    """Measured balancing summary returns NaNs for cycles without balancing labels."""
    class Cycle:
        source_path = type("P", (), {"name": "fixture.parquet"})()
        cell_voltage_flat_V = np.array([[4.0, 4.1], [4.0, 4.1]])
        semicycle = np.array(["WLTP", "Stand by"])
        time_s = np.array([0.0, 1.0])

    metrics = measured_balancing_metrics(Cycle())

    assert metrics["balancing_samples"] == 0
    assert np.isnan(metrics["balancing_spread_start_mV"])
