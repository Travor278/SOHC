from __future__ import annotations

import numpy as np
import pandas as pd

from craic_pipeline.pack_dataset_upc import (
    N_CELLS,
    cell_labels,
    export_simulink_csv,
    load_upc_pack_cycle,
    summarize_upc_pack_cycle,
    summarize_upc_pack_directory,
)


def _write_upc_fixture(path):
    """Create a tiny UPC-like Parquet file for loader tests."""
    n = 5
    data = {
        "Cycle": ["Cycle_7"] * n,
        "Semicycle": ["WLTP", "WLTP", "Balancing", "Stand by", "Constant Charge"],
        "Timestamp": pd.date_range("2025-01-01", periods=n, freq="s"),
        "Current_Actual_Battery [A]": np.linspace(-2.0, 1.0, n).astype("float32"),
        "Voltage_Actual_Battery [V]": np.linspace(40.0, 45.0, n).astype("float32"),
        "SoC_Actual_Battery [percent]": np.linspace(40.0, 80.0, n).astype("float32"),
        "Temperature_IN_Chamber [degC]": np.full(n, 25.0, dtype="float32"),
        "Temperature_OUT_Chamber [degC]": np.full(n, 24.0, dtype="float32"),
        "Humidity_IN_Chamber [RH_percent]": np.full(n, 50.0, dtype="float32"),
        "Resistance_Actual_Battery_IT5101 [Ohm]": np.full(n, 0.1, dtype="float64"),
    }
    for p in range(1, 4):
        data[f"Current_Actual_P{p} [A]"] = np.linspace(-1.0, 0.5, n).astype("float32") + p * 0.1
        data[f"Voltage_Actual_P{p} [V]"] = np.linspace(40.0, 45.0, n).astype("float32") + p * 0.01
        for s in range(1, 13):
            data[f"Voltage_Cell_P{p}S{s} [V]"] = np.full(n, 3.5 + 0.01 * s + 0.001 * p, dtype="float32")
            data[f"Temperature_Cell_Top_P{p}S{s} [degC]"] = np.full(n, 25.0 + p, dtype="float32")
            data[f"Temperature_Cell_Bottom_P{p}S{s} [degC]"] = np.full(n, 26.0 + s * 0.01, dtype="float32")
    pd.DataFrame(data).to_parquet(path, index=False)


def test_load_upc_pack_cycle_shapes_and_fields(tmp_path):
    """UPC loader returns aligned 12S3P arrays and normalized BMS SOC."""
    path = tmp_path / "Qtzl_Cycle_007_WLTP_partial_data.parquet"
    _write_upc_fixture(path)

    cycle = load_upc_pack_cycle(path)

    assert cycle.cycle_id == "Cycle_7"
    assert cycle.cycle_type == "WLTP"
    assert cycle.time_s.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert cycle.cell_voltage_V.shape == (5, 3, 12)
    assert cycle.branch_current_A.shape == (5, 3)
    assert cycle.cell_temperature_C.shape == (5, 3, 12, 2)
    assert cycle.cell_voltage_flat_V.shape == (5, N_CELLS)
    assert cycle.cell_temperature_mean_C.shape == (5, N_CELLS)
    assert np.isclose(cycle.bms_soc[0], 0.4)
    assert np.isclose(cycle.bms_soc[-1], 0.8)


def test_summarize_upc_pack_directory_and_balancing_count(tmp_path):
    """UPC summary records spread, semicycle names, and balancing samples."""
    path = tmp_path / "Qtzl_Cycle_007_WLTP_partial_data.parquet"
    _write_upc_fixture(path)

    summary = summarize_upc_pack_directory(tmp_path)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["cycle_type"] == "WLTP"
    assert row["balancing_samples"] == 1
    assert row["cell_voltage_spread_max_mV"] > 0.0
    assert "Balancing" in row["semicycles"]


def test_export_simulink_csv_uses_stable_column_names(tmp_path):
    """Simulink CSV export avoids brackets/spaces and includes all cell channels."""
    path = tmp_path / "Qtzl_Cycle_007_WLTP_partial_data.parquet"
    out = tmp_path / "simulink.csv"
    _write_upc_fixture(path)
    cycle = load_upc_pack_cycle(path)

    export_simulink_csv(cycle, out)
    exported = pd.read_csv(out)

    assert "time_s" in exported.columns
    assert "branch_current_P1_A" in exported.columns
    assert "cell_voltage_P1S1_V" in exported.columns
    assert "cell_temperature_P3S12_C" in exported.columns
    assert len([col for col in exported.columns if col.startswith("cell_voltage_")]) == len(cell_labels())
