from __future__ import annotations

import numpy as np
from scipy.io import savemat

from craic_pipeline.nasa_loader import (
    load_arc_fy08q4,
    load_pcoe_basic,
    load_randomized_usage,
)


def _write_mat(path, root_name, cycles):
    """Write a tiny NASA-like MATLAB file for loader tests."""
    savemat(path, {root_name: {"cycle": np.array(cycles, dtype=object)}})


def test_load_pcoe_basic_returns_unified_arrays(tmp_path):
    """PCoE loader returns aligned V/I/T/time/metadata arrays."""
    mat_path = tmp_path / "B0005.mat"
    _write_mat(
        mat_path,
        "B0005",
        [
            {
                "type": "discharge",
                "ambient_temperature": 24,
                "data": {
                    "Voltage_measured": np.array([4.1, 4.0, 3.9]),
                    "Current_measured": np.array([1.0, 1.0, 1.0]),
                    "Temperature_measured": np.array([24.0, 24.2, 24.4]),
                    "Time": np.array([0.0, 1.0, 2.0]),
                    "Capacity": np.array([[1.82]]),
                },
            }
        ],
    )

    V, I, T, t, cycle_id, ambient, capacity = load_pcoe_basic(mat_path)

    assert V.shape == I.shape == T.shape == t.shape == cycle_id.shape == ambient.shape == capacity.shape
    assert V.tolist() == [4.1, 4.0, 3.9]
    assert ambient.tolist() == [24.0, 24.0, 24.0]
    assert capacity.tolist() == [1.82, 1.82, 1.82]


def test_load_arc_fy08q4_uses_ambient_temperature_when_temperature_missing(tmp_path):
    """ARC loader fills missing temperature samples from ambient metadata."""
    mat_path = tmp_path / "B0025_43C.mat"
    _write_mat(
        mat_path,
        "B0025",
        [
            {
                "ambient_temperature": 43,
                "data": {
                    "Voltage_measured": np.array([4.2, 4.1]),
                    "Current_measured": np.array([2.0, 2.0]),
                    "Time": np.array([0.0, 10.0]),
                    "Capacity": 1.7,
                },
            }
        ],
    )

    V, I, T, t, cycle_id, ambient, capacity = load_arc_fy08q4(mat_path)

    assert V.shape == (2,)
    assert np.allclose(T, [43.0, 43.0])
    assert np.allclose(ambient, [43.0, 43.0])
    assert np.allclose(capacity, [1.7, 1.7])


def test_load_randomized_usage_filters_large_current_steps(tmp_path):
    """Randomized loader splits cycles at current jumps larger than 1 A."""
    mat_path = tmp_path / "RW1.mat"
    _write_mat(
        mat_path,
        "RW1",
        [
            {
                "ambient_temperature": 24,
                "data": {
                    "Voltage": np.array([4.0, 3.99, 3.97, 3.96]),
                    "Current": np.array([0.5, 0.7, 2.2, 2.4]),
                    "Temperature": np.array([25.0, 25.1, 25.3, 25.4]),
                    "Time": np.array([0.0, 1.0, 2.0, 3.0]),
                    "Capacity": 1.6,
                },
            }
        ],
    )

    V, I, T, t, cycle_id, ambient, capacity = load_randomized_usage(mat_path)

    assert V.shape == (4,)
    assert np.allclose(I, [0.5, 0.7, 2.2, 2.4])
    assert cycle_id.tolist() == [0.0, 0.0, 1.0, 1.0]
    for cid in np.unique(cycle_id):
        idx = np.where(cycle_id == cid)[0]
        assert np.all(np.abs(np.diff(I[idx])) <= 1.0)
