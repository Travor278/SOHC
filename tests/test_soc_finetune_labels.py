from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import savemat

from craic_pipeline.soc_finetune import (
    _estimate_soc_labels,
    build_pcoe_cell_split,
)


def test_strict_soc_labels_are_cycle_local_and_endpoint_calibrated():
    """Strict SOC labels start at 1 and end at the calibrated cycle endpoint."""
    df = pd.DataFrame(
        {
            "t": [0.0, 1800.0, 3600.0, 0.0, 1800.0, 3600.0],
            "voltage": [4.2, 3.7, 3.2, 4.2, 4.0, 3.8],
            "current": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            "temperature": [25.0] * 6,
            "cycle_id": [0, 0, 0, 1, 1, 1],
            "ambient_T": [25.0] * 6,
            "capacity": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )

    labels = _estimate_soc_labels(df, mode="strict")

    assert np.allclose(labels[:3], [1.0, 0.5, 0.0])
    assert np.isclose(labels[3], 1.0)
    assert np.isclose(labels[5], 0.5)
    assert np.all(np.diff(labels[:3]) <= 0.0)
    assert np.all(np.diff(labels[3:]) <= 0.0)


def test_build_pcoe_cell_split_uses_named_holdout_cells(tmp_path):
    """PCoE split builds train/holdout windows from requested cell ids."""
    for stem in ("B0005", "B0018"):
        savemat(
            tmp_path / f"{stem}.mat",
            {
                stem: {
                    "cycle": np.array(
                        [
                            {
                                "data": {
                                    "Voltage_measured": np.array([4.2, 4.0, 3.8, 3.6]),
                                    "Current_measured": np.array([-1.0, -1.0, -1.0, -1.0]),
                                    "Temperature_measured": np.array([25.0, 25.1, 25.2, 25.3]),
                                    "Time": np.array([0.0, 1.0, 2.0, 3.0]),
                                    "Capacity": 1.0,
                                }
                            }
                        ],
                        dtype=object,
                    )
                }
            },
        )

    X_train, y_train, X_val, y_val = build_pcoe_cell_split(
        tmp_path,
        window=2,
        train_cells=("B0005",),
        holdout_cells=("B0018",),
        stride=1,
        max_samples=10,
    )

    assert X_train.shape == (3, 2, 3)
    assert X_val.shape == (2, 2, 3)
    assert np.isfinite(y_train).all()
    assert np.isfinite(y_val).all()
