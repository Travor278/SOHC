from __future__ import annotations

import numpy as np

from craic_pipeline.ecm_safety_layer import ECMParams, ECMSafetyLayer, cross_check_against_matlab, load_params_from_mat


def test_ecm_project_keeps_voltage_inside_bounds():
    """ECM projection clips an unsafe current to the voltage boundary."""
    params = ECMParams(
        R0=0.05,
        R1=0.01,
        R2=0.02,
        C1=1000.0,
        C2=1000.0,
        ocv_coeffs=(3.7,),
        V_min=3.0,
        V_max=4.0,
    )
    layer = ECMSafetyLayer(params, dt=1.0)

    current = layer.project(soc=0.5, action_current=30.0)
    layer.reset()
    voltage = layer.predict_voltage(soc=0.5, current=current)

    assert voltage >= params.V_min - 1e-9
    assert voltage <= params.V_max + 1e-9


def test_cross_check_against_matlab_reference_is_millivolt_close():
    """ECM Python implementation matches an independent MATLAB-form reference."""
    metrics = cross_check_against_matlab()

    assert metrics["max_abs_error_mV"] < 1.0


def test_ecm_random_projection_keeps_1000_actions_safe():
    """Projection keeps random action currents inside configured voltage bounds."""
    params = load_params_from_mat()
    params.V_min = 2.8
    params.V_max = 4.2
    layer = ECMSafetyLayer(params, dt=0.2)
    rng = np.random.default_rng(2026)

    for _ in range(1000):
        soc = float(rng.uniform(0.05, 0.95))
        action_current = float(rng.uniform(-20.0, 20.0))
        current = layer.project(soc=soc, action_current=action_current)
        voltage = layer._ocv(soc) - current * params.R0 - layer.V1 - layer.V2

        assert voltage >= params.V_min - 1e-8
        assert voltage <= params.V_max + 1e-8
