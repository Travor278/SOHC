from __future__ import annotations

from craic_pipeline.ecm_safety_layer import ECMParams, ECMSafetyLayer


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
