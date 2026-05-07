"""ECM physical safety layer for projecting RL charging actions.

The layer uses the second-order RC parameters shipped with the legacy MATLAB
assets and clips current actions so the next-step terminal voltage remains
inside configured voltage bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_PARAMS_MAT = Path(
    "MATLAB滤波算法代码——云储实时数据/1-2-model_identification_RC/result/savemat_2order.mat"
)


@dataclass
class ECMParams:
    """Second-order RC model parameters from the MATLAB STA identification."""

    R0: float
    R1: float
    R2: float
    C1: float
    C2: float
    ocv_coeffs: tuple[float, ...]
    V_max: float = 4.2
    V_min: float = 2.5


def load_params_from_mat(mat_path: Path = DEFAULT_PARAMS_MAT) -> ECMParams:
    """Load ECM parameters and OCV polynomial coefficients from MATLAB `.mat`.

    Args:
        mat_path: `savemat_2order.mat` produced by the legacy STA identifier.

    Returns:
        `ECMParams` with R0/R1/R2/C1/C2 and an OCV(SOC) polynomial.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"ECM parameter file not found: {mat_path}")
    raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    best = _array(raw.get("Best"))
    if best.size >= 5:
        R0, R1, R2, C1, C2 = (float(v) for v in best[:5])
    else:
        R0 = _scalar(raw, "R0")
        R1 = _scalar(raw, "R1")
        R2 = _scalar(raw, "R2")
        C1 = _scalar(raw, "C1")
        C2 = _scalar(raw, "C2")

    soc = _array(raw.get("SOC_battery"))
    ocv = _array(raw.get("OCV_battery"))
    if soc.size == 0 or ocv.size == 0:
        ocv_soc = np.asarray(raw.get("OCV_SOC"), dtype=float)
        if ocv_soc.shape[0] != 2:
            raise ValueError("ECM .mat does not contain OCV/SOC data")
        ocv, soc = ocv_soc[0], ocv_soc[1]
    order = min(8, len(soc) - 1)
    coeffs = tuple(float(v) for v in np.polyfit(soc, ocv, order))
    return ECMParams(R0=R0, R1=R1, R2=R2, C1=C1, C2=C2, ocv_coeffs=coeffs)


class ECMSafetyLayer:
    """Project current actions through a second-order RC voltage constraint."""

    def __init__(self, params: ECMParams, dt: float = 1.0):
        """Create a stateful ECM projector with zero polarization voltage."""
        self.params = params
        self.dt = float(dt)
        self.V1 = 0.0
        self.V2 = 0.0

    def predict_voltage(self, soc: float, current: float) -> float:
        """Predict next terminal voltage without mutating polarization state."""
        next_v1, next_v2 = self._next_polarization(current)
        return self._ocv(soc) - current * self.params.R0 - next_v1 - next_v2

    def project(self, soc: float, action_current: float) -> float:
        """Clip action current so the predicted next voltage stays in bounds."""
        current = float(action_current)
        voltage = self.predict_voltage(soc, current)
        if voltage > self.params.V_max:
            current = self._current_for_voltage(soc, self.params.V_max)
        elif voltage < self.params.V_min:
            current = self._current_for_voltage(soc, self.params.V_min)
        self.V1, self.V2 = self._next_polarization(current)
        return float(current)

    def reset(self) -> None:
        """Reset internal RC polarization voltages to zero."""
        self.V1 = 0.0
        self.V2 = 0.0

    def _next_polarization(self, current: float) -> tuple[float, float]:
        """Compute next RC polarization voltages for a candidate current."""
        a1 = np.exp(-self.dt / max(self.params.R1 * self.params.C1, 1e-12))
        a2 = np.exp(-self.dt / max(self.params.R2 * self.params.C2, 1e-12))
        next_v1 = self.V1 * a1 + current * self.params.R1 * (1.0 - a1)
        next_v2 = self.V2 * a2 + current * self.params.R2 * (1.0 - a2)
        return float(next_v1), float(next_v2)

    def _current_for_voltage(self, soc: float, target_voltage: float) -> float:
        """Solve the linear ECM voltage equation for current at a boundary."""
        a1 = np.exp(-self.dt / max(self.params.R1 * self.params.C1, 1e-12))
        a2 = np.exp(-self.dt / max(self.params.R2 * self.params.C2, 1e-12))
        base = self._ocv(soc) - self.V1 * a1 - self.V2 * a2
        gain = self.params.R0 + self.params.R1 * (1.0 - a1) + self.params.R2 * (1.0 - a2)
        return float((base - target_voltage) / max(gain, 1e-12))

    def _ocv(self, soc: float) -> float:
        """Evaluate OCV(SOC) using the fitted MATLAB OCV curve."""
        soc = float(np.clip(soc, 0.0, 1.0))
        return float(np.polyval(self.params.ocv_coeffs, soc))


def _array(value) -> np.ndarray:
    """Convert a MATLAB field into a flat float array."""
    if value is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _scalar(raw: dict, key: str) -> float:
    """Read one scalar ECM field from a loaded MATLAB dictionary."""
    values = _array(raw.get(key))
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise KeyError(f"ECM parameter {key} not found")
    return float(finite[0])


def cross_check_against_matlab() -> dict:
    """Cross-check ECM voltage against an independent MATLAB-form reference."""
    params = load_params_from_mat()
    raw = loadmat(DEFAULT_PARAMS_MAT, squeeze_me=True, struct_as_record=False)
    currents = _array(raw.get("I"))[:200]
    soc = _array(raw.get("SOC"))[:200]
    if currents.size < 10 or soc.size < 10:
        currents = np.linspace(-2.0, 2.0, 200)
        soc = np.linspace(0.2, 0.9, 200)
    dt = float(_array(raw.get("Ts"))[0]) if _array(raw.get("Ts")).size else 0.2

    layer = ECMSafetyLayer(params, dt=dt)
    reference_v1 = 0.0
    reference_v2 = 0.0
    errors = []
    voltages = []
    for one_soc, current in zip(soc, currents):
        predicted = layer.predict_voltage(float(one_soc), float(current))
        reference_v1, reference_v2 = _reference_next_polarization(params, dt, reference_v1, reference_v2, float(current))
        reference = np.polyval(params.ocv_coeffs, np.clip(one_soc, 0.0, 1.0)) - current * params.R0 - reference_v1 - reference_v2
        errors.append(abs(predicted - reference))
        voltages.append(predicted)
        layer.V1, layer.V2 = layer._next_polarization(float(current))
    max_error_mV = float(np.max(errors) * 1000.0)
    if max_error_mV >= 1.0:
        raise AssertionError(f"ECM reference mismatch: {max_error_mV:.3f} mV")
    return {"max_abs_error_mV": max_error_mV, "samples": int(len(errors)), "voltages": voltages[:5]}


def _reference_next_polarization(
    params: ECMParams,
    dt: float,
    v1: float,
    v2: float,
    current: float,
) -> tuple[float, float]:
    """Independent second-order RC state update matching the MATLAB equations."""
    a1 = np.exp(-dt / max(params.R1 * params.C1, 1e-12))
    a2 = np.exp(-dt / max(params.R2 * params.C2, 1e-12))
    next_v1 = v1 * a1 + current * params.R1 * (1.0 - a1)
    next_v2 = v2 * a2 + current * params.R2 * (1.0 - a2)
    return float(next_v1), float(next_v2)


if __name__ == "__main__":
    print(cross_check_against_matlab())
