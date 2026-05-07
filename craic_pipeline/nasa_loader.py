"""NASA PCoE .mat loaders for the CRAIC2026 v0.2 data strategy.

The public loaders normalize the three NASA subsets used by W1/W2 into the
same tuple of one-dimensional arrays:
    (V, I, T, t, cycle_id, ambient_temp, capacity)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat


LoaderOutput = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


_META_KEYS = {"__header__", "__version__", "__globals__"}
_VOLTAGE_KEYS = (
    "Voltage_measured",
    "voltage_measured",
    "voltage",
    "Voltage",
    "Voltage_load",
    "voltage_in_V",
    "V",
)
_CURRENT_KEYS = (
    "Current_measured",
    "current_measured",
    "current",
    "Current",
    "Current_load",
    "current_in_A",
    "I",
)
_TEMP_KEYS = (
    "Temperature_measured",
    "temperature_measured",
    "temperature",
    "Temperature",
    "temperature_in_C",
    "T",
)
_TIME_KEYS = ("Time", "time", "t", "time_in_s", "Test_Time")
_CAPACITY_KEYS = (
    "Capacity",
    "capacity",
    "discharge_capacity",
    "discharge_capacity_in_Ah",
    "charge_capacity_in_Ah",
)
_AMBIENT_KEYS = ("ambient_temperature", "Ambient_temperature", "ambient_temp", "ambient_T")


def load_pcoe_basic(path: Path | str) -> LoaderOutput:
    """Load NASA B0005-B0018 .mat files into unified V/I/T/time arrays.

    Args:
        path: A B0005-B0018 `.mat` file or a directory containing such files.

    Returns:
        Tuple `(V, I, T, t, cycle_id, ambient_temp, capacity)` from NASA PCoE.
    """
    return _load_many(Path(path), subset="pcoe")


def load_arc_fy08q4(path: Path | str) -> LoaderOutput:
    """Load NASA ARC-FY08Q4 .mat files with ambient temperature metadata.

    Args:
        path: An ARC `.mat` file or directory containing B0025-B0056 files.

    Returns:
        Tuple `(V, I, T, t, cycle_id, ambient_temp, capacity)` from NASA ARC.
    """
    return _load_many(Path(path), subset="arc")


def load_randomized_usage(path: Path | str) -> LoaderOutput:
    """Load NASA Randomized Battery Usage data and drop current-jump samples.

    Args:
        path: An RW `.mat` file or directory containing randomized usage files.

    Returns:
        Unified arrays after removing samples where adjacent current changes by
        more than 1 A inside each parsed cycle.
    """
    arrays = _load_many(Path(path), subset="randomized")
    return _filter_current_jumps(arrays, threshold_A=1.0)


def _load_many(path: Path, *, subset: str) -> LoaderOutput:
    """Load every NASA `.mat` file under a path and concatenate arrays."""
    files = _mat_files(path)
    records: list[LoaderOutput] = []
    cycle_offset = 0
    for file_path in files:
        loaded = _load_one_file(file_path, subset=subset, cycle_offset=cycle_offset)
        if loaded[0].size:
            records.append(loaded)
            cycle_offset = int(np.nanmax(loaded[4])) + 1
    return _concat_outputs(records)


def _mat_files(path: Path) -> list[Path]:
    """Resolve a NASA file or directory into sorted `.mat` paths."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.mat"))
    raise FileNotFoundError(f"NASA .mat path does not exist: {path}")


def _load_one_file(path: Path, *, subset: str, cycle_offset: int) -> LoaderOutput:
    """Parse one NASA `.mat` file into normalized cycle arrays."""
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    payload = _select_payload(raw, path)
    cycles = _extract_cycles(payload)
    if not cycles:
        cycles = [payload]

    outputs: list[LoaderOutput] = []
    file_ambient = _ambient_from_name(path) if subset == "arc" else np.nan
    for local_id, cycle in enumerate(cycles):
        parsed = _parse_cycle(cycle, cycle_offset + local_id, file_ambient)
        if parsed[0].size:
            outputs.append(parsed)
    return _concat_outputs(outputs)


def _select_payload(raw: dict, path: Path):
    """Select the non-metadata MATLAB payload matching the file stem."""
    candidates = {k: v for k, v in raw.items() if k not in _META_KEYS}
    stem = path.stem
    if stem in candidates:
        return _to_plain(candidates[stem])
    if len(candidates) == 1:
        return _to_plain(next(iter(candidates.values())))
    for key, value in candidates.items():
        if key.lower() == stem.lower():
            return _to_plain(value)
    return _to_plain(candidates)


def _extract_cycles(obj) -> list:
    """Extract cycle-like MATLAB structs from a NASA payload object."""
    plain = _to_plain(obj)
    if isinstance(plain, dict):
        for key in ("cycle", "cycles", "Cycle", "data"):
            if key in plain:
                value = _to_plain(plain[key])
                if isinstance(value, list):
                    return value
                return [value]
        found: list = []
        for value in plain.values():
            found.extend(_extract_cycles(value))
        return found
    if isinstance(plain, list):
        if any(_has_measurements(item) for item in plain):
            return plain
        found = []
        for item in plain:
            found.extend(_extract_cycles(item))
        return found
    return []


def _parse_cycle(cycle, cycle_id: int, default_ambient: float) -> LoaderOutput:
    """Convert one NASA cycle struct into V/I/T/time/capacity arrays."""
    cycle = _to_plain(cycle)
    data = _to_plain(_field(cycle, "data", "Data", default=cycle))

    voltage = _series(data, _VOLTAGE_KEYS)
    current = _series(data, _CURRENT_KEYS)
    if voltage.size == 0 or current.size == 0:
        return _empty_output()

    n = min(voltage.size, current.size)
    temp = _series(data, _TEMP_KEYS)
    time = _series(data, _TIME_KEYS)
    if temp.size == 0:
        ambient = _scalar(_field(cycle, *_AMBIENT_KEYS, default=default_ambient), default_ambient)
        temp = np.full(n, ambient, dtype=float)
    if time.size == 0:
        time = np.arange(n, dtype=float)

    n = min(n, temp.size, time.size)
    voltage = voltage[:n]
    current = current[:n]
    temp = temp[:n]
    time = time[:n]

    ambient = _scalar(
        _field(cycle, *_AMBIENT_KEYS, default=_field(data, *_AMBIENT_KEYS, default=default_ambient)),
        default_ambient,
    )
    capacity = _capacity_array(cycle, data, n)
    return (
        voltage,
        current,
        temp,
        time,
        np.full(n, float(cycle_id), dtype=float),
        np.full(n, ambient, dtype=float),
        capacity,
    )


def _capacity_array(cycle, data, n: int) -> np.ndarray:
    """Broadcast a NASA capacity value or series to sample length `n`."""
    capacity_value = _field(data, *_CAPACITY_KEYS, default=_field(cycle, *_CAPACITY_KEYS, default=np.nan))
    capacity_series = _as_numeric_1d(capacity_value)
    if capacity_series.size:
        finite = capacity_series[np.isfinite(capacity_series)]
        value = float(finite[-1]) if finite.size else np.nan
    else:
        value = _scalar(capacity_value, np.nan)
    return np.full(n, value, dtype=float)


def _filter_current_jumps(arrays: LoaderOutput, *, threshold_A: float) -> LoaderOutput:
    """Split randomized cycles at large current jumps and keep stable spans."""
    V, I, T, t, cycle_id, ambient, capacity = arrays
    if I.size <= 1:
        return arrays
    kept_indices: list[np.ndarray] = []
    new_cycle_ids: list[np.ndarray] = []
    next_cycle = 0
    for cid in np.unique(cycle_id):
        idx = np.where(cycle_id == cid)[0]
        if idx.size <= 1:
            continue
        jumps = np.where(np.abs(np.diff(I[idx])) > threshold_A)[0] + 1
        bounds = np.concatenate(([0], jumps, [idx.size]))
        for start, stop in zip(bounds[:-1], bounds[1:]):
            segment = idx[start:stop]
            if segment.size < 2:
                continue
            kept_indices.append(segment)
            new_cycle_ids.append(np.full(segment.size, float(next_cycle), dtype=float))
            next_cycle += 1
    if not kept_indices:
        return _empty_output()
    keep = np.concatenate(kept_indices)
    filtered = [arr[keep] for arr in (V, I, T, t, cycle_id, ambient, capacity)]
    filtered[4] = np.concatenate(new_cycle_ids)
    return tuple(filtered)  # type: ignore[return-value]


def _concat_outputs(outputs: Sequence[LoaderOutput]) -> LoaderOutput:
    """Concatenate normalized NASA loader outputs along sample dimension."""
    if not outputs:
        return _empty_output()
    return tuple(np.concatenate([out[i] for out in outputs]) for i in range(7))  # type: ignore[return-value]


def _empty_output() -> LoaderOutput:
    """Return an empty normalized NASA loader tuple."""
    empty = np.array([], dtype=float)
    return empty, empty, empty, empty, empty, empty, empty


def _to_plain(obj):
    """Convert scipy MATLAB structs and object arrays into Python containers."""
    if hasattr(obj, "_fieldnames"):
        return {name: _to_plain(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            return {name: _to_plain(obj[name]) for name in obj.dtype.names}
        if obj.dtype == object:
            return [_to_plain(item) for item in obj.ravel()]
        if obj.ndim == 0:
            return _to_plain(obj.item())
        return obj
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: _to_plain(obj[name]) for name in obj.dtype.names}
    return obj


def _field(obj, *names: str, default=None):
    """Read a case-insensitive field from a NASA MATLAB-derived object."""
    obj = _to_plain(obj)
    if isinstance(obj, dict):
        lowered = {key.lower(): key for key in obj}
        for name in names:
            if name in obj:
                return obj[name]
            match = lowered.get(name.lower())
            if match is not None:
                return obj[match]
    return default


def _series(obj, names: Iterable[str]) -> np.ndarray:
    """Return the first numeric one-dimensional series matching field names."""
    for name in names:
        values = _as_numeric_1d(_field(obj, name, default=None))
        if values.size:
            return values
    return np.array([], dtype=float)


def _as_numeric_1d(value) -> np.ndarray:
    """Convert MATLAB scalar/vector/list data into a flat float array."""
    if value is None:
        return np.array([], dtype=float)
    value = _to_plain(value)
    if isinstance(value, list):
        value = np.array(value, dtype=object)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    return arr


def _scalar(value, default: float) -> float:
    """Return the first finite scalar from MATLAB data, or a default."""
    arr = _as_numeric_1d(value)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        return float(finite[0])
    return float(default) if np.isfinite(default) else np.nan


def _has_measurements(obj) -> bool:
    """Check whether a MATLAB object contains voltage and current samples."""
    obj = _to_plain(obj)
    data = _field(obj, "data", "Data", default=obj)
    return _series(data, _VOLTAGE_KEYS).size > 0 and _series(data, _CURRENT_KEYS).size > 0


def _ambient_from_name(path: Path) -> float:
    """Infer ARC ambient temperature from NASA filename tokens when present."""
    text = path.stem.lower()
    for token, value in (("4c", 4.0), ("24c", 24.0), ("43c", 43.0)):
        if token in text or token.replace("c", "degc") in text:
            return value
    return np.nan
