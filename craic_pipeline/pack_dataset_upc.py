"""Loader for the UPC 36-cell pack WLTP/CC-CV Parquet dataset.

Data source: CORA.RDR / UPC Dataverse, DOI `10.34810/DATA2395`.
The pack is represented as three parallel branches of twelve series cells
(`12S3P`) with cell voltage, branch current, surface temperature, and BMS SOC.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


N_PARALLEL = 3
N_SERIES = 12
N_CELLS = N_PARALLEL * N_SERIES


@dataclass
class UPCPackCycle:
    """One UPC pack cycle with arrays aligned on the original timestamp grid."""

    source_path: Path
    cycle_id: str
    cycle_type: str
    time_s: np.ndarray
    timestamps: np.ndarray
    semicycle: np.ndarray
    cell_voltage_V: np.ndarray
    branch_current_A: np.ndarray
    cell_temperature_C: np.ndarray
    pack_voltage_V: np.ndarray
    branch_voltage_V: np.ndarray
    bms_soc: np.ndarray
    ambient_temperature_C: np.ndarray
    humidity_percent: np.ndarray
    pack_resistance_ohm: np.ndarray

    @property
    def cell_voltage_flat_V(self) -> np.ndarray:
        """Return cell voltages as `(N, 36)` ordered P1S1..P3S12."""
        return self.cell_voltage_V.reshape(len(self.time_s), N_CELLS)

    @property
    def cell_temperature_mean_C(self) -> np.ndarray:
        """Return mean top/bottom cell temperatures as `(N, 36)`."""
        return np.nanmean(self.cell_temperature_C, axis=-1).reshape(len(self.time_s), N_CELLS)


def load_upc_pack_cycle(path: Path, *, downsample: int = 1) -> UPCPackCycle:
    """Load one UPC Parquet cycle into structured `12S3P` arrays.

    Args:
        path: `Qtzl_Cycle_*_partial_data.parquet` file from UPC DATA2395.
        downsample: Keep every nth row after loading; `1` preserves raw data.

    Returns:
        `UPCPackCycle` with voltage `(N, 3, 12)`, branch current `(N, 3)`,
        temperature `(N, 3, 12, 2)`, pack voltage, BMS SOC, and semicycle labels.
    """
    path = Path(path)
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    df = pd.read_parquet(path)
    if downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)
    _require_columns(df, _required_columns())

    timestamps = pd.to_datetime(df["Timestamp"]).to_numpy()
    time_s = (pd.to_datetime(df["Timestamp"]) - pd.to_datetime(df["Timestamp"]).iloc[0]).dt.total_seconds().to_numpy(
        dtype=np.float64
    )
    cell_voltage = np.stack([df[col].to_numpy(dtype=np.float32) for col in _cell_voltage_columns()], axis=1).reshape(
        len(df), N_PARALLEL, N_SERIES
    )
    branch_current = np.stack([df[f"Current_Actual_P{p} [A]"].to_numpy(dtype=np.float32) for p in range(1, 4)], axis=1)
    branch_voltage = np.stack([df[f"Voltage_Actual_P{p} [V]"].to_numpy(dtype=np.float32) for p in range(1, 4)], axis=1)
    cell_temp = np.stack([df[col].to_numpy(dtype=np.float32) for col in _cell_temperature_columns()], axis=1).reshape(
        len(df), N_PARALLEL, N_SERIES, 2
    )
    cycle_id = _first_string(df["Cycle"], fallback=_cycle_id_from_name(path))
    return UPCPackCycle(
        source_path=path,
        cycle_id=cycle_id,
        cycle_type=_cycle_type_from_name(path),
        time_s=time_s,
        timestamps=timestamps,
        semicycle=df["Semicycle"].astype(str).to_numpy(),
        cell_voltage_V=cell_voltage,
        branch_current_A=branch_current,
        cell_temperature_C=cell_temp,
        pack_voltage_V=df["Voltage_Actual_Battery [V]"].to_numpy(dtype=np.float32),
        branch_voltage_V=branch_voltage,
        bms_soc=df["SoC_Actual_Battery [percent]"].to_numpy(dtype=np.float32) / 100.0,
        ambient_temperature_C=df[["Temperature_IN_Chamber [degC]", "Temperature_OUT_Chamber [degC]"]].to_numpy(
            dtype=np.float32
        ),
        humidity_percent=df["Humidity_IN_Chamber [RH_percent]"].to_numpy(dtype=np.float32),
        pack_resistance_ohm=df["Resistance_Actual_Battery_IT5101 [Ohm]"].to_numpy(dtype=np.float64),
    )


def iter_upc_pack_cycles(
    root: Path,
    *,
    pattern: str = "*.parquet",
    limit: int | None = None,
    downsample: int = 1,
) -> Iterator[UPCPackCycle]:
    """Yield UPC pack cycles from a directory without concatenating all data."""
    files = sorted(Path(root).glob(pattern))
    if limit is not None:
        files = files[:limit]
    for file_path in files:
        yield load_upc_pack_cycle(file_path, downsample=downsample)


def summarize_upc_pack_cycle(cycle: UPCPackCycle) -> dict:
    """Compute pack-level spread, current, temperature, and safety metrics."""
    voltage_flat = cycle.cell_voltage_flat_V
    temp_flat = cycle.cell_temperature_mean_C
    valid_temp = temp_flat[(temp_flat > -40.0) & (temp_flat < 120.0)]
    spread_mV = (np.nanmax(voltage_flat, axis=1) - np.nanmin(voltage_flat, axis=1)) * 1000.0
    semicycles = sorted(set(str(item) for item in cycle.semicycle))
    balancing_mask = np.char.find(cycle.semicycle.astype(str), "Balancing") >= 0
    return {
        "file": cycle.source_path.name,
        "cycle_id": cycle.cycle_id,
        "cycle_type": cycle.cycle_type,
        "samples": int(len(cycle.time_s)),
        "duration_s": float(cycle.time_s[-1] - cycle.time_s[0]) if len(cycle.time_s) else 0.0,
        "semicycles": "|".join(semicycles),
        "bms_soc_start": _finite_edge(cycle.bms_soc, first=True),
        "bms_soc_end": _finite_edge(cycle.bms_soc, first=False),
        "pack_voltage_min_V": float(np.nanmin(cycle.pack_voltage_V)),
        "pack_voltage_max_V": float(np.nanmax(cycle.pack_voltage_V)),
        "cell_voltage_min_V": float(np.nanmin(voltage_flat)),
        "cell_voltage_max_V": float(np.nanmax(voltage_flat)),
        "cell_voltage_spread_mean_mV": float(np.nanmean(spread_mV)),
        "cell_voltage_spread_max_mV": float(np.nanmax(spread_mV)),
        "branch_current_abs_max_A": float(np.nanmax(np.abs(cycle.branch_current_A))),
        "temperature_max_raw_C": float(np.nanmax(temp_flat)),
        "temperature_median_valid_C": float(np.nanmedian(valid_temp)) if valid_temp.size else float("nan"),
        "temperature_p95_valid_C": float(np.nanpercentile(valid_temp, 95)) if valid_temp.size else float("nan"),
        "temperature_valid_fraction": float(valid_temp.size / max(temp_flat.size, 1)),
        "balancing_samples": int(np.sum(balancing_mask)),
    }


def summarize_upc_pack_directory(
    root: Path,
    *,
    pattern: str = "*.parquet",
    limit: int | None = None,
    downsample: int = 1,
) -> pd.DataFrame:
    """Summarize multiple UPC Parquet cycles into one DataFrame."""
    rows = [
        summarize_upc_pack_cycle(cycle)
        for cycle in iter_upc_pack_cycles(root, pattern=pattern, limit=limit, downsample=downsample)
    ]
    return pd.DataFrame(rows)


def export_simulink_csv(cycle: UPCPackCycle, out_path: Path) -> Path:
    """Export one cycle as MATLAB/Simulink-friendly wide CSV time series."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {
        "time_s": cycle.time_s,
        "pack_voltage_V": cycle.pack_voltage_V,
        "bms_soc": cycle.bms_soc,
        "ambient_in_C": cycle.ambient_temperature_C[:, 0],
        "ambient_out_C": cycle.ambient_temperature_C[:, 1],
        "humidity_percent": cycle.humidity_percent,
    }
    for p in range(N_PARALLEL):
        data[f"branch_current_P{p + 1}_A"] = cycle.branch_current_A[:, p]
        data[f"branch_voltage_P{p + 1}_V"] = cycle.branch_voltage_V[:, p]
    voltage_flat = cycle.cell_voltage_flat_V
    temp_flat = cycle.cell_temperature_mean_C
    for idx, label in enumerate(cell_labels()):
        data[f"cell_voltage_{label}_V"] = voltage_flat[:, idx]
        data[f"cell_temperature_{label}_C"] = temp_flat[:, idx]
    pd.DataFrame(data).to_csv(out_path, index=False)
    return out_path


def cell_labels() -> list[str]:
    """Return stable cell labels ordered P1S1..P3S12."""
    return [f"P{p}S{s}" for p in range(1, N_PARALLEL + 1) for s in range(1, N_SERIES + 1)]


def _required_columns() -> list[str]:
    """List columns required by the structured UPC loader."""
    return [
        "Cycle",
        "Semicycle",
        "Timestamp",
        "Current_Actual_Battery [A]",
        "Voltage_Actual_Battery [V]",
        "SoC_Actual_Battery [percent]",
        "Temperature_IN_Chamber [degC]",
        "Temperature_OUT_Chamber [degC]",
        "Humidity_IN_Chamber [RH_percent]",
        "Resistance_Actual_Battery_IT5101 [Ohm]",
        *_cell_voltage_columns(),
        *[f"Current_Actual_P{p} [A]" for p in range(1, 4)],
        *[f"Voltage_Actual_P{p} [V]" for p in range(1, 4)],
        *_cell_temperature_columns(),
    ]


def _cell_voltage_columns() -> list[str]:
    """Return UPC cell-voltage column names ordered P1S1..P3S12."""
    return [f"Voltage_Cell_P{p}S{s} [V]" for p in range(1, 4) for s in range(1, 13)]


def _cell_temperature_columns() -> list[str]:
    """Return UPC cell-temperature columns ordered P1S1 top/bottom..P3S12."""
    cols = []
    for p in range(1, 4):
        for s in range(1, 13):
            cols.append(f"Temperature_Cell_Top_P{p}S{s} [degC]")
            cols.append(f"Temperature_Cell_Bottom_P{p}S{s} [degC]")
    return cols


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a helpful error when a UPC file is missing required columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"UPC pack file missing columns: {missing[:8]}")


def _cycle_id_from_name(path: Path) -> str:
    """Extract a cycle id from the UPC filename."""
    match = re.search(r"Cycle_(\d+)", path.name)
    return f"Cycle_{int(match.group(1))}" if match else path.stem


def _cycle_type_from_name(path: Path) -> str:
    """Extract `WLTP` or `Capacity_check` from the UPC filename."""
    stem = path.stem
    if "WLTP" in stem:
        return "WLTP"
    if "Capacity_check" in stem:
        return "Capacity_check"
    return "unknown"


def _first_string(series: pd.Series, *, fallback: str) -> str:
    """Return the first non-empty string from a Series."""
    for value in series:
        if pd.notna(value) and str(value):
            return str(value)
    return fallback


def _finite_edge(values: np.ndarray, *, first: bool) -> float:
    """Return the first or last finite value from an array."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[0] if first else finite[-1])


def main() -> None:
    """Summarize UPC pack files and optionally export one cycle for Simulink."""
    parser = argparse.ArgumentParser(description="Load and summarize UPC 36-cell pack Parquet files")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pack_wltp_upc"))
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--summary-out", type=Path, default=Path("outputs/upc_pack_summary.csv"))
    parser.add_argument("--simulink-csv-out", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize_upc_pack_directory(
        args.data_dir,
        pattern=args.pattern,
        limit=args.limit,
        downsample=args.downsample,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    print(summary.to_string(index=False))
    print(f"summary={args.summary_out}")
    if args.simulink_csv_out is not None:
        first = next(iter_upc_pack_cycles(args.data_dir, pattern=args.pattern, limit=1, downsample=args.downsample))
        path = export_simulink_csv(first, args.simulink_csv_out)
        print(f"simulink_csv={path}")


if __name__ == "__main__":
    main()
