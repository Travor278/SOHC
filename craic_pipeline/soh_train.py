"""SOH 估计器训练：NASA 同源数据 + BatteryML trainer (v0.2)。

数据策略 v0.2：
    - 训练数据：NASA B0005-B0018（容量退化时序）+ ARC-FY08Q4（多温度多倍率补充）
    - 全程 NMC 18650 同源，无需跨化学体系 fine-tune
    - HUST 数据（仓库已携带）退为可选展示，不进训练管线

用途：
    - W1：在 NASA 数据上训 baseline SOH（Variance / 浅层 CNN）
    - W4：BatteryML 内挂 Mamba head 跑 SOH 对比表（架构创新点）

输入：
    --config : configs/nasa_soh_*.yaml（BatteryML 风格）
    --data   : NASA .mat 目录（默认 data/nasa_pcoe/）
    --out    : 权重保存路径（默认 outputs/soh_baseline.pt）

输出：
    .pt 权重 + JSON 训练日志（loss / RMSE 曲线）

依赖：
    external/BatteryML 已 clone（见 external/README.md）
    自写 NASA loader（craic_pipeline.nasa_loader），不复用 BatteryML 的 HUST loader
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from craic_pipeline.nasa_loader import load_arc_fy08q4, load_pcoe_basic


def _batteryml_classes():
    """Return BatteryML data classes, or local fallbacks if import fails."""
    root = Path(__file__).resolve().parents[1]
    batteryml_path = root / "external" / "BatteryML"
    if batteryml_path.exists() and str(batteryml_path) not in sys.path:
        sys.path.insert(0, str(batteryml_path))
    try:
        from batteryml.data.battery_data import BatteryData, CycleData

        return BatteryData, CycleData
    except Exception:
        return _FallbackBatteryData, _FallbackCycleData


class _FallbackCycleData:
    """Minimal CycleData-compatible container for NASA SOH fallback training."""

    def __init__(self, cycle_number: int, **kwargs):
        """Store one NASA cycle with BatteryML-like attributes."""
        self.cycle_number = cycle_number
        self.additional_data = {}
        for key, value in kwargs.items():
            if key in {
                "voltage_in_V",
                "current_in_A",
                "temperature_in_C",
                "time_in_s",
                "discharge_capacity_in_Ah",
            }:
                setattr(self, key, value)
            else:
                self.additional_data[key] = value


class _FallbackBatteryData:
    """Minimal BatteryData-compatible container for converted NASA cells."""

    def __init__(self, cell_id: str, *, cycle_data=None, **kwargs):
        """Store one NASA cell and its converted cycle list."""
        self.cell_id = cell_id
        self.cycle_data = cycle_data or []
        for key, value in kwargs.items():
            setattr(self, key, value)


class VarianceSohBaseline:
    """Small Ridge baseline over NASA cycle statistics for W1 SOH fallback."""

    def __init__(self, alpha: float = 1e-12):
        """Initialize a Ridge model over standardized NASA cycle features."""
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, cells):
        """Fit SOH from BatteryData cycle features and capacity-derived labels."""
        X, y = _cells_to_xy(cells)
        if len(y) == 0:
            raise ValueError("no finite SOH labels found in NASA cycles")
        self.model.fit(X, y)
        return self

    def predict(self, cells) -> np.ndarray:
        """Predict SOH fraction for all cycles with finite labels/features."""
        X, _ = _cells_to_xy(cells)
        if len(X) == 0:
            return np.array([], dtype=float)
        return np.clip(self.model.predict(X), 0.0, 1.2)

    def save(self, path: Path, metrics: dict) -> None:
        """Persist the sklearn baseline and W1 metrics to a `.pt` path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fout:
            pickle.dump({"model": self.model, "metrics": metrics}, fout)


def load_nasa_for_batteryml(data_dir: Path):
    """Load NASA .mat files and convert them to BatteryML BatteryData cells.

    Args:
        data_dir: `data/nasa_pcoe` or one of its B000x/ARC subdirectories.

    Returns:
        List of BatteryML `BatteryData` objects with capacity-derived SOH labels.
    """
    BatteryData, CycleData = _batteryml_classes()
    data_dir = Path(data_dir)
    files = _nasa_soh_files(data_dir)
    cells = []
    for file_path, loader in files:
        try:
            V, I, T, t, cycle_id, ambient, capacity = loader(file_path)
        except Exception:
            continue
        if V.size == 0:
            continue
        finite_caps = _valid_capacity_values(capacity)
        fresh_capacity = float(np.nanmax(finite_caps)) if finite_caps.size else np.nan
        cycles = []
        for cid in np.unique(cycle_id):
            idx = np.where(cycle_id == cid)[0]
            cap_values = capacity[idx]
            finite = _valid_capacity_values(cap_values)
            cap = float(finite[-1]) if finite.size else np.nan
            ratio = cap / fresh_capacity if np.isfinite(cap) and np.isfinite(fresh_capacity) else np.nan
            soh = float(np.clip(ratio, 0.0, 1.0)) if np.isfinite(ratio) else np.nan
            cycles.append(
                CycleData(
                    int(cid),
                    voltage_in_V=V[idx].astype(float).tolist(),
                    current_in_A=I[idx].astype(float).tolist(),
                    temperature_in_C=T[idx].astype(float).tolist(),
                    time_in_s=t[idx].astype(float).tolist(),
                    discharge_capacity_in_Ah=None if not np.isfinite(cap) else [cap],
                    capacity_in_Ah=cap,
                    soh=soh,
                    ambient_temperature=float(np.nanmedian(ambient[idx])) if idx.size else np.nan,
                )
            )
        cells.append(
            BatteryData(
                file_path.stem,
                cycle_data=cycles,
                form_factor="18650",
                cathode_material="NMC",
                nominal_capacity_in_Ah=fresh_capacity,
                reference="NASA PCoE",
                description="Converted by craic_pipeline.nasa_loader for CRAIC2026 W1 SOH",
            )
        )
    return cells


def _valid_capacity_values(values) -> np.ndarray:
    """Filter NASA capacity readings to plausible 18650 Ah values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    plausible = arr[(arr >= 0.5) & (arr <= 2.2)]
    if plausible.size:
        return plausible
    return arr[arr > 0]


def build_model(cfg):
    """Build the W1 SOH baseline model from config."""
    model_type = (cfg.get("model", {}) or {}).get("type", "variance").lower()
    if model_type != "variance":
        raise ValueError(f"W1 currently supports variance baseline only, got {model_type}")
    return VarianceSohBaseline()


def train(model, train_cells, val_cells, cfg):
    """Train the W1 SOH baseline and return NASA holdout RMSE metrics."""
    model.fit(train_cells)
    train_pred = model.predict(train_cells)
    val_pred = model.predict(val_cells)
    _, train_y = _cells_to_xy(train_cells)
    _, val_y = _cells_to_xy(val_cells)
    metrics = {
        "train_cells": [cell.cell_id for cell in train_cells],
        "val_cells": [cell.cell_id for cell in val_cells],
        "train_cycles": int(len(train_y)),
        "val_cycles": int(len(val_y)),
        "train_rmse_soh": _rmse(train_y, train_pred),
        "val_rmse_soh": _rmse(val_y, val_pred),
        "val_rmse_percent": _rmse(val_y, val_pred) * 100.0,
    }
    return metrics


def split_by_cell_id(cells, val_ratio: float = 0.2):
    """Split BatteryData cells by cell id for NASA holdout validation."""
    cells = sorted(cells, key=lambda cell: cell.cell_id)
    if len(cells) < 2:
        raise ValueError("need at least two NASA cells for cell-id holdout")
    n_val = max(1, int(round(len(cells) * val_ratio)))
    return cells[:-n_val], cells[-n_val:]


def _nasa_soh_files(data_dir: Path):
    """List NASA PCoE/ARC files and their matching loader functions."""
    candidates: list[tuple[Path, object]] = []
    search_roots = []
    if data_dir.name.lower() in {"b000x", "arc-fy08q4"}:
        search_roots = [data_dir]
    else:
        search_roots = [data_dir / "B000x", data_dir / "ARC-FY08Q4"]
    for root in search_roots:
        if not root.exists():
            continue
        loader = load_arc_fy08q4 if "arc" in root.name.lower() else load_pcoe_basic
        candidates.extend((path, loader) for path in sorted(root.rglob("*.mat")))
    return candidates


def _cells_to_xy(cells) -> tuple[np.ndarray, np.ndarray]:
    """Convert BatteryData cycles into SOH feature and label arrays."""
    X = []
    y = []
    for cell in cells:
        nominal_capacity = _cell_nominal_capacity(cell)
        for cycle in cell.cycle_data:
            feature = _cycle_features(cycle, nominal_capacity)
            target = _cycle_soh(cycle)
            if feature is not None and np.isfinite(target):
                X.append(feature)
                y.append(target)
    if not X:
        return np.empty((0, 10), dtype=float), np.array([], dtype=float)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _cycle_features(cycle, nominal_capacity: float) -> np.ndarray | None:
    """Compute fixed SOH baseline features from one NASA cycle."""
    voltage = _array(getattr(cycle, "voltage_in_V", None))
    current = _array(getattr(cycle, "current_in_A", None))
    temp = _array(getattr(cycle, "temperature_in_C", None))
    time = _array(getattr(cycle, "time_in_s", None))
    if voltage.size == 0 or current.size == 0:
        return None
    capacity = _cycle_capacity(cycle)
    capacity_ratio = capacity / nominal_capacity if np.isfinite(capacity) and nominal_capacity > 0 else np.nan
    capacity_ratio = float(np.clip(capacity_ratio, 0.0, 1.0)) if np.isfinite(capacity_ratio) else np.nan
    duration = float(np.nanmax(time) - np.nanmin(time)) if time.size else float(len(voltage))
    return np.array(
        [
            float(cycle.cycle_number),
            float(np.nanmean(voltage)),
            float(np.nanmin(voltage)),
            float(np.nanmax(voltage)),
            float(np.nanmean(current)),
            float(np.nanstd(current)),
            float(np.nanmean(temp)) if temp.size else np.nan,
            duration,
            capacity,
            capacity_ratio,
        ],
        dtype=float,
    )


def _cell_nominal_capacity(cell) -> float:
    """Read fresh capacity from a NASA BatteryData cell."""
    value = getattr(cell, "nominal_capacity_in_Ah", np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _cycle_capacity(cycle) -> float:
    """Read NASA cycle capacity in Ah from BatteryML or fallback storage."""
    additional = getattr(cycle, "additional_data", {})
    value = additional.get("capacity_in_Ah", getattr(cycle, "capacity_in_Ah", np.nan))
    try:
        return float(value)
    except (TypeError, ValueError):
        values = _array(value)
        finite = values[np.isfinite(values)]
        return float(finite[-1]) if finite.size else np.nan


def _cycle_soh(cycle) -> float:
    """Read capacity-derived SOH from a BatteryML or fallback cycle."""
    value = getattr(cycle, "additional_data", {}).get("soh", np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _array(values) -> np.ndarray:
    """Convert optional BatteryML sequence data to a flat float array."""
    if values is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _rmse(y_true, y_pred) -> float:
    """Compute RMSE for SOH labels and predictions."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    """Run the NASA SOH training CLI and persist model plus metrics."""
    parser = argparse.ArgumentParser(description="SOH training on NASA via BatteryML/fallback")
    parser.add_argument("--config", type=Path, default=Path("configs/nasa_soh_baseline.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/nasa_pcoe"))
    parser.add_argument("--out", type=Path, default=Path("outputs/soh_baseline.pt"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    cells = load_nasa_for_batteryml(args.data)
    if not cells:
        raise FileNotFoundError(f"no NASA SOH .mat files found under {args.data}")
    val_ratio = float((cfg.get("dataset", {}) or {}).get("val_ratio", 0.2))
    train_cells, val_cells = split_by_cell_id(cells, val_ratio=val_ratio)
    model = build_model(cfg)
    metrics = train(model, train_cells, val_cells, cfg)
    model.save(args.out, metrics)
    log_path = args.out.with_suffix(".metrics.json")
    log_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
