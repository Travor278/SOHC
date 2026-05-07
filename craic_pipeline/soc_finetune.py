"""Fine-tune KeiLongW SOC LSTM on NASA ARC-FY08Q4 data.

The script warm-starts from KeiLongW `.h5` artifacts, freezes the first two
LSTM layers, trains the final recurrent/dense head on ARC V/I/T sequences, and
writes `outputs/soc_finetuned.h5`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from craic_pipeline.nasa_loader import load_arc_fy08q4, load_pcoe_basic
from craic_pipeline.soc_inference import infer_model_window, load_keilongw_model, preprocess_sequence


def build_soc_training_set(
    data_dir: Path,
    window: int,
    *,
    stride: int = 10,
    max_samples: int = 200_000,
    limit_files: int | None = None,
    label_mode: str = "strict",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ARC train/holdout LSTM windows with Coulomb-counted SOC labels."""
    files = sorted(Path(data_dir).rglob("*.mat"))
    if limit_files is not None:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"no ARC .mat files found under {data_dir}")

    train_X: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    val_X: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    holdout_start = max(1, int(len(files) * 0.8))

    for idx, file_path in enumerate(files):
        df = _frame_from_loader(file_path, load_arc_fy08q4)
        if len(df) < window:
            continue
        labels = _estimate_soc_labels(df, mode=label_mode)
        X, y = _labeled_cycle_windows(df, labels, window=window, stride=stride)
        if len(y) == 0:
            continue
        if idx >= holdout_start:
            val_X.append(X)
            val_y.append(y)
        else:
            train_X.append(X)
            train_y.append(y)

    if not train_X or not val_X:
        raise ValueError("not enough ARC data to create train/holdout splits")
    X_train, y_train = _cap_samples(np.concatenate(train_X), np.concatenate(train_y), max_samples)
    X_val, y_val = _cap_samples(np.concatenate(val_X), np.concatenate(val_y), max(1, max_samples // 5))
    return X_train, y_train, X_val, y_val


def build_pcoe_cell_split(
    pcoe_dir: Path,
    window: int,
    *,
    train_cells: Sequence[str] = ("B0005", "B0006", "B0007"),
    holdout_cells: Sequence[str] = ("B0018",),
    stride: int = 10,
    max_samples: int = 200_000,
    label_mode: str = "strict",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build NASA B000x train/holdout windows split by cell id."""
    files = sorted(Path(pcoe_dir).rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no PCoE .mat files found under {pcoe_dir}")

    train_set = {cell.upper() for cell in train_cells}
    holdout_set = {cell.upper() for cell in holdout_cells}
    train_X: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    val_X: list[np.ndarray] = []
    val_y: list[np.ndarray] = []
    for file_path in files:
        cell_id = file_path.stem.upper()
        if cell_id not in train_set and cell_id not in holdout_set:
            continue
        df = _frame_from_loader(file_path, load_pcoe_basic)
        if len(df) < window:
            continue
        labels = _estimate_soc_labels(df, mode=label_mode)
        X, y = _labeled_cycle_windows(df, labels, window=window, stride=stride)
        if len(y) == 0:
            continue
        if cell_id in holdout_set:
            val_X.append(X)
            val_y.append(y)
        else:
            train_X.append(X)
            train_y.append(y)

    if not train_X or not val_X:
        raise ValueError(
            "not enough PCoE data for cell split; "
            f"train_cells={list(train_cells)}, holdout_cells={list(holdout_cells)}"
        )
    X_train, y_train = _cap_samples(np.concatenate(train_X), np.concatenate(train_y), max_samples)
    X_val, y_val = _cap_samples(np.concatenate(val_X), np.concatenate(val_y), max(1, max_samples // 5))
    return X_train, y_train, X_val, y_val


def finetune_soc_model(
    weights: Path,
    arc_dir: Path,
    out: Path,
    *,
    pcoe_dir: Path | None = None,
    split_mode: str = "arc",
    train_cells: Sequence[str] = ("B0005", "B0006", "B0007"),
    holdout_cells: Sequence[str] = ("B0018",),
    window: int | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    stride: int = 10,
    max_samples: int = 200_000,
    limit_files: int | None = None,
    learning_rate: float = 1e-3,
    early_stop_patience: int = 5,
    label_mode: str = "strict",
    two_stage: bool = False,
    stage1_epochs: int | None = None,
    stage2_epochs: int | None = None,
    stage3_epochs: int = 0,
    stage1_learning_rate: float = 3e-4,
    stage2_learning_rate: float | None = None,
    stage3_learning_rate: float = 1e-5,
) -> dict:
    """Fine-tune the KeiLongW SOC model and save a Keras `.h5` artifact."""
    from tensorflow import keras

    model = load_keilongw_model(weights)
    fixed_window = infer_model_window(model)
    window = window or fixed_window or 100
    if fixed_window is not None and window != fixed_window:
        raise ValueError(f"weights require window={fixed_window}, got window={window}")
    if split_mode == "arc":
        X_train, y_train, X_val, y_val = build_soc_training_set(
            arc_dir,
            window,
            stride=stride,
            max_samples=max_samples,
            limit_files=limit_files,
            label_mode=label_mode,
        )
    elif split_mode == "pcoe-cell":
        if pcoe_dir is None:
            raise ValueError("pcoe_dir is required when split_mode='pcoe-cell'")
        X_train, y_train, X_val, y_val = build_pcoe_cell_split(
            pcoe_dir,
            window,
            train_cells=train_cells,
            holdout_cells=holdout_cells,
            stride=stride,
            max_samples=max_samples,
            label_mode=label_mode,
        )
    else:
        raise ValueError(f"unknown split_mode={split_mode!r}; expected 'arc' or 'pcoe-cell'")

    out.parent.mkdir(parents=True, exist_ok=True)

    histories: list[dict] = []
    best_mae = _evaluate_model_mae(model, X_val, y_val)
    model.save(out)
    stages = (
        [
            ("head", stage1_epochs if stage1_epochs is not None else max(1, epochs // 3), stage1_learning_rate),
            ("last_lstm", stage2_epochs if stage2_epochs is not None else epochs, stage2_learning_rate or learning_rate),
        ]
        if two_stage
        else [("last_lstm", epochs, learning_rate)]
    )
    if stage3_epochs > 0:
        stages.append(("all", stage3_epochs, stage3_learning_rate))

    for stage_name, stage_epochs, stage_lr in stages:
        if stage_epochs <= 0:
            continue
        _set_trainable_stage(model, stage_name)
        model.compile(optimizer=keras.optimizers.Adam(stage_lr), loss="mse", metrics=["mae"])
        model, best_mae, stage_result = _fit_stage_with_global_best(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            out=out,
            stage_name=stage_name,
            epochs=stage_epochs,
            batch_size=batch_size,
            learning_rate=stage_lr,
            early_stop_patience=early_stop_patience,
        )
        histories.append(stage_result)

    if out.exists():
        model = load_keilongw_model(out)
    mae = _evaluate_model_mae(model, X_val, y_val)
    if not out.exists():
        model.save(out)
    return {
        "train_samples": int(len(X_train)),
        "holdout_samples": int(len(X_val)),
        "limit_files": limit_files,
        "split_mode": split_mode,
        "train_cells": list(train_cells) if split_mode == "pcoe-cell" else None,
        "holdout_cells": list(holdout_cells) if split_mode == "pcoe-cell" else None,
        "label_mode": label_mode,
        "two_stage": two_stage,
        "learning_rate": learning_rate,
        "initial_holdout_mae_fraction": float(histories[0]["initial_mae_fraction"]) if histories else best_mae,
        "holdout_mae_fraction": mae,
        "holdout_mae_percent": mae * 100.0,
        "stages": histories,
        "history": _merge_histories(histories),
    }


def evaluate_on_pcoe(
    weights: Path,
    pcoe_dir: Path,
    *,
    window: int | None = None,
    stride: int = 10,
    max_samples: int = 50_000,
    cells: Sequence[str] | None = None,
    label_mode: str = "strict",
) -> dict:
    """Evaluate a fine-tuned SOC model on NASA B0005-B0018 holdout files."""
    model = load_keilongw_model(weights)
    window = window or infer_model_window(model) or 100
    files = sorted(Path(pcoe_dir).rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no PCoE .mat files found under {pcoe_dir}")
    cell_filter = {cell.upper() for cell in cells} if cells else None
    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for file_path in files:
        if cell_filter is not None and file_path.stem.upper() not in cell_filter:
            continue
        df = _frame_from_loader(file_path, load_pcoe_basic)
        if len(df) < window:
            continue
        labels = _estimate_soc_labels(df, mode=label_mode)
        X, y = _labeled_cycle_windows(df, labels, window=window, stride=stride)
        if len(y):
            X_all.append(X)
            y_all.append(y)
    if not X_all:
        raise ValueError("not enough PCoE data to evaluate SOC")
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    X, y = _cap_samples(X, y, max_samples)
    pred = np.asarray(model.predict(X, verbose=0)).reshape(len(X), -1)[:, 0]
    mae = float(np.mean(np.abs(np.clip(pred, 0.0, 1.0) - y)))
    return {
        "samples": int(len(y)),
        "cells": sorted(cell_filter) if cell_filter else None,
        "label_mode": label_mode,
        "mae_fraction": mae,
        "mae_percent": mae * 100.0,
    }


def _frame_from_loader(file_path: Path, loader) -> pd.DataFrame:
    """Convert one NASA loader output into a SOC training DataFrame."""
    V, I, T, t, cycle_id, ambient, capacity = loader(file_path)
    return pd.DataFrame(
        {
            "t": t,
            "voltage": V,
            "current": I,
            "temperature": T,
            "cycle_id": cycle_id,
            "ambient_T": ambient,
            "capacity": capacity,
        }
    )


def _estimate_soc_labels(df: pd.DataFrame, *, mode: str = "strict") -> np.ndarray:
    """Create SOC labels by per-discharge-cycle Coulomb counting on NASA data."""
    if mode not in {"strict", "legacy"}:
        raise ValueError(f"unknown SOC label mode={mode!r}; expected 'strict' or 'legacy'")
    labels = np.full(len(df), np.nan, dtype=float)
    for _, group in df.groupby("cycle_id", sort=False):
        idx = group.index.to_numpy()
        current = np.nan_to_num(group["current"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        time = np.nan_to_num(group["t"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        voltage = np.nan_to_num(group["voltage"].to_numpy(dtype=float), nan=np.nan)
        capacity = group["capacity"].replace([np.inf, -np.inf], np.nan).dropna()
        if capacity.empty or capacity.iloc[0] <= 0:
            continue
        cap = float(capacity.iloc[0])
        discharge_amps = _discharge_current_magnitude(current, voltage)
        if discharge_amps is None:
            continue
        dt = np.diff(time, prepend=time[0])
        dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
        ah = np.cumsum(discharge_amps * dt) / 3600.0
        terminal_ah = float(ah[-1]) if np.isfinite(ah[-1]) else 0.0
        if terminal_ah <= 1e-6:
            continue
        if mode == "legacy":
            span = max(terminal_ah, cap, 1e-6)
            labels[idx] = np.clip(1.0 - ah / span, 0.0, 1.0)
            continue

        terminal_soc = _terminal_soc_from_capacity_and_cutoff(terminal_ah, cap, voltage)
        consumed_fraction = np.clip(ah / terminal_ah, 0.0, 1.0)
        soc = 1.0 - consumed_fraction * (1.0 - terminal_soc)
        if soc.size:
            soc[0] = 1.0
            soc[-1] = terminal_soc
        labels[idx] = np.clip(soc, 0.0, 1.0)
    return labels


def _discharge_current_magnitude(current: np.ndarray, voltage: np.ndarray) -> np.ndarray | None:
    """Infer NASA discharge polarity and return positive discharge current."""
    if current.size == 0:
        return None
    finite_current = current[np.isfinite(current)]
    if finite_current.size == 0:
        return None
    median_current = float(np.nanmedian(finite_current))
    voltage_drop = _finite_endpoint_delta(voltage)
    if median_current < -0.05:
        amps = np.clip(-current, 0.0, None)
    elif median_current > 0.05 and voltage_drop < -0.05:
        amps = np.clip(current, 0.0, None)
    else:
        return None
    amps = np.where(np.isfinite(amps), amps, 0.0)
    return amps if np.count_nonzero(amps > 0.05) >= 2 else None


def _finite_endpoint_delta(values: np.ndarray) -> float:
    """Return last-minus-first delta over finite values, or zero."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    return float(finite[-1] - finite[0])


def _terminal_soc_from_capacity_and_cutoff(terminal_ah: float, capacity: float, voltage: np.ndarray) -> float:
    """Estimate end-of-cycle SOC from capacity consistency and cutoff voltage."""
    capacity_ratio = terminal_ah / max(capacity, 1e-6)
    finite_voltage = voltage[np.isfinite(voltage)]
    end_voltage = float(finite_voltage[-1]) if finite_voltage.size else np.nan
    reached_cutoff = np.isfinite(end_voltage) and end_voltage <= 3.35
    if reached_cutoff or capacity_ratio >= 0.85:
        return 0.0
    return float(np.clip(1.0 - capacity_ratio, 0.0, 0.95))


def _labeled_cycle_windows(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build SOC windows within labeled NASA cycles without crossing cycles."""
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for _, group in df.groupby("cycle_id", sort=False):
        group_idx = group.index.to_numpy(dtype=int)
        if not np.isfinite(labels[group_idx]).any() or len(group) < window:
            continue
        try:
            X, rows = preprocess_sequence(group, window, stride=stride)
        except ValueError:
            continue
        y = labels[rows.index.to_numpy(dtype=int)]
        finite = np.isfinite(y)
        if np.any(finite):
            X_parts.append(X[finite])
            y_parts.append(y[finite])
    if not X_parts:
        return np.empty((0, window, 3), dtype=np.float32), np.array([], dtype=float)
    return np.concatenate(X_parts), np.concatenate(y_parts)


def _freeze_first_lstm_layers(model, *, count: int) -> None:
    """Freeze the first `count` LSTM layers in a KeiLongW-compatible model."""
    frozen = 0
    for layer in model.layers:
        if "lstm" in layer.__class__.__name__.lower() or "lstm" in layer.name.lower():
            if frozen < count:
                layer.trainable = False
                frozen += 1


def _set_trainable_stage(model, stage_name: str) -> None:
    """Set Keras trainable flags for staged SOC fine-tuning."""
    if stage_name == "head":
        for layer in model.layers:
            is_lstm = "lstm" in layer.__class__.__name__.lower() or "lstm" in layer.name.lower()
            layer.trainable = not is_lstm
        return
    if stage_name == "last_lstm":
        for layer in model.layers:
            layer.trainable = True
        _freeze_first_lstm_layers(model, count=2)
        return
    if stage_name == "all":
        for layer in model.layers:
            layer.trainable = True
        return
    raise ValueError(f"unknown fine-tune stage={stage_name!r}")


def _fit_stage_with_global_best(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    out: Path,
    stage_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stop_patience: int,
) -> tuple[object, float, dict]:
    """Fit one SOC stage and keep the best validation model seen so far."""
    from tensorflow import keras

    initial_mae = _evaluate_model_mae(model, X_val, y_val)
    temp_out = out.with_name(f"{out.stem}.{stage_name}.candidate{out.suffix}")
    if temp_out.exists():
        temp_out.unlink()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(temp_out),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
    ]
    if early_stop_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_mae",
                mode="min",
                patience=early_stop_patience,
                restore_best_weights=True,
            )
        )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    candidate = load_keilongw_model(temp_out) if temp_out.exists() else model
    candidate_mae = _evaluate_model_mae(candidate, X_val, y_val)
    previous_best = _evaluate_model_mae(load_keilongw_model(out), X_val, y_val) if out.exists() else np.inf
    improved = candidate_mae <= previous_best
    if improved:
        candidate.save(out)
        model = candidate
        best_mae = candidate_mae
    else:
        model = load_keilongw_model(out)
        best_mae = previous_best
    return (
        model,
        best_mae,
        {
            "stage": stage_name,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "initial_mae_fraction": float(initial_mae),
            "candidate_mae_fraction": float(candidate_mae),
            "best_mae_fraction": float(best_mae),
            "improved": bool(improved),
            "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        },
    )


def _evaluate_model_mae(model, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate clipped SOC MAE for a Keras model on prepared windows."""
    pred = np.asarray(model.predict(X, verbose=0)).reshape(len(X), -1)[:, 0]
    return float(np.mean(np.abs(np.clip(pred, 0.0, 1.0) - y)))


def _merge_histories(stages: Sequence[dict]) -> dict:
    """Flatten staged Keras histories for backward-compatible metrics JSON."""
    merged: dict[str, list[float]] = {}
    for stage in stages:
        for key, values in stage.get("history", {}).items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def _cap_samples(X: np.ndarray, y: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomly cap SOC windows to keep CPU fine-tune memory bounded."""
    if len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(2026)
    idx = np.sort(rng.choice(len(y), size=max_samples, replace=False))
    return X[idx], y[idx]


def main() -> None:
    """Run the SOC fine-tuning CLI and write metrics next to the model."""
    parser = argparse.ArgumentParser(description="Fine-tune KeiLongW SOC model on NASA ARC-FY08Q4")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--arc-dir", type=Path, default=Path("data/nasa_pcoe/ARC-FY08Q4"))
    parser.add_argument("--pcoe-dir", type=Path, default=Path("data/nasa_pcoe/B000x"))
    parser.add_argument("--out", type=Path, default=Path("outputs/soc_finetuned.h5"))
    parser.add_argument("--split-mode", choices=("arc", "pcoe-cell"), default="arc")
    parser.add_argument("--train-cells", nargs="+", default=["B0005", "B0006", "B0007"])
    parser.add_argument("--holdout-cells", nargs="+", default=["B0018"])
    parser.add_argument("--label-mode", choices=("strict", "legacy"), default="strict")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage3-epochs", type=int, default=0)
    parser.add_argument("--stage1-learning-rate", type=float, default=3e-4)
    parser.add_argument("--stage2-learning-rate", type=float, default=None)
    parser.add_argument("--stage3-learning-rate", type=float, default=1e-5)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=200_000)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--eval-stride", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=50_000)
    args = parser.parse_args()

    metrics = finetune_soc_model(
        args.weights,
        args.arc_dir,
        args.out,
        pcoe_dir=args.pcoe_dir,
        split_mode=args.split_mode,
        train_cells=args.train_cells,
        holdout_cells=args.holdout_cells,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        max_samples=args.max_samples,
        limit_files=args.limit_files,
        learning_rate=args.learning_rate,
        early_stop_patience=args.early_stop_patience,
        label_mode=args.label_mode,
        two_stage=args.two_stage,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        stage1_learning_rate=args.stage1_learning_rate,
        stage2_learning_rate=args.stage2_learning_rate,
        stage3_learning_rate=args.stage3_learning_rate,
    )
    if args.pcoe_dir.exists():
        try:
            metrics["pcoe_holdout"] = evaluate_on_pcoe(
                args.out,
                args.pcoe_dir,
                window=args.window,
                stride=args.eval_stride,
                max_samples=args.eval_max_samples,
                cells=args.holdout_cells if args.split_mode == "pcoe-cell" else None,
                label_mode=args.label_mode,
            )
        except Exception as exc:
            metrics["pcoe_holdout_error"] = repr(exc)
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
