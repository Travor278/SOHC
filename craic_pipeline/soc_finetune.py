"""Fine-tune KeiLongW SOC LSTM on NASA ARC-FY08Q4 data.

The script warm-starts from KeiLongW `.h5` artifacts, freezes the first two
LSTM layers, trains the final recurrent/dense head on ARC V/I/T sequences, and
writes `outputs/soc_finetuned.h5`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        labels = _estimate_soc_labels(df)
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


def finetune_soc_model(
    weights: Path,
    arc_dir: Path,
    out: Path,
    *,
    window: int | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    stride: int = 10,
    max_samples: int = 200_000,
    limit_files: int | None = None,
    learning_rate: float = 1e-3,
    early_stop_patience: int = 5,
) -> dict:
    """Fine-tune the KeiLongW SOC model and save a Keras `.h5` artifact."""
    from tensorflow import keras

    model = load_keilongw_model(weights)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=["mae"])
    fixed_window = infer_model_window(model)
    window = window or fixed_window or 100
    if fixed_window is not None and window != fixed_window:
        raise ValueError(f"weights require window={fixed_window}, got window={window}")
    _freeze_first_lstm_layers(model, count=2)
    X_train, y_train, X_val, y_val = build_soc_training_set(
        arc_dir,
        window,
        stride=stride,
        max_samples=max_samples,
        limit_files=limit_files,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(out),
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
    if out.exists():
        model = load_keilongw_model(out)
    pred = np.asarray(model.predict(X_val, verbose=0)).reshape(len(X_val), -1)[:, 0]
    mae = float(np.mean(np.abs(np.clip(pred, 0.0, 1.0) - y_val)))
    if not out.exists():
        model.save(out)
    return {
        "train_samples": int(len(X_train)),
        "holdout_samples": int(len(X_val)),
        "limit_files": limit_files,
        "learning_rate": learning_rate,
        "holdout_mae_fraction": mae,
        "holdout_mae_percent": mae * 100.0,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }


def evaluate_on_pcoe(
    weights: Path,
    pcoe_dir: Path,
    *,
    window: int | None = None,
    stride: int = 10,
    max_samples: int = 50_000,
) -> dict:
    """Evaluate a fine-tuned SOC model on NASA B0005-B0018 holdout files."""
    model = load_keilongw_model(weights)
    window = window or infer_model_window(model) or 100
    files = sorted(Path(pcoe_dir).rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no PCoE .mat files found under {pcoe_dir}")
    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for file_path in files:
        df = _frame_from_loader(file_path, load_pcoe_basic)
        if len(df) < window:
            continue
        labels = _estimate_soc_labels(df)
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
    return {"samples": int(len(y)), "mae_fraction": mae, "mae_percent": mae * 100.0}


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


def _estimate_soc_labels(df: pd.DataFrame) -> np.ndarray:
    """Create approximate SOC labels from NASA current/time/capacity samples."""
    labels = np.full(len(df), np.nan, dtype=float)
    for _, group in df.groupby("cycle_id", sort=False):
        idx = group.index.to_numpy()
        current = np.nan_to_num(group["current"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        time = np.nan_to_num(group["t"].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        capacity = group["capacity"].replace([np.inf, -np.inf], np.nan).dropna()
        if capacity.empty or capacity.iloc[0] <= 0:
            continue
        cap = float(capacity.iloc[0])
        mean_current = float(np.nanmean(current)) if current.size else 0.0
        if mean_current >= -0.05:
            continue
        dt = np.diff(time, prepend=time[0])
        dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
        ah = np.cumsum(np.abs(current) * dt) / 3600.0
        terminal_ah = float(ah[-1]) if np.isfinite(ah[-1]) else 0.0
        span = max(terminal_ah, cap, 1e-6)
        soc = 1.0 - ah / span
        labels[idx] = np.clip(soc, 0.0, 1.0)
    return labels


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
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        max_samples=args.max_samples,
        limit_files=args.limit_files,
        learning_rate=args.learning_rate,
        early_stop_patience=args.early_stop_patience,
    )
    if args.pcoe_dir.exists():
        try:
            metrics["pcoe_holdout"] = evaluate_on_pcoe(
                args.out,
                args.pcoe_dir,
                window=args.window,
                stride=args.eval_stride,
                max_samples=args.eval_max_samples,
            )
        except Exception as exc:
            metrics["pcoe_holdout_error"] = repr(exc)
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
