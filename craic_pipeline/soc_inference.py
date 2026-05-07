"""SOC 推断：KeiLongW Stacked LSTM (TF/Keras) warm-start + NASA fine-tune (v0.2)。

数据策略 v0.2：
    - KeiLongW 预训练权重（在 LG 18650HG2 上训）作为 warm-start
    - 用 NASA BatteryAgingARC-FY08Q4（B0025-B0056，多温度多倍率）做 fine-tune
    - fine-tune 后的模型在 NASA B0005-B0018 上 inference，给 Mamba 世界模型打 SOC 软标签

输入：
    --weights      : KeiLongW release 里的 .h5（warm-start 用）
    --finetune-data: NASA ARC-FY08Q4 .mat 目录（fine-tune 用）
    --data         : 推断输入 V/I/T 时序 CSV / .mat
    --window       : 滑动窗口长度（KeiLongW 默认 100 / 200 / 300）
    --out          : 输出 CSV 路径

输出：
    CSV 含列 [t, voltage, current, temperature, ambient_T, soc_pred]

W1 任务：fine-tune + 在 NASA holdout 上验证 MAE < 1.5%。
W2 任务：跑 NASA B0005-B0018 + Randomized，输出软标签供 Mamba 训练。
W5 任务：跑 Zenodo 6985321 (WLTP) 定量验证 + Zenodo 18471156 定性展示。
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


_FEATURE_COLUMNS = {
    "voltage": ("voltage", "Voltage", "V", "voltage_in_V", "Voltage_measured"),
    "current": ("current", "Current", "I", "current_in_A", "Current_measured"),
    "temperature": ("temperature", "Temperature", "T", "temperature_in_C", "Temperature_measured"),
}


def build_keilongw_model(input_shape: tuple[int | None, int] = (None, 3)):
    """Build the W1 KeiLongW-compatible LSTM for V/I/T SOC sequences."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(256, activation="selu", return_sequences=True, name="lstm_0"),
            keras.layers.LSTM(256, activation="selu", return_sequences=True, name="lstm_1"),
            keras.layers.LSTM(128, activation="selu", return_sequences=False, name="lstm_2"),
            keras.layers.Dense(64, activation="selu", name="dense_0"),
            keras.layers.Dense(1, activation="linear", name="soc"),
        ],
        name="keilongw_nasa_soc",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def load_keilongw_model(weights_path: Path):
    """Load a KeiLongW Keras SOC model or weights file.

    Args:
        weights_path: `.h5` from KeiLongW release or `outputs/soc_finetuned.h5`.

    Returns:
        A compiled Keras model for SOC prediction from `(V, I, T)` windows.
    """
    from tensorflow import keras

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"SOC weights not found: {weights_path}")

    try:
        model = keras.models.load_model(weights_path, compile=False)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        return model
    except Exception as load_error:
        model = build_keilongw_model()
        try:
            model.load_weights(weights_path)
            return model
        except Exception as weight_error:
            raise RuntimeError(
                f"Unable to load {weights_path} as a full model or planned W1 weights. "
                f"load_model error={load_error!r}; load_weights error={weight_error!r}"
            ) from weight_error


def preprocess_sequence(df, window: int, scaler_path: Path | None = None):
    """Convert NASA/LG V/I/T samples into normalized sliding SOC windows.

    Args:
        df: DataFrame with voltage/current/temperature columns.
        window: Number of previous samples per LSTM input window.
        scaler_path: Optional KeiLongW scaler pickle from release artifacts.

    Returns:
        `(X, rows)` where `X` is `(N, window, 3)` and `rows` are aligned input rows.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    features = _extract_vit(df)
    if len(features) < window:
        raise ValueError(f"need at least {window} samples, got {len(features)}")

    scaled = _scale_features(features, scaler_path)
    X = np.stack([scaled[i - window + 1 : i + 1] for i in range(window - 1, len(scaled))])
    aligned_rows = df.iloc[window - 1 :].reset_index(drop=True)
    return X.astype(np.float32), aligned_rows


def predict_soc(model, X) -> "np.ndarray":
    """Run one Keras forward pass and return `(N,)` SOC predictions."""
    pred = model.predict(X, verbose=0)
    pred = np.asarray(pred)
    if pred.ndim == 3:
        pred = pred[:, -1, :]
    pred = pred.reshape(pred.shape[0], -1)[:, 0]
    return np.clip(pred.astype(float), 0.0, 1.0)


def _extract_vit(df: pd.DataFrame) -> np.ndarray:
    """Extract finite voltage/current/temperature columns from input samples."""
    columns = []
    for logical_name, candidates in _FEATURE_COLUMNS.items():
        match = next((col for col in candidates if col in df.columns), None)
        if match is None:
            lowered = {col.lower(): col for col in df.columns}
            match = next((lowered.get(col.lower()) for col in candidates if col.lower() in lowered), None)
        if match is None:
            raise ValueError(f"missing {logical_name} column; available={list(df.columns)}")
        columns.append(match)
    values = df[columns].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    return values[mask]


def _scale_features(features: np.ndarray, scaler_path: Path | None) -> np.ndarray:
    """Scale V/I/T features with a KeiLongW scaler or physical fallback ranges."""
    if scaler_path and scaler_path.exists():
        with open(scaler_path, "rb") as fin:
            scaler = pickle.load(fin)
        if isinstance(scaler, (list, tuple)):
            scaled = np.empty_like(features, dtype=float)
            for idx, one_scaler in enumerate(scaler[: features.shape[1]]):
                scaled[:, idx] = one_scaler.transform(features[:, [idx]]).reshape(-1)
            return scaled
        return scaler.transform(features)

    # LG loader uses min-max scaling. Without release scalers, use conservative
    # physical ranges so inference can run on NASA before fine-tune artifacts exist.
    mins = np.array([2.0, -5.0, -20.0], dtype=float)
    maxs = np.array([4.5, 5.0, 80.0], dtype=float)
    return np.clip((features - mins) / (maxs - mins), 0.0, 1.0)


def _read_input_frame(path: Path) -> pd.DataFrame:
    """Read CSV or NASA `.mat` input into a SOC inference DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".mat":
        from craic_pipeline.nasa_loader import (
            load_arc_fy08q4,
            load_pcoe_basic,
            load_randomized_usage,
        )

        loaders = (load_pcoe_basic, load_arc_fy08q4, load_randomized_usage)
        last_error: Exception | None = None
        for loader in loaders:
            try:
                V, I, T, t, cycle_id, ambient, capacity = loader(path)
                if V.size:
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
            except Exception as exc:
                last_error = exc
        raise ValueError(f"could not parse .mat input {path}") from last_error
    raise ValueError(f"unsupported SOC input file type: {path.suffix}")


def main():
    """Run the SOC inference CLI and write predictions to CSV."""
    parser = argparse.ArgumentParser(description="SOC inference via KeiLongW LSTM")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/soc_pred.csv"))
    args = parser.parse_args()

    df = _read_input_frame(args.data)
    X, rows = preprocess_sequence(df, args.window, args.scaler)
    model = load_keilongw_model(args.weights)
    rows = rows.copy()
    rows["soc_pred"] = predict_soc(model, X)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep = [col for col in ["t", "voltage", "current", "temperature", "ambient_T", "soc_pred"] if col in rows]
    rows[keep].to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
