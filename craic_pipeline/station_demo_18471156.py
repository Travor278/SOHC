"""Qualitative station-data demo for Zenodo 18471156.

The dataset provides real-world energy-storage station monitoring CSV files
with eight cell voltages, eight temperatures, and pack current per file. It
does not provide SOC/SOH labels or capacity checks, so this module produces a
qualitative W5 figure rather than a quantitative benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ZENODO_18471156_DOI = "10.5281/zenodo.18471156"


def list_station_csvs(root: Path) -> list[Path]:
    """Return all Zenodo 18471156 station CSV files under an extracted root."""
    return sorted(Path(root).rglob("*.csv"))


def load_station_csv(csv_path: Path) -> pd.DataFrame:
    """Load one station CSV containing vol_*, temp_*, and cur columns."""
    frame = pd.read_csv(csv_path)
    vol_cols = _ordered_columns(frame, "vol_")
    temp_cols = _ordered_columns(frame, "temp_")
    if not vol_cols or "cur" not in frame.columns:
        raise ValueError(f"{csv_path} is missing vol_* or cur columns")
    keep = vol_cols + temp_cols + ["cur"]
    frame = frame[keep].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return frame


def select_high_spread_file(root: Path, *, max_files: int | None = None) -> tuple[Path, dict]:
    """Select a station CSV with high cell-voltage spread for demonstration."""
    candidates = list_station_csvs(root)
    if max_files:
        candidates = candidates[:max_files]
    if not candidates:
        raise FileNotFoundError(f"no station CSV files found under {root}")

    best_path: Path | None = None
    best_stats: dict | None = None
    best_score = -np.inf
    for path in candidates:
        try:
            frame = pd.read_csv(path, nrows=4000)
        except Exception:
            continue
        vol_cols = _ordered_columns(frame, "vol_")
        if not vol_cols:
            continue
        values = frame[vol_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        spread_mv = (np.nanmax(values, axis=1) - np.nanmin(values, axis=1)) * 1000.0
        if not np.isfinite(spread_mv).any():
            continue
        current = pd.to_numeric(frame.get("cur", pd.Series(np.zeros(len(frame)))), errors="coerce").to_numpy(dtype=float)
        stats = {
            "file": str(path),
            "rows_scanned": int(len(frame)),
            "spread_p95_mV": float(np.nanpercentile(spread_mv, 95)),
            "spread_max_mV": float(np.nanmax(spread_mv)),
            "current_abs_p95_A": float(np.nanpercentile(np.abs(current), 95)) if np.isfinite(current).any() else 0.0,
        }
        score = stats["spread_p95_mV"] + 0.01 * stats["current_abs_p95_A"]
        if score > best_score:
            best_score = score
            best_path = path
            best_stats = stats
    if best_path is None or best_stats is None:
        raise RuntimeError(f"could not select a valid station CSV under {root}")
    return best_path, best_stats


def build_station_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute voltage/current/temperature envelopes and consistency proxy."""
    vol_cols = _ordered_columns(frame, "vol_")
    temp_cols = _ordered_columns(frame, "temp_")
    voltage = frame[vol_cols].to_numpy(dtype=float)
    temps = frame[temp_cols].to_numpy(dtype=float) if temp_cols else np.full_like(voltage, np.nan)
    spread_mv = (np.nanmax(voltage, axis=1) - np.nanmin(voltage, axis=1)) * 1000.0
    temp_spread = np.nanmax(temps, axis=1) - np.nanmin(temps, axis=1) if temp_cols else np.zeros(len(frame))
    consistency_proxy = 100.0 - np.clip(spread_mv / 1000.0 * 50.0 + temp_spread * 0.5, 0.0, 45.0)
    return pd.DataFrame(
        {
            "sample": np.arange(len(frame), dtype=int),
            "current_A": frame["cur"].to_numpy(dtype=float),
            "voltage_min_V": np.nanmin(voltage, axis=1),
            "voltage_mean_V": np.nanmean(voltage, axis=1),
            "voltage_max_V": np.nanmax(voltage, axis=1),
            "voltage_spread_mV": spread_mv,
            "temperature_min_C": np.nanmin(temps, axis=1) if temp_cols else np.nan,
            "temperature_mean_C": np.nanmean(temps, axis=1) if temp_cols else np.nan,
            "temperature_max_C": np.nanmax(temps, axis=1) if temp_cols else np.nan,
            "consistency_proxy_percent": pd.Series(consistency_proxy).rolling(15, min_periods=1, center=True).mean(),
        }
    )


def run_station_soc_lstm(frame: pd.DataFrame, weights: Path, *, stride: int = 20) -> tuple[pd.DataFrame, str]:
    """Run W1 SOC LSTM over each station cell using a qualitative current adapter."""
    from craic_pipeline.soc_inference import infer_model_window, load_keilongw_model, predict_soc, preprocess_sequence

    model = load_keilongw_model(weights)
    window = infer_model_window(model) or 100
    vol_cols = _ordered_columns(frame, "vol_")
    temp_cols = _ordered_columns(frame, "temp_")
    current = frame["cur"].to_numpy(dtype=float)
    current_scale = max(float(np.nanpercentile(np.abs(current), 95)), 1.0)
    adapted_current = np.clip(-current / current_scale * 2.0, -5.0, 5.0)
    preds = []
    aligned_samples: np.ndarray | None = None
    for idx, vol_col in enumerate(vol_cols):
        temp_col = temp_cols[min(idx, len(temp_cols) - 1)] if temp_cols else None
        sample = pd.DataFrame(
            {
                "t": np.arange(len(frame), dtype=float),
                "voltage": frame[vol_col].to_numpy(dtype=float),
                "current": adapted_current,
                "temperature": frame[temp_col].to_numpy(dtype=float) if temp_col else np.full(len(frame), 25.0),
            }
        )
        X, rows = preprocess_sequence(sample, window=window, scaler_path=None, stride=stride)
        pred = predict_soc(model, X)
        if aligned_samples is None:
            aligned_samples = rows["t"].to_numpy(dtype=int)
        preds.append(pred)
    stack = np.vstack(preds)
    output = pd.DataFrame(
        {
            "sample": aligned_samples,
            "soc_mean_percent": np.mean(stack, axis=0) * 100.0,
            "soc_min_percent": np.min(stack, axis=0) * 100.0,
            "soc_max_percent": np.max(stack, axis=0) * 100.0,
        }
    )
    note = f"W1 LSTM qualitative SOC, current_adapter=-cur/p95_abs_current*2A, p95_abs_current={current_scale:.3f}A"
    return output, note


def voltage_soc_proxy(frame: pd.DataFrame, *, stride: int = 20) -> tuple[pd.DataFrame, str]:
    """Return a voltage-normalized SOC proxy when the W1 LSTM is unavailable."""
    vol_cols = _ordered_columns(frame, "vol_")
    voltage = frame[vol_cols].to_numpy(dtype=float)
    mean_v = np.nanmean(voltage, axis=1)
    soc = np.clip((mean_v - 3.0) / (4.2 - 3.0), 0.0, 1.0) * 100.0
    samples = np.arange(0, len(frame), stride, dtype=int)
    out = pd.DataFrame(
        {
            "sample": samples,
            "soc_mean_percent": soc[samples],
            "soc_min_percent": np.clip((np.nanmin(voltage, axis=1)[samples] - 3.0) / 1.2, 0.0, 1.0) * 100.0,
            "soc_max_percent": np.clip((np.nanmax(voltage, axis=1)[samples] - 3.0) / 1.2, 0.0, 1.0) * 100.0,
        }
    )
    return out, "voltage-normalized SOC proxy; W1 LSTM was not used"


def make_station_demo(
    data_root: Path,
    out_dir: Path,
    *,
    weights: Path | None = None,
    csv_path: Path | None = None,
    max_files: int | None = None,
    stride: int = 20,
) -> dict:
    """Generate Zenodo 18471156 station diagnostics CSV, metrics, and figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_stats = {}
    if csv_path is None:
        csv_path, selected_stats = select_high_spread_file(data_root, max_files=max_files)
    frame = load_station_csv(csv_path)
    diagnostics = build_station_diagnostics(frame)
    soc_note = ""
    if weights and Path(weights).exists():
        try:
            soc, soc_note = run_station_soc_lstm(frame, Path(weights), stride=stride)
        except Exception as exc:
            soc, soc_note = voltage_soc_proxy(frame, stride=stride)
            soc_note = f"{soc_note}; LSTM failed: {exc!r}"
    else:
        soc, soc_note = voltage_soc_proxy(frame, stride=stride)
    merged = pd.merge_asof(
        diagnostics.sort_values("sample"),
        soc.sort_values("sample"),
        on="sample",
        direction="nearest",
    )
    merged.to_csv(out_dir / "zenodo_18471156_station_demo.csv", index=False)
    metrics = {
        "source": ZENODO_18471156_DOI,
        "selected_file": str(csv_path),
        "selection": selected_stats,
        "rows": int(len(frame)),
        "voltage_spread_initial_mV": float(diagnostics["voltage_spread_mV"].iloc[0]),
        "voltage_spread_p95_mV": float(diagnostics["voltage_spread_mV"].quantile(0.95)),
        "voltage_spread_max_mV": float(diagnostics["voltage_spread_mV"].max()),
        "current_abs_p95_A": float(np.nanpercentile(np.abs(diagnostics["current_A"]), 95)),
        "soc_output_note": soc_note,
        "soh_output_note": "No capacity/SOH labels are available; consistency_proxy_percent is a voltage/temp spread proxy, not quantitative SOH.",
    }
    (out_dir / "zenodo_18471156_station_demo_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_station_demo(merged, out_dir, metrics)
    return metrics


def _plot_station_demo(frame: pd.DataFrame, out_dir: Path, metrics: dict) -> None:
    """Save the IEEE-style Zenodo 18471156 qualitative station figure."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.35,
            "grid.alpha": 0.35,
            "svg.fonttype": "none",
        }
    )
    stride = max(len(frame) // 1600, 1)
    plot = frame.iloc[::stride].copy()
    sample = plot["sample"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True)
    axes[0, 0].plot(sample, plot["current_A"], color="#1F77B4")
    axes[0, 0].axhline(0.0, color="#303030", linestyle=":", linewidth=0.7)
    axes[0, 0].set_title("Station current")
    axes[0, 0].set_ylabel("Current (A)")
    axes[0, 0].grid(True)

    axes[0, 1].fill_between(sample, plot["voltage_min_V"], plot["voltage_max_V"], color="#BFD7EA", alpha=0.65, label="cell envelope")
    axes[0, 1].plot(sample, plot["voltage_mean_V"], color="#1F77B4", label="mean")
    axes[0, 1].set_title(f"Voltage envelope, p95 spread={metrics['voltage_spread_p95_mV']:.1f} mV")
    axes[0, 1].set_ylabel("Voltage (V)")
    axes[0, 1].legend(frameon=False, loc="best")
    axes[0, 1].grid(True)

    axes[1, 0].fill_between(sample, plot["temperature_min_C"], plot["temperature_max_C"], color="#F6C6A8", alpha=0.70, label="cell envelope")
    axes[1, 0].plot(sample, plot["temperature_mean_C"], color="#A23B3B", label="mean")
    axes[1, 0].set_title("Temperature envelope")
    axes[1, 0].set_ylabel("Temperature (deg C)")
    axes[1, 0].set_xlabel("Sample index")
    axes[1, 0].legend(frameon=False, loc="best")
    axes[1, 0].grid(True)

    axes[1, 1].fill_between(sample, plot["soc_min_percent"], plot["soc_max_percent"], color="#D8E9D3", alpha=0.65, label="cell SOC range")
    axes[1, 1].plot(sample, plot["soc_mean_percent"], color="#2A7F62", label="SOC output")
    axes[1, 1].plot(sample, plot["consistency_proxy_percent"], color="#6B6B6B", linestyle="--", label="consistency proxy")
    axes[1, 1].set_title("Qualitative SOC and consistency output")
    axes[1, 1].set_ylabel("Output (%)")
    axes[1, 1].set_xlabel("Sample index")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].legend(frameon=False, loc="best")
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / "fig16_zenodo_18471156_station_demo.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig16_zenodo_18471156_station_demo.svg", bbox_inches="tight")
    plt.close(fig)


def _ordered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Sort prefixed columns by their numeric suffix."""
    cols = [col for col in frame.columns if str(col).startswith(prefix)]
    return sorted(cols, key=lambda col: int(str(col).split("_")[-1]))


def main() -> None:
    """CLI entry point for Zenodo 18471156 station qualitative plotting."""
    parser = argparse.ArgumentParser(description="Zenodo 18471156 station qualitative demo")
    parser.add_argument("--data-root", type=Path, default=Path("data/zenodo_18471156/BatteryData"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures/zenodo_18471156"))
    parser.add_argument("--weights", type=Path, default=Path("outputs/soc_finetuned.h5"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--stride", type=int, default=20)
    args = parser.parse_args()
    metrics = make_station_demo(
        args.data_root,
        args.out_dir,
        weights=args.weights,
        csv_path=args.csv,
        max_files=args.max_files,
        stride=args.stride,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
