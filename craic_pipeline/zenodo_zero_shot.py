"""Zenodo 6985321 zero-shot SOC/SOH evaluation and plotting.

The dataset contains fresh/aged lithium-ion cell experiments plus an OCV-SOC
curve. This module reconstructs a Coulomb-counted SOC reference with voltage
anchoring, runs the W1 SOC LSTM with C-rate-normalized current, and estimates
cycle-level SOH from full half-cycle charge throughput. The SOH part is a
label-reconstruction diagnostic, not a deployed capacity-free SOH estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ZENODO_6985321_DOI = "10.5281/zenodo.6985321"
NOMINAL_CAPACITY_AH = 19.96
NASA_EQUIVALENT_CAPACITY_AH = 2.0


def load_zenodo_cell(csv_path: Path) -> pd.DataFrame:
    """Load a Zenodo 6985321 fresh/aged CSV with normalized column names."""
    frame = pd.read_csv(csv_path)
    rename = {
        "Time": "time_s",
        "Current": "current_A",
        "Voltage": "voltage_V",
        "Temperature": "temperature_C",
    }
    frame = frame.rename(columns=rename)
    required = list(rename.values())
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"missing columns in {csv_path}: {missing}")
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna().reset_index(drop=True)
    return frame


def reconstruct_soc_reference(frame: pd.DataFrame, *, capacity_ah: float = NOMINAL_CAPACITY_AH) -> pd.Series:
    """Reconstruct SOC from current integration with voltage endpoint anchoring.

    Zenodo 6985321 starts from a fully discharged cell and uses positive current
    as discharge in the accompanying Matlab script. We follow the same sign
    convention and anchor SOC at lower/upper voltage endpoints.
    """
    time = frame["time_s"].to_numpy(dtype=float)
    current = frame["current_A"].to_numpy(dtype=float)
    voltage = frame["voltage_V"].to_numpy(dtype=float)
    dt = np.diff(time, prepend=time[0])
    dt = np.clip(dt, 0.0, 10.0)
    soc = np.zeros(len(frame), dtype=float)
    for idx in range(1, len(frame)):
        soc[idx] = soc[idx - 1] - dt[idx] * current[idx] / (capacity_ah * 3600.0)
        if (voltage[idx] > 4.19 and abs(current[idx]) < 4.0) or soc[idx] > 1.0:
            soc[idx] = 1.0
        elif (voltage[idx] < 3.01 and abs(current[idx]) < 4.0) or soc[idx] < 0.0:
            soc[idx] = 0.0
    return pd.Series(np.clip(soc, 0.0, 1.0), name="soc_ref")


def estimate_half_cycle_soh(frame: pd.DataFrame, soc_ref: pd.Series, *, nominal_ah: float = NOMINAL_CAPACITY_AH) -> pd.DataFrame:
    """Estimate SOH from half-cycle throughput between SOC endpoint anchors."""
    time = frame["time_s"].to_numpy(dtype=float)
    current = frame["current_A"].to_numpy(dtype=float)
    soc = soc_ref.to_numpy(dtype=float)
    endpoint = np.full(len(soc), np.nan)
    endpoint[soc <= 0.01] = 0.0
    endpoint[soc >= 0.99] = 1.0

    rows = []
    last_idx = None
    last_endpoint = None
    for idx, value in enumerate(endpoint):
        if not np.isfinite(value):
            continue
        if last_endpoint is None:
            last_idx, last_endpoint = idx, value
            continue
        if value == last_endpoint:
            last_idx = idx
            continue
        segment = slice(last_idx, idx + 1)
        dt = np.diff(time[segment], prepend=time[last_idx])
        throughput_ah = float(np.sum(np.abs(current[segment]) * np.clip(dt, 0.0, 10.0)) / 3600.0)
        if throughput_ah > 0.25 * nominal_ah:
            rows.append(
                {
                    "time_h": float(time[idx] / 3600.0),
                    "start_soc": float(last_endpoint),
                    "end_soc": float(value),
                    "capacity_Ah": throughput_ah,
                    "soh_est": float(np.clip(throughput_ah / nominal_ah, 0.0, 1.2)),
                }
            )
        last_idx, last_endpoint = idx, value
    return pd.DataFrame(rows)


def run_soc_lstm(frame: pd.DataFrame, weights: Path, *, stride: int = 30) -> pd.DataFrame:
    """Run the W1 SOC LSTM on Zenodo CSV data with sign/C-rate alignment."""
    from craic_pipeline.soc_inference import infer_model_window, load_keilongw_model, predict_soc, preprocess_sequence

    model = load_keilongw_model(weights)
    window = infer_model_window(model) or 100
    adapted = pd.DataFrame(
        {
            "t": frame["time_s"],
            "voltage": frame["voltage_V"],
            # Zenodo: positive is discharge. NASA W1 current sign is aligned by
            # converting to a 2 Ah 18650-equivalent C-rate and flipping sign.
            "current": -frame["current_A"] / NOMINAL_CAPACITY_AH * NASA_EQUIVALENT_CAPACITY_AH,
            "temperature": frame["temperature_C"],
        }
    )
    X, rows = preprocess_sequence(adapted, window=window, scaler_path=None, stride=stride)
    pred = predict_soc(model, X)
    return pd.DataFrame({"time_s": rows["t"].to_numpy(dtype=float), "soc_pred": pred})


def evaluate_zero_shot(data_dir: Path, weights: Path, out_dir: Path, *, stride: int = 30) -> dict:
    """Evaluate fresh and aged Zenodo cells and write CSV/PNG artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    frames = {}
    soh_frames = {}
    for label, filename in {
        "fresh": "Experimental_data_fresh_cell.csv",
        "aged": "Experimental_data_aged_cell.csv",
    }.items():
        frame = load_zenodo_cell(data_dir / filename)
        frame["soc_ref"] = reconstruct_soc_reference(frame)
        pred = run_soc_lstm(frame, weights, stride=stride)
        merged = pd.merge_asof(
            pred.sort_values("time_s"),
            frame[["time_s", "soc_ref", "voltage_V", "current_A", "temperature_C"]].sort_values("time_s"),
            on="time_s",
            direction="nearest",
        )
        merged["soc_abs_error"] = (merged["soc_pred"] - merged["soc_ref"]).abs()
        soh = estimate_half_cycle_soh(frame, frame["soc_ref"])
        frames[label] = merged
        soh_frames[label] = soh
        merged.to_csv(out_dir / f"zenodo_6985321_{label}_soc_predictions.csv", index=False)
        soh.to_csv(out_dir / f"zenodo_6985321_{label}_soh_half_cycles.csv", index=False)
        results[label] = {
            "samples": int(len(merged)),
            "duration_h": float(frame["time_s"].iloc[-1] / 3600.0),
            "soc_mae_percent": float(merged["soc_abs_error"].mean() * 100.0),
            "soc_p95_percent": float(merged["soc_abs_error"].quantile(0.95) * 100.0),
            "half_cycle_count": int(len(soh)),
            "median_half_cycle_soh_percent": float(soh["soh_est"].median() * 100.0) if len(soh) else float("nan"),
        }
    _plot_zero_shot(frames, soh_frames, out_dir)
    metrics = {
        "source": ZENODO_6985321_DOI,
        "soc_weights": str(weights),
        "prediction_stride": int(stride),
        "current_adapter": "I_lstm=-I_zenodo/19.96Ah*2.0Ah",
        "results": results,
        "note": "SOC is a cross-dataset zero-shot diagnostic; SOH is reconstructed from endpoint-anchored half-cycle throughput.",
    }
    (out_dir / "zenodo_6985321_zero_shot_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _plot_zero_shot(frames: dict[str, pd.DataFrame], soh_frames: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Save IEEE-style zero-shot SOC/SOH figure."""
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
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex="col")
    for col, label in enumerate(["fresh", "aged"]):
        frame = frames[label].iloc[:: max(len(frames[label]) // 2500, 1)]
        t_h = frame["time_s"] / 3600.0
        axes[0, col].plot(t_h, frame["soc_ref"] * 100.0, color="#111111", label="Coulomb/OCV reference")
        axes[0, col].plot(t_h, frame["soc_pred"] * 100.0, color="#1F77B4", linestyle="--", label="W1 LSTM zero-shot")
        axes[0, col].set_title(f"{label.capitalize()} cell SOC")
        axes[0, col].set_ylabel("SOC (%)")
        axes[0, col].grid(True)
        axes[0, col].legend(frameon=False, loc="best")

        soh = soh_frames[label]
        if len(soh):
            axes[1, col].plot(soh["time_h"], soh["soh_est"] * 100.0, color="#A23B3B", marker="o", markersize=2.5)
        reference = 100.0 if label == "fresh" else 83.2
        axes[1, col].axhline(reference, color="#303030", linestyle=":", label=f"reference {reference:.1f}%")
        axes[1, col].set_xlabel("Time (h)")
        axes[1, col].set_ylabel("SOH (%)")
        axes[1, col].set_title("Half-cycle throughput SOH")
        axes[1, col].grid(True)
        axes[1, col].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig15_zenodo_6985321_zero_shot.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig15_zenodo_6985321_zero_shot.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point for Zenodo 6985321 zero-shot evaluation."""
    parser = argparse.ArgumentParser(description="Zenodo 6985321 zero-shot SOC/SOH evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/zenodo_6985321"))
    parser.add_argument("--weights", type=Path, default=Path("outputs/soc_finetuned.h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures/zenodo_6985321"))
    parser.add_argument("--stride", type=int, default=30)
    args = parser.parse_args()
    metrics = evaluate_zero_shot(args.data_dir, args.weights, args.out_dir, stride=args.stride)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
