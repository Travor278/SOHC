"""Plot SOC prediction curves against NASA Coulomb-counted references."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from craic_pipeline.soc_finetune import _estimate_soc_labels
from craic_pipeline.soc_inference import (
    _read_input_frame,
    infer_model_window,
    load_keilongw_model,
    predict_soc,
    preprocess_sequence,
)


def build_soc_plot_frame(
    weights: Path,
    data: Path,
    *,
    cycle_id: int | None = None,
    stride: int = 20,
    max_points: int = 2000,
) -> pd.DataFrame:
    """Build a plot DataFrame with SOC prediction and NASA reference labels."""
    df = _read_input_frame(data)
    labels = _estimate_soc_labels(df, mode="strict")
    model = load_keilongw_model(weights)
    window = infer_model_window(model) or 100
    frames: list[pd.DataFrame] = []

    groups = df.groupby("cycle_id", sort=False) if "cycle_id" in df else [(0, df)]
    for cid, group in groups:
        if cycle_id is not None and int(cid) != int(cycle_id):
            continue
        group_idx = group.index.to_numpy(dtype=int)
        if len(group) < window or not np.isfinite(labels[group_idx]).any():
            continue
        try:
            X, rows = preprocess_sequence(group, window, stride=stride)
        except ValueError:
            continue
        row_idx = rows.index.to_numpy(dtype=int)
        y_ref = labels[row_idx]
        finite = np.isfinite(y_ref)
        if not np.any(finite):
            continue
        rows = rows.loc[finite].copy()
        rows["soc_ref"] = y_ref[finite]
        rows["soc_pred"] = predict_soc(model, X[finite])
        frames.append(rows)
        if cycle_id is not None:
            break

    if not frames:
        raise ValueError(f"no labeled SOC windows available for {data}")
    out = pd.concat(frames, ignore_index=True)
    if len(out) > max_points:
        positions = np.linspace(0, len(out) - 1, max_points).astype(int)
        out = out.iloc[positions].reset_index(drop=True)
    return out


def plot_soc_prediction(frame: pd.DataFrame, out: Path, *, title: str | None = None) -> None:
    """Save a PNG SOC prediction plot for NASA W1 reporting."""
    out.parent.mkdir(parents=True, exist_ok=True)
    t = frame["t"].to_numpy(dtype=float)
    if len(t) and np.nanmax(t) > 600:
        x = t / 60.0
        xlabel = "time (min)"
    else:
        x = t
        xlabel = "time (s)"

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, frame["soc_ref"] * 100.0, label="NASA reference", linewidth=1.8)
    axes[0].plot(x, frame["soc_pred"] * 100.0, label="SOC prediction", linewidth=1.5)
    axes[0].set_ylabel("SOC (%)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, frame["voltage"], color="#2f6f9f", linewidth=1.2)
    axes[1].set_ylabel("V (V)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, frame["current"], color="#9f4f2f", linewidth=1.2)
    axes[2].set_ylabel("I (A)")
    axes[2].set_xlabel(xlabel)
    axes[2].grid(True, alpha=0.25)

    if title:
        fig.suptitle(title)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    """Run SOC plotting CLI and save PNG plus companion CSV."""
    parser = argparse.ArgumentParser(description="Plot SOC prediction against NASA reference labels")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/soc_prediction.png"))
    parser.add_argument("--cycle-id", type=int, default=None)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    frame = build_soc_plot_frame(
        args.weights,
        args.data,
        cycle_id=args.cycle_id,
        stride=args.stride,
        max_points=args.max_points,
    )
    plot_soc_prediction(frame, args.out, title=args.title)
    csv_path = args.out.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)
    mae = float(np.mean(np.abs(frame["soc_pred"].to_numpy() - frame["soc_ref"].to_numpy())) * 100.0)
    print(f"wrote {args.out} and {csv_path}; plotted_mae_percent={mae:.3f}; rows={len(frame)}")


if __name__ == "__main__":
    main()
