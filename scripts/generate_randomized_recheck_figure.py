"""Generate the Randomized full-rollout recheck figure for the report."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = Path("outputs/world_model_randomized_full_stride64.metrics.json")
OUT_DIR = Path("paper_figures")


def set_style() -> None:
    """Configure compact, print-friendly plotting defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.35,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rw_sort_key(name: str) -> int:
    """Sort RW file ids by numeric suffix."""
    match = re.search(r"\d+", name)
    return int(match.group(0)) if match else 999


def weighted_mean(rows: pd.DataFrame, value: str, weight: str) -> float:
    """Compute a weighted mean for per-file metric summaries."""
    return float(np.average(rows[value].to_numpy(float), weights=rows[weight].to_numpy(float)))


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Create full and outlier-excluded metric rows for the figure."""
    normal = rows[~rows["file"].isin(["RW2", "RW3"])].copy()
    return pd.DataFrame(
        [
            {
                "scope": "Full 28 RW",
                "files": len(rows),
                "one_step_mV": weighted_mean(rows, "one_step_voltage_mae_mV", "windows"),
                "rollout_mV": weighted_mean(rows, "rollout_voltage_mae_mV", "rollout_errors"),
            },
            {
                "scope": "Without RW2/RW3",
                "files": len(normal),
                "one_step_mV": weighted_mean(normal, "one_step_voltage_mae_mV", "windows"),
                "rollout_mV": weighted_mean(normal, "rollout_voltage_mae_mV", "rollout_errors"),
            },
        ]
    )


def main() -> None:
    """Draw a two-panel Randomized full-rollout diagnostic figure."""
    set_style()
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = pd.DataFrame(data["files"]).sort_values("file", key=lambda col: col.map(rw_sort_key))
    summary = build_summary(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "fig19_randomized_rollout_recheck.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [2.15, 1.0]})
    ax = axes[0]
    colors = np.where(rows["file"].isin(["RW2", "RW3"]), "#B84A3A", "#2A7F62")
    x = np.arange(len(rows))
    ax.bar(x, rows["rollout_voltage_mae_mV"], color=colors, width=0.72, edgecolor="#303030", linewidth=0.25)
    ax.axhline(50, color="#303030", linestyle=":", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(5, 1300)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["file"], rotation=90)
    ax.set_ylabel("20-step rollout V MAE (mV, log)")
    ax.set_title("Per-file dynamic-load rollout error")
    ax.grid(True, axis="y", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(len(rows) - 0.2, 53, "50 mV ref.", ha="right", va="bottom", fontsize=7, color="#303030")
    for file_id in ["RW2", "RW3"]:
        idx = int(rows.index[rows["file"] == file_id][0])
        x_pos = int(np.where(rows["file"].to_numpy() == file_id)[0][0])
        y = float(rows.loc[idx, "rollout_voltage_mae_mV"])
        ax.text(x_pos, y * 1.08, file_id, ha="center", va="bottom", color="#B84A3A", fontsize=7)

    ax = axes[1]
    width = 0.34
    x = np.arange(len(summary))
    ax.bar(x - width / 2, summary["one_step_mV"], width, label="1-step", color="#4A90A4", edgecolor="#303030", linewidth=0.35)
    ax.bar(x + width / 2, summary["rollout_mV"], width, label="20-step", color="#A77D2D", edgecolor="#303030", linewidth=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.scope}\n({int(r.files)} files)" for r in summary.itertuples()], linespacing=1.1)
    ax.set_ylabel("Weighted V MAE (mV)")
    ax.set_title("Full vs robust summary")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=7)
    ax.set_ylim(0, max(summary["rollout_mV"]) * 1.22)

    fig.suptitle("NASA Randomized full recheck: strict stride=64, 28/28 RW files", fontsize=9.5, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.01,
        "RW2/RW3 dominate the full 20-step metric; the result is reported as a dynamic-load stress boundary, not a pass metric.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), w_pad=1.8)
    fig.savefig(OUT_DIR / "fig19_randomized_rollout_recheck.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig19_randomized_rollout_recheck.svg", bbox_inches="tight")
    print(json.dumps({"figure": "paper_figures/fig19_randomized_rollout_recheck.png", "summary": summary.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
