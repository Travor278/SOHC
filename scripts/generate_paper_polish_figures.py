"""Generate polished summary figures for the CRAIC2026 report.

These figures are intentionally derived from already accepted local metrics.
They do not introduce new experimental claims; they make the report and PPT
easier to read while Randomized full evaluation runs in the background.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT_DIR = Path("paper_figures")


def set_style() -> None:
    """Use a compact, print-friendly IEEE-like style."""
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


def save(fig, name: str) -> None:
    """Save a figure to PNG and SVG in the paper figure directory."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight")


def box(ax, x: float, y: float, w: float, h: float, text: str, *, fc: str, ec: str = "#303030") -> None:
    """Draw a rounded text box in axes coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.016,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.15)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Draw a slim arrow in axes coordinates."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=0.85, color="#303030", shrinkA=3, shrinkB=3),
    )


def metric_card(ax, x: float, y: float, w: float, h: float, value: str, label: str, *, fc: str) -> None:
    """Draw a metric card for summary dashboards."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=fc,
        edgecolor="#303030",
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.61, value, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, label, ha="center", va="center", fontsize=7.5, linespacing=1.1)


def fig_graphical_abstract() -> None:
    """Create a one-page graphical abstract for the system."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.95,
        "Safe AI Fast-Charging: State Estimation + Mamba World Model + SAC + ECM Safety",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    ax.text(
        0.5,
        0.89,
        "Training backbone: NASA PCoE NMC 18650; pack validation: UPC 36-cell; station display: Zenodo 18471156",
        ha="center",
        va="center",
        fontsize=7.7,
        color="#555555",
    )

    colors = ["#E9F5EE", "#FFF4E4", "#F3ECF8", "#FCEBEB", "#EEF2F2"]
    labels = [
        "W1\nSOC / SOH\nstate estimation",
        "W2\nMamba\nworld model",
        "W3\nSAC\nfast-charge policy",
        "L3\nECM safety\naction projection",
        "W5\nPack balancing\ncoordination",
    ]
    xs = [0.045, 0.245, 0.445, 0.645, 0.805]
    widths = [0.15, 0.15, 0.15, 0.15, 0.15]
    for i, (x, w, text, color) in enumerate(zip(xs, widths, labels, colors)):
        box(ax, x, 0.62, w, 0.17, text, fc=color)
        if i:
            arrow(ax, (xs[i - 1] + widths[i - 1], 0.705), (x, 0.705))

    metric_card(ax, 0.055, 0.34, 0.18, 0.18, "30.97%", "faster to 80% SOC\nSAC vs 3A CC-CV", fc="#E8EEF7")
    metric_card(ax, 0.285, 0.34, 0.18, 0.18, "17.37%", "lower single-cycle\nΔSOH", fc="#E9F5EE")
    metric_card(ax, 0.515, 0.34, 0.18, 0.18, "0", "actual over-voltage\nevents", fc="#FCEBEB")
    metric_card(ax, 0.745, 0.34, 0.18, 0.18, "46.30%", "UPC-initialized pack\nspread reduction", fc="#FFF4E4")

    ax.add_patch(Rectangle((0.055, 0.14), 0.87, 0.12, facecolor="#F7F7F7", edgecolor="#303030", linewidth=0.8))
    ax.text(
        0.49,
        0.20,
        "Current caveat: SOC holdout MAE = 3.48% (target 1.5% not yet reached).\n"
        "Zenodo station data has no SOC/SOH labels, so it is qualitative only.",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#333333",
    )

    save(fig, "fig00_graphical_abstract")
    plt.close(fig)


def fig_results_dashboard() -> None:
    """Create a compact dashboard of the accepted metrics and caveats."""
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.4))
    fig.suptitle("CRAIC2026 Current Evidence Map", fontsize=11, weight="bold", y=0.99)

    ax = axes[0, 0]
    vals = [596.5, 411.75]
    ax.bar(["CC-CV", "SAC"], vals, color=["#8C8C8C", "#1F77B4"], width=0.55)
    ax.set_ylabel("Time to 80% SOC (s)")
    ax.set_title("Single-cell fast charging")
    ax.grid(True, axis="y")
    ax.text(0.5, max(vals) * 0.92, "30.97% faster", ha="center", color="#1F77B4", weight="bold")

    ax = axes[0, 1]
    vals = [1.42, 8.04, 2.39]
    ax.bar(["1-step\nB0018", "20-step\nB0018", "1-step\nRandomized"], vals, color=["#2A7F62", "#2A7F62", "#A77D2D"], width=0.58)
    ax.set_ylabel("Voltage MAE (mV)")
    ax.set_title("World-model voltage accuracy")
    ax.grid(True, axis="y")
    ax.text(0.05, 7.15, "1-step target <5 mV\n20-step target <50 mV", fontsize=7, color="#303030")

    ax = axes[1, 0]
    vals = [1121, 668]
    ax.bar(["CC-CV", "SAC"], vals, color=["#8C8C8C", "#1F77B4"], width=0.55)
    ax.set_ylabel("Pack min-cell time (s)")
    ax.set_title("6S1P policy replication")
    ax.grid(True, axis="y")
    ax.text(0.5, max(vals) * 0.90, "40.41% faster\n23.01% lower ΔSOH", ha="center", color="#1F77B4", weight="bold")

    ax = axes[1, 1]
    labels = ["NASA\nSOC", "Zenodo\nfresh", "Zenodo\naged"]
    vals = [3.48, 16.11, 14.03]
    colors = ["#2A7F62", "#A23B3B", "#A23B3B"]
    ax.bar(labels, vals, color=colors, width=0.58)
    ax.axhline(1.5, color="#303030", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 18.5)
    ax.set_ylabel("SOC MAE (%)")
    ax.set_title("Generalization boundary")
    ax.grid(True, axis="y")
    ax.text(1.5, 17.1, "zero-shot gap", ha="center", color="#A23B3B", fontsize=7.5, weight="bold")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.2, h_pad=1.4)
    save(fig, "fig18_results_dashboard")
    plt.close(fig)


def fig_single_cell_trajectory_polished() -> None:
    """Create a cleaner W4 trajectory figure with direct evidence annotations."""
    traj = pd.read_csv("outputs/eval_w4_final_default/trajectories.csv")
    metrics = pd.read_csv("outputs/eval_w4_final_default/metrics_by_episode.csv")
    episode = 0
    names = {"cc_cv": "CC-CV", "mfcc": "MFCC", "ours": "SAC"}
    colors = {"cc_cv": "#6B6B6B", "mfcc": "#D9822B", "ours": "#1F77B4"}
    fields = [
        ("current_A", "Current (A)", "Charging current"),
        ("voltage", "Voltage (V)", "Terminal voltage"),
        ("soc", "SOC (%)", "SOC trajectory"),
        ("temperature", "Temperature (deg C)", "Temperature"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.4), sharex=True)
    for ax, (field, ylabel, title) in zip(axes.flat, fields):
        for strategy in ["cc_cv", "mfcc", "ours"]:
            part = traj[(traj["episode"] == episode) & (traj["strategy"] == strategy)].copy()
            if part.empty:
                continue
            y = part[field].to_numpy(dtype=float)
            if field == "soc":
                y = y * 100.0
            ax.plot(part["time_s"], y, color=colors[strategy], lw=1.5, label=names[strategy])
        if field == "voltage":
            ax.axhline(4.2, color="#303030", linestyle=":", lw=0.8)
            ax.text(0.98, 0.90, "4.2 V safety limit", transform=ax.transAxes, ha="right", va="center", fontsize=7)
        if field == "soc":
            ax.axhline(80, color="#303030", linestyle=":", lw=0.8)
            label_y = {"ours": 82.0, "cc_cv": 86.0}
            for strategy in ["ours", "cc_cv"]:
                row = metrics[(metrics["episode"] == episode) & (metrics["strategy"] == strategy)]
                if not row.empty and np.isfinite(float(row["time_to_80_s"].iloc[0])):
                    t_hit = float(row["time_to_80_s"].iloc[0])
                    ax.axvline(t_hit, color=colors[strategy], linestyle="--", lw=0.8, alpha=0.75)
                    ax.text(t_hit, label_y[strategy], f"{names[strategy]} {t_hit:.0f}s", color=colors[strategy], fontsize=7, ha="center")
            ax.set_ylim(12, 90)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.50, 0.925))
    fig.suptitle("Single-cell fast-charging trajectories, representative paired episode", y=0.99, fontsize=11, weight="bold")
    fig.text(0.52, 0.02, "SAC reaches 80% earlier while the ECM safety layer keeps terminal voltage at or below the limit.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), w_pad=1.2, h_pad=1.4)
    save(fig, "fig09_charging_comparison_polished")
    plt.close(fig)


def fig_pack_trajectory_polished() -> None:
    """Create a cleaner W5 6S1P paired-pack comparison figure."""
    traj = pd.read_csv("outputs/eval_pack_6s1p_h1200/pack_trajectories.csv")
    metrics = pd.read_csv("outputs/eval_pack_6s1p_h1200/pack_metrics_by_episode.csv")
    episode = 2
    names = {"cc_cv": "CC-CV", "mfcc": "MFCC", "ours": "SAC"}
    colors = {"cc_cv": "#6B6B6B", "mfcc": "#D9822B", "ours": "#1F77B4"}

    # Keep one pack-level row per time step; per-cell rows duplicate pack columns.
    pack = traj[traj["episode"] == episode].drop_duplicates(["strategy", "step"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.4), sharex=True)
    panels = [
        ("soc_min", "Min-cell SOC (%)", "Pack target variable"),
        ("soc_spread", "SOC spread (%)", "Cell-to-cell imbalance"),
        ("current_A", "Mean cell current (A)", "Mean charging current"),
        ("temperature_max", "Max temperature (deg C)", "Thermal envelope"),
    ]
    for ax, (field, ylabel, title) in zip(axes.flat, panels):
        for strategy in ["cc_cv", "mfcc", "ours"]:
            part = pack[pack["strategy"] == strategy].sort_values("time_s")
            if part.empty:
                continue
            if field == "current_A":
                y = (
                    traj[(traj["episode"] == episode) & (traj["strategy"] == strategy)]
                    .groupby("time_s")["current_A"]
                    .mean()
                    .to_numpy(dtype=float)
                )
                x = (
                    traj[(traj["episode"] == episode) & (traj["strategy"] == strategy)]
                    .groupby("time_s")["current_A"]
                    .mean()
                    .index.to_numpy(dtype=float)
                )
            else:
                x = part["time_s"].to_numpy(dtype=float)
                y = part[field].to_numpy(dtype=float)
                if field in {"soc_min", "soc_spread"}:
                    y = y * 100.0
            ax.plot(x, y, color=colors[strategy], lw=1.5, label=names[strategy])
        if field == "soc_min":
            ax.axhline(80, color="#303030", linestyle=":", lw=0.8)
            label_y = {"ours": 82.0, "cc_cv": 86.0}
            for strategy in ["ours", "cc_cv"]:
                row = metrics[(metrics["episode"] == episode) & (metrics["strategy"] == strategy)]
                if not row.empty and np.isfinite(float(row["time_to_target_s"].iloc[0])):
                    t_hit = float(row["time_to_target_s"].iloc[0])
                    ax.axvline(t_hit, color=colors[strategy], linestyle="--", lw=0.8, alpha=0.75)
                    ax.text(t_hit, label_y[strategy], f"{names[strategy]} {t_hit:.0f}s", color=colors[strategy], fontsize=7, ha="center")
            ax.set_ylim(16, 90)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.50, 0.925))
    fig.suptitle("6S1P policy replication with SOC-spread balancing, paired episode", y=0.99, fontsize=11, weight="bold")
    fig.text(0.52, 0.02, "Pack-level stopping uses the weakest cell; SAC reaches the min-cell target earlier and lowers final spread.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), w_pad=1.2, h_pad=1.4)
    save(fig, "fig10_pack_comparison_polished")
    plt.close(fig)


def update_manifest() -> None:
    """Refresh the paper figure manifest with all figure PNG files."""
    manifest = {
        "generated": sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("fig*.png")),
        "note": "Figures generated from local CRAIC2026 artifacts; third-party reference images are not included.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Generate polished summary figures."""
    set_style()
    fig_graphical_abstract()
    fig_results_dashboard()
    fig_single_cell_trajectory_polished()
    fig_pack_trajectory_polished()
    update_manifest()
    print(json.dumps(json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
