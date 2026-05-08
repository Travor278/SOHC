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
    update_manifest()
    print(json.dumps(json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
