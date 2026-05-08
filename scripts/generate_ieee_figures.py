"""Generate IEEE-style figures for the CRAIC2026 report draft.

The script writes publication-oriented PNG/SVG figures into `paper_figures/`.
Most figures use CSV/JSON artifacts already produced by W1-W5. The world-model
rollout figure requires a Mamba-capable environment because `outputs/world_model.pt`
was trained with `mamba-ssm`.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("paper_figures")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def set_ieee_style() -> None:
    """Configure matplotlib for compact IEEE-like report figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "patch.linewidth": 0.8,
            "grid.linewidth": 0.35,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, out_base: Path) -> None:
    """Save one figure as PNG and SVG."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")


def draw_box(ax, xy, wh, text: str, *, fc: str = "#F7F7F7", ec: str = "#303030") -> None:
    """Draw a labeled rounded rectangle in axes coordinates."""
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.15)


def draw_arrow(ax, start, end, *, color: str = "#303030") -> None:
    """Draw a slim arrow in axes coordinates."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=color, shrinkA=2, shrinkB=2),
    )


def fig_system_architecture(out_dir: Path) -> None:
    """Create the main three-layer architecture diagram."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    colors = {
        "data": "#E8EEF7",
        "est": "#E9F5EE",
        "wm": "#FFF4E4",
        "rl": "#F3ECF8",
        "safe": "#FCEBEB",
        "out": "#F7F7F7",
    }
    draw_box(ax, (0.05, 0.76), (0.18, 0.14), "NASA PCoE\nB0005-18 / ARC\nRandomized", fc=colors["data"])
    draw_box(ax, (0.30, 0.76), (0.18, 0.14), "W1 State\nEstimator\nSOC / SOH", fc=colors["est"])
    draw_box(ax, (0.55, 0.76), (0.20, 0.14), "W2 Mamba\nWorld Model\nnext-state + aging", fc=colors["wm"])
    draw_box(ax, (0.55, 0.50), (0.20, 0.14), "W3 SAC Policy\ncontinuous\ncharging current", fc=colors["rl"])
    draw_box(ax, (0.30, 0.50), (0.18, 0.14), "L3 ECM Safety\nvoltage-constrained\naction projection", fc=colors["safe"])
    draw_box(ax, (0.05, 0.50), (0.18, 0.14), "Safe Charging\ntrajectory\nI / V / SOC / T", fc=colors["out"])
    draw_box(ax, (0.30, 0.18), (0.45, 0.16), "Pack Extension\nsingle-cell policy replication + SOC-spread active balancing\nUPC 36-cell pack validation / Simulink interface", fc="#EEF2F2")

    draw_arrow(ax, (0.23, 0.83), (0.30, 0.83))
    draw_arrow(ax, (0.48, 0.83), (0.55, 0.83))
    draw_arrow(ax, (0.65, 0.76), (0.65, 0.64))
    draw_arrow(ax, (0.55, 0.57), (0.48, 0.57))
    draw_arrow(ax, (0.30, 0.57), (0.23, 0.57))
    draw_arrow(ax, (0.14, 0.50), (0.30, 0.30))
    draw_arrow(ax, (0.65, 0.50), (0.58, 0.34))
    draw_arrow(ax, (0.40, 0.50), (0.43, 0.34))

    ax.text(0.65, 0.69, "state-action history", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(0.40, 0.69, "[SOC, SOH, V, I, T]", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(0.39, 0.43, "safe current", ha="center", va="center", fontsize=7, color="#555555")
    ax.set_title("EV Fast-Charging Decision Framework", pad=6)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig01_system_architecture")
    plt.close(fig)


def fig_data_flow(out_dir: Path) -> None:
    """Create a dataset provenance and usage diagram."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_box(ax, (0.04, 0.70), (0.20, 0.16), "LG 18650HG2\nKeiLongW weights", fc="#F2F2F2")
    draw_box(ax, (0.04, 0.46), (0.20, 0.16), "NASA ARC-FY08Q4\nmulti-temp/rate", fc="#E9F5EE")
    draw_box(ax, (0.04, 0.22), (0.20, 0.16), "NASA B0005-18\naging cycles", fc="#E8EEF7")
    draw_box(ax, (0.04, 0.02), (0.20, 0.14), "NASA Randomized\nstable dynamic load", fc="#FFF4E4")

    draw_box(ax, (0.36, 0.58), (0.20, 0.16), "SOC LSTM\nwarm-start +\nNASA fine-tune", fc="#E9F5EE")
    draw_box(ax, (0.36, 0.30), (0.20, 0.16), "SOH baseline\ncapacity ratio /\nBatteryML API", fc="#E8EEF7")
    draw_box(ax, (0.65, 0.44), (0.24, 0.18), "World-model tensor\nX=(N,L,6)\ny=(N,4)", fc="#FFF4E4")
    draw_box(ax, (0.65, 0.14), (0.24, 0.16), "RL evaluation\nCC-CV / MFCC / SAC", fc="#F3ECF8")
    draw_box(ax, (0.65, 0.74), (0.24, 0.14), "UPC 36-cell pack\nW5 validation", fc="#EEF2F2")
    draw_box(ax, (0.36, 0.02), (0.53, 0.10), "Zenodo 6985321 / 18471156\nzero-shot diagnostic and station qualitative display only", fc="#F7F7F7")

    draw_arrow(ax, (0.24, 0.78), (0.36, 0.68))
    draw_arrow(ax, (0.24, 0.54), (0.36, 0.66))
    draw_arrow(ax, (0.24, 0.30), (0.36, 0.38))
    draw_arrow(ax, (0.56, 0.66), (0.65, 0.54))
    draw_arrow(ax, (0.56, 0.38), (0.65, 0.52))
    draw_arrow(ax, (0.24, 0.09), (0.65, 0.48))
    draw_arrow(ax, (0.77, 0.44), (0.77, 0.30))
    draw_arrow(ax, (0.77, 0.74), (0.77, 0.62))

    ax.set_title("Dataset Provenance and Usage", pad=6)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig02_data_flow")
    plt.close(fig)


def fig_world_model_rollout(out_dir: Path, *, device: str = "auto") -> None:
    """Plot an open-loop Mamba rollout on a B0018 holdout trace."""
    import matplotlib.pyplot as plt
    import torch

    from craic_pipeline.world_model_mamba import load_world_model_checkpoint

    bundle = torch.load("outputs/world_model_train_data.pt", map_location="cpu", weights_only=False)
    model, _ = load_world_model_checkpoint(Path("outputs/world_model.pt"))
    use_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    if device == "auto" and not torch.cuda.is_available():
        use_device = torch.device("cpu")
    model = model.to(use_device)
    model.eval()
    seq_len = int(bundle.get("meta", {}).get("seq_len", 64))
    traces = [
        t
        for t in bundle.get("traces", [])
        if str(t.get("cell", "")).upper() == "B0018" and len(t.get("features", [])) > seq_len + 180
    ]
    if not traces:
        raise RuntimeError("No B0018 trace long enough for world-model rollout figure")
    trace = max(traces, key=lambda item: len(item["features"]))
    features = np.asarray(trace["features"], dtype=np.float32)
    horizon = min(20, len(features) - seq_len - 1)
    start = max(0, (len(features) - seq_len - horizon) // 2)
    hist = torch.from_numpy(features[start : start + seq_len].copy()).to(use_device).unsqueeze(0)
    pred_soh = float(hist[0, -1, 1].detach().cpu())
    pred_v, true_v, true_soc, pred_soc, err = [], [], [], [], []
    with torch.no_grad():
        for step in range(horizon):
            pred = model(hist).squeeze(0).detach()
            actual = features[start + seq_len + step]
            pv = float(pred[1].detach().cpu())
            ps = float(pred[0].detach().cpu())
            tv = float(actual[2])
            pred_v.append(pv)
            pred_soc.append(ps)
            true_v.append(tv)
            true_soc.append(float(actual[0]))
            err.append((pv - tv) * 1000.0)
            pred_soh = float(np.clip(pred_soh - max(float(pred[3].detach().cpu()), 0.0), 0.0, 1.2))
            next_feature = torch.tensor(
                [
                    [
                        float(pred[0].clamp(0.0, 1.0)),
                        pred_soh,
                        pv,
                        float(actual[3]),
                        float(pred[2].detach().cpu()),
                        float(actual[5]),
                    ]
                ],
                dtype=hist.dtype,
                device=use_device,
            )
            hist = torch.cat([hist[:, 1:], next_feature.unsqueeze(0)], dim=1)

    t = np.arange(horizon)
    mae = float(np.mean(np.abs(err)))
    pd.DataFrame(
        {
            "step": t,
            "voltage_true_V": true_v,
            "voltage_pred_V": pred_v,
            "voltage_error_mV": err,
            "soc_true": true_soc,
            "soc_pred": pred_soc,
        }
    ).to_csv(out_dir / "fig03_world_model_rollout.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.0), sharex=True)
    axes[0].plot(t, true_v, color="#111111", label="Measured")
    axes[0].plot(t, pred_v, color="#1F77B4", linestyle="--", label="Mamba rollout")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(True)
    axes[0].legend(frameon=False, loc="best")
    axes[0].set_title(f"B0018 20-step open-loop rollout, MAE={mae:.2f} mV")
    axes[1].plot(t, err, color="#A23B3B")
    axes[1].axhline(0, color="#303030", linewidth=0.7)
    axes[1].set_ylabel("Error (mV)")
    axes[1].set_xlabel("Rollout step")
    axes[1].grid(True)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig03_world_model_rollout")
    plt.close(fig)


def fig_w4_metrics(out_dir: Path) -> None:
    """Plot paired W4 speed and aging improvements."""
    import matplotlib.pyplot as plt

    paired = pd.read_csv("outputs/eval_w4_final_default/paired_vs_cc_cv.csv").iloc[0]
    labels = ["CC-CV", "SAC"]
    time_vals = [paired["cc_cv_time_to_80_s"], paired["strategy_time_to_80_s"]]
    soh_vals = [paired["cc_cv_delta_soh"] * 1000.0, paired["strategy_delta_soh"] * 1000.0]
    colors = ["#6B6B6B", "#1F77B4"]
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 1.85))
    axes[0].bar(labels, time_vals, color=colors, width=0.62)
    axes[0].set_ylabel("Time to 80% SOC (s)")
    axes[0].set_title(f"+{paired['speed_improvement_pct']:.1f}% faster")
    axes[0].grid(axis="y")
    axes[1].bar(labels, soh_vals, color=colors, width=0.62)
    axes[1].set_ylabel(r"$\Delta$SOH ($\times10^{-3}$)")
    axes[1].set_title(f"{paired['delta_soh_reduction_pct']:.1f}% lower aging")
    axes[1].grid(axis="y")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, out_dir / "fig04_w4_metrics_bar")
    plt.close(fig)


def fig_ecm_projection(out_dir: Path) -> None:
    """Plot ECM action projection at high SOC."""
    import matplotlib.pyplot as plt

    from craic_pipeline.ecm_safety_layer import ECMSafetyLayer, load_params_from_mat

    params = load_params_from_mat()
    soc = 0.95
    requested = np.linspace(0.0, 8.0, 161)
    raw_v, safe_v, safe_i = [], [], []
    for req in requested:
        raw_layer = ECMSafetyLayer(params, dt=1.0)
        raw_v.append(raw_layer.predict_voltage(soc, -float(req)))
        project_layer = ECMSafetyLayer(params, dt=1.0)
        projected = -project_layer.project(soc, -float(req))
        safe_i.append(projected)
        check_layer = ECMSafetyLayer(params, dt=1.0)
        safe_v.append(check_layer.predict_voltage(soc, -projected))
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.0), sharex=True)
    axes[0].plot(requested, raw_v, color="#A23B3B", label="Before projection")
    axes[0].plot(requested, safe_v, color="#1F77B4", label="After ECM projection")
    axes[0].axhline(params.V_max, color="#303030", linestyle=":", label=r"$V_{max}$")
    axes[0].set_ylabel("Predicted V (V)")
    axes[0].legend(frameon=False, loc="best")
    axes[0].grid(True)
    axes[1].plot(requested, requested, color="#A23B3B", linestyle=":", label="Requested")
    axes[1].plot(requested, safe_i, color="#1F77B4", label="Safe")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_xlabel("Requested charging current (A)")
    axes[1].grid(True)
    axes[1].legend(frameon=False, loc="best")
    axes[0].set_title("ECM safety projection at SOC=95%")
    fig.tight_layout()
    save_figure(fig, out_dir / "fig05_ecm_safety_projection")
    plt.close(fig)


def fig_sac_training(out_dir: Path) -> None:
    """Plot SAC rollout reward and critic loss from TensorBoard events."""
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator

    event_files = sorted(Path("outputs/runs/w3_horizon600/tb/SAC_1").glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError("No TensorBoard event file found for W3 horizon600")
    ea = event_accumulator.EventAccumulator(str(event_files[-1]))
    ea.Reload()

    def scalar(tag: str) -> tuple[np.ndarray, np.ndarray]:
        vals = ea.Scalars(tag)
        return np.asarray([v.step for v in vals], dtype=float), np.asarray([v.value for v in vals], dtype=float)

    steps_r, rewards = scalar("rollout/ep_rew_mean")
    steps_c, critic = scalar("train/critic_loss")
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 3.0), sharex=False)
    axes[0].plot(steps_r / 1000.0, rewards, color="#1F77B4")
    axes[0].set_ylabel("Episode return")
    axes[0].set_title("SAC training diagnostics")
    axes[0].grid(True)
    axes[1].semilogy(steps_c / 1000.0, critic, color="#6B6B6B")
    axes[1].set_ylabel("Critic loss")
    axes[1].set_xlabel("Training steps (k)")
    axes[1].grid(True, which="both")
    fig.tight_layout()
    save_figure(fig, out_dir / "fig06_sac_training_curve")
    plt.close(fig)


def fig_soh_baseline(out_dir: Path) -> None:
    """Plot SOH baseline predictions on NASA validation cells."""
    import matplotlib.pyplot as plt

    from craic_pipeline.soh_train import (
        _cell_nominal_capacity,
        _cycle_features,
        _cycle_soh,
        load_nasa_for_batteryml,
        split_by_cell_id,
    )

    with open("outputs/soh_baseline.pt", "rb") as fin:
        payload = pickle.load(fin)
    model = payload["model"]
    cells = load_nasa_for_batteryml(Path("data/nasa_pcoe"))
    _, val_cells = split_by_cell_id(cells, val_ratio=0.2)
    rows = []
    for cell in val_cells:
        nominal = _cell_nominal_capacity(cell)
        for cycle in cell.cycle_data:
            feature = _cycle_features(cycle, nominal)
            target = _cycle_soh(cycle)
            if feature is None or not np.isfinite(target):
                continue
            pred = float(np.clip(model.predict(np.asarray([feature], dtype=float))[0], 0.0, 1.2))
            rows.append({"cell": cell.cell_id, "cycle": cycle.cycle_number, "soh_true": target, "soh_pred": pred})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "fig07_soh_predictions.csv", index=False)
    if frame.empty:
        raise RuntimeError("No SOH validation predictions available")
    rep_cell = frame["cell"].iloc[-1]
    rep = frame[frame["cell"] == rep_cell]
    rmse = float(np.sqrt(np.mean((frame["soh_true"] - frame["soh_pred"]) ** 2)) * 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35))
    axes[0].scatter(frame["soh_true"] * 100.0, frame["soh_pred"] * 100.0, s=8, alpha=0.65, color="#1F77B4")
    lo = min(frame["soh_true"].min(), frame["soh_pred"].min()) * 100.0 - 1
    hi = max(frame["soh_true"].max(), frame["soh_pred"].max()) * 100.0 + 1
    axes[0].plot([lo, hi], [lo, hi], color="#303030", linestyle=":")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("Reference SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")
    axes[0].set_title(f"Validation RMSE={rmse:.3g}%")
    axes[0].grid(True)
    axes[1].plot(rep["cycle"], rep["soh_true"] * 100.0, color="#111111", label="Reference")
    axes[1].plot(rep["cycle"], rep["soh_pred"] * 100.0, color="#1F77B4", linestyle="--", label="Predicted")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("SOH (%)")
    axes[1].set_title(f"Representative cell {rep_cell}")
    axes[1].grid(True)
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, out_dir / "fig07_soh_baseline")
    plt.close(fig)


def copy_existing_figures(out_dir: Path) -> None:
    """Collect existing result figures into the paper figure directory."""
    mapping = {
        "outputs/figures/soc_b0018_prediction.png": "fig08_soc_b0018_prediction.png",
        "outputs/eval_w4_final_default/charging_comparison.png": "fig09_charging_comparison.png",
        "outputs/eval_pack_6s1p_h1200/pack_comparison.png": "fig10_pack_comparison.png",
        "outputs/upc_pack_paper/fig_active_balancer_topology.png": "fig11_active_balancer_topology.png",
        "outputs/upc_pack_paper/fig_upc_measured_profile.png": "fig12_upc_measured_profile.png",
        "outputs/upc_pack_paper/fig_upc_real_balancing_semicycle.png": "fig13_upc_real_balancing_semicycle.png",
        "outputs/upc_pack_paper/fig_python_balancing_short_sim.png": "fig14_python_balancing_short_sim.png",
    }
    for src, dst in mapping.items():
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, out_dir / dst)


def fig_simulink_pack_workflow(out_dir: Path) -> None:
    """Create a workflow diagram for using existing Simulink pack assets."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_box(ax, (0.04, 0.70), (0.18, 0.16), "Python pack\nsimulator\n6S1P / 30S1P", fc="#E8EEF7")
    draw_box(ax, (0.29, 0.70), (0.18, 0.16), "Policy export\nper-cell I(t)\nSOC/V/T init", fc="#E9F5EE")
    draw_box(ax, (0.54, 0.70), (0.18, 0.16), "Simulink pack\nplant\n30 modules", fc="#FFF4E4")
    draw_box(ax, (0.78, 0.70), (0.18, 0.16), "Buck-boost\nbalancer\non/off paired", fc="#FCEBEB")

    draw_box(ax, (0.04, 0.36), (0.20, 0.16), "Input files\npack_to_simulink.csv\nparams.mat", fc="#F7F7F7")
    draw_box(ax, (0.31, 0.36), (0.20, 0.16), "Signal builder\ncurrent source\ninitial states", fc="#F7F7F7")
    draw_box(ax, (0.58, 0.36), (0.20, 0.16), "Logged outputs\ncell V/I/T/SOC\nspread", fc="#F7F7F7")
    draw_box(ax, (0.76, 0.10), (0.20, 0.14), "Paper metrics\ntime-to-80\nspread / safety", fc="#EEF2F2")

    draw_arrow(ax, (0.22, 0.78), (0.29, 0.78))
    draw_arrow(ax, (0.47, 0.78), (0.54, 0.78))
    draw_arrow(ax, (0.72, 0.78), (0.78, 0.78))
    draw_arrow(ax, (0.13, 0.70), (0.14, 0.52))
    draw_arrow(ax, (0.24, 0.44), (0.31, 0.44))
    draw_arrow(ax, (0.51, 0.44), (0.58, 0.44))
    draw_arrow(ax, (0.68, 0.70), (0.68, 0.52))
    draw_arrow(ax, (0.84, 0.70), (0.84, 0.24))
    draw_arrow(ax, (0.78, 0.44), (0.84, 0.24))

    ax.text(0.38, 0.61, "CSV/MAT bridge", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(0.68, 0.61, "same initial condition", ha="center", va="center", fontsize=7, color="#555555")
    ax.text(0.52, 0.25, "validation boundary: qualitative/electrical smoke test,\nnot used to replace UPC quantitative pack results", ha="center", va="center", fontsize=7, color="#555555")
    ax.set_title("Simulink Pack-Validation Workflow", pad=6)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig17_simulink_pack_workflow")
    plt.close(fig)


def main() -> None:
    """Generate all currently available report figures."""
    parser = argparse.ArgumentParser(description="Generate IEEE-style CRAIC2026 report figures")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-world", action="store_true", help="Skip Mamba checkpoint rollout figure")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_ieee_style()

    fig_system_architecture(args.out_dir)
    fig_data_flow(args.out_dir)
    if not args.skip_world:
        fig_world_model_rollout(args.out_dir, device=args.device)
    fig_w4_metrics(args.out_dir)
    fig_ecm_projection(args.out_dir)
    fig_sac_training(args.out_dir)
    fig_soh_baseline(args.out_dir)
    copy_existing_figures(args.out_dir)
    fig_simulink_pack_workflow(args.out_dir)

    manifest = {
        "generated": sorted(path.relative_to(args.out_dir).as_posix() for path in args.out_dir.rglob("fig*.png")),
        "note": "Figures generated from local CRAIC2026 artifacts; third-party reference images are not included.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
