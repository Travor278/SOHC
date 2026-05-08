"""Paper-style UPC pack evaluation and pure-Python balancing demo.

The figures generated here use the trusted UPC 36-cell pack dataset for the
measured pack evidence, plus a lightweight active-balancing digital twin
initialized from a real high-spread UPC sample. It is not a switching-level
Simulink replacement; it provides the reproducible pack-level result figures.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from craic_pipeline.pack_dataset_upc import UPCPackCycle, iter_upc_pack_cycles, load_upc_pack_cycle, summarize_upc_pack_cycle


@dataclass
class BalancerConfig:
    """Parameters for the active voltage-spread balancing digital twin."""

    capacity_Ah: float = 3.2
    voltage_per_soc: float = 1.2
    max_balance_current_A: float = 0.8
    efficiency: float = 0.92
    threshold_mV: float = 10.0
    dt_s: float = 1.0
    duration_s: float = 1800.0


def find_representative_cycles(data_dir: Path) -> tuple[Path, Path]:
    """Pick one WLTP profile and one balancing cycle from UPC files."""
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no UPC parquet files found in {data_dir}")
    wltp = next((path for path in files if "WLTP" in path.name), files[0])
    balancing = None
    for path in files:
        cycle = load_upc_pack_cycle(path, downsample=100)
        if any("Balancing" in str(item) for item in cycle.semicycle):
            balancing = path
            break
    return wltp, balancing or wltp


def simulate_voltage_balancer(initial_voltage_V: np.ndarray, cfg: BalancerConfig) -> pd.DataFrame:
    """Simulate active buck-boost-like equalization from real UPC cell voltages.

    Args:
        initial_voltage_V: Initial cell voltages `(36,)` from a UPC sample.
        cfg: Balancing capacity, current, efficiency, threshold, and duration.

    Returns:
        Long-form DataFrame with per-cell voltage and balance current traces.
    """
    initial = np.asarray(initial_voltage_V, dtype=float).reshape(-1)
    if initial.size == 0:
        raise ValueError("initial_voltage_V cannot be empty")
    n_steps = int(cfg.duration_s / cfg.dt_s) + 1
    active = initial.copy()
    off = initial.copy()
    rows = []
    max_dv = cfg.max_balance_current_A / max(cfg.capacity_Ah * 3600.0, 1e-12) * cfg.voltage_per_soc * cfg.dt_s
    threshold = cfg.threshold_mV / 1000.0
    for step in range(n_steps):
        time_s = step * cfg.dt_s
        for case, values, currents in [
            ("balancing_off", off, np.zeros_like(off)),
            ("active_buck_boost", active, np.zeros_like(active) if step == 0 else last_current),
        ]:
            for cell_id, voltage in enumerate(values):
                rows.append(
                    {
                        "case": case,
                        "time_s": time_s,
                        "cell_id": cell_id,
                        "voltage_V": float(voltage),
                        "balance_current_A": float(currents[cell_id]),
                    }
                )
        target = float(np.nanmean(active))
        desired = target - active
        update = np.zeros_like(active)
        mask = np.abs(desired) > threshold
        update[mask] = np.clip(0.20 * desired[mask], -max_dv, max_dv)
        update[update > 0] *= cfg.efficiency
        active = active + update
        last_current = np.abs(update) / max(cfg.voltage_per_soc, 1e-12) * cfg.capacity_Ah * 3600.0 / cfg.dt_s
    return pd.DataFrame(rows)


def compute_balancing_metrics(sim: pd.DataFrame) -> pd.DataFrame:
    """Compute spread reduction and current metrics from a balancing simulation."""
    rows = []
    for case, group in sim.groupby("case"):
        per_step = group.groupby("time_s").agg(
            v_min=("voltage_V", "min"),
            v_max=("voltage_V", "max"),
            current_mean=("balance_current_A", "mean"),
            current_max=("balance_current_A", "max"),
        )
        spread = (per_step["v_max"] - per_step["v_min"]) * 1000.0
        rows.append(
            {
                "case": case,
                "initial_spread_mV": float(spread.iloc[0]),
                "final_spread_mV": float(spread.iloc[-1]),
                "spread_reduction_pct": 100.0 * (float(spread.iloc[0]) - float(spread.iloc[-1])) / max(float(spread.iloc[0]), 1e-12),
                "mean_balance_current_A": float(per_step["current_mean"].mean()),
                "max_balance_current_A": float(per_step["current_max"].max()),
            }
        )
    return pd.DataFrame(rows)


def measured_profile_metrics(cycle: UPCPackCycle) -> dict:
    """Summarize a measured UPC cycle for the paper results table."""
    summary = summarize_upc_pack_cycle(cycle)
    voltage = cycle.cell_voltage_flat_V
    spread = (np.nanmax(voltage, axis=1) - np.nanmin(voltage, axis=1)) * 1000.0
    return {
        **summary,
        "cell_voltage_spread_p95_mV": float(np.nanpercentile(spread, 95)),
        "branch_current_abs_mean_A": float(np.nanmean(np.abs(cycle.branch_current_A))),
    }


def measured_balancing_metrics(cycle: UPCPackCycle) -> dict:
    """Measure spread behavior during a real UPC balancing semicycle."""
    voltage = cycle.cell_voltage_flat_V
    spread = (np.nanmax(voltage, axis=1) - np.nanmin(voltage, axis=1)) * 1000.0
    mask = np.char.find(cycle.semicycle.astype(str), "Balancing") >= 0
    if not np.any(mask):
        return {
            "file": cycle.source_path.name,
            "balancing_samples": 0,
            "balancing_duration_s": 0.0,
            "balancing_spread_start_mV": float("nan"),
            "balancing_spread_end_mV": float("nan"),
            "balancing_spread_min_mV": float("nan"),
            "balancing_spread_max_mV": float("nan"),
        }
    idx = np.where(mask)[0]
    segment = spread[idx]
    return {
        "file": cycle.source_path.name,
        "balancing_samples": int(len(idx)),
        "balancing_duration_s": float(cycle.time_s[idx[-1]] - cycle.time_s[idx[0]]),
        "balancing_spread_start_mV": float(segment[0]),
        "balancing_spread_end_mV": float(segment[-1]),
        "balancing_spread_min_mV": float(np.nanmin(segment)),
        "balancing_spread_max_mV": float(np.nanmax(segment)),
    }


def plot_topology(out_dir: Path) -> Path:
    """Draw an original active buck-boost balancing topology schematic."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_axis_off()
    blocks = {
        "high": (0.06, 0.58, 0.20, 0.24, "High-SOC cell/module\n(source)"),
        "switch": (0.36, 0.58, 0.18, 0.24, "Bidirectional\nswitch matrix"),
        "converter": (0.64, 0.58, 0.24, 0.24, "Buck-boost stage\nL + MOSFET bridge"),
        "low": (0.64, 0.16, 0.24, 0.22, "Low-SOC cell/module\n(sink)"),
        "ctrl": (0.08, 0.12, 0.34, 0.22, "BMS coordinator\nSOC/V spread -> duty / pair select"),
    }
    for x, y, w, h, text in blocks.values():
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.26, 0.70), (0.36, 0.70), "select high"),
        ((0.54, 0.70), (0.64, 0.70), "buck mode"),
        ((0.76, 0.58), (0.76, 0.38), "energy buffer"),
        ((0.64, 0.27), (0.54, 0.58), "boost mode"),
        ((0.42, 0.23), (0.64, 0.23), "commands"),
    ]
    for start, end, label in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=13, lw=1.4))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035, label, ha="center", fontsize=8)
    ax.text(
        0.5,
        0.02,
        "Original schematic for this paper: active energy transfer from high-voltage/high-SOC cells to low-voltage/low-SOC cells.",
        ha="center",
        fontsize=8,
    )
    path = out_dir / "fig_active_balancer_topology.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_measured_cycle(cycle: UPCPackCycle, out_dir: Path) -> Path:
    """Plot measured UPC cell-voltage envelope, spread, currents, and SOC."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    time_h = cycle.time_s / 3600.0
    voltage = cycle.cell_voltage_flat_V
    spread = (np.nanmax(voltage, axis=1) - np.nanmin(voltage, axis=1)) * 1000.0
    temp = cycle.cell_temperature_mean_C
    valid_temp = np.where((temp > -40.0) & (temp < 120.0), temp, np.nan)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].fill_between(time_h, np.nanmin(voltage, axis=1), np.nanmax(voltage, axis=1), alpha=0.25, label="cell min-max")
    axes[0].plot(time_h, np.nanmean(voltage, axis=1), lw=1.4, label="cell mean")
    axes[0].set_ylabel("Cell voltage (V)")
    axes[1].plot(time_h, spread, color="tab:red", lw=1.2)
    axes[1].set_ylabel("Voltage spread (mV)")
    for idx in range(cycle.branch_current_A.shape[1]):
        axes[2].plot(time_h, cycle.branch_current_A[:, idx], lw=1.0, label=f"P{idx + 1}")
    axes[2].set_ylabel("Branch current (A)")
    axes[2].legend(loc="best", ncols=3)
    axes[3].plot(time_h, cycle.bms_soc, label="BMS SOC")
    axes[3].plot(time_h, np.nanpercentile(valid_temp, 95, axis=1) / 100.0, label="T p95 / 100")
    axes[3].set_ylabel("SOC / scaled T")
    axes[3].set_xlabel("Time (h)")
    axes[3].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"UPC measured pack profile: {cycle.source_path.name}", y=0.995)
    fig.tight_layout()
    path = out_dir / "fig_upc_measured_profile.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_real_balancing_cycle(cycle: UPCPackCycle, out_dir: Path) -> Path:
    """Plot the real UPC cycle segment that contains a balancing semicycle."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    time_h = cycle.time_s / 3600.0
    voltage = cycle.cell_voltage_flat_V
    spread = (np.nanmax(voltage, axis=1) - np.nanmin(voltage, axis=1)) * 1000.0
    mask = np.char.find(cycle.semicycle.astype(str), "Balancing") >= 0
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    axes[0].fill_between(time_h, np.nanmin(voltage, axis=1), np.nanmax(voltage, axis=1), alpha=0.25)
    axes[0].plot(time_h, np.nanmean(voltage, axis=1), lw=1.2)
    axes[0].set_ylabel("Cell voltage (V)")
    axes[1].plot(time_h, spread, color="tab:red", lw=1.2)
    axes[1].set_ylabel("Spread (mV)")
    axes[2].plot(time_h, cycle.bms_soc, label="BMS SOC")
    axes[2].plot(time_h, np.nanmean(cycle.branch_current_A, axis=1), label="mean branch I")
    axes[2].set_ylabel("SOC / I")
    axes[2].set_xlabel("Time (h)")
    axes[2].legend(loc="best")
    if np.any(mask):
        start = time_h[np.where(mask)[0][0]]
        end = time_h[np.where(mask)[0][-1]]
        for ax in axes:
            ax.axvspan(start, end, color="tab:green", alpha=0.12, label="Balancing")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"UPC measured balancing semicycle: {cycle.source_path.name}", y=0.995)
    fig.tight_layout()
    path = out_dir / "fig_upc_real_balancing_semicycle.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_balancing_sim(sim: pd.DataFrame, out_dir: Path) -> Path:
    """Plot active balancing spread reduction initialized from UPC real voltages."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for case, group in sim.groupby("case"):
        per_step = group.groupby("time_s").agg(
            v_min=("voltage_V", "min"),
            v_max=("voltage_V", "max"),
            v_mean=("voltage_V", "mean"),
            i_max=("balance_current_A", "max"),
        )
        time_min = per_step.index.to_numpy() / 60.0
        axes[0].plot(time_min, per_step["v_mean"], label=case)
        axes[0].fill_between(time_min, per_step["v_min"], per_step["v_max"], alpha=0.18)
        axes[1].plot(time_min, (per_step["v_max"] - per_step["v_min"]) * 1000.0, label=case)
        axes[2].plot(time_min, per_step["i_max"], label=case)
    axes[0].set_ylabel("Cell voltage (V)")
    axes[1].set_ylabel("Spread (mV)")
    axes[2].set_ylabel("Max balance I (A)")
    axes[2].set_xlabel("Time (min)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Pure-Python active buck-boost balancing short simulation", y=0.995)
    fig.tight_layout()
    path = out_dir / "fig_python_balancing_short_sim.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _initial_high_spread_voltage(cycle: UPCPackCycle) -> np.ndarray:
    """Return the measured cell-voltage vector at maximum spread."""
    voltage = cycle.cell_voltage_flat_V
    spread = np.nanmax(voltage, axis=1) - np.nanmin(voltage, axis=1)
    return voltage[int(np.nanargmax(spread))].astype(float)


def write_paper_results(
    out_dir: Path,
    *,
    measured_metrics: dict,
    real_balancing_metrics: dict,
    balance_metrics: pd.DataFrame,
    references: list[tuple[str, str]],
) -> Path:
    """Write a compact paper-style results markdown file."""
    active = balance_metrics[balance_metrics["case"] == "active_buck_boost"].iloc[0]
    off = balance_metrics[balance_metrics["case"] == "balancing_off"].iloc[0]
    text = f"""# UPC Pack-Level Results

## Experimental Setting

- Dataset: UPC 36-cell pack WLTP/CC-CV, 12S3P topology.
- Measured profile: `{measured_metrics['file']}`.
- Measured signals: 36 cell voltages, 3 branch currents, BMS SOC, cell temperatures, semicycle labels.
- Python balancing demo: active buck-boost-like equalizer initialized from a real UPC high-spread sample.

## Measured UPC Pack Evidence

| Metric | Value |
|---|---:|
| Samples | {int(measured_metrics['samples'])} |
| Duration | {measured_metrics['duration_s'] / 3600.0:.2f} h |
| Mean cell voltage spread | {measured_metrics['cell_voltage_spread_mean_mV']:.2f} mV |
| P95 cell voltage spread | {measured_metrics['cell_voltage_spread_p95_mV']:.2f} mV |
| Max cell voltage spread | {measured_metrics['cell_voltage_spread_max_mV']:.2f} mV |
| Max branch current | {measured_metrics['branch_current_abs_max_A']:.2f} A |
| Valid temperature p95 | {measured_metrics['temperature_p95_valid_C']:.2f} degC |

## Real UPC Balancing Semicycle

| Metric | Value |
|---|---:|
| File | `{real_balancing_metrics['file']}` |
| Balancing duration | {real_balancing_metrics['balancing_duration_s'] / 3600.0:.2f} h |
| Spread at balancing start | {real_balancing_metrics['balancing_spread_start_mV']:.2f} mV |
| Minimum spread during balancing | {real_balancing_metrics['balancing_spread_min_mV']:.2f} mV |
| Spread at balancing end | {real_balancing_metrics['balancing_spread_end_mV']:.2f} mV |

## Python Active Balancing Short Simulation

| Case | Initial spread | Final spread | Reduction | Max balance current |
|---|---:|---:|---:|---:|
| Balancing off | {off['initial_spread_mV']:.2f} mV | {off['final_spread_mV']:.2f} mV | {off['spread_reduction_pct']:.2f}% | {off['max_balance_current_A']:.2f} A |
| Active buck-boost | {active['initial_spread_mV']:.2f} mV | {active['final_spread_mV']:.2f} mV | {active['spread_reduction_pct']:.2f}% | {active['max_balance_current_A']:.2f} A |

## Figures

- `fig_active_balancer_topology.png`: original active buck-boost balancing topology schematic.
- `fig_upc_measured_profile.png`: real UPC measured pack profile.
- `fig_upc_real_balancing_semicycle.png`: real UPC balancing semicycle highlighted.
- `fig_python_balancing_short_sim.png`: pure-Python balancing short simulation.

## Reference Sources

"""
    for label, url in references:
        text += f"- {label}: {url}\n"
    path = out_dir / "paper_results.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    """Generate UPC paper-style pack results and figures."""
    parser = argparse.ArgumentParser(description="Generate UPC pack paper figures")
    parser.add_argument("--data-dir", type=Path, default=Path("data/pack_wltp_upc"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/upc_pack_paper"))
    parser.add_argument("--measured-cycle", type=Path, default=None)
    parser.add_argument("--balancing-cycle", type=Path, default=None)
    parser.add_argument("--downsample", type=int, default=20)
    parser.add_argument("--balance-duration-s", type=float, default=1800.0)
    args = parser.parse_args()

    wltp_path, balancing_path = find_representative_cycles(args.data_dir)
    measured_path = args.measured_cycle or wltp_path
    balancing_path = args.balancing_cycle or balancing_path
    measured = load_upc_pack_cycle(measured_path, downsample=args.downsample)
    balancing_cycle = load_upc_pack_cycle(balancing_path, downsample=max(args.downsample, 10))
    cfg = BalancerConfig(duration_s=args.balance_duration_s)
    sim = simulate_voltage_balancer(_initial_high_spread_voltage(balancing_cycle), cfg)
    metrics = measured_profile_metrics(measured)
    real_balance = measured_balancing_metrics(balancing_cycle)
    balance_metrics = compute_balancing_metrics(sim)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.out_dir / "upc_measured_metrics.csv", index=False)
    pd.DataFrame([real_balance]).to_csv(args.out_dir / "upc_real_balancing_metrics.csv", index=False)
    balance_metrics.to_csv(args.out_dir / "python_balancing_metrics.csv", index=False)
    sim.to_csv(args.out_dir / "python_balancing_trace.csv", index=False)
    plot_topology(args.out_dir)
    plot_measured_cycle(measured, args.out_dir)
    plot_real_balancing_cycle(balancing_cycle, args.out_dir)
    plot_balancing_sim(sim, args.out_dir)
    write_paper_results(
        args.out_dir,
        measured_metrics=metrics,
        real_balancing_metrics=real_balance,
        balance_metrics=balance_metrics,
        references=[
            ("MathWorks Simscape Battery cell-balancing examples", "https://www.mathworks.com/help/simscape-battery/cell-balancing.html"),
            ("MathWorks Battery Pack Cell Balancing example", "https://www.mathworks.com/help/sps/ug/lithium-pack-cell-balancing.html"),
            (
                "Open-source active balancing Simulink reference",
                "https://github.com/yavuzhanocak/Single-switch-capacitor-battery-balance",
            ),
            (
                "UPC 36-cell pack dataset",
                "https://doi.org/10.34810/DATA2395",
            ),
        ],
    )
    print(f"measured_cycle={measured_path}")
    print(f"balancing_cycle={balancing_path}")
    print(balance_metrics.to_string(index=False))
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
