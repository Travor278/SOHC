"""Pack-level strategy replication and SOC-spread balancing simulation.

This module keeps the W5 pack prototype close to the W3/W4 single-cell loop:
each cell runs the same world-model + ECM environment, a single-cell policy or
baseline proposes a current per cell, and a lightweight coordinator adds active
balancing trims before stepping the cells.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from craic_pipeline.eval_compare import DEFAULT_ECM_PARAMS, _current_to_action, _parse_mfcc_stages


@dataclass
class PackConfig:
    """Configuration for a series/parallel pack assembled from W3 cell envs."""

    n_series: int = 6
    n_parallel: int = 1
    max_steps: int = 800
    dt: float = 1.0
    soc_target: float = 0.8
    i_max_amps: float = 5.0
    pack_current_limit_A: float | None = None
    balance_gain_A_per_soc: float = 8.0
    max_balance_current_A: float = 0.8
    balance_tolerance_soc: float = 0.015
    initial_soc_low: float = 0.16
    initial_soc_high: float = 0.34
    soc_spread_std: float = 0.035
    soh_spread_std: float = 0.025
    temp_spread_std: float = 2.0
    initial_soh_mean: float = 0.96
    initial_temp_C: float = 25.0
    initial_voltage: float = 3.7

    def __post_init__(self) -> None:
        """Validate pack dimensions and derive the charger current limit."""
        if self.n_series < 1 or self.n_parallel < 1:
            raise ValueError("n_series and n_parallel must both be positive")
        if self.pack_current_limit_A is None:
            self.pack_current_limit_A = self.i_max_amps * self.n_parallel

    @property
    def n_cells(self) -> int:
        """Return total cell count in the simulated pack."""
        return int(self.n_series * self.n_parallel)


def build_pack_envs(
    world_model_path: Path,
    ecm_params_path: Path = DEFAULT_ECM_PARAMS,
    *,
    cfg: PackConfig,
    device: str = "auto",
):
    """Create one W3 cell environment per pack cell from W2/W3 artifacts."""
    from craic_pipeline.ecm_safety_layer import ECMSafetyLayer, load_params_from_mat
    from craic_pipeline.rl_env import BatteryChargingEnv, EnvConfig
    from craic_pipeline.train_sac import _resolve_device
    from craic_pipeline.world_model_mamba import load_world_model_checkpoint

    world_model, metrics = load_world_model_checkpoint(Path(world_model_path))
    params = load_params_from_mat(Path(ecm_params_path))
    device_name = _resolve_device(device)
    envs = []
    for _ in range(cfg.n_cells):
        cell_cfg = EnvConfig(
            max_steps=cfg.max_steps,
            dt=cfg.dt,
            soc_target=cfg.soc_target,
            V_min=params.V_min,
            V_max=params.V_max,
            I_max_amps=cfg.i_max_amps,
            initial_soc_low=cfg.initial_soc_low,
            initial_soc_high=cfg.initial_soc_high,
            initial_soh_low=max(cfg.initial_soh_mean - 3.0 * cfg.soh_spread_std, 0.5),
            initial_soh_high=min(cfg.initial_soh_mean + 3.0 * cfg.soh_spread_std, 1.0),
            initial_voltage=cfg.initial_voltage,
            seq_len=int(getattr(world_model.cfg, "seq_len", 64)),
            device=device_name,
        )
        env = BatteryChargingEnv(world_model, ECMSafetyLayer(params, dt=cfg.dt), cell_cfg)
        env.metadata["world_model_metrics"] = metrics
        envs.append(env)
    return envs


def initial_pack_states(cfg: PackConfig, seed: int | None = None) -> np.ndarray:
    """Sample reproducible per-cell `[SOC, SOH, V, I, T]` initial states."""
    rng = np.random.default_rng(seed)
    base_soc = float(rng.uniform(cfg.initial_soc_low, cfg.initial_soc_high))
    if cfg.n_cells == 1:
        offsets = np.array([0.0], dtype=float)
    else:
        offsets = np.linspace(-1.0, 1.0, cfg.n_cells) * cfg.soc_spread_std
        offsets += rng.normal(0.0, cfg.soc_spread_std * 0.25, cfg.n_cells)
    soc = np.clip(base_soc + offsets, 0.02, 0.95)
    soh = np.clip(rng.normal(cfg.initial_soh_mean, cfg.soh_spread_std, cfg.n_cells), 0.75, 1.0)
    temp = np.clip(rng.normal(cfg.initial_temp_C, cfg.temp_spread_std, cfg.n_cells), -10.0, 60.0)
    voltage = np.clip(cfg.initial_voltage + 0.7 * (soc - 0.25), 2.5, 4.2)
    current = np.zeros(cfg.n_cells, dtype=float)
    return np.stack([soc, soh, voltage, current, temp], axis=1).astype(np.float32)


class PackChargingSimulator:
    """Roll out replicated cell controllers plus optional balancing trims."""

    def __init__(self, cell_envs: Sequence, cfg: PackConfig):
        """Create a simulator from already-built W3 cell environments."""
        if len(cell_envs) != cfg.n_cells:
            raise ValueError(f"expected {cfg.n_cells} cell envs, got {len(cell_envs)}")
        self.cell_envs = list(cell_envs)
        self.cfg = cfg

    def rollout(
        self,
        controller: Callable[[np.ndarray, int], np.ndarray],
        *,
        strategy: str,
        seed: int | None = None,
        balance: bool = True,
    ) -> pd.DataFrame:
        """Run one pack episode and return per-cell trajectory rows."""
        states = initial_pack_states(self.cfg, seed=seed)
        for env, state in zip(self.cell_envs, states):
            env.reset(seed=seed)
            env.safety.reset()
            env.state = state.astype(np.float32)
            env._reset_history(env.state)

        rows: list[dict] = []
        for step in range(self.cfg.max_steps):
            obs = np.stack([env.state.copy() for env in self.cell_envs], axis=0)
            requested = np.asarray(controller(obs.copy(), step), dtype=float).reshape(-1)
            if requested.size != self.cfg.n_cells:
                raise ValueError(f"controller returned {requested.size} currents for {self.cfg.n_cells} cells")
            requested = np.clip(requested, 0.0, self.cfg.i_max_amps)
            balanced = apply_soc_balancer(requested, obs[:, 0], self.cfg) if balance else requested.copy()
            balance_current = balanced - requested

            next_infos = []
            rewards = []
            for cell_id, (env, current) in enumerate(zip(self.cell_envs, balanced)):
                state_before = env.state.copy()
                action = _current_to_action(float(current), env.cfg.I_max_amps)
                _, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
                rewards.append(float(reward))
                next_infos.append((cell_id, state_before, info, terminated, truncated))

            pack_snapshot = _pack_snapshot([env.state for env in self.cell_envs], self.cfg)
            for cell_id, state_before, info, terminated, truncated in next_infos:
                module_id = cell_id // self.cfg.n_parallel
                rows.append(
                    {
                        "strategy": strategy,
                        "step": step + 1,
                        "time_s": (step + 1) * self.cfg.dt,
                        "module_id": module_id,
                        "cell_id": cell_id,
                        "soc_before": float(state_before[0]),
                        "soh_before": float(state_before[1]),
                        "voltage_before": float(state_before[2]),
                        "temperature_before": float(state_before[4]),
                        "soc": float(info["soc"]),
                        "soh": float(info["soh"]),
                        "voltage": float(info["voltage"]),
                        "temperature": float(info["temperature"]),
                        "current_A": float(info["safe_current"]),
                        "requested_current_A": float(requested[cell_id]),
                        "balance_current_A": float(balance_current[cell_id]),
                        "world_model_voltage_raw": float(info.get("world_model_voltage_raw", info["voltage"])),
                        "model_delta_soh": float(info.get("model_delta_soh", 0.0)),
                        "aging_proxy_delta_soh": float(info.get("aging_proxy_delta_soh", 0.0)),
                        "reward": float(rewards[cell_id]),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        **pack_snapshot,
                    }
                )

            if pack_snapshot["soc_min"] >= self.cfg.soc_target:
                break
            if pack_snapshot["voltage_max"] > self.cell_envs[0].cfg.V_max + 1e-6:
                break
            if pack_snapshot["temperature_max"] > self.cell_envs[0].cfg.T_max:
                break

        return pd.DataFrame(rows)


def apply_soc_balancer(currents_A: np.ndarray, soc: np.ndarray, cfg: PackConfig) -> np.ndarray:
    """Add active-balancing current trims from SOC spread and enforce limits."""
    currents = np.asarray(currents_A, dtype=float).copy()
    soc = np.asarray(soc, dtype=float)
    trims = cfg.balance_gain_A_per_soc * (float(np.mean(soc)) - soc)
    trims = np.clip(trims, -cfg.max_balance_current_A, cfg.max_balance_current_A)
    balanced = np.clip(currents + trims, 0.0, cfg.i_max_amps)
    mean_current = float(np.mean(balanced))
    if mean_current > float(cfg.pack_current_limit_A):
        balanced *= float(cfg.pack_current_limit_A) / max(mean_current, 1e-12)
    return np.clip(balanced, 0.0, cfg.i_max_amps)


def make_cc_cv_controller(current_A: float) -> Callable[[np.ndarray, int], np.ndarray]:
    """Create a replicated CC-CV-like controller returning per-cell currents."""

    def controller(obs: np.ndarray, _step: int) -> np.ndarray:
        current = np.full(obs.shape[0], float(current_A), dtype=float)
        current[obs[:, 0] >= 0.8] = 0.0
        return current

    return controller


def make_mfcc_controller(stages: Sequence[tuple[float, float]]) -> Callable[[np.ndarray, int], np.ndarray]:
    """Create a replicated multi-stage constant-current controller."""
    stages = [(float(current), float(soc_upper)) for current, soc_upper in stages]

    def controller(obs: np.ndarray, _step: int) -> np.ndarray:
        currents = np.zeros(obs.shape[0], dtype=float)
        for idx, soc in enumerate(obs[:, 0]):
            for current, soc_upper in stages:
                if float(soc) < soc_upper:
                    currents[idx] = current
                    break
        return currents

    return controller


def make_sac_controller(policy, i_max_amps: float) -> Callable[[np.ndarray, int], np.ndarray]:
    """Create a replicated SAC controller that predicts each cell independently."""

    def controller(obs: np.ndarray, _step: int) -> np.ndarray:
        currents = []
        for cell_obs in obs:
            action, _ = policy.predict(cell_obs.astype(np.float32), deterministic=True)
            action_value = float(np.asarray(action, dtype=float).reshape(-1)[0])
            currents.append(0.5 * (np.clip(action_value, -1.0, 1.0) + 1.0) * i_max_amps)
        return np.asarray(currents, dtype=float)

    return controller


def compute_pack_metrics(
    trajectory: pd.DataFrame,
    *,
    soc_target: float = 0.8,
    v_min: float = 2.5,
    v_max: float = 4.2,
) -> dict:
    """Compute pack-level time, degradation, safety, and balancing metrics."""
    if trajectory.empty:
        raise ValueError("trajectory is empty")
    per_step = (
        trajectory.groupby("step", as_index=False)
        .agg(
            time_s=("time_s", "first"),
            soc_min=("soc", "min"),
            soc_max=("soc", "max"),
            soc_mean=("soc", "mean"),
            soh_mean=("soh", "mean"),
            soh_start_mean=("soh_before", "mean"),
            voltage_min=("voltage", "min"),
            voltage_max=("voltage", "max"),
            temperature_max=("temperature", "max"),
            current_mean=("current_A", "mean"),
            current_max=("current_A", "max"),
            abs_balance_current_sum=("balance_current_A", lambda x: float(np.abs(x).sum())),
            reward_sum=("reward", "sum"),
            pack_voltage=("pack_voltage", "first"),
        )
    )
    hit = per_step[per_step["soc_min"] >= soc_target]
    start = trajectory[trajectory["step"] == trajectory["step"].min()]
    end = trajectory[trajectory["step"] == trajectory["step"].max()]
    overvoltage = ((trajectory["voltage"] > v_max) | (trajectory["voltage"] < v_min)).sum()
    raw_overvoltage = ((trajectory["world_model_voltage_raw"] > v_max) | (trajectory["world_model_voltage_raw"] < v_min)).sum()
    return {
        "steps": int(per_step["step"].max()),
        "return": float(per_step["reward_sum"].sum()),
        "time_to_target_s": float(hit["time_s"].iloc[0]) if len(hit) else np.nan,
        "hit_target": bool(len(hit)),
        "soc_min_start": float(start["soc_before"].min()),
        "soc_max_start": float(start["soc_before"].max()),
        "soc_spread_start": float(start["soc_before"].max() - start["soc_before"].min()),
        "soc_min_end": float(end["soc"].min()),
        "soc_mean_end": float(end["soc"].mean()),
        "soc_spread_end": float(end["soc"].max() - end["soc"].min()),
        "soc_spread_max": float((per_step["soc_max"] - per_step["soc_min"]).max()),
        "delta_soh_mean": max(float(start["soh_before"].mean() - end["soh"].mean()), 0.0),
        "delta_soh_sum": max(float(start["soh_before"].sum() - end["soh"].sum()), 0.0),
        "overvoltage_count": int(overvoltage),
        "raw_overvoltage_count": int(raw_overvoltage),
        "mean_current_A": float(per_step["current_mean"].mean()),
        "max_cell_current_A": float(per_step["current_max"].max()),
        "max_voltage": float(per_step["voltage_max"].max()),
        "max_temperature": float(per_step["temperature_max"].max()),
        "max_pack_voltage": float(per_step["pack_voltage"].max()),
        "balance_throughput_A_s": float(per_step["abs_balance_current_sum"].sum()),
    }


def plot_pack_comparison(trajectories: pd.DataFrame, out_dir: Path) -> Path:
    """Plot pack-level SOC envelope, spread, current, voltage, and temperature."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for strategy, group in trajectories.groupby("strategy"):
        per_step = (
            group.groupby("step", as_index=False)
            .agg(
                time_s=("time_s", "first"),
                soc_min=("soc", "min"),
                soc_max=("soc", "max"),
                soc_mean=("soc", "mean"),
                current_mean=("current_A", "mean"),
                voltage_max=("voltage", "max"),
                voltage_min=("voltage", "min"),
                temperature_max=("temperature", "max"),
            )
        )
        axes[0].plot(per_step["time_s"], per_step["soc_mean"], label=f"{strategy} mean")
        axes[0].fill_between(per_step["time_s"], per_step["soc_min"], per_step["soc_max"], alpha=0.15)
        axes[1].plot(per_step["time_s"], per_step["soc_max"] - per_step["soc_min"], label=strategy)
        axes[2].plot(per_step["time_s"], per_step["current_mean"], label=strategy)
        axes[3].plot(per_step["time_s"], per_step["voltage_max"], label=f"{strategy} max")
        axes[3].plot(per_step["time_s"], per_step["voltage_min"], linestyle="--", alpha=0.7)
        axes[4].plot(per_step["time_s"], per_step["temperature_max"], label=strategy)
    labels = ["SOC", "SOC spread", "Mean current (A)", "Cell voltage (V)", "Max temperature (C)"]
    for ax, ylabel in zip(axes, labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    path = out_dir / "pack_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _pack_snapshot(states: Sequence[np.ndarray], cfg: PackConfig) -> dict:
    """Aggregate per-cell states into pack-level diagnostics."""
    arr = np.stack(states, axis=0).astype(float)
    module_voltage = arr[:, 2].reshape(cfg.n_series, cfg.n_parallel).mean(axis=1)
    return {
        "soc_min": float(arr[:, 0].min()),
        "soc_mean": float(arr[:, 0].mean()),
        "soc_max": float(arr[:, 0].max()),
        "soc_spread": float(arr[:, 0].max() - arr[:, 0].min()),
        "voltage_min": float(arr[:, 2].min()),
        "voltage_max": float(arr[:, 2].max()),
        "pack_voltage": float(module_voltage.sum()),
        "temperature_max": float(arr[:, 4].max()),
    }


def _paired_pack_against_cc_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare pack strategies with CC-CV on episodes where both hit target."""
    rows = []
    cc = metrics[metrics["strategy"] == "cc_cv"].set_index("episode")
    for strategy, group in metrics.groupby("strategy"):
        if strategy == "cc_cv":
            continue
        other = group.set_index("episode")
        common = cc.index.intersection(other.index)
        common = [idx for idx in common if bool(cc.loc[idx, "hit_target"]) and bool(other.loc[idx, "hit_target"])]
        if not common:
            continue
        cc_hit = cc.loc[common]
        other_hit = other.loc[common]
        cc_time = float(cc_hit["time_to_target_s"].mean())
        strategy_time = float(other_hit["time_to_target_s"].mean())
        cc_soh = float(cc_hit["delta_soh_mean"].mean())
        strategy_soh = float(other_hit["delta_soh_mean"].mean())
        cc_spread = float(cc_hit["soc_spread_end"].mean())
        strategy_spread = float(other_hit["soc_spread_end"].mean())
        rows.append(
            {
                "strategy": strategy,
                "paired_episodes": int(len(common)),
                "cc_cv_time_to_target_s": cc_time,
                "strategy_time_to_target_s": strategy_time,
                "speed_improvement_pct": 100.0 * (cc_time - strategy_time) / max(cc_time, 1e-12),
                "cc_cv_delta_soh_mean": cc_soh,
                "strategy_delta_soh_mean": strategy_soh,
                "delta_soh_reduction_pct": 100.0 * (cc_soh - strategy_soh) / max(cc_soh, 1e-12),
                "cc_cv_soc_spread_end": cc_spread,
                "strategy_soc_spread_end": strategy_spread,
                "soc_spread_reduction_pct": 100.0 * (cc_spread - strategy_spread) / max(cc_spread, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run pack-level CC-CV / MFCC / SAC comparison and save artifacts."""
    parser = argparse.ArgumentParser(description="Replicate W4 single-cell strategies onto a pack")
    parser.add_argument("--sac-policy", type=Path, default=Path("outputs/sac_policy.zip"))
    parser.add_argument("--world-model", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--ecm-params", type=Path, default=DEFAULT_ECM_PARAMS)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval_pack"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--policy-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--strategies", nargs="+", default=["cc_cv", "mfcc", "ours"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-series", type=int, default=6)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--i-max-amps", type=float, default=5.0)
    parser.add_argument("--cc-current", type=float, default=3.0)
    parser.add_argument("--mfcc-stages", type=str, default="3.5:0.50,2.5:0.65,1.25:0.80")
    parser.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance-gain", type=float, default=8.0)
    parser.add_argument("--max-balance-current", type=float, default=0.8)
    parser.add_argument("--soc-spread-std", type=float, default=0.035)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = PackConfig(
        n_series=args.n_series,
        n_parallel=args.n_parallel,
        max_steps=args.max_steps,
        i_max_amps=args.i_max_amps,
        balance_gain_A_per_soc=args.balance_gain,
        max_balance_current_A=args.max_balance_current,
        soc_spread_std=args.soc_spread_std,
    )

    from stable_baselines3 import SAC

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = SAC.load(str(args.sac_policy), device=args.policy_device) if "ours" in args.strategies else None
    stages = _parse_mfcc_stages(args.mfcc_stages)
    controllers = {
        "cc_cv": make_cc_cv_controller(args.cc_current),
        "mfcc": make_mfcc_controller(stages),
    }
    if policy is not None:
        controllers["ours"] = make_sac_controller(policy, cfg.i_max_amps)

    all_trajs: list[pd.DataFrame] = []
    metric_rows = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        for strategy in args.strategies:
            envs = build_pack_envs(args.world_model, args.ecm_params, cfg=cfg, device=args.device)
            simulator = PackChargingSimulator(envs, cfg)
            traj = simulator.rollout(controllers[strategy], strategy=strategy, seed=episode_seed, balance=args.balance)
            traj["episode"] = episode
            all_trajs.append(traj)
            metrics = compute_pack_metrics(traj, soc_target=cfg.soc_target, v_min=envs[0].cfg.V_min, v_max=envs[0].cfg.V_max)
            metrics["strategy"] = strategy
            metrics["episode"] = episode
            metric_rows.append(metrics)

    trajectories = pd.concat(all_trajs, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("strategy", as_index=False).agg(
        episodes=("episode", "count"),
        hit_rate=("hit_target", "mean"),
        time_to_target_s_mean=("time_to_target_s", "mean"),
        time_to_target_s_std=("time_to_target_s", "std"),
        delta_soh_mean=("delta_soh_mean", "mean"),
        soc_spread_end_mean=("soc_spread_end", "mean"),
        soc_spread_max_mean=("soc_spread_max", "mean"),
        overvoltage_count_sum=("overvoltage_count", "sum"),
        raw_overvoltage_count_sum=("raw_overvoltage_count", "sum"),
        max_temperature_mean=("max_temperature", "mean"),
        balance_throughput_A_s_mean=("balance_throughput_A_s", "mean"),
        return_mean=("return", "mean"),
    )
    paired = _paired_pack_against_cc_cv(metrics)

    trajectories.to_csv(args.out_dir / "pack_trajectories.csv", index=False)
    metrics.to_csv(args.out_dir / "pack_metrics_by_episode.csv", index=False)
    summary.to_csv(args.out_dir / "pack_metrics_summary.csv", index=False)
    paired.to_csv(args.out_dir / "pack_paired_vs_cc_cv.csv", index=False)
    if args.plot:
        plot_pack_comparison(trajectories, args.out_dir)
    print(summary.to_string(index=False))
    if not paired.empty:
        print("\nPack paired vs CC-CV:")
        print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
