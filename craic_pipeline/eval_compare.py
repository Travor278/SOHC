"""W4 evaluation: compare SAC charging against CC-CV and MFCC baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ECM_PARAMS = Path("MATLAB滤波算法代码——云储实时数据/1-2-model_identification_RC/result/savemat_2order.mat")


def build_eval_env(
    world_model_path: Path,
    ecm_params_path: Path = DEFAULT_ECM_PARAMS,
    *,
    device: str = "auto",
    max_steps: int = 600,
    i_max_amps: float = 5.0,
    soc_target: float = 0.8,
):
    """Create a W4 evaluation env from the W2 world model and ECM params."""
    from craic_pipeline.ecm_safety_layer import ECMSafetyLayer, load_params_from_mat
    from craic_pipeline.rl_env import BatteryChargingEnv, EnvConfig
    from craic_pipeline.train_sac import _resolve_device
    from craic_pipeline.world_model_mamba import load_world_model_checkpoint

    world_model, metrics = load_world_model_checkpoint(Path(world_model_path))
    params = load_params_from_mat(Path(ecm_params_path))
    cfg = EnvConfig(
        max_steps=max_steps,
        soc_target=soc_target,
        V_min=params.V_min,
        V_max=params.V_max,
        I_max_amps=i_max_amps,
        seq_len=int(getattr(world_model.cfg, "seq_len", 64)),
        device=_resolve_device(device),
    )
    env = BatteryChargingEnv(world_model, ECMSafetyLayer(params, dt=cfg.dt), cfg)
    env.metadata["world_model_metrics"] = metrics
    return env


def run_baseline_cc_cv(env, soc_target=0.8, current_A: float | None = None, seed: int | None = None):
    """Run a CC-CV baseline where ECM projection supplies the CV clipping."""
    request_A = float(current_A if current_A is not None else 0.8 * env.cfg.I_max_amps)

    def controller(obs, _env, _step):
        if float(obs[0]) >= soc_target:
            return _current_to_action(0.0, _env.cfg.I_max_amps)
        return _current_to_action(request_A, _env.cfg.I_max_amps)

    return _rollout(env, controller, label="cc_cv", seed=seed)


def run_baseline_mfcc(env, stages=None, seed: int | None = None):
    """Run a multi-stage constant-current baseline from `(current_A, soc_upper)` stages."""
    if stages is None:
        stages = [
            (0.85 * env.cfg.I_max_amps, 0.50),
            (0.60 * env.cfg.I_max_amps, 0.65),
            (0.35 * env.cfg.I_max_amps, env.cfg.soc_target),
        ]
    stages = [(float(current), float(soc_upper)) for current, soc_upper in stages]

    def controller(obs, _env, _step):
        soc = float(obs[0])
        for current, soc_upper in stages:
            if soc < soc_upper:
                return _current_to_action(current, _env.cfg.I_max_amps)
        return _current_to_action(0.0, _env.cfg.I_max_amps)

    return _rollout(env, controller, label="mfcc", seed=seed)


def run_policy(env, sac_policy_path: Path, seed: int | None = None, policy=None, deterministic: bool = True):
    """Deploy a trained SAC policy and record `(state, action, reward)` per step."""
    if policy is None:
        from stable_baselines3 import SAC

        policy = SAC.load(str(sac_policy_path))

    def controller(obs, _env, _step):
        action, _ = policy.predict(obs, deterministic=deterministic)
        return float(np.asarray(action, dtype=float).reshape(-1)[0])

    return _rollout(env, controller, label="ours", seed=seed)


def compute_metrics(trajectory, *, soc_target: float = 0.8, v_min: float = 2.5, v_max: float = 4.2) -> dict:
    """Compute W4 charging time, degradation, safety, and temperature metrics."""
    df = pd.DataFrame(trajectory)
    if df.empty:
        raise ValueError("trajectory is empty")
    hit = df[df["soc"] >= soc_target]
    time_to_target = float(hit["time_s"].iloc[0]) if len(hit) else np.nan
    raw_v = df.get("world_model_voltage_raw", df["voltage"])
    overvoltage = ((df["voltage"] > v_max) | (df["voltage"] < v_min)).sum()
    raw_overvoltage = ((raw_v > v_max) | (raw_v < v_min)).sum()
    start_soh = float(df["soh_before"].iloc[0])
    end_soh = float(df["soh"].iloc[-1])
    start_soc = float(df["soc_before"].iloc[0])
    end_soc = float(df["soc"].iloc[-1])
    return {
        "steps": int(len(df)),
        "return": float(df["reward"].sum()),
        "time_to_80_s": time_to_target,
        "hit_target": bool(len(hit)),
        "soc_start": start_soc,
        "soc_end": end_soc,
        "delta_soc": end_soc - start_soc,
        "delta_soh": max(start_soh - end_soh, 0.0),
        "overvoltage_count": int(overvoltage),
        "raw_overvoltage_count": int(raw_overvoltage),
        "mean_T": float(df["temperature"].mean()),
        "max_T": float(df["temperature"].max()),
        "mean_current_A": float(df["current_A"].mean()),
        "max_voltage": float(df["voltage"].max()),
    }


def plot_comparison(trajectories: dict, out_dir: Path):
    """Plot four W4 comparison curves: I(t), V(t), SOC(t), and T(t)."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    series = [
        ("current_A", "Current (A)"),
        ("voltage", "Voltage (V)"),
        ("soc", "SOC"),
        ("temperature", "Temperature (C)"),
    ]
    for label, traj in trajectories.items():
        df = pd.DataFrame(traj)
        if "episode" in df:
            df = df[df["episode"] == df["episode"].min()]
        for ax, (col, ylabel) in zip(axes, series):
            ax.plot(df["time_s"], df[col], label=label)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    out_path = out_dir / "charging_comparison.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _rollout(env, controller, *, label: str, seed: int | None = None) -> pd.DataFrame:
    """Roll out one deterministic controller inside a W4 env."""
    obs, _ = env.reset(seed=seed)
    rows = []
    for step in range(env.cfg.max_steps):
        state_before = np.asarray(obs, dtype=float).copy()
        action = float(np.clip(controller(obs, env, step), -1.0, 1.0))
        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        rows.append(
            {
                "strategy": label,
                "step": step + 1,
                "time_s": (step + 1) * env.cfg.dt,
                "action_norm": action,
                "soc_before": state_before[0],
                "soh_before": state_before[1],
                "voltage_before": state_before[2],
                "temperature_before": state_before[4],
                "soc": info["soc"],
                "soh": info["soh"],
                "voltage": info["voltage"],
                "temperature": info["temperature"],
                "current_A": info["safe_current"],
                "requested_current_A": info["requested_current"],
                "world_model_voltage_raw": info.get("world_model_voltage_raw", info["voltage"]),
                "model_delta_soh": info.get("model_delta_soh", 0.0),
                "aging_proxy_delta_soh": info.get("aging_proxy_delta_soh", 0.0),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


def _current_to_action(current_A: float, i_max_amps: float) -> float:
    """Map a physical charging current in amps to normalized env action."""
    current = float(np.clip(current_A, 0.0, i_max_amps))
    return float(np.clip(2.0 * current / max(i_max_amps, 1e-12) - 1.0, -1.0, 1.0))


def _paired_against_cc_cv(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare strategies with CC-CV on episodes where both hit SOC target."""
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
        cc_time = float(cc_hit["time_to_80_s"].mean())
        strategy_time = float(other_hit["time_to_80_s"].mean())
        cc_soh = float(cc_hit["delta_soh"].mean())
        strategy_soh = float(other_hit["delta_soh"].mean())
        rows.append(
            {
                "strategy": strategy,
                "paired_episodes": int(len(common)),
                "cc_cv_time_to_80_s": cc_time,
                "strategy_time_to_80_s": strategy_time,
                "speed_improvement_pct": 100.0 * (cc_time - strategy_time) / max(cc_time, 1e-12),
                "cc_cv_delta_soh": cc_soh,
                "strategy_delta_soh": strategy_soh,
                "delta_soh_reduction_pct": 100.0 * (cc_soh - strategy_soh) / max(cc_soh, 1e-12),
                "cc_cv_overvoltage": int(cc_hit["overvoltage_count"].sum()),
                "strategy_overvoltage": int(other_hit["overvoltage_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _parse_mfcc_stages(raw: str) -> list[tuple[float, float]]:
    """Parse CLI stages formatted as `I:soc,I:soc`."""
    stages = []
    for piece in raw.split(","):
        if not piece.strip():
            continue
        current, soc_upper = piece.split(":")
        stages.append((float(current), float(soc_upper)))
    if not stages:
        raise ValueError("MFCC stages cannot be empty")
    return stages


def main():
    """Run W4 policy/baseline comparison and save trajectories plus metrics."""
    parser = argparse.ArgumentParser(description="Compare SAC policy vs CC-CV / MFCC")
    parser.add_argument("--sac-policy", type=Path, default=Path("outputs/sac_policy.zip"))
    parser.add_argument("--world-model", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--ecm-params", type=Path, default=DEFAULT_ECM_PARAMS)
    parser.add_argument("--baselines", nargs="+", default=["cc_cv", "mfcc"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--policy-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--i-max-amps", type=float, default=5.0)
    parser.add_argument("--cc-current", type=float, default=3.0)
    parser.add_argument("--mfcc-stages", type=str, default="3.5:0.50,2.5:0.65,1.25:0.80")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import SAC

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy = SAC.load(str(args.sac_policy), device=args.policy_device)
    stages = _parse_mfcc_stages(args.mfcc_stages)
    all_trajs: list[pd.DataFrame] = []
    metrics_rows = []

    for episode in range(args.episodes):
        seed = args.seed + episode
        env = build_eval_env(
            args.world_model,
            args.ecm_params,
            device=args.device,
            max_steps=args.max_steps,
            i_max_amps=args.i_max_amps,
        )
        runners = {}
        if "cc_cv" in args.baselines:
            runners["cc_cv"] = lambda one_env: run_baseline_cc_cv(one_env, current_A=args.cc_current, seed=seed)
        if "mfcc" in args.baselines:
            runners["mfcc"] = lambda one_env: run_baseline_mfcc(one_env, stages=stages, seed=seed)
        runners["ours"] = lambda one_env: run_policy(one_env, args.sac_policy, seed=seed, policy=policy)

        for strategy, runner in runners.items():
            traj = runner(env)
            traj["episode"] = episode
            all_trajs.append(traj)
            metrics = compute_metrics(traj, soc_target=env.cfg.soc_target, v_min=env.cfg.V_min, v_max=env.cfg.V_max)
            metrics["strategy"] = strategy
            metrics["episode"] = episode
            metrics_rows.append(metrics)
        env.close()

    trajectories = pd.concat(all_trajs, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    summary = metrics.groupby("strategy", as_index=False).agg(
        episodes=("episode", "count"),
        hit_rate=("hit_target", "mean"),
        time_to_80_s_mean=("time_to_80_s", "mean"),
        time_to_80_s_std=("time_to_80_s", "std"),
        delta_soh_mean=("delta_soh", "mean"),
        overvoltage_count_sum=("overvoltage_count", "sum"),
        raw_overvoltage_count_sum=("raw_overvoltage_count", "sum"),
        mean_T_mean=("mean_T", "mean"),
        max_T_mean=("max_T", "mean"),
        return_mean=("return", "mean"),
        soc_end_mean=("soc_end", "mean"),
    )
    paired = _paired_against_cc_cv(metrics)

    trajectories.to_csv(args.out_dir / "trajectories.csv", index=False)
    metrics.to_csv(args.out_dir / "metrics_by_episode.csv", index=False)
    summary.to_csv(args.out_dir / "metrics_summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired_vs_cc_cv.csv", index=False)
    if args.plot:
        plot_comparison({key: df for key, df in trajectories.groupby("strategy")}, args.out_dir)
    print(summary.to_string(index=False))
    if not paired.empty:
        print("\nPaired vs CC-CV:")
        print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
