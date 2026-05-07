"""SAC training entry point for the W3 charging policy."""
from __future__ import annotations

import argparse
from pathlib import Path


def make_env(
    world_model_path: Path,
    ecm_params_path: Path,
    *,
    device: str = "auto",
    max_steps: int = 600,
    soc_target: float = 0.8,
    i_max_amps: float = 5.0,
    reward_weights: dict[str, float] | None = None,
    seed: int | None = None,
):
    """Create a monitored W3 SAC environment from W2 and ECM artifacts."""
    from stable_baselines3.common.monitor import Monitor

    from craic_pipeline.ecm_safety_layer import ECMSafetyLayer, load_params_from_mat
    from craic_pipeline.rl_env import BatteryChargingEnv, EnvConfig, RewardWeights
    from craic_pipeline.world_model_mamba import load_world_model_checkpoint

    resolved_device = _resolve_device(device)
    try:
        world_model, metrics = load_world_model_checkpoint(Path(world_model_path))
    except Exception as exc:
        raise RuntimeError(
            "Failed to load world-model checkpoint. If this checkpoint was trained with "
            "Mamba, run SAC in the WSL/Linux environment where mamba-ssm is installed, "
            "or retrain W2 with --gru-fallback."
        ) from exc
    params = load_params_from_mat(Path(ecm_params_path))
    reward = RewardWeights(**reward_weights) if reward_weights else None
    cfg = EnvConfig(
        max_steps=max_steps,
        soc_target=soc_target,
        V_min=params.V_min,
        V_max=params.V_max,
        I_max_amps=i_max_amps,
        seq_len=int(getattr(world_model.cfg, "seq_len", 64)),
        device=resolved_device,
        reward=reward,
    )
    env = BatteryChargingEnv(world_model, ECMSafetyLayer(params, dt=cfg.dt), cfg)
    env.metadata["world_model_metrics"] = metrics
    if seed is not None:
        env.reset(seed=seed)
    return Monitor(env)


def _resolve_device(requested: str) -> str:
    """Resolve `auto/cuda/cpu` and verify CUDA can run one tiny tensor op."""
    import torch

    requested = requested.lower()
    if requested == "cpu":
        return "cpu"
    if requested not in {"auto", "cuda"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return "cpu"
    try:
        _ = (torch.ones(1, device="cuda") + 1).cpu()
        return "cuda"
    except Exception:
        if requested == "cuda":
            raise
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train SAC charging policy")
    parser.add_argument("--world-model", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--ecm-params", type=Path,
                        default=Path("MATLAB滤波算法代码——云储实时数据/1-2-model_identification_RC/result/savemat_2order.mat"))
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("outputs/sac_policy.zip"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/runs/sac"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--policy-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--soc-target", type=float, default=0.8)
    parser.add_argument("--i-max-amps", type=float, default=5.0)
    parser.add_argument("--reward-speed", type=float, default=12.0)
    parser.add_argument("--reward-voltage", type=float, default=50.0)
    parser.add_argument("--reward-temperature", type=float, default=0.2)
    parser.add_argument("--reward-aging", type=float, default=80.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        args.world_model,
        args.ecm_params,
        device=args.device,
        max_steps=args.max_steps,
        soc_target=args.soc_target,
        i_max_amps=args.i_max_amps,
        reward_weights={
            "speed": args.reward_speed,
            "voltage": args.reward_voltage,
            "temperature": args.reward_temperature,
            "aging": args.reward_aging,
        },
        seed=args.seed,
    )
    policy_device = _resolve_device(args.policy_device)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=min(1000, max(1, args.total_steps // 10)),
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        device=policy_device,
        verbose=1,
    )
    callbacks = []
    if args.checkpoint_freq > 0:
        checkpoint_dir = args.log_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix=args.out.stem,
            )
        )
    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=args.total_steps, callback=callback, progress_bar=args.progress_bar)
    model.save(args.out)
    env.close()
    print(f"saved SAC policy to {args.out}")


if __name__ == "__main__":
    main()
