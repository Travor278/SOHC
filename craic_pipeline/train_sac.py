"""SAC 训练入口（stable-baselines3）。

W3：训练 SAC 策略，目标 100k steps。
依赖：rl_env.BatteryChargingEnv, world_model_mamba.BatteryWorldModel,
       ecm_safety_layer.ECMSafetyLayer

运行：
    python -m craic_pipeline.train_sac \
        --world-model outputs/world_model.pt \
        --total-steps 100000 \
        --out outputs/sac_policy.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path


def make_env(world_model_path: Path, ecm_params_path: Path):
    """工厂函数，被 SubprocVecEnv 调用。"""
    # from stable_baselines3.common.monitor import Monitor
    # wm = load_world_model(world_model_path)
    # safety = ECMSafetyLayer(load_params_from_mat(ecm_params_path))
    # env = BatteryChargingEnv(wm, safety, EnvConfig())
    # return Monitor(env)
    raise NotImplementedError("W3")


def main():
    parser = argparse.ArgumentParser(description="Train SAC charging policy")
    parser.add_argument("--world-model", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--ecm-params", type=Path,
                        default=Path("MATLAB滤波算法代码——云储实时数据/1-2-model_identification_RC/result/savemat_2order.mat"))
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=Path("outputs/sac_policy.zip"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/runs/sac"))
    args = parser.parse_args()

    # TODO (W3):
    # from stable_baselines3 import SAC
    # env = make_env(args.world_model, args.ecm_params)
    # model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=int(1e5),
    #             tensorboard_log=str(args.log_dir), verbose=1)
    # model.learn(total_timesteps=args.total_steps, progress_bar=True)
    # model.save(args.out)
    raise NotImplementedError("W3")


if __name__ == "__main__":
    main()
