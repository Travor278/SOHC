"""W4-W5 评估：本方案 vs CC-CV / MFCC / MIUKF + Zenodo 泛化展示 (v0.2)。

输出（W4 定量）：
    - 充电曲线对比图（I-t / V-t / SOC-t / T-t）
    - 指标表 (CSV)：充至 80% SOC 耗时 / ΔSOH 单循环 / 过压报警次数 / 平均 T

输出（W5 泛化）：
    - L3 定量：Zenodo 6985321 (WLTP) zero-shot 误差曲线
    - L4 定性：Zenodo 18471156 真实电站监测数据上 SOC/SOH 输出曲线（PPT 末尾 1 张图）

基线：
    cc_cv  : 工业标准（V<4.2 时恒流，达 4.2 后恒压）
    mfcc   : 多阶段恒流（论文常用 baseline）
    miukf  : 复用本仓库 MATLAB MIUKF 输出（仅 SOC 维度对比）
    socnet : 本仓库 神经网络/SOCtarget LSTM 输出（仅 SOC 维度对比）
    ours   : SAC 策略
"""
from __future__ import annotations

import argparse
from pathlib import Path


def run_baseline_cc_cv(env, soc_target=0.8):
    """CC-CV 充电策略：固定电流到上限电压，再恒压。"""
    raise NotImplementedError("W4")


def run_baseline_mfcc(env, stages):
    """多阶段恒流：[(I_1, soc_1), (I_2, soc_2), ...]"""
    raise NotImplementedError("W4")


def run_policy(env, sac_policy_path: Path):
    """部署训好的 SAC 策略，记录每步 (state, action, reward)。"""
    raise NotImplementedError("W4")


def compute_metrics(trajectory):
    """从轨迹计算关键指标。"""
    # return dict(time_to_80_soc=..., delta_soh=..., overvoltage_count=..., mean_T=...)
    raise NotImplementedError("W4")


def plot_comparison(trajectories: dict, out_dir: Path):
    """4 联子图：I(t), V(t), SOC(t), T(t)，每条曲线一个策略。"""
    raise NotImplementedError("W4")


def main():
    parser = argparse.ArgumentParser(description="Compare SAC policy vs CC-CV / MFCC")
    parser.add_argument("--sac-policy", type=Path, default=Path("outputs/sac_policy.zip"))
    parser.add_argument("--world-model", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--baselines", nargs="+", default=["cc_cv", "mfcc"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    # TODO (W4):
    # 1. env = make_env(...)
    # 2. trajs = {}
    # 3. for b in args.baselines: trajs[b] = run_baseline_*(env)
    # 4. trajs["ours"] = run_policy(env, args.sac_policy)
    # 5. metrics_table = {k: compute_metrics(v) for k, v in trajs.items()}
    # 6. save CSV + 如果 --plot 调 plot_comparison
    raise NotImplementedError("W4")


if __name__ == "__main__":
    main()
