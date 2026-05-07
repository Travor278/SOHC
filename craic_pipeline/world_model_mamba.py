"""Mamba 世界模型：电池单步动力学的端到端学习器。

定位：方向三三层架构的【层 1】。RL 的 env.step() 调它。

输入序列（per timestep）：
    [SOC_t, SOH_t, V_t, I_t, T_t, action_t]   shape (B, L, 6)

输出（next-step prediction）：
    [SOC_{t+1}, V_{t+1}, T_{t+1}, ΔSOH_step]  shape (B, 4)

训练数据：
    NASA PCoE B0005/06/07/18 连续放电时序
    SOC_t、SOH_t 由 craic_pipeline.soc_inference / soh_train 推断填入

退化方案（mamba-ssm 装不上时）：
    把 MambaBlock 换成 nn.GRU(hidden=128, num_layers=2)，其余结构保持。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    state_dim: int = 5     # [SOC, SOH, V, I, T]
    action_dim: int = 1    # I_charge
    hidden_dim: int = 128
    n_layers: int = 4
    seq_len: int = 64
    use_mamba: bool = True  # False → fall back to GRU


class BatteryWorldModel:
    """Mamba 主干，输入 (B, L, state+action)，输出 (B, 4) 下一步预测。

    参考 mamba-ssm 用法：
        from mamba_ssm import Mamba
        self.layers = nn.ModuleList([Mamba(d_model=H) for _ in range(N)])
    """

    def __init__(self, cfg: WorldModelConfig):
        self.cfg = cfg
        # TODO (W2): 初始化 Mamba layers + 输入投影 + 输出 head
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def step(self, state, action):
        """单步推进：(state_t, action_t) -> state_{t+1}。RL env 调用。"""
        raise NotImplementedError


def build_training_dataset(nasa_dir: Path, soc_csv: Path, soh_ckpt: Path):
    """构造 NASA + 软标签训练集。

    流程：
        1. 读 NASA .mat → V, I, T 时序
        2. 用 SOC 估计器推断 SOC_t（CSV 已有）
        3. 用 SOH 估计器在每个循环开始时推断 SOH_t
        4. ΔSOH_step 用前后两个循环 SOH 差分插值
        5. action_t 取 NASA 原始电流（已知）
    """
    raise NotImplementedError("W2: 整合三个数据源拼成训练 tensor")


def main():
    parser = argparse.ArgumentParser(description="Train Mamba world model on NASA PCoE")
    parser.add_argument("--nasa-dir", type=Path, default=Path("data/nasa_pcoe"))
    parser.add_argument("--soc-csv", type=Path, default=Path("outputs/soc_pred_nasa.csv"))
    parser.add_argument("--soh-ckpt", type=Path, default=Path("outputs/soh_baseline.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gru-fallback", action="store_true", help="Mamba 装不上时用 GRU")
    args = parser.parse_args()

    # TODO (W2):
    # 1. dataset = build_training_dataset(...)
    # 2. cfg = WorldModelConfig(use_mamba=not args.gru_fallback)
    # 3. model = BatteryWorldModel(cfg)
    # 4. 训练循环：MSE loss on next-step prediction
    # 5. eval：1 步 MAE < 5mV，20 步漂移 < 50mV (验证标准见 PLAN.md)
    raise NotImplementedError


if __name__ == "__main__":
    main()
