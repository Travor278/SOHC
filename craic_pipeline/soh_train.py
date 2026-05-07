"""SOH 估计器训练：NASA 同源数据 + BatteryML trainer (v0.2)。

数据策略 v0.2：
    - 训练数据：NASA B0005-B0018（容量退化时序）+ ARC-FY08Q4（多温度多倍率补充）
    - 全程 NMC 18650 同源，无需跨化学体系 fine-tune
    - HUST 数据（仓库已携带）退为可选展示，不进训练管线

用途：
    - W1：在 NASA 数据上训 baseline SOH（Variance / 浅层 CNN）
    - W4：BatteryML 内挂 Mamba head 跑 SOH 对比表（架构创新点）

输入：
    --config : configs/nasa_soh_*.yaml（BatteryML 风格）
    --data   : NASA .mat 目录（默认 data/nasa_pcoe/）
    --out    : 权重保存路径（默认 outputs/soh_baseline.pt）

输出：
    .pt 权重 + JSON 训练日志（loss / RMSE 曲线）

依赖：
    external/BatteryML 已 clone（见 external/README.md）
    自写 NASA loader（craic_pipeline.nasa_loader），不复用 BatteryML 的 HUST loader
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_nasa_for_batteryml(data_dir: Path):
    """从 NASA .mat 加载并转换成 BatteryML 的 BatteryData 类。

    BatteryML 的 BatteryData 期望字段（参考其 batteryml/data/battery_data.py）：
        cell_id, cycle_data: List[CycleData]
        每个 CycleData: voltage_in_V, current_in_A, temperature_in_C,
                       discharge_capacity_in_Ah, time_in_s, ...
    NASA loader 见 craic_pipeline.nasa_loader（W1 待写）。
    """
    raise NotImplementedError("W1: 写 NASA -> BatteryData 适配层")


def build_model(cfg):
    """根据 cfg.model.type 构造模型：
        - "variance" : sklearn Ridge on hand-crafted features (BatteryML 自带)
        - "cnn"      : BatteryML 内置 CNN
        - "mamba"    : 我们自己接的 Mamba head（W4 加）
    """
    raise NotImplementedError("W1 baseline → W4 Mamba head")


def train(model, train_cells, val_cells, cfg):
    """训练循环。BatteryML 有自己的 Trainer，可以直接复用其 trainer.fit()。"""
    raise NotImplementedError("W1")


def main():
    parser = argparse.ArgumentParser(description="SOH training on HUST via BatteryML")
    parser.add_argument("--config", type=Path, default=Path("configs/nasa_soh_baseline.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/nasa_pcoe"))
    parser.add_argument("--out", type=Path, default=Path("outputs/soh_baseline.pt"))
    args = parser.parse_args()

    # TODO (W1):
    # 1. cfg = yaml.safe_load(args.config)
    # 2. cells = load_nasa_for_batteryml(args.data)
    # 3. train_cells, val_cells = split_by_cell_id(cells, val_ratio=0.2)
    # 4. model = build_model(cfg)
    # 5. trainer.fit(model, train_cells, val_cells)
    # 6. torch.save(model.state_dict(), args.out)
    raise NotImplementedError("实现见 W1 任务（TODO.md）。")


if __name__ == "__main__":
    main()
