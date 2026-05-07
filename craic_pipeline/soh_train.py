"""SOH 估计器训练：调用 BatteryML 的 HUST loader + 自定义模型头。

用途：
    - W1：在 data/HUST data/ 上训 baseline SOH（Variance / CNN / 简单 MLP）
    - W4：在 BatteryML 内挂一个 Mamba head 跑 SOH 对比表
    - W2：在 NASA PCoE 上做最后一层 fine-tune（解决 LFP→NMC 跨化学体系）

输入：
    --config : configs/hust_soh_*.yaml（BatteryML 风格）
    --data   : HUST CSV 目录（默认 data/HUST data/）
    --out    : 权重保存路径（默认 outputs/soh_baseline.pt）

输出：
    .pt 权重 + JSON 训练日志（loss / RMSE 曲线）

依赖：
    external/BatteryML 已 clone（见 external/README.md）
    BatteryML 的 HUST loader: external/BatteryML/batteryml/data/preprocess/preprocess_HUST.py
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_hust_via_batteryml(data_dir: Path):
    """调 BatteryML 的 HUST 预处理 → BatteryData 列表。

    BatteryML 调用方式（参考其 examples/）：
        from batteryml.data import BatteryData
        from batteryml.builders import PREPROCESSORS
        proc = PREPROCESSORS.build({'name': 'HUSTPreprocessor', ...})
        cells = proc.process(data_dir)
    """
    raise NotImplementedError("W1: 跑通 BatteryML HUST loader")


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
    parser.add_argument("--config", type=Path, default=Path("configs/hust_soh_baseline.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/HUST data"))
    parser.add_argument("--out", type=Path, default=Path("outputs/soh_baseline.pt"))
    args = parser.parse_args()

    # TODO (W1):
    # 1. cfg = yaml.safe_load(args.config)
    # 2. cells = load_hust_via_batteryml(args.data)
    # 3. train_cells, val_cells = split_by_cell_id(cells, val_ratio=0.2)
    # 4. model = build_model(cfg)
    # 5. trainer.fit(model, train_cells, val_cells)
    # 6. torch.save(model.state_dict(), args.out)
    raise NotImplementedError("实现见 W1 任务（TODO.md）。")


if __name__ == "__main__":
    main()
