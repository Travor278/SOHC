"""SOC 推断：KeiLongW Stacked LSTM (TF/Keras) warm-start + NASA fine-tune (v0.2)。

数据策略 v0.2：
    - KeiLongW 预训练权重（在 LG 18650HG2 上训）作为 warm-start
    - 用 NASA BatteryAgingARC-FY08Q4（B0025-B0056，多温度多倍率）做 fine-tune
    - fine-tune 后的模型在 NASA B0005-B0018 上 inference，给 Mamba 世界模型打 SOC 软标签

输入：
    --weights      : KeiLongW release 里的 .h5（warm-start 用）
    --finetune-data: NASA ARC-FY08Q4 .mat 目录（fine-tune 用）
    --data         : 推断输入 V/I/T 时序 CSV / .mat
    --window       : 滑动窗口长度（KeiLongW 默认 100 / 200 / 300）
    --out          : 输出 CSV 路径

输出：
    CSV 含列 [t, voltage, current, temperature, ambient_T, soc_pred]

W1 任务：fine-tune + 在 NASA holdout 上验证 MAE < 1.5%。
W2 任务：跑 NASA B0005-B0018 + Randomized，输出软标签供 Mamba 训练。
W5 任务：跑 Zenodo 6985321 (WLTP) 定量验证 + Zenodo 18471156 定性展示。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def load_keilongw_model(weights_path: Path):
    """加载 KeiLongW 的 Keras 模型。

    KeiLongW 仓库结构：
        experiments/lg/lstm_soc_lg_*_steps.ipynb 训练
        results/lg/<run>/model.h5 保存权重
    本函数应该重建网络结构（Stacked LSTM 256→256→128→Dense64→Dense1, selu）
    然后 load_weights()。
    """
    raise NotImplementedError("W1: 复现 KeiLongW 网络结构后接预训练权重")


def preprocess_sequence(df, window: int):
    """把 (V, I, T) 时序切成 (N, window, 3) 张量，做归一化。

    归一化参数应与 KeiLongW 训练时一致——读 results/lg/<run>/scaler.pkl。
    """
    raise NotImplementedError("W1: 对齐 KeiLongW 的归一化器")


def predict_soc(model, X) -> "np.ndarray":
    """单批前向，返回 (N,) SOC 预测。"""
    raise NotImplementedError("W1")


def main():
    parser = argparse.ArgumentParser(description="SOC inference via KeiLongW LSTM")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("outputs/soc_pred.csv"))
    args = parser.parse_args()

    # TODO (W1):
    # 1. df = pd.read_csv(args.data)
    # 2. X, t = preprocess_sequence(df, args.window)
    # 3. model = load_keilongw_model(args.weights)
    # 4. soc = predict_soc(model, X)
    # 5. write CSV with columns [t, V, I, T, soc_pred]
    raise NotImplementedError(
        "实现见 W1 任务（TODO.md）。先跑 external/KeiLongW 原始 notebook 拿到权重，再适配此包装。"
    )


if __name__ == "__main__":
    main()
