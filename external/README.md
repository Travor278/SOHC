# external/ — 外部仓库

这里存放 git submodule 或手动 clone 的外部依赖仓库。本目录在 `.gitignore` 中，不会被提交。

## 必需仓库

### 1. KeiLongW/battery-state-estimation（SOC，TF/Keras）

```bash
git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW
```

- Stars: 185+
- 用途：SOC 估计的 Stacked LSTM 模型
- 关键文件：
  - `experiments/lg/lstm_soc_lg_*.ipynb` — 训练 notebook
  - `data_processing/` — LG 18650HG2 数据预处理
  - GitHub Releases 提供预训练 `.h5` 权重

我们的代码 `craic_pipeline/soc_inference.py` 会加载它的预训练权重做 inference。

### 2. microsoft/BatteryML（SOH，PyTorch）

```bash
git clone https://github.com/microsoft/BatteryML.git external/BatteryML
```

- Stars: 740+
- 用途：SOH 估计的多算法 benchmark + HUST 数据 loader
- 关键文件：
  - `batteryml/data/preprocess/preprocess_HUST.py` — HUST 数据加载
  - `examples/soh_example.ipynb` — SOH 训练范例
  - `configs/` — 多种模型配置

我们的代码 `craic_pipeline/soh_train.py` 调用它的 HUST loader + Trainer。

## 可选

如果你想做迁移学习对比，还可以 clone：

```bash
# ArpanBiswas99 PyTorch 版 SOC（备用基线）
git clone https://github.com/ArpanBiswas99/Battery-State-of-Charge-Estimation.git external/ArpanBiswas99
```
