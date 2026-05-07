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
- 用途：SOH 估计的多算法 benchmark + Trainer 框架
- 关键文件：
  - `batteryml/data/battery_data.py` — `BatteryData` 数据类（v0.2 用我们自己的 NASA loader 转成此类型）
  - `examples/soh_example.ipynb` — SOH 训练范例
  - `configs/` — 多种模型配置参考

v0.2 用法：复用其 `BatteryData` 类型 + Trainer，但 loader 我们自写 NASA 适配（`craic_pipeline/nasa_loader.py`），不复用其 HUST/CALCE 等内置 loader。

## 可选

如果你想做迁移学习对比，还可以 clone：

```bash
# ArpanBiswas99 PyTorch 版 SOC（备用基线）
git clone https://github.com/ArpanBiswas99/Battery-State-of-Charge-Estimation.git external/ArpanBiswas99
```
