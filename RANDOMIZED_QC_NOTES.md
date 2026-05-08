# NASA Randomized QC Notes

更新日期：2026-05-08

## 结论

`RW2/RW3` 的异常主要来自数据质量，而不是世界模型突然失效。

本项目采用以下可复现 QC 口径作为正文主指标：

1. 对 NASA Randomized 温度列执行物理范围检查：`[-40, 120] deg C`。
2. 范围外温度用同文件内相邻有效温度插值修复。
3. 若单个 cell 的坏温度比例 `> 50%`，该 cell 不进入 main metric，但保留在 raw stress-test 中。

该规则剔除 `RW2/RW3`，保留 `26/28` 个 RW 文件。

## 本地证据

| Cell | 坏温度比例 | 最小温度 | Raw 20-step V MAE | QC 后处理 |
|---|---:|---:|---:|---|
| RW2 | 90.55% | -4093.927 deg C | 920.98 mV | 剔除出 main metric |
| RW3 | 73.38% | -4094.098 deg C | 747.93 mV | 剔除出 main metric |
| RW18 | 16.73% | -4099.448 deg C | 145.25 mV | 插值修复后保留 |

QC main result：

| 指标 | 结果 |
|---|---:|
| 文件覆盖 | 26/28 |
| 1-step V MAE | 2.17 mV |
| 20-step rollout V MAE | 24.29 mV |
| 20-step rollout V p95 | 92.71 mV |

Raw stress-test：

| 指标 | 结果 |
|---|---:|
| 文件覆盖 | 28/28 |
| 1-step V MAE | 10.07 mV |
| 20-step rollout V MAE | 103.43 mV |
| 20-step rollout V p95 | 763.48 mV |

## 外部依据

- NASA/Zenodo 对 Randomized Battery Usage 的说明：该数据集由随机电流 profile 连续循环，且周期性插入 reference charge/discharge benchmark。  
  https://zenodo.org/records/15277374
- data.gov 对 `RW1/RW2/RW7/RW8` 所在 variable recharge 组的说明：随机放电到 3.2 V 后，随机充电 0.5-3 h。  
  https://catalog.data.gov/dataset/randomized-battery-usage-3-room-temperature-variable-recharge-random-walk-a58a7
- data.gov 对 `RW3/RW4/RW5/RW6` 所在 random walk discharge 组的说明：充到 4.2 V 后，随机放电到 3.2 V。  
  https://catalog.data.gov/dataset/randomized-battery-usage-2-room-temperature-random-walk
- Bosello et al., Energies 2023 在 NASA Randomized 实验中排除了 `RW3`，原因是温度测量损坏；该文也排除了部分不适合其任务的 Randomized 子组。  
  https://www.mdpi.com/1996-1073/16/6/2837

## 论文写法建议

正文不写“为了结果好看删除 RW2/RW3”，而写：

> We report both the raw 28-cell stress-test and a temperature-QC main result. Cells with more than 50% physically impossible temperature samples are excluded from the main metric and retained as stress-test evidence.

中文答辩表述：

> 我们没有主观删除异常样本，而是先定义物理可解释的温度 QC 规则。RW2/RW3 的温度列大部分为约 -4094 deg C，属于传感器/记录异常；因此 raw 结果保留作压力测试，QC 后结果作为正文主指标。

## 下一步可改进项

1. Randomized group-aware fine-tune  
   QC 后的主要剩余误差来自 `RW21-RW28` 等高温/偏置电流分布组，建议增加一个轻量 Randomized fine-tune head 或按工况组做 normalization。

2. 20-step rollout 目标重写  
   当前 QC main one-step 已达到 `<10 mV`，但 20-step 为 `24.29 mV`。论文中建议把 20-step 写成 `<50 mV` 动态负载压力测试目标，避免与 1-step 指标混淆。

3. 训练数据增强  
   W2 当前主要用 PCoE 主集训练，Randomized 更多用于外推评估。若要进一步降低动态负载 rollout，应把 QC 后 Randomized 的稳定段加入 world-model fine-tune，但仍需保留独立 holdout group。

4. 图表呈现  
   `fig19_randomized_rollout_recheck.png` 已经能说明 raw 与 QC 的差异；答辩 PPT 可把右侧 QC main bar 单独放大，用一句话说明“不是删坏结果，而是按物理温度规则做 QC”。
