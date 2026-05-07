# data/ — 数据集

## 已携带（仓库内）

### `data/HUST data/`

华中科技大学 LFP 18650 老化数据集（Hong et al., Nature Comm 2020 配套）。
80+ CSV，特征工程后的统计量（V/I 均值/方差/峰度/偏度、CC/CV 容量等），每行对应一个充放电循环。
**用途**：W1 训练 SOH 估计器。

## 需手动下载

### `data/lg_hg2/` — LG 18650HG2（Mendeley，~3 GB）

- 来源：https://data.mendeley.com/datasets/cp3473x7xv/3
- 论文：Kollmeyer et al., 2020
- 数据：单节 NMC 18650，3 Ah，多温度（-20°C ~ 40°C），多 driving cycle（HWFET/LA92/UDDS/US06/Mixed）
- 用途：W1 训练 KeiLongW SOC 模型（实际上 KeiLongW 已提供预训练权重，本数据集主要用于本地测试集验证）
- 步骤：
  1. 注册 Mendeley 账号
  2. 下载 ZIP 解压到 `data/lg_hg2/`
  3. 验证目录结构包含 `25degC/`、`40degC/`、`n10degC/` 等温度子目录

### `data/nasa_pcoe/` — NASA PCoE Battery（~50 MB）

- 来源：https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- 数据：B0005, B0006, B0007, B0018 共 4 节 NMC 18650，每节 ~150 个充放电循环 + 容量退化记录
- 用途：**W2 训练 Mamba 世界模型 + RL 环境主数据**
- 步骤：
  1. 进 PCoE Repository 找 "Battery Data Set"
  2. 下载 `Battery_Data_Set.zip`（含 BatteryAgingARC-FY08Q4 等子集）
  3. 解压把 `B0005.mat`、`B0006.mat`、`B0007.mat`、`B0018.mat` 放到 `data/nasa_pcoe/`

### `data/zenodo_6985321/` — Offenburg WLTP+老化（~30 MB）

- 来源：https://zenodo.org/records/6985321
- 论文：Braun et al., J. Power Sources, 2022
- 数据：2 节锂离子电池（fresh + pre-aged），WLTP 动态行驶剖面 + 长时实验，1 Hz 采样
- 用途：**W4/W5 zero-shot 泛化验证**（不参与训练，仅用于"动态驾驶 + 老化"双重工况下的端到端测试）
- 步骤：
  1. 直接 Zenodo 下载 ZIP
  2. 解压所有 CSV 与 .m 脚本到 `data/zenodo_6985321/`
  3. 注意：SOC/SOH 标签需要用配套 .m 脚本里的 OCV-SOC 表（1001 点）+ 库仑积分自行重建

## 验证下载完成

跑一个简单的 sanity check（待实现）：

```bash
python -m craic_pipeline.utils.verify_data
```
