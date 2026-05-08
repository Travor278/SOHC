# Pack-Level Dataset Candidates

更新日期：2026-05-08

本文件记录 W5 包级扩展的可信数据来源选择。原则是：不把仓库内未知来源的 Simulink 30 模组资产作为论文定量依据；优先使用带 DOI、论文说明、公开仓库或国家/高校数据门户的数据集。

## 推荐主线：UPC 36-Cell Pack WLTP + CC-CV Dataset

**结论：最适合作为包级均衡/快充验证的主数据集。**

- 名称：Lithium-ion battery pack cycling dataset with CC-CV charging and WLTP/constant discharge profiles
- 论文：Scientific Data 12, Article 1942 (2025)
- 论文 DOI：https://doi.org/10.1038/s41597-025-06229-5
- 数据 DOI：https://doi.org/10.34810/DATA2395
- 数据门户：https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi:10.34810/DATA2395
- 许可：CC BY 4.0
- 数据大小：约 1.3 GB，412 个文件，主要为 Apache Parquet

### 为什么可信

- 数据由 Universitat Politecnica de Catalunya 研究团队发布，并有 Scientific Data 数据论文。
- 数据门户提供 DOI、版本、作者 ORCID、机构、资助信息、MD5 和文件级元数据。
- 测试台包含商业汽车 BMS、CAN 通信、校准温度传感器、分支电流传感器和受控温箱。

### 和本项目的匹配点

- 36 个 Panasonic NCR18650B cell，三组并联支路，每支路 12 串，可抽象为 `12S3P`。
- 每个 cycle 文件包含 WLTP 动态放电与 CC-CV 充电，适合验证快充策略和动态负载外推。
- 字段覆盖：
  - 36 个单体电压
  - 3 个支路电流
  - pack 电压/电阻
  - 72 个 cell 表面温度
  - BMS SOC
  - balancing semicycle
- 论文明确说明：当 cell/branch 电压不平衡超过 100 mV 时触发 balancing semicycle。

### 建议用途

- W5 主定量验证：
  - cell voltage spread
  - min-cell 到 80% SOC 时间
  - pack / branch current 约束
  - balancing 前后电压/SOC spread
  - 真实 WLTP 动态工况下的策略稳健性
- 不用于 W1-W4 主训练，避免破坏 v0.2 的 NASA 同源训练策略。

### 建议落地路径

目标目录：

```text
data/pack_wltp_upc/
```

初步 adapter：

```text
craic_pipeline/pack_dataset_upc.py
```

统一输出接口：

```text
time_s: (N,)
cell_voltage_V: (N, 36)
branch_current_A: (N, 3)
cell_temperature_C: (N, 36 or 72)
pack_voltage_V: (N,)
bms_soc: (N,)
semicycle: list[str]
cycle_id: int
```

第一阶段只下载少量 Parquet 文件做 smoke，例如：

```text
Qtzl_Cycle_001_Capacity_check_partial_data.parquet
Qtzl_Cycle_003_WLTP_partial_data.parquet
Qtzl_Cycle_010_WLTP_partial_data.parquet
```

## 推荐补充：BattGP 8S LFP Field Dataset

**结论：适合作为“真实服役 pack 异常/弱单体分析”补充，不适合作快充主验证。**

- 名称：Lithium-Ion Battery Field Data: 28 LFP battery systems with 8 cells in series, up to 5 years of operation
- 数据 DOI：https://doi.org/10.5281/zenodo.13715694
- 数据门户：https://zenodo.org/records/13715694
- 论文：Cell Reports Physical Science, 2024
- 论文 DOI：https://doi.org/10.1016/j.xcrp.2024.102258
- 代码：https://github.com/JoachimSchaeffer/BattGP
- 数据大小：`field_data.zip` 约 1.7 GB
- 许可：CC BY-NC 4.0

### 为什么可信

- TU Darmstadt / MIT 团队发布，有 Cell Reports Physical Science 论文和配套 BattGP Python 包。
- 数据门户说明 28 个 24 V LFP battery systems，每个系统 8 个串联 prismatic cells。
- 总量约 133M rows，覆盖 1 个月到 5 年真实使用。

### 和本项目的匹配点

- 有 cell-level voltage、pack current、temperature、active cell balancing。
- 适合展示“弱单体/不均衡/故障概率”这类包级健康监测问题。

### 局限

- LFP、160 Ah prismatic cell，与本项目 NASA NMC 18650 主线化学体系不同。
- 所有系统来自返修/异常样本，数据集本身有偏，不能代表正常 EV 快充。
- 用途应限定为 W5 field-data 定性/补充验证。

## 备选：DOE / INL AVTA Pack Testing

**结论：可信但不优先。**

- 页面：https://www.energy.gov/eere/vehicles/avta-battery-testing-data
- 相关 pack testing DOI：https://doi.org/10.15483/1876633
- 优点：DOE/INL，车辆 pack 级测试，流程可信。
- 局限：更偏整车/pack 性能测试报告，通常缺少完整 cell-level voltage/time-series，不适合作均衡算法主验证。

## 不作为主依据的数据

### 仓库自带 Simulink 30 模组资产

- 路径：
  - `batterpack.slx`
  - `buck_boost_balance.slx`
  - `Rebattery_Modeling-master/`
- 处理方式：只保留为可选接口演示或电路拓扑参考。
- 不作为论文定量数据来源，因为来源、参数标定和实验依据不清晰。

### Zenodo 18471156

- 处理方式：保留 W5 末尾定性展示。
- 不作为训练或定量验证，因为无 SOC/SOH 标签、无 RPT、无化学体系与配套论文。

## 当前决策

1. 论文主包级验证：使用 UPC 36-cell pack dataset。
2. 工程接口：保留 Python `pack_balance.py` 的 `6S1P` / `30S1P` simulator。
3. Simulink：降级为可选演示，不承担可信定量指标。
4. field-data 补充：视时间下载 BattGP，做弱单体/电压 spread 定性图。

## 当前本地状态

- `scripts/download_upc_pack.py` 已实现 Dataverse API 下载、断点跳过、MD5 校验。
- UPC DATA2395 已在本机下载完成：`412/412` 文件，约 1.32 GB。
- `craic_pipeline/pack_dataset_upc.py` 已实现：
  - 单个 Parquet cycle 加载
  - 目录流式 summary
  - `12S3P` 原生数组输出
  - 36-cell flatten 输出
  - Simulink 友好宽 CSV 导出
- 全量 summary 输出：`outputs/upc_pack_summary_full.csv`
  - 410 Parquet cycles
  - 295 WLTP / 115 Capacity_check
  - 3 个 cycle 含 Balancing semicycle
  - 平均 cell voltage spread 约 69.34 mV
  - 最大 cell voltage spread 约 1312 mV
- 注意：部分温度列存在约 650°C 级占位/异常值；分析时使用 `temperature_median_valid_C`、`temperature_p95_valid_C` 和 `temperature_valid_fraction`，不直接使用 raw max 做热安全结论。
