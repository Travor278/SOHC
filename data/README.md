# data/ — 数据集 (v0.2)

## 数据策略概览

v0.2 采用**纯 NASA Plus 多子集同源训练**：

| 角色 | 数据集 | 训练 / 评估 |
|---|---|---|
| SOC 估计器 fine-tune | NASA BatteryAgingARC-FY08Q4 (B0025-B0056) | 训练 |
| SOH 估计器训练 | NASA B0005-B0018 + ARC-FY08Q4 容量退化 | 训练 |
| Mamba 世界模型 | NASA B0005-B0018 + Randomized Battery Usage 1-7 | 训练 |
| RL（SAC） | 在 Mamba env 上跑，无独立数据 | — |
| 真实驾驶定量泛化 | Zenodo 6985321 (WLTP+老化) | W5 定量 zero-shot |
| 真实电站定性外推 | Zenodo 18471156 | W5 PPT 定性展示 |
| LFP 跨化学体系泛化（可选） | HUST CSV（仓库已携带）| W5 可选展示 |

---

## 已携带（仓库内，无需下载）

### `data/HUST data/`

华中科技大学 LFP 18650 老化数据集（Hong et al., Nature Comm 2020 配套）。
80+ CSV，特征工程后的统计量。
**v0.2 用途**：可选——作为 LFP 跨化学体系泛化的定性展示，不进训练管线。

---

## 需手动下载

### `data/nasa_pcoe/B000x/` — NASA PCoE Battery (B0005-B0018)

- 来源：[NASA PCoE Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) → "Battery Data Set"
- 数据：B0005, B0006, B0007, B0018 共 4 节 NMC 18650，每节 ~150 充放电循环 + 容量退化记录
- 充电协议：CC 1.5A → 4.2V，CV 到 20mA
- 放电协议：CC 2A，截止电压 2.7/2.5/2.2/2.5V
- 环境温度：全 24°C
- 总大小：~50 MB
- 用途：**SOH 时序训练 + Mamba 世界模型主训练**
- 步骤：
  1. 进 PCoE Repository 找 "Battery Data Set"
  2. 下载 ZIP 解压
  3. 把 `B0005.mat`、`B0006.mat`、`B0007.mat`、`B0018.mat` 放到 `data/nasa_pcoe/B000x/`

### `data/nasa_pcoe/ARC-FY08Q4/` — NASA BatteryAgingARC-FY08Q4 (B0025-B0056)

- 来源：[NASA Open Data Portal - Li-ion Battery Aging Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)
- 数据：~32 节 NMC 18650
- **多温度**：4°C / 24°C / 43°C
- **多放电倍率**：1A / 2A / 4A
- 总大小：~300-500 MB
- 用途：**SOC 估计器 fine-tune（多温度多倍率覆盖）+ SOH 容量退化补充**
- 步骤：
  1. 在 NASA Open Data Portal 搜 "BatteryAgingARC-FY08Q4"
  2. 下载所有 .mat 到 `data/nasa_pcoe/ARC-FY08Q4/`
  3. 验证：每节电池有 `ambient_temperature` 字段（4 / 24 / 43）

### `data/nasa_pcoe/Randomized/` — NASA Randomized Battery Usage 1-7

- 来源：[NASA - Randomized Battery Usage](https://data.nasa.gov/dataset/randomized-and-recommissioned-battery-dataset)
- 数据：7 套电池，**动态负载**（0.5-4A 随机游走），模拟真实使用场景
- 总大小：~200-300 MB
- 用途：**Mamba 世界模型动态外推训练**（让世界模型见过非 CC-CV 的多样动作，避免 RL 探索分布外）
- 步骤：
  1. 下载 RW1-RW7 数据
  2. 解压到 `data/nasa_pcoe/Randomized/`
  3. 在 loader 里筛选稳定段（电流变化 < 1A），剔除剧烈跳变样本

### `data/zenodo_6985321/` — Offenburg WLTP+老化 (~30 MB)

- 来源：[Zenodo 6985321](https://zenodo.org/records/6985321)
- 论文：Braun et al., J. Power Sources, 2022
- 数据：2 节锂离子电池（fresh + pre-aged），WLTP 动态行驶剖面 + 长时实验，1 Hz 采样
- 用途：**W5 真实驾驶定量泛化验证**（zero-shot，不参与训练）
- 步骤：
  1. Zenodo 直接下载 ZIP
  2. 解压所有 CSV + .m 脚本到 `data/zenodo_6985321/`
  3. 注意：SOC/SOH 标签需要用配套 .m 脚本里的 OCV-SOC 表（1001 点）+ 库仑积分自行重建

### `data/zenodo_18471156/` — Zenodo 真实储能电站监测数据 (~19 MB)

- 来源：[Zenodo 18471156](https://zenodo.org/records/18471156)
- 上传者：Slam Idea（个人，2026-02-03）
- 数据：BatteryData.zip 19.2 MB（解压未知）；多节单体 V/I/T 时序；**无 SOC/SOH 标签、无 RPT、无化学体系信息、无配套论文**
- 用途：**W5 PPT 末尾 1 张图定性展示**（"在真实电站工况下方案输出曲线合理性"），**绝不进训练管线、不参与定量评估**
- 步骤：
  1. Zenodo 直接下载 BatteryData.zip
  2. 解压到 `data/zenodo_18471156/`
  3. 选其中一节电池一段时序，跑训好的 SOC/SOH 估计器 inference，画曲线

## 验证下载完成

跑一个 sanity check（待实现）：

```bash
python -m craic_pipeline.utils.verify_data
```

预期输出：
- ✅ B0005.mat / B0006.mat / B0007.mat / B0018.mat 存在
- ✅ ARC-FY08Q4 至少 ~30 节电池数据
- ✅ Randomized RW1-RW7 至少 7 套
- ✅ Zenodo 6985321 CSV + .m 脚本存在
- ✅ Zenodo 18471156 BatteryData.zip 解压后非空（仅警告，不阻塞）
