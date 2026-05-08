# CRAIC2026 EV 智能快充系统当前状态

更新日期：2026-05-08  
仓库：`https://github.com/Travor278/SOHC`  
当前阶段：W0-W4 主链路已跑通，W5 包级/展示扩展已形成论文初稿产物。

## 1. 系统整体框架

当前系统按 v0.2 方案实现为三层架构：

```text
NASA PCoE 数据
  ├─ B0005/B0006/B0007/B0018 主集
  ├─ ARC-FY08Q4 多温度/多倍率子集
  └─ Randomized Battery Usage 动态负载子集
        │
        ▼
W1: SOC / SOH 估计层
  ├─ SOC: KeiLongW LSTM warm-start + NASA fine-tune
  └─ SOH: NASA Capacity ratio / BatteryML-compatible baseline
        │
        ▼
W2: Mamba 世界模型 + ECM 物理安全层
  ├─ Mamba/GRU world model: 输入 [SOC, SOH, V, I, T, action]
  └─ ECM safety layer: 动作投影，保证电压硬约束
        │
        ▼
W3: SAC 强化学习快充策略
  ├─ BatteryChargingEnv(gymnasium.Env)
  ├─ reward = speed - voltage risk - temperature - aging
  └─ 输出 SAC policy
        │
        ▼
W4: CC-CV / MFCC / SAC 对比评估
  ├─ 轨迹: I(t), V(t), SOC(t), T(t)
  ├─ 指标: 到 80% 时间、ΔSOH、过压次数、平均温度
  └─ paired-vs-CCCV 核心对比表
        │
        ▼
W5: 多单体 / 包级扩展
  ├─ 6S1P Python pack prototype
  ├─ 单体策略复制: CC-CV / MFCC / SAC per cell
  └─ SOC-spread balancing coordinator
```

## 2. 数据策略

当前严格遵守 v0.2 数据策略：

- 训练主线全部使用 NASA PCoE 同源 NMC 18650 数据。
- NASA 三个子集：
  - `data/nasa_pcoe/B000x/`
  - `data/nasa_pcoe/ARC-FY08Q4/`
  - `data/nasa_pcoe/Randomized/`
- LG 18650HG2 只作为 KeiLongW SOC 预训练权重来源。
- HUST 数据仅保留为可选展示，不进训练主线。
- Zenodo 6985321 已下载并完成 zero-shot 诊断，只作跨数据集泛化分析，不进训练。
- Zenodo 18471156 已下载并完成真实电站定性展示，不进训练、不作定量 SOH 结论。

## 3. 已实现代码模块

| 模块 | 状态 | 作用 |
|---|---:|---|
| `craic_pipeline/nasa_loader.py` | 已实现 | 统一解析 PCoE / ARC-FY08Q4 / Randomized `.mat` |
| `craic_pipeline/soc_inference.py` | 已实现 | KeiLongW LSTM SOC 推断 |
| `craic_pipeline/soc_finetune.py` | 已实现主流程 | NASA fine-tune、严格 SOC 标签、cell holdout |
| `craic_pipeline/soh_train.py` | 已实现 | BatteryML-compatible SOH baseline |
| `craic_pipeline/world_model_mamba.py` | 已实现 | Mamba/GRU residual 世界模型训练与评估 |
| `craic_pipeline/ecm_safety_layer.py` | 已实现 | 二阶 ECM 安全动作投影 |
| `craic_pipeline/rl_env.py` | 已实现 | SAC 环境，含 reward 和 L3 安全约束 |
| `craic_pipeline/train_sac.py` | 已实现 | SAC 训练入口，支持 reward sweep/checkpoint |
| `craic_pipeline/eval_compare.py` | 已实现 | CC-CV / MFCC / SAC 评估与画图 |
| `craic_pipeline/pack_balance.py` | 已实现初版 | 多单体策略复制、SOC-spread 均衡、包级指标与画图 |
| `craic_pipeline/pack_dataset_upc.py` | 已实现 | UPC 36-cell pack Parquet 加载、summary、Simulink CSV 导出 |
| `craic_pipeline/eval_upc_pack.py` | 已实现 | UPC 真实数据论文图表 + Python active buck-boost 短仿真 |
| `craic_pipeline/zenodo_zero_shot.py` | 已实现 | Zenodo 6985321 fresh/aged cell zero-shot SOC/SOH 诊断 |
| `craic_pipeline/station_demo_18471156.py` | 已实现 | Zenodo 18471156 真实电站定性展示图 |
| `craic_pipeline/soh_mamba_head.py` | 已实现 | W2 Mamba embedding + Ridge head 的 SOH 架构 ablation |

## 3.1 W1 主要算法

W1 是状态估计层，目标是给后续世界模型和 RL 环境提供 `SOC` / `SOH` 软标签与推断能力。

### SOC 估计

当前 SOC 主算法是 KeiLongW 风格的 stacked LSTM：

```text
输入: 滑动窗口 V/I/T 序列
shape: (N, window, 3)
特征: [voltage, current, temperature]

网络:
LSTM(256, selu, return_sequences=True)
LSTM(256, selu, return_sequences=True)
LSTM(128, selu)
Dense(64, selu)
Dense(1, linear)

输出:
SOC ∈ [0, 1]
```

训练策略：

1. 使用 KeiLongW release 中的 LG 18650HG2 `.h5` 作为 warm-start。
2. 用 NASA ARC-FY08Q4 多温度/多倍率数据 fine-tune。
3. 默认冻结前两层 LSTM，只微调第三层 LSTM 和 Dense head。
4. SOC 标签采用 NASA discharge cycle 内的严格库仑积分构造：
   - 每个 discharge cycle 独立积分。
   - 起点强制 `SOC=1`。
   - 终点按容量/截止点校准。
   - 避免跨 cycle 滑窗污染。

当前状态：

- 推断链路已跑通。
- 当前 best B0018 holdout MAE 约 `3.48%`。
- 尚未达到 `<1.5%` 理想指标。

### SOH 估计

当前 SOH 主算法是 BatteryML-compatible 的容量比 baseline：

```text
SOH = capacity / fresh_capacity
```

实现方式：

1. `nasa_loader.py` 解析 NASA `.mat`。
2. `soh_train.py` 将 NASA cycle 转成 BatteryML 风格 `BatteryData / CycleData`。
3. 从 NASA `Capacity` 字段构造 SOH 标签。
4. 使用 Ridge / Variance-style baseline 学习 cycle-level 统计特征到 SOH 的映射。

cycle-level 特征包括：

- 电压均值、标准差、最小值、最大值。
- 电流均值、绝对电流均值。
- 温度均值、最大值。
- cycle 时间跨度。
- 容量一致性特征。

当前状态：

- NASA holdout RMSE 已满足 `<2% SOH`。
- 注意：该 SOH baseline 依赖 NASA `Capacity` 字段，适合作软标签/一致性基准，不等价于真实部署时无容量标签的 SOH 估计器。

## 3.2 W2 输入输出接口

W2 包含两个核心模块：

1. Mamba 世界模型。
2. ECM 物理安全层。

### W2 数据张量

训练数据包：

```text
outputs/world_model_train_data.pt
```

核心字段：

```text
X:      (N, L, 6)
y:      (N, 4)
traces: 连续 rollout 评估轨迹
meta:   数据来源、seq_len、SOC/SOH 标签来源等
```

其中 `L=64` 是默认历史窗口长度。

### Mamba 世界模型输入

世界模型输入是历史状态-动作序列：

```text
X[t-L:t] shape = (B, L, 6)
```

每个时间步 6 个通道：

| 通道 | 名称 | 说明 |
|---:|---|---|
| 0 | `SOC` | 当前 SOC，范围 `[0, 1]` |
| 1 | `SOH` | 当前 SOH，通常范围 `[0, 1]` |
| 2 | `V` | 端电压，单位 V |
| 3 | `I` | 当前电流，NASA/W2 口径：正值为充电 |
| 4 | `T` | 温度，单位 °C |
| 5 | `action_current` | 动作电流，单位 A，正值为充电 |

### Mamba 世界模型输出

世界模型输出下一步动力学：

```text
y_hat shape = (B, 4)
```

4 个输出通道：

| 通道 | 名称 | 说明 |
|---:|---|---|
| 0 | `SOC_next` | 下一步 SOC |
| 1 | `V_next` | 下一步端电压 |
| 2 | `T_next` | 下一步温度 |
| 3 | `delta_SOH` | 单步 SOH 损耗 |

当前实现采用 residual head：

```text
SOC_next = SOC_last + ΔSOC
V_next   = V_last   + ΔV
T_next   = T_last   + ΔT
delta_SOH = predicted aging loss
```

这样可以让模型初始接近 persistence baseline，显著降低电压预测漂移。

### ECM 安全层输入输出

ECM 安全层用于 W2/W3 的 L3 物理约束。

输入：

```text
soc: 当前 SOC
action_current: 候选动作电流
```

输出：

```text
safe_current: 投影后的安全电流
```

约束目标：

```text
V_min <= V_pred <= V_max
```

当前电流符号适配：

- NASA/W2/RL 口径：正电流表示充电。
- legacy MATLAB ECM 口径更接近正电流放电。
- 因此 RL 环境传入 ECM 前会对电流反号，ECM 投影后再转回 W2 口径。

### W2 到 W3 的接口

W3 环境每一步执行：

```text
1. SAC 输出 normalized action ∈ [-1, 1]
2. 映射到 charging current ∈ [0, I_max]
3. ECM safety layer 投影得到 safe_current
4. 将 [SOC, SOH, V, I, T, safe_current] 写入历史窗口
5. Mamba 世界模型输出 [SOC_next, V_next, T_next, delta_SOH]
6. 环境更新 state = [SOC_next, SOH - delta_SOH, V_next, safe_current, T_next]
7. 计算 reward 并返回给 SAC
```

## 4. 已有关键产物

### W1 产物

| 产物 | 路径 | 说明 |
|---|---|---|
| SOC fine-tuned 模型 | `outputs/soc_finetuned.h5` | 当前 best 来自 PCoE split，B0018 holdout MAE 约 3.48% |
| SOC metrics | `outputs/soc_finetuned.metrics.json` | SOC 训练/验证记录 |
| SOH baseline | `outputs/soh_baseline.pt` | Ridge/容量比 baseline |
| SOH metrics | `outputs/soh_baseline.metrics.json` | NASA holdout RMSE 达到 W1 指标 |

### W2 产物

| 产物 | 路径 | 说明 |
|---|---|---|
| 世界模型训练数据 | `outputs/world_model_train_data.pt` | PCoE-only tensor bundle，含 traces |
| Mamba 世界模型 | `outputs/world_model.pt` | WSL GPU/Mamba 训练产物 |
| 世界模型 metrics | `outputs/world_model.metrics.json` | 1-step V MAE 等指标 |
| Randomized 子集评估 | `outputs/world_model_randomized_subset_eval.metrics.json` | 动态负载子集外推评估 |
| Randomized 全量复核 | `outputs/world_model_randomized_full_stride64.metrics.json` | 28/28 RW，strict stride=64，20-step rollout 压力边界 |
| Randomized 复核图 | `paper_figures/fig19_randomized_rollout_recheck.png` | 逐文件误差和 RW2/RW3 异常贡献 |
| ECM 参数来源 | `MATLAB滤波算法代码——云储实时数据/.../savemat_2order.mat` | legacy MATLAB STA/MIUKF 参数源 |

### W3 产物

| 产物 | 路径 | 说明 |
|---|---|---|
| SAC policy | `outputs/sac_policy.zip` | 当前最终策略，来自 600-step horizon 训练 |
| SAC smoke policy | `outputs/sac_policy_smoke.zip` | 短链路验证产物 |
| W3 decisions | `outputs/runs/w3_rl/decisions.md` | W3 关键决策记录 |
| SAC TensorBoard | `outputs/runs/sac_reward_balanced/`、`outputs/runs/w3_horizon600/` | reward sweep 和正式训练日志 |

### W4 产物

正式输出目录：

`outputs/eval_w4_final_default/`

| 文件 | 说明 |
|---|---|
| `trajectories.csv` | SAC / CC-CV / MFCC 每步轨迹 |
| `metrics_by_episode.csv` | 每个策略每个 episode 的指标 |
| `metrics_summary.csv` | 汇总指标 |
| `paired_vs_cc_cv.csv` | 与 CC-CV 成对成功 episode 的核心对比 |
| `charging_comparison.png` | I/V/SOC/T 四联图 |

## 5. 当前关键指标

### SOC

- 当前 SOC fine-tune 已跑通。
- B0018 holdout 最好 MAE 约 `3.48%`。
- 尚未达到 TODO 中的 `<1.5%` 理想指标。
- 当前判断：瓶颈主要来自 B0018 cell-domain 差异和 NASA 标签噪声，不是简单训练轮数不足。

### SOH

- NASA holdout RMSE 满足 `<2%`。
- 但该 SOH baseline 依赖 NASA `Capacity` 字段构造 ratio，更适合作软标签和一致性验证，不代表真实部署时无容量标签能力。
- Mamba-head ablation 已完成：在 SOH 通道置中、B0005/B0006/B0007 → B0018 划分下，Mamba embedding Ridge 的 RMSE/MAE 为 `2.87%` / `1.18%`，优于 physical stats Ridge 的 `3.18%` / `1.71%`，但尚未达到 `<2% RMSE`。

### Mamba 世界模型

- B0005/B0006/B0007 -> B0018 holdout：
  - 1-step V MAE：约 `1.42 mV`
  - 20-step rollout drift：约 `8.04 mV`
- Randomized 子集：
  - 1-step V MAE：约 `2.39 mV`
  - sampled 20-step rollout V MAE：约 `7.71 mV`
- Randomized strict full recheck：
  - 覆盖 `28/28` 个 RW 文件，统一 `stride=64`，20-step rollout stride `256`。
  - one-step V MAE：`10.07 mV`
  - 20-step rollout V MAE / p95：`103.43 mV` / `763.48 mV`
  - 去除 `RW2/RW3` 后 weighted one-step/rollout V MAE：`2.26 mV` / `25.63 mV`
  - 结论：全量 Randomized 是当前世界模型的动态负载压力边界，不能写成 `<10 mV` 达标。

### SAC / W4 核心对比

正式 W4 输出使用：

- SAC policy：`outputs/sac_policy.zip`
- CC-CV baseline：3A
- horizon：800 step
- 输出目录：`outputs/eval_w4_final_default/`

在双方都充至 80% SOC 的 paired episodes 上：

| 指标 | CC-CV | SAC | 改善 |
|---|---:|---:|---:|
| 充至 80% 时间 | `596.5 s` | `411.75 s` | `+30.97%` |
| ΔSOH 单循环 | `0.001859` | `0.001536` | `-17.37%` |
| 过压次数 | `0` | `0` | 持平 |

这已经满足 W4 核心交付：

- 充电速度提升 ≥ 15%
- ΔSOH 降低 ≥ 10%
- 过压 = 0

### 包级 6S1P 初版对比

当前包级原型输出使用：

- 模块：`craic_pipeline/pack_balance.py`
- 默认包规模：`6S1P`
- 扩展烟测：`30S1P`
- 策略：CC-CV / MFCC / SAC 单体策略复制到每个 cell
- 均衡：SOC-spread active balancing trim
- horizon：1200 step
- 输出目录：`outputs/eval_pack_6s1p_h1200/`

包级停止口径比 W4 更严格：要求 **pack 内最低 SOC cell** 达到 80%。

| 指标 | CC-CV | MFCC | SAC |
|---|---:|---:|---:|
| hit_rate | `1/3` | `0/3` | `3/3` |
| 平均到目标时间 | `1121 s` | `NaN` | `699.67 s` |
| 平均 ΔSOH | `0.003147` | `0.003252` | `0.002513` |
| 末端 SOC spread | `0.02016` | `0.03787` | `0.02496` |
| 实际过压次数 | `0` | `0` | `0` |

在双方都命中的 paired episode 上，SAC vs CC-CV：

| 指标 | CC-CV | SAC | 改善 |
|---|---:|---:|---:|
| pack min-cell 到 80% 时间 | `1121 s` | `668 s` | `+40.41%` |
| 平均 ΔSOH | `0.003117` | `0.002400` | `-23.01%` |
| 末端 SOC spread | `0.04544` | `0.03272` | `-28.00%` |

注意：`raw_overvoltage_count` 统计的是世界模型未经 L3 裁剪的风险趋势；实际轨迹电压已由 ECM safety layer 投影，过压次数为 0。

30S1P 短烟测也已跑通：

- 输出目录：`outputs/eval_pack_30s1p_smoke/`
- 设置：30S1P，120 step，1 episode
- 结果：CC-CV / MFCC / SAC 三策略实际过压均为 0
- 用途：作为 `batterpack.slx` / `buck_boost_balance.slx` 的 per-cell current/SOC 轨迹接口验证，不作为最终性能对比。

### W5 泛化展示

Zenodo 6985321 已完成 zero-shot 诊断：

| 数据 | SOC MAE | SOC P95 | 半循环 SOH 中位数 |
|---|---:|---:|---:|
| fresh cell | `16.11%` | `35.13%` | `98.02%` |
| aged cell | `14.03%` | `34.01%` | `81.23%` |

结论：SOC zero-shot 误差较高，说明跨 cell 尺度 / 工况协议迁移仍是短板；SOH throughput 重建能给出接近 aged reference `83.2%` 的诊断结果，但不代表部署式无标签 SOH。

Zenodo 18471156 已完成真实储能电站定性展示：

- 数据：`600` 个 CSV 片段，字段为 `vol_1..vol_8`、`temp_1..temp_8`、`cur`。
- 选中片段：`real_world_05_06/battery_03_cells_017-024_t_04000-07999.csv`。
- 指标：电压 spread P95 `21.00 mV`，最大 `138.00 mV`，电流绝对值 P95 `135.70 A`。
- 图中 SOC 为 W1 LSTM 定性输出；SOH 因无容量标签改为 consistency proxy，不能作 SOH 定量结论。

## 6. 当前重要工程决策

### 包级可信数据

- 仓库内 `batterpack.slx` / `buck_boost_balance.slx` 来源未知，降级为可选接口演示。
- W5 包级定量验证改用 UPC 36-cell pack WLTP+CC-CV 数据集。
- 本地 `data/pack_wltp_upc/` 已下载并校验 `412/412` 文件，约 1.32GB；数据被 `.gitignore` 排除。
- `outputs/upc_pack_summary_full.csv` 已生成全量 downsample=100 概览：
  - 410 Parquet cycles
  - 295 WLTP / 115 Capacity_check
  - 3 个 cycle 含 Balancing semicycle
  - 平均 cell voltage spread 约 `69.34 mV`
  - 最大 cell voltage spread 约 `1312 mV`
- UPC 温度列存在约 650°C 级占位/异常值；当前 summary 保留 `temperature_max_raw_C`，并额外使用 `-40~120°C` 有效温度分位数做分析。
- `SIMULINK_PACK_WORKFLOW.md` 已说明已有 pack 资产时的数据回放、策略闭环和 balancing on/off paired test 流程。
- `PAPER_UPC_PACK_RESULTS.md` 已整理 UPC 包级论文结果：
  - Cycle 003 WLTP 平均 spread `244.56 mV`，P95 `510.00 mV`，最大 `590.00 mV`
  - Cycle 027 real balancing semicycle 起点 `308.00 mV`，段内最小 `127.00 mV`，终点 `308.00 mV`
  - Python active buck-boost 短仿真 `622.00 mV -> 334.00 mV`，降幅 `46.30%`
- 本地效果图输出在 `outputs/upc_pack_paper/`：
  - `fig_active_balancer_topology.png`
  - `fig_upc_measured_profile.png`
  - `fig_upc_real_balancing_semicycle.png`
  - `fig_python_balancing_short_sim.png`

### 电流符号

- NASA/W2 张量：正电流表示充电，负电流表示放电。
- legacy MATLAB ECM 参数口径更接近正电流放电。
- 因此 `BatteryChargingEnv` 对 RL/世界模型暴露 `[0, I_max]` 正充电电流，传给 ECM safety layer 时内部反号。

### L3 安全层

- ECM safety layer 是硬安全约束。
- 世界模型 raw voltage 会进入 reward 的电压风险惩罚。
- 对外 observation 中的 voltage 使用 L3-clipped 电压，保证策略轨迹物理安全。

### Aging reward

当前 reward aging 来自两部分：

- 世界模型输出的 `delta_SOH`
- 物理 proxy：
  - 电流应力
  - raw high-voltage stress
  - 温度应力
  - calendar aging floor

最终 W3 policy 训练参数：

```text
total_steps = 60000
max_steps = 600
buffer_size = 20000
batch_size = 64
speed = 30
voltage = 300
temperature = 0.02
aging = 120
```

## 7. 已知问题

1. SOC `<1.5% MAE` 尚未达到。
   - 当前 best 约 `3.48%`。
   - 论文图像可用，但若要严苛指标，需要继续做标签清洗或改模型。

2. Randomized 全量动态负载是当前 W2 的主要短板。
   - 严格 `stride=64` 已缓存并评估 28/28 个 RW 文件。
   - 全量 20-step rollout V MAE 为 `103.43 mV`，主要由 `RW2/RW3` 拉高。
   - 去除 `RW2/RW3` 后为 `25.63 mV`，仍应作为压力边界而非最终达标指标。

3. W4 的 paired-vs-CCCV 口径需要在论文/答辩中说明。
   - 因为随机初始 SOC 下，CC-CV/MFCC 在固定 horizon 内并非每次都能到 80%。
   - 核心百分比使用“同初始条件且双方都到 80%”的 paired episodes。
   - 全量 summary 同时保留 hit_rate 和 soc_end_mean。

4. BatteryML Mamba-head SOH 已完成轻量 ablation，但仍需加强。
   - 当前是 BatteryML-compatible 目标 + W2 Mamba embedding + Ridge head。
   - 尚未接入 external BatteryML trainer，也未达到 `<2% RMSE`。

5. Zenodo 6985321 zero-shot 已完成但 SOC 精度不足。
   - fresh/aged SOC MAE 分别为 `16.11%` / `14.03%`。
   - 可作为跨尺度迁移瓶颈诊断，不宜写成泛化达标结果。

6. 包级原型仍是 Python supervisory simulator。
   - 当前没有直接求解 buck-boost 开关电路。
   - 30S1P 短烟测已完成。
   - 仓库内 `batterpack.slx` / `buck_boost_balance.slx` 来源未知，后续只作可选接口演示，不作为论文定量依据。
   - 包级定量验证改用可信公开数据集，详见 `PACK_DATASETS.md`。

## 8. 下一步建议

建议下一步优先顺序：

### A. Randomized 动态负载指标复核

1. 已统一 `SYSTEM_STATUS.md` / `TODO.md` / metrics JSON 中 Randomized 20-step rollout 口径。
2. 已完成统一 `stride=64` 的 28/28 文件复核。
3. 已输出 `paper_figures/fig19_randomized_rollout_recheck.png`。正文建议写成“动态负载压力测试暴露 RW2/RW3 域差异”，不写成 Randomized 全量达标。

### B. 继续加强 SOH Mamba-head

1. 当前轻量对比表已完成，见 `SOH_MAMBA_HEAD_RESULTS.md`。
2. 下一步可尝试：
   - 更严格的 cycle-level split / cell-level feature aggregation。
   - 用 shallow MLP/Huber loss 替代 Ridge head。
   - 真正接入 external BatteryML trainer。
3. 目标是把 Mamba-head 从“有增益的 ablation”推进到 `<2% RMSE` 或更有说服力的架构对比。

### C. 补泛化与展示的可信边界

1. Zenodo 6985321 已完成，正文只写“zero-shot 诊断”，不写“泛化达标”。
2. Zenodo 18471156 已完成，正文只写“真实电站定性展示”，不写“SOH 定量准确”。
3. 若要更强泛化结论，需要补有标签容量校准或 BMS SOC/SOH 真值。

### D. 准备答辩图表包

优先整理以下图：

1. 三层架构图。
2. W2 世界模型预测 vs 真实曲线。
3. W4 `charging_comparison.png`。
4. W4 paired-vs-CCCV 指标表。
5. W5 `pack_comparison.png`。
6. Zenodo 6985321 zero-shot 曲线。
7. Zenodo 18471156 真实电站定性图。
8. Simulink pack validation workflow 图。

### E. 再考虑 SOC 精度攻坚

如果时间允许，再回头攻 SOC `<1.5%`：

1. 进一步按 cell/cycle 清洗异常 discharge 段。
2. 引入更可靠的物理 SOC label smoothing。
3. 尝试 TCN/Transformer/Mamba SOC head，而不是只用 KeiLongW LSTM。

## 9. 推荐立即执行项

最建议现在做：

```text
针对 RW2/RW3 做 Randomized fine-tune，并加强 BatteryML/Mamba-head SOH 对比。
```

原因：

- 主链路、包级 UPC、Zenodo 展示图都已具备。
- Randomized 指标口径已统一；新的问题是 28-file strict full 暴露 `RW2/RW3` 域差异。
- Mamba-head SOH 轻量版已补齐，但仍可继续优化到更像正式 BatteryML trainer 的实验。
