# CRAIC2026 EV 智能快充系统当前状态

更新日期：2026-05-08  
仓库：`https://github.com/Travor278/SOHC`  
当前阶段：W0-W4 主链路已跑通，W5/展示扩展待推进。

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
- Zenodo 6985321 已下载，可作为 W5 定量泛化输入。
- Zenodo 18471156 仍未下载，仅计划用于 W5 末尾定性展示，不进训练。

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

### Mamba 世界模型

- B0005/B0006/B0007 -> B0018 holdout：
  - 1-step V MAE：约 `1.42 mV`
  - 20-step rollout drift：约 `8.04 mV`
- Randomized 子集：
  - 1-step V MAE：约 `2.39 mV`
  - sampled 20-step rollout V MAE：约 `7.71 mV`
- 注意：Randomized 是 6 个最小文件子集，不是 28 个 `.mat` 全量。

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

## 6. 当前重要工程决策

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

2. Randomized 全量评估尚未完成。
   - 当前是 6 文件子集。
   - 全量 28 文件建议继续分 shard 缓存长跑。

3. W4 的 paired-vs-CCCV 口径需要在论文/答辩中说明。
   - 因为随机初始 SOC 下，CC-CV/MFCC 在固定 horizon 内并非每次都能到 80%。
   - 核心百分比使用“同初始条件且双方都到 80%”的 paired episodes。
   - 全量 summary 同时保留 hit_rate 和 soc_end_mean。

4. BatteryML Mamba-head SOH 尚未做。
   - 这是 W4 剩余的架构创新点展示项。

5. Zenodo 6985321 zero-shot 尚未完成。
   - 数据和 OCV-SOC 曲线已在 `data/zenodo_6985321/`。
   - 需要用 OCV-SOC 表 + 库仑积分重建参考 SOC/SOH，再跑 zero-shot 曲线。

## 8. 下一步建议

建议下一步优先顺序：

### A. 补 W4 剩余创新点

1. 在 BatteryML-compatible SOH 流程里接 Mamba world-model features/head。
2. 输出一张 SOH 对比表：
   - capacity-ratio baseline
   - variance/Ridge baseline
   - Mamba feature/head variant
3. 目标不是重新大幅提升 RMSE，而是形成“BatteryML 内挂 Mamba head”的架构创新证据。

### B. 做 Zenodo 6985321 zero-shot

1. 读取：
   - `Experimental_data_fresh_cell.csv`
   - `Experimental_data_aged_cell.csv`
   - `OCV_vs_SOC_curve.csv`
2. 用 OCV-SOC 表和库仑积分重建参考 SOC。
3. 跑当前 SOC/SOH inference。
4. 输出：
   - `outputs/zenodo_6985321_zero_shot_metrics.csv`
   - `outputs/figures/zenodo_6985321_soc_soh.png`

### C. 准备答辩图表包

优先整理以下图：

1. 三层架构图。
2. W2 世界模型预测 vs 真实曲线。
3. W4 `charging_comparison.png`。
4. W4 paired-vs-CCCV 指标表。
5. Zenodo 6985321 zero-shot 曲线。

### D. 再考虑 SOC 精度攻坚

如果时间允许，再回头攻 SOC `<1.5%`：

1. 进一步按 cell/cycle 清洗异常 discharge 段。
2. 引入更可靠的物理 SOC label smoothing。
3. 尝试 TCN/Transformer/Mamba SOC head，而不是只用 KeiLongW LSTM。

## 9. 推荐立即执行项

最建议现在做：

```text
先做 Zenodo 6985321 zero-shot + 图表输出。
```

原因：

- 数据已经下载。
- 对 W5/PPT 价值高。
- 不会破坏现有 W3/W4 主链路。
- 可以快速补上“泛化验证”这一块答辩材料。
