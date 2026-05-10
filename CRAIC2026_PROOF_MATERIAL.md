# CRAIC2026 方向三项目证明材料

项目：面向电动汽车智能快充的 Mamba 世界模型、SAC 强化学习与 ECM 物理安全层协同决策系统  
仓库：<https://github.com/Travor278/SOHC>  
整理日期：2026-05-10  
对应长版草稿：[CRAIC2026_REPORT_DRAFT.md](CRAIC2026_REPORT_DRAFT.md)

> 本材料用于比赛提交、答辩附件或项目验收说明。内容仅保留关键证据、关键图片、指标表和伪代码。真实代码见 `craic_pipeline/`；本文中的算法均以伪代码表达，避免把工程实现细节误写成论文正文。

## 1. 项目证明目标

本项目证明一条端到端智能快充决策链路已经跑通：

```text
NASA 同源电池数据
  -> SOC/SOH 状态估计
  -> Mamba 世界模型预测电池短期动力学
  -> SAC 强化学习输出动态快充电流
  -> ECM 物理安全层投影电流，保证电压边界
  -> CC-CV / MFCC / SAC 对比评估
  -> 多单体 supervisory prototype 与 UPC 真实 pack 数据展示
```

核心结论按当前证据应限定为：

- 单体闭环中，SAC 相比 3A CC-CV 在 paired episodes 上充电时间缩短 `30.97%`，代理 ΔSOH 降低 `17.37%`，实际过压次数为 `0`。
- Mamba 世界模型在 B0018 holdout 上短期电压预测有效：1-step V MAE `1.42 mV`，20-step V MAE `8.04 mV`。
- 600-step replay 漂移明显：V MAE `170.56 mV`，因此世界模型只作为短期滚动决策环境，不声称替代真实电池 plant。
- 当前 ΔSOH 改善由物理应力代理项主导，Mamba 老化头尚未形成独立寿命预测贡献。
- 多单体结果是 independent-cell supervisory prototype，不是串联包 KCL/KVL 电路级仿真。

![图 0 系统与关键结果总览](paper_figures/fig00_graphical_abstract.png)

图 0 说明：该图汇总了 W1-W5 的证据链、主要指标和当前边界。最重要的是同时展示“已完成结果”和“未完成边界”，避免夸大。

## 2. 代码与数据来源证明

| 模块 | 代码来源 | 数据来源 | 证明边界 |
|---|---|---|---|
| SOC 估计器 | 结构和 warm-start 权重参考 `KeiLongW/battery-state-estimation`；本仓库实现 `soc_inference.py`、`soc_finetune.py` | LG 18650HG2 仅作为 `.h5` 权重来源；fine-tune/验证使用 NASA ARC-FY08Q4 + B0005/B0006/B0007/B0018 | 当前 B0018 holdout MAE `3.48%`，未达 `<1.5%` |
| SOH 估计器 | BatteryML-compatible 接口；本仓库实现 `soh_train.py` | NASA B0005/B0006/B0007/B0018 `Capacity` 字段 | 容量比 baseline，可作软标签/一致性基准，不代表无容量标签部署 |
| SOH Mamba-head | 本仓库实现 `soh_mamba_head.py` | NASA B0005/B0006/B0007 -> B0018 | Mamba embedding Ridge RMSE `2.87%`，是架构补强证据 |
| Mamba 世界模型 | 本仓库实现 `world_model_mamba.py`；底层依赖 `mamba-ssm` | PCoE B0005/B0006/B0007 训练，B0018 holdout；Randomized 用于动态负载复核 | 短期动力学有效，600-step 漂移较大 |
| SAC 策略 | 本仓库实现 `rl_env.py`、`train_sac.py`；算法库 `stable-baselines3` | 在 W2 世界模型 + ECM 安全环境中交互训练 | 策略结果依赖短期滚动模型和 ECM 硬安全层 |
| ECM 安全层 | 本仓库 Python 重写；参数来自仓库内 MATLAB STA/MIUKF 资产 | `savemat_2order.mat` | 不是 NASA 重新标定 ECM；已加入一阶 SOH-R0 保守修正 |
| 多单体与 pack 展示 | 本仓库实现 `pack_balance.py`、`pack_dataset_upc.py`、`eval_upc_pack.py` | UPC 36-cell pack WLTP+CC-CV；Zenodo 展示数据 | Python 多单体不是真实串联包电路仿真 |

![图 1 三层智能快充决策框架](paper_figures/fig01_system_architecture.png)

图 1 说明：系统由状态估计层、Mamba 世界模型、SAC 策略和 ECM 安全层组成。ECM 是动作执行前的硬安全投影，不依赖策略网络自行学会保守。

![图 2 数据来源与用途划分](paper_figures/fig02_data_flow.png)

图 2 说明：训练主线只使用 NASA PCoE 同源 NMC 18650 数据；LG 仅用于 SOC warm-start 权重；UPC 和 Zenodo 只用于展示/边界诊断。

## 3. W1 证明：SOC 与 SOH 状态估计

### 3.1 SOC 估计伪代码

输入：NASA cycle 中的电压 `V`、电流 `I`、温度 `T`、时间 `t`、容量 `Capacity`。  
输出：SOC 预测模型 `soc_finetuned.h5` 与 B0018 holdout 误差。

```text
Algorithm 1: Strict-cycle SOC label reconstruction and LSTM fine-tune

Load KeiLongW stacked-LSTM architecture
Load LG 18650HG2 pretrained .h5 weights as warm-start

For each NASA discharge cycle:
    Sort samples by time
    Compute discharge throughput by Coulomb counting
    Set SOC at cycle start = 1.0
    Calibrate cycle endpoint by measured Capacity / cutoff point
    Reject windows crossing cycle boundary

Build training windows:
    X = sliding_window([V, I, T], length=L)
    y = strict_cycle_SOC_at_window_end

Train schedule:
    Stage 1: freeze first two LSTM layers, train last LSTM + Dense head
    Stage 2: optionally unfreeze later layers with lower learning rate
    Select best checkpoint by B0018 holdout MAE

Return best SOC model and metrics
```

关键结果：

| 项目 | 数据划分 | 指标 |
|---|---|---:|
| SOC LSTM | B0005/B0006/B0007 -> B0018 | MAE `3.48%` |

![图 3 B0018 holdout SOC 预测曲线](paper_figures/fig08_soc_b0018_prediction.png)

图 3 说明：SOC 曲线整体可跟随 B0018 discharge cycle 趋势，已经可作为 W2/W3 软标签工程基线；但低 SOC 区和跨 cell 迁移仍有误差，因此不声称达到 `<1.5%`。

### 3.2 SOH 估计伪代码

输入：NASA cycle-level `Capacity` 字段和 V/I/T 统计特征。  
输出：SOH baseline 与容量退化一致性验证。

```text
Algorithm 2: BatteryML-compatible SOH baseline

For each NASA cell:
    fresh_capacity = first valid Capacity
    For each cycle:
        SOH_label = Capacity / fresh_capacity
        features = [
            mean(V), std(V), min(V), max(V),
            mean(I), mean(abs(I)),
            mean(T), max(T),
            cycle_duration,
            capacity_consistency_features
        ]

Split cells into train/holdout
Fit lightweight Ridge / variance-style baseline
Evaluate RMSE and MAE on holdout
Save soh_baseline.pt and metrics
```

关键结果：

| SOH 方法 | 输入特征 | 验证划分 | Val RMSE | Val MAE | 说明 |
|---|---|---|---:|---:|---|
| Capacity-ratio oracle | NASA `Capacity` ratio | NASA holdout | 约 0 | 未记录 | 软标签 oracle |
| Physical stats Ridge | SOC/V/I/T/action 统计量 | B0018 | `3.18%` | `1.71%` | 无 Mamba 表征 |
| Mamba embedding Ridge | W2 frozen Mamba embedding | B0018 | `2.87%` | `1.18%` | 有表征增益 |

![图 4 NASA SOH baseline 验证结果](paper_figures/fig07_soh_baseline.png)

图 4 说明：SOH baseline 可提供容量退化状态变量和软标签一致性证据。它依赖 NASA `Capacity` 字段，不等价于真实部署时无容量标签的在线 SOH 估计器。

## 4. W2 证明：Mamba 世界模型与 ECM 安全层

### 4.1 Mamba 世界模型伪代码

输入：历史窗口 `[SOC, SOH, V, I, T, action_current]`。  
输出：下一步 `[SOC_next, V_next, T_next, delta_SOH]`。

```text
Algorithm 3: Residual Mamba world model

Build W2 tensor bundle from NASA traces:
    For each cycle:
        SOC = strict SOC label or W1 SOC inference
        SOH = capacity_ratio soft label
        action_current = measured current under NASA sign convention
        X_t = [SOC_t, SOH_t, V_t, I_t, T_t, action_current_t]
        y_t = [SOC_{t+1}, V_{t+1}, T_{t+1}, SOH_t - SOH_{t+1}]

Model:
    z = InputProjection(X_window)
    z = MambaBlocks(z) or GRU fallback
    h = LastToken(z)
    residual = DenseHead(h)

Residual output:
    SOC_next = SOC_last + residual_SOC
    V_next   = V_last   + residual_V
    T_next   = T_last   + residual_T
    delta_SOH = positive_part(residual_SOH)

Train:
    Minimize MSE(prediction, target)
    Validate by NASA cell holdout and multi-step rollout
```

关键指标：

| 指标 | 结果 | 解释 |
|---|---:|---|
| B0018 1-step V MAE | `1.42 mV` | 一步电压预测精度高 |
| B0018 20-step V MAE / p95 | `8.04 / 22.03 mV` | 短期 rolling dynamics 可用 |
| B0018 100-step V MAE / p95 | `22.36 / 85.83 mV` | 开始出现可见漂移 |
| B0018 600-step V MAE / p95 | `170.56 / 550.20 mV` | 不适合作长时间开环 plant 替代 |
| Randomized QC 1-step V MAE | `2.17 mV` | 26/28 文件，温度 QC 后 |
| Randomized QC 20-step V MAE / p95 | `24.29 / 92.71 mV` | 动态负载边界仍存在 |

![图 5 B0018 20-step Mamba 世界模型开环预测](paper_figures/fig03_world_model_rollout.png)

图 5 说明：20-step 误差仍在 mV 到十几 mV 级，是 W3 SAC 短期滚动环境可用的关键证据。

![补充图 20 B0018 世界模型长 horizon replay 漂移](paper_figures/fig20_closed_loop_replay.png)

图 20 说明：100/600-step replay 证明长 horizon 会明显漂移。因此本项目将世界模型定位为短期滚动决策模型，而不是真实电池 plant 的长开环替代。

![补充图 19 Randomized 全量 20-step rollout 复核](paper_figures/fig19_randomized_rollout_recheck.png)

图 19 说明：Randomized raw 28-file stress-test 暴露出异常文件和动态负载边界；温度 QC 后主指标更稳定，但 20-step 仍未达 `<10 mV`。

### 4.2 ECM 安全层伪代码

输入：SOC、SOH、候选电流 `action_current`。  
输出：满足电压边界的 `safe_current`。

```text
Algorithm 4: SOH-aware ECM current projection

Given ECM parameters R0, R1, R2, C1, C2 and OCV(SOC)

For each candidate charging action:
    Convert RL current sign to ECM sign
    R0_eff = R0 * (2 - clip(SOH, 0.5, 1.0))

    Predict next polarization:
        V1_next = f_RC(V1, current, R1, C1)
        V2_next = f_RC(V2, current, R2, C2)

    Predict terminal voltage:
        V_pred = OCV(SOC) - current * R0_eff - V1_next - V2_next

    If V_pred > V_max:
        Solve linear ECM equation for current at V_max
    Else if V_pred < V_min:
        Solve linear ECM equation for current at V_min
    Else:
        Keep current

    Update ECM polarization state
    Return safe_current under RL sign convention
```

关键结果：

| 项目 | 结果 |
|---|---:|
| 与 MATLAB 二阶 RC 参考公式误差 | `< 1 mV` |
| 随机 1000 条动作投影通过率 | `100%` |

![图 6 高 SOC 下 ECM 安全动作投影效果](paper_figures/fig05_ecm_safety_projection.png)

图 6 说明：ECM 安全层把潜在越压动作投影到安全电流区间，是项目中最直接的物理安全证明。

## 5. W3/W4 证明：SAC 快充策略优于 CC-CV/MFCC

### 5.1 SAC 环境和奖励伪代码

```text
Algorithm 5: One step in BatteryChargingEnv

Input:
    observation = [SOC, SOH, V, I, T]
    action_norm in [-1, 1]

Map action:
    requested_current = scale(action_norm, 0, I_max)

Safety projection:
    safe_current = ECM_Project(SOC, SOH, requested_current)

World-model transition:
    prediction = Mamba(history + safe_current)
    model_delta_soh = max(prediction.delta_SOH, 0)
    aging_proxy_delta_soh = stress_proxy(safe_current, raw_voltage, temperature)
    delta_soh = max(model_delta_soh, aging_proxy_delta_soh)

Update state:
    SOC_next = clip(prediction.SOC_next)
    SOH_next = SOH - delta_soh
    V_next   = clip(prediction.V_next, V_min, V_max)
    T_next   = prediction.T_next

Reward:
    reward =
        30  * delta_SOC
        -300 * voltage_risk(raw_voltage)
        -0.02 * temperature_risk(T_next)
        -120 * delta_soh

Return next_state, reward, done, diagnostics
```

温度风险采用 soft-hard 形式：

```text
If T <= 40 deg C:
    temperature_risk = 0
Else if T <= T_max:
    temperature_risk = ((T - 40) / (T_max - 40))^2
Else:
    temperature_risk = 10 + 5 * (T - T_max)
```

老化代理项：

```text
aging_proxy_delta_soh =
    calendar_floor
    + k * (
        current_stress
        + high_voltage_stress
        + temperature_stress
      )
```

当前评估显示 `model_delta_soh = 0`，所以 ΔSOH 改善来自物理应力 proxy，而非 Mamba 老化头。

### 5.2 单体策略对比证据

评估设置：

- 策略：SAC、3A CC-CV、MFCC。
- horizon：800 step。
- 核心百分比：只统计同初始条件下双方均达到 80% SOC 的 paired episodes。
- 安全判据：实际过压次数为 0。

| 指标 | CC-CV | SAC | 改善 |
|---|---:|---:|---:|
| 充至 80% 时间 | `596.5 s` | `411.75 s` | `+30.97%` |
| ΔSOH 单循环 | `0.001859` | `0.001536` | `-17.37%` |
| 实际过压次数 | `0` | `0` | 持平 |

![图 8 单体 paired episodes 核心指标对比](paper_figures/fig04_w4_metrics_bar.png)

图 8 说明：SAC 在 paired episodes 上同时提升速度并降低代理 ΔSOH，满足 W4 核心交付目标。

![图 9 单体 SAC、CC-CV 与 MFCC 充电轨迹对比](paper_figures/fig09_charging_comparison_polished.png)

图 9 说明：SAC 不是简单拉高恒流，而是在接近安全边界时形成动态电流曲线；ECM 层保证实际电压不越界。

### 5.3 评估与老化分解伪代码

```text
Algorithm 6: Paired evaluation and aging-source decomposition

For each evaluation seed:
    Reset identical initial state for all strategies
    Run CC-CV, MFCC, SAC until target/horizon/end condition
    Record per-step:
        SOC, SOH, V, T, current
        model_delta_soh
        aging_proxy_delta_soh
        overvoltage flag

For each strategy:
    Compute:
        hit_rate
        time_to_80
        delta_soh
        overvoltage_count
        mean/max temperature

For paired comparison:
    Select episodes where both CC-CV and SAC hit target
    Report mean improvement in time and delta_soh

For aging decomposition:
    Count steps where model_delta_soh >= aging_proxy_delta_soh
    Count steps where aging_proxy_delta_soh > model_delta_soh
    Report dominant source ratio
```

老化分解结果：

| 策略 | Mamba ΔSOH 总和 | Proxy ΔSOH 总和 | Proxy 主导步数 |
|---|---:|---:|---:|
| CC-CV | `0.000000` | `0.020191` | `100%` |
| MFCC | `0.000000` | `0.023817` | `100%` |
| SAC | `0.000000` | `0.019842` | `100%` |

结论：当前不能说“Mamba 已经准确预测老化收益”；应表述为“SAC 在物理应力代理老化项约束下实现更温和快充”。

## 6. W5 证明：多单体扩展与真实 pack 数据展示

### 6.1 多单体 supervisory prototype 伪代码

```text
Algorithm 7: Independent-cell supervisory strategy replication

Initialize N independent cell environments:
    Each cell has [SOC, SOH, V, I, T]
    Each cell runs the same W2 world model and ECM projector

For each time step:
    For each cell:
        base_current_i = strategy(cell_state_i)

    Compute SOC spread:
        mean_soc = mean(SOC_i)

    Supervisory trim:
        trim_i = gain * (mean_soc - SOC_i)
        trim_i = clip(trim_i, -max_balance_current, +max_balance_current)
        current_i = clip(base_current_i + trim_i, 0, I_max)

    Step each independent cell with current_i
    Stop when min(SOC_i) >= 0.8 or safety/horizon reached

Return per-cell trajectories and spread metrics
```

边界说明：该原型允许 per-cell current 独立变化，因此不是物理 6S1P/30S1P 串联包仿真；它只证明策略复制和 SOC-spread 协调逻辑。

关键结果：

| 指标 | CC-CV | MFCC | SAC |
|---|---:|---:|---:|
| hit_rate | `1/3` | `0/3` | `3/3` |
| 平均到目标时间 | `1121 s` | NaN | `699.67 s` |
| 平均 ΔSOH | `0.003147` | `0.003252` | `0.002513` |
| 末端 SOC spread | `0.02016` | `0.03787` | `0.02496` |
| 实际过压次数 | `0` | `0` | `0` |

![图 10 6-cell 多单体策略复制与均衡协调结果](paper_figures/fig10_pack_comparison_polished.png)

图 10 说明：停止条件为最低 SOC cell 到 80%，比单体条件更严格。结果证明 SAC 策略可复制到多个 cell 的 supervisory prototype，但不替代真实串联包 plant。

### 6.2 UPC 真实 pack 数据与主动均衡短仿真

UPC 数据集包含：

- 真实 12S3P pack。
- 36 个 cell voltage。
- 3 个 branch current。
- 72 个 cell temperature。
- BMS SOC 和 balancing semicycle 标记。

```text
Algorithm 8: UPC pack spread analysis and active-balancing short simulation

Load UPC cycle parquet
Extract:
    cell_voltage[time, 36]
    branch_current[time, 3]
    temperature[time, 36]
    BMS_SOC[time]

Compute measured metrics:
    voltage_spread = max(cell_voltage) - min(cell_voltage)
    spread_mean, spread_p95, spread_max
    balancing_semicycle segments

For synthetic active-balancing test:
    Select high-spread initial cell voltages
    Case A: balancing off
    Case B: active buck-boost supervisory current on
    Simulate spread change for 30 minutes
    Compare final spread and max balance current
```

关键结果：

| Case | 初始 spread | 终点 spread | 降幅 | 最大均衡电流 |
|---|---:|---:|---:|---:|
| balancing off | `622.00 mV` | `622.00 mV` | `0.00%` | `0.00 A` |
| active buck-boost | `622.00 mV` | `334.00 mV` | `46.30%` | `0.80 A` |

![图 11 本项目主动均衡拓扑示意](paper_figures/fig11_active_balancer_topology.png)

图 11 说明：该图是本项目重画的 active buck-boost supervisory topology，不复用第三方未授权截图。

![图 12 UPC 36-cell pack 实测 WLTP 工况](paper_figures/fig12_upc_measured_profile.png)

图 12 说明：UPC 实测数据证明真实动态工况下 pack 内 voltage spread 可以达到数百 mV，包级均衡问题真实存在。

![图 14 基于 UPC 高 spread 初值的 Python active buck-boost 短仿真](paper_figures/fig14_python_balancing_short_sim.png)

图 14 说明：在相同高 spread 初值下，active buck-boost supervisory law 可在短仿真中降低 spread；但该结果仍需 Simulink/Simscape 电路级验证开关损耗和器件约束。

## 7. 泛化展示与真实电站接口证明

这些实验不进入训练主线，仅证明数据接口和泛化边界。

### 7.1 Zenodo 6985321 zero-shot 诊断

| 数据 | SOC MAE | SOC P95 | 半循环 SOH 中位数 | 说明 |
|---|---:|---:|---:|---|
| fresh cell | `16.11%` | `35.13%` | `98.02%` | 20 Ah cell |
| aged cell | `14.03%` | `34.01%` | `81.23%` | reference aged SOH 约 `83.2%` |

![图 15 Zenodo 6985321 fresh/aged cell zero-shot SOC 与半循环 SOH 诊断](paper_figures/zenodo_6985321/fig15_zenodo_6985321_zero_shot.png)

图 15 说明：SOC zero-shot 误差较高，说明跨尺度迁移仍是短板；SOH throughput 能区分 fresh/aged，可作为诊断展示。

### 7.2 Zenodo 18471156 真实电站定性展示

选中片段：

```text
real_world_05_06/battery_03_cells_017-024_t_04000-07999.csv
```

关键特征：

- 4000 行真实电站监测数据。
- 包含 8 个 cell 电压/温度和 pack current。
- 无 SOC/SOH/容量标签，因此不能做定量 MAE/RMSE。

![图 16 Zenodo 18471156 真实储能电站片段定性展示](paper_figures/zenodo_18471156/fig16_zenodo_18471156_station_demo.png)

图 16 说明：图中 SOC 是 W1 LSTM 定性输出，灰色 consistency proxy 不是 SOH 真值。该图只证明真实工业 CSV 接口可跑通。

## 8. 复现证据路径

| 证据 | 路径 |
|---|---|
| SOC 模型 | `outputs/soc_finetuned.h5` |
| SOC 指标 | `outputs/soc_finetuned.metrics.json` |
| SOH baseline | `outputs/soh_baseline.pt` |
| SOH 指标 | `outputs/soh_baseline.metrics.json` |
| 世界模型训练数据 | `outputs/world_model_train_data.pt` |
| Mamba 世界模型 | `outputs/world_model.pt` |
| 世界模型指标 | `outputs/world_model.metrics.json` |
| B0018 长 horizon replay | `outputs/closed_loop_replay_b0018.metrics.json` |
| Randomized QC 指标 | `outputs/world_model_randomized_full_stride64_tempqc.metrics.json` |
| SAC policy | `outputs/sac_policy.zip` |
| W4 单体评估 | `outputs/eval_w4_final_default/` |
| 多单体评估 | `outputs/eval_pack_6s1p_h1200/` |
| UPC pack 图表 | `outputs/upc_pack_paper/` |
| 技术审评决策记录 | `outputs/runs/technical_review_20260509/decisions.md` |
| IEEE 风格插图 | `paper_figures/` |

## 9. 当前边界与答辩口径

1. SOC 当前 MAE `3.48%`，可作软标签工程基线，但不声称达到 `<1.5%`。
2. SOH capacity-ratio baseline 依赖 NASA `Capacity` 字段，不等价于无容量标签部署。
3. Mamba 世界模型短期预测有效，但 600-step replay 漂移大，不声称替代真实电池 plant。
4. 当前 ΔSOH 改善由物理应力 proxy 主导，不声称 Mamba 老化头已学到可靠寿命模型。
5. 多单体 prototype 不是真实串联包 KCL/KVL 电路仿真。
6. UPC pack 和 Zenodo 电站实验属于展示/边界诊断，不进入 NASA 训练主线。
7. Simulink 30 模组资产来源和参数依据未知，只能作为接口/电路烟测流程，不作为论文定量依据。

## 10. 可放入答辩 PPT 的最短证明链

```text
1. 数据可信：NASA 同源 NMC 18650 主线，避免跨化学体系混训。
2. 状态估计可跑：SOC B0018 MAE 3.48%，SOH baseline 达到 NASA 容量比一致性。
3. 世界模型有效但边界清楚：1-step 1.42 mV，20-step 8.04 mV；600-step 漂移已主动报告。
4. 安全层硬约束：ECM 投影 1000 随机动作全部满足电压边界。
5. 策略有效：SAC vs CC-CV，时间 -30.97%，代理 ΔSOH -17.37%，过压 0。
6. 包级扩展谨慎：多单体 supervisory prototype + UPC 真实 pack 数据，不冒充真实串联包仿真。
```

