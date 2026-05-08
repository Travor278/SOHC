# CRAIC2026 论文图注与答辩讲解清单

更新日期：2026-05-08

> 用途：正式论文可取“推荐图注”；答辩 PPT 可取“讲解重点”。所有图均来自本仓库本地结果或本项目重画示意，不直接使用第三方未授权截图。

| 图 | 路径 | 推荐图注 | 讲解重点 | 边界说明 |
|---|---|---|---|---|
| 图形摘要 | `paper_figures/fig00_graphical_abstract.png` | 本文智能快充系统的三层闭环架构、包级扩展与当前关键指标总览。 | 一页讲清 W1-W5 结构和最核心结果。 | SOC 1.5% 目标未达；电站数据无标签。 |
| 图 1 | `paper_figures/fig01_system_architecture.png` | Mamba 世界模型、SAC 策略与 ECM 安全层协同的 EV 快充决策框架。 | 说明 AI 决策层不替代安全层，动作先经 ECM 投影。 | 不是量产 BMS 全栈，只是决策原型。 |
| 图 2 | `paper_figures/fig02_data_flow.png` | NASA、LG、UPC 与 Zenodo 数据在训练、验证和展示中的用途划分。 | 强调训练主线 NASA 同源，LG 只给权重，UPC/Zenodo 不进训练。 | Zenodo 是诊断/展示，不是训练集。 |
| 图 3 | `paper_figures/fig08_soc_b0018_prediction.png` | B0018 holdout 上 SOC 参考标签与 W1 LSTM 预测曲线。 | SOC 走势可跟随，但 MAE 3.48% 未达 1.5%。 | 不夸大 SOC 精度。 |
| 图 4 | `paper_figures/fig07_soh_baseline.png` | NASA 容量比 SOH baseline 的预测一致性与代表 cell 退化轨迹。 | SOH 可作为世界模型软标签。 | 使用 `Capacity` 字段，不是无标签部署模型。 |
| 图 5 | `paper_figures/fig03_world_model_rollout.png` | B0018 holdout 上 Mamba 世界模型 20-step open-loop 电压预测。 | 证明 W2 短期动力学预测稳定。 | Randomized full 见补充图 19，是动态负载压力边界。 |
| 图 6 | `paper_figures/fig05_ecm_safety_projection.png` | 高 SOC 条件下 ECM 安全层对充电电流的电压约束投影。 | L3 安全层保证动作物理安全。 | 参数来自 legacy 二阶 ECM，需要后续电芯级标定。 |
| 补充图 19 | `paper_figures/fig19_randomized_rollout_recheck.png` | NASA Randomized 28/28 文件 strict stride=64 下的 20-step rollout 全量复核。 | 主动说明 `RW2/RW3` 拉高全量动态负载误差，避免只展示顺滑子集。 | 这是压力边界，不是 `<10 mV` 达标结果。 |
| 图 7 | `paper_figures/fig06_sac_training_curve.png` | SAC 训练过程中的 episode return 与 critic loss。 | reward sweep 后策略可学习稳定行为。 | return 不是严格单调。 |
| 图 8 | `paper_figures/fig04_w4_metrics_bar.png` | 单体 paired episodes 中 SAC 与 CC-CV 的充电时间和 ΔSOH 对比。 | W4 核心结论：更快且更少老化。 | paired 口径需说明。 |
| 图 9 | `paper_figures/fig09_charging_comparison_polished.png` | SAC、CC-CV 与 MFCC 的 I/V/SOC/T 轨迹对比。 | SAC 形成动态电流轨迹，423 s 达到 80% SOC，且电压保持在 4.2 V 安全线内。 | 轨迹来自世界模型环境。 |
| 图 10 | `paper_figures/fig10_pack_comparison_polished.png` | 6S1P 包级策略复制与 SOC-spread 均衡协调结果。 | 单体策略复制到 pack 后，SAC 在 paired episode 中更早达到 min-cell 80% 目标并降低末端 spread。 | 仍是 Python supervisory simulator。 |
| 图 11 | `paper_figures/fig11_active_balancer_topology.png` | 本项目重画的主动 buck-boost 均衡协调拓扑示意。 | 说明均衡控制逻辑和 per-cell 电流修正。 | 不是开关级 PWM/器件模型。 |
| 图 12 | `paper_figures/fig12_upc_measured_profile.png` | UPC 36-cell pack 在 WLTP 工况下的实测电压、电流和 SOC profile。 | 真实 pack voltage spread 显著，说明包级问题真实存在。 | UPC 温度列有异常占位，分析使用有效温度范围。 |
| 图 13 | `paper_figures/fig13_upc_real_balancing_semicycle.png` | UPC 真实 balancing semicycle 中 cell voltage spread 的动态变化。 | 真实 balancing spread 不一定单调下降。 | 不能截取局部片段夸大效果。 |
| 图 14 | `paper_figures/fig14_python_balancing_short_sim.png` | 基于 UPC 高 spread 初值的 Python active buck-boost 短仿真。 | active balancing 30 min spread 降低 46.30%。 | 需 Simulink/Simscape 进一步验证电路约束。 |
| 图 15 | `paper_figures/zenodo_6985321/fig15_zenodo_6985321_zero_shot.png` | Zenodo 6985321 fresh/aged cell 上的 zero-shot SOC 与半循环 SOH 诊断。 | SOH throughput 能区分 fresh/aged；SOC zero-shot 误差较高。 | 是迁移瓶颈诊断，不是泛化达标。 |
| 图 16 | `paper_figures/zenodo_18471156/fig16_zenodo_18471156_station_demo.png` | Zenodo 18471156 真实储能电站片段的 V/I/T 输入与 SOC/consistency 输出。 | 数据接口能落到真实站端 CSV，曲线范围可解释。 | 无 SOC/SOH/容量真值，不能报精度。 |
| 图 17 | `paper_figures/fig17_simulink_pack_workflow.png` | Python pack 策略轨迹接入 Simulink 30 模组与 buck-boost 均衡资产的工作流。 | 明确 CSV/MAT bridge 和 balancing on/off paired test。 | Simulink 资产只作接口/烟测，不替代 UPC 定量。 |
| 图 18 | `paper_figures/fig18_results_dashboard.png` | 当前系统证据地图：单体快充、世界模型、包级复制与泛化边界。 | 最适合答辩“结果总览”页。 | 红色 panel 主动标注 zero-shot gap。 |

## PPT 建议顺序

1. 图形摘要。
2. 图 1 + 图 2：架构与数据策略。
3. 图 3 + 图 4：W1 状态估计。
4. 图 5 + 图 6 + 补充图 19：W2 世界模型、安全层与动态负载边界。
5. 图 8 + 图 9：W4 单体快充核心结果。
6. 图 10 + 图 14：包级复制与均衡效果。
7. 图 15 + 图 16：泛化诊断与真实电站展示。
8. 图 18：总结页，主动说明已验证结果和边界。
