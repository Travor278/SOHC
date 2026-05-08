# 论文图像缺口清单

更新日期：2026-05-08  
对应报告草稿：[CRAIC2026_REPORT_DRAFT.md](CRAIC2026_REPORT_DRAFT.md)

## 1. 当前结论

目前已经具备可支撑“单体智能快充 + 包级扩展”的核心结果图。本轮已补齐 P0/P1 主链路插图，并统一汇总到 `paper_figures/`：

1. **总览类图**：系统架构图、数据流图。
2. **可信度类图**：世界模型预测 vs 真实、ECM 安全投影、SAC 训练曲线。
3. **实验结果类图**：单体指标柱状图、SOH baseline 图、W4/W5 已有结果图汇总。

泛化展示类图已补齐：Zenodo 6985321 zero-shot、Zenodo 18471156 真实电站定性图。当前主要缺口收敛为 Randomized 动态负载 rollout 指标复核和 BatteryML Mamba-head SOH 对比。

外部 Simulink / 电路图已经拉取到 `external_refs/simulink_balance/`，但只能做参考，不能作为本项目论文结果图。

## 2. 已有可用图

| 图像 | 路径 | 论文用途 | 状态 |
|---|---|---|---|
| B0018 SOC 预测曲线 | `outputs/figures/soc_b0018_prediction.png` | W1 SOC 估计结果 | 可用，但需注明 MAE 3.48% 未达 1.5% |
| B0018 representative cycle | `outputs/figures/soc_b0018_cycle98_representative.png` | SOC 曲线补充 | 可用 |
| B0018 best-case cycle | `outputs/figures/soc_b0018_cycle316_bestcase.png` | SOC 最佳示例 | 可用，避免单独作为主结果 |
| 单体快充四联图 | `outputs/eval_w4_final_default/charging_comparison.png` | SAC vs CC-CV vs MFCC 核心结果 | 可用 |
| 6S1P pack 对比图 | `outputs/eval_pack_6s1p_h1200/pack_comparison.png` | 包级策略复制结果 | 可用 |
| 30S1P smoke 对比图 | `outputs/eval_pack_30s1p_smoke/pack_comparison.png` | Simulink 接口烟测 | 可用但不建议进论文定量 |
| 主动均衡拓扑示意 | `outputs/upc_pack_paper/fig_active_balancer_topology.png` | 本项目原创 topology 示意 | 可用 |
| UPC 实测 pack profile | `outputs/upc_pack_paper/fig_upc_measured_profile.png` | 真实 36-cell pack 不一致性 | 可用 |
| UPC balancing semicycle | `outputs/upc_pack_paper/fig_upc_real_balancing_semicycle.png` | 真实 BMS balancing 行为 | 可用 |
| Python active buck-boost 短仿真 | `outputs/upc_pack_paper/fig_python_balancing_short_sim.png` | 包级主动均衡效果 | 可用 |

本轮已将上述可用图复制到 `paper_figures/`，并额外生成以下 IEEE 风格插图：

| 图像 | 路径 | 说明 |
|---|---|---|
| 系统总体架构图 | `paper_figures/fig01_system_architecture.png` | 三层智能快充决策框架 |
| 数据来源与用途划分 | `paper_figures/fig02_data_flow.png` | NASA / LG / UPC 数据流向 |
| 世界模型 20-step rollout | `paper_figures/fig03_world_model_rollout.png` | B0018 holdout 开环预测 |
| W4 核心指标柱状图 | `paper_figures/fig04_w4_metrics_bar.png` | time-to-80 与 ΔSOH paired 对比 |
| ECM 安全投影图 | `paper_figures/fig05_ecm_safety_projection.png` | 高 SOC 下限流防过压 |
| SAC 训练曲线 | `paper_figures/fig06_sac_training_curve.png` | episode return 与 critic loss |
| SOH baseline 图 | `paper_figures/fig07_soh_baseline.png` | 验证集 scatter + 代表 cell 曲线 |
| Zenodo 6985321 zero-shot 图 | `paper_figures/zenodo_6985321/fig15_zenodo_6985321_zero_shot.png` | 跨数据集 SOC/SOH 诊断 |
| Zenodo 18471156 电站定性图 | `paper_figures/zenodo_18471156/fig16_zenodo_18471156_station_demo.png` | 真实电站 V/I/T + SOC/consistency 输出 |
| Simulink pack workflow | `paper_figures/fig17_simulink_pack_workflow.png` | Python pack 轨迹到 Simulink 资产接口流程 |

## 3. 缺失图像

| 优先级 | 图像 | 目的 | 当前缺口 | 建议生成方式 |
|---:|---|---|---|---|
| P0 | 系统总体架构图 | 一页解释“三层架构 + 数据流” | 已生成 | `paper_figures/fig01_system_architecture.png` |
| P0 | W2 世界模型预测 vs 真实 | 证明 Mamba world model 可信 | 已生成 | `paper_figures/fig03_world_model_rollout.png` |
| P0 | W4 指标柱状图 | 把 30.97% / 17.37% 结果做成论文表图 | 已生成 | `paper_figures/fig04_w4_metrics_bar.png` |
| P1 | ECM 安全投影图 | 证明 L3 safety layer 不是口头约束 | 已生成 | `paper_figures/fig05_ecm_safety_projection.png` |
| P1 | SAC reward / learning curve | 证明训练调参有效 | 已生成 | `paper_figures/fig06_sac_training_curve.png` |
| P1 | SOH 预测 vs 真实 | 证明 W1 SOH baseline | 已生成 | `paper_figures/fig07_soh_baseline.png` |
| P1 | 数据集流向图 | 解释 NASA / LG / UPC / Zenodo 各自用途 | 已生成 | `paper_figures/fig02_data_flow.png` |
| P2 | Randomized 动态负载外推图 | 证明动态负载泛化 | 指标记录存在版本差异 | 先复核 Randomized rollout，再画 profile 与误差曲线 |
| P2 | Zenodo 6985321 zero-shot 曲线 | W5 定量泛化诊断 | 已生成；SOC zero-shot 误差较高 | `paper_figures/zenodo_6985321/fig15_zenodo_6985321_zero_shot.png` |
| P2 | Zenodo 18471156 定性图 | 真实电站展示 | 已生成；无标签，仅定性 | `paper_figures/zenodo_18471156/fig16_zenodo_18471156_station_demo.png` |
| P2 | BatteryML Mamba-head SOH 对比图 | 架构创新点补强 | 任务未完成 | 实现 Mamba feature/head 后画 SOH 对比表 |
| P3 | Simulink 接口流程图 | 说明已有 pack 资产如何接入 | 已生成 | `paper_figures/fig17_simulink_pack_workflow.png` |

## 4. 不应直接使用的图

| 素材 | 原因 | 可接受用法 |
|---|---|---|
| `external_refs/simulink_balance/images/*.png` | 第三方仓库未发现 LICENSE | 本地参考拓扑/排版，不放入正式论文结果图 |
| `external_refs/simulink_balance/Single-switch-capacitor-battery-balance/Matlab Simulink/ssc1.slx` 截图 | 第三方模型，许可不明 | 学习搭建方式，引用源链接 |
| MathWorks 官方示例截图 | 官方文档版权，不应直接复用 | 引用链接，自己重画 workflow 或 topology |
| 仓库自带 `batterpack.slx` / `buck_boost_balance.slx` 结果图 | 来源与参数依据未知 | 可做接口演示，不做论文定量依据 |

## 5. 指标一致性待复核

当前发现一个需要在出正式论文前处理的小裂缝：

- `SYSTEM_STATUS.md` / `TODO.md` 记录 Randomized 子集 sampled 20-step rollout V MAE 约 `7.71 mV`。
- 当前 `outputs/world_model_randomized_subset_eval.metrics.json` 中记录：
  - one-step V MAE：`2.39 mV`
  - 20-step rollout V MAE：`26.88 mV`

建议正式写论文前重新跑一次 Randomized 子集外推，并把最终指标、命令、输出文件统一到一个新目录，例如：

```text
outputs/paper_figures/randomized_rollout/
```

在未复核前，正文只写 “Randomized one-step V MAE 2.39 mV”，不要写 “20-step < 10 mV”。

## 6. 建议下一步生成顺序

本轮已完成最关键的主链路图。下一步推荐补：

1. `fig18_randomized_rollout_recheck.png`
2. `fig19_batteryml_mamba_soh_ablation.png`
3. 按报告最终模板导出 PDF / PPT 版本图。

其中第一张用于修正 Randomized 动态负载指标口径；第二张用于补强 SOH 架构创新点。
