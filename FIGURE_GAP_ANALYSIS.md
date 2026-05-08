# 论文图像缺口清单

更新日期：2026-05-08  
对应报告草稿：[CRAIC2026_REPORT_DRAFT.md](CRAIC2026_REPORT_DRAFT.md)

## 1. 当前结论

目前已经具备可支撑“单体智能快充 + 包级扩展”的核心结果图，但还缺三类关键图：

1. **总览类图**：系统架构图、数据流图。
2. **可信度类图**：世界模型预测 vs 真实、ECM 安全投影、SAC 训练曲线。
3. **泛化展示图**：Zenodo 6985321 zero-shot、Zenodo 18471156 真实电站定性图。

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

## 3. 缺失图像

| 优先级 | 图像 | 目的 | 当前缺口 | 建议生成方式 |
|---:|---|---|---|---|
| P0 | 系统总体架构图 | 一页解释“三层架构 + 数据流” | 只有 Markdown ASCII 图 | 用 Mermaid/Matplotlib 生成正式 PNG/SVG |
| P0 | W2 世界模型预测 vs 真实 | 证明 Mamba world model 可信 | 只有 metrics JSON，没有曲线图 | 从 `outputs/world_model_train_data.pt` + `outputs/world_model.pt` 采样 B0018 rollout 画 V/SOC/T |
| P0 | W4 指标柱状图 | 把 30.97% / 17.37% 结果做成论文表图 | 只有 CSV 和四联曲线 | 从 `outputs/eval_w4_final_default/paired_vs_cc_cv.csv` 画 bar chart |
| P1 | ECM 安全投影图 | 证明 L3 safety layer 不是口头约束 | 缺投影前后电流/电压图 | 随机采样 action，画 raw V vs clipped V、raw I vs safe I |
| P1 | SAC reward / learning curve | 证明训练调参有效 | TensorBoard event 未导出 PNG | 解析 `outputs/runs/w3_horizon600/tb/SAC_1/events*` 画 reward 曲线 |
| P1 | SOH 预测 vs 真实 | 证明 W1 SOH baseline | 只有 metrics JSON，没有 per-cycle 图 | 重新跑/扩展 `soh_train.py` 保存 predictions CSV 后画散点/退化曲线 |
| P1 | 数据集流向图 | 解释 NASA / LG / UPC / Zenodo 各自用途 | 只有文字表 | 画 data provenance diagram |
| P2 | Randomized 动态负载外推图 | 证明动态负载泛化 | 指标记录存在版本差异 | 先复核 Randomized rollout，再画 profile 与误差曲线 |
| P2 | Zenodo 6985321 zero-shot 曲线 | W5 定量泛化 | 任务未完成 | 按 OCV-SOC 表重建标签后画 SOC/SOH |
| P2 | Zenodo 18471156 定性图 | 真实电站展示 | 数据未下载 | 下载后跑 inference，画 V/I/T + SOC/SOH |
| P2 | BatteryML Mamba-head SOH 对比图 | 架构创新点补强 | 任务未完成 | 实现 Mamba feature/head 后画 SOH 对比表 |
| P3 | Simulink 接口流程图 | 说明已有 pack 资产如何接入 | 只有 workflow 文档 | 用 `SIMULINK_PACK_WORKFLOW.md` 画流程图 |

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

最推荐先补 5 张图，性价比最高：

1. `fig_system_architecture.png`
2. `fig_world_model_rollout.png`
3. `fig_w4_metrics_bar.png`
4. `fig_ecm_safety_projection.png`
5. `fig_sac_training_curve.png`

这五张补完后，报告主链路就比较完整；再往后做 Zenodo 6985321 / 18471156，属于泛化与答辩加分项。
