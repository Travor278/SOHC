# UPC Pack-Level Paper Results

更新日期：2026-05-08

本结果小节使用 UPC 36-cell pack WLTP+CC-CV 真实数据和本仓库纯 Python 包级 simulator 生成。外部 Simulink / buck-boost 参考仅用于拓扑和流程，不借用他人结果图。

## 数据与设置

- 数据集：UPC 36-cell pack WLTP+CC-CV，`12S3P`
- 数据 DOI：`10.34810/DATA2395`
- 实测 profile：`Qtzl_Cycle_003_WLTP_partial_data.parquet`
- 实测 balancing cycle：`Qtzl_Cycle_027_Capacity_check_partial_data.parquet`
- Python 脚本：`craic_pipeline/eval_upc_pack.py`
- 输出目录：`outputs/upc_pack_paper/`

## 图表产物

| 图 | 路径 | 说明 |
|---|---|---|
| 主动均衡拓扑示意 | `outputs/upc_pack_paper/fig_active_balancer_topology.png` | 本项目原创示意图，基于 active buck-boost 高 SOC → 低 SOC 能量转移思路 |
| UPC 实测 pack profile | `outputs/upc_pack_paper/fig_upc_measured_profile.png` | 36 cell voltage envelope、voltage spread、3 branch current、BMS SOC |
| UPC 实测 balancing semicycle | `outputs/upc_pack_paper/fig_upc_real_balancing_semicycle.png` | 高亮真实 balancing semicycle，展示实测 spread 变化 |
| Python 主动均衡短仿真 | `outputs/upc_pack_paper/fig_python_balancing_short_sim.png` | 从 UPC 高 spread 样本初始化，比较 balancing off / active buck-boost |

## 关键结果

### 真实 UPC WLTP 工况

| 指标 | 数值 |
|---|---:|
| 实测样本数（downsample=20） | `2985` |
| 持续时间 | `6.63 h` |
| 平均 cell voltage spread | `244.56 mV` |
| P95 cell voltage spread | `510.00 mV` |
| 最大 cell voltage spread | `590.00 mV` |
| 最大 branch current | `6.51 A` |
| 有效温度 P95 | `30.23 °C` |

### 真实 UPC balancing semicycle

| 指标 | 数值 |
|---|---:|
| balancing 持续时间 | `4.63 h` |
| balancing 起点 spread | `308.00 mV` |
| balancing 段内最小 spread | `127.00 mV` |
| balancing 终点 spread | `308.00 mV` |
| balancing 段内最大 spread | `622.00 mV` |

解释：真实 BMS balancing semicycle 并不保证 voltage spread 单调下降，spread 会受并行支路电流、SOC 变化、测试阶段切换和 BMS 策略共同影响。

### Python active buck-boost 短仿真

| Case | 初始 spread | 终点 spread | 降幅 | 最大均衡电流 |
|---|---:|---:|---:|---:|
| balancing off | `622.00 mV` | `622.00 mV` | `0.00%` | `0.00 A` |
| active buck-boost | `622.00 mV` | `334.00 mV` | `46.30%` | `0.80 A` |

解释：该仿真是包级控制数字孪生，不是 PWM / 开关电路级仿真。它说明在真实 UPC 高 spread 初值下，主动能量转移式均衡策略能在 30 分钟短仿真内显著压低 spread。

## 可直接放入论文的文字

> 为验证单体快充策略向包级均衡场景的可迁移性，本文进一步采用 UPC 36-cell 真实电池包数据集进行包级分析。该数据集为 12S3P 结构，包含 36 个单体电压、3 个支路电流、BMS SOC 与 cell 表面温度。以 Cycle 003 WLTP 工况为例，实测 cell voltage spread 均值为 244.56 mV，P95 为 510.00 mV，最大值达到 590.00 mV，说明真实动态负载下包内不一致性显著。进一步选取含 balancing semicycle 的 Cycle 027 作为高不一致性初值，在纯 Python active buck-boost 数字孪生中进行 30 min 短仿真。结果显示，相比 balancing off 的 spread 保持 622.00 mV，主动 buck-boost 均衡将终点 spread 降至 334.00 mV，降幅为 46.30%，且最大均衡电流受限于 0.80 A。该结果表明，本文提出的单体决策策略可以自然扩展到包级协调控制，并能在真实 pack 数据初始化条件下改善电压一致性。

## 引用来源

- MathWorks Battery Pack Cell Balancing：https://www.mathworks.com/help/sps/ug/lithium-pack-cell-balancing.html
- MathWorks Simscape Battery Cell Balancing：https://www.mathworks.com/help/simscape-battery/cell-balancing.html
- Active balancing Simulink reference：https://github.com/yavuzhanocak/Single-switch-capacitor-battery-balance
- Buck-boost active balancing paper：https://doi.org/10.1016/j.compeleceng.2025.110736
- UPC 36-cell pack dataset：https://doi.org/10.34810/DATA2395
