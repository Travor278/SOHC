# Simulink / 电路图参考素材索引

更新日期：2026-05-08

本文件记录已经拉取到本地的第三方 Simulink 仿真图、主动均衡拓扑图和电路图参考。**这些素材仅用于理解拓扑、控制流程和 Simulink 搭建方式，不作为本项目论文结果图，也不提交进 git。**

## 本地目录

```text
external_refs/simulink_balance/
├─ Single-switch-capacitor-battery-balance/
│  ├─ Matlab Simulink/ssc1.slx
│  └─ Altium designer/
└─ images/
   ├─ manifest.json
   ├─ single_switch_capacitor_01.png
   ├─ ...
   └─ single_switch_capacitor_11.png
```

重新拉取命令：

```powershell
git clone --depth 1 https://github.com/yavuzhanocak/Single-switch-capacitor-battery-balance.git `
  .\external_refs\simulink_balance\Single-switch-capacitor-battery-balance
.\.venv_craic\python.exe .\scripts\download_simulink_balance_refs.py
```

## 已拉取参考

来源仓库：<https://github.com/yavuzhanocak/Single-switch-capacitor-battery-balance>

该仓库包含 single switched capacitor active balancing、bidirectional converter、Simulink `.slx` 短仿真和 Altium 设计文件。仓库根目录未发现 `LICENSE` 文件，因此当前只按“本地阅读参考”处理。

| 本地文件 | 内容 | 对本项目的参考价值 |
|---|---|---|
| `single_switch_capacitor_01.png` | 单开关电容主动均衡拓扑说明图 | 对照 active balancing 连接关系 |
| `single_switch_capacitor_02.png` | 双向 converter 与 SSC 系统连接 | 对照 buck/boost 能量转移方向 |
| `single_switch_capacitor_03.png` | 低能量 cell 充电 PWM 示例 | 对照均衡控制占空比 |
| `single_switch_capacitor_04.png` | 高能量 cell 放电 PWM 示例 | 对照均衡控制占空比 |
| `single_switch_capacitor_05.png` | 双向开关结构 | 对照 MOSFET bidirectional switch 画法 |
| `single_switch_capacitor_06.png` | 双向 converter 结构 | 对照 buck/boost 子系统画法 |
| `single_switch_capacitor_07.png` | MATLAB/Simulink 总体模型 | 对照 Simulink plant + control + scope 布局 |
| `single_switch_capacitor_08.png` | SOC 仿真结果 | 只看结果呈现方式，不借用数值 |
| `single_switch_capacitor_09.png` | cell voltage 仿真结果 | 只看结果呈现方式，不借用数值 |
| `single_switch_capacitor_10.png` | Altium PCB / 电路图 1 | 对照电路级实现复杂度 |
| `single_switch_capacitor_11.png` | Altium PCB / 电路图 2 | 对照电路级实现复杂度 |

## 官方 Simulink 参考链接

这些链接更适合作为论文/答辩中的“建模流程参考”，但不直接下载或复用其图片：

- MathWorks Battery Pack Cell Balancing：<https://www.mathworks.com/help/sps/ug/lithium-pack-cell-balancing.html>
- MathWorks Battery Passive Cell Balancing：<https://www.mathworks.com/help/simscape-battery/ug/battery-cell-balancing.html>
- MathWorks Build Model of Battery Pack with Cell Balancing Circuit：<https://www.mathworks.com/help/simscape-battery/ug/build-battery-pack-cell-balancing.html>

## 使用边界

- 可以参考：拓扑关系、模块命名、Simulink 信号组织、scope/结果图的排版方式。
- 不建议参考：对方参数、对方仿真数值、对方性能结论。
- 不应使用：直接把第三方图片放进论文结果页，或把第三方仿真曲线当作本项目实验结果。
- 若必须在 PPT 中展示第三方拓扑图，需要明确标注来源；若进入论文正文，先确认 license 或获得作者许可。
