# Simulink Pack Simulation Workflow

更新日期：2026-05-08

本文件回答一个前提问题：**如果已经有可信的 pack / buck-boost balance Simulink 资产，如何把本项目的单体策略、UPC 36-cell 数据和包级均衡仿真接进去。**

注意：仓库自带 `batterpack.slx` / `buck_boost_balance.slx` 来源未知，因此只作可选接口演示。论文定量验证优先使用 UPC 36-cell pack 数据。

## 0. 最契合的外部 Simulink / 拓扑参考

这些参考只用于说明拓扑和仿真流程，**不直接借用对方结果图作为本项目结果**。

| 参考 | 类型 | 和本项目的关系 | 用法 |
|---|---|---|---|
| MathWorks Battery Pack Cell Balancing | 官方 Simscape Electrical 示例 | 提供 pack cell balancing、series/parallel 参数、初始 SOC/温度、Simscape logging 的可信流程 | 作为 Simulink 数据回放/日志接口参考 |
| MathWorks Simscape Battery Cell Balancing | 官方 Simscape Battery 示例集合 | 提供 cell balancing 示例入口，适合作为建模规范参考 | 作为 workflow 参考，不作为主动 buck-boost 结果 |
| `yavuzhanocak/Single-switch-capacitor-battery-balance` | GitHub 开源 Simulink + Altium | 含 active balancing、bidirectional converter、4-cell module 短仿真，最贴近“短仿真演示” | 只借鉴拓扑/控制思路，不借用结果图 |
| “A modularized active cell balancing ... buck-boost converter ... EV applications” | 2025 论文 | buck-boost 主动均衡、MATLAB/Simulink、静态/充电/放电模式 | 作为 active buck-boost 方向引用 |

链接：

- https://www.mathworks.com/help/sps/ug/lithium-pack-cell-balancing.html
- https://www.mathworks.com/help/simscape-battery/cell-balancing.html
- https://github.com/yavuzhanocak/Single-switch-capacitor-battery-balance
- https://doi.org/10.1016/j.compeleceng.2025.110736

## 1. 推荐仿真分三层做

### A. 数据回放校准

目标：先确认 Simulink pack plant 本身能复现实测 UPC 数据，不让控制策略背锅。

输入：

- UPC measured branch current：`Current_Actual_P1/P2/P3`
- UPC initial cell voltage：36 个 `Voltage_Cell_P#S#`
- UPC ambient / cell temperature

输出对比：

- Simulink cell voltage vs UPC measured cell voltage
- pack voltage RMSE
- cell voltage spread RMSE / max spread

通过条件建议：

```text
cell voltage RMSE < 20-30 mV
cell voltage spread trend direction correct
no numerical instability
```

### B. 策略闭环仿真

目标：验证 SAC/CC-CV/MFCC 在 pack plant 上的电流指令是否安全。

输入：

- `pack_balance.py` 生成的 per-cell current command
- 或 `pack_dataset_upc.py` 导出的实测 WLTP/CC-CV current profile

Simulink plant 执行：

```text
current command -> buck/boost 或 charger subsystem -> cell/module states
```

输出指标：

- min-cell 到 80% SOC 时间
- cell voltage spread
- branch current / pack current limit
- overvoltage / undervoltage count
- temperature max / p95

### C. 均衡电路验证

目标：验证 buck-boost balance 控制器是否能降低不一致性。

建议做 paired test：

```text
same initial cell SOC / voltage / SOH
same charger current limit
case 1: balancing off
case 2: balancing on
```

核心指标：

- end SOC spread reduction
- end cell voltage spread reduction
- balancing energy / balancing current throughput
- no overvoltage / no branch overcurrent

## 2. Python 侧导出 Simulink 输入

### UPC 实测数据导出

```powershell
.\.venv_craic\python.exe -m craic_pipeline.pack_dataset_upc `
  --data-dir data/pack_wltp_upc `
  --pattern "Qtzl_Cycle_003_WLTP_partial_data.parquet" `
  --downsample 10 `
  --summary-out outputs/simulink/upc_cycle003_summary.csv `
  --simulink-csv-out outputs/simulink/upc_cycle003_inputs.csv
```

导出 CSV 采用 Simulink 友好列名：

```text
time_s
pack_voltage_V
bms_soc
branch_current_P1_A ... branch_current_P3_A
branch_voltage_P1_V ... branch_voltage_P3_V
cell_voltage_P1S1_V ... cell_voltage_P3S12_V
cell_temperature_P1S1_C ... cell_temperature_P3S12_C
```

### Python pack 策略轨迹导出

```powershell
wsl -d Ubuntu2404 -- bash -lc "source ~/.venvs/sohc-craic-py312/bin/activate && \
python -m craic_pipeline.pack_balance \
  --episodes 1 \
  --max-steps 1200 \
  --n-series 30 \
  --n-parallel 1 \
  --strategies ours \
  --out-dir outputs/simulink/pack_policy_30s1p"
```

输出：

```text
outputs/simulink/pack_policy_30s1p/pack_trajectories.csv
```

其中每行是一个 cell 在某个 step 的状态和 current，可在 MATLAB 里 pivot 成：

```text
time_s x current_cell_001 ... current_cell_030
```

## 3. MATLAB / Simulink 接入骨架

### 读取 UPC CSV

```matlab
T = readtable("outputs/simulink/upc_cycle003_inputs.csv", ...
    "VariableNamingRule", "preserve");

t = T.time_s;
branchCurrent = [ ...
    T.branch_current_P1_A, ...
    T.branch_current_P2_A, ...
    T.branch_current_P3_A ...
];

cellVoltageVars = startsWith(T.Properties.VariableNames, "cell_voltage_");
cellVoltage = T{:, cellVoltageVars};   % N x 36, order P1S1..P3S12

cellTempVars = startsWith(T.Properties.VariableNames, "cell_temperature_");
cellTemp = T{:, cellTempVars};         % N x 36

branchCurrentTs = timeseries(branchCurrent, t);
cellVoltageInit = cellVoltage(1, :);
cellTempInit = cellTemp(1, :);
```

### 传入 Simulink

```matlab
model = "your_pack_model";
load_system(model);

simIn = Simulink.SimulationInput(model);
simIn = simIn.setVariable("branchCurrentCmd", branchCurrentTs);
simIn = simIn.setVariable("cellVoltageInit", cellVoltageInit);
simIn = simIn.setVariable("cellTempInit", cellTempInit);
simIn = simIn.setModelParameter("StopTime", string(t(end)));

out = sim(simIn);
```

### 记录输出

Simulink 模型建议 log：

```text
cellVoltageSim: N x cells
cellSocSim:     N x cells
cellTempSim:    N x cells
branchCurrent:  N x branches
packVoltage:    N x 1
balanceCurrent: N x cells or N x converter channels
```

## 4. 36-cell UPC 与 30 模组资产不匹配怎么办

优先级如下：

1. **最好**：把 Simulink pack 配成 `12S3P`，直接对齐 UPC 36 cell。
2. **可接受**：只取 UPC 前 30 个 cell，构成 `30S1P` 接口 smoke；这只能验证接口，不适合论文定量。
3. **可接受**：将 36 cell 按 series index 聚合成 12 module，每个 module 取 3 parallel cell 的均值/最大 spread。
4. **不建议**：把 30 模组结果直接宣称为 UPC 实测验证。

## 5. 电流符号必须先校验

本项目 Python 侧约定：

```text
positive current = charging
```

Simscape / 自建电池模型可能采用：

```text
positive current = discharging
```

接入前做一个 10 秒小测试：

```text
I = +1 A
若 SOC 上升，则符号一致
若 SOC 下降，则进入 Simulink 前取反
```

## 6. 最小验收清单

一次可信 Simulink pack run 至少输出：

- 输入文件路径与 git commit
- 数据来源：UPC DOI `10.34810/DATA2395`
- pack topology：`12S3P` 或明确说明 adapter
- 电流符号测试结果
- cell voltage RMSE / spread error
- min-cell 到 80% 时间
- end SOC spread
- overvoltage / undervoltage count
- balance throughput
- 输出图：cell SOC envelope、cell voltage envelope、balance current、pack current

## 7. 建议当前路线

```text
先用 UPC Cycle_003 WLTP 做数据回放校准；
再用 pack_balance.py 的 SAC current profile 做策略闭环；
最后打开 buck-boost balance，对比 balancing on/off。
```
