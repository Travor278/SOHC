# 技术审评报告：框架科学性与严谨性

> 审评基准：2024–2026 年同领域顶刊/arxiv 论文 + 开源仓库对标  
> 审评对象：`CRAIC2026_REPORT_DRAFT.md` 所描述的系统架构与实验设计  
> 日期：2026-05-08

---

## 文献基准速查

在给出问题列表之前，先确立本工作各模块在当前文献中的参照水位，供自我定位使用。

| 模块 | 本工作 | 2024–2026 文献水位 | 代表论文 |
|---|---|---|---|
| SOC MAE (同化学体系 holdout) | 3.48% | **< 0.7%**（Mamba 系） | MambaLithium (2024), IWOA-Mamba (Springer Ionics 2025) |
| SOH RMSE (NASA holdout) | 2.87% (Mamba head) | **< 0.7%** | SambaMixer arXiv:2411.00233 |
| 世界模型 1-step V MAE | 1.42 mV | 无直接可比基准（电压预测任务定义不同） | — |
| 快充速度提升 vs CC-CV | +30.97% | 有正向结果但多数论文未报告统一百分比 | SAC+ROM (Energies 2025), MPC-guided DRL (IEEE 2024) |
| 包级 SOC variance 降低 | −28% spread | MARL 文献报告 **−69.4%** SOC variance | Multi-Agent RL for Balancing (ScienceDirect 2025) |
| RL 环境框架 | 自写 Gymnasium env | **PyBaMM + liionpack** 是社区标准 | pybamm-team/liionpack |

---

## 问题一览（按严重程度）

```
● 严重（Critical）— 影响科学主张的物理/逻辑正确性，必须修正
▲ 主要（Major）  — 削弱论证说服力，建议修正或补 caveat
◇ 中等（Minor）  — 完整性缺口，答辩时需主动说明
```

---

## ● C1：包级仿真拓扑错误（6S1P ≠ 独立电流控制）

### 问题描述

`pack_balance.py` 中 `apply_soc_balancer()` 给每个 cell 分配**独立不同的充电电流**：

```python
trims = cfg.balance_gain_A_per_soc * (mean_soc - soc)   # 每 cell 不同 trim
balanced = np.clip(currents + trims, 0.0, cfg.i_max_amps)
```

在 **6串1并（6S1P）** 拓扑中，6 个串联 cell 只有一条电流路径，由基尔霍夫电流定律，**所有串联 cell 电流必须相等**，不能独立控制。

### 文献确认

PyBaMM/liionpack（社区标准包级仿真框架）对串联包的建模是单一 string current。NXP AN4428 及 Nature 2024 均明确：主动均衡实现的是 H 桥**旁路开关**（按占空比旁路某个 cell），而非给每个 cell 施加独立模拟电流。

> "Active balancing only provides bypass switching (ON/OFF per cell), not independent analog current control."  
> — Active Cell Balancing for Li-ion (Nature, 2024)

当前代码实际仿真的是 **6 个完全独立的单体**，而非串联包。

### 解决方案（两选一）

**方案 A（推荐，改动最小）：重新定性，不改代码**

在报告中将"6S1P 包级仿真"重新表述为：

> "6-cell 独立策略复制仿真（independent cell simulation）：验证单体策略向多 cell 的策略泛化性，以及 SOC-spread 感知协调器的控制逻辑。该仿真不求解串联字符串约束，不替代基于 KVL 的包级物理验证。"

这与 MARL 文献中"每个 cell 是独立智能体"的建模框架一致，只需诚实说明边界。

**方案 B（改代码，更严谨）：加入串联电流约束**

```python
def apply_series_pack_with_balancing(
    base_current_A: float,           # 整串唯一充电电流
    soc: np.ndarray,
    cfg: PackConfig,
) -> np.ndarray:
    """
    串联包：所有 cell 流过相同 base_current。
    均衡电流只在 cell 间转移能量（能量守恒）：
      - 高 SOC cell 旁路（有效充电电流 = 0）
      - 低 SOC cell 全流通（有效充电电流 = base_current）
    均衡增益以"旁路占空比"而非绝对电流表示。
    """
    duty = np.ones(cfg.n_cells, dtype=float)                # 全部全流通
    mean_soc = float(np.mean(soc))
    for i, s in enumerate(soc):
        if s > mean_soc + cfg.balance_tolerance_soc:        # 超前 cell 旁路
            duty[i] = max(0.0, 1.0 - cfg.balance_gain_A_per_soc * (s - mean_soc))
    effective_current = np.clip(duty * base_current_A, 0.0, cfg.i_max_amps)
    return effective_current
```

配合将 `rollout()` 改为先由外部 charger 决定 `base_current`，再用上述函数生成 per-cell 有效电流。

---

## ● C2：老化优化实际由手工代理公式主导，而非 Mamba 预测

### 问题描述

`rl_env.py` 中：

```python
model_delta_soh = max(float(pred[3]), 0.0)          # Mamba 输出
proxy_delta_soh  = self._aging_proxy_delta_soh(...)  # 手工应力公式
delta_soh = max(model_delta_soh, proxy_delta_soh)   # 取大值
```

`_aging_proxy_delta_soh` 基底为 `calendar_aging_scale=2.5e-6`/步。600 步基础贡献：

```
600 × 2.5e-6 = 0.0015
```

报告中 CC-CV ΔSOH=0.001859，SAC ΔSOH=0.001536，两者量级恰好与代理公式底线相符。SAC 通过减少高电流/高电压动作来降低 `aging_proxy_delta_soh`——这是在**优化手工公式，而非数据驱动的老化模型**。

### 文献对照

SAC + reduced-order electrochemical model（Energies 2025）的老化 reward 直接来自模型输出的副反应电流（side reaction current），而非手工代理。这是该工作的核心创新点之一。

### 解决方案

**步骤 1：测量两者各自主导的比例（必做）**

```python
# 在 eval_compare.py 的评估循环中添加统计
model_dominated = 0
proxy_dominated = 0
for info in episode_infos:
    if info['model_delta_soh'] >= info['aging_proxy_delta_soh']:
        model_dominated += 1
    else:
        proxy_dominated += 1
print(f"Mamba 主导步数: {model_dominated}, Proxy 主导步数: {proxy_dominated}")
```

**步骤 2：根据结果选择处理方式**

- 若 Mamba 主导比例 < 30%：在报告中诚实说明"老化惩罚目前主要由基于物理应力的代理公式驱动，Mamba ΔSOH 输出作为补充上界"。这不等于造假，是真实的 hybrid 设计，但需要如实描述。
- 若 Mamba 主导比例 > 50%：正面结论，可作为世界模型有效性的证据之一。

**步骤 3（可选，加强工作）**：去掉 `max(model, proxy)`，改用加权融合：

```python
# 可调参数 alpha ∈ [0, 1]，alpha=1 纯 Mamba，alpha=0 纯 proxy
delta_soh = alpha * model_delta_soh + (1 - alpha) * proxy_delta_soh
```

并消融 alpha 的影响，展示 Mamba 输出对老化预测的独立贡献。

---

## ● C3：世界模型 20-step 验证 vs 600-step 部署，存在 30× 外推未量化

### 问题描述

世界模型在 B0018 上做了 20-step **开环** rollout 验证（8.04 mV）。SAC 训练是 600 步**闭环**（每步用自身预测更新历史），开环误差不等于闭环误差。

此外，`_reset_history()` 将 64 步历史全部填充为相同初始状态（电流=0），Mamba 训练时从未见过这种全静止历史，这是分布外输入。

### 文献对照

模型基强化学习（MBRL）社区的标准做法（Dreamer, TDMPC2 等）：
1. 多步训练目标（k-step rollout loss），而非只用 1-step loss
2. 在推理时进行短步 rollout（16–50 步），不做 600 步单次 rollout
3. 报告 **闭环** 误差，不只报开环误差

### 解决方案

**最小修正（不改训练，只补量化）**：

在 `eval_compare.py` 中添加闭环误差测量：

```python
def measure_closed_loop_error(world_model, ecm_params, real_trajectory_df):
    """
    用真实 NASA 充电轨迹作为 ground truth，
    world model 用真实初始状态启动，后续全靠自身预测（不注入真值）。
    每步记录预测电压 vs 真实电压的差值。
    """
    errors = []
    # ... 实现：replay real trajectory, feed predicted state as next input
    return {
        "closed_loop_100step_V_MAE": ...,
        "closed_loop_600step_V_MAE": ...,
        "closed_loop_600step_V_p95": ...,
    }
```

在报告 §5.2 中补充这个表格，哪怕结果不好也要报，这是科学诚信的体现，且文献中多数 MBRL 工作都展示了这个数字。

**改善历史初始化**：

```python
def _reset_history(self, state: np.ndarray) -> None:
    # 在初始状态上加微小噪声，避免 64 步完全相同
    base_row = np.array([state[0], state[1], state[2], state[3], state[4], 0.0])
    noise = np.random.normal(0, 1e-3, (self.cfg.seq_len, 6))
    noise[:, 0] = np.clip(noise[:, 0], -0.01, 0.01)   # SOC 噪声小
    self.history = np.clip(
        np.repeat(base_row[None, :], self.cfg.seq_len, axis=0) + noise,
        0.0, None
    )
```

---

## ▲ M1：ECM 参数固定，不随 SOH 更新

### 问题描述

`ecm_safety_layer.py` 从单个 `.mat` 文件加载一组固定 R0/R1/R2/C1/C2。但：

1. 内阻 R0 随老化线性增大，SOH=0.85 的电池 R0 可比新电池大 20–40%
2. 参数文件路径为 `MATLAB滤波算法代码——云储实时数据/...`，这是**云储实时数据**而非 NASA 18650 cells，OCV 曲线来源存疑

### 解决方案（轻量）

添加 SOH 自适应的 R0 修正，不需要重新标定：

```python
def predict_voltage_soh_aware(self, soc: float, current: float, soh: float) -> float:
    """
    用线性 SOH 退化因子修正 R0：
    R0_eff = R0_fresh * (2 - soh)  （近似：SOH=1 → R0，SOH=0.8 → 1.2×R0）
    这是文献中常用的一阶近似（参考 Hu et al., J. Power Sources 2012）。
    """
    r0_eff = self.params.R0 * (2.0 - float(np.clip(soh, 0.5, 1.0)))
    next_v1, next_v2 = self._next_polarization(current)
    return self._ocv(soc) - current * r0_eff - next_v1 - next_v2
```

并在报告中说明使用一阶 SOH-R0 线性修正，指向引用。

---

## ▲ M2：温度惩罚权重实质为零

### 问题描述

最终权重 `temperature=0.02`，典型步温度惩罚约 `0.02 × (15/10)² ≈ 0.045`，而速度奖励约 `30 × 0.001 = 0.03`。温度约束可以说是激活的，但在速度和电压惩罚主导的训练信号下，策略对温度几乎无感知，训练曲线不会因温度超限出现有意义的惩罚峰。

### 解决方案

将温度从软惩罚改为**软硬混合**：

```python
# 温度惩罚：T < T_soft 时为零，T_soft 到 T_max 线性增大，超 T_max 后 cliff
T_soft = 40.0   # 开始惩罚的温度阈值
if T <= T_soft:
    temp_penalty = 0.0
elif T <= self.cfg.T_max:
    temp_penalty = ((T - T_soft) / (self.cfg.T_max - T_soft)) ** 2
else:
    temp_penalty = 10.0 + (T - self.cfg.T_max) * 5.0   # cliff

reward -= self.cfg.reward.temperature * temp_penalty
```

并将 `temperature` 权重提升到与 `aging` 同量级（~100）。这样温度约束才真正被学习。

---

## ▲ M3：Reward 权重代码默认值与报告不一致

### 问题描述

| 参数 | 代码默认（RewardWeights dataclass）| 报告 §3.5 声称的最终权重 |
|---|---|---|
| speed | 12 | 30 |
| voltage | 50 | 300 |
| temperature | 0.2 | 0.02 |
| aging | 80 | 120 |

这说明最终训练时通过命令行参数覆盖了默认值，但这一信息没有记录在代码或报告中。

### 解决方案

**步骤 1**：将最终权重写入代码（更新 dataclass 默认值或添加 `FINAL_WEIGHTS` 常量）：

```python
# train_sac.py 或 rl_env.py 头部
FINAL_REWARD_WEIGHTS = RewardWeights(
    speed=30.0,
    voltage=300.0,
    temperature=0.02,
    aging=120.0,
)
```

**步骤 2**：在报告方法节补充 reward sweep 过程的一句话说明（不需要完整表格，只需说明是怎么得到的）。

---

## ▲ M4：Mamba vs GRU 未消融，创新必要性未证明

### 问题描述

代码中存在完整的 GRU fallback（`world_model_mamba.py`，Windows 上自动切换），但从未对同一任务做 Mamba vs GRU 对比实验。报告声称使用 Mamba 是架构创新点，却没有实验支撑。

SambaMixer（arXiv:2411.00233）和 MambaLithium 都有明确的 Mamba vs LSTM/Transformer 对比表格。

### 解决方案

在已有 GRU fallback 的基础上，用**完全相同超参数**分别训练 Mamba 和 GRU 版本，对比 B0018 holdout 的 1-step 和 20-step MAE。预计只需一次额外训练，即可得到如下表格：

| 后端 | 1-step V MAE | 20-step rollout V MAE | 训练时间 |
|---|---|---|---|
| GRU (baseline) | ? | ? | ? |
| Mamba (ours) | 1.42 mV | 8.04 mV | — |

即使 Mamba 仅有 10–20% 改善，该对比也是 CRAIC 创新度维度的直接证据。

---

## ◇ N1：SOC 估计误差在下游链路中的传播未分析

### 问题描述

SOC 估计 MAE 3.48% 会直接注入世界模型输入（SOC 通道）和 SAC 观测空间。这个误差如何影响下游预测精度和策略质量，当前完全未分析。

### 解决方案（轻量）

做一个简单的 oracle vs estimated SOC 对比实验：

```python
# 评估时运行两个版本：
# 版本 A：SOC 通道注入 oracle SOC（库仑积分真值）
# 版本 B：SOC 通道注入 LSTM 估计值
# 比较世界模型 20-step V MAE 的差异
```

结论无论好坏都有价值：若差异小，说明系统对 SOC 误差鲁棒；若差异大，说明需要先提升 SOC 精度。

---

## ◇ N2：SAC 训练统计样本量不足

### 问题描述

- 包级 paired episodes n=1（CC-CV 3/3 中只有 1 个达到目标）
- 单体 30.97% 速度提升基于的 paired episodes 数量在报告中未明确报告

### 解决方案

单体实验至少补充 10 对 paired episodes，包级补充至 5 对。报告结果时加均值 ± 标准差，例如：

> "SAC 相比 CC-CV 将充电时间减少 30.97% ± 4.2%（n=10 paired episodes）"

---

## ◇ N3：NASA 4-cell 数据集多样性极低

### 问题描述

B0005/B0006/B0007/B0018 均来自同一实验室、相同协议、相同温度的 18650 NMC 电池。世界模型泛化边界实际上是"同款电池同条件"，而非"同化学体系"。

### 解决方案

在报告中明确说明：

> "当前世界模型在 NASA 同源 4-cell 数据上训练和验证，其泛化边界是相同型号、相同实验室条件下的不同老化程度。ARC-FY08Q4 提供了温度和倍率多样性，Randomized 子集提供了负载 profile 多样性，但化学体系和容量规格多样性仍受 NASA 同源数据限制。"

---

## 需要补充的实验（按 ROI 排序）

| 实验 | 工作量 | 对报告增益 | 优先级 |
|---|---|---|---|
| Mamba vs GRU 消融（B0018 holdout） | 低（已有 fallback，跑一次） | 高（直接支撑创新度） | ★★★ |
| `model_delta_soh` vs `proxy_delta_soh` 分解统计 | 极低（加几行 log） | 高（澄清老化模型设计） | ★★★ |
| 世界模型闭环误差（100/600 步） | 中（写评估脚本） | 高（填补核心验证缺口） | ★★★ |
| 单体 paired episodes 扩展到 n=10 | 低（重跑 eval_compare.py） | 中（统计可信度） | ★★ |
| SOH 自适应 ECM 修正 | 低（几行代码） | 中（ECM 严谨性） | ★★ |
| 包级声明修正（重新定性） | 极低（改报告文字） | 高（修正物理谬误） | ★★★ |

---

## 对标文献清单（建议在相关工作节引用）

| 引用方向 | 文献 | 用途 |
|---|---|---|
| Mamba for battery | SambaMixer, arXiv:2411.00233 (2024) | SOH RMSE < 0.7% 基准，说明本文世界模型任务不同（非直接比较） |
| Mamba for battery | MambaLithium, github.com/zshicode/MambaLithium (2024) | Mamba 在 SOC/SOH 领域已有验证 |
| RL fast charging | SAC + ROM, Energies MDPI (2025) | 相同算法（SAC），不同物理模型，区分本文创新点（Mamba world model） |
| Physics-informed RL | Residual-Corrected ECM, arXiv:2605.06419 (2025) | 与本文 ECM 安全层设计思路相近，需区分贡献 |
| Pack balancing RL | Multi-Agent RL for Cell-Level Balancing, ScienceDirect (2025) | 包级 MARL 标杆，指出本文当前是单 agent 策略复制 |
| RL environment | PyBaMM + liionpack, pybamm-team/liionpack | 说明本文选择自写 env 的原因（数据驱动 vs 物理 env） |

---

## 总结

| 优先级 | 问题 | 解决方案 | 是否影响结论 |
|---|---|---|---|
| ● 必须 | C1 包级拓扑错误 | 重新定性为"独立 cell 仿真"，或改为串联约束 | 影响包级所有定量结论的物理解读 |
| ● 必须 | C2 老化代理主导 | 统计 Mamba/proxy 比例，报告中如实说明 hybrid 设计 | 影响"数据驱动老化优化"的主张 |
| ● 建议 | C3 闭环误差未量化 | 补充 100/600 步闭环误差评估 | 影响世界模型质量主张的完整性 |
| ▲ 建议 | M1 ECM 不随 SOH 更新 | 加一阶 R0-SOH 修正 | 不影响结论，但提升物理严谨性 |
| ▲ 建议 | M2 温度约束失效 | 改为软硬混合惩罚 + 调权重 | 不影响当前结论，但温度安全主张站不住 |
| ▲ 必须 | M3 权重不一致 | 更新代码默认值 + 报告补一句话 | 影响可复现性 |
| ▲ 建议 | M4 缺 Mamba 消融 | 跑 GRU vs Mamba 对比 | 直接支撑 CRAIC 创新度得分 |
| ◇ 建议 | N1 SOC 误差传播 | oracle vs estimated 对比实验 | 不影响结论，提升系统理解 |
| ◇ 建议 | N2 样本量不足 | 扩展到 n=10 paired episodes | 影响统计可信度 |
| ◇ 说明 | N3 数据多样性窄 | 在报告中明确泛化边界 | 诚信声明，不影响结论 |
