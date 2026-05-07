# CRAIC2026 创新赛 · 方向三整合实施方案

> 已批准计划副本。原始可编辑版本：`C:\Users\34886\.claude\plans\soc-https-github-com-arpanbiswas99-batt-sprightly-cocke.md`

## Context

**目标**：把现有大创资产（MIUKF MATLAB 实现、STA 二阶 ECM 辨识、Simulink 30 模组、HUST 数据集、神经网络/SOCtarget）与外部高星仓库结合，实现 [CRAIC2026_方案调研.md](CRAIC2026_方案调研.md) 方向三的"AI 智能快充决策系统"端到端 demo——即 **Mamba 世界模型 + SAC 强化学习 + ECM 物理安全层** 的三层架构，输出动态充电策略并通过 RL 奖励函数显式惩罚电池老化损耗。

**资产盘点**：
- 本地仓库已有：MIUKF/STA（MATLAB）、Simulink 30 模组、LSTM+Attention SOC 模型、HUST 数据 80+ CSV
- 外部仓库选型：
  - SOC：[KeiLongW/battery-state-estimation](https://github.com/KeiLongW/battery-state-estimation)（185⭐，TF/Keras Stacked LSTM，含预训练权重）
  - SOH：[microsoft/BatteryML](https://github.com/microsoft/BatteryML)（740⭐，PyTorch + sklearn，原生支持 HUST loader）
  - 三层架构主干：`mamba-ssm` + `stable-baselines3` + 自有二阶 ECM
- 外部验证集：[Zenodo 6985321](https://zenodo.org/records/6985321)（WLTP+老化，仅供 W5 zero-shot 泛化验证，不参与训练）

**关键决策**：
1. SOC = LG 18650HG2 + KeiLongW 预训练权重（TF inference → CSV → PyTorch 下游不混框架）
2. SOH = HUST 数据 + BatteryML PyTorch 流水线
3. 电池包：先单体跑通三层，再接 Simulink 30 模组扩展
4. 减老化方案 = RL 奖励函数显式惩罚 SOH 衰减/温度积分

**数据分工合理性**：SOC、SOH 估计器独立训练符合 BMS 行业标准做法（"软传感器"角色）。RL 三层架构必须有同时含连续时序+老化标签的数据 → 选 NASA PCoE B0005/06/07/18，由训好的估计器在每时刻 inference 出 SOC/SOH 软标签。Zenodo 6985321 WLTP 数据在 W5 用于泛化验证。

---

## 系统架构

```
┌────────────────────────────────────────────────────────────────┐
│  阶段 A — 状态感知                                              │
│   LG 18650HG2 → KeiLongW LSTM (TF) → SOC                       │
│   HUST CSV    → BatteryML (PyTorch) → SOH                      │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼ [SOC, SOH, T, V, I]
┌────────────────────────────────────────────────────────────────┐
│  阶段 B — 三层决策架构                                          │
│  层1  Mamba 世界模型（mamba-ssm）                              │
│       训练数据：NASA PCoE B0005/06/07/18                       │
│  层2  SAC 强化学习（stable-baselines3）                        │
│       reward: +ΔSOC −α·|V−V_safe| −β·(T−T_ref)² −γ·ΔSOH_step  │
│  层3  ECM 物理安全层（PyTorch 重写自有二阶 RC）                │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
   单体最优充电曲线 / vs CC-CV/MFCC 对比 / 30 模组扩展
```

---

## 工作分解

| 周 | 主线任务 | 关键产出 |
|---|---|---|
| **W1** | (1) clone KeiLongW + 下 LG 18650HG2 + 跑通推断脚本 (2) clone BatteryML + 跑通 HUST loader + 训 baseline SOH | `soc_pred.csv`, `soh_baseline.pt` |
| **W2** | (1) 下 NASA PCoE + inference 软标签 → (V,I,T,SOC,SOH,action) 序列 (2) Mamba 世界模型训练 (3) ECM 安全层 PyTorch 重写 + 单元测试 vs MATLAB | `world_model.pt`, ECM 测试通过 |
| **W3** | (1) gymnasium 环境搭建 (2) 奖励函数调权重 (3) SAC 训练 100k steps | `sac_policy.zip` |
| **W4** | (1) vs CC-CV/MFCC 对比 (2) BatteryML 内 Mamba head SOH 对比表 (3) 拉 Zenodo 6985321 重建 SOC/SOH 标签 | 对比表 + 充电曲线图 |
| **W5** | (1) Simulink 30 模组协同 (2) Zenodo WLTP zero-shot 验证 (3) 答辩 Demo + PPT | Demo + 完整代码包 |

---

## 验证方案

### 估计器分层评估

| 验证层 | 数据 | SOC 目标 | SOH 目标 | 何时 |
|---|---|---|---|---|
| L1 训练域 | LG holdout / HUST holdout | MAE < 1% | RMSE < 2% SOH | W1 末 |
| L2 同源域（RL 用）| NASA PCoE B0005-B0018 | MAE < 2% | RMSE < 3%（fine-tune 后）| W2 中 |
| L3 自有基线 | LG + 本仓库 MIUKF/SOCtarget | 不弱于 MIUKF + SOCtarget | — | W2 末 |
| L4 真实驾驶 | Zenodo 6985321 (WLTP) | zero-shot 误差曲线 | 老化轨迹合理性 | W5 |

**SOH 跨化学体系兜底**：HUST LFP 1.1Ah → NASA NMC ~2Ah，预期裸迁移 RMSE 上升 2-3 倍。对策：在 NASA 上做最后一层 fine-tune（仅 head ~5% 参数，几分钟），把目标域误差压回阈值。

### 端到端测试（W4 末）

1. SOC 准确性：L1+L2 双达标
2. SOH 准确性：L1+L2 双达标
3. 世界模型：1 步 V 预测 MAE < 5 mV，20 步漂移 < 50 mV
4. ECM 安全层：随机 1000 条 SAC 动作，100% 满足电压边界
5. **RL 策略对比（核心交付）**：vs CC-CV 充至 80% 耗时 ↓ ≥ 15%、单循环 ΔSOH ↓ ≥ 10%、过压报警 = 0
6. 可解释性可视化：注意力热图、世界模型预测曲线、I(t) 对照图、跨数据集迁移误差曲线

---

## 风险与回退

| 风险 | 触发条件 | 回退策略 |
|---|---|---|
| TF/PyTorch 共存难 | 同进程加载冲突 | 解耦：TF 推断输出 CSV，PyTorch 只读 CSV |
| Mamba 世界模型不收敛 | W2 末 1 步预测 MAE > 20 mV | 退化为 2 层 GRU，方向三的 Mamba 创新点由 SOH 头继续承担 |
| BatteryML 装机失败 | W1 conda 装超 4h | 仅用其 HUST loader，自己 PyTorch 写训练循环 |
| RL 不收敛 | W3 末奖励曲线无改善 | 缩 state 维度 + PPO 替代 SAC + 降低 horizon |
| Simulink 协同接不上 | W5 时间不足 | 不做联仿，PPT 给"30 模组扩展规划"作为方向四演示 |

---

## 答辩故事线

- **创新度（C）**：Mamba 世界模型 + RL 奖励显式老化项是 2026 年文献空白
- **可行性（A）**：HUST/LG 公开数据 + 高星开源仓库降低复现门槛
- **难度（B）**：三层架构（世界模型 + SAC + 物理安全层）三模块叠加
- **团队贡献（E，权重 40%）**：自有 STA→ECM 安全层、Simulink 30 模组→包级扩展、自有 LSTM SOCtarget→SOC 第二基线、与现有专利（分数阶 KF）保持延续叙事
