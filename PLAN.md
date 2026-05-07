# CRAIC2026 创新赛 · 方向三整合实施方案 (v0.2)

> v0.2 变更：训练数据从"LG SOC + HUST SOH + NASA RL"切换为"纯 NASA Plus 多子集"。
> 详细缘由见 [README.md](README.md) 末尾"v0.2 变更日志"，决策过程见 GitHub PR / commit log。
>
> 已批准计划镜像：`C:\Users\34886\.claude\plans\soc-https-github-com-arpanbiswas99-batt-sprightly-cocke.md`

## Context

**目标**：把现有大创资产（MIUKF MATLAB、STA 二阶 ECM 辨识、Simulink 30 模组、神经网络/SOCtarget LSTM）与外部高星仓库结合，实现 [CRAIC2026_方案调研.md](CRAIC2026_方案调研.md) 方向三的"AI 智能快充决策系统"端到端 demo——即 **Mamba 世界模型 + SAC 强化学习 + ECM 物理安全层** 的三层架构，输出动态充电策略并通过 RL 奖励函数显式惩罚电池老化损耗。

**v0.2 数据策略**：**所有训练 + 定量评估都在 NASA PCoE 同源数据上完成**，避免跨数据集迁移误差。NASA 内部多子集组合提供温度/倍率/动态负载多样性。LG/HUST 退为辅助角色，Zenodo 18471156 作为答辩末尾定性展示。

**资产盘点**：
- 本地仓库已有：MIUKF/STA（MATLAB）、Simulink 30 模组、LSTM+Attention SOC 模型、HUST CSV 80+
- 外部仓库选型：
  - SOC：[KeiLongW/battery-state-estimation](https://github.com/KeiLongW/battery-state-estimation)（185⭐，TF/Keras Stacked LSTM）—— 仅取**预训练权重做 warm-start**，再用 NASA 数据 fine-tune
  - SOH：[microsoft/BatteryML](https://github.com/microsoft/BatteryML)（740⭐，PyTorch 多算法 benchmark）—— 用其 trainer 框架，loader 自写 NASA 版本
  - 三层架构主干：`mamba-ssm` + `stable-baselines3` + 自有二阶 ECM
- **训练数据（NASA Plus 三子集）**：
  - **NASA PCoE B0005/06/07/18**（4 节 NMC 18650，~150 循环/节，24°C，CC-CV 充电）→ SOH 时序退化训练 + 世界模型主训练
  - **NASA BatteryAgingARC-FY08Q4 (B0025-B0056)**（~32 节 NMC 18650，**4°C / 24°C / 43°C 多温度，多放电倍率**）→ SOC 估计器训练（多温度多倍率）
  - **NASA Randomized Battery Usage 1-7**（动态负载随机游走，0.5-4A）→ 世界模型动态外推训练（对抗 RL 探索分布外问题）
- **辅助数据**：
  - LG 18650HG2：仅作 KeiLongW 预训练权重的训练源（不下载到本地）
  - HUST 数据（仓库已携带）：可选的 LFP 跨化学体系泛化展示
  - Zenodo 6985321（WLTP+老化）：W5 真实驾驶 zero-shot 验证
  - Zenodo 18471156（电站现场监测）：W5 答辩末尾定性展示，**不参与训练 / 不参与定量评估**

**关键决策**：
1. **SOC** = KeiLongW 预训练权重 warm-start + NASA ARC 多温度多倍率 fine-tune
2. **SOH** = NASA B0005-B0018 + ARC 子集训练（同源 NMC 18650，无跨化学体系问题）
3. **Mamba 世界模型** = NASA B0005-B0018 + Randomized Battery Usage（解决动作外推）
4. **SAC** = 在 Mamba env 上 100k steps，reward 显式惩罚 ΔSOH
5. **电池包**：先单体跑通三层，再接 Simulink 30 模组扩展
6. **18471156 用法**：仅 W5 PPT 末尾 1 张图（"在真实电站监测数据上的输出曲线合理性"），不进训练管线

**v0.2 vs v0.1 差异**：
- ✅ 砍掉 SOH 跨化学体系（LFP→NMC）fine-tune 的兜底逻辑（W2 任务量 −20%）
- ✅ 答辩故事重定向：从"跨多数据集泛化"改为"NASA 标准基准 + 真实电站定性外推"——评委更熟悉、更易验证
- ✅ 化学体系全程同源（NMC 18650），数据流更稳健
- ⚠️ 总下载量：~50 MB → ~500 MB-1 GB（多了 ARC 和 Randomized 子集）
- ⚠️ 需写 3 个 NASA loader（每子集 .mat 字段命名略有差异）

---

## 系统架构（v0.2 数据流）

```
┌────────────────────────────────────────────────────────────────┐
│  阶段 A — 状态感知                                              │
│                                                                │
│   LG 18650HG2 (KeiLongW 预训练) ─┐                              │
│                                  │ warm-start                  │
│   NASA ARC-FY08Q4 (多温度多倍率)─┼─► SOC 估计器 (TF/Keras LSTM) │
│                                                                │
│   NASA B0005-B0018 + ARC 容量退化 ──► SOH 估计器 (BatteryML)    │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼ [SOC, SOH, T, V, I]
┌────────────────────────────────────────────────────────────────┐
│  阶段 B — 三层决策架构                                          │
│  层1  Mamba 世界模型                                           │
│       训练数据：NASA B0005-B0018 (CC-CV) + Randomized (动态)    │
│       输入: [SOC_t, SOH_t, V_t, I_t, T_t, action_t]             │
│       输出: [SOC_{t+1}, V_{t+1}, T_{t+1}, ΔSOH_step]            │
│  层2  SAC 强化学习 (stable-baselines3)                         │
│       reward: +ΔSOC −α·|V−V_safe| −β·(T−T_ref)² −γ·ΔSOH_step    │
│  层3  ECM 物理安全层（自有 STA 辨识参数 PyTorch 重写）          │
└────────────────────────────────────────────────────────────────┘
                            │
                            ▼
   单体最优充电曲线 / vs CC-CV/MFCC 对比 / 30 模组扩展
   + Zenodo 6985321 WLTP zero-shot 泛化曲线
   + Zenodo 18471156 真实电站定性展示（PPT 末尾）
```

---

## 工作分解

| 周 | 主线任务 | 关键产出 |
|---|---|---|
| **W1** | (1) clone KeiLongW + 提取预训练权重 (2) 下 NASA ARC-FY08Q4，写 loader (3) 在 ARC 上 fine-tune SOC 估计器（KeiLongW 权重打底）(4) clone BatteryML + 写 NASA loader 适配 BatteryML trainer，跑 baseline SOH | `soc_finetuned.h5`, `soh_baseline.pt`, NASA ARC loader |
| **W2** | (1) 下 NASA B0005-B0018 + Randomized Battery Usage (2) 用 W1 估计器 inference 出软标签 → (V,I,T,SOC,SOH,action) 序列 (3) Mamba 世界模型训练（B0005-B0018 主 + Randomized 增强动作多样性）(4) ECM 安全层 PyTorch 重写 + 单元测试 vs MATLAB | `world_model.pt`, ECM 测试通过 |
| **W3** | (1) gymnasium 环境搭建 (2) 奖励函数调权重（speed→safety→aging）(3) SAC 训练 100k steps | `sac_policy.zip` |
| **W4** | (1) vs CC-CV/MFCC 对比，指标=充至 80% 时间 / ΔSOH / 过压报警次数 (2) BatteryML 内挂 Mamba head 跑 SOH 对比表（架构创新点） (3) 拉 Zenodo 6985321，重建 SOC/SOH 参考标签 | 对比表 + 充电曲线图 |
| **W5** | (1) Simulink 30 模组协同 (2) Zenodo 6985321 WLTP zero-shot 验证（定量）(3) Zenodo 18471156 电站定性展示（1 张图）(4) 答辩 Demo + PPT | Demo + 完整代码包 |

---

## 验证方案

### 估计器分层评估（v0.2 简化版）

| 验证层 | 数据 | SOC 目标 | SOH 目标 | 何时 |
|---|---|---|---|---|
| **L1 NASA 训练域** | NASA holdout（按电池 ID 切）| MAE < 1.5% | RMSE < 2% SOH | W1 末 |
| **L2 自有基线对比** | NASA + MIUKF + 神经网络/SOCtarget | 不弱于 MIUKF | — | W2 末 |
| **L3 真实驾驶（定量）** | Zenodo 6985321 (WLTP) | zero-shot MAE 曲线（不要求阈值，看趋势）| 老化轨迹合理性 | W5 |
| **L4 真实电站（定性）** | Zenodo 18471156 | 输出曲线单调/范围/温度响应合理 | 同左 | W5 |

**v0.2 砍掉的兜底**：原 v0.1 里"SOH 跨化学体系 fine-tune"已不必要（化学体系全程同源 NMC 18650）。

### 端到端测试（W4 末）

1. SOC 准确性：L1 NASA 达标
2. SOH 准确性：L1 NASA 达标
3. 世界模型：1 步 V 预测 MAE < 5 mV，20 步漂移 < 50 mV，**动态负载（Randomized 子集）外推 MAE < 10 mV**
4. ECM 安全层：随机 1000 条 SAC 动作，100% 满足电压边界
5. **RL 策略对比（核心交付）**：vs CC-CV 充至 80% 耗时 ↓ ≥ 15%、单循环 ΔSOH ↓ ≥ 10%、过压报警 = 0
6. 可解释性可视化：注意力热图、世界模型预测曲线、I(t) 对照、L3+L4 泛化曲线

---

## 风险与回退

| 风险 | 触发条件 | 回退策略 |
|---|---|---|
| TF/PyTorch 共存难 | 同进程加载冲突 | 解耦：TF 推断输出 CSV，PyTorch 只读 CSV |
| Mamba 世界模型不收敛 | W2 末 1 步预测 MAE > 20 mV | 退化为 2 层 GRU，方向三的 Mamba 创新点由 SOH 头继续承担 |
| BatteryML 装机失败 | W1 conda 装超 4h | 仅用其 trainer 接口，HUST loader 自写 PyTorch 版本 |
| RL 不收敛 | W3 末奖励曲线无改善 | 缩 state 维度 + PPO 替代 SAC + 降低 horizon |
| NASA 子集 .mat 字段对不齐 | W1/W2 解析报错 | 写 3 个独立 loader，不强求统一 schema |
| Randomized 子集动作太随机 | 世界模型在 Randomized 上学不到稳定动力学 | 仅用 Randomized 的轻度扰动段（电流变化 < 1A），剔除剧烈段 |
| Simulink 协同接不上 | W5 时间不足 | 不做联仿，PPT 给"30 模组扩展规划"作为方向四演示 |

---

## 答辩故事线（v0.2）

- **创新度（C）**：Mamba 世界模型 + RL 奖励显式老化项是 2026 年文献空白；BatteryML 内 Mamba head 跑 SOH（W4）是架构创新点
- **可行性（A）**：NASA 标准基准 + 高星开源仓库降低复现门槛；Zenodo 18471156 真实电站监测数据展示工业部署潜力
- **难度（B）**：三层架构（世界模型 + SAC + 物理安全层）三模块叠加；多 NASA 子集 loader 工程化
- **团队贡献（E，权重 40%）**：自有 STA→ECM 安全层；自有 Simulink 30 模组→包级扩展；自有 LSTM SOCtarget→SOC 第二基线；与现有专利（分数阶 KF）保持延续叙事
- **新故事重点**：从"跨多数据集泛化"改为 **"NASA 标准基准 SOTA + 真实电站定性外推"**——前者是论文常规故事，后者是评委更易验证的工业落地论证
