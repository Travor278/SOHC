# CRAIC2026 创新赛 · 方向三 — AI 智能快充决策系统

**Mamba 世界模型 + SAC 强化学习 + ECM 物理安全层** 的三层架构，实现 EV 电池动态最优充电策略，奖励函数显式惩罚电池老化损耗。

> **当前版本：v0.2** — 训练数据策略调整为"纯 NASA Plus 多子集同源"，详见底部"v0.2 变更日志"。

---

## 仓库结构

```
.
├── PLAN.md                          已批准方案 v0.2
├── TODO.md                          5 周滚动任务清单 v0.2
├── CRAIC2026_方案调研.md            原始三方向调研
├── requirements.txt                 Python 依赖
│
├── craic_pipeline/                  本作品的 Python 主代码包（骨架已搭，逐周填充）
│   ├── soc_inference.py             SOC：KeiLongW Stacked LSTM warm-start + NASA fine-tune
│   ├── soh_train.py                 SOH：BatteryML trainer + NASA loader
│   ├── world_model_mamba.py         层 1：Mamba 世界模型（NASA 训练）
│   ├── ecm_safety_layer.py          层 3：二阶 RC 投影器
│   ├── rl_env.py                    gymnasium 环境
│   ├── train_sac.py                 层 2：SAC 训练入口
│   └── eval_compare.py              vs CC-CV / MFCC / MIUKF 评估
│
├── configs/                         YAML 训练配置
├── external/                        外部仓库（git ignore，setup 时 clone）
├── data/                            数据集（HUST 已携带，其余 git ignore，按 README 下载）
├── outputs/                         模型权重、CSV、图（git ignore）
├── scripts/                         环境搭建脚本
│
├── MATLAB滤波算法代码——云储实时数据/    现有大创资产：MIUKF + STA 二阶 ECM 辨识
├── Rebattery_Modeling-master/           现有大创资产：Simulink 30 模组
└── 神经网络/                            现有大创资产：LSTM SOC 模型（自训）
```

---

## 快速开始（在另一台机器上）

```powershell
# Windows
git clone https://github.com/Travor278/SOHC.git
cd SOHC
.\scripts\setup_env.ps1
git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW
git clone https://github.com/microsoft/BatteryML.git external/BatteryML
# 按 data/README.md 下载 NASA 三个子集 + Zenodo 6985321
```

```bash
# Linux / macOS
git clone https://github.com/Travor278/SOHC.git
cd SOHC
bash scripts/setup_env.sh
git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW
git clone https://github.com/microsoft/BatteryML.git external/BatteryML
```

---

## 数据流（v0.2，按周交付）

| 周 | 输入 | 模块 | 输出 |
|---|---|---|---|
| W1 | KeiLongW 预训练权重 + NASA ARC-FY08Q4（多温度多倍率）| `soc_inference.py` + fine-tune | `outputs/soc_finetuned.h5` |
| W1 | NASA B0005-B0018 + ARC 容量退化 | `soh_train.py` (BatteryML) | `outputs/soh_baseline.pt` |
| W2 | NASA B0005-B0018 + Randomized Battery Usage + W1 估计器 | `world_model_mamba.py` | `outputs/world_model.pt` |
| W2 | `savemat_2order.mat` | `ecm_safety_layer.py` | 单元测试 vs MATLAB |
| W3 | W2 世界模型 + W2 ECM | `train_sac.py` | `outputs/sac_policy.zip` |
| W4 | W3 策略 + 基线 + Zenodo 6985321 | `eval_compare.py` | 对比表 + 充电曲线图 + WLTP 泛化曲线 |
| W5 | Simulink 30 模组 + Zenodo 18471156 | (联仿 + 定性展示) | Demo + PPT |

**所有训练数据在化学体系上同源（NMC 18650）**，无需跨化学体系 fine-tune。

---

## 当前状态

- ✅ W0：项目骨架就绪（v0.2 已 push）
- ⬜ W1：SOC + SOH 估计器（NASA 同源训练）
- ⬜ W2：Mamba 世界模型 + ECM 安全层
- ⬜ W3：SAC 训练
- ⬜ W4：评估与基线对比
- ⬜ W5：包级扩展 + 真实电站定性展示 + 答辩

详细进度见 [TODO.md](TODO.md)。

---

## v0.2 变更日志

**v0.1 → v0.2**（数据策略重构）：

| 维度 | v0.1 | v0.2 |
|---|---|---|
| SOC 训练数据 | LG 18650HG2（KeiLongW 复用预训练）| KeiLongW 预训练权重 warm-start + NASA ARC-FY08Q4 多温度多倍率 fine-tune |
| SOH 训练数据 | HUST CSV（LFP 1.1Ah）| NASA B0005-B0018 + ARC-FY08Q4（NMC 18650 同源）|
| Mamba 世界模型数据 | NASA B0005-B0018 + LFP→NMC fine-tune | NASA B0005-B0018 + Randomized Battery Usage（动态负载多样性）|
| 跨化学体系 fine-tune | 必需（HUST LFP → NASA NMC）| **取消**（全程同源）|
| 答辩故事 | 跨多数据集泛化 | NASA 标准基准 SOTA + 真实电站定性外推 |
| Zenodo 6985321 | W4/W5 zero-shot 验证 | 不变（W5 定量泛化）|
| Zenodo 18471156 | 未提及 | **新增**（W5 PPT 末尾 1 张图，定性展示）|
| HUST 数据 | 训练主集 | 退为可选 LFP 跨化学体系泛化展示（不进训练管线）|

**主要收益**：
- 砍掉跨化学体系 fine-tune 兜底逻辑（W2 工作量 −20%）
- 数据流稳健性提升（同源数据，无迁移误差风险）
- 答辩故事更易被评委验证（NASA 是 SOC/SOH 论文金标准基准）

**新增成本**：
- 总下载量从 ~50 MB 增至 ~500 MB-1 GB
- 需写 3 个 NASA 子集 loader（每子集 .mat schema 略有差异）
