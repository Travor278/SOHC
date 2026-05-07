# CRAIC2026 创新赛 · 方向三 — AI 智能快充决策系统

**Mamba 世界模型 + SAC 强化学习 + ECM 物理安全层** 的三层架构，实现 EV 电池动态最优充电策略，奖励函数显式惩罚电池老化损耗。

---

## 仓库结构

```
.
├── PLAN.md                          已批准的整体方案
├── TODO.md                          5 周滚动任务清单
├── CRAIC2026_方案调研.md            原始三方向调研（方向一/二/三）
├── requirements.txt                 Python 依赖
│
├── craic_pipeline/                  本作品的 Python 主代码包（骨架已搭，逐周填充）
│   ├── soc_inference.py             SOC：包装 KeiLongW Stacked LSTM (TF) 做 inference
│   ├── soh_train.py                 SOH：调 BatteryML HUST loader 训练
│   ├── world_model_mamba.py         层 1：Mamba 世界模型（NASA PCoE 训练）
│   ├── ecm_safety_layer.py          层 3：二阶 RC 投影器（PyTorch 重写自有 MATLAB）
│   ├── rl_env.py                    gymnasium 环境
│   ├── train_sac.py                 层 2：SAC 训练入口
│   └── eval_compare.py              vs CC-CV / MFCC / MIUKF 评估
│
├── configs/                         YAML 训练配置
│   └── hust_soh_baseline.yaml
│
├── external/                        外部仓库（git ignore，setup 时 clone）
│   └── README.md                    clone 指引
│
├── data/                            数据集（HUST 已携带，其余 git ignore）
│   ├── HUST data/                   80+ CSV，本仓库已含，用于 SOH 训练
│   ├── lg_hg2/                      LG 18650HG2 (Mendeley) — W1 下载
│   ├── nasa_pcoe/                   NASA PCoE B0005-B0018 — W2 下载
│   ├── zenodo_6985321/              Offenburg WLTP+老化 — W4/W5 下载
│   └── README.md                    数据集获取指引
│
├── scripts/
│   ├── setup_env.ps1                Windows venv 搭建脚本
│   ├── setup_env.sh                 Linux/macOS venv 搭建脚本
│   └── download_datasets.md         数据下载分步说明
│
├── outputs/                         模型权重、CSV、图（git ignore）
│
├── MATLAB滤波算法代码——云储实时数据/    现有大创资产：MIUKF + STA 二阶 ECM 辨识
├── Rebattery_Modeling-master/           现有大创资产：Simulink 30 模组
└── 神经网络/                            现有大创资产：LSTM SOC 模型（自训）
```

---

## 快速开始（在另一台机器上）

```powershell
# Windows
git clone <this-repo-url>
cd <repo>
.\scripts\setup_env.ps1
git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW
git clone https://github.com/microsoft/BatteryML.git external/BatteryML
# 按 data/README.md 下载三个数据集
```

```bash
# Linux / macOS
git clone <this-repo-url>
cd <repo>
bash scripts/setup_env.sh
git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW
git clone https://github.com/microsoft/BatteryML.git external/BatteryML
```

---

## 数据流（按周交付）

| 周 | 输入 | 模块 | 输出 |
|---|---|---|---|
| W1 | LG 18650HG2 V/I/T 时序 | `soc_inference.py` (KeiLongW) | `outputs/soc_pred_lg.csv` |
| W1 | HUST CSV 80+ 节 | `soh_train.py` (BatteryML) | `outputs/soh_baseline.pt` |
| W2 | NASA PCoE B0005-B0018 + W1 估计器 | `world_model_mamba.py` | `outputs/world_model.pt` |
| W2 | `savemat_2order.mat` | `ecm_safety_layer.py` | 单元测试 vs MATLAB |
| W3 | W2 世界模型 + W2 ECM | `train_sac.py` | `outputs/sac_policy.zip` |
| W4 | W3 策略 + 基线 | `eval_compare.py` | 对比表 + 充电曲线图 |
| W5 | Simulink 30 模组 + Zenodo WLTP | (联仿 + 泛化测试) | Demo + PPT |

---

## 当前状态

✅ 项目骨架就绪（W0 完成）
⬜ W1：SOC + SOH 估计器训练
⬜ W2：Mamba 世界模型 + ECM 安全层
⬜ W3：SAC 训练
⬜ W4：评估与基线对比
⬜ W5：包级扩展 + 答辩

详细进度见 [TODO.md](TODO.md)。
