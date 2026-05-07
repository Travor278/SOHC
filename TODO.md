# TODO — 5 周执行清单

> 详细方案见 [PLAN.md](PLAN.md)。本文件是滚动的可勾选清单，跑到哪改到哪。

## W0 · 环境准备（推到远端后在另一台机器上做）

- [ ] 在目标机器上 `git clone` 本仓库
- [ ] 跑 `scripts/setup_env.ps1`（Windows）或 `scripts/setup_env.sh`（Linux/macOS）创建 venv 并安装 `requirements.txt`
- [ ] `git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW`
- [ ] `git clone https://github.com/microsoft/BatteryML.git external/BatteryML`
- [ ] 按 `data/README.md` 指引下载 LG 18650HG2、NASA PCoE、Zenodo 6985321 三个数据集
- [ ] 验证 `data/HUST data/` 已存在（本仓库携带）
- [ ] 验证 `MATLAB滤波算法代码——云储实时数据/1-2-model_identification_RC/result/savemat_2order.mat` 已存在

## W1 · SOC + SOH 估计器

### SOC（KeiLongW）
- [ ] 跑 `external/KeiLongW/experiments/lg/lstm_soc_lg_*.ipynb` 拿到预训练权重 `.h5`
- [ ] 在 `craic_pipeline/soc_inference.py` 实现 `load_keilongw_model()`、`preprocess_sequence()`、`predict_soc()`
- [ ] 在 LG 18650HG2 测试集跑通，对齐论文 MAE 0.22%（允许偏差 → MAE < 1%）
- [ ] 输出 `outputs/soc_pred_lg.csv`

### SOH（BatteryML）
- [ ] 跑 `external/BatteryML/examples/soh_example.ipynb` 在内置数据上跑通
- [ ] 切到 HUST loader（`preprocess_HUST.py`），跑 `craic_pipeline/soh_train.py --config configs/hust_soh_baseline.yaml`
- [ ] 调通 baseline（Variance / CNN / MLP），HUST holdout RMSE < 2% SOH
- [ ] 保存 `outputs/soh_baseline.pt` + 训练日志

### 自有基线对接（L3 验证）
- [ ] 把本仓库 MATLAB MIUKF 输出的 SOC 导出 CSV，与 KeiLongW 输出对比
- [ ] 把本仓库 `神经网络/SOCtarget/` LSTM 推断的 SOC 也导出 CSV，三方对比

## W2 · Mamba 世界模型 + ECM 安全层

### NASA 软标签构造
- [ ] 下载 NASA PCoE B0005/06/07/18 `.mat` → `data/nasa_pcoe/`
- [ ] 写 NASA loader：解析 `.mat`，输出 (V, I, T, t, cycle_id) 时序
- [ ] 用 W1 的 SOC 估计器在 NASA 上 inference → SOC 软标签
- [ ] 用 W1 的 SOH 估计器在 NASA 每循环开始 inference → SOH 软标签
- [ ] **关键**：评估跨数据集迁移误差（L2），SOH RMSE 若 > 5% 则做 last-layer fine-tune
- [ ] 拼成 `(V, I, T, SOC, SOH, action)` shape (N, L, 6) tensor

### Mamba 世界模型
- [ ] 在 `craic_pipeline/world_model_mamba.py` 实现 `BatteryWorldModel`、`build_training_dataset()`
- [ ] 训练循环（MSE on next-step），50 epochs
- [ ] 验证：1 步 V 预测 MAE < 5 mV，20 步漂移 < 50 mV
- [ ] 退化预案：若 mamba-ssm 装不上，加 `--gru-fallback` 跑 GRU baseline
- [ ] 保存 `outputs/world_model.pt`

### ECM 安全层
- [ ] 在 `craic_pipeline/ecm_safety_layer.py` 实现 `load_params_from_mat()`、`ECMSafetyLayer`
- [ ] 单元测试 `cross_check_against_matlab()`：与 MIUK.m 输出电压差 < 1 mV
- [ ] 投影测试：随机 1000 条动作，100% 满足 V_min ≤ V_pred ≤ V_max

## W3 · RL 训练

- [ ] 在 `craic_pipeline/rl_env.py` 实现 `BatteryChargingEnv`（继承 `gymnasium.Env`）
- [ ] 实现 `compute_reward()`：4 项加权和（speed / V / T / aging）
- [ ] 单跑环境 1000 步，确认无异常
- [ ] 调奖励权重：先单项调（速度→安全→老化），再加权
- [ ] 跑 `train_sac.py --total-steps 100000`
- [ ] tensorboard 监控曲线：episode_return 应单调上升
- [ ] 保存 `outputs/sac_policy.zip`

## W4 · 评估与对比

- [ ] 在 `craic_pipeline/eval_compare.py` 实现 CC-CV、MFCC 基线
- [ ] 部署 SAC 策略，记录轨迹
- [ ] 计算指标表：充至 80% 耗时、ΔSOH 单循环、过压报警次数、平均 T
- [ ] 画 4 联子图（I/V/SOC/T over t）
- [ ] **核心交付**：vs CC-CV 充电速度 ↑ ≥ 15%、ΔSOH ↓ ≥ 10%、过压 = 0
- [ ] 在 BatteryML 里挂 Mamba head 跑一版 SOH，加进对比表
- [ ] 下载 Zenodo 6985321，用配套 .m 脚本里的 OCV-SOC 表 + 库仑积分重建标签

## W5 · 包级扩展 + 答辩

- [ ] Simulink 30 模组：每模组独立 SAC 策略，导出包级仿真
- [ ] Zenodo WLTP 上跑 zero-shot：本方案 SOC/SOH 估计在动态行驶下的误差曲线
- [ ] 端到端 Demo：插入电池参数 → 输出最优充电策略曲线
- [ ] 可视化：注意力热图 + 世界模型预测 vs 真实 + RL I(t) 对照 + 迁移误差曲线
- [ ] PPT：架构图、对比表、Demo 视频
- [ ] 完整代码包打 zip 提交

## 备忘 / 已知坑

- mamba-ssm 在 Windows + CUDA 12 上偶有装机问题。退路：用 WSL2 / Linux GPU 机；或 GRU fallback。
- BatteryML 依赖较重（含 PyTorch、PyG 等），首次 conda 装机预计 30-60 min。
- TF 和 PyTorch 同时 import 在某些 CUDA 版本下会冲突。原则：TF inference 出 CSV → 退出进程 → PyTorch 流水线读 CSV，不混进程。
- HUST 数据是循环聚合后的统计量（CSV），不是逐采样点时序——做 SOH 是对的（每循环一个标签），但不能直接拿来训 SOC 时序模型。
- NASA PCoE 是 NMC 18650，HUST 是 LFP 1.1Ah，化学体系不同 → SOH head fine-tune 是必经之路，别跳。
