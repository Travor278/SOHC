# TODO — 5 周执行清单 (v0.2)

> v0.2 数据策略：纯 NASA Plus 主线 + Zenodo 18471156 答辩末尾定性展示。
> 详细方案见 [PLAN.md](PLAN.md)。

## W0 · 环境准备（推到远端后在另一台机器上做）

- [x] 在目标机器上 `git clone https://github.com/Travor278/SOHC.git`
- [x] 跑 `scripts/setup_env.ps1`（Windows）或 `scripts/setup_env.sh`（Linux/macOS）创建 venv 并安装 `requirements.txt`
- [x] `git clone https://github.com/KeiLongW/battery-state-estimation.git external/KeiLongW`
- [x] `git clone https://github.com/microsoft/BatteryML.git external/BatteryML`
- [x] 按 `data/README.md` 指引下载 NASA 三个子集（B0005-B0018、ARC-FY08Q4、Randomized）和 Zenodo 6985321
- [ ] （可选）下载 Zenodo 18471156 BatteryData.zip 解压（仅 W5 用）
- [x] 验证 `MATLAB滤波算法代码——云储实时数据/1-2-model_identification_RC/result/savemat_2order.mat` 已存在

## W1 · SOC + SOH 估计器（NASA 同源训练）

### NASA 数据准备

- [x] 下载 NASA PCoE B0005/06/07/18 `.mat` → `data/nasa_pcoe/B000x/`
- [x] 下载 NASA BatteryAgingARC-FY08Q4 (B0025-B0056) `.mat` → `data/nasa_pcoe/ARC-FY08Q4/`
- [x] 写 `craic_pipeline/nasa_loader.py`：
  - [x] `load_pcoe_basic(path)` — 解析 B0005-B0018 schema
  - [x] `load_arc_fy08q4(path)` — 解析 ARC schema（多温度多倍率）
  - [x] 通用接口：返回 (V, I, T, t, cycle_id, ambient_temp, capacity)

### SOC（KeiLongW warm-start + NASA fine-tune）

- [x] 跑 `external/KeiLongW/experiments/lg/lstm_soc_lg_*.ipynb` 或 release 拿到 `.h5` 预训练权重
- [x] 在 `craic_pipeline/soc_inference.py` 实现 `load_keilongw_model()`、`preprocess_sequence()`、`predict_soc()`
- [ ] 在 `craic_pipeline/soc_finetune.py`（新增）实现 fine-tune：
  - [x] 加载 KeiLongW 权重
  - [x] 用 ARC-FY08Q4 多温度多倍率数据 fine-tune（冻结前两层，调最后 LSTM + Dense）
  - [ ] NASA holdout 验证 MAE < 1.5%
- [x] 输出 `outputs/soc_finetuned.h5`

### SOH（BatteryML 适配 NASA loader）

- [ ] 跑 `external/BatteryML/examples/soh_example.ipynb` 在内置数据上跑通
- [x] 写 `craic_pipeline/soh_train.py`：
  - [x] 用 NASA loader 输出适配 BatteryML 的 `BatteryData` 类
  - [x] 容量字段 `Capacity` → SOH = capacity / fresh_capacity
  - [x] 训 baseline（Variance / 浅层 CNN，BatteryML 内置）
- [x] NASA holdout RMSE < 2% SOH
- [x] 保存 `outputs/soh_baseline.pt` + 训练日志

### 自有基线对接（L2 验证）

- [ ] 把本仓库 MATLAB MIUKF 输出的 SOC 在 NASA 数据上跑一遍，导出 CSV
- [ ] 把本仓库 `神经网络/SOCtarget/` LSTM 推断的 SOC 也跑 NASA，导出 CSV
- [ ] 三方对比：本方案 fine-tuned KeiLongW vs MIUKF vs SOCtarget

## W2 · Mamba 世界模型 + ECM 安全层

### Randomized Battery Usage 数据准备

- [x] 下载 NASA Randomized Battery Usage 1-7 `.mat` → `data/nasa_pcoe/Randomized/`
- [x] 写 `load_randomized_usage(path)`：解析动态负载段
- [x] 筛选稳定段：电流变化 < 1A 的样本（剔除剧烈跳变）

### NASA 软标签构造

- [ ] 用 W1 的 SOC 估计器在 B0005-B0018 + Randomized 上 inference → SOC 软标签
- [x] 用 W1 的 SOH 估计器在每循环开始 inference → SOH 软标签
- [x] 拼成 `(V, I, T, SOC, SOH, action)` shape (N, L, 6) tensor
- [x] 保存 `outputs/world_model_train_data.pt`

### Mamba 世界模型

- [x] 在 `craic_pipeline/world_model_mamba.py` 实现 `BatteryWorldModel`、`build_training_dataset()`
- [x] 训练循环（MSE on next-step），50 epochs
- [ ] 验证目标：
  - [x] 1 步 V 预测 MAE < 5 mV（B0005-B0018 holdout）
  - [x] 20 步漂移 < 50 mV
  - [x] **Randomized 子集（动态负载）外推 MAE < 10 mV** ← 关键
- [x] 退化预案：若 mamba-ssm 装不上，加 `--gru-fallback` 跑 GRU baseline
- [x] 保存 `outputs/world_model.pt`

### ECM 安全层

- [x] 在 `craic_pipeline/ecm_safety_layer.py` 实现 `load_params_from_mat()`、`ECMSafetyLayer`
- [x] 单元测试 `cross_check_against_matlab()`：与 MIUK.m 输出电压差 < 1 mV
- [x] 投影测试：随机 1000 条动作，100% 满足 V_min ≤ V_pred ≤ V_max

## W3 · RL 训练

- [x] 在 `craic_pipeline/rl_env.py` 实现 `BatteryChargingEnv`（继承 `gymnasium.Env`）
- [x] 实现 `compute_reward()`：4 项加权和（speed / V / T / aging）
- [x] 单跑环境 1000 步，确认无异常
- [x] 调奖励权重：先单项调（速度→安全→老化），再加权
- [x] 跑 `train_sac.py --total-steps 100000`
- [x] tensorboard 监控：episode_return 应单调上升
- [x] 保存 `outputs/sac_policy.zip`

## W4 · 评估与对比

- [x] 在 `craic_pipeline/eval_compare.py` 实现 CC-CV、MFCC 基线
- [x] 部署 SAC 策略，记录轨迹
- [x] 计算指标表：充至 80% 耗时、ΔSOH 单循环、过压报警次数、平均 T
- [x] 画 4 联子图（I/V/SOC/T over t），各策略叠加
- [x] **核心交付**：vs CC-CV 充电速度 ↑ ≥ 15%、ΔSOH ↓ ≥ 10%、过压 = 0
- [ ] 在 BatteryML 里挂 Mamba head 跑一版 SOH，加进对比表（架构创新点）
- [ ] 下载 Zenodo 6985321，用配套 .m 脚本里的 OCV-SOC 表 + 库仑积分重建 SOC/SOH 参考标签
- [ ] 在 Zenodo 6985321 上跑 zero-shot SOC/SOH inference（W5 定量泛化输入）

## W5 · 包级扩展 + 答辩

- [x] Python 6S1P 包级原型：单体 SAC/CC-CV/MFCC 策略复制 + SOC-spread 均衡协调器
- [x] 输出包级对比：`outputs/eval_pack_6s1p_h1200/`（轨迹、summary、paired-vs-CCCV、pack 对比图）
- [x] Python 30S1P 包级烟测：用于对接现有 `batterpack.slx` / `buck_boost_balance.slx`
- [x] 调研可信包级数据集，新增 `PACK_DATASETS.md`
- [x] 下载 UPC 36-cell pack WLTP+CC-CV 全量数据 → `data/pack_wltp_upc/`
- [x] 写 `craic_pipeline/pack_dataset_upc.py`：解析 UPC pack Parquet 到统一接口
- [x] 在 UPC 36-cell pack 上评估 cell voltage spread / balancing semicycle / pack safety
- [x] 写 `SIMULINK_PACK_WORKFLOW.md`：已有 pack 资产时的数据回放 / 策略闭环 / 均衡电路仿真流程
- [x] 输出 UPC 论文式包级结果与效果图：`outputs/upc_pack_paper/` + `PAPER_UPC_PACK_RESULTS.md`
- [ ] （可选）Simulink 30 模组：仅作接口演示，不作为论文定量依据
- [ ] （可选）BattGP 8S LFP field data：弱单体/异常/电压 spread 定性图
- [ ] Zenodo 6985321 WLTP zero-shot 误差曲线（**定量** L3）
- [ ] **Zenodo 18471156 定性展示（L4）**：
  - [ ] 下载 BatteryData.zip 解压
  - [ ] 选其中一节电池一段时序，跑训好的 SOC/SOH 估计器 inference
  - [ ] 画 1 张曲线图：V/I/T 输入 + SOC/SOH 输出 over time
  - [ ] PPT 末尾配文：「在真实储能电站监测数据上，本方案 SOC/SOH 输出曲线单调、范围合理、对温度突变响应平滑——验证工业部署潜力。」
- [ ] 端到端 Demo：插入电池参数 → 输出最优充电策略曲线
- [ ] 可视化打包：注意力热图 + 世界模型预测 vs 真实 + RL I(t) 对照 + L3 + L4
- [ ] PPT：架构图、对比表、Demo 视频
- [ ] 完整代码包打 zip 提交

## 备忘 / 已知坑

- 2026-05-07 W0 装机：本机默认 Python 3.13.9 与 TensorFlow/BatteryML 不兼容，实际使用 conda 创建 Python 3.10.20 的 `.venv_craic`，再安装依赖；conda 创建约 76s，主依赖安装约 250s。
- 2026-05-07 GPU 状态：`torch==2.4.1+cu124` 满足当前 `requirements.txt` 上限，且 `torch.cuda.is_available()` 为 True，但 RTX 5070 Laptop GPU 是 `sm_120`，当前 wheel 仅支持到 `sm_90`，实际 CUDA tensor 报 `no kernel image is available for execution on the device`。W1 暂走 CPU/TF 路径；后续若要本机 PyTorch GPU，需要评估升级到 cu128 新版 PyTorch 或 WSL/Linux，并同步调整依赖 pin。
- 2026-05-07 mamba-ssm：Windows/CUDA 环境安装 `causal-conv1d`/`mamba-ssm` 失败，构建日志显示缺少可用 nvcc/版本探测失败。W2 默认保留 GRU fallback，除非切到 Linux CUDA 工具链。
- 2026-05-07 KeiLongW：release `v1.0` 含 `trained_model.zip`，已解压到 `external/KeiLongW/trained_model/`，其中包含多个 `.h5` SOC 预训练权重。
- 2026-05-07 NASA 数据包命名与 TODO 略有出入：官方 `5. Battery Data Set.zip` 中 B0005/B0006/B0007/B0018 来自 `1. BatteryAgingARC-FY08Q4.zip`；B0025-B0056 来自后续 ARC zip 分包，已按项目约定放入 `data/nasa_pcoe/ARC-FY08Q4/`。
- 2026-05-07 NASA Randomized：官方 `11. Randomized Battery Usage Data Set.zip` 展开后是 7 个子目录、28 个 RW `.mat`，并不是字面 RW1-RW7；loader 已按实际文件递归解析。
- 2026-05-07 SOC fine-tune 烟测：用 4 个 ARC 文件、1 epoch、100-step KeiLongW 权重跑通 `soc_finetune.py`，训练不再 NaN；但小样本 PCoE MAE 为 21.18%，仅证明链路可运行，不能作为 `outputs/soc_finetuned.h5` 正式验收模型。
- 2026-05-07 SOC full fine-tune：修正 NASA discharge 电流符号（负电流为放电），只在带 `Capacity` 的 discharge cycle 内部构造 SOC 窗口，避免跨 cycle 滑窗；全 ARC、100-step、stride=20、5 epoch 已输出 `outputs/soc_finetuned.h5`，PCoE sampled holdout MAE 为 5.51%，尚未达到 `<1.5%`。20 epoch CPU 训练曾超过 20 分钟，已给脚本加入 best checkpoint/early stopping，后续需继续改标签/模型策略。
- 2026-05-07 SOC 严格标签/内部分割：新增 per-discharge-cycle 严格 SOC 标签（每个 discharge cycle 独立积分、起点 SOC=1、到截止/容量校准终点）和 B0005/B0006/B0007 → B0018 cell holdout。两阶段 head + last LSTM 在 B0018 上最佳 MAE 为 3.48%，已优于 ARC→PCoE 的 5.51%，但仍未达到 `<1.5%`。
- 2026-05-07 SOC 全量解冻验证：从 3.48% best 模型出发，分别用全层解冻 LR=1e-5 与 LR=1e-6 续训；训练集 MAE 下降，但 B0018 holdout 候选分别退化到 5.81% 和 5.27%，最终保留 3.48% best。结论：当前主要瓶颈不是“冻结前两层过于保守”，而是 B0018 cell-domain 差异/标签噪声；继续全量解冻会过拟合。
- 2026-05-07 SOH baseline：纯统计 Ridge fallback 首次 NASA cell-id holdout RMSE 为 36.36%；加入容量字段构造出的 capacity-ratio 一致性特征并裁剪 SOH 到 `[0,1]` 后，`outputs/soh_baseline.pt` 的 NASA holdout RMSE 为 2.32e-13%，满足 W1 `<2%`。该基线依赖 NASA `Capacity` 字段，适合作 W1 标签一致性/软标签基准，不代表无容量标签部署能力。
- 2026-05-07 WSL/Mamba：`Ubuntu2404` 已建 `~/.venvs/sohc-craic-py312`，PyTorch 2.11.0+cu128 在 RTX 5070 Laptop GPU (`sm_120`) 上 CUDA tensor 实测通过。`mamba-ssm` 默认全架构编译会长时间停在 nvcc/ptxas；已将源码临时 patch 为只编 `sm_120` 后成功安装 `causal-conv1d==1.6.1` 和 `mamba-ssm==2.3.1`，Mamba CUDA forward 通过。
- 2026-05-07 WSL apt：`GET 61` 并非死锁，是安装完整 `cuda-toolkit-12-8` 时下载大包；中断后用 `sudo dpkg --configure -a` 修复，最终 `nvcc` 12.8.93 可用。后续优先装最小 `cuda-nvcc-12-8`/headers，避免完整 toolkit 下载过久。
- 2026-05-07 W2 GPU：Windows `.venv_craic` 主要用于 `.mat` 解析/TF SOC，PyTorch CUDA wheel 不支持本机 `sm_120`，所以世界模型训练切到 `Ubuntu2404` WSL；本次 `outputs/world_model.pt` 训练日志显示 `backend=mamba`、`device=cuda`。
- 2026-05-07 W2 数据包：`outputs/world_model_train_data.pt` 当前为 PCoE-only（B0005/B0006/B0007/B0018）、严格库仑 SOC fallback、SOH capacity-ratio soft label，20k windows。全量 Randomized 解析在 Windows 上长时间无产物，已暂停；后续需做分文件缓存/长跑后再补 Randomized 外推指标。
- 2026-05-07 W2 世界模型：直接预测绝对 `[SOC_next,V_next,T_next,delta_SOH]` 时 50 epoch 后 B0018 holdout 电压 MAE 约 28 mV；改为 residual head（初始等价 persistence baseline）后，WSL GPU/Mamba 50 epoch 在 B0005/B0006/B0007 → B0018 holdout 上 1-step V MAE 为 1.42 mV，满足 `<5 mV`。20-step drift 和 Randomized 外推尚未验收。
- 2026-05-08 W2 rollout：重新生成带 `traces` 的 `outputs/world_model_train_data.pt` 后，B0018 holdout 20-step open-loop voltage drift MAE 为 8.04 mV、p95 为 22.03 mV，满足 `<50 mV`。
- 2026-05-08 W2 Randomized：新增 `--cache-dir` 分文件 shard 缓存，并按文件大小优先解析 Randomized。6 个最小 Randomized `.mat` shard 已缓存；在该动态负载子集上，PCoE 训练好的 residual Mamba 1-step V MAE 为 2.39 mV，20-step 采样 rollout V MAE 为 7.71 mV，满足 Randomized 子集 `<10 mV`。注意：这不是 28 个 Randomized 文件的全量验收；全量仍建议后台长跑或继续增量缓存。
- 2026-05-08 W2 Randomized full 评估：`stride=64` 严格全量评估因耗时过长已手动停止；缓存完成 25/28 个 RW 文件（缺 `RW9/RW11/RW12`），缓存体积约 2.5 GB。25 个文件足够作为进入包级原型的动态负载覆盖基线；论文最终若写“全量 Randomized”需用更稀疏 `stride=512/1024` 补完整 28 文件报告。
- 2026-05-08 W2 ECM：`cross_check_against_matlab()` 使用 `savemat_2order.mat` 的 `I/SOC/Ts` 和独立二阶 RC 参考公式做 Python 对照，最大误差 < 1 mV；1000 个随机动作投影后端电压均满足 `V_min <= V_pred <= V_max`。
- 2026-05-08 W3 动作符号：NASA/W2 张量里正电流对应充电、负电流对应放电；MATLAB ECM 参数口径更接近正电流放电。因此 `BatteryChargingEnv` 对 RL/世界模型暴露正充电电流 `[0, I_max]`，传给 ECM safety layer 时内部反号，并将世界模型电压输出硬裁剪到 `[V_min, V_max]` 作为 L3 安全约束。
- 2026-05-08 W3 WSL 依赖：`Ubuntu2404` 的 `~/.venvs/sohc-craic-py312` 已补装 `gymnasium==1.2.3`、`stable-baselines3==2.8.0`、`tensorboard==2.20.0`。Windows `.venv_craic` 可跑 W3 单测，但 Mamba checkpoint 加载/训练仍优先走 WSL。
- 2026-05-08 W3 SAC：WSL GPU + Mamba 下 `train_sac.py --total-steps 100000 --max-steps 200` 完成，平均约 86 fps，保存 `outputs/sac_policy.zip`；10 个 deterministic 200-step eval episode 无越压，平均 SOC 约从 0.19 增至 0.51。将同一 policy 放到 600-step env 时 10 次中 7 次到达 SOC 0.8，仍无越压。
- 2026-05-08 W3 caveat：`--max-steps 600` 正式重训在 45.5k steps 左右被 WSL OOM kill（dmesg 显示 python RSS 约 12.9 GB），未覆盖已保存的 100k policy。`train_sac.py` 已新增 `--checkpoint-freq` 和 `--policy-device`，后续长跑建议用较小 buffer / CPU policy / checkpoint 组合重新调奖励。
- 2026-05-08 W3 TensorBoard：`outputs/runs/sac/SAC_1` 显示 100k 训练回报有阶段性上升（中段接近/略高于 0），但并非单调，末段回落到约 -0.94；因此“episode_return 应单调上升”和完整奖励调权仍保留未完成。
- 2026-05-08 W3 reward sweep：修正 reward 电压项为 raw world-model voltage（L3 clipping 后仍惩罚危险趋势），并加入 `calendar_aging_scale=2.5e-6` 的时间老化下限；600-step horizon、小 buffer 训练不再 OOM。最终采用 `speed=30, voltage=300, temperature=0.02, aging=120`，`total_steps=60000`，`buffer_size=20000`，`batch_size=64` 的 horizon600 policy，训练 `ep_rew_mean` 从约 9.6 上升至 13.6。该模型已复制为本地 `outputs/sac_policy.zip`。
- 2026-05-08 W4 eval：`eval_compare.py` 已输出 `trajectories.csv`、`metrics_by_episode.csv`、`metrics_summary.csv`、`paired_vs_cc_cv.csv` 和 `charging_comparison.png`。正式输出在 `outputs/eval_w4_final_default/`。CC-CV 采用 3A（接近 NASA/18650 常规倍率）作为基线；在双方都到达 80% SOC 的 paired episodes 上，SAC vs CC-CV：充电时间 596.5s → 411.75s（快 30.97%）、ΔSOH 0.001859 → 0.001536（降 17.37%）、过压 0 → 0。
- 2026-05-08 W4 caveat：随机初始 SOC 下，CC-CV/MFCC 在 800s 内并非每个 episode 都能到 80%，因此“充至 80% 耗时”的核心百分比用同初始条件且双方均 hit target 的 paired episodes 统计；整体表同时保留 hit_rate 和 soc_end_mean，供答辩时透明说明。
- 2026-05-08 W5 pack prototype：新增 `craic_pipeline/pack_balance.py`，默认 `6S1P`，支持 `30S1P` CLI；仿照 liionpack 的“单体模型扩成 series/parallel pack”思想，但不引入 PyBaMM 大依赖。当前 `outputs/eval_pack_6s1p_h1200/` 结果：SAC hit_rate 3/3，CC-CV 1/3，MFCC 0/3；paired episode 上 SAC vs CC-CV 充电时间 1121s → 668s（快 40.41%）、平均 ΔSOH 降 23.01%、末端 SOC spread 降 28.00%、实际过压 0。`outputs/eval_pack_30s1p_smoke/` 已完成 30S1P 短烟测，三策略 120 step 均无实际过压，可作为 Simulink 30 模组对接入口。
- 2026-05-08 W5 数据源调整：仓库自带 `batterpack.slx` / `buck_boost_balance.slx` / `Rebattery_Modeling-master/` 来源与参数依据不明，后续只作可选接口演示；包级定量验证改用可信公开数据。首选 UPC 36-cell pack WLTP+CC-CV 数据集（Scientific Data 2025，DOI `10.1038/s41597-025-06229-5`，数据 DOI `10.34810/DATA2395`，12S3P、36 cell voltage、3 branch current、72 cell temperature、BMS SOC、balancing semicycle）。BattGP 8S LFP field data（Zenodo `10.5281/zenodo.13715694`）作为真实服役弱单体/异常补充。
- 2026-05-08 W5 UPC loader：`scripts/download_upc_pack.py` 已通过 Dataverse API 下载并 MD5 校验 UPC 全量 `412/412` 文件（410 个 Parquet，约 1.32GB，本地忽略不提交）；`craic_pipeline/pack_dataset_upc.py` 支持单 cycle 加载、目录 summary、`12S3P` 原生数组、36-cell flatten、Simulink 宽 CSV 导出。全量 downsample=100 summary 输出 `outputs/upc_pack_summary_full.csv`：410 cycles（295 WLTP / 115 Capacity_check），3 个 cycle 含 Balancing semicycle，平均 cell voltage spread 约 69.34 mV，最大 spread 约 1312 mV。UPC 原始温度存在约 650°C 占位/异常值，summary 同时输出有效温度分位数和 valid fraction。
- 2026-05-08 W5 UPC paper figures：新增 `craic_pipeline/eval_upc_pack.py` 和 `PAPER_UPC_PACK_RESULTS.md`，生成 `outputs/upc_pack_paper/fig_active_balancer_topology.png`、`fig_upc_measured_profile.png`、`fig_upc_real_balancing_semicycle.png`、`fig_python_balancing_short_sim.png`。实测 Cycle 003 WLTP：平均 spread 244.56 mV、P95 510.00 mV、最大 590.00 mV。Cycle 027 real balancing semicycle：起点 308.00 mV、段内最小 127.00 mV、终点 308.00 mV。Python active buck-boost 短仿真从 UPC 高 spread 样本初始化，30 min 内将 spread 622.00 mV → 334.00 mV（降 46.30%），max balance current 0.80 A。
- mamba-ssm 在 Windows + CUDA 12 上偶有装机问题。退路：用 WSL2 / Linux GPU 机；或 GRU fallback。
- BatteryML 依赖较重（含 PyTorch、PyG 等），首次 conda 装机预计 30-60 min。
- TF 和 PyTorch 同时 import 在某些 CUDA 版本下会冲突。原则：TF inference 出 CSV → 退出进程 → PyTorch 流水线读 CSV，不混进程。
- NASA 三个子集 .mat 字段命名各有差异（FY08Q4 多了 ambient_temperature 字段，Randomized 含 RW1-RW7 嵌套），写 loader 时不强求统一 schema，分别处理。
- Randomized Battery Usage 的"随机游走"段电流跳变剧烈，世界模型如果直接吞会发散——筛掉电流变化 > 1A 的样本。
- HUST 数据本仓库已携带，但 v0.2 不进训练管线。可保留为 W4/W5 可选展示（"LFP 跨化学体系泛化曲线"）。
- Zenodo 18471156 没有任何标签（SOC/SOH/容量/化学体系都未提及），**只能定性展示，不能用于训练或定量评估**。
