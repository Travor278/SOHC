"""CRAIC2026 方向三 — Mamba 世界模型 + SAC + ECM 安全层 (v0.2)。

模块组织（按数据流）：
    soc_inference     KeiLongW LSTM warm-start + NASA ARC fine-tune → SOC
    soh_train         BatteryML trainer + NASA loader → SOH
    world_model_mamba Mamba 世界模型，NASA B0005-B0018 主集 + Randomized 增强
    ecm_safety_layer  二阶 RC 投影器（PyTorch 重写自 MATLAB MIUKF）
    rl_env            gymnasium 环境，step() 调 Mamba
    train_sac         stable-baselines3 SAC 训练
    eval_compare      vs CC-CV / MFCC / MIUKF + Zenodo 6985321/18471156 泛化展示
    pack_balance      多单体策略复制 + 包级 SOC-spread 均衡仿真

数据策略 v0.2：纯 NASA Plus 多子集同源训练，避免跨化学体系迁移误差。
"""

__version__ = "0.2.0"
