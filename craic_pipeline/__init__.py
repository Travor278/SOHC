"""CRAIC2026 方向三 — Mamba 世界模型 + SAC + ECM 安全层。

模块组织（按数据流）：
    soc_inference     KeiLongW LSTM 推断 → SOC CSV
    soh_train         BatteryML 在 HUST 上训 SOH → .pt
    world_model_mamba Mamba 世界模型，吃 NASA PCoE
    ecm_safety_layer  二阶 RC 投影器（PyTorch）
    rl_env            gymnasium 环境，step() 调 Mamba
    train_sac         stable-baselines3 SAC 训练
    eval_compare      vs CC-CV / MFCC / MIUKF 对比
"""

__version__ = "0.1.0"
