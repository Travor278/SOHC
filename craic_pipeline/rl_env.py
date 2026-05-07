"""Gymnasium 环境：把 Mamba 世界模型 + ECM 安全层包成 RL env。

定位：方向三三层架构的【层 2】(SAC) 接入点。

state  : np.array([SOC, SOH, V, T, age_proxy])  shape (5,)
action : np.array([I_charge_normalized])        shape (1,) ∈ [-1, 1]
reward : +ΔSOC*w_speed
         −α*max(0, V−V_safe)
         −β*(T−T_ref)²
         −γ*ΔSOH_step           ◄── 显式惩罚老化
done   : SOC ≥ 0.8 (充电完成) | V 越界 | T 越界 | step ≥ max_steps

env.step():
    1. 用 ECMSafetyLayer.project(action) 把动作投影到安全区
    2. 用 BatteryWorldModel.step(state, safe_action) 推下一步
    3. 计算 reward + done
"""
from __future__ import annotations

from dataclasses import dataclass

# 这里用 typing 占位，实际 import 在 main 实现里做
# import gymnasium as gym
# from gymnasium import spaces


@dataclass
class RewardWeights:
    speed: float = 1.0     # +ΔSOC
    voltage: float = 5.0   # −α
    temperature: float = 0.1  # −β
    aging: float = 10.0    # −γ，老化项权重最大（方向三的 USP）


@dataclass
class EnvConfig:
    max_steps: int = 600
    dt: float = 1.0
    soc_target: float = 0.8
    V_max: float = 4.2
    V_min: float = 2.5
    T_max: float = 50.0
    T_ref: float = 25.0
    I_max_amps: float = 5.0
    reward: RewardWeights = None

    def __post_init__(self):
        if self.reward is None:
            self.reward = RewardWeights()


class BatteryChargingEnv:
    """gymnasium.Env 子类。world_model 和 safety_layer 在 __init__ 注入。"""

    def __init__(self, world_model, safety_layer, cfg: EnvConfig):
        self.wm = world_model
        self.safety = safety_layer
        self.cfg = cfg
        self.steps = 0
        # action_space, observation_space 在实现时定义
        raise NotImplementedError("W3: 继承 gym.Env 并定义 spaces")

    def reset(self, *, seed=None, options=None):
        """随机初始化 SOC ∈ [0.1, 0.4], SOH ∈ [0.7, 1.0], T ~ N(25, 3)。"""
        raise NotImplementedError("W3")

    def step(self, action):
        # TODO (W3):
        # 1. action_amps = denormalize(action) * cfg.I_max_amps
        # 2. safe_amps = self.safety.project(self.state.soc, action_amps)
        # 3. next_state = self.wm.step(self.state, safe_amps)
        # 4. reward = compute_reward(self.state, next_state)
        # 5. done = check_termination(next_state)
        # 6. return next_state, reward, done, truncated, info
        raise NotImplementedError("W3")

    def compute_reward(self, s_t, s_tp1) -> float:
        """4 项奖励的加权和，权重见 RewardWeights。"""
        raise NotImplementedError("W3")
