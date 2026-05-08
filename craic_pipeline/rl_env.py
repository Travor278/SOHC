"""Gymnasium environment wrapping the Mamba world model and ECM safety layer.

State: `[SOC, SOH, V, I, T]`.
Action: normalized charging command in `[-1, 1]`, mapped to current
`[0, I_max]` A in the NASA/W2 sign convention, where positive current is
charge. The legacy MATLAB ECM uses the opposite current polarity internally.

Reward uses the W3 four-term design: speed, voltage safety, temperature, and
explicit SOH-aging penalty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - import error is clearer at construction.
    gym = None
    spaces = None


@dataclass
class RewardWeights:
    """Reward weights for speed, safety, temperature, and aging terms."""

    speed: float = 12.0
    voltage: float = 50.0
    temperature: float = 0.2
    aging: float = 80.0


@dataclass
class EnvConfig:
    """Configuration for the SAC charging environment."""

    max_steps: int = 600
    dt: float = 1.0
    soc_target: float = 0.8
    V_max: float = 4.2
    V_min: float = 2.5
    T_max: float = 50.0
    T_ref: float = 25.0
    I_max_amps: float = 5.0
    initial_soc_low: float = 0.1
    initial_soc_high: float = 0.4
    initial_soh_low: float = 0.85
    initial_soh_high: float = 1.0
    initial_voltage: float = 3.7
    seq_len: int = 64
    device: str = "cpu"
    aging_proxy_scale: float = 1e-6
    calendar_aging_scale: float = 2.5e-6
    reward: RewardWeights = None

    def __post_init__(self):
        if self.reward is None:
            self.reward = RewardWeights()


class BatteryChargingEnv(gym.Env if gym is not None else object):
    """Gymnasium environment for W3 SAC charging policy training."""

    def __init__(self, world_model, safety_layer, cfg: EnvConfig):
        """Create an env from a trained world model and ECM safety projector."""
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for BatteryChargingEnv")
        self.wm = world_model
        self.safety = safety_layer
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.wm.to(self.device)
        self.wm.eval()
        self.steps = 0
        self.state = np.zeros(5, dtype=np.float32)
        self.history = np.zeros((cfg.seq_len, 6), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -20.0], dtype=np.float32),
            high=np.array([1.2, 1.2, 5.0, cfg.I_max_amps, 100.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.last_world_model_voltage_raw = np.nan
        self.last_model_delta_soh = 0.0
        self.last_aging_proxy_delta_soh = 0.0

    def reset(self, *, seed=None, options=None):
        """Reset to a random low-SOC charging state and clear ECM/history."""
        super().reset(seed=seed)
        rng = self.np_random
        soc = float(rng.uniform(self.cfg.initial_soc_low, self.cfg.initial_soc_high))
        soh = float(rng.uniform(self.cfg.initial_soh_low, self.cfg.initial_soh_high))
        temp = float(np.clip(rng.normal(self.cfg.T_ref, 3.0), -10.0, self.cfg.T_max - 1.0))
        voltage = float(np.clip(self.cfg.initial_voltage + 0.7 * (soc - 0.25), self.cfg.V_min, self.cfg.V_max))
        self.state = np.array([soc, soh, voltage, 0.0, temp], dtype=np.float32)
        self.steps = 0
        self.safety.reset()
        self._reset_history(self.state)
        return self.state.copy(), {}

    def step(self, action):
        """Project action through ECM, advance the world model, and score reward."""
        raw_action = float(np.asarray(action, dtype=float).reshape(-1)[0])
        raw_action = float(np.clip(raw_action, -1.0, 1.0))
        requested_current = 0.5 * (raw_action + 1.0) * self.cfg.I_max_amps
        safe_ecm_current = self.safety.project(soc=float(self.state[0]), action_current=-requested_current)
        safe_current = float(np.clip(-safe_ecm_current, 0.0, self.cfg.I_max_amps))
        previous = self.state.copy()
        next_state = self._world_step(previous, safe_current)
        self.state = next_state.astype(np.float32)
        self.steps += 1
        reward, reward_terms = self.compute_reward(previous, self.state, raw_voltage=self.last_world_model_voltage_raw)
        terminated = bool(
            self.state[0] >= self.cfg.soc_target
            or self.state[2] < self.cfg.V_min - 1e-5
            or self.state[2] > self.cfg.V_max + 1e-5
            or self.state[4] > self.cfg.T_max
        )
        truncated = bool(self.steps >= self.cfg.max_steps)
        info = {
            "requested_current": requested_current,
            "safe_current": safe_current,
            "safe_ecm_current": safe_ecm_current,
            "world_model_voltage_raw": float(self.last_world_model_voltage_raw),
            "model_delta_soh": float(self.last_model_delta_soh),
            "aging_proxy_delta_soh": float(self.last_aging_proxy_delta_soh),
            "reward_terms": reward_terms,
            "soc": float(self.state[0]),
            "soh": float(self.state[1]),
            "voltage": float(self.state[2]),
            "temperature": float(self.state[4]),
        }
        return self.state.copy(), float(reward), terminated, truncated, info

    def compute_reward(self, s_t, s_tp1, *, raw_voltage: float | None = None) -> tuple[float, dict]:
        """Compute the four-term W3 reward and expose component diagnostics."""
        delta_soc = float(s_tp1[0] - s_t[0])
        delta_soh = max(float(s_t[1] - s_tp1[1]), 0.0)
        voltage_for_penalty = float(s_tp1[2] if raw_voltage is None else raw_voltage)
        high_v = max(float(voltage_for_penalty - self.cfg.V_max), 0.0)
        low_v = max(float(self.cfg.V_min - voltage_for_penalty), 0.0)
        voltage_penalty = high_v * high_v + low_v * low_v
        temp_penalty = ((float(s_tp1[4]) - self.cfg.T_ref) / 10.0) ** 2
        reward = (
            self.cfg.reward.speed * delta_soc
            - self.cfg.reward.voltage * voltage_penalty
            - self.cfg.reward.temperature * temp_penalty
            - self.cfg.reward.aging * delta_soh
        )
        return float(reward), {
            "speed": self.cfg.reward.speed * delta_soc,
            "voltage_penalty": self.cfg.reward.voltage * voltage_penalty,
            "temperature_penalty": self.cfg.reward.temperature * temp_penalty,
            "aging_penalty": self.cfg.reward.aging * delta_soh,
        }

    def _world_step(self, state: np.ndarray, safe_current: float) -> np.ndarray:
        """Run one sequence-conditioned world-model transition."""
        model_input = self.history.copy()
        model_input[-1] = np.array([state[0], state[1], state[2], state[3], state[4], safe_current], dtype=np.float32)
        with torch.no_grad():
            pred = self.wm(torch.from_numpy(model_input).unsqueeze(0).to(self.device)).detach().cpu().numpy()[0]
        model_delta_soh = max(float(pred[3]), 0.0)
        proxy_delta_soh = self._aging_proxy_delta_soh(safe_current, raw_voltage=float(pred[1]), temperature=float(pred[2]))
        delta_soh = max(model_delta_soh, proxy_delta_soh)
        self.last_model_delta_soh = model_delta_soh
        self.last_aging_proxy_delta_soh = proxy_delta_soh
        self.last_world_model_voltage_raw = float(pred[1])
        next_state = np.array(
            [
                np.clip(pred[0], 0.0, 1.0),
                np.clip(state[1] - delta_soh, 0.0, 1.2),
                np.clip(pred[1], self.cfg.V_min, self.cfg.V_max),
                safe_current,
                pred[2],
            ],
            dtype=np.float32,
        )
        self.history = np.roll(model_input, shift=-1, axis=0)
        self.history[-1] = np.array(
            [next_state[0], next_state[1], next_state[2], next_state[3], next_state[4], safe_current],
            dtype=np.float32,
        )
        return next_state

    def _reset_history(self, state: np.ndarray) -> None:
        """Fill Mamba history with a repeated initial no-current state."""
        row = np.array([state[0], state[1], state[2], state[3], state[4], 0.0], dtype=np.float32)
        self.history = np.repeat(row[None, :], self.cfg.seq_len, axis=0)

    def _aging_proxy_delta_soh(self, current_A: float, *, raw_voltage: float, temperature: float) -> float:
        """Estimate one-step SOH loss from current, high-voltage, and heat stress."""
        current_stress = (float(current_A) / max(self.cfg.I_max_amps, 1e-12)) ** 2
        voltage_stress = max((float(raw_voltage) - 4.0) / max(self.cfg.V_max - 4.0, 1e-6), 0.0) ** 2
        temperature_stress = max((float(temperature) - self.cfg.T_ref) / 15.0, 0.0) ** 2
        stress = 0.40 * current_stress + 1.40 * voltage_stress + 0.20 * temperature_stress
        return float(max(self.cfg.calendar_aging_scale + self.cfg.aging_proxy_scale * stress, 0.0))
