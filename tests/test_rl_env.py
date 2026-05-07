from __future__ import annotations

import numpy as np
import torch

from craic_pipeline.ecm_safety_layer import ECMParams, ECMSafetyLayer
from craic_pipeline.rl_env import BatteryChargingEnv, EnvConfig, RewardWeights


class DummyWorldModel(torch.nn.Module):
    """Tiny deterministic dynamics model for W3 environment tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return plausible `[SOC_next, V_next, T_next, delta_SOH]` outputs."""
        last = x[:, -1]
        current = last[:, 5]
        charge = current.clamp(min=0.0, max=5.0)
        soc_next = (last[:, 0] + 0.003 * charge).clamp(0.0, 1.0)
        voltage_next = (last[:, 2] + 0.0004 * charge).clamp(2.5, 4.19)
        temp_next = last[:, 4] + 0.01 * charge
        delta_soh = 1e-5 * charge
        return torch.stack([soc_next, voltage_next, temp_next, delta_soh], dim=-1)


def _make_env(max_steps: int = 80) -> BatteryChargingEnv:
    """Build a small deterministic W3 env for unit tests."""
    params = ECMParams(
        R0=0.05,
        R1=0.01,
        R2=0.02,
        C1=1000.0,
        C2=1000.0,
        ocv_coeffs=(3.7,),
        V_min=2.5,
        V_max=4.2,
    )
    cfg = EnvConfig(
        max_steps=max_steps,
        seq_len=8,
        device="cpu",
        reward=RewardWeights(speed=12.0, voltage=50.0, temperature=0.2, aging=80.0),
    )
    return BatteryChargingEnv(DummyWorldModel(), ECMSafetyLayer(params), cfg)


def test_battery_charging_env_reset_and_step_contract():
    """W3 env exposes Gymnasium reset/step shapes and finite diagnostics."""
    env = _make_env()

    obs, info = env.reset(seed=2026)
    next_obs, reward, terminated, truncated, step_info = env.step(np.array([1.0], dtype=np.float32))

    assert obs.shape == (5,)
    assert info == {}
    assert next_obs.shape == (5,)
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info["safe_current"] >= 0.0
    assert step_info["safe_current"] <= env.cfg.I_max_amps + 1e-8
    assert step_info["safe_ecm_current"] <= 0.0
    assert env.cfg.V_min <= step_info["voltage"] <= env.cfg.V_max
    assert {"speed", "voltage_penalty", "temperature_penalty", "aging_penalty"} <= set(
        step_info["reward_terms"]
    )


def test_compute_reward_penalizes_aging_loss():
    """A larger SOH drop lowers reward when all other terms are equal."""
    env = _make_env()
    state = np.array([0.2, 1.0, 3.7, 0.0, 25.0], dtype=np.float32)
    no_aging = np.array([0.21, 1.0, 3.7, 1.0, 25.0], dtype=np.float32)
    aging = np.array([0.21, 0.99, 3.7, 1.0, 25.0], dtype=np.float32)

    reward_no_aging, _ = env.compute_reward(state, no_aging)
    reward_aging, terms = env.compute_reward(state, aging)

    assert reward_aging < reward_no_aging
    assert terms["aging_penalty"] > 0.0


def test_battery_charging_env_random_1000_steps_has_no_nan():
    """Random-action smoke test keeps the W3 env numerically stable."""
    env = _make_env(max_steps=120)
    obs, _ = env.reset(seed=7)
    rewards = []

    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        rewards.append(reward)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert env.cfg.V_min <= info["voltage"] <= env.cfg.V_max
        if terminated or truncated:
            obs, _ = env.reset()

    assert len(rewards) == 1000
