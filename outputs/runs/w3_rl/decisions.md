# W3 RL Decisions

- 2026-05-08: Use `BatteryChargingEnv` state `[SOC, SOH, V, I, T]` and action in `[-1, 1]`, mapped to NASA/W2 charging current `[0, I_max]` A.
- 2026-05-08: Keep ECM safety layer in the legacy MATLAB polarity by passing `-requested_current` into `ECMSafetyLayer.project()`, then converting the projected ECM current back to NASA/W2 polarity for the world model.
- 2026-05-08: Treat ECM as the hard L3 guard: action is projected before the world model step, and the environment voltage observation is clipped to `[V_min, V_max]`. The raw world-model voltage is still exposed in `info["world_model_voltage_raw"]`.
- 2026-05-08: Initial reward weights are `speed=12`, `voltage=50`, `temperature=0.2`, `aging=80`. This keeps the explicit aging penalty in the reward, but the TensorBoard curve is not yet monotonic, so W3 reward tuning remains open.
- 2026-05-08: Final saved W3 policy is `outputs/sac_policy.zip` from WSL GPU + Mamba, `total_steps=100000`, `max_steps=200`, `batch_size=128`, `buffer_size=100000`. A 600-step retrain hit WSL OOM at about 45.5k steps and did not replace the saved policy.
- 2026-05-08: Added `--reward-speed`, `--reward-voltage`, `--reward-temperature`, `--reward-aging`, `--policy-device`, and `--checkpoint-freq` to `train_sac.py` so future reward sweeps and long SAC runs do not require code edits and can keep intermediate policies.
