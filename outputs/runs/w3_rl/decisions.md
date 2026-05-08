# W3 RL Decisions

- 2026-05-08: Use `BatteryChargingEnv` state `[SOC, SOH, V, I, T]` and action in `[-1, 1]`, mapped to NASA/W2 charging current `[0, I_max]` A.
- 2026-05-08: Keep ECM safety layer in the legacy MATLAB polarity by passing `-requested_current` into `ECMSafetyLayer.project()`, then converting the projected ECM current back to NASA/W2 polarity for the world model.
- 2026-05-08: Treat ECM as the hard L3 guard: action is projected before the world model step, and the environment voltage observation is clipped to `[V_min, V_max]`. The raw world-model voltage is still exposed in `info["world_model_voltage_raw"]`.
- 2026-05-08: Reward voltage penalty uses raw world-model voltage before L3 clipping; L3-clipped voltage remains the physical observation and hard safety output.
- 2026-05-08: Add a small calendar-aging floor (`calendar_aging_scale=2.5e-6`) on top of current/high-voltage/temperature stress so shorter safe charging can reduce total one-cycle SOH loss.
- 2026-05-08: Initial W3 policy was `total_steps=100000`, `max_steps=200`, but it did not learn the later high-voltage taper region well. A first `max_steps=600` retrain with large buffer hit WSL OOM at about 45.5k steps.
- 2026-05-08: Final selected W3 policy is `outputs/sac_policy.zip` copied from `outputs/runs/w3_horizon600/sac_policy_h600.zip`: WSL GPU + Mamba, `total_steps=60000`, `max_steps=600`, `batch_size=64`, `buffer_size=20000`, `speed=30`, `voltage=300`, `temperature=0.02`, `aging=120`. TensorBoard `ep_rew_mean` rose from about 9.6 to 13.6.
- 2026-05-08: Added `--reward-speed`, `--reward-voltage`, `--reward-temperature`, `--reward-aging`, `--policy-device`, and `--checkpoint-freq` to `train_sac.py` so future reward sweeps and long SAC runs do not require code edits and can keep intermediate policies.
- 2026-05-08: W4 comparison reports both all-episode metrics and a paired-vs-CCCV table. The paired table is used for the core “charge to 80%” claim because CC-CV does not reach 80% within the fixed horizon for every randomized initial SOC.
