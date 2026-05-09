# Technical Review Fix Decisions

Date: 2026-05-09

## Decisions

1. Reclassify the Python W5 multi-cell simulator as an `independent-cell supervisory prototype`.
   - Reason: `pack_balance.py` assigns independent per-cell currents, so it does not enforce series-string KCL/KVL.
   - Consequence: report and status docs no longer claim physical 6S1P/30S1P pack simulation from this prototype.

2. Keep the current hybrid aging design, but report it as proxy-driven.
   - Existing W4/W5 trajectories show `model_delta_soh_sum = 0` and `aging_proxy_delta_soh` dominance ratio = 100%.
   - Consequence: speed and delta-SOH improvements are framed as physical stress-proxy optimization, not independent Mamba aging prediction.

3. Add conservative SOH-aware ECM projection.
   - Formula: `R0_eff = R0 * (2 - clip(SOH, 0.5, 1.0))`.
   - Consequence: lower-SOH cells receive more conservative voltage-bound current projection.

4. Align reward defaults with the final trained policy.
   - Defaults: `speed=30`, `voltage=300`, `temperature=0.02`, `aging=120`.
   - Temperature risk now uses a soft-hard penalty: no penalty below 40 deg C, quadratic risk until `T_max`, cliff beyond `T_max`.

5. Reduce Mamba reset-history distribution shift.
   - `_reset_history()` now initializes around no-current state with small physical noise instead of 64 identical rows.
   - WSL/Mamba/CUDA replay has now quantified the long-horizon drift: B0018 100-step V MAE/P95 = 22.36/85.83 mV, 600-step V MAE/P95 = 170.56/550.20 mV.
   - Consequence: report W2 as a short-horizon rolling dynamics model, not a 600-step open-loop plant substitute.

## Verification

- Targeted tests: `21 passed` with `.venv_craic`:
  - `tests/test_ecm_safety_layer.py`
  - `tests/test_rl_env.py`
  - `tests/test_eval_compare.py`
  - `tests/test_pack_balance.py`
- Note: Windows `.venv_craic` printed a pyarrow access-violation stack after pytest completed; pytest exit code was 0.
