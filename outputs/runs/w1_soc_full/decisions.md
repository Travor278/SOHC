# W1 SOC Full Fine-Tune

Date: 2026-05-07

## Fixes Before Run

- NASA capacity-bearing discharge cycles use negative current. The SOC label builder now treats negative mean current as discharge and uses `SOC = 1 - Ah / span`.
- SOC labels are generated only for cycles with finite positive `Capacity`.
- Sliding windows are generated inside each labeled cycle, so windows no longer cross charge/discharge/rest cycle boundaries.
- The training script now writes best checkpoints every epoch and supports early stopping, because a 20-epoch CPU-only run exceeded 20 minutes before final save.

## Command

`python -m craic_pipeline.soc_finetune --weights external/KeiLongW/trained_model/2021-01-13-18-48-52_lstm_soc_percentage_lg_positive_temp_100_steps_mixed_cycle_test_best.h5 --arc-dir data/nasa_pcoe/ARC-FY08Q4 --pcoe-dir data/nasa_pcoe/B000x --out outputs/soc_finetuned.h5 --epochs 5 --batch-size 128 --stride 20 --max-samples 20000 --learning-rate 0.001 --early-stop-patience 2 --eval-stride 50 --eval-max-samples 5000`

## Result

- ARC train windows: 16935
- ARC holdout windows: 2575
- Best ARC holdout MAE: 8.63% SOC
- PCoE sampled holdout windows: 2781
- PCoE sampled holdout MAE: 5.51% SOC

## Decision

`outputs/soc_finetuned.h5` is a valid W1 plumbing artifact, but it does not satisfy the W1 acceptance target of NASA holdout MAE `< 1.5%`. Keep the TODO acceptance item open.

The next SOC iteration should improve cross-cell/generalization behavior. Candidate paths:

- add a NASA capacity-cycle-specific calibration head;
- train on a broader NASA split if the project accepts using B000x train/holdout rather than ARC-only fine-tune;
- replace approximate Coulomb labels with a more faithful full-cycle SOC reconstruction.
