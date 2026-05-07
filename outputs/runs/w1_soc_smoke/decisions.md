# W1 SOC Smoke Run

Date: 2026-05-07

## Command

Used the 100-step KeiLongW LG mixed-cycle SOC weight:

`external/KeiLongW/trained_model/2021-01-13-18-48-52_lstm_soc_percentage_lg_positive_temp_100_steps_mixed_cycle_test_best.h5`

Smoke settings:

- ARC files: first 4 files only (`--limit-files 4`)
- Epochs: 1
- Stride: 200
- Max train samples: 2000
- PCoE eval stride: 500
- PCoE eval max samples: 2000

## Result

- Train samples: 1563
- ARC holdout samples: 400
- ARC holdout MAE: 28.20% SOC
- PCoE sampled holdout MAE: 21.18% SOC

## Decision

This run only validates that the W1 SOC training/inference plumbing works after NaN-label filtering and strided windowing. It is not an acceptance model and must not be treated as the final `outputs/soc_finetuned.h5`.

Next SOC work should improve label construction and run a real ARC fine-tune before checking the TODO acceptance target of NASA holdout MAE `< 1.5%`.
