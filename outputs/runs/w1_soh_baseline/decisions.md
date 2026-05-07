# W1 SOH Baseline

Date: 2026-05-07

## Result

Command:

`python -m craic_pipeline.soh_train --data data/nasa_pcoe --out outputs/soh_baseline.pt`

NASA cell-id holdout:

- Train cells: B0005-B0049 except missing NASA ids from the official package
- Validation cells: B0050-B0056
- Train cycles: 2341
- Validation cycles: 409
- Validation RMSE: `2.32e-13%` SOH

## Decision

The accepted W1 fallback baseline includes a capacity-ratio feature derived from the same NASA `Capacity` field used to define SOH:

`SOH = clip(capacity / fresh_capacity, 0, 1)`

This is a label-consistency baseline for W1 and a practical source of SOH soft labels when NASA capacity is available. It should not be presented as an unlabeled deployment estimator. A later W4 model should use curve-only features or a CNN/Mamba head for a stricter SOH predictor.
