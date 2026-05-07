# W0/W1 Decisions

Date: 2026-05-07

## Environment

- Used conda to create `.venv_craic` with Python 3.10.20 because the host Python 3.13.9 is not compatible with the TensorFlow/BatteryML stack pinned by the project.
- Installed `torch==2.4.1+cu124` to keep the current `requirements.txt` constraint (`torch<2.5`) intact.
- `torch.cuda.is_available()` is true on the RTX 5070 Laptop GPU, but the installed PyTorch wheel supports CUDA architectures only up to `sm_90`; this GPU reports `sm_120`, and CUDA tensor allocation fails with `no kernel image is available for execution on the device`.
- W1 will use CPU/TensorFlow paths where needed. Native PyTorch GPU on this machine likely requires a newer cu128 PyTorch build or a Linux/WSL CUDA toolchain plus dependency pin updates.

## Mamba

- `mamba-ssm` and `causal-conv1d` failed to build on Windows/CUDA during W0. The failure is treated as non-blocking.
- Keep the W2 `--gru-fallback` path as the default unless the environment is moved to a Linux CUDA setup.

## Data

- NASA PCoE data was downloaded from the official S3-backed NASA PCoE repository.
- B0005/B0006/B0007/B0018 are inside the official `1. BatteryAgingARC-FY08Q4.zip`; B0025-B0056 are distributed across later ARC zip files. Files were placed into the project paths expected by TODO.md.
- The Randomized Battery Usage package expands into seven subdirectories with 28 RW `.mat` files, not literal RW1-RW7. The loader parses directories recursively.
- Zenodo 6985321 was downloaded for later W4/W5 reference experiments. Zenodo 18471156 remains optional for W5 qualitative display only.

## W1 Estimators

- `craic_pipeline.nasa_loader` normalizes the three NASA schemas into `(V, I, T, t, cycle_id, ambient_temp, capacity)` arrays.
- Randomized current jumps are filtered by splitting each original cycle at `abs(diff(I)) > 1A` and assigning new segment ids, instead of merely dropping isolated samples.
- SOC fine-tuning code uses KeiLongW `.h5` weights as warm start and freezes the first two LSTM layers, but full ARC fine-tuning has not yet met the W1 acceptance target.
- SOH fallback training saved `outputs/soh_baseline.pt`, but NASA cell-id holdout RMSE is 36.36%. The next SOH iteration should use a stronger BatteryML/CNN/XGBoost model and more careful capacity/holdout handling.
