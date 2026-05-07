# W2 World Model Decisions

## 2026-05-07

- Use Windows `.venv_craic` only for NASA `.mat` parsing and TensorFlow SOC utilities. These steps are CPU/I/O bound and do not benefit from the RTX 5070 GPU in the current Windows environment.
- Use `Ubuntu2404` WSL for PyTorch/Mamba training. The WSL environment has PyTorch `2.11.0+cu128`, CUDA visible on the RTX 5070 Laptop GPU, and `mamba-ssm==2.3.1` installed with an `sm_120`-only build.
- Build the first W2 tensor artifact as PCoE-only to keep the main W2/W3 chain moving: `outputs/world_model_train_data.pt` contains 20k windows from B0005/B0006/B0007/B0018, strict Coulomb SOC fallback labels, and SOH capacity-ratio labels.
- Do not claim Randomized extrapolation yet. Full Randomized parsing on Windows ran for many minutes without producing `outputs/world_model_train_data.pt`; it needs file-level caching or a WSL/Linux long run.
- Use a residual world-model head by default. Direct absolute next-state prediction reached about 28 mV on B0018 holdout after 50 GPU epochs; residual prediction reached 1.42 mV on B0005/B0006/B0007 -> B0018 holdout after 50 GPU epochs.
