# W2 WSL Mamba Environment

Date: 2026-05-07

## Distribution

- Checked both WSL distributions.
- `Ubuntu2204` has Python 3.10.12 and GPU passthrough, but no ready conda/nvcc environment.
- `Ubuntu2404` is now the active W2 GPU environment because PyTorch cu128 and Mamba were verified there.

## Installed Environment

- Venv: `~/.venvs/sohc-craic-py312`
- Python: 3.12
- PyTorch: 2.11.0+cu128
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU, compute capability `sm_120`
- CUDA toolkit/nvcc: 12.8.93

## Mamba Build Decision

`mamba-ssm` and `causal-conv1d` hardcode CUDA `-gencode` flags for many architectures. That made builds appear stuck in `nvcc`/`ptxas` even after setting `TORCH_CUDA_ARCH_LIST=12.0`.

To finish the local install, the source distributions were downloaded to `~/sohc-build` and their `setup.py` architecture block was patched to build only:

`arch=compute_120,code=sm_120`

Installed packages:

- `causal-conv1d==1.6.1`
- `mamba-ssm==2.3.1`

Verification:

- CUDA tensor allocation on RTX 5070: passed
- `Mamba(d_model=16)` forward on CUDA tensor: passed
- W2 smoke train: `backend=mamba`, `device=cuda`, 512 NASA windows, 1 epoch

## Note

The original `GET 61` pause was the full `cuda-toolkit-12-8` apt install downloading large CUDA packages. After interruption, `sudo dpkg --configure -a` repaired the partial apt state and left `nvcc` usable.
