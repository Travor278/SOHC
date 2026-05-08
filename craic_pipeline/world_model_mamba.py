"""Mamba/GRU world model for NASA battery next-step dynamics.

The W2 model consumes sequences shaped `(B, L, 6)`:
`[SOC, SOH, V, I, T, action]`, and predicts the next
`[SOC, V, T, delta_SOH]` target. On Windows where `mamba-ssm` is unavailable,
the same API falls back to a two-layer GRU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from craic_pipeline.nasa_loader import load_pcoe_basic, load_randomized_usage


@dataclass
class WorldModelConfig:
    """Configuration for the W2 battery world model."""

    state_dim: int = 5
    action_dim: int = 1
    hidden_dim: int = 128
    n_layers: int = 4
    seq_len: int = 64
    use_mamba: bool = True
    residual_head: bool = True


class BatteryWorldModel(nn.Module):
    """Sequence model mapping battery state-action histories to next dynamics."""

    def __init__(self, cfg: WorldModelConfig):
        """Initialize a Mamba backend when available, otherwise a GRU fallback."""
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.state_dim + cfg.action_dim
        self.input_proj = nn.Linear(self.input_dim, cfg.hidden_dim)
        self.backend = "gru"
        self.layers: nn.Module
        if cfg.use_mamba:
            try:
                from mamba_ssm import Mamba

                self.layers = nn.ModuleList([Mamba(d_model=cfg.hidden_dim) for _ in range(cfg.n_layers)])
                self.norms = nn.ModuleList([nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.n_layers)])
                self.backend = "mamba"
            except Exception:
                self.layers = nn.GRU(
                    input_size=cfg.hidden_dim,
                    hidden_size=cfg.hidden_dim,
                    num_layers=max(1, min(cfg.n_layers, 2)),
                    batch_first=True,
                )
        else:
            self.layers = nn.GRU(
                input_size=cfg.hidden_dim,
                hidden_size=cfg.hidden_dim,
                num_layers=max(1, min(cfg.n_layers, 2)),
                batch_first=True,
            )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 4),
        )
        self._init_residual_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict `[SOC_next, V_next, T_next, delta_SOH]` from `(B,L,6)`."""
        h = self.input_proj(x.float())
        if self.backend == "mamba":
            for norm, layer in zip(self.norms, self.layers):
                h = h + layer(norm(h))
        else:
            h, _ = self.layers(h)
        raw = self.head(h[:, -1])
        if not self.cfg.residual_head:
            return raw
        last = x[:, -1].float()
        return torch.stack(
            [
                last[:, 0] + raw[:, 0],
                last[:, 2] + raw[:, 1],
                last[:, 4] + raw[:, 2],
                raw[:, 3],
            ],
            dim=-1,
        )

    @torch.no_grad()
    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Advance one state with one action for later RL environment usage."""
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=-1).unsqueeze(1)
        pred = self.forward(x)
        next_state = state.clone()
        next_state[:, 0] = pred[:, 0].clamp(0.0, 1.0)
        next_state[:, 1] = (state[:, 1] - pred[:, 3].clamp_min(0.0)).clamp(0.0, 1.2)
        next_state[:, 2] = pred[:, 1]
        next_state[:, 3] = action.reshape(-1)
        next_state[:, 4] = pred[:, 2]
        return next_state.squeeze(0)

    def _init_residual_head(self) -> None:
        """Initialize residual predictions close to a persistence baseline."""
        final = self.head[-1]
        if isinstance(final, nn.Linear) and self.cfg.residual_head:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)


def build_training_dataset(
    pcoe_dir: Path,
    randomized_dir: Path,
    soc_csv: Path | None = None,
    soh_ckpt: Path | None = None,
    *,
    seq_len: int = 64,
    stride: int = 8,
    max_windows: int = 100_000,
) -> TensorDataset:
    """Build NASA PCoE + Randomized world-model tensors.

    Args:
        pcoe_dir: Directory with B0005-B0018 `.mat` files.
        randomized_dir: Directory with NASA Randomized `.mat` files.
        soc_csv: Optional SOC prediction CSV; unused when unavailable.
        soh_ckpt: Optional SOH checkpoint path; capacity ratios are used here.
        seq_len: Input history length.
        stride: Window step used to subsample long traces.
        max_windows: Maximum number of windows kept in memory.

    Returns:
        `TensorDataset(X, y)` with `X=(N,L,6)` and `y=(N,4)`.
    """
    bundle = build_world_tensors(
        pcoe_dir,
        randomized_dir,
        soc_weights=soc_csv,
        soh_ckpt=soh_ckpt,
        seq_len=seq_len,
        stride=stride,
        max_windows=max_windows,
    )
    return tensors_to_dataset(bundle)


def build_world_tensors(
    pcoe_dir: Path,
    randomized_dir: Path,
    *,
    soc_weights: Path | None = None,
    soh_ckpt: Path | None = None,
    seq_len: int = 64,
    stride: int = 8,
    max_windows: int = 100_000,
    use_soc_model: bool = True,
    limit_pcoe_files: int | None = None,
    limit_randomized_files: int | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Build W2 tensor package with SOC/SOH soft labels and source metadata."""
    del soh_ckpt
    soc_model = None
    soc_window = None
    soc_source = "strict_coulomb_fallback"
    if use_soc_model and soc_weights and Path(soc_weights).exists():
        try:
            from craic_pipeline.soc_inference import infer_model_window, load_keilongw_model

            soc_model = load_keilongw_model(Path(soc_weights))
            soc_window = infer_model_window(soc_model) or 100
            soc_source = f"keras:{Path(soc_weights).as_posix()}"
        except Exception as exc:
            soc_source = f"strict_coulomb_fallback_after_soc_error:{exc!r}"

    pieces: list[tuple[np.ndarray, np.ndarray, list[str], list[str]]] = []
    traces: list[dict] = []
    for subset, loader, root in (
        ("pcoe", load_pcoe_basic, Path(pcoe_dir)),
        ("randomized", load_randomized_usage, Path(randomized_dir)),
    ):
        if not root.exists():
            continue
        files = _mat_files(root)
        if subset == "randomized":
            files = sorted(files, key=lambda item: item.stat().st_size)
        limit = limit_pcoe_files if subset == "pcoe" else limit_randomized_files
        if limit is not None:
            files = files[:limit]
        for file_path in files:
            shard = _load_or_build_file_shard(
                file_path,
                subset=subset,
                loader=loader,
                soc_model=soc_model,
                soc_window=soc_window,
                seq_len=seq_len,
                stride=stride,
                cache_dir=cache_dir,
            )
            if shard is None:
                continue
            traces.extend(shard["traces"])
            pieces.append((shard["X"], shard["y"], shard["subset"].tolist(), shard["cell"].tolist()))
    if not pieces:
        raise FileNotFoundError("no NASA windows found for world-model training")

    X = np.concatenate([item[0] for item in pieces], axis=0)
    y = np.concatenate([item[1] for item in pieces], axis=0)
    subsets = np.asarray([value for item in pieces for value in item[2]], dtype=object)
    cells = np.asarray([value for item in pieces for value in item[3]], dtype=object)
    X, y, subsets, cells = _cap_samples_with_meta(X, y, subsets, cells, max_windows)
    return {
        "X": X.astype(np.float32),
        "y": y.astype(np.float32),
        "subset": subsets,
        "cell": cells,
        "traces": traces,
        "schema": ["SOC", "SOH", "V", "I", "T", "action_current"],
        "target_schema": ["SOC_next", "V_next", "T_next", "delta_SOH"],
        "meta": {
            "seq_len": seq_len,
            "stride": stride,
            "max_windows": max_windows,
            "soc_source": soc_source,
            "soh_source": "capacity_ratio_per_file",
            "samples": int(len(y)),
            "subsets": {name: int(np.sum(subsets == name)) for name in sorted(set(subsets.tolist()))},
        },
    }


def tensors_to_dataset(bundle: dict) -> TensorDataset:
    """Convert a saved W2 tensor bundle to `TensorDataset(X, y)`."""
    return TensorDataset(torch.from_numpy(np.asarray(bundle["X"])).float(), torch.from_numpy(np.asarray(bundle["y"])).float())


def save_world_tensors(bundle: dict, out: Path) -> None:
    """Persist W2 train tensors to a `.pt` package."""
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)


def load_world_tensors(path: Path) -> dict:
    """Load W2 train tensors from a `.pt` package."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_or_build_file_shard(
    file_path: Path,
    *,
    subset: str,
    loader,
    soc_model,
    soc_window: int | None,
    seq_len: int,
    stride: int,
    cache_dir: Path | None,
) -> dict | None:
    """Load or build one cached NASA file shard for W2 tensors."""
    shard_path = _shard_path(cache_dir, subset, file_path) if cache_dir else None
    if shard_path is not None and shard_path.exists():
        return torch.load(shard_path, map_location="cpu", weights_only=False)

    V, I, T, t, cycle_id, ambient, capacity = loader(file_path)
    if not V.size:
        return None
    df = _frame_from_arrays(V, I, T, t, cycle_id, ambient, capacity)
    soc = _soc_soft_labels(df, model=soc_model, window=soc_window, fast_when_unlabeled=subset == "randomized")
    soh = _soh_soft_labels(capacity)
    windows = _series_to_windows(
        V,
        I,
        T,
        cycle_id,
        capacity,
        soc=soc,
        soh=soh,
        seq_len=seq_len,
        stride=stride,
    )
    if not windows:
        return None
    X = np.concatenate([item[0] for item in windows], axis=0)
    y = np.concatenate([item[1] for item in windows], axis=0)
    shard = {
        "X": X.astype(np.float32),
        "y": y.astype(np.float32),
        "subset": np.asarray([subset] * len(y), dtype=object),
        "cell": np.asarray([file_path.stem] * len(y), dtype=object),
        "traces": _series_to_traces(V, I, T, cycle_id, soc=soc, soh=soh, subset=subset, cell=file_path.stem),
    }
    if shard_path is not None:
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(shard, shard_path)
    return shard


def _shard_path(cache_dir: Path | None, subset: str, file_path: Path) -> Path:
    """Return a stable shard cache path for one NASA `.mat` file."""
    safe_parent = hashlib.sha1(str(file_path.parent).encode("utf-8")).hexdigest()[:8]
    return Path(cache_dir) / subset / f"{file_path.stem}_{safe_parent}.pt"


def train_world_model(
    dataset: TensorDataset,
    cfg: WorldModelConfig,
    *,
    val_dataset: TensorDataset | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple[BatteryWorldModel, dict]:
    """Train the W2 world model with MSE next-step loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BatteryWorldModel(cfg).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    val_history: list[dict] = []
    for _ in range(epochs):
        losses = []
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
        if val_dataset is not None:
            val_history.append(evaluate_world_model(model, val_dataset, batch_size=batch_size, device=device))
    model = model.cpu()
    metrics = {
        "loss": history,
        "val": val_history,
        "backend": model.backend,
        "samples": len(dataset),
        "val_samples": 0 if val_dataset is None else len(val_dataset),
        "device": str(device),
    }
    return model, metrics


@torch.no_grad()
def evaluate_world_model(
    model: BatteryWorldModel,
    dataset: TensorDataset,
    *,
    batch_size: int = 1024,
    device: torch.device | str | None = None,
) -> dict:
    """Evaluate one-step SOC/V/T/SOH metrics on a world-model dataset."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    was_training = model.training
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    targets = []
    for X, y in loader:
        preds.append(model(X.to(device)).detach().cpu())
        targets.append(y.detach().cpu())
    if was_training:
        model.train()
    pred = torch.cat(preds).numpy()
    target = torch.cat(targets).numpy()
    err = pred - target
    return {
        "soc_mae_percent": float(np.mean(np.abs(err[:, 0])) * 100.0),
        "voltage_mae_mV": float(np.mean(np.abs(err[:, 1])) * 1000.0),
        "temperature_mae_C": float(np.mean(np.abs(err[:, 2]))),
        "delta_soh_mae_percent": float(np.mean(np.abs(err[:, 3])) * 100.0),
    }


@torch.no_grad()
def evaluate_rollout_drift(
    model: BatteryWorldModel,
    bundle: dict,
    *,
    horizon: int = 20,
    seq_len: int | None = None,
    stride: int = 64,
    max_rollouts: int | None = None,
    cells: list[str] | None = None,
    subsets: list[str] | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Evaluate open-loop multi-step V drift with true future action current."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    seq_len = int(seq_len or bundle.get("meta", {}).get("seq_len", 64))
    cell_filter = {cell.upper() for cell in cells} if cells else None
    subset_filter = {subset.lower() for subset in subsets} if subsets else None
    errors_mV: list[float] = []
    counts = 0
    for trace in bundle.get("traces", []):
        cell = str(trace.get("cell", "")).upper()
        subset = str(trace.get("subset", "")).lower()
        if cell_filter is not None and cell not in cell_filter:
            continue
        if subset_filter is not None and subset not in subset_filter:
            continue
        features = np.asarray(trace["features"], dtype=np.float32)
        if len(features) <= seq_len + horizon:
            continue
        for start in range(0, len(features) - seq_len - horizon, stride):
            hist = torch.from_numpy(features[start : start + seq_len].copy()).to(device).unsqueeze(0)
            pred_soh = float(hist[0, -1, 1].detach().cpu())
            for step_idx in range(horizon):
                pred = model(hist).squeeze(0).detach()
                future_row = features[start + seq_len + step_idx]
                errors_mV.append(abs(float(pred[1].detach().cpu()) - float(future_row[2])) * 1000.0)
                next_current = float(future_row[3])
                next_action = float(future_row[5])
                delta_soh = max(float(pred[3].detach().cpu()), 0.0)
                pred_soh = float(np.clip(pred_soh - delta_soh, 0.0, 1.2))
                next_feature = torch.tensor(
                    [[float(pred[0].clamp(0.0, 1.0)), pred_soh, float(pred[1]), next_current, float(pred[2]), next_action]],
                    dtype=hist.dtype,
                    device=device,
                )
            hist = torch.cat([hist[:, 1:], next_feature.unsqueeze(0)], dim=1)
            counts += 1
            if max_rollouts is not None and counts >= max_rollouts:
                break
        if max_rollouts is not None and counts >= max_rollouts:
            break
    if not errors_mV:
        raise ValueError("no traces available for rollout drift evaluation")
    return {
        "horizon": int(horizon),
        "rollouts": int(counts),
        "max_rollouts": None if max_rollouts is None else int(max_rollouts),
        "voltage_mae_mV": float(np.mean(errors_mV)),
        "voltage_p95_mV": float(np.percentile(errors_mV, 95)),
        "cells": sorted(cell_filter) if cell_filter else None,
        "subsets": sorted(subset_filter) if subset_filter else None,
    }


@torch.no_grad()
def evaluate_randomized_directory(
    model: BatteryWorldModel,
    randomized_dir: Path,
    *,
    cache_dir: Path,
    seq_len: int = 64,
    stride: int = 64,
    batch_size: int = 1024,
    rollout_horizon: int = 20,
    rollout_stride: int = 256,
    max_files: int | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Evaluate W2 on every NASA Randomized `.mat` file using cached shards.

    Args:
        model: Trained W2 world model.
        randomized_dir: NASA Randomized root with RW `.mat` files.
        cache_dir: Per-file shard cache directory.
        seq_len: History length used for shard windows.
        stride: Window stride used when building missing shards.
        batch_size: One-step evaluation batch size.
        rollout_horizon: Open-loop rollout horizon.
        rollout_stride: Trace stride used for rollout sampling.
        max_files: Optional debug cap; `None` means all `.mat` files.
        device: Torch device.

    Returns:
        JSON-serializable metrics covering all evaluated Randomized files.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    files = sorted(_mat_files(Path(randomized_dir)), key=lambda item: item.stat().st_size)
    if max_files is not None:
        files = files[:max_files]
    one_step = _empty_error_accumulator()
    rollout_errors_mV: list[float] = []
    file_rows = []
    for file_path in files:
        shard = _load_or_build_file_shard(
            file_path,
            subset="randomized",
            loader=load_randomized_usage,
            soc_model=None,
            soc_window=None,
            seq_len=seq_len,
            stride=stride,
            cache_dir=cache_dir,
        )
        if shard is None:
            file_rows.append({"file": file_path.stem, "status": "empty"})
            continue
        file_one_step = _one_step_error_accumulator(
            model,
            shard["X"],
            shard["y"],
            batch_size=batch_size,
            device=device,
        )
        _merge_error_accumulator(one_step, file_one_step)
        file_rollout = _rollout_errors_from_traces(
            model,
            shard.get("traces", []),
            seq_len=seq_len,
            horizon=rollout_horizon,
            stride=rollout_stride,
            device=device,
        )
        rollout_errors_mV.extend(file_rollout)
        file_rows.append(
            {
                "file": file_path.stem,
                "status": "ok",
                "windows": int(len(shard["y"])),
                "traces": int(len(shard.get("traces", []))),
                "one_step_voltage_mae_mV": _safe_div(file_one_step["voltage_abs_sum"], file_one_step["count"])
                * 1000.0,
                "rollout_errors": int(len(file_rollout)),
                "rollout_voltage_mae_mV": float(np.mean(file_rollout)) if file_rollout else np.nan,
            }
        )
    one_count = int(one_step["count"])
    rollout_count = int(len(rollout_errors_mV))
    return {
        "subset": "randomized_full",
        "files_total": int(len(files)),
        "files_evaluated": int(sum(row["status"] == "ok" for row in file_rows)),
        "seq_len": int(seq_len),
        "stride": int(stride),
        "batch_size": int(batch_size),
        "one_step_samples": one_count,
        "one_step": {
            "soc_mae_percent": _safe_div(one_step["soc_abs_sum"], one_count) * 100.0,
            "voltage_mae_mV": _safe_div(one_step["voltage_abs_sum"], one_count) * 1000.0,
            "temperature_mae_C": _safe_div(one_step["temperature_abs_sum"], one_count),
            "delta_soh_mae_percent": _safe_div(one_step["delta_soh_abs_sum"], one_count) * 100.0,
        },
        "rollout": {
            "horizon": int(rollout_horizon),
            "stride": int(rollout_stride),
            "errors": rollout_count,
            "voltage_mae_mV": float(np.mean(rollout_errors_mV)) if rollout_errors_mV else np.nan,
            "voltage_p95_mV": float(np.percentile(rollout_errors_mV, 95)) if rollout_errors_mV else np.nan,
        },
        "files": file_rows,
    }


def _empty_error_accumulator() -> dict:
    """Create an accumulator for one-step absolute errors."""
    return {
        "count": 0,
        "soc_abs_sum": 0.0,
        "voltage_abs_sum": 0.0,
        "temperature_abs_sum": 0.0,
        "delta_soh_abs_sum": 0.0,
    }


@torch.no_grad()
def _one_step_error_accumulator(
    model: BatteryWorldModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Accumulate one-step W2 absolute errors without materializing predictions."""
    stats = _empty_error_accumulator()
    dataset = TensorDataset(torch.from_numpy(np.asarray(X)).float(), torch.from_numpy(np.asarray(y)).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for X_batch, y_batch in loader:
        pred = model(X_batch.to(device)).detach().cpu()
        target = y_batch.detach().cpu()
        err = (pred - target).abs().numpy()
        stats["count"] += int(len(err))
        stats["soc_abs_sum"] += float(err[:, 0].sum())
        stats["voltage_abs_sum"] += float(err[:, 1].sum())
        stats["temperature_abs_sum"] += float(err[:, 2].sum())
        stats["delta_soh_abs_sum"] += float(err[:, 3].sum())
    return stats


def _merge_error_accumulator(total: dict, part: dict) -> None:
    """Add one one-step error accumulator into another."""
    for key in total:
        total[key] += part[key]


@torch.no_grad()
def _rollout_errors_from_traces(
    model: BatteryWorldModel,
    traces: list[dict],
    *,
    seq_len: int,
    horizon: int,
    stride: int,
    device: torch.device,
) -> list[float]:
    """Return sampled rollout voltage errors in mV for a list of traces."""
    errors_mV: list[float] = []
    for trace in traces:
        features = np.asarray(trace["features"], dtype=np.float32)
        if len(features) <= seq_len + horizon:
            continue
        for start in range(0, len(features) - seq_len - horizon, stride):
            hist = torch.from_numpy(features[start : start + seq_len].copy()).to(device).unsqueeze(0)
            pred_soh = float(hist[0, -1, 1].detach().cpu())
            for step_idx in range(horizon):
                pred = model(hist).squeeze(0).detach()
                future_row = features[start + seq_len + step_idx]
                errors_mV.append(abs(float(pred[1].detach().cpu()) - float(future_row[2])) * 1000.0)
                next_action = float(future_row[5])
                delta_soh = max(float(pred[3].detach().cpu()), 0.0)
                pred_soh = float(np.clip(pred_soh - delta_soh, 0.0, 1.2))
                next_feature = torch.tensor(
                    [[float(pred[0].clamp(0.0, 1.0)), pred_soh, float(pred[1]), next_action, float(pred[2]), next_action]],
                    dtype=hist.dtype,
                    device=device,
                )
                hist = torch.cat([hist[:, 1:], next_feature.unsqueeze(0)], dim=1)
    return errors_mV


def _safe_div(value: float, count: int) -> float:
    """Divide by a positive count, returning NaN for empty accumulators."""
    return float(value / count) if count else float("nan")


def load_world_model_checkpoint(path: Path) -> tuple[BatteryWorldModel, dict]:
    """Load a saved W2 world-model checkpoint and metrics."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = WorldModelConfig(**ckpt["cfg"])
    model = BatteryWorldModel(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt.get("metrics", {})


def split_dataset(dataset: TensorDataset, val_ratio: float = 0.2) -> tuple[TensorDataset, TensorDataset]:
    """Deterministically split a tensor dataset into train/validation parts."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    X, y = dataset.tensors
    n = len(dataset)
    n_val = max(1, int(round(n * val_ratio)))
    rng = np.random.default_rng(2026)
    idx = rng.permutation(n)
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return TensorDataset(X[train_idx], y[train_idx]), TensorDataset(X[val_idx], y[val_idx])


def split_bundle_by_cell(bundle: dict, val_cells: list[str]) -> tuple[TensorDataset, TensorDataset]:
    """Split a W2 tensor bundle by NASA cell id for holdout validation."""
    X = torch.from_numpy(np.asarray(bundle["X"])).float()
    y = torch.from_numpy(np.asarray(bundle["y"])).float()
    cells = np.asarray(bundle.get("cell"))
    if cells.size != len(y):
        raise ValueError("bundle does not contain cell metadata aligned with samples")
    val_set = {cell.upper() for cell in val_cells}
    val_mask = np.asarray([str(cell).upper() in val_set for cell in cells], dtype=bool)
    train_mask = ~val_mask
    if not val_mask.any() or not train_mask.any():
        raise ValueError(f"val_cells={val_cells} do not create a non-empty train/val split")
    return TensorDataset(X[train_mask], y[train_mask]), TensorDataset(X[val_mask], y[val_mask])


def _series_to_windows(
    V: np.ndarray,
    I: np.ndarray,
    T: np.ndarray,
    cycle_id: np.ndarray,
    capacity: np.ndarray,
    *,
    soc: np.ndarray | None = None,
    soh: np.ndarray | None = None,
    seq_len: int,
    stride: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert continuous NASA series into per-cycle world-model windows."""
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for cid in np.unique(cycle_id):
        idx = np.where(cycle_id == cid)[0]
        if idx.size <= seq_len:
            continue
        soc_cycle = soc[idx] if soc is not None else _approx_soc(I[idx], capacity[idx])
        soh_cycle = soh[idx] if soh is not None else _approx_soh(capacity[idx])
        features = np.column_stack([soc_cycle, soh_cycle, V[idx], I[idx], T[idx], I[idx]])
        finite = np.isfinite(features).all(axis=1)
        if finite.sum() <= seq_len:
            continue
        features = features[finite]
        X, y = _window_one_cycle(features, seq_len=seq_len, stride=stride)
        if len(y):
            outputs.append((X, y))
    return outputs


def _series_to_traces(
    V: np.ndarray,
    I: np.ndarray,
    T: np.ndarray,
    cycle_id: np.ndarray,
    *,
    soc: np.ndarray,
    soh: np.ndarray,
    subset: str,
    cell: str,
) -> list[dict]:
    """Convert per-cycle NASA series into continuous W2 rollout traces."""
    traces = []
    for cid in np.unique(cycle_id):
        idx = np.where(cycle_id == cid)[0]
        if idx.size < 2:
            continue
        features = np.column_stack([soc[idx], soh[idx], V[idx], I[idx], T[idx], I[idx]])
        finite = np.isfinite(features).all(axis=1)
        if finite.sum() < 2:
            continue
        traces.append(
            {
                "subset": subset,
                "cell": cell,
                "cycle_id": float(cid),
                "features": features[finite].astype(np.float32),
            }
        )
    return traces


def _window_one_cycle(features: np.ndarray, *, seq_len: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Create `(X,y)` next-step windows for one NASA cycle."""
    X_parts = []
    y_parts = []
    for start in range(0, len(features) - seq_len, stride):
        end = start + seq_len
        current = features[end - 1]
        nxt = features[end]
        delta_soh = max(float(current[1] - nxt[1]), 0.0)
        X_parts.append(features[start:end])
        y_parts.append([nxt[0], nxt[2], nxt[4], delta_soh])
    if not X_parts:
        return np.empty((0, seq_len, 6), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
    return np.asarray(X_parts, dtype=np.float32), np.asarray(y_parts, dtype=np.float32)


def _mat_files(path: Path) -> list[Path]:
    """Return sorted NASA `.mat` files from a file or directory path."""
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.mat"))


def _frame_from_arrays(
    V: np.ndarray,
    I: np.ndarray,
    T: np.ndarray,
    t: np.ndarray,
    cycle_id: np.ndarray,
    ambient: np.ndarray,
    capacity: np.ndarray,
) -> pd.DataFrame:
    """Create a DataFrame compatible with the W1 SOC preprocessing helpers."""
    return pd.DataFrame(
        {
            "t": t,
            "voltage": V,
            "current": I,
            "temperature": T,
            "cycle_id": cycle_id,
            "ambient_T": ambient,
            "capacity": capacity,
        }
    )


def _soc_soft_labels(df: pd.DataFrame, *, model=None, window: int | None = None, fast_when_unlabeled: bool = False) -> np.ndarray:
    """Infer SOC with W1 Keras weights, falling back to strict NASA labels."""
    from craic_pipeline.soc_finetune import _estimate_soc_labels

    capacity = df["capacity"].to_numpy(dtype=float) if "capacity" in df else np.array([], dtype=float)
    if fast_when_unlabeled and not np.any(np.isfinite(capacity) & (capacity > 0)):
        return _approx_soc(df["current"].to_numpy(dtype=float), capacity)

    labels = np.full(len(df), np.nan, dtype=float)
    if model is not None:
        from craic_pipeline.soc_inference import predict_soc, preprocess_sequence

        for _, group in df.groupby("cycle_id", sort=False):
            if len(group) < (window or 100):
                continue
            try:
                X, rows = preprocess_sequence(group, window or 100, stride=1)
            except ValueError:
                continue
            labels[rows.index.to_numpy(dtype=int)] = predict_soc(model, X)
    fallback = _estimate_soc_labels(df, mode="strict")
    missing = ~np.isfinite(labels)
    labels[missing] = fallback[missing]
    if np.isfinite(labels).any():
        labels = _fill_nan_1d(labels)
    else:
        labels = _approx_soc(df["current"].to_numpy(dtype=float), capacity)
    return labels


def _soh_soft_labels(capacity: np.ndarray) -> np.ndarray:
    """Create per-sample SOH labels from NASA capacity ratio."""
    return _approx_soh(capacity)


def _fill_nan_1d(values: np.ndarray) -> np.ndarray:
    """Interpolate finite labels and clamp any all-NaN case to one."""
    values = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(values)
    if not finite.any():
        return np.ones_like(values, dtype=float)
    x = np.arange(len(values))
    values[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return np.clip(values, 0.0, 1.0)


def _approx_soc(current: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Estimate SOC labels from current integration for world-model bootstrapping."""
    dt = np.ones_like(current, dtype=float)
    ah = np.cumsum(np.abs(np.nan_to_num(current, nan=0.0)) * dt) / 3600.0
    finite_cap = capacity[np.isfinite(capacity) & (capacity > 0)]
    span = float(finite_cap[-1]) if finite_cap.size else max(float(ah[-1]), 1.0)
    if np.nanmean(current) < 0:
        soc = 1.0 - ah / max(span, 1e-6)
    else:
        soc = ah / max(span, 1e-6)
    return np.clip(soc, 0.0, 1.0)


def _approx_soh(capacity: np.ndarray) -> np.ndarray:
    """Estimate SOH from NASA capacity ratios when available."""
    finite = capacity[np.isfinite(capacity) & (capacity > 0)]
    if not finite.size:
        return np.ones_like(capacity, dtype=float)
    fresh = max(float(np.nanmax(finite)), 1e-6)
    soh = np.where(np.isfinite(capacity) & (capacity > 0), capacity / fresh, 1.0)
    return np.clip(soh, 0.0, 1.0)


def _cap_samples(X: np.ndarray, y: np.ndarray, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsample world-model windows deterministically when a run is too large."""
    if len(y) <= max_windows:
        return X, y
    rng = np.random.default_rng(2026)
    idx = np.sort(rng.choice(len(y), size=max_windows, replace=False))
    return X[idx], y[idx]


def _cap_samples_with_meta(
    X: np.ndarray,
    y: np.ndarray,
    subset: np.ndarray,
    cell: np.ndarray,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample world-model windows while preserving metadata alignment."""
    if len(y) <= max_windows:
        return X, y, subset, cell
    rng = np.random.default_rng(2026)
    idx = np.sort(rng.choice(len(y), size=max_windows, replace=False))
    return X[idx], y[idx], subset[idx], cell[idx]


def main() -> None:
    """Train the W2 Mamba/GRU world model CLI."""
    parser = argparse.ArgumentParser(description="Train Mamba/GRU world model on NASA PCoE")
    parser.add_argument("--pcoe-dir", type=Path, default=Path("data/nasa_pcoe/B000x"))
    parser.add_argument("--randomized-dir", type=Path, default=Path("data/nasa_pcoe/Randomized"))
    parser.add_argument("--soc-weights", type=Path, default=Path("outputs/soc_finetuned.h5"))
    parser.add_argument("--no-soc-model", action="store_true", help="Use strict Coulomb SOC labels only")
    parser.add_argument("--soh-ckpt", type=Path, default=Path("outputs/soh_baseline.pt"))
    parser.add_argument("--dataset", type=Path, default=None, help="Load a prebuilt W2 tensor package")
    parser.add_argument("--dataset-out", type=Path, default=Path("outputs/world_model_train_data.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=100_000)
    parser.add_argument("--limit-pcoe-files", type=int, default=None)
    parser.add_argument("--limit-randomized-files", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-cells", nargs="*", default=None)
    parser.add_argument("--eval-rollout", action="store_true")
    parser.add_argument("--eval-randomized-full", action="store_true")
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--rollout-stride", type=int, default=64)
    parser.add_argument("--max-rollouts", type=int, default=None)
    parser.add_argument("--rollout-cells", nargs="*", default=None)
    parser.add_argument("--rollout-subsets", nargs="*", default=None)
    parser.add_argument("--randomized-report-out", type=Path, default=Path("outputs/world_model_randomized_full_eval.metrics.json"))
    parser.add_argument("--randomized-eval-stride", type=int, default=64)
    parser.add_argument("--randomized-rollout-stride", type=int, default=256)
    parser.add_argument("--gru-fallback", action="store_true", help="Force GRU when mamba-ssm is unavailable")
    parser.add_argument("--build-only", action="store_true", help="Only build and save W2 train tensors")
    args = parser.parse_args()

    cfg = WorldModelConfig(seq_len=args.seq_len, use_mamba=not args.gru_fallback)
    if args.eval_randomized_full:
        model, _ = load_world_model_checkpoint(args.out)
        report = evaluate_randomized_directory(
            model,
            args.randomized_dir,
            cache_dir=args.cache_dir or Path("outputs/cache/world_model_randomized_full"),
            seq_len=args.seq_len,
            stride=args.randomized_eval_stride,
            batch_size=args.batch_size,
            rollout_horizon=args.rollout_horizon,
            rollout_stride=args.randomized_rollout_stride,
            max_files=args.limit_randomized_files,
        )
        args.randomized_report_out.parent.mkdir(parents=True, exist_ok=True)
        args.randomized_report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if args.dataset is not None and args.dataset.exists():
        bundle = load_world_tensors(args.dataset)
    else:
        bundle = build_world_tensors(
            args.pcoe_dir,
            args.randomized_dir,
            soc_weights=args.soc_weights,
            soh_ckpt=args.soh_ckpt,
            seq_len=args.seq_len,
            stride=args.stride,
            max_windows=args.max_windows,
            use_soc_model=not args.no_soc_model,
            limit_pcoe_files=args.limit_pcoe_files,
            limit_randomized_files=args.limit_randomized_files,
            cache_dir=args.cache_dir,
        )
        save_world_tensors(bundle, args.dataset_out)
    if args.build_only:
        print(json.dumps(bundle["meta"], indent=2))
        return
    if args.eval_rollout:
        model, previous_metrics = load_world_model_checkpoint(args.out)
        rollout = evaluate_rollout_drift(
            model,
            bundle,
            horizon=args.rollout_horizon,
            seq_len=args.seq_len,
            stride=args.rollout_stride,
            max_rollouts=args.max_rollouts,
            cells=args.rollout_cells,
            subsets=args.rollout_subsets,
        )
        metrics = {**previous_metrics, "rollout": rollout}
        args.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(rollout, indent=2))
        return

    dataset = tensors_to_dataset(bundle)
    if args.val_cells:
        train_ds, val_ds = split_bundle_by_cell(bundle, args.val_cells)
    else:
        train_ds, val_ds = split_dataset(dataset, val_ratio=args.val_ratio)
    model, metrics = train_world_model(
        train_ds,
        cfg,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    metrics["dataset"] = bundle.get("meta", {})
    metrics["final_val"] = evaluate_world_model(model, val_ds, batch_size=args.batch_size)
    if bundle.get("traces"):
        try:
            metrics["rollout"] = evaluate_rollout_drift(
                model,
                bundle,
                horizon=args.rollout_horizon,
                seq_len=args.seq_len,
                stride=args.rollout_stride,
                max_rollouts=args.max_rollouts,
                cells=args.rollout_cells or args.val_cells,
                subsets=args.rollout_subsets,
            )
        except Exception as exc:
            metrics["rollout_error"] = repr(exc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "metrics": metrics}, args.out)
    args.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
