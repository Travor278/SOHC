"""Mamba/GRU world model for NASA battery next-step dynamics.

The W2 model consumes sequences shaped `(B, L, 6)`:
`[SOC, SOH, V, I, T, action]`, and predicts the next
`[SOC, V, T, delta_SOH]` target. On Windows where `mamba-ssm` is unavailable,
the same API falls back to a two-layer GRU.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict `[SOC_next, V_next, T_next, delta_SOH]` from `(B,L,6)`."""
        h = self.input_proj(x.float())
        if self.backend == "mamba":
            for norm, layer in zip(self.norms, self.layers):
                h = h + layer(norm(h))
        else:
            h, _ = self.layers(h)
        return self.head(h[:, -1])

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
    del soc_csv, soh_ckpt
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for loader, root in ((load_pcoe_basic, Path(pcoe_dir)), (load_randomized_usage, Path(randomized_dir))):
        if not root.exists():
            continue
        V, I, T, _, cycle_id, _, capacity = loader(root)
        if V.size:
            arrays.extend(_series_to_windows(V, I, T, cycle_id, capacity, seq_len=seq_len, stride=stride))
    if not arrays:
        raise FileNotFoundError("no NASA windows found for world-model training")
    X = np.concatenate([item[0] for item in arrays], axis=0)
    y = np.concatenate([item[1] for item in arrays], axis=0)
    X, y = _cap_samples(X, y, max_windows)
    return TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())


def train_world_model(
    dataset: TensorDataset,
    cfg: WorldModelConfig,
    *,
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
    model = model.cpu()
    metrics = {"loss": history, "backend": model.backend, "samples": len(dataset), "device": str(device)}
    return model, metrics


def _series_to_windows(
    V: np.ndarray,
    I: np.ndarray,
    T: np.ndarray,
    cycle_id: np.ndarray,
    capacity: np.ndarray,
    *,
    seq_len: int,
    stride: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert continuous NASA series into per-cycle world-model windows."""
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for cid in np.unique(cycle_id):
        idx = np.where(cycle_id == cid)[0]
        if idx.size <= seq_len:
            continue
        soc = _approx_soc(I[idx], capacity[idx])
        soh = _approx_soh(capacity[idx])
        features = np.column_stack([soc, soh, V[idx], I[idx], T[idx], I[idx]])
        finite = np.isfinite(features).all(axis=1)
        if finite.sum() <= seq_len:
            continue
        features = features[finite]
        X, y = _window_one_cycle(features, seq_len=seq_len, stride=stride)
        if len(y):
            outputs.append((X, y))
    return outputs


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


def main() -> None:
    """Train the W2 Mamba/GRU world model CLI."""
    parser = argparse.ArgumentParser(description="Train Mamba/GRU world model on NASA PCoE")
    parser.add_argument("--pcoe-dir", type=Path, default=Path("data/nasa_pcoe/B000x"))
    parser.add_argument("--randomized-dir", type=Path, default=Path("data/nasa_pcoe/Randomized"))
    parser.add_argument("--soc-csv", type=Path, default=Path("outputs/soc_pred_nasa.csv"))
    parser.add_argument("--soh-ckpt", type=Path, default=Path("outputs/soh_baseline.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gru-fallback", action="store_true", help="Force GRU when mamba-ssm is unavailable")
    args = parser.parse_args()

    cfg = WorldModelConfig(seq_len=args.seq_len, use_mamba=not args.gru_fallback)
    dataset = build_training_dataset(
        args.pcoe_dir,
        args.randomized_dir,
        args.soc_csv,
        args.soh_ckpt,
        seq_len=args.seq_len,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    model, metrics = train_world_model(dataset, cfg, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "metrics": metrics}, args.out)
    args.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
