"""SOH comparison with a frozen W2 Mamba encoder plus a shallow head.

This module is a W4/W5 architecture ablation. It reuses the BatteryML-compatible
NASA SOH target definition, but extracts sequence embeddings from the trained
W2 world model and fits a small Ridge head to predict SOH. To avoid target
leakage, the SOH input channel in the world-model tensor is neutralized before
embedding extraction.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from craic_pipeline.world_model_mamba import load_world_model_checkpoint


def physical_stats_features(X: np.ndarray) -> np.ndarray:
    """Compute non-SOH sequence statistics for a leakage-free SOH baseline."""
    use = X.copy()
    use[:, :, 1] = 1.0
    channels = [0, 2, 3, 4, 5]
    feats = []
    for idx in channels:
        arr = use[:, :, idx]
        feats.extend(
            [
                arr[:, -1],
                np.nanmean(arr, axis=1),
                np.nanstd(arr, axis=1),
                np.nanmin(arr, axis=1),
                np.nanmax(arr, axis=1),
            ]
        )
    return np.vstack(feats).T.astype(np.float32)


@torch.no_grad()
def extract_mamba_embeddings(model, X: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    """Extract frozen W2 hidden embeddings after neutralizing the SOH channel."""
    model = model.to(device)
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        batch = X[start : start + batch_size].copy()
        batch[:, :, 1] = 1.0
        tensor = torch.from_numpy(batch.astype(np.float32)).to(device)
        outputs.append(model.encode(tensor).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_ridge_head(features: np.ndarray, target: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> tuple[object, dict]:
    """Fit a standardized Ridge head and return validation metrics."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(features[train_mask], target[train_mask])
    pred_train = np.clip(model.predict(features[train_mask]), 0.0, 1.2)
    pred_val = np.clip(model.predict(features[val_mask]), 0.0, 1.2)
    metrics = {
        "train_rmse_percent": _rmse(target[train_mask], pred_train) * 100.0,
        "val_rmse_percent": _rmse(target[val_mask], pred_val) * 100.0,
        "val_mae_percent": float(mean_absolute_error(target[val_mask], pred_val) * 100.0),
        "val_samples": int(val_mask.sum()),
    }
    return model, metrics


def run_ablation(
    dataset_path: Path,
    world_model_path: Path,
    soh_baseline_path: Path,
    out: Path,
    *,
    train_cells: list[str],
    val_cells: list[str],
    batch_size: int,
    device_name: str,
) -> dict:
    """Run the SOH feature/head ablation and persist a compact comparison."""
    bundle = torch.load(dataset_path, map_location="cpu", weights_only=False)
    X = np.asarray(bundle["X"], dtype=np.float32)
    cells = np.asarray(bundle["cell"]).astype(str)
    target = np.asarray(X[:, -1, 1], dtype=np.float32)
    finite = np.isfinite(target)
    train_mask = finite & np.isin(cells, train_cells)
    val_mask = finite & np.isin(cells, val_cells)
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"empty train/val split: train={train_mask.sum()} val={val_mask.sum()}")

    physical_model, physical_metrics = train_ridge_head(physical_stats_features(X), target, train_mask, val_mask)

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    world_model, world_metrics = load_world_model_checkpoint(world_model_path)
    embeddings = extract_mamba_embeddings(world_model, X, batch_size=batch_size, device=device)
    mamba_head, mamba_metrics = train_ridge_head(embeddings, target, train_mask, val_mask)

    capacity_ratio_metrics = _load_capacity_ratio_metrics(soh_baseline_path)
    result = {
        "dataset": str(dataset_path),
        "world_model": str(world_model_path),
        "world_model_backend": getattr(world_model, "backend", "unknown"),
        "device": str(device),
        "target": "SOH = capacity / fresh_capacity from W2 tensor channel; SOH channel neutralized before feature extraction",
        "train_cells": train_cells,
        "val_cells": val_cells,
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "comparison": {
            "capacity_ratio_oracle": capacity_ratio_metrics,
            "physical_stats_ridge_no_soh": physical_metrics,
            "mamba_embedding_ridge_no_soh": mamba_metrics,
        },
        "note": "This is an architecture ablation for evidence; the capacity-ratio oracle is not deployment-realistic.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fout:
        pickle.dump(
            {
                "physical_model": physical_model,
                "mamba_head": mamba_head,
                "metrics": result,
                "train_cells": train_cells,
                "val_cells": val_cells,
            },
            fout,
        )
    out.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _load_capacity_ratio_metrics(path: Path) -> dict:
    """Read the existing W1 capacity-ratio baseline metrics if available."""
    metrics_path = path.with_suffix(".metrics.json")
    if not metrics_path.exists():
        return {"val_rmse_percent": float("nan"), "note": "metrics not found"}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "val_rmse_percent": float(metrics.get("val_rmse_percent", float("nan"))),
        "val_cells": metrics.get("val_cells", []),
        "note": "uses NASA Capacity-derived ratio feature; oracle-like baseline",
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE for SOH fractions."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    """CLI entry point for the Mamba-head SOH ablation."""
    parser = argparse.ArgumentParser(description="SOH ablation with frozen W2 Mamba embeddings")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/world_model_train_data.pt"))
    parser.add_argument("--world-model", type=Path, default=Path("outputs/world_model.pt"))
    parser.add_argument("--soh-baseline", type=Path, default=Path("outputs/soh_baseline.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/soh_mamba_head.pt"))
    parser.add_argument("--train-cells", nargs="+", default=["B0005", "B0006", "B0007"])
    parser.add_argument("--val-cells", nargs="+", default=["B0018"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_ablation(
        args.dataset,
        args.world_model,
        args.soh_baseline,
        args.out,
        train_cells=args.train_cells,
        val_cells=args.val_cells,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
