"""
Metriche di valutazione per i modelli di analisi portuali.

Metriche calcolate:
- MAE  (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R²   (Coefficiente di determinazione)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train import PortoDataset


def predict(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Genera le previsioni di un modello su un array numpy.

    Parameters
    ----------
    model : nn.Module
    X : np.ndarray
    batch_size : int
    device : torch.device or None

    Returns
    -------
    preds : np.ndarray, shape (n,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Dataset senza target (usiamo zeri come placeholder)
    dummy_y = np.zeros(len(X), dtype=np.float32)
    ds = PortoDataset(X, dummy_y)
    loader = DataLoader(ds, batch_size=batch_size)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).cpu().numpy()
            preds.append(out)

    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calcola le metriche di regressione.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    dict con chiavi: mae, rmse, mape, r2
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE: evita divisione per zero
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    """Stampa le metriche in formato leggibile."""
    print(f"\n{'=' * 45}")
    print(f"  Modello: {name}")
    print(f"{'=' * 45}")
    print(f"  MAE  : {metrics['mae']:.3f} ore")
    print(f"  RMSE : {metrics['rmse']:.3f} ore")
    print(f"  MAPE : {metrics['mape']:.2f} %")
    print(f"  R²   : {metrics['r2']:.4f}")
    print(f"{'=' * 45}")
