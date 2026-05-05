"""
Utility di addestramento per i modelli di analisi portuali.

Fornisce:
- `train_epoch`  : un'epoca di addestramento
- `eval_epoch`   : un'epoca di valutazione
- `fit`          : ciclo di addestramento completo con early stopping
- `PortoDataset` : dataset PyTorch per dati tabulari e sequenziali
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PortoDataset(Dataset):
    """
    Dataset PyTorch compatibile sia con dati piatti (MLP)
    sia con sequenze (LSTM / Transformer).

    Parameters
    ----------
    X : np.ndarray, shape (n, ...) 
        Feature (può essere 2-D per MLP o 3-D per modelli sequenziali).
    y : np.ndarray, shape (n,)
        Target (tempo di sosta in ore).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Esegue una singola epoca di addestramento e restituisce la loss media."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Esegue una singola epoca di valutazione e restituisce la loss media."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def fit(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Ciclo di addestramento completo con early stopping.

    Parameters
    ----------
    model : nn.Module
    X_train, y_train : dati di addestramento
    X_val, y_val     : dati di validazione
    epochs           : numero massimo di epoche
    batch_size       : dimensione del batch
    lr               : learning rate iniziale
    patience         : numero di epoche senza miglioramento prima di fermarsi
    device           : dispositivo PyTorch (default: auto-detect)
    verbose          : stampa la loss ad ogni epoca

    Returns
    -------
    history : dict con chiavi "train_loss" e "val_loss"
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience // 2, factor=0.5
    )

    train_ds = PortoDataset(X_train, y_train)
    val_ds = PortoDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state: dict = {}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:>4}/{epochs}  "
                f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}"
            )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping all'epoca {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    return history
