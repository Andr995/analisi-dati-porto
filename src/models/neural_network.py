"""
Rete neurale LSTM per la previsione del tempo di sosta in porto.

Architettura:
    input  →  LSTM (stacked)  →  Dropout  →  Linear  →  output (scalar)

Il modello riceve in input sequenze di operazioni portuali precedenti
e prevede il tempo di sosta (in ore) dell'operazione successiva.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPorto(nn.Module):
    """
    Rete LSTM per regressione sul tempo di sosta in porto.

    Parameters
    ----------
    input_size : int
        Numero di feature per ogni timestep.
    hidden_size : int
        Dimensione dello stato nascosto dell'LSTM.
    num_layers : int
        Numero di layer LSTM impilati.
    dropout : float
        Probabilità di dropout tra i layer LSTM (attivo solo se num_layers > 1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        out, _ = self.lstm(x)
        # Prendiamo solo l'output dell'ultimo timestep
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.head(last).squeeze(-1)


class MLPPorto(nn.Module):
    """
    Rete feed-forward (MLP) per regressione su singoli record portuali.

    Accetta feature piatte (non sequenze) e restituisce il tempo di sosta.

    Parameters
    ----------
    input_size : int
        Numero di feature di input.
    hidden_sizes : list[int]
        Dimensioni degli strati nascosti.
    dropout : float
        Probabilità di dropout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_size)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        return self.net(x).squeeze(-1)
