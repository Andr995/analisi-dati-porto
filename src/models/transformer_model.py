"""
Modello Transformer per l'estrazione di pattern e la previsione
del tempo di sosta nelle operazioni portuali.

Architettura:
    input  →  Linear embedding  →  Positional Encoding
           →  N × TransformerEncoderLayer
           →  Global Average Pooling
           →  Linear head  →  output (scalar)

Il Transformer è particolarmente adatto ad estrarre dipendenze
a lungo raggio tra operazioni portuali non-consecutive.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Codifica posizionale sinusoidale standard (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Dimensione dell'embedding.
    max_len : int
        Lunghezza massima della sequenza supportata.
    dropout : float
        Dropout applicato dopo la somma con il positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerPorto(nn.Module):
    """
    Transformer Encoder per regressione sul tempo di sosta in porto.

    Parameters
    ----------
    input_size : int
        Numero di feature per ogni timestep.
    d_model : int
        Dimensione dell'embedding interno del Transformer.
    nhead : int
        Numero di teste di attenzione (deve dividere d_model).
    num_encoder_layers : int
        Numero di layer TransformerEncoder impilati.
    dim_feedforward : int
        Dimensione dello strato feed-forward interno.
    dropout : float
        Probabilità di dropout.
    max_seq_len : int
        Lunghezza massima della sequenza supportata dal positional encoding.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        # Proiezione lineare delle feature → d_model
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_seq_len, dropout=dropout
        )

        # Stack di TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm (più stabile)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Testa di regressione
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)
        src_key_padding_mask : torch.Tensor or None
            Maschera booleana (batch, seq_len) — True dove il token è padding.

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        # Proiezione e positional encoding
        x = self.input_projection(x)          # (B, T, d_model)
        x = self.pos_encoding(x)               # (B, T, d_model)

        # Encoder Transformer
        enc = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, T, d_model)

        # Global average pooling sull'asse temporale
        pooled = enc.mean(dim=1)               # (B, d_model)

        return self.head(pooled).squeeze(-1)   # (B,)
