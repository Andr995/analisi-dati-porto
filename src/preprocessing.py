"""
Pipeline di pre-elaborazione per i dati portuali.

Gestisce:
- Encoding delle variabili categoriche
- Normalizzazione delle feature numeriche
- Costruzione di sequenze temporali per i modelli sequenziali
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


CATEGORICAL_COLS = [
    "tipo_nave",
    "tipo_cargo",
    "porto_provenienza",
    "condizioni_meteo",
]

NUMERIC_FEATURE_COLS = [
    "lunghezza_nave_m",
    "stazza_lorda_t",
    "volume_cargo_t",
    "n_gru",
    "mese",
    "giorno_settimana",
]

TARGET_COL = "sosta_ore"


class PortoPreprocessor:
    """
    Preprocessore per il dataset di operazioni portuali.

    Trasforma le colonne categoriche in interi e normalizza
    le feature numeriche con media 0 e deviazione standard 1.
    """

    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Pubblico
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Adatta il preprocessore sul dataset e restituisce (X, y).

        Parameters
        ----------
        df : pd.DataFrame
            Dataset grezzo generato da `data_generator`.

        Returns
        -------
        X : np.ndarray, shape (n, n_features)
        y : np.ndarray, shape (n,)
        """
        df = df.copy()
        df = self._encode_categoricals(df, fit=True)
        self.feature_cols = NUMERIC_FEATURE_COLS + list(self.label_encoders.keys())
        X_raw = df[self.feature_cols].values.astype(np.float32)
        X = self.scaler.fit_transform(X_raw).astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32)
        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Applica la trasformazione su un dataset non visto.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        X : np.ndarray
        y : np.ndarray
        """
        if not self._fitted:
            raise RuntimeError("Chiama fit_transform prima di transform.")
        df = df.copy()
        df = self._encode_categoricals(df, fit=False)
        X_raw = df[self.feature_cols].values.astype(np.float32)
        X = self.scaler.transform(X_raw).astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32)
        return X, y

    # ------------------------------------------------------------------
    # Privato
    # ------------------------------------------------------------------

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))
        return df


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Costruisce sequenze di lunghezza fissa per modelli sequenziali.

    I dati sono ordinati per data di arrivo, quindi le sequenze
    catturano la dipendenza temporale tra operazioni consecutive.

    Parameters
    ----------
    X : np.ndarray, shape (n, n_features)
    y : np.ndarray, shape (n,)
    seq_len : int
        Numero di timestep per ogni sequenza.

    Returns
    -------
    X_seq : np.ndarray, shape (n - seq_len, seq_len, n_features)
    y_seq : np.ndarray, shape (n - seq_len,)
        Il target è il valore che segue ogni sequenza.
    """
    n = len(X)
    X_seq = np.stack([X[i : i + seq_len] for i in range(n - seq_len)], axis=0)
    y_seq = y[seq_len:]
    return X_seq, y_seq


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[np.ndarray, ...]:
    """
    Suddivide i dati in train / validation / test preservando l'ordine temporale.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    n = len(X)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    return X_train, y_train, X_val, y_val, X_test, y_test
