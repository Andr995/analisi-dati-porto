"""
Test per i componenti del progetto analisi-dati-porto.

Coprono:
- data_generator    : forma e tipologie del dataset
- preprocessing     : encoding, normalizzazione, sequenze, split
- neural_network    : forward pass MLP e LSTM
- transformer_model : forward pass Transformer
- train / evaluate  : singola epoca di training e predizione
"""

import numpy as np
import pytest
import torch

# ──────────────────────────────────────────────────────────────────────────────
# data_generator
# ──────────────────────────────────────────────────────────────────────────────

from src.data_generator import generate_porto_dataset


def test_dataset_shape():
    df = generate_porto_dataset(n_samples=100)
    assert df.shape[0] == 100
    assert "sosta_ore" in df.columns
    assert "data_arrivo" in df.columns


def test_dataset_no_nulls():
    df = generate_porto_dataset(n_samples=200)
    assert df.isnull().sum().sum() == 0


def test_dataset_target_range():
    df = generate_porto_dataset(n_samples=500)
    assert df["sosta_ore"].min() >= 1
    assert df["sosta_ore"].max() <= 120


def test_dataset_reproducible():
    df1 = generate_porto_dataset(n_samples=50, seed=0)
    df2 = generate_porto_dataset(n_samples=50, seed=0)
    assert (df1["sosta_ore"].values == df2["sosta_ore"].values).all()


# ──────────────────────────────────────────────────────────────────────────────
# preprocessing
# ──────────────────────────────────────────────────────────────────────────────

from src.preprocessing import (
    PortoPreprocessor,
    build_sequences,
    train_val_test_split,
)


def _make_preprocessed(n=300):
    df = generate_porto_dataset(n_samples=n, seed=1)
    prep = PortoPreprocessor()
    X, y = prep.fit_transform(df)
    return X, y, prep, df


def test_preprocessor_output_shapes():
    X, y, _, df = _make_preprocessed(200)
    assert X.shape[0] == 200
    assert y.shape[0] == 200
    assert X.ndim == 2
    assert y.ndim == 1


def test_preprocessor_float32():
    X, y, _, _ = _make_preprocessed(100)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_preprocessor_transform_consistent():
    df = generate_porto_dataset(n_samples=200, seed=2)
    prep = PortoPreprocessor()
    X_fit, y_fit = prep.fit_transform(df)
    X_tr, y_tr = prep.transform(df)
    np.testing.assert_array_almost_equal(X_fit, X_tr)
    np.testing.assert_array_equal(y_fit, y_tr)


def test_preprocessor_transform_before_fit_raises():
    prep = PortoPreprocessor()
    df = generate_porto_dataset(n_samples=50)
    with pytest.raises(RuntimeError):
        prep.transform(df)


def test_build_sequences_shape():
    X, y, _, _ = _make_preprocessed(200)
    seq_len = 20
    X_seq, y_seq = build_sequences(X, y, seq_len=seq_len)
    n_feat = X.shape[1]
    assert X_seq.shape == (200 - seq_len, seq_len, n_feat)
    assert y_seq.shape == (200 - seq_len,)


def test_train_val_test_split_sizes():
    X = np.zeros((1000, 5), dtype=np.float32)
    y = np.zeros(1000, dtype=np.float32)
    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)
    total = len(y_tr) + len(y_val) + len(y_te)
    assert total == 1000
    assert len(y_tr) > len(y_val)
    assert len(y_tr) > len(y_te)


# ──────────────────────────────────────────────────────────────────────────────
# neural_network (MLP + LSTM)
# ──────────────────────────────────────────────────────────────────────────────

from src.models.neural_network import LSTMPorto, MLPPorto


def test_mlp_forward():
    model = MLPPorto(input_size=10)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8,)


def test_lstm_forward():
    model = LSTMPorto(input_size=10, hidden_size=32, num_layers=2)
    x = torch.randn(8, 15, 10)
    out = model(x)
    assert out.shape == (8,)


def test_mlp_no_nan():
    model = MLPPorto(input_size=8)
    x = torch.randn(16, 8)
    out = model(x)
    assert not torch.isnan(out).any()


def test_lstm_no_nan():
    model = LSTMPorto(input_size=8)
    x = torch.randn(16, 10, 8)
    out = model(x)
    assert not torch.isnan(out).any()


# ──────────────────────────────────────────────────────────────────────────────
# transformer_model
# ──────────────────────────────────────────────────────────────────────────────

from src.models.transformer_model import TransformerPorto, PositionalEncoding


def test_positional_encoding_shape():
    pe = PositionalEncoding(d_model=32, max_len=50)
    x = torch.zeros(4, 20, 32)
    out = pe(x)
    assert out.shape == (4, 20, 32)


def test_transformer_forward():
    model = TransformerPorto(
        input_size=10, d_model=32, nhead=4, num_encoder_layers=1, dim_feedforward=64
    )
    x = torch.randn(8, 15, 10)
    out = model(x)
    assert out.shape == (8,)


def test_transformer_no_nan():
    model = TransformerPorto(input_size=8, d_model=16, nhead=2, num_encoder_layers=1)
    x = torch.randn(4, 10, 8)
    out = model(x)
    assert not torch.isnan(out).any()


def test_transformer_with_mask():
    model = TransformerPorto(input_size=6, d_model=16, nhead=2, num_encoder_layers=1)
    x = torch.randn(4, 10, 6)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[:, -2:] = True  # ultimi 2 token mascherati
    out = model(x, src_key_padding_mask=mask)
    assert out.shape == (4,)


# ──────────────────────────────────────────────────────────────────────────────
# train + evaluate
# ──────────────────────────────────────────────────────────────────────────────

from src.train import PortoDataset, train_epoch, eval_epoch
from src.evaluate import predict, compute_metrics


def _tiny_data(n=64, n_feat=8):
    X = np.random.randn(n, n_feat).astype(np.float32)
    y = np.random.rand(n).astype(np.float32) * 10
    return X, y


def test_porto_dataset():
    X, y = _tiny_data()
    ds = PortoDataset(X, y)
    assert len(ds) == len(y)
    x0, y0 = ds[0]
    assert x0.shape == (8,)
    assert y0.shape == ()


def test_train_eval_epoch():
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn

    X, y = _tiny_data(n=64, n_feat=8)
    ds = PortoDataset(X, y)
    loader = DataLoader(ds, batch_size=16)
    model = MLPPorto(input_size=8, hidden_sizes=[16])
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    device = torch.device("cpu")

    tr_loss = train_epoch(model, loader, optimizer, criterion, device)
    vl_loss = eval_epoch(model, loader, criterion, device)
    assert tr_loss >= 0
    assert vl_loss >= 0


def test_predict_shape():
    X, _ = _tiny_data(n=50, n_feat=8)
    model = MLPPorto(input_size=8, hidden_sizes=[16])
    preds = predict(model, X)
    assert preds.shape == (50,)


def test_compute_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)


def test_compute_metrics_keys():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y, y + 0.5)
    assert set(m.keys()) == {"mae", "rmse", "mape", "r2"}
