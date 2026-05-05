"""
Analisi Dati Porto — Script principale
======================================

Esegue l'intera pipeline di analisi dati portuali con:
  1. Generazione del dataset sintetico
  2. Pre-elaborazione e costruzione delle sequenze
  3. Addestramento del modello LSTM (rete neurale)
  4. Addestramento del modello Transformer
  5. Confronto delle prestazioni e visualizzazione

Utilizzo
--------
    python analisi_porto.py

Per un ciclo rapido (meno epoche):
    python analisi_porto.py --epochs 10 --samples 2000
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # backend non-interattivo per ambienti headless
import matplotlib.pyplot as plt
import numpy as np

from src.data_generator import generate_porto_dataset
from src.preprocessing import PortoPreprocessor, build_sequences, train_val_test_split
from src.models.neural_network import LSTMPorto, MLPPorto
from src.models.transformer_model import TransformerPorto
from src.train import fit
from src.evaluate import predict, compute_metrics, print_metrics


def plot_history(
    histories: dict[str, dict[str, list[float]]],
    out_path: str = "risultati_training.png",
) -> None:
    """Salva un grafico comparativo delle loss di training."""
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    if len(histories) == 1:
        axes = [axes]

    for ax, (name, hist) in zip(axes, histories.items()):
        ax.plot(hist["train_loss"], label="Train Loss")
        ax.plot(hist["val_loss"], label="Val Loss")
        ax.set_title(f"{name} — Loss (MSE)")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("MSE (ore²)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nGrafico di training salvato in '{out_path}'")


def plot_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_points: int = 200,
    out_path: str = "risultati_previsioni.png",
) -> None:
    """Salva un grafico delle previsioni vs valori reali."""
    n = min(n_points, len(y_true))
    fig, axes = plt.subplots(1, len(predictions), figsize=(7 * len(predictions), 4))
    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.plot(y_true[:n], label="Reale", alpha=0.7)
        ax.plot(y_pred[:n], label="Previsto", alpha=0.7)
        ax.set_title(f"{name} — Previsioni vs Reale")
        ax.set_xlabel("Campione")
        ax.set_ylabel("Sosta (ore)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Grafico previsioni salvato in '{out_path}'")


def main(epochs: int = 30, n_samples: int = 5000, seq_len: int = 30) -> None:
    print("=" * 55)
    print("  Analisi Dati Porto — Transformer & Reti Neurali")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 1. Generazione e pre-elaborazione dei dati
    # ------------------------------------------------------------------
    print(f"\n[1/5] Generazione dataset ({n_samples} record)...")
    df = generate_porto_dataset(n_samples=n_samples)
    print(f"      Dataset: {df.shape[0]} righe × {df.shape[1]} colonne")
    print(f"      Target  : sosta_ore — min={df['sosta_ore'].min():.1f}h "
          f"max={df['sosta_ore'].max():.1f}h "
          f"media={df['sosta_ore'].mean():.1f}h")

    print("\n[2/5] Pre-elaborazione...")
    preprocessor = PortoPreprocessor()
    X_flat, y = preprocessor.fit_transform(df)
    n_features = X_flat.shape[1]
    print(f"      Feature totali: {n_features}")

    # Split per MLP (dati piatti)
    X_tr_f, y_tr, X_val_f, y_val, X_te_f, y_te = train_val_test_split(X_flat, y)
    print(f"      Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_te)}")

    # Split per modelli sequenziali
    from src.preprocessing import build_sequences
    X_seq, y_seq = build_sequences(X_flat, y, seq_len=seq_len)
    X_tr_s, y_tr_s, X_val_s, y_val_s, X_te_s, y_te_s = train_val_test_split(
        X_seq, y_seq
    )

    histories: dict[str, dict[str, list[float]]] = {}
    test_preds: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 3. Rete neurale — MLP
    # ------------------------------------------------------------------
    print("\n[3/5] Addestramento MLP...")
    mlp = MLPPorto(input_size=n_features)
    hist_mlp = fit(
        mlp, X_tr_f, y_tr, X_val_f, y_val,
        epochs=epochs, batch_size=256, lr=1e-3, patience=8,
    )
    histories["MLP"] = hist_mlp
    preds_mlp = predict(mlp, X_te_f)
    test_preds["MLP"] = preds_mlp
    print_metrics("MLP", compute_metrics(y_te, preds_mlp))

    # ------------------------------------------------------------------
    # 4. Rete neurale — LSTM
    # ------------------------------------------------------------------
    print("\n[4/5] Addestramento LSTM...")
    lstm = LSTMPorto(input_size=n_features, hidden_size=128, num_layers=2)
    hist_lstm = fit(
        lstm, X_tr_s, y_tr_s, X_val_s, y_val_s,
        epochs=epochs, batch_size=128, lr=5e-4, patience=8,
    )
    histories["LSTM"] = hist_lstm
    preds_lstm = predict(lstm, X_te_s)
    test_preds["LSTM"] = preds_lstm
    print_metrics("LSTM", compute_metrics(y_te_s, preds_lstm))

    # ------------------------------------------------------------------
    # 5. Transformer
    # ------------------------------------------------------------------
    print("\n[5/5] Addestramento Transformer...")
    transformer = TransformerPorto(
        input_size=n_features,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
    )
    hist_tr = fit(
        transformer, X_tr_s, y_tr_s, X_val_s, y_val_s,
        epochs=epochs, batch_size=128, lr=5e-4, patience=8,
    )
    histories["Transformer"] = hist_tr
    preds_tf = predict(transformer, X_te_s)
    test_preds["Transformer"] = preds_tf
    print_metrics("Transformer", compute_metrics(y_te_s, preds_tf))

    # ------------------------------------------------------------------
    # Visualizzazioni
    # ------------------------------------------------------------------
    plot_history(histories)
    # Per LSTM/Transformer usiamo y_te_s; per MLP y_te
    plot_predictions(
        y_te_s,
        {"LSTM": preds_lstm, "Transformer": preds_tf},
    )

    print("\nAnalisi completata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi Dati Porto")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Numero massimo di epoche (default: 30)")
    parser.add_argument("--samples", type=int, default=5000,
                        help="Dimensione del dataset sintetico (default: 5000)")
    parser.add_argument("--seq_len", type=int, default=30,
                        help="Lunghezza sequenza temporale (default: 30)")
    args = parser.parse_args()
    main(epochs=args.epochs, n_samples=args.samples, seq_len=args.seq_len)
