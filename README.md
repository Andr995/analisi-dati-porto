# AIS ETA Prediction - Spatio-Temporal Transformer

Questo repository contiene un Jupyter Notebook (`AIS_ETA_Transformer_Fixed.ipynb`) che implementa un modello di Deep Learning basato su architettura **Spatio-Temporal Transformer** per prevedere l'Estimated Time of Arrival (ETA) di navi mercantili. 
Il modello utilizza dati AIS (Automatic Identification System) e rileva anomalie di navigazione.

##  Caratteristiche Principali

Il notebook è stato profondamente ottimizzato per la ricerca applicata (R&D) in ambito aziendale, con i seguenti miglioramenti metodologici e tecnici rispetto alle versioni precedenti:

*   **Valutazione Rigorosa (Train/Val/Test Split):** Implementato uno split `70/15/15` (Train, Validation, Test). Questo elimina il *data leakage* durante la fase di inferenza, garantendo metriche di valutazione robuste su dati mai visti dal modello. L'Early Stopping utilizza correttamente solo il Validation set.
*   **Machine Learning Baseline:** Prima del modello Deep Learning, viene addestrato un modello **Random Forest Regressor** per stabilire una solida baseline. Questo passaggio è cruciale per dimostrare il reale vantaggio (in termini di rapporto costi/benefici computazionali) offerto dall'architettura Transformer rispetto al ML classico sui dati tabulari "piatti".
*   **Deep Learning Baseline (LSTM):** Integrata un'architettura ricorrente **Spatio-Temporal LSTM** a due layer come step intermedio. Questo crea un benchmark per le serie temporali, permettendo di verificare empiricamente se il meccanismo di Self-Attention del Transformer cattura pattern spaziotemporali non lineari che una RNN standard fatica a memorizzare a lungo termine.
*   **Exploratory Data Analysis (EDA):** Aggiunta una sezione di EDA iniziale per "ascoltare" i dati prima della modellazione (es. distribuzione SOG e matrici di correlazione).
*   **Explainability Geografica Interattiva (Folium):** I pesi di attenzione reali dell'ultimo layer del Transformer (estratti tramite PyTorch forward hooks) sono mappati direttamente sulle coordinate GPS usando la libreria `folium`. Questo permette di visualizzare interattivamente l'esatto momento/luogo in cui il modello rileva anomalie.
*   **Fallback CPU:** Esecuzione sicura su macchine sprovviste di GPU.
*   **Feature Engineering:** Distanza geodetica calcolata con l'equazione di Haversine, variabili temporali codificate ciclicamente (sin/cos).

## 🛠️ Architettura dei Modelli Analizzati

1.  **Baseline ML (Random Forest):** Approccio standard su feature appiattite.
2.  **Baseline DL (LSTM):** Rete ricorrente a 2 layer (hidden_dim=64), ottimizzata per serie temporali classiche.
3.  **Spatio-Temporal Transformer:** Encoder basato su `nn.TransformerEncoderLayer` con Positional Encoding.
    *   `d_model` = 64
    *   `n_heads` = 4
    *   `n_layers` = 2
    *   `dim_feedforward` = 256
    *   `dropout` = 0.1

## 📊 Panoramica delle Metriche (Esempio su dati Mock)

Su un dataset mock di test (viaggio con simulazione di rallentamento anomalo), i tre approcci hanno mostrato le seguenti differenze sostanziali:

| Modello | MAE | RMSE | R² | MAPE | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 74.67 min | 211.28 min | 0.8820 | 4.63% | Ottima baseline per dati tabulari. |
| **LSTM** | 618.72 min | 827.71 min | -0.8103 | 29.52% | Difficoltà di convergenza (vanishing gradient) in assenza di pattern ciclici forti. |
| **Transformer** | 151.16 min | 197.09 min | 0.8974 | 9.29% | Buone performance, superiore all'LSTM ma non batte l'efficienza del Random Forest in questo specifico scenario tabulare. |

## 📦 Prerequisiti e Installazione

Il notebook richiede Python 3.8+ e le seguenti librerie principali:

```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn folium
