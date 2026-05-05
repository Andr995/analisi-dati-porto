# analisi-dati-porto

Analisi dati sulle operazioni portuali con **reti neurali** (MLP, LSTM) e **modello Transformer**.

Il progetto dimostra come applicare tecniche di deep learning per estrarre pattern e prevedere
il **tempo di sosta in porto** delle navi, a partire da dati storici di operazioni portuali.

---

## Struttura del progetto

```
analisi-dati-porto/
├── analisi_porto.py          # Script principale — esegue l'intera pipeline
├── requirements.txt          # Dipendenze Python
├── src/
│   ├── data_generator.py     # Generatore di dataset sintetico
│   ├── preprocessing.py      # Pre-elaborazione e costruzione sequenze
│   ├── train.py              # Ciclo di addestramento con early stopping
│   ├── evaluate.py           # Metriche (MAE, RMSE, MAPE, R²) e predizione
│   └── models/
│       ├── neural_network.py # MLP e LSTM (PyTorch)
│       └── transformer_model.py  # Transformer Encoder (PyTorch)
├── data/                     # Dataset generati (non inclusi nel repo)
└── tests/
    └── test_porto.py         # Test unitari
```

---

## Installazione

```bash
pip install -r requirements.txt
```

---

## Utilizzo

```bash
# Esecuzione completa (30 epoche, 5000 campioni)
python analisi_porto.py

# Ciclo rapido per prove
python analisi_porto.py --epochs 10 --samples 2000

# Parametri disponibili
python analisi_porto.py --help
```

Al termine vengono prodotti:
- `risultati_training.png`   — curve di loss per i tre modelli
- `risultati_previsioni.png` — previsioni vs valori reali (LSTM e Transformer)

---

## Dataset sintetico

Il dataset simula operazioni portuali con le seguenti caratteristiche:

| Colonna              | Descrizione                                   |
|----------------------|-----------------------------------------------|
| `data_arrivo`        | Data di arrivo della nave                     |
| `tipo_nave`          | Tipologia (portacontainer, petroliera, …)     |
| `tipo_cargo`         | Tipo di merce trasportata                     |
| `porto_provenienza`  | Porto di origine                              |
| `condizioni_meteo`   | Meteo al momento dell'operazione              |
| `lunghezza_nave_m`   | Lunghezza della nave in metri                 |
| `stazza_lorda_t`     | Stazza lorda in tonnellate                    |
| `volume_cargo_t`     | Volume di cargo movimentato (t)               |
| `n_gru`              | Numero di gru/mezzi impiegati                 |
| **`sosta_ore`**      | **Target**: ore di sosta in porto             |
| `costo_operativo_eur`| Costo operativo dell'operazione (EUR)         |

---

## Modelli

### MLP (Multi-Layer Perceptron)
Rete feed-forward su singoli record:
`input → Linear(256) → ReLU → Dropout → Linear(128) → … → output`

### LSTM
Rete ricorrente su sequenze di 30 operazioni consecutive:
`input → LSTM(128, 2 layers) → Dropout → Linear → output`

### Transformer
Encoder Transformer su sequenze di 30 operazioni:
`input → Linear embedding → Positional Encoding → N × TransformerEncoderLayer → Average Pooling → head → output`

---

## Test

```bash
pip install pytest
pytest tests/ -v
```

---

## Dipendenze principali

- [PyTorch](https://pytorch.org/) — framework deep learning
- [scikit-learn](https://scikit-learn.org/) — preprocessing e metriche
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — manipolazione dati
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — visualizzazione
