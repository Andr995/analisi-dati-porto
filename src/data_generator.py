"""
Generatore di dati sintetici per le operazioni portuali.

Crea un dataset che simula:
- Arrivi e partenze delle navi
- Tipologie di cargo
- Volumi movimentati
- Tempi di sosta in porto
- Condizioni meteo-marine
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


SHIP_TYPES = ["portacontainer", "petroliera", "bulk_carrier", "ro_ro", "passeggeri"]
CARGO_TYPES = ["container", "petrolio", "cereali", "acciaio", "veicoli", "generale"]
ORIGINS = ["Genova", "Valencia", "Rotterdam", "Amburgo", "Barcellona", "Marsiglia",
           "Istanbul", "Pireo", "Algeciras", "Anversa"]
WEATHER = ["sereno", "nuvoloso", "pioggia", "vento_forte", "nebbia"]


def generate_porto_dataset(
    n_samples: int = 5000,
    start_date: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un dataset sintetico di operazioni portuali.

    Parameters
    ----------
    n_samples : int
        Numero di record da generare.
    start_date : str
        Data di inizio della serie storica (formato YYYY-MM-DD).
    seed : int
        Seme per la riproducibilità.

    Returns
    -------
    pd.DataFrame
        Dataset con le caratteristiche delle operazioni portuali.
    """
    rng = np.random.default_rng(seed)

    # Date distribuite nell'arco temporale
    base = datetime.strptime(start_date, "%Y-%m-%d")
    days_offset = rng.integers(0, 365 * 4, size=n_samples)
    dates = [base + timedelta(days=int(d)) for d in days_offset]
    dates.sort()

    # Caratteristiche categoriche
    ship_type = rng.choice(SHIP_TYPES, size=n_samples)
    cargo_type = rng.choice(CARGO_TYPES, size=n_samples)
    origin = rng.choice(ORIGINS, size=n_samples)
    weather = rng.choice(WEATHER, size=n_samples, p=[0.35, 0.30, 0.15, 0.12, 0.08])

    # Lunghezza nave (metri)
    ship_length = np.clip(rng.normal(200, 60, n_samples), 50, 400).astype(int)

    # Stazza lorda (tonnellate)
    gross_tonnage = (ship_length ** 2.1 * rng.uniform(0.8, 1.2, n_samples)).astype(int)

    # Volume di cargo (tonnellate) correlato alla stazza
    cargo_volume = (gross_tonnage * rng.uniform(0.3, 0.75, n_samples)).astype(int)

    # Effetti meteo sul volume movimentato
    weather_factor = np.where(
        np.isin(weather, ["vento_forte", "nebbia"]), 0.75,
        np.where(weather == "pioggia", 0.90, 1.0),
    )
    cargo_volume = (cargo_volume * weather_factor).astype(int)

    # Numero di gru/mezzi impiegati
    n_cranes = np.clip(rng.integers(1, 8, n_samples), 1, 7)

    # Tempo di sosta in porto (ore) — target principale
    base_stay = cargo_volume / (n_cranes * rng.uniform(400, 600, n_samples))
    weather_delay = np.where(
        np.isin(weather, ["vento_forte", "nebbia"]),
        rng.uniform(2, 8, n_samples),
        0.0,
    )
    sosta_ore = np.clip(base_stay + weather_delay + rng.normal(0, 1, n_samples), 1, 120)

    # Costo operativo (EUR)
    cost_per_hour = rng.uniform(800, 2500, n_samples)
    costo_operativo = (sosta_ore * cost_per_hour).astype(int)

    df = pd.DataFrame(
        {
            "data_arrivo": dates,
            "tipo_nave": ship_type,
            "tipo_cargo": cargo_type,
            "porto_provenienza": origin,
            "condizioni_meteo": weather,
            "lunghezza_nave_m": ship_length,
            "stazza_lorda_t": gross_tonnage,
            "volume_cargo_t": cargo_volume,
            "n_gru": n_cranes,
            "sosta_ore": sosta_ore.round(2),
            "costo_operativo_eur": costo_operativo,
        }
    )

    # Feature temporali
    df["anno"] = df["data_arrivo"].dt.year
    df["mese"] = df["data_arrivo"].dt.month
    df["giorno_settimana"] = df["data_arrivo"].dt.dayofweek  # 0=lunedì

    return df


if __name__ == "__main__":
    df = generate_porto_dataset()
    out_path = "data/operazioni_porto.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset salvato in '{out_path}' ({len(df)} record, {df.shape[1]} colonne)")
    print(df.describe())
