# data/

Questa cartella contiene i dataset generati dalla pipeline.

I file CSV vengono creati automaticamente eseguendo:

```bash
python src/data_generator.py
```

oppure come effetto collaterale di:

```bash
python analisi_porto.py
```

I file di dati non sono inclusi nel repository perché vengono generati
in modo deterministico a partire dal seed configurabile.
