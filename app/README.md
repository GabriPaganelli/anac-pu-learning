# app — APPalti: Risk Scoring degli Appalti Pubblici
# *app — APPalti: Risk Scoring for Public Procurement*

---

## Italiano

Applicazione Streamlit per il calcolo del **rischio corruttivo** su singoli appalti o batch CSV. Usa RE LightGBM (nnPU loss, γ=1) addestrato on-the-fly sulla distribuzione storica ANAC (2008–oggi).

La cartella è **standalone**: non dipende da `selected_models/` o `metrics/` a runtime.

---

## Struttura

```
app/
├── app.py                  # Interfaccia Streamlit (form + scoring + bootstrap + SHAP)
├── scorer.py               # Training e scoring wrapper (run_scoring, run_bootstrap, run_shap)
├── re_lgbm.py              # RE LightGBM production — fork di selected_models/re_lgbm.py
│                           #   (run_oof rimosso, num_threads aggiunto per parallelismo bootstrap)
├── prepare_data.py                # Script one-time: pre-encoding parquet nativi → data/
├── smoke_test.py          # Smoke test (dati reali, nessuna scrittura su data/)
└── data/
    ├── M1_ready.parquet    # Feature matrix M1 (9.5M contratti, float32 + label)
    ├── M2_ready.parquet    # Feature matrix M2
    ├── M3_ready.parquet    # Feature matrix M3
    ├── encodings.json      # {colonna: [sorted_unique_values]} per categoriali
    ├── feature_registry.json   # Metadati feature (stage, tipo input, etichette UI)
    └── contesto_province.csv   # Dati ISTAT provinciali (disoccupazione, reddito, omicidi)
```

---

## Setup (una tantum)

I file `data/` sono già presenti nel repository — rieseguire `prepare_data.py` solo se i parquet sorgente cambiano.

Eseguire **prima** di avviare l'app, dalla root del progetto:

```bash
python app/prepare_data.py
```

Legge i parquet nativi da `anac/output/parquet/model/nativi/` e produce:
- `data/M{1,2,3}_ready.parquet` — feature matrix float32 + label (tutte le righe, no filtro fold)
- `data/encodings.json` — mapping categoriali per l'encoding del nuovo contratto

---

## Avvio

```bash
streamlit run app/app.py
```

---

## Funzionamento

### Scoring singolo
Il form raccoglie i dati disponibili dell'appalto (M1 obbligatorio, M2/M3 opzionali).
Il modello viene addestrato on-the-fly su tutta la distribuzione storica e restituisce
il **rank percentile** del contratto (quanti appalti storici supera per rischio).

### Bootstrap CI
Ripete lo scoring su B sottoinsiemi bootstrap dei dati storici.
Restituisce il percentile 5°–95° della distribuzione del rank — misura la stabilità del risultato.
Parallelizzato via `ThreadPoolExecutor` (LightGBM rilascia il GIL durante il training C++).

### SHAP
Contributi per-feature calcolati con TreeSHAP built-in di LightGBM (`pred_contrib=True`).
Si applicano agli score grezzi (pre-Platt): il ranking è invariante a trasformazioni monotone.

### Batch CSV
Carica un CSV con N appalti. Le righe con lo stesso set di feature disponibili vengono
raggruppate ed elaborate con un unico training — significativamente più veloce di N scoring singoli.

---

## Modello

| Parametro | Valore |
|-----------|--------|
| Algoritmo | RE LightGBM (nnPU loss, PNPU) |
| γ | 1.0 (N certi a peso pieno) |
| π | 0.02 per tutti i dataset |
| \|U\_train\| | \|P\| / π, bounded in [3 000, 50 000] |
| num\_leaves | 15 |
| learning\_rate | 0.02 |
| Early stopping | ROC-AUC su 15% holdout, patience=200 |

---

## Prerequisiti

```bash
pip install streamlit pandas numpy lightgbm scipy pyarrow plotly
```

---

## English

Streamlit application for computing **corruption risk** on individual contracts or batch CSV files. Uses RE LightGBM (nnPU loss, γ=1) trained on-the-fly on the historical ANAC distribution (2008–present).

The folder is **standalone**: it does not depend on `selected_models/` or `metrics/` at runtime.

---

## Structure

```
app/
├── app.py                  # Streamlit UI (form + scoring + bootstrap + SHAP)
├── scorer.py               # Training and scoring wrapper (run_scoring, run_bootstrap, run_shap)
├── re_lgbm.py              # RE LightGBM production — fork of selected_models/re_lgbm.py
│                           #   (run_oof removed, num_threads added for bootstrap parallelism)
├── prepare_data.py                # One-time script: pre-encode native parquets → data/
├── smoke_test.py          # Smoke test (real data, no writes to data/)
└── data/
    ├── M1_ready.parquet    # Feature matrix M1 (9.5M contracts, float32 + label)
    ├── M2_ready.parquet    # Feature matrix M2
    ├── M3_ready.parquet    # Feature matrix M3
    ├── encodings.json      # {column: [sorted_unique_values]} for categorical features
    ├── feature_registry.json   # Feature metadata (stage, input type, UI labels)
    └── contesto_province.csv   # ISTAT provincial data (unemployment, income, homicide rate)
```

---

## Setup (one-time)

Run **before** starting the app, from the project root:

```bash
python app/prepare_data.py
```

Reads native parquets from `anac/output/parquet/model/nativi/` and produces:
- `data/M{1,2,3}_ready.parquet` — float32 feature matrix + label (all rows, no fold filter)
- `data/encodings.json` — categorical mappings for encoding new contracts

The `data/` files are already present in the repository — re-run `prepare_data.py` only if the source parquets change.

---

## Running

```bash
streamlit run app/app.py
```

---

## How it works

### Single scoring
The form collects the available contract data (M1 required, M2/M3 optional).
The model is trained on-the-fly on the full historical distribution and returns
the **rank percentile** of the contract (how many historical contracts it beats in risk).

### Bootstrap CI
Repeats scoring on B bootstrap subsets of the historical data.
Returns the 5th–95th percentile of the rank distribution — measures result stability.
Parallelised via `ThreadPoolExecutor` (LightGBM releases the GIL during C++ training).

### SHAP
Per-feature contributions computed with LightGBM's built-in TreeSHAP (`pred_contrib=True`).
Applied to raw scores (pre-Platt): ranking is invariant to monotone transformations.

### Batch CSV
Loads a CSV with N contracts. Rows sharing the same set of available features are
grouped and processed with a single training run — significantly faster than N individual scorings.

---

## Model

| Parameter | Value |
|-----------|-------|
| Algorithm | RE LightGBM (nnPU loss, PNPU) |
| γ | 1.0 (certain negatives at full weight) |
| π | 0.02 for all datasets |
| \|U\_train\| | \|P\| / π, bounded in [3 000, 50 000] |
| num\_leaves | 15 |
| learning\_rate | 0.02 |
| Early stopping | ROC-AUC on 15% holdout, patience=200 |

---

## Prerequisites

```bash
pip install streamlit pandas numpy lightgbm scipy pyarrow plotly
```
