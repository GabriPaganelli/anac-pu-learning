# Appalti ANAC — Rilevamento corruzione con PU Learning
# *Appalti ANAC — Corruption Detection with PU Learning*

---

## Italiano

Pipeline per la costruzione di dataset e modelli di rilevamento della corruzione negli appalti pubblici italiani (BDNCP ANAC, 2008-2025).

Il progetto usa **Positive-Unlabeled (PU) Learning**: le gare d'appalto con sentenza TAR favorevole al ricorrente o risolte per reato/antimafia sono etichettate come positive (corrotte), quelle con sentenza negativa come negative (scagionate), tutto il resto rimane non etichettato.

---

## Struttura

```
anac/
├── 01_sentenze/          # Pipeline di estrazione etichette da sentenze TAR/CdS (OpenGA)
├── 02_pipeline/          # Pipeline di feature engineering da CSV ANAC (17 step)
├── data/
│   ├── raw/              # CSV ANAC originali (non versionati)
│   ├── territorial/      # Contesto provinciale ISTAT/MEF (contesto_province.csv)
│   └── lookup/           # Tabelle di lookup manuali e generate
├── labels/
│   ├── cig_condannati.csv  # CIG label=1 (sentenza + fine-contratto)
│   └── cig_scagionati.csv  # CIG label=0 (ricorso rigettato in tribunale)
├── output/
│   └── parquet/
│       ├── bando_cig_all.parquet      # Parquet sorgente (66+ colonne)
│       └── model/
│           ├── nativi/                # M1/M2/M3 per XGBoost/LightGBM
│           └── preprocessed/         # M1/M2/M3 per Logistica/SVM
├── utils/
│   ├── download_bdncp.py              # Scarica CSV BDNCP da dati.anticorruzione.it (CKAN)
│   ├── download_contesto.py           # Scarica dati territoriali ISTAT/MEF
│   └── create_data_dictionary.py      # Genera variables/data_dictionary.xlsx
├── variables/
│   ├── variable_selection.xlsx        # Selezione e configurazione variabili
│   └── data_dictionary.xlsx           # Dizionario dati generato automaticamente
```

---

## Esecuzione dall'inizio

### Prerequisiti

```bash
pip install pandas numpy pyarrow openpyxl requests scikit-learn lightgbm xgboost
```

### 1. Dati territoriali (ISTAT/MEF)

```bash
python utils/download_contesto.py
```

Scarica tasso di disoccupazione, reddito IRPEF pro capite e tasso di omicidi per 100k per ogni provincia italiana (2007-2024). Output: `data/territorial/contesto_province.csv`.

### 2. Etichette da sentenze TAR/CdS

```bash
cd 01_sentenze
python run_pipeline.py
```

Scarica le sentenze dal portale OpenGA, identifica i CIG coinvolti, e produce:
- `labels/cig_condannati.csv` — CIG con sentenza favorevole al ricorrente o risolti per reato/antimafia
- `labels/cig_scagionati.csv` — CIG con ricorso rigettato dal tribunale

Vedi `01_sentenze/README.md` per i dettagli.

### 3. Pipeline feature engineering

```bash
cd 02_pipeline
python run_pipeline.py
```

Legge i CSV ANAC in `data/raw/` e costruisce progressivamente `output/parquet/bando_cig_all.parquet`, poi produce i 6 parquet model (`M1`, `M2`, `M3` × nativi/preprocessed) con la colonna `fold` per la cross-validation.

I CSV ANAC vengono scaricati automaticamente dalla CKAN API di dati.anticorruzione.it:

```bash
python utils/download_bdncp.py
```

Opzioni: `--no-cache` (forza re-download), `--dry-run` (stampa URL senza scaricare), `--since ANNO` (scarica i bandi-cig solo dall'anno indicato).

Vedi `02_pipeline/README.md` per i dettagli.

---

## Architettura dei modelli

Tre modelli temporali annidati (feature set cumulativo):

| Modello | Fase         | Feature disponibili                     |
|---------|--------------|------------------------------------------|
| M1      | Ex ante      | Dati del bando al momento della pubblicazione |
| M2      | Durante      | M1 + dati aggiudicazione                |
| M3      | Ex post      | M2 + dati esecuzione contratto          |

Ogni modello è disponibile in due versioni:
- **Nativi**: variabili continue senza preprocessamento (per XGBoost/LightGBM)
- **Preprocessed**: discretizzazione log-quantile e encoding categoriale (per Logistica/SVM)

La colonna `fold` (0–3) permette cross-validation stratificata: i fold sono assegnati in modo che positivi e negativi siano distribuiti uniformemente, con propagazione bottom-up M3→M2→M1.

---

## Fonti dati

| Dato | Fonte |
|------|-------|
| Bandi CIG | ANAC BDNCP |
| Aggiudicazioni, varianti, SAL, ... | ANAC BDNCP |
| Sentenze TAR/CdS | openga.giustizia-amministrativa.it |
| Disoccupazione provinciale | ISTAT SDMX |
| Reddito IRPEF pro capite | MEF |
| Tasso omicidi per 100k | ISTAT BES |

---

## English

Pipeline for building datasets and models to detect corruption in Italian public procurement (BDNCP ANAC, 2008–2025).

The project uses **Positive-Unlabeled (PU) Learning**: procurement tenders (gare d'appalto) with a court ruling (sentenza) favorable to the claimant (ricorrente) at an administrative court, or contracts terminated for proven crimes or anti-mafia reasons, are labeled positive (corrupt). Tenders where the appeal (ricorso) was rejected are labeled negative (cleared). Everything else remains unlabeled.

---

## Structure

```
anac/
├── 01_sentenze/          # Label extraction pipeline from TAR/CdS court rulings (OpenGA)
├── 02_pipeline/          # Feature engineering pipeline from ANAC CSVs (17 steps)
├── data/
│   ├── raw/              # Raw ANAC CSV files (not versioned)
│   ├── territorial/      # Provincial context ISTAT/MEF (contesto_province.csv)
│   └── lookup/           # Manual and generated lookup tables
├── labels/
│   ├── cig_condannati.csv  # CIG (tender IDs) label=1 (ruling + contract termination)
│   └── cig_scagionati.csv  # CIG label=0 (appeal rejected in court)
├── output/
│   └── parquet/
│       ├── bando_cig_all.parquet      # Source parquet (66+ columns)
│       └── model/
│           ├── nativi/                # M1/M2/M3 for XGBoost/LightGBM
│           └── preprocessed/         # M1/M2/M3 for Logistic/SVM
├── utils/
│   ├── download_bdncp.py              # Downloads BDNCP CSV files from dati.anticorruzione.it (CKAN)
│   ├── download_contesto.py           # Downloads territorial data from ISTAT/MEF
│   └── create_data_dictionary.py      # Generates variables/data_dictionary.xlsx
└── variables/
    ├── variable_selection.xlsx        # Variable selection and configuration
    └── data_dictionary.xlsx           # Auto-generated data dictionary
```

---

## Running from scratch

### Prerequisites

```bash
pip install pandas numpy pyarrow openpyxl requests scikit-learn lightgbm xgboost
```

### 1. Territorial data (ISTAT/MEF)

```bash
python utils/download_contesto.py
```

Downloads unemployment rate (tasso di disoccupazione), per-capita IRPEF income (reddito IRPEF pro capite) and homicide rate per 100k inhabitants for every Italian province (2007–2024). Output: `data/territorial/contesto_province.csv`.

### 2. Labels from TAR/CdS court rulings (sentenze)

```bash
cd 01_sentenze
python run_pipeline.py
```

Downloads court rulings from the OpenGA portal, identifies the tender IDs (CIG — Codice Identificativo Gara) involved in litigation, and produces:
- `labels/cig_condannati.csv` — CIG labeled positive: ruling favorable to the claimant (ricorrente), or contract terminated for crime/anti-mafia reasons
- `labels/cig_scagionati.csv` — CIG labeled negative: appeal (ricorso) rejected by the court

See `01_sentenze/README.md` for details.

### 3. Feature engineering pipeline

```bash
cd 02_pipeline
python run_pipeline.py
```

Reads ANAC CSV files from `data/raw/` and incrementally builds `output/parquet/bando_cig_all.parquet`, then produces the 6 model parquets (`M1`, `M2`, `M3` × native/preprocessed) with a `fold` column for cross-validation.

ANAC CSV files are downloaded automatically from the dati.anticorruzione.it CKAN API:

```bash
python utils/download_bdncp.py
```

Options: `--no-cache` (force re-download), `--dry-run` (print URLs without downloading), `--since YEAR` (download bandi-cig from that year onward).

See `02_pipeline/README.md` for details.

---

## Model architecture

Three temporally nested models (cumulative feature sets):

| Model | Phase        | Available features                          |
|-------|--------------|---------------------------------------------|
| M1    | Ex ante      | Tender notice (bando) data at publication   |
| M2    | During       | M1 + award (aggiudicazione) data            |
| M3    | Ex post      | M2 + contract execution data                |

Each model is available in two variants:
- **Native**: continuous variables without preprocessing (for XGBoost/LightGBM)
- **Preprocessed**: log-quantile discretization and categorical encoding (for Logistic Regression/SVM)

The `fold` column (0–3) enables stratified cross-validation: folds are assigned so that positives and negatives are evenly distributed, with bottom-up propagation M3→M2→M1.

---

## Data sources

| Data | Source | Update frequency |
|------|--------|-----------------|
| CIG tender notices (bandi) | ANAC BDNCP | Annual (one file per year) |
| Awards, variants, SAL, ... | ANAC BDNCP | Single file |
| TAR/CdS court rulings (sentenze) | openga.giustizia-amministrativa.it | On demand |
| Provincial unemployment | ISTAT SDMX | Annual |
| Per-capita IRPEF income | MEF | Annual |
| Homicide rate per 100k | ISTAT BES | Multi-year intervals |
