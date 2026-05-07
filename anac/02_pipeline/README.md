# 02_pipeline — Pipeline feature engineering ANAC
# *02_pipeline — ANAC feature engineering pipeline*

---

## Italiano

Pipeline di 17 step che trasforma i CSV grezzi ANAC in dataset pronti per il training di modelli di machine learning. Produce 6 parquet (M1/M2/M3 × nativi/preprocessed) con la colonna `fold` per la cross-validation a 4 fold.

---

## Prerequisiti

Prima di eseguire questa pipeline devono essere disponibili:

```
data/raw/                  # CSV ANAC (scaricati manualmente da dati.anticorruzione.it)
data/territorial/          # contesto_province.csv (da utils/download_contesto.py)
labels/                    # cig_condannati.csv, cig_scagionati.csv (da 01_sentenze/)
data/lookup/               # lavorazioni_tipo.csv, categorie_opera.csv (manuali)
```

---

## Esecuzione

```bash
python run_pipeline.py
```

Per un run parziale, commentare le righe non necessarie in `run_pipeline.py`.

---

## Step della pipeline

### Pre-processing (idempotente)

| Script | Descrizione |
|--------|-------------|
| `01_filter_cig_annual.py` | Normalizza i CSV CIG annuali 2008-2025 (separatore, selezione colonne) |
| `02_filter_columns.py` | Filtra le colonne di tutti i dataset ANAC secondo `variables/variable_selection.xlsx` |
| `03_build_lookups.py` | Genera le tabelle di lookup codice→descrizione in `data/lookup/` |

### Costruzione parquet base

| Script | Descrizione |
|--------|-------------|
| `04_build_bando_cig.py` | Crea `bando_cig_all.parquet` da CIG annuali + join label + join territoriale |

### Arricchimento progressivo

Ogni script legge e riscrive `bando_cig_all.parquet`:

| Script | Feature aggiunte |
|--------|-----------------|
| `05_build_aggiudicazioni.py` | Dati aggiudicazione, flag procedurali, filtro Opzione A (rimuove CIG con esito anomalo senza override) |
| `06_build_aggiudicatari.py` | `tipo_soggetto_agg`: SINGOLA / ATI / CONSORZIO / ... |
| `07_build_stazione_appaltante.py` | `natura_giuridica_SA`: 8 categorie |
| `08_build_quadro_economico.py` | `pct_riserva_base`, `pct_overrun_core`, `pct_riserva_consumata`, fallback `importo_sicurezza_pct` |
| `09_build_avvio_contratto.py` | `lag_stipula`, `durata_pianificata`, `consegna_frazionata/sotto_riserva` |
| `10_build_varianti.py` | `n_varianti`, `flag_variante_sostanziale`, `pct_overrun_variante`, ... |
| `11_build_sospensioni.py` | `n_sospensioni`, `flag_sospensione`, `pct_durata_sospesa`, ... |
| `12_build_sal.py` | `n_sal`, `flag_in_ritardo`, `flag_proroga` |
| `13_build_subappalti.py` | `flag_subappalto` (OR tra bando e subappalti.csv) |
| `14_build_lavorazioni.py` | `tipo_lavorazione_macro`: COSTRUZIONE / RISANAMENTO / MANUTENZIONE |
| `15_build_collaudo.py` | `esito_collaudo`: POSITIVO / NEGATIVO (6.5% coverage) |

### Costruzione dataset model

| Script | Descrizione |
|--------|-------------|
| `16_build_model_datasets.py` | Separa M1/M2/M3 × nativi/preprocessed; droppa `data_pubblicazione` dal sorgente |
| `17_assign_folds.py` | Assegna la colonna `fold` (0–3) con strategia stratificata P+N e round-robin U |

---

## Output

```
output/parquet/
├── bando_cig_all.parquet              # Sorgente completo (66+ colonne)
└── model/
    ├── nativi/                        # Per XGBoost / LightGBM (NA nativi)
    │   ├── M1.parquet
    │   ├── M2.parquet
    │   └── M3.parquet
    └── preprocessed/                  # Per Logistica / SVM (discretizzazione + encoding)
        ├── M1.parquet
        ├── M2.parquet
        └── M3.parquet
```

Ogni parquet contiene le colonne:
- `cig` — identificativo gara (chiave)
- `label` — 1 (condannato) / 0 (scagionato) / NaN (non etichettato)
- `fold` — 0–3 per gli esempi inclusi nella cross-validation, NaN per il resto
- Feature del modello specifico (M1 ⊂ M2 ⊂ M3)

---

## Architettura dei modelli

```
M1 (ex ante)  — feature disponibili alla pubblicazione del bando
M2 (durante)  — M1 + dati post-aggiudicazione
M3 (ex post)  — M2 + dati di esecuzione contratto (varianti, SAL, sospensioni, collaudo)
```

**Fold assignment** (script 17):
- Labeled (P+N): stratificato per label, propagazione bottom-up M3→M2→M1
  - M3 determina la base del bilanciamento (dataset più piccolo)
  - M2 eredita i fold di M3 via join CIG; i CIG esclusivi M2 ricevono assegnazione casuale
  - M1 eredita i fold di M2 in modo analogo
- Unlabeled (U): campione casuale, round-robin sui 4 fold
  - M1: 10.000 U per fold — M2: 6.000 — M3: 3.000
- Il seed viene scelto tra 30 candidati per minimizzare la deviazione delle distribuzioni U tra fold

---

## Dizionario variabili

Il file `variables/data_dictionary.xlsx` contiene la documentazione completa di ogni feature: descrizione, fonte, fase modello, tipo dato, percentuale di null, trattamento NA e note sul segnale predittivo.

Per rigenerarlo dopo modifiche al parquet:

```bash
python utils/create_data_dictionary.py
```

---

## English

17-step pipeline that transforms raw ANAC CSVs into datasets ready for machine learning model training. Produces 6 parquets (M1/M2/M3 × nativi/preprocessed) with a `fold` column for 4-fold cross-validation.

---

## Prerequisites

Before running this pipeline, the following must be available:

```
data/raw/                  # ANAC CSVs (manually downloaded from dati.anticorruzione.it)
data/territorial/          # contesto_province.csv (from utils/download_contesto.py)
labels/                    # cig_condannati.csv, cig_scagionati.csv (from 01_sentenze/)
data/lookup/               # lavorazioni_tipo.csv, categorie_opera.csv (manual)
```

---

## Execution

```bash
python run_pipeline.py
```

For a partial run, comment out the unneeded lines in `run_pipeline.py`.

---

## Pipeline steps

### Pre-processing (idempotent)

| Script | Description |
|--------|-------------|
| `01_filter_cig_annual.py` | Normalises annual CIG CSVs 2008–2025 (separator, column selection) |
| `02_filter_columns.py` | Filters columns across all ANAC datasets per `variables/variable_selection.xlsx` |
| `03_build_lookups.py` | Generates code→description lookup tables in `data/lookup/` |

### Base parquet construction

| Script | Description |
|--------|-------------|
| `04_build_bando_cig.py` | Creates `bando_cig_all.parquet` from annual CIGs + label join + territorial join |

### Progressive enrichment

Each script reads and rewrites `bando_cig_all.parquet`:

| Script | Features added |
|--------|----------------|
| `05_build_aggiudicazioni.py` | Award data, procedural flags, Option A filter (removes CIGs with anomalous outcome without override) |
| `06_build_aggiudicatari.py` | `tipo_soggetto_agg`: SINGOLA / ATI / CONSORZIO / ... |
| `07_build_stazione_appaltante.py` | `natura_giuridica_SA`: 8 categories |
| `08_build_quadro_economico.py` | `pct_riserva_base`, `pct_overrun_core`, `pct_riserva_consumata`, fallback `importo_sicurezza_pct` |
| `09_build_avvio_contratto.py` | `lag_stipula`, `durata_pianificata`, `consegna_frazionata/sotto_riserva` |
| `10_build_varianti.py` | `n_varianti`, `flag_variante_sostanziale`, `pct_overrun_variante`, ... |
| `11_build_sospensioni.py` | `n_sospensioni`, `flag_sospensione`, `pct_durata_sospesa`, ... |
| `12_build_sal.py` | `n_sal`, `flag_in_ritardo`, `flag_proroga` |
| `13_build_subappalti.py` | `flag_subappalto` (OR across bando and subappalti.csv) |
| `14_build_lavorazioni.py` | `tipo_lavorazione_macro`: COSTRUZIONE / RISANAMENTO / MANUTENZIONE |
| `15_build_collaudo.py` | `esito_collaudo`: POSITIVO / NEGATIVO (6.5% coverage) |

### Model dataset construction

| Script | Description |
|--------|-------------|
| `16_build_model_datasets.py` | Splits M1/M2/M3 × nativi/preprocessed; drops `data_pubblicazione` from source |
| `17_assign_folds.py` | Assigns `fold` column (0–3) using stratified P+N strategy and round-robin U |

---

## Output

```
output/parquet/
├── bando_cig_all.parquet              # Full source (66+ columns)
└── model/
    ├── nativi/                        # For XGBoost / LightGBM (native NAs)
    │   ├── M1.parquet
    │   ├── M2.parquet
    │   └── M3.parquet
    └── preprocessed/                  # For Logistic / SVM (discretisation + encoding)
        ├── M1.parquet
        ├── M2.parquet
        └── M3.parquet
```

Each parquet contains:
- `cig` — tender identifier (key)
- `label` — 1 (condemned) / 0 (cleared) / NaN (unlabelled)
- `fold` — 0–3 for examples included in cross-validation, NaN otherwise
- Model-specific features (M1 ⊂ M2 ⊂ M3)

---

## Model architecture

```
M1 (ex ante)  — features available at tender publication
M2 (durante)  — M1 + post-award data
M3 (ex post)  — M2 + contract execution data (variants, SAL, suspensions, testing)
```

**Fold assignment** (script 17):
- Labeled (P+N): stratified by label, bottom-up propagation M3→M2→M1
  - M3 determines the balancing baseline (smallest dataset)
  - M2 inherits M3 folds via CIG join; M2-exclusive CIGs receive random assignment
  - M1 inherits M2 folds analogously
- Unlabeled (U): random sample, round-robin across 4 folds
  - M1: 10,000 U per fold — M2: 6,000 — M3: 3,000
- Seed chosen among 30 candidates to minimise deviation of U distributions across folds

---

## Variable dictionary

`variables/data_dictionary.xlsx` contains full documentation for every feature: description, source, model stage, data type, null percentage, NA treatment, and notes on predictive signal.

To regenerate after changes to the parquet:

```bash
python utils/create_data_dictionary.py
```
