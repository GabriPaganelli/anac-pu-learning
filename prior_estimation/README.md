# prior_estimation — Stima della prior di classe (α / π)
# *prior_estimation — Class prior estimation (α / π)*

---

## Italiano

Stima della proporzione di contratti corrotti (α) nel framework PU Learning su dati di appalti pubblici italiani (ANAC).

Quattro metodi implementati: **Elkan-Noto**, **Blanchard Quantili**, **KM2** (DEDPUL/KMPE) e **PULSNAR**.

---

## Struttura

```
prior_estimation/
├── run_prior_estimation.py   # Script unico: esegue tutti e 4 i metodi
├── KMPE.py                   # Vendored da dimonenka/DEDPUL (MIT License)
└── results/
    ├── prior_estimates_M1.txt
    ├── prior_estimates_M2.txt
    ├── prior_estimates_M3.txt
    ├── alpha_estimates.tsv
    ├── bic_vs_cluster_count.png
    └── predictions.tsv
```

---

## Prerequisiti

```bash
pip install pandas numpy scikit-learn lightgbm scipy pyarrow
```

**KM2** è incluso tramite `KMPE.py` (vendored, nessuna installazione aggiuntiva).

**PULSNAR** richiede installazione separata:

```bash
pip install git+https://github.com/unmtransinfo/PULSNAR.git
```

PULSNAR usa internamente R via `rpy2`. Installare R (https://www.r-project.org/) e `rpy2`:

```bash
pip install rpy2 xgboost catboost
```

Se KM2 o PULSNAR non sono disponibili, lo script salta il metodo corrispondente e continua.

---

## Dataset

Lo script legge i parquet prodotti dalla pipeline `anac/`:

| Metodo | Dataset |
|--------|---------|
| Elkan-Noto, Blanchard | `anac/output/parquet/model/nativi/M{n}.parquet` |
| KM2, PULSNAR | `anac/output/parquet/model/preprocessed/M{n}.parquet` |

I dataset nativi vengono usati con LightGBM (gestisce NA nativamente). I dataset preprocessed vengono usati con KM2 (kernel RBF) e PULSNAR (GMM interno), che non tollerano NA.

---

## Utilizzo

```bash
python run_prior_estimation.py
```

Modificare la sezione **CONFIG** in testa allo script per cambiare modello o parametri:

```python
MODEL_NUMBER         = 3        # 1=M1, 2=M2, 3=M3
TEST_MODE            = False    # True = smoke test rapido (~1 min)
N_RUNS_PULSNAR       = 30
MAX_UNL_EN_BLANCHARD = 200_000
MAX_UNL_KM2          = 1_500
N_RUNS_KM2           = 10
MAX_UNL_PULSNAR      = 50_000
```

**Tempi indicativi (produzione, `TEST_MODE = False`):**

| Metodo | Tempo per modello |
|--------|-------------------|
| Elkan-Noto + Blanchard | 30–60 min (LightGBM 5-fold CV) |
| KM2 (10 run) | 30–60 min |
| PULSNAR (30 run) | 2.5–5 ore |

L'output viene salvato in `results/prior_estimates_M{n}.txt`.

---

## Modelli

Tre modelli temporali annidati (feature set cumulativo):

| Modello | Fase | Feature disponibili |
|---------|------|---------------------|
| M1 | Ex ante | Dati del bando al momento della pubblicazione |
| M2 | Durante | M1 + dati aggiudicazione |
| M3 | Ex post | M2 + dati esecuzione contratto |

---

## Note metodologiche

In breve:
- **Elkan-Noto** stima `c = P(S=1|Y=1)` (label frequency) — tende a sovrastimare α sotto SAR.
- **Blanchard Quantili** è un upper bound teorico di π — più robusto a violazioni di SCAR.
- **KM2** e **PULSNAR** stimano direttamente `P(Y=1)` nel set non etichettato — più instabili con pochi positivi (<500).

Il valore operativo riportato è quello di PULSNAR, unico metodo robusto sotto SAR, arrotondato per eccesso (approccio conservativo per ridurre i falsi negativi).

---

## English

Estimation of the proportion of corrupt contracts (α) in the PU Learning framework on Italian public procurement (appalti pubblici) data from ANAC.

Four methods implemented: **Elkan-Noto**, **Blanchard Quantile**, **KM2** (DEDPUL/KMPE), and **PULSNAR**.

---

## Structure

```
prior_estimation/
├── run_prior_estimation.py   # Single script: runs all 4 methods
├── KMPE.py                   # Vendored from dimonenka/DEDPUL (MIT License)
└── results/
    ├── prior_estimates_M1.txt
    ├── prior_estimates_M2.txt
    ├── prior_estimates_M3.txt
    ├── alpha_estimates.tsv
    ├── bic_vs_cluster_count.png
    └── predictions.tsv
```

---

## Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm scipy pyarrow
```

**KM2** is included via `KMPE.py` (vendored, no additional installation needed).

**PULSNAR** requires a separate installation:

```bash
pip install git+https://github.com/unmtransinfo/PULSNAR.git
```

PULSNAR uses R internally via `rpy2`. Install R (https://www.r-project.org/) and `rpy2`:

```bash
pip install rpy2 xgboost catboost
```

If KM2 or PULSNAR are unavailable, the script skips the corresponding method and continues.

---

## Dataset

The script reads the parquet files produced by the `anac/` pipeline:

| Method | Dataset |
|--------|---------|
| Elkan-Noto, Blanchard | `anac/output/parquet/model/nativi/M{n}.parquet` |
| KM2, PULSNAR | `anac/output/parquet/model/preprocessed/M{n}.parquet` |

Native (nativi) datasets are used with LightGBM (handles NA natively). Preprocessed datasets are used with KM2 (RBF kernel) and PULSNAR (internal GMM), which cannot handle NA.

---

## Usage

```bash
python run_prior_estimation.py
```

Modify the **CONFIG** section at the top of the script to change model or parameters:

```python
MODEL_NUMBER         = 3        # 1=M1, 2=M2, 3=M3
TEST_MODE            = False    # True = quick smoke test (~1 min)
N_RUNS_PULSNAR       = 30
MAX_UNL_EN_BLANCHARD = 200_000
MAX_UNL_KM2          = 1_500
N_RUNS_KM2           = 10
MAX_UNL_PULSNAR      = 50_000
```

**Indicative runtimes (production, `TEST_MODE = False`):**

| Method | Time per model |
|--------|----------------|
| Elkan-Noto + Blanchard | 30–60 min (LightGBM 5-fold CV) |
| KM2 (10 runs) | 30–60 min |
| PULSNAR (30 runs) | 2.5–5 hours |

Output is saved to `results/prior_estimates_M{n}.txt`.

---

## Temporal models

| Model | Phase   | Available features                            |
|-------|---------|-----------------------------------------------|
| M1    | Ex ante | Tender notice (bando) data at publication     |
| M2    | During  | M1 + award (aggiudicazione) data              |
| M3    | Ex post | M2 + contract execution data                  |

---

## Methodological notes

In brief:
- **Elkan-Noto** estimates `c = P(S=1|Y=1)` (label frequency) — tends to overestimate α under SAR (Selected At Random) assumption violations.
- **Blanchard Quantile** is a theoretical lower bound on π — more robust to SCAR violations.
- **KM2** and **PULSNAR** directly estimate `P(Y=1)` in the unlabeled set — less stable with few positives (<500).

The reported operational value is that of PULSNAR, the only method robust under SAR, rounded up (conservative approach to reduce false negatives).
