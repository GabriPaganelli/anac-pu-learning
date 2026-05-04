# models — Stimatori PU Learning per la rilevazione di corruzione negli appalti

Quattro famiglie di modelli che implementano strategie diverse per il problema Positive-Unlabeled (PU) su dati di appalti pubblici italiani (ANAC).

Ogni famiglia supporta tre dataset temporali (**M1/M2/M3**) e due varianti di loss (**PNPU** / **PU puro**), configurabili tramite flag in testa agli script.

---

## Struttura

```
models/
├── risk_estimators/         # Stimatori nnPU/PNPU (loss corretta per π)
│   ├── lgbm/nnpu_lgbm.py    #   LightGBM + custom objective
│   ├── logit/nnpu_logit.py  #   Regressione logistica custom (gradient ascent)
│   └── nn/nnpu_nn.py        #   MLP PyTorch
├── bagging/                 # PU Bagging (Mordelet & Vert 2014)
│   ├── pu_bagging_lightgbm.py
│   └── pu_extratrees.py
├── biased/                  # Biased learning (U come negativi pesati)
│   ├── gbm.py
│   ├── logit.py
│   ├── rf.py
│   ├── svm.py
│   └── run_all.py           #   Runner parallelo per tutti i modelli biased
└── em_like/                 # Iterativo EM-like (R + LightGBM)
    ├── lgbSoft.R            #   Soft: pesi frazionari da score LightGBM
    └── lgbHard.R            #   Hard: Reliable Negatives adattivi
```

---

## Famiglie di modelli

### risk_estimators

Implementano il **risk estimator nnPU** (Kiryo et al. 2017) con estensione **PNPU** che aggiunge un termine correttivo per i negativi certi N.

- **Loss nnPU**: `R = π·R_P(+) + max(0, R_U(-) - π·R_P(-))`
- **Loss PNPU**: aggiunge `(1-π)/n_U_tot · Σ_N BCE(f, -)`
- **Inner CV** annidato: selezione iperparametri su inner val (lift@1%), poi fit finale su tutto il training del fold.
- Flag `RETRAIN_FINAL = True` per rialleno su tutto il dataset con i parametri ottimali.

| Script | Base learner | Dataset |
|--------|-------------|---------|
| `nnpu_lgbm.py` | LightGBM (custom objective) | nativi |
| `nnpu_logit.py` | Logistica (gradient ascent custom) | preprocessed |
| `nnpu_nn.py` | MLP PyTorch (1-2 hidden layer) | preprocessed |

### bagging

**PU Bagging** (Mordelet & Vert 2014): B bootstrap da U di dimensione S, ogni base learner allena su P ∪ campione_U; score finale = media delle probabilità.

- `PNPU=True`: N certi fissi in ogni bootstrap (mai campionati).
- `PNPU=False`: N certi nel pool U, campionati come gli U.

| Script | Base learner | Dataset |
|--------|-------------|---------|
| `pu_bagging_lightgbm.py` | LightGBM (BCE standard) | nativi |
| `pu_extratrees.py` | ExtraTreesClassifier | preprocessed |

### biased

**Biased learning**: U trattati come negativi con peso ridotto `W_UNLABELED`.

- `PNPU=True`: N certi peso 1, U peso `W_UNLABELED`.
- `PNPU=False`: N certi e U entrambi con peso `W_UNLABELED`.

`run_all.py` lancia in parallelo tutti i modelli × M1/M2/M3 × PNPU/PU.

| Script | Modello | Dataset |
|--------|---------|---------|
| `gbm.py` | LightGBM (BCE pesata) | nativi |
| `logit.py` | Logistica Elastic Net | preprocessed |
| `rf.py` | Random Forest | preprocessed |
| `svm.py` | SVM lineare | preprocessed |

### em_like

**Iterativo EM-like** in R + LightGBM: ad ogni iterazione il modello riclassifica gli U e aggiorna il training set.

- `lgbSoft.R`: pesi frazionari (raw score del modello corrente come peso dell'esempio U).
- `lgbHard.R`: soglia adattiva — U con score < h_k diventano Reliable Negatives (RN).

---

## Modelli temporali

| Modello | Fase | Feature disponibili |
|---------|------|---------------------|
| M1 | Ex ante | Dati del bando alla pubblicazione |
| M2 | Durante | M1 + dati aggiudicazione |
| M3 | Ex post | M2 + dati esecuzione contratto |

---

## Configurazione comune

Ogni script ha una sezione **CONFIGURAZIONE** in testa con i parametri principali:

```python
MODEL_NUMBER = 1      # 1=M1, 2=M2, 3=M3
PNPU         = True   # True=PNPU, False=PU puro
TEST_MODE    = False  # True = smoke test rapido (1 fold, dataset ridotto)
```

I percorsi sono risolti dinamicamente tramite `Path(__file__).resolve().parents[N]` — nessun percorso assoluto hardcoded.

---

## Prerequisiti

**Python:**
```bash
pip install pandas numpy scikit-learn lightgbm torch pyarrow scipy
```

**R (solo em_like):**
```r
install.packages(c("arrow", "lightgbm", "caret", "dplyr", "pROC", "PRROC", "ggplot2"))
```

---

## Output

Ogni script salva i risultati in `<famiglia>/<sottocartella>/results/` come CSV con le metriche per fold (lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC).

I file seguono la convenzione `{modello}_M{n}_{variante}_fold_metrics.csv`.

---

## English version below

# models — PU Learning estimators for corruption detection in public procurement

Four model families implementing different strategies for the Positive-Unlabeled (PU) problem on Italian public procurement (appalti pubblici) data (ANAC).

Each family supports three temporal datasets (**M1/M2/M3**) and two loss variants (**PNPU** / **pure PU**), configurable via flags at the top of each script.

---

## Structure

```
models/
├── risk_estimators/         # nnPU/PNPU risk estimators (π-corrected loss)
│   ├── lgbm/nnpu_lgbm.py    #   LightGBM + custom objective
│   ├── logit/nnpu_logit.py  #   Custom logistic regression (gradient ascent)
│   └── nn/nnpu_nn.py        #   MLP PyTorch
├── bagging/                 # PU Bagging (Mordelet & Vert 2014)
│   ├── pu_bagging_lightgbm.py
│   └── pu_extratrees.py
├── biased/                  # Biased learning (unlabeled as down-weighted negatives)
│   ├── gbm.py
│   ├── logit.py
│   ├── rf.py
│   ├── svm.py
│   └── run_all.py           #   Parallel runner for all biased models
└── em_like/                 # Iterative EM-like (R + LightGBM)
    ├── lgbSoft.R            #   Soft: fractional weights from LightGBM scores
    └── lgbHard.R            #   Hard: adaptive Reliable Negatives
```

---

## Model families

### risk_estimators

Implement the **nnPU risk estimator** (Kiryo et al. 2017) with the **PNPU** extension that adds a corrective term for confirmed negatives (negativi certi).

- **nnPU loss**: `R = π·R_P(+) + max(0, R_U(-) - π·R_P(-))`
- **PNPU loss**: adds `(1-π)/n_U_tot · Σ_N BCE(f, -)`
- **Nested inner CV**: hyperparameter selection on inner validation (lift@1%), then final fit on the full training fold.
- Flag `RETRAIN_FINAL = True` to retrain on the full dataset with optimal parameters.

| Script | Base learner | Dataset |
|--------|-------------|---------|
| `nnpu_lgbm.py` | LightGBM (custom objective) | native (nativi) |
| `nnpu_logit.py` | Logistic regression (gradient ascent) | preprocessed |
| `nnpu_nn.py` | MLP PyTorch (1–2 hidden layers) | preprocessed |

### bagging

**PU Bagging** (Mordelet & Vert 2014): B bootstraps from U of size S; each base learner trains on P ∪ U_sample; final score = average probabilities.

- `PNPU=True`: confirmed negatives (negativi certi) fixed in every bootstrap (never sampled).
- `PNPU=False`: confirmed negatives in the U pool, sampled like unlabeled examples.

| Script | Base learner | Dataset |
|--------|-------------|---------|
| `pu_bagging_lightgbm.py` | LightGBM (standard BCE) | native |
| `pu_extratrees.py` | ExtraTreesClassifier | preprocessed |

### biased

**Biased learning**: unlabeled examples treated as negatives with reduced weight `W_UNLABELED`.

- `PNPU=True`: confirmed negatives weight 1, unlabeled weight `W_UNLABELED`.
- `PNPU=False`: both confirmed negatives and unlabeled at weight `W_UNLABELED`.

`run_all.py` launches all models × M1/M2/M3 × PNPU/PU in parallel.

| Script | Model | Dataset |
|--------|-------|---------|
| `gbm.py` | LightGBM (weighted BCE) | native |
| `logit.py` | Elastic Net logistic regression | preprocessed |
| `rf.py` | Random Forest | preprocessed |
| `svm.py` | Linear SVM | preprocessed |

### em_like

**Iterative EM-like** in R + LightGBM: at each iteration the current model reclassifies unlabeled examples and updates the training set.

- `lgbSoft.R`: fractional weights (pesi frazionari) — raw model score used as unlabeled weight.
- `lgbHard.R`: adaptive threshold (soglia adattiva) — unlabeled examples below h_k become Reliable Negatives (RN).

---

## Temporal models

| Model | Phase   | Available features                           |
|-------|---------|----------------------------------------------|
| M1    | Ex ante | Tender notice (bando) data at publication    |
| M2    | During  | M1 + award (aggiudicazione) data             |
| M3    | Ex post | M2 + contract execution data                 |

---

## Common configuration

Each script has a **CONFIGURAZIONE** section at the top with the main parameters:

```python
MODEL_NUMBER = 1      # 1=M1, 2=M2, 3=M3
PNPU         = True   # True=PNPU, False=pure PU
TEST_MODE    = False  # True = quick smoke test (1 fold, reduced dataset)
```

Paths are resolved dynamically via `Path(__file__).resolve().parents[N]` — no hardcoded absolute paths.

---

## Prerequisites

**Python:**
```bash
pip install pandas numpy scikit-learn lightgbm torch pyarrow scipy
```

**R (em_like only):**
```r
install.packages(c("arrow", "lightgbm", "caret", "dplyr", "pROC", "PRROC", "ggplot2"))
```

---

## Output

Each script saves results in `<family>/<subfolder>/results/` as CSV files with per-fold metrics (lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC).

Files follow the naming convention `{model}_M{n}_{variant}_fold_metrics.csv`.
