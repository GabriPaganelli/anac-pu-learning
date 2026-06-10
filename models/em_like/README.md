# em_like — Iterativo EM-like (R + LightGBM)
# *em_like — Iterative EM-like (R + LightGBM)*

---

## Italiano

Approccio iterativo ispirato all'algoritmo EM: ad ogni iterazione il modello LightGBM corrente riclassifica gli esempi Unlabeled e aggiorna il training set.

Due varianti implementate in R: **Soft** (pesi frazionari) e **Hard** (Reliable Negatives adattivi).

---

## Struttura

```
em_like/
├── lgbm_soft.R      # EM Soft: pesi frazionari da score del modello corrente
├── lgbm_hard.R      # EM Hard: Reliable Negatives con soglia adattiva
└── results/
    ├── soft/
    └── hard/
```

---

## Varianti

### lgbm_soft.R — Soft EM

Ogni U riceve come peso il raw score del modello corrente (interpretato come P(corrotto | X)). Il modello successivo allena su P ∪ U con pesi frazionari. Iterazione fino a convergenza o `MAX_ITER_EM`.

### lgbm_hard.R — Hard EM (Reliable Negatives)

Soglia adattiva `h_k`: gli U con score < `h_k` diventano **Reliable Negatives** (RN) e vengono aggiunti come negativi certi al training del ciclo successivo.

- `h_k` parte al 5° percentile e aumenta in base al margine di stabilità tra iterazioni.
- Criteri di stop: recall_P < `RECALL_P_MIN` oppure RN > 50% degli U nel training.

---

## Configurazione

```r
PI_HAT        <- 0.02        # stima della prior π
MODEL_NUMBER  <- 3           # 1, 2 o 3
DATA_SOURCE   <- "preprocessed"  # "nativi" | "preprocessed"
MAX_ITER_EM   <- 5
```

---

## Utilizzo

```bash
Rscript lgbm_soft.R
Rscript lgbm_hard.R
```

Modificare `MODEL_NUMBER` e `DATA_SOURCE` in CONFIG prima dell'esecuzione.

---

## Output

| File | Descrizione |
|------|-------------|
| `results/soft/em_soft_M{n}_{source}_fold_metrics.csv` | Metriche per fold (Soft) |
| `results/soft/em_soft_M{n}_{source}_results.rds` | Oggetto R completo (Soft) |
| `results/hard/em_hard_M{n}_{source}_fold_metrics.csv` | Metriche per fold (Hard) |
| `results/hard/em_hard_M{n}_{source}_results.rds` | Oggetto R completo (Hard) |

I file `.rds` (binari) sono versionati con DVC: `dvc pull` per scaricarli.

---

## Prerequisiti

```r
install.packages(c("arrow", "lightgbm", "caret", "dplyr", "pROC", "PRROC", "ggplot2"))
```

---

## English

An iterative approach inspired by the EM algorithm: at each iteration the current LightGBM model reclassifies unlabeled (Unlabeled) examples and updates the training set.

Two variants implemented in R: **Soft** (fractional weights / pesi frazionari) and **Hard** (adaptive Reliable Negatives).

---

## Structure

```
em_like/
├── lgbm_soft.R      # Soft EM: fractional weights from current model score
├── lgbm_hard.R      # Hard EM: Reliable Negatives with adaptive threshold
└── results/
    ├── soft/
    └── hard/
```

---

## Variants

### lgbm_soft.R — Soft EM

Each unlabeled example receives the raw score of the current model as its weight (interpreted as P(corrupt | X)). The next model trains on P ∪ U with fractional weights (pesi frazionari). Iteration continues until convergence or `MAX_ITER_EM`.

### lgbm_hard.R — Hard EM (Reliable Negatives)

Adaptive threshold `h_k`: unlabeled examples with score < `h_k` become **Reliable Negatives** (RN) and are added as confirmed negatives in the next training cycle.

- `h_k` starts at the 5th percentile and increases based on the stability margin between iterations.
- Stopping criteria: recall_P < `RECALL_P_MIN` or RN > 50% of unlabeled in training.

---

## Configuration

```r
PI_HAT        <- 0.02        # prior estimate π
MODEL_NUMBER  <- 3           # 1, 2, or 3
DATA_SOURCE   <- "preprocessed"  # "nativi" | "preprocessed"
MAX_ITER_EM   <- 5
```

Paths are resolved dynamically via `dirname(normalizePath(...))` relative to the script location.

---

## Usage

```bash
Rscript lgbm_soft.R
Rscript lgbm_hard.R
```

Modify `MODEL_NUMBER` and `DATA_SOURCE` in CONFIG before running.

---

## Output

| File | Description |
|------|-------------|
| `results/soft/em_soft_M{n}_{source}_fold_metrics.csv` | Per-fold metrics (Soft) |
| `results/soft/em_soft_M{n}_{source}_results.rds` | Full R object (Soft) |
| `results/hard/em_hard_M{n}_{source}_fold_metrics.csv` | Per-fold metrics (Hard) |
| `results/hard/em_hard_M{n}_{source}_results.rds` | Full R object (Hard) |

The binary `.rds` files are versioned with DVC: run `dvc pull` to download them.

---

## Prerequisites

```r
install.packages(c("arrow", "lightgbm", "caret", "dplyr", "pROC", "PRROC", "ggplot2"))
```
