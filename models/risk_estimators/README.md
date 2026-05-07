# risk_estimators — Stimatori nnPU / PNPU
# *risk_estimators — nnPU / PNPU risk estimators*

---

## Italiano

Implementazione del risk estimator **nnPU** (Kiryo et al. 2017) con estensione **PNPU** per il progetto di rilevazione corruzione negli appalti ANAC.

Tre base learner disponibili: LightGBM, regressione logistica custom e MLP PyTorch.

---

## Struttura

```
risk_estimators/
├── lgbm/
│   ├── nnpu_lgbm.py    # LightGBM con custom objective nnPU/PNPU
│   └── results/
├── logit/
│   ├── nnpu_logit.py   # Logistica con gradient ascent custom
│   └── results/
└── nn/
    ├── nnpu_nn.py      # MLP PyTorch
    └── results/
```

---

## Loss functions

**nnPU** (Kiryo 2017):
```
R_nnPU = π · R_P(+) + max(0, R_U(-) - π · R_P(-))
```

**PNPU** (estensione con negativi certi):
```
R_PNPU = π · R_P(+) + max(0, R_U(-) - π · R_P(-)) + (1-π)/n_U_tot · Σ_N BCE(f, -)
```

Il flag `PNPU = True/False` in CONFIG seleziona la variante.

---

## Configurazione

Sezione **CONFIGURAZIONE** in testa a ogni script:

```python
MODEL_NUMBER   = 1      # 1=M1, 2=M2, 3=M3
PNPU           = True   # True=PNPU, False=nnPU puro
TEST_MODE      = False  # True = smoke test rapido (1 fold, dati ridotti)
RETRAIN_FINAL  = False  # True = riallena su tutto il dataset con parametri ottimali
```

---

## Dataset

| Script | Dataset | Motivo |
|--------|---------|--------|
| `nnpu_lgbm.py` | `nativi/` | LightGBM gestisce NA e categoriali nativamente |
| `nnpu_logit.py` | `preprocessed/` | Gradient ascent richiede input numerici senza NA |
| `nnpu_nn.py` | `preprocessed/` | PyTorch richiede tensori numerici |

---

## Schema CV

```
Dataset
└── Outer fold k (4 fold OOF)
    ├── Validation: metriche OOF finali
    └── Training
        └── Inner CV (N_INNER_FOLDS fold)
            └── Selezione iperparametri (metrica: lift@1%)
```

---

## Utilizzo

```bash
python nnpu_lgbm.py   # LightGBM
python nnpu_logit.py  # Logistica
python nnpu_nn.py     # Neural network
```

Modificare `MODEL_NUMBER` e `PNPU` in CONFIG prima dell'esecuzione.

---

## Output

`results/{modello}_M{n}_{pnpu|pu}_fold_metrics.csv` con colonne per fold, iperparametri selezionati e metriche (lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC).

---

## Prerequisiti

```bash
# LightGBM
pip install lightgbm pandas numpy pyarrow

# Logistica
pip install pandas numpy scikit-learn pyarrow scipy

# NN
pip install torch pandas numpy scikit-learn pyarrow
```

---

## English

Implementation of the **nnPU risk estimator** (Kiryo et al. 2017) with the **PNPU** extension for the ANAC procurement corruption detection project.

Three base learners available: LightGBM, custom logistic regression, and MLP PyTorch.

---

## Structure

```
risk_estimators/
├── lgbm/
│   ├── nnpu_lgbm.py    # LightGBM with nnPU/PNPU custom objective
│   └── results/
├── logit/
│   ├── nnpu_logit.py   # Logistic regression with custom gradient ascent
│   └── results/
└── nn/
    ├── nnpu_nn.py      # MLP PyTorch
    └── results/
```

---

## Loss functions

**nnPU** (Kiryo 2017):
```
R_nnPU = π · R_P(+) + max(0, R_U(-) - π · R_P(-))
```

**PNPU** (extension with confirmed negatives / negativi certi):
```
R_PNPU = π · R_P(+) + max(0, R_U(-) - π · R_P(-)) + (1-π)/n_U_tot · Σ_N BCE(f, -)
```

The flag `PNPU = True/False` in CONFIG selects the variant.

---

## Configuration

**CONFIGURAZIONE** section at the top of each script:

```python
MODEL_NUMBER   = 1      # 1=M1, 2=M2, 3=M3
PNPU           = True   # True=PNPU, False=pure nnPU
TEST_MODE      = False  # True = quick smoke test (1 fold, reduced data)
RETRAIN_FINAL  = False  # True = retrain on full dataset with optimal parameters
```

---

## Dataset

| Script | Dataset | Reason |
|--------|---------|--------|
| `nnpu_lgbm.py` | `nativi/` | LightGBM handles NA and categoricals natively |
| `nnpu_logit.py` | `preprocessed/` | Gradient ascent requires numeric input without NA |
| `nnpu_nn.py` | `preprocessed/` | PyTorch requires numeric tensors |

Path resolved dynamically: `Path(__file__).resolve().parents[3] / "anac" / "output" / "parquet" / "model" / ...`

---

## CV scheme

```
Dataset
└── Outer fold k (4-fold OOF)
    ├── Validation: final OOF metrics
    └── Training
        └── Inner CV (N_INNER_FOLDS folds)
            └── Hyperparameter selection (metric: lift@1%)
```

---

## Usage

```bash
python nnpu_lgbm.py   # LightGBM
python nnpu_logit.py  # Logistic regression
python nnpu_nn.py     # Neural network
```

Modify `MODEL_NUMBER` and `PNPU` in CONFIG before running.

---

## Output

`results/{model}_M{n}_{pnpu|pu}_fold_metrics.csv` with columns for fold, selected hyperparameters, and metrics (lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC).

---

## Prerequisites

```bash
# LightGBM
pip install lightgbm pandas numpy pyarrow

# Logistic regression
pip install pandas numpy scikit-learn pyarrow scipy

# Neural network
pip install torch pandas numpy scikit-learn pyarrow
```
