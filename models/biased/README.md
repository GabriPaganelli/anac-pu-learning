# biased — Biased Learning (baseline)

Baseline PU in cui gli esempi **Unlabeled** vengono trattati come negativi con peso ridotto `W_UNLABELED`.

Quattro classificatori implementati: GBM, regressione logistica, random forest e SVM.  
`run_all.py` esegue tutti i modelli in parallelo su tutte le combinazioni di dataset e variante.

---

## Struttura

```
biased/
├── gbm.py       # LightGBM (BCE pesata)
├── logit.py     # Logistica Elastic Net (sklearn)
├── rf.py        # Random Forest (sklearn)
├── svm.py       # SVM lineare (sklearn)
├── run_all.py   # Runner parallelo
└── results/
```

---

## Schema dei pesi

| Variante | P | N certi | U |
|----------|---|---------|---|
| `PNPU=True` | 1.0 | 1.0 | `W_UNLABELED` |
| `PNPU=False` | 1.0 | `W_UNLABELED` | `W_UNLABELED` |

Il valore di default `W_UNLABELED = 0.025` dà peso trascurabile agli U preservando l'utilità dei P.

---

## Configurazione

Sezione **CONFIGURAZIONE** in testa a ogni script:

```python
MODEL_NUMBER = 3       # 1=M1, 2=M2, 3=M3
PNPU         = True    # True=PNPU, False=PU puro
W_UNLABELED  = 0.025   # peso degli U (e dei N in PU puro)
```

---

## Utilizzo

Script singolo:
```bash
python gbm.py      # oppure logit.py, rf.py, svm.py
```

Tutti i job in parallelo:
```bash
python run_all.py
```

`run_all.py` lancia tutti i modelli × M1/M2/M3 × PNPU/PU come sottoprocessi paralleli e stampa un riepilogo a fine run.

---

## Dataset

| Script | Dataset | Motivo |
|--------|---------|--------|
| `gbm.py` | `nativi/` | LightGBM gestisce NA nativamente |
| `logit.py`, `rf.py`, `svm.py` | `preprocessed/` | sklearn richiede input numerici senza NA |

---

## Output

`results/{modello}_M{n}_{pnpu|pu}_fold_metrics.csv` con lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC per fold e iperparametro selezionato.

---

## Prerequisiti

```bash
pip install lightgbm scikit-learn pandas numpy pyarrow
```

---

## English version below

# biased — Biased Learning (baseline)

Baseline PU approach in which **Unlabeled** examples are treated as negatives with reduced weight `W_UNLABELED`.

Four classifiers implemented: GBM, logistic regression, random forest, and SVM.
`run_all.py` runs all models in parallel across all dataset and variant combinations.

---

## Structure

```
biased/
├── gbm.py       # LightGBM (weighted BCE)
├── logit.py     # Elastic Net logistic regression (sklearn)
├── rf.py        # Random Forest (sklearn)
├── svm.py       # Linear SVM (sklearn)
├── run_all.py   # Parallel runner
└── results/
```

---

## Weight scheme

| Variant | P | Confirmed negatives (N certi) | Unlabeled (U) |
|---------|---|-------------------------------|---------------|
| `PNPU=True`  | 1.0 | 1.0 | `W_UNLABELED` |
| `PNPU=False` | 1.0 | `W_UNLABELED` | `W_UNLABELED` |

The default `W_UNLABELED = 0.025` gives negligible weight to unlabeled examples while preserving the utility of positives.

---

## Configuration

**CONFIGURAZIONE** section at the top of each script:

```python
MODEL_NUMBER = 3       # 1=M1, 2=M2, 3=M3
PNPU         = True    # True=PNPU, False=pure PU
W_UNLABELED  = 0.025   # weight for unlabeled (and confirmed negatives in pure PU)
```

---

## Usage

Single script:
```bash
python gbm.py      # or logit.py, rf.py, svm.py
```

All jobs in parallel:
```bash
python run_all.py
```

`run_all.py` launches all models × M1/M2/M3 × PNPU/PU as parallel subprocesses and prints a summary at the end.

---

## Dataset

| Script | Dataset | Reason |
|--------|---------|--------|
| `gbm.py` | `nativi/` | LightGBM handles NA natively |
| `logit.py`, `rf.py`, `svm.py` | `preprocessed/` | sklearn requires numeric input without NA |

---

## Output

`results/{model}_M{n}_{pnpu|pu}_fold_metrics.csv` with lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC per fold and selected hyperparameter.

---

## Prerequisites

```bash
pip install lightgbm scikit-learn pandas numpy pyarrow
```
