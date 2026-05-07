# bagging — PU Bagging
# *bagging — PU Bagging*

---

## Italiano

Implementazione del **PU Bagging** (Mordelet & Vert 2014) con due base learner: LightGBM ed ExtraTrees.

Idea: B bootstrap da U di dimensione S, ogni base learner allena su P ∪ campione_U; lo score finale è la media delle probabilità sui B modelli. La correzione per π emerge implicitamente dal rapporto s/|U| nel campionamento.

---

## Struttura

```
bagging/
├── pu_bagging_lightgbm.py   # B bootstrap LightGBM + inner CV
├── pu_extratrees.py         # B bootstrap ExtraTreesClassifier
└── results/
```

---

## Varianti

| Flag | P | N certi | U |
|------|---|---------|---|
| `PNPU=True` | Sempre nel training | Fissi in ogni bootstrap (mai campionati) | B campionamenti casuali |
| `PNPU=False` | Sempre nel training | Nel pool U, campionati come gli U | B campionamenti casuali |

---

## Configurazione

```python
MODEL     = 1       # 1=M1, 2=M2, 3=M3
PNPU      = True    # True=PNPU, False=PU puro
TEST_MODE = False   # True = smoke test rapido

B         = 150     # numero di bootstrap
S         = 15_000  # dimensione campione U per bootstrap
```

**pu_bagging_lightgbm.py** include inner CV (grid num_leaves × N_ROUNDS, metrica lift@1%) per selezionare gli iperparametri del base learner prima del bagging.

---

## Utilizzo

```bash
python pu_bagging_lightgbm.py
python pu_extratrees.py
```

---

## Dataset

| Script | Dataset | Motivo |
|--------|---------|--------|
| `pu_bagging_lightgbm.py` | `nativi/` | LightGBM gestisce NA nativamente |
| `pu_extratrees.py` | `preprocessed/` | sklearn richiede input numerici senza NA |

---

## Output

`results/{modello}_M{n}_{pnpu|pu}_fold_metrics.csv` con lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC per fold.

---

## Prerequisiti

```bash
pip install lightgbm scikit-learn pandas numpy pyarrow
```

---

## English

Implementation of **PU Bagging** (Mordelet & Vert 2014) with two base learners: LightGBM and ExtraTrees.

Idea: B bootstraps from U of size S; each base learner trains on P ∪ U_sample; the final score is the average probability across the B models. The correction for π (prior) emerges implicitly from the ratio s/|U| in the sampling.

---

## Structure

```
bagging/
├── pu_bagging_lightgbm.py   # B LightGBM bootstraps + inner CV
├── pu_extratrees.py         # B ExtraTreesClassifier bootstraps
└── results/
```

---

## Variants

| Flag | P | Confirmed negatives (N certi) | Unlabeled (U) |
|------|---|-------------------------------|---------------|
| `PNPU=True`  | Always in training | Fixed in every bootstrap (never sampled) | B random samples |
| `PNPU=False` | Always in training | In U pool, sampled like unlabeled examples | B random samples |

---

## Configuration

```python
MODEL     = 1       # 1=M1, 2=M2, 3=M3
PNPU      = True    # True=PNPU, False=pure PU
TEST_MODE = False   # True = quick smoke test

B         = 150     # number of bootstraps
S         = 15_000  # unlabeled sample size per bootstrap
```

**pu_bagging_lightgbm.py** includes inner CV (grid num_leaves × N_ROUNDS, metric lift@1%) to select base learner hyperparameters before bagging.

---

## Usage

```bash
python pu_bagging_lightgbm.py
python pu_extratrees.py
```

---

## Dataset

| Script | Dataset | Reason |
|--------|---------|--------|
| `pu_bagging_lightgbm.py` | `nativi/` | LightGBM handles NA natively |
| `pu_extratrees.py` | `preprocessed/` | sklearn requires numeric input without NA |

---

## Output

`results/{model}_M{n}_{pnpu|pu}_fold_metrics.csv` with lift@1%, lift@2%, lift@5%, PR-AUC, ROC-AUC per fold.

---

## Prerequisites

```bash
pip install lightgbm scikit-learn pandas numpy pyarrow
```
