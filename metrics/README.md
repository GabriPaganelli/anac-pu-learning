# metrics — Modulo condiviso di metriche e preprocessing

Funzioni condivise tra tutti gli script di modellazione PU del progetto.  
Importato da ogni script tramite `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))`.

---

## Struttura

```
metrics/
├── pu_metrics.py      # Lift@k%, PR-AUC, ROC-AUC per framework PU
└── preprocessing.py   # One-hot encoding colonne categoriali
```

---

## pu_metrics.py

| Funzione | Descrizione |
|----------|-------------|
| `lift_at_k(scores_pos, scores_neg, scores_unl, k)` | Lift@k% su P vs (N + U) |
| `pr_auc(scores_pos, scores_neg)` | PR-AUC su soli esempi etichettati (P vs N) |
| `roc_auc(scores_pos, scores_neg)` | ROC-AUC su soli esempi etichettati (P vs N) |
| `eval_all(scores_pos, scores_neg, scores_unl, ks)` | Calcola tutte le metriche in un colpo |

**Lift@k%** è la metrica primaria del progetto: misura quante volte più positivi cadono nel top-k% rispetto al caso casuale.  
I punteggi sono separati in tre array (P, N, U) perché il denominatore del lift usa l'intero pool (P + N + U), mentre numeratore e ROC/PR usano solo gli etichettati.

## preprocessing.py

| Funzione | Descrizione |
|----------|-------------|
| `encode_categoricals(df, extra_drop, drop_first, verbose)` | One-hot encoding colonne object/category |

Le colonne amministrative (`cig`, `esito`, `anno_pubblicazione`, `regione`, `label`, `fold`) sono sempre escluse dall'encoding. L'encoding viene applicato sull'intero dataset prima del CV split (livelli fissi per costruzione — nessun data leakage).

---

## Utilizzo

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N = livelli verso Appalti/

from metrics.pu_metrics import lift_at_k, eval_all
from metrics.preprocessing import encode_categoricals
```

---

## Prerequisiti

```bash
pip install numpy scikit-learn pandas
```

---

## English version below

# metrics — Shared metrics and preprocessing module

Shared functions used across all PU modelling scripts in the project.
Imported by each script via `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))`.

---

## Structure

```
metrics/
├── pu_metrics.py      # Lift@k%, PR-AUC, ROC-AUC for the PU framework
└── preprocessing.py   # One-hot encoding of categorical columns
```

---

## pu_metrics.py

| Function | Description |
|----------|-------------|
| `lift_at_k(scores_pos, scores_neg, scores_unl, k)` | Lift@k% on P vs (N + U) |
| `pr_auc(scores_pos, scores_neg)` | PR-AUC on labeled examples only (P vs N) |
| `roc_auc(scores_pos, scores_neg)` | ROC-AUC on labeled examples only (P vs N) |
| `eval_all(scores_pos, scores_neg, scores_unl, ks)` | Computes all metrics at once |

**Lift@k%** is the project's primary metric: it measures how many more positives fall in the top-k% compared to random chance.
Scores are split into three arrays (P, N, U) because the lift denominator uses the full pool (P + N + U), while the numerator and ROC/PR use only labeled examples.

## preprocessing.py

| Function | Description |
|----------|-------------|
| `encode_categoricals(df, extra_drop, drop_first, verbose)` | One-hot encoding of object/category columns |

Administrative columns (`cig`, `esito`, `anno_pubblicazione`, `regione`, `label`, `fold`) are always excluded from encoding. Encoding is applied on the full dataset before the CV split (fixed levels by construction — no data leakage).

---

## Usage

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N = levels up to Appalti/

from metrics.pu_metrics import lift_at_k, eval_all
from metrics.preprocessing import encode_categoricals
```

---

## Prerequisites

```bash
pip install numpy scikit-learn pandas
```
