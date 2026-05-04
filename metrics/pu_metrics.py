"""
Modulo condiviso di metriche per PU learning.
Chiamato da tutti gli script della pipeline (punti 1-5 della roadmap).
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def lift_at_k(scores_pos: np.ndarray,
              scores_neg: np.ndarray,
              scores_unl: np.ndarray,
              k: float = 0.01) -> float:
    """
    Lift@k% su P vs (N + U).
    Tutti gli esempi vengono ordinati per score; la metrica misura
    quante volte più P cadono nel top-k% rispetto al caso casuale.

    Parameters
    ----------
    scores_pos : score degli esempi positivi certi nel fold di validazione
    scores_neg : score dei negativi certi nel fold di validazione
    scores_unl : score degli unlabeled nel fold di validazione
    k          : soglia percentuale (0.01 = top 1%)
    """
    n_pos  = len(scores_pos)
    all_s  = np.concatenate([scores_pos, scores_neg, scores_unl])
    n_top  = max(1, int(np.ceil(k * len(all_s))))
    thresh = np.partition(all_s, -n_top)[-n_top]   # k-esimo score più alto
    pos_in_top = (scores_pos >= thresh).sum()
    expected   = k * n_pos
    return float(pos_in_top / expected) if expected > 0 else np.nan


def pr_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """PR-AUC su soli esempi etichettati (P vs N)."""
    y = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    s = np.concatenate([scores_pos, scores_neg])
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def roc_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """ROC-AUC su soli esempi etichettati (P vs N)."""
    y = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    s = np.concatenate([scores_pos, scores_neg])
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def eval_all(scores_pos, scores_neg, scores_unl,
             ks=(0.01, 0.02, 0.05)) -> dict:
    """Calcola tutte le metriche in un colpo solo."""
    out = {}
    for k in ks:
        out[f"lift@{k*100:g}%"] = lift_at_k(scores_pos, scores_neg, scores_unl, k)
    out["pr_auc"]  = pr_auc(scores_pos, scores_neg)
    out["roc_auc"] = roc_auc(scores_pos, scores_neg)
    return out
