"""
PUET (PU Extra Trees) — finalista PNU Mixing.
Basato su PUET-Bagging/pu_extratrees.py (PNPU, Roadmap §4).

Iperparametri fissi (identici all'originale — nessun inner CV in PUET):
  B                = 150    (bag)
  S                = 15_000 (bootstrap size per bag)
  NTREE_ET         = 50     (alberi per bag)
  MAX_FEATURES     = 1.0    (randomizzazione solo sulle soglie, Geurts 2006)
  MIN_SAMPLES_LEAF = 1

Usa il parquet preprocessed (no NA, categoriali label-encoded)
perché ExtraTreesClassifier (sklearn) non gestisce NA nativamente.

γ (PNU Mixing):
  γ ∈ [0, 1]: sample_weight degli N certi in ogni bootstrap.
  P ha weight=1; U campionato ha weight=1.
  γ=0: N certi assenti dal training (weight=0 → non inclusi nel vstack).
  γ=1: comportamento originale PNPU (N certi a peso pieno).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import check_random_state

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.pu_metrics import eval_all  # noqa: E402

B                = 150
S                = 15_000
NTREE_ET         = 50
MAX_FEATURES     = 1.0
MIN_SAMPLES_LEAF = 1
RANDOM_SEED      = 42


def fit_predict(X_P_tr: np.ndarray, X_N_tr: np.ndarray,
                X_U_tr: np.ndarray, X_te: np.ndarray,
                gamma: float, rng) -> np.ndarray:
    """Bagging Extra Trees con γ-weighted N certi. Restituisce score su X_te."""
    X_pool  = X_U_tr
    s_fold  = min(S, len(X_pool))
    replace = s_fold == len(X_pool)
    if replace:
        print(f"    Nota: pool ({len(X_pool)}) < S ({S}), campionamento con rimpiazzo")

    acc = np.zeros(len(X_te))

    for b in range(B):
        idx   = rng.choice(len(X_pool), size=s_fold, replace=replace)
        X_neg = X_pool[idx]

        if gamma > 0 and len(X_N_tr) > 0:
            X_tr = np.vstack([X_P_tr, X_N_tr, X_neg])
            y_tr = np.concatenate([
                np.ones(len(X_P_tr)),
                np.zeros(len(X_N_tr)),
                np.zeros(s_fold),
            ])
            w_tr = np.concatenate([
                np.ones(len(X_P_tr)),
                np.full(len(X_N_tr), gamma),
                np.ones(s_fold),
            ])
        else:
            X_tr = np.vstack([X_P_tr, X_neg])
            y_tr = np.concatenate([np.ones(len(X_P_tr)), np.zeros(s_fold)])
            w_tr = None

        et = ExtraTreesClassifier(
            n_estimators     = NTREE_ET,
            max_features     = MAX_FEATURES,
            min_samples_leaf = MIN_SAMPLES_LEAF,
            bootstrap        = False,
            n_jobs           = -1,
            random_state     = rng.randint(0, 99999),
        )
        et.fit(X_tr, y_tr, sample_weight=w_tr)
        acc += et.predict_proba(X_te)[:, 1]

        if (b + 1) % 50 == 0:
            print(f"    bag {b+1}/{B}")

    return acc / B


def run_oof(df: pd.DataFrame, gamma: float,
            features: list) -> tuple[np.ndarray, dict]:
    """4-fold OOF PUET con γ per il dataset indicato.

    df deve essere il parquet preprocessed (no NA) con fold già reshuffato.
    Ritorna (score_oof_array, fold_metrics_dict).
    """
    rng      = check_random_state(RANDOM_SEED)
    label    = df["label"].values.astype(float)
    fold_col = df["fold"].values.astype(float)
    X        = df[features].to_numpy(dtype=np.float32)
    scores   = np.full(len(df), np.nan)
    fold_metrics = {}

    for fold_test in range(4):
        print(f"  [PUET gamma={gamma:.2f}] fold {fold_test}")
        tr = fold_col != fold_test
        te = fold_col == fold_test

        X_P_tr = X[(label == 1) & tr]
        X_N_tr = X[(label == 0) & tr]
        X_U_tr = X[np.isnan(label) & tr]
        X_te   = X[te]
        label_te = label[te]

        sc_te = fit_predict(X_P_tr, X_N_tr, X_U_tr, X_te, gamma, rng)
        scores[te] = sc_te

        sp = sc_te[label_te == 1]
        sn = sc_te[label_te == 0]
        su = sc_te[np.isnan(label_te)]
        m = eval_all(sp, sn, su)
        fold_metrics[fold_test] = {"fold": fold_test, "gamma": gamma, **m}
        print(f"    lift@1%={m['lift@1%']:.2f}x pr_auc={m['pr_auc']:.3f}")

    return scores, fold_metrics
