"""
LightGBM supervisionato P vs N certi — finalista PNU Mixing.
Riscrittura Python di em_like/lgbHard.v4.R (iterazione 0).

In EM-Hard, best_iter=0 in quasi tutti i fold su tutti i dataset:
il modello coincide con un LightGBM standard addestrato su P vs N certi.
Questa versione implementa solo iter=0, senza la pipeline EM iterativa.

Parametri fissi (idem R):
  feature_fraction = 0.7
  bagging_fraction = 0.7
  bagging_freq     = 5
  lambda_l2        = 0.5
  objective        = binary   (BCE standard)
  early stop su AUC su 15% holdout dei labeled di training

Tuning globale (idem R):
  Eseguito UNA VOLTA per dataset su tutti i P+N labeled (non per fold).
  Giustificazione: con ~2600 obs e 27 combo, il leakage del fold di test
  è trascurabile rispetto alla granularità della griglia. 4× più veloce
  del per-fold tuning senza differenza misurabile (vedi §3.5 riepilogo).

γ (PNU Mixing):
  γ ∈ [0.1, 1.0]: sample_weight degli N certi nel training set.
  P ha sempre weight=1. U non entra in training (modello supervisionato).
  γ=0 escluso: senza N certi il classificatore binario non ha negativi
  e la loss BCE collassa (tutti i label=1).
  γ=1: comportamento originale EM-Hard (N certi a peso pieno).
"""

import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.pu_metrics import eval_all  # noqa: E402

LGBM_FIXED = dict(
    objective        = "binary",
    metric           = "auc",       # early stopping su ROC-AUC
    boosting_type    = "gbdt",
    feature_fraction = 0.7,
    bagging_fraction = 0.7,
    bagging_freq     = 5,
    lambda_l1        = 0.0,
    lambda_l2        = 0.5,
    num_threads      = -1,
    verbosity        = -1,
    seed             = 17,          # stesso seed del R originale
)

# Griglia tuning (idem R)
LEAVES_GRID    = [15, 31, 63]
MIN_DATA_GRID  = [10, 20, 50]
LR_GRID        = [0.01, 0.05, 0.1]

N_ROUNDS_MAX   = 500
EARLY_STOP     = 30
ES_FRAC        = 0.15   # frazione del training labeled per early stopping
N_TUNE_FOLDS   = 5      # fold per il tuning globale (idem R)
RANDOM_SEED    = 17


def _build_params(combo: dict) -> dict:
    p = dict(LGBM_FIXED)
    p.update(combo)
    return p


def global_tune(X_P_all: np.ndarray, X_N_all: np.ndarray,
                cat_idx: list = None) -> dict:
    """Tuning globale su tutti i P+N labeled via N_TUNE_FOLDS-fold CV interna.

    Metrica: PR AUC medio sui fold interni (idem R originale).
    Ritorna dict con iperparametri ottimali usati per tutti i γ e tutti i fold.

    Nota: usa tutti i P+N inclusi quelli che saranno fold di test nell'OOF esterno.
    Il leakage è documentato e giustificato (vedi §3.5 riepilogo e lgbHard.v4.R).

    cat_idx: lista di indici 0-based delle colonne categoriali nella feature matrix.
    """
    X_all = np.vstack([X_P_all, X_N_all])
    y_all = np.concatenate([np.ones(len(X_P_all)), np.zeros(len(X_N_all))])

    combos = list(itertools.product(LEAVES_GRID, MIN_DATA_GRID, LR_GRID))
    cat_spec = cat_idx if cat_idx else "auto"
    print(f"  [Tuning globale] {len(combos)} combo × {N_TUNE_FOLDS} fold "
          f"su {len(X_all)} obs labeled")

    rng = np.random.default_rng(RANDOM_SEED)
    idx_p = np.where(y_all == 1)[0]; rng.shuffle(idx_p)
    idx_n = np.where(y_all == 0)[0]; rng.shuffle(idx_n)
    folds_p = idx_p % N_TUNE_FOLDS
    folds_n = idx_n % N_TUNE_FOLDS

    combo_scores = {c: [] for c in combos}

    for kf in range(N_TUNE_FOLDS):
        va_p = idx_p[folds_p == kf];  tr_p = idx_p[folds_p != kf]
        va_n = idx_n[folds_n == kf];  tr_n = idx_n[folds_n != kf]
        tr_idx = np.concatenate([tr_p, tr_n])
        va_idx = np.concatenate([va_p, va_n])

        X_tr_k = X_all[tr_idx]; y_tr_k = y_all[tr_idx]
        X_va_k = X_all[va_idx]; y_va_k = y_all[va_idx]

        dtrain_k = lgb.Dataset(X_tr_k, label=y_tr_k,
                                categorical_feature=cat_spec,
                                free_raw_data=False)
        dval_k   = lgb.Dataset(X_va_k, label=y_va_k, reference=dtrain_k)

        for combo in combos:
            nl, md, lr = combo
            params = _build_params({"num_leaves": nl,
                                    "min_data_in_leaf": md,
                                    "learning_rate": lr})
            model_k = lgb.train(
                params, dtrain_k, N_ROUNDS_MAX,
                valid_sets=[dval_k],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            scores_va = model_k.predict(X_va_k)
            if len(np.unique(y_va_k)) < 2:
                continue
            pr = average_precision_score(y_va_k, scores_va)
            combo_scores[combo].append(pr)

    mean_scores = {c: np.mean(v) if v else -np.inf for c, v in combo_scores.items()}
    best_combo  = max(mean_scores, key=mean_scores.get)
    nl, md, lr  = best_combo
    print(f"  [Tuning globale] Ottimali: leaves={nl} min_leaf={md} lr={lr:.2f} "
          f"PR_AUC={mean_scores[best_combo]:.4f}")
    return {"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": lr}


def fit_predict(X_P_tr: np.ndarray, X_N_tr: np.ndarray,
                X_te: np.ndarray, gamma: float,
                best_params: dict,
                cat_idx: list = None) -> np.ndarray:
    """Addestra LGB su P + γ·N certi, restituisce score su X_te.

    X_U non entra in training: modello puramente supervisionato.
    γ controlla il contributo degli N certi come sample_weight.
    Early stopping su 15% holdout stratificato del training labeled.
    """
    assert gamma > 0, "γ=0 non valido per LGB supervisionato (nessun negativo)"

    X_tr = np.vstack([X_P_tr, X_N_tr])
    y_tr = np.concatenate([np.ones(len(X_P_tr)), np.zeros(len(X_N_tr))])
    w_tr = np.concatenate([np.ones(len(X_P_tr)),
                           np.full(len(X_N_tr), gamma)])

    # Early stopping: 15% holdout stratificato
    rng  = np.random.default_rng(RANDOM_SEED)
    n_es = max(10, int(ES_FRAC * len(X_tr)))
    # Stratificato: mantieni proporzioni P/N nell'ES set
    idx_p = np.where(y_tr == 1)[0]; idx_n = np.where(y_tr == 0)[0]
    n_es_p = max(1, int(ES_FRAC * len(idx_p)))
    n_es_n = max(1, int(ES_FRAC * len(idx_n)))
    es_p   = rng.choice(idx_p, size=min(n_es_p, len(idx_p)), replace=False)
    es_n   = rng.choice(idx_n, size=min(n_es_n, len(idx_n)), replace=False)
    es_idx = np.concatenate([es_p, es_n])
    tr_idx = np.setdiff1d(np.arange(len(X_tr)), es_idx)

    cat = cat_idx or "auto"

    dtrain = lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx],
                         weight=w_tr[tr_idx],
                         categorical_feature=cat, free_raw_data=False)
    dval   = lgb.Dataset(X_tr[es_idx], label=y_tr[es_idx],
                         reference=dtrain)

    params = _build_params(best_params)
    model  = lgb.train(
        params, dtrain, N_ROUNDS_MAX,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    return model.predict(X_te)


def run_oof(df: pd.DataFrame, gamma: float,
            best_params: dict, features: list,
            cat_idx: list = None) -> tuple[np.ndarray, dict]:
    """Esegue 4-fold OOF e restituisce (score_oof, fold_metrics).

    best_params: dal tuning globale (chiamare global_tune() prima del loop γ).
    cat_idx: indici di colonne categoriali nella feature matrix (lista di int).
    """
    label    = df["label"].values.astype(float)
    fold_col = df["fold"].values.astype(float)
    X        = df[features].to_numpy(dtype=np.float32)
    scores   = np.full(len(df), np.nan)
    fold_metrics = {}

    for fold_test in range(4):
        tr = fold_col != fold_test
        te = fold_col == fold_test

        X_P_tr = X[(label == 1) & tr]
        X_N_tr = X[(label == 0) & tr]
        X_te   = X[te]
        label_te = label[te]

        sc_te = fit_predict(X_P_tr, X_N_tr, X_te, gamma, best_params, cat_idx)
        scores[te] = sc_te

        sp = sc_te[label_te == 1]
        sn = sc_te[label_te == 0]
        su = sc_te[np.isnan(label_te)]
        m = eval_all(sp, sn, su)
        fold_metrics[fold_test] = {"fold": fold_test, "gamma": gamma, **m}

    return scores, fold_metrics
