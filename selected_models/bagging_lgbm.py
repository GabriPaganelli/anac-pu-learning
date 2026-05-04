"""
Bagging LightGBM — finalista PNU Mixing.
Basato su PUET-Bagging/pu_bagging_lightgbm.py (PNPU, Roadmap §4).

Iperparametri fissi:
  num_leaves  = 15   (valore dominante in RE LGB stesso base learner;
                       inner CV non salvato per Bagging — scelta coerente)
  n_rounds    = 100  (valore centrale della griglia {50,100,200} di inner CV)
  lr          = 0.05 (fisso nell'originale)
  tutti gli altri LGBM_FIXED invariati dall'originale

γ (PNU Mixing):
  γ ∈ [0, 1]: sample_weight degli N certi in ogni bootstrap.
  P ha weight=1; U campionato ha weight=1.
  γ=0: N certi esclusi dal training (weight=0, non entrano nel vstack).
  γ=1: comportamento originale PNPU (N certi a peso pieno).

  Nota: γ=0 NON è equivalente alla variante PU originale (dove N certi
  andavano nel pool U come negativi campionabili). Qui a γ=0 i N certi
  sono semplicemente assenti — semantica più pulita per il mixing continuo.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.utils import check_random_state

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.pu_metrics import eval_all  # noqa: E402

NUM_LEAVES = 15    # dominante in RE LGB (stesso base learner); vedi docstring
N_ROUNDS   = 100   # valore centrale griglia inner CV {50, 100, 200}

LGBM_FIXED = dict(
    objective        = "binary",
    boosting_type    = "gbdt",
    learning_rate    = 0.05,
    min_data_in_leaf = 20,
    feature_fraction = 0.8,
    bagging_fraction = 0.5,
    bagging_freq     = 1,
    lambda_l1        = 0.0,
    lambda_l2        = 0.5,
    num_threads      = -1,
    verbosity        = -1,
)

B           = 150      # numero di bag
S           = 15_000   # dimensione bootstrap per bag
RANDOM_SEED = 42


def _make_dataset(X: np.ndarray, y: np.ndarray,
                  w: np.ndarray = None,
                  cat_idx: list = None,
                  reference: lgb.Dataset = None) -> lgb.Dataset:
    return lgb.Dataset(
        X, label=y,
        weight=w,
        categorical_feature=cat_idx or "auto",
        reference=reference,
        free_raw_data=False,
    )


def _one_bag(X_P: np.ndarray, X_N: np.ndarray, X_pool: np.ndarray,
             X_score: np.ndarray, gamma: float,
             params: dict, rng,
             cat_idx: list = None) -> np.ndarray:
    """Addestra un singolo bag e restituisce le probabilità su X_score."""
    s       = min(S, len(X_pool))
    replace = s == len(X_pool)
    idx     = rng.choice(len(X_pool), size=s, replace=replace)
    X_neg   = X_pool[idx]

    if gamma > 0 and len(X_N) > 0:
        X_tr = np.vstack([X_P, X_N, X_neg])
        y_tr = np.concatenate([np.ones(len(X_P)),
                               np.zeros(len(X_N)),
                               np.zeros(s)])
        w_tr = np.concatenate([np.ones(len(X_P)),
                               np.full(len(X_N), gamma),
                               np.ones(s)])
    else:
        # γ=0 oppure nessun N disponibile: addestramento su solo P + U campionato
        X_tr = np.vstack([X_P, X_neg])
        y_tr = np.concatenate([np.ones(len(X_P)), np.zeros(s)])
        w_tr = None  # pesi uniformi

    p         = dict(params)
    p["seed"] = int(rng.randint(0, 99999))
    model     = lgb.train(p, _make_dataset(X_tr, y_tr, w_tr, cat_idx),
                          num_boost_round=N_ROUNDS)
    return model.predict(X_score)


def fit_predict(X_P_tr: np.ndarray, X_N_tr: np.ndarray,
                X_U_tr: np.ndarray, X_te: np.ndarray,
                gamma: float, rng,
                cat_idx: list = None) -> np.ndarray:
    """Bagging PU con γ-weighted N certi. Restituisce score aggregati su X_te."""
    params = dict(LGBM_FIXED, num_leaves=NUM_LEAVES)
    # cat_idx va al Dataset, NON a params (LightGBM ignora categorical_feature in params)

    acc = np.zeros(len(X_te))
    for b in range(B):
        acc += _one_bag(X_P_tr, X_N_tr, X_U_tr, X_te,
                        gamma, params, rng, cat_idx)
        if (b + 1) % 50 == 0:
            print(f"    bag {b+1}/{B}")
    return acc / B


def run_oof(df: pd.DataFrame, gamma: float,
            features: list, cat_idx: list = None) -> tuple[np.ndarray, dict]:
    """4-fold OOF con Bagging LGB per il γ dato.

    Ritorna (score_oof_array, fold_metrics_dict).
    """
    rng      = check_random_state(RANDOM_SEED)
    label    = df["label"].values.astype(float)
    fold_col = df["fold"].values.astype(float)
    X        = df[features].to_numpy(dtype=np.float32)
    scores   = np.full(len(df), np.nan)
    fold_metrics = {}

    for fold_test in range(4):
        print(f"  [Bagging gamma={gamma:.2f}] fold {fold_test}")
        tr = fold_col != fold_test
        te = fold_col == fold_test

        X_P_tr = X[(label == 1) & tr]
        X_N_tr = X[(label == 0) & tr]
        X_U_tr = X[np.isnan(label) & tr]
        X_te   = X[te]
        label_te = label[te]

        sc_te = fit_predict(X_P_tr, X_N_tr, X_U_tr, X_te,
                            gamma, rng, cat_idx)
        scores[te] = sc_te

        sp = sc_te[label_te == 1]
        sn = sc_te[label_te == 0]
        su = sc_te[np.isnan(label_te)]
        m = eval_all(sp, sn, su)
        fold_metrics[fold_test] = {"fold": fold_test, "gamma": gamma, **m}
        print(f"    lift@1%={m['lift@1%']:.2f}x pr_auc={m['pr_auc']:.3f}")

    return scores, fold_metrics
