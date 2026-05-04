"""
RE LightGBM (nnPU loss) — finalista PNU Mixing.
Basato su risk_estimators/lgbm/nnpu_lgbm_pnpu.py.

Iperparametri fissi:
  num_leaves       = 15   (unanime su M1; maggioranza su M2/M3; vedi riepilogo)
  learning_rate    = 0.02  (unanime su tutti i dataset)
  min_data_in_leaf = 20   (valore centrale della griglia {10,20,50};
                            non salvato nel CSV di output originale)
  n_rounds: ADATTIVO via early stopping (non è un HP architetturale)

π per dataset:
  M1, M2, M3: π=0.02  (uniforme per coerenza con stima PULSNAR e interpretabilità)
  Nota: π=0.03 dava +15% lift su M3 ma è probabile artefatto del gradiente nnPU
  (PULSNAR stima π piccolo per M3); vedi §3.0 riepilogo per confronto sensitivity.

γ (PNU Mixing):
  γ ∈ [0, 1]: scala il termine gradiente dei N certi nella loss nnPU.

  Loss PNPU (Kiryo 2017 + termine N certi):
    R_PNPU = π·R_P(+) + max(0, R_U(−) − π·R_P(−)) + (1−π)/n_UL · Σ_{k∈N} BCE(f_k,−)

  Con γ, il termine N certi diventa:
    γ · (1−π)/n_UL · Σ_{k∈N} BCE(f_k,−)

  γ=0 → termine N certi annullato = nnPU puro (PU puro senza N certi)
  γ=1 → PNPU completo (comportamento originale)

  L'implementazione modifica make_pnpu_objective() passando gamma
  come parametro che scala w_N = γ·(1−π)/n_UL.

  Questa è la semantica teoricamente corretta per il PNU mixing
  nel framework nnPU (confronta Sakai et al. 2017, Eq. 7).
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.pu_metrics import eval_all  # noqa: E402

PI_BY_MODEL = {1: 0.02, 2: 0.02, 3: 0.02}  # uniformato; 0.03 era artefatto gradiente

NUM_LEAVES       = 15
LEARNING_RATE    = 0.02
MIN_DATA_IN_LEAF = 20   # centrale della griglia {10,20,50}, non salvato in CSV

LGBM_FIXED = dict(
    objective               = "mse",   # sovrascritto dalla custom fobj
    boosting_type           = "gbdt",
    lambda_l1               = 0.0,
    lambda_l2               = 0.5,
    min_sum_hessian_in_leaf = 1e-5,
    num_threads             = -1,
    verbosity               = -1,
    seed                    = 42,
)

N_ROUNDS_MAX    = 1000
EARLY_STOP      = 200
ES_FRAC         = 0.15   # 15% del training come holdout per early stopping
RANDOM_SEED     = 42



def make_pnpu_objective(pi: float, gamma: float,
                        n_P: int, n_U: int, n_unl_total: int):
    """Custom objective nnPU PNPU con coefficiente mixing γ sul termine N certi.

    Gradienti per-campione (BCE version, hessiane sempre positive):

      P (label=1):
        g = (π/n_P) · (σ(f) − 1)
        h = (π/n_P) · σ(f)·(1−σ(f))

      U (label=0, unlabeled): attivi solo se neg_term > 0
        g = (1/n_U) · σ(f)
        h = (1/n_U) · σ(f)·(1−σ(f))

      N certi (label=2, sentinella): SEMPRE attivi, scalati per γ
        g = γ · (1−π)/n_UL · σ(f)
        h = γ · (1−π)/n_UL · σ(f)·(1−σ(f))

    γ=0 → w_N=0 → termine N certi annullato → nnPU puro
    γ=1 → comportamento originale PNPU

    Tutti i gradienti moltiplicati per n_total=(n_P+n_U) per portare
    le hessiane alla scala BCE standard (~0.25), necessario affinché
    i split gain siano positivi con lambda_l2=0.5.
    """
    _pi      = float(pi)
    _gamma   = float(gamma)
    _n_P     = float(max(n_P, 1))
    _n_U     = float(max(n_U, 1))
    _n_UL    = float(max(n_unl_total, 1))
    _n_total = float(max(n_P + n_U, 1))

    def _obj(preds, data):
        labels = data.get_label()
        is_P = labels == 1
        is_U = labels == 0
        is_N = labels == 2

        sig = 1.0 / (1.0 + np.exp(-preds))

        # Non-negative correction
        r_P_neg  = np.mean(np.log1p(np.exp(preds[is_P]))) if is_P.any() else 0.0
        r_U_neg  = np.mean(np.log1p(np.exp(preds[is_U]))) if is_U.any() else 0.0
        neg_term = r_U_neg - _pi * r_P_neg

        grads = np.zeros_like(preds)
        hess  = np.zeros_like(preds)

        if is_P.any():
            grads[is_P] = (_pi / _n_P) * (sig[is_P] - 1.0)
            hess[is_P]  = (_pi / _n_P) * sig[is_P] * (1.0 - sig[is_P])

        if is_U.any() and neg_term > 0.0:
            grads[is_U] = (1.0 / _n_U) * sig[is_U]
            hess[is_U]  = (1.0 / _n_U) * sig[is_U] * (1.0 - sig[is_U])

        # Termine N certi scalato per γ: w_N = γ·(1−π)/n_UL
        if is_N.any() and _gamma > 0.0:
            w_N = _gamma * (1.0 - _pi) / _n_UL
            grads[is_N] = w_N * sig[is_N]
            hess[is_N]  = w_N * sig[is_N] * (1.0 - sig[is_N])

        grads = grads * _n_total
        hess  = np.clip(hess * _n_total, 1e-6, None)
        return grads, hess

    return _obj


def _make_roc_eval():
    """feval per early stopping su holdout labeled (P vs tutto il resto)."""
    def _eval(preds, data):
        labels = data.get_label()
        y_true = (labels == 1).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return "roc_auc", 0.5, True
        return "roc_auc", float(roc_auc_score(y_true, preds)), True
    return _eval


def _make_dataset(X: np.ndarray, y: np.ndarray,
                  features: list, cat_idx: list,
                  reference=None) -> lgb.Dataset:
    return lgb.Dataset(
        X, label=y,
        feature_name=features,
        categorical_feature=cat_idx if cat_idx else "auto",
        reference=reference,
        free_raw_data=False,
    )


def fit_predict(X_P_tr: np.ndarray, X_N_tr: np.ndarray,
                X_U_tr: np.ndarray, X_te: np.ndarray,
                gamma: float, pi: float,
                features: list, cat_idx: list = None,
                rng=None,
                return_model: bool = False):
    """Addestra RE LGB PNPU con γ-scaled N term, restituisce (score_te, best_iter).

    Encoding label nel Dataset:
      1 = P (positivi certi)
      0 = U (unlabeled)
      2 = N certi (sentinella; gestita dalla custom fobj, non da obiettivi built-in)

    Early stopping su 15% holdout stratificato (P+N+U), ROC-AUC come metrica.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n_P = len(X_P_tr); n_U = len(X_U_tr); n_N = len(X_N_tr)
    n_unl_total = n_U   # denominatore termine N (= |U| nel fold di training)

    # Dataset PNPU: P (1) + U (0) + N (2)
    X_all_tr = np.vstack([X_P_tr, X_U_tr, X_N_tr])
    y_all_tr = np.concatenate([
        np.ones(n_P),
        np.zeros(n_U),
        np.full(n_N, 2.0),
    ])

    # Early stopping: 15% holdout
    n_es   = max(10, int(ES_FRAC * len(X_all_tr)))
    es_idx = rng.choice(len(X_all_tr), size=n_es, replace=False)
    tr_idx = np.setdiff1d(np.arange(len(X_all_tr)), es_idx)

    y_es_bin = np.where(y_all_tr[es_idx] == 2, 0.0, y_all_tr[es_idx])
    n_P_tr   = int((y_all_tr[tr_idx] == 1).sum())
    n_U_tr   = int((y_all_tr[tr_idx] == 0).sum())

    fobj  = make_pnpu_objective(pi, gamma, n_P_tr, n_U_tr, n_unl_total)
    feval = _make_roc_eval()

    params = dict(LGBM_FIXED,
                  num_leaves=NUM_LEAVES,
                  min_data_in_leaf=MIN_DATA_IN_LEAF,
                  learning_rate=LEARNING_RATE,
                  objective=fobj)

    cat = cat_idx if cat_idx else "auto"
    dtrain = _make_dataset(X_all_tr[tr_idx], y_all_tr[tr_idx], features, cat)
    dval   = _make_dataset(X_all_tr[es_idx], y_es_bin, features, cat, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=N_ROUNDS_MAX,
        valid_sets=[dval],
        valid_names=["es_val"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = model.best_iteration if model.best_iteration > 0 else N_ROUNDS_MAX
    scores_te = sigmoid(model.predict(X_te, raw_score=True))
    if return_model:
        return scores_te, best_iter, model
    return scores_te, best_iter


def run_oof(df: pd.DataFrame, gamma: float, model_number: int,
            features: list, cat_idx: list = None) -> tuple[np.ndarray, dict]:
    """4-fold OOF RE LGB con γ per il dataset indicato.

    π viene scelto da PI_BY_MODEL (0.02 per tutti i dataset).
    Ritorna (score_oof_array, fold_metrics_dict).
    """
    pi       = PI_BY_MODEL[model_number]
    rng      = np.random.default_rng(RANDOM_SEED)
    label    = df["label"].values.astype(float)
    fold_col = df["fold"].values.astype(float)
    X        = df[features].to_numpy(dtype=np.float32)
    scores   = np.full(len(df), np.nan)
    fold_metrics = {}

    for fold_test in range(4):
        print(f"  [RE LGB gamma={gamma:.2f} pi={pi}] fold {fold_test}")
        tr = fold_col != fold_test
        te = fold_col == fold_test

        X_P_tr = X[(label == 1) & tr]
        X_N_tr = X[(label == 0) & tr]
        X_U_tr = X[np.isnan(label) & tr]
        X_te   = X[te]
        label_te = label[te]

        sc_te, best_iter = fit_predict(
            X_P_tr, X_N_tr, X_U_tr, X_te,
            gamma=gamma, pi=pi,
            features=features, cat_idx=cat_idx,
            rng=np.random.default_rng(RANDOM_SEED + fold_test),
        )
        scores[te] = sc_te

        sp = sc_te[label_te == 1]
        sn = sc_te[label_te == 0]
        su = sc_te[np.isnan(label_te)]
        m = eval_all(sp, sn, su)
        fold_metrics[fold_test] = {"fold": fold_test, "gamma": gamma,
                                   "pi": pi, "best_iter": best_iter, **m}
        print(f"    lift@1%={m['lift@1%']:.2f}x pr_auc={m['pr_auc']:.3f} "
              f"best_iter={best_iter}")

    return scores, fold_metrics
