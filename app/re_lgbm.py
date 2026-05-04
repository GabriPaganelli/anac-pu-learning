"""
RE LightGBM (nnPU loss) — finalista PNU Mixing.
Versione production per app/ — run_oof rimosso (solo pipeline di training).

Iperparametri fissi:
  num_leaves       = 15
  learning_rate    = 0.02
  min_data_in_leaf = 20

π per dataset:
  M1, M2, M3: π=0.02  (uniforme per coerenza con stima PULSNAR)

γ (PNU Mixing):
  Loss PNPU:
    R_PNPU = π·R_P(+) + max(0, R_U(−) − π·R_P(−)) + γ·(1−π)/n_UL · Σ_{k∈N} BCE(f_k,−)
  γ=0 → nnPU puro  |  γ=1 → PNPU completo
"""

import numpy as np
import lightgbm as lgb
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score

PI_BY_MODEL = {1: 0.02, 2: 0.02, 3: 0.02}

NUM_LEAVES       = 15
LEARNING_RATE    = 0.02
MIN_DATA_IN_LEAF = 20

LGBM_FIXED = dict(
    objective               = "mse",
    boosting_type           = "gbdt",
    lambda_l1               = 0.0,
    lambda_l2               = 0.5,
    min_sum_hessian_in_leaf = 1e-5,
    num_threads             = -1,
    verbosity               = -1,
    seed                    = 42,
)

N_ROUNDS_MAX = 1000
EARLY_STOP   = 200
ES_FRAC      = 0.15
RANDOM_SEED  = 42



def make_pnpu_objective(pi: float, gamma: float,
                        n_P: int, n_U: int, n_unl_total: int):
    """Custom objective nnPU PNPU con coefficiente mixing γ sul termine N certi."""
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

        if is_N.any() and _gamma > 0.0:
            w_N = _gamma * (1.0 - _pi) / _n_UL
            grads[is_N] = w_N * sig[is_N]
            hess[is_N]  = w_N * sig[is_N] * (1.0 - sig[is_N])

        grads = grads * _n_total
        hess  = np.clip(hess * _n_total, 1e-6, None)
        return grads, hess

    return _obj


def _make_roc_eval():
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
                return_model: bool = False,
                num_threads: int = -1):
    """Addestra RE LGB PNPU con γ-scaled N term, restituisce (score_te, best_iter).

    num_threads: thread LightGBM per questa chiamata (-1 = tutti i core disponibili).
                 Impostare a 1 quando si eseguono più chiamate in parallelo per
                 evitare oversubscription dei core.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    n_P = len(X_P_tr); n_U = len(X_U_tr); n_N = len(X_N_tr)
    n_unl_total = n_U

    X_all_tr = np.vstack([X_P_tr, X_U_tr, X_N_tr])
    y_all_tr = np.concatenate([
        np.ones(n_P),
        np.zeros(n_U),
        np.full(n_N, 2.0),
    ])

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
                  objective=fobj,
                  num_threads=num_threads)

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
