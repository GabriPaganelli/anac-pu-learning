"""
nnPU Risk Estimator + LightGBM
PNPU=True : loss estesa con termine N certi (label=2, peso (1-pi)/n_unl_total).
PNPU=False: loss nnPU pura (Kiryo 2017); N certi esclusi dal training, usati solo in val.

Custom objective BCE su LightGBM (gradients/hessians per-campione).
Categoriali: gestione nativa LightGBM da file nativi/.
"""

MODEL_NUMBER   = 3          # 1=M1, 2=M2, 3=M3
PNPU           = True       # True = PNPU (N certi nel training); False = nnPU puro
TEST_MODE      = False      # True → smoke test rapido
RETRAIN_FINAL  = False      # True → riallena su tutti i dati con (pi*, params*)
PI_FINAL       = None
PARAMS_FINAL   = None       # dict {num_leaves, min_data_in_leaf, learning_rate}

from pathlib import Path as _P
BASE_PATH = str(_P(__file__).resolve().parents[3] / "anac" / "output" / "parquet" / "model" / "nativi")
OUT_DIR   = str(_P(__file__).resolve().parent / "results")

LABEL_COL    = "label"
FOLD_COL     = "fold"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]

N_OUTER_FOLDS  = 4
N_INNER_FOLDS  = 3

PI_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]

K_GRID = [0.01, 0.02, 0.03, 0.05]

LEAVES_GRID   = [15, 31, 63]      if not TEST_MODE else [15]
MIN_DATA_GRID = [10, 20, 50]      if not TEST_MODE else [10]
LR_GRID       = [0.02, 0.05, 0.1] if not TEST_MODE else [0.05]

LGBM_FIXED = dict(
    objective               = "regression",  # sovrascritto da fobj nel training
    boosting_type           = "gbdt",
    lambda_l1               = 0,
    lambda_l2               = 0.5,
    min_sum_hessian_in_leaf = 1e-5,
    num_threads             = -1,
    verbosity               = -1,
    seed                    = 42,
)

N_INNER_ROUNDS    = 300   if not TEST_MODE else 30
N_ROUNDS_MAX      = 1000  if not TEST_MODE else 50
EARLY_STOP_ROUNDS = 200   if not TEST_MODE else 5
ES_FRAC           = 0.15

RANDOM_SEED = 42
U_PER_FOLD  = 200       # solo TEST_MODE


import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from scipy.special import expit as sigmoid

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from metrics.pu_metrics import eval_all, lift_at_k

_CAT_FEATURES: list = []


def make_pnpu_objective(pi: float, n_P: int, n_U: int, n_unl_total: int):
    """
    PNPU: R_PNPU = pi*R_P(+) + max(0, R_U(-) - pi*R_P(-)) + (1-pi)/n_unl_total * sum_{k in N} BCE(f_k, -)

    BCE invece di margin loss perché d2l/df2 della margin loss può essere negativo,
    rendendo LightGBM instabile. BCE ha hessiana sigma*(1-sigma) > 0 sempre.

    Gradiente P formulato per-campione (pi/n_P * (sigma-1)) anziché collassato
    a costante (-pi/n_P): la versione collassata elimina il segnale per-campione
    e impedisce a LightGBM di apprendere quali P sono mal classificati.

    Scala n_total: moltiplica grads/hess per (n_P+n_U) così le hessiane
    per-campione risultano ~0.25 (scala BCE standard); i split gain rimangono
    positivi con lambda_l2=0.5. L'argmin non cambia.

    Encoding label: 1=P, 0=U, 2=N certi (sentinella; non usato da built-in).
    """
    _pi      = float(pi)
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

        r_P_neg = np.mean(np.log1p(np.exp(preds[is_P]))) if is_P.any() else 0.0
        r_U_neg = np.mean(np.log1p(np.exp(preds[is_U]))) if is_U.any() else 0.0
        neg_term = r_U_neg - _pi * r_P_neg

        grads = np.zeros_like(preds)
        hess  = np.zeros_like(preds)

        if is_P.any():
            grads[is_P] = (_pi / _n_P) * (sig[is_P] - 1.0)
            hess[is_P]  = (_pi / _n_P) * sig[is_P] * (1.0 - sig[is_P])

        if is_U.any() and neg_term > 0.0:
            grads[is_U] = (1.0 / _n_U) * sig[is_U]
            hess[is_U]  = (1.0 / _n_U) * sig[is_U] * (1.0 - sig[is_U])

        if is_N.any():
            w_N = (1.0 - _pi) / _n_UL
            grads[is_N] = w_N * sig[is_N]
            hess[is_N]  = w_N * sig[is_N] * (1.0 - sig[is_N])

        grads = grads * _n_total
        hess  = np.clip(hess * _n_total, 1e-6, None)
        return grads, hess

    return _obj


def make_nnpu_objective(pi: float, n_P: int, n_U: int):
    """
    nnPU puro (Kiryo 2017): R_nnPU = pi*R_P(+) + max(0, R_U(-) - pi*R_P(-))

    Gradiente P per-campione (non collassato) per lo stesso motivo di PNPU.
    Termine U contribuisce solo quando neg_term >= 0 (non-negative correction).
    Scala n_total per avere hessiane ~0.25 (split gain positivi con lambda_l2=0.5).
    """
    _pi      = float(pi)
    _n_P     = float(max(n_P, 1))
    _n_U     = float(max(n_U, 1))
    _n_total = float(max(n_P + n_U, 1))

    def _obj(preds: np.ndarray, data: lgb.Dataset):
        labels = data.get_label()
        is_P   = labels == 1
        is_U   = labels == 0
        sig    = sigmoid(preds)

        r_U_neg  = np.mean(np.log1p(np.exp(preds[is_U]))) if is_U.any() else 0.0
        r_P_neg  = np.mean(np.log1p(np.exp(preds[is_P]))) if is_P.any() else 0.0
        neg_term = r_U_neg - _pi * r_P_neg

        grads = np.zeros_like(preds)
        hess  = np.zeros_like(preds)

        grads[is_P] = (_pi / _n_P) * (sig[is_P] - 1.0)
        hess[is_P]  = (_pi / _n_P) * sig[is_P] * (1.0 - sig[is_P])

        if neg_term >= 0.0:
            grads[is_U] = (1.0 / _n_U) * sig[is_U]
            hess[is_U]  = (1.0 / _n_U) * sig[is_U] * (1.0 - sig[is_U])

        grads = grads * _n_total
        hess  = np.clip(hess * _n_total, 1e-6, None)
        return grads, hess

    return _obj


def make_roc_eval():
    # P (label==1) vs tutto il resto; label=2 → non-positivo per ROC.
    def _eval(preds: np.ndarray, data: lgb.Dataset):
        labels = data.get_label()
        y_true = (labels == 1).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return "roc_auc", 0.5, True
        return "roc_auc", float(roc_auc_score(y_true, preds)), True
    return _eval


def load_data() -> pd.DataFrame:
    """
    Legge il parquet nativi filtrando righe con fold assegnato.
    Converte categoriali (object/category) in integer codes; NA → NaN (gestito da LightGBM).
    """
    global _CAT_FEATURES
    path = Path(BASE_PATH) / f"M{MODEL_NUMBER}.parquet"
    print(f"  Caricamento {path.name} [nativi]...")
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    _schema = pq.read_schema(str(path))
    if FOLD_COL in _schema.names:
        _table = pq.read_table(str(path), filters=pc.field(FOLD_COL).is_valid())
        df = _table.to_pandas()
        print(f"  {len(df):,} righe con fold assegnato (U non campionati esclusi)")
    else:
        df = pd.read_parquet(path)
        print(f"  {df.shape[0]:,} righe (colonna fold assente)")

    _drop = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    cat_cols = [c for c in df.columns
                if c not in _drop and df[c].dtype.kind not in "iufb"]
    _CAT_FEATURES = cat_cols
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan).astype("float32")
    print(f"  {len(cat_cols)} colonne categoriali convertite a integer codes: {cat_cols}")

    if FOLD_COL not in df.columns:
        if TEST_MODE:
            print(f"  [WARN] Colonna '{FOLD_COL}' assente -- creo fold casuali (TEST_MODE).")
            df = _make_temp_folds(df)
        else:
            raise ValueError(f"Colonna '{FOLD_COL}' mancante.")
    return df


def _make_temp_folds(df: pd.DataFrame) -> pd.DataFrame:
    rng   = np.random.default_rng(RANDOM_SEED)
    label = df[LABEL_COL]
    is_p  = label.eq(1).fillna(False).values
    is_n  = label.eq(0).fillna(False).values
    is_u  = label.isna().values
    fold  = np.full(len(df), -1, dtype=int)
    for mask in [is_p, is_n]:
        idx = np.where(mask)[0]; rng.shuffle(idx)
        for j, i in enumerate(idx): fold[i] = j % N_OUTER_FOLDS
    u_idx  = np.where(is_u)[0]
    n_u    = min(len(u_idx), U_PER_FOLD * N_OUTER_FOLDS)
    chosen = rng.choice(u_idx, size=n_u, replace=False); rng.shuffle(chosen)
    for j, i in enumerate(chosen): fold[i] = j % N_OUTER_FOLDS
    df = df.copy(); df[FOLD_COL] = fold
    return df


def get_features(df: pd.DataFrame) -> list:
    # Cat codes (int16, kind='i') inclusi automaticamente.
    drop = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "iufb"]


def split_fold(df: pd.DataFrame, features: list, outer_k: int) -> dict:
    label   = df[LABEL_COL]
    fold_np = df[FOLD_COL].to_numpy(dtype=float, na_value=np.nan)

    is_p = label.eq(1).fillna(False).to_numpy(dtype=bool)
    is_n = label.eq(0).fillna(False).to_numpy(dtype=bool)
    is_u = label.isna().to_numpy(dtype=bool)

    tr = (fold_np != outer_k) & ~np.isnan(fold_np)
    va = (fold_np == outer_k)

    X = df[features].to_numpy(dtype=np.float32)

    return dict(
        X_pos_tr=X[is_p & tr], X_neg_tr=X[is_n & tr], X_unl_tr=X[is_u & tr],
        X_pos_va=X[is_p & va], X_neg_va=X[is_n & va], X_unl_va=X[is_u & va],
    )


def _build_params(combo: dict) -> dict:
    p = dict(LGBM_FIXED)
    p.update(combo)
    return p


def _make_dataset(X: np.ndarray, y: np.ndarray,
                  features: list, cat_features: list,
                  reference: Optional[lgb.Dataset] = None) -> lgb.Dataset:
    cat_idx = [features.index(f) for f in cat_features if f in features]
    return lgb.Dataset(
        X, label=y,
        feature_name=features,
        categorical_feature=cat_idx if cat_idx else "auto",
        reference=reference,
        free_raw_data=False,
    )


def train_lgbm(X_tr: np.ndarray, y_tr: np.ndarray,
               X_es: np.ndarray, y_es: np.ndarray,
               pi: float, n_P: int, n_U: int, n_unl_total: int,
               params: dict, features: list, cat_features: list,
               n_rounds: int, early_stop: Optional[int] = None,
               verbose_eval: int = 0) -> lgb.Booster:
    """
    LightGBM con custom objective (PNPU o nnPU secondo flag PNPU).
    raw_score=True in predict_proba garantisce logits grezzi.
    """
    if PNPU:
        fobj = make_pnpu_objective(pi, n_P, n_U, n_unl_total)
    else:
        fobj = make_nnpu_objective(pi, n_P, n_U)
    feval = make_roc_eval()

    # LightGBM 4.x: custom objective va in params["objective"]
    params_run = dict(params)
    params_run["objective"] = fobj

    dtrain = _make_dataset(X_tr, y_tr, features, cat_features)
    dval   = _make_dataset(X_es, y_es, features, cat_features, reference=dtrain)

    callbacks = [lgb.log_evaluation(period=verbose_eval if verbose_eval > 0 else 999999)]
    if early_stop is not None:
        callbacks.append(lgb.early_stopping(early_stop, verbose=False))

    return lgb.train(
        params_run, dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dval], valid_names=["es_val"],
        feval=feval, callbacks=callbacks,
    )


def predict_proba(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """raw_score=True evita trasformazioni interne di LightGBM."""
    return sigmoid(model.predict(X, raw_score=True))


def _inner_cv(X_pos_tr, X_unl_tr, X_neg_tr,
              pi: float, rng, features: list, cat_features: list) -> dict:
    """
    Inner CV (N_INNER_FOLDS) per (num_leaves, min_data_in_leaf, learning_rate).
    PNPU=True: N certi inclusi nel training inner (label=2).
    PNPU=False: N certi esclusi dal training, inclusi nella val per lift@k.
    """
    combos = list(itertools.product(LEAVES_GRID, MIN_DATA_GRID, LR_GRID))
    combo_scores = {c: [] for c in combos}

    n_p = len(X_pos_tr); n_u = len(X_unl_tr); n_n = len(X_neg_tr)
    inner_fold_p = rng.permutation(n_p) % N_INNER_FOLDS
    inner_fold_u = rng.permutation(n_u) % N_INNER_FOLDS
    inner_fold_n = rng.permutation(n_n) % N_INNER_FOLDS if n_n > 0 else np.array([])

    for ifold in range(N_INNER_FOLDS):
        t_if = time.time()
        mask_p_tr = inner_fold_p != ifold;  mask_p_va = inner_fold_p == ifold
        mask_u_tr = inner_fold_u != ifold;  mask_u_va = inner_fold_u == ifold
        mask_n_tr = (inner_fold_n != ifold) if n_n > 0 else np.zeros(0, dtype=bool)
        mask_n_va = (inner_fold_n == ifold) if n_n > 0 else np.zeros(0, dtype=bool)

        Xp_itr = X_pos_tr[mask_p_tr]; Xp_iva = X_pos_tr[mask_p_va]
        Xu_itr = X_unl_tr[mask_u_tr]; Xu_iva = X_unl_tr[mask_u_va]
        Xn_itr = X_neg_tr[mask_n_tr] if n_n > 0 else np.empty((0, X_pos_tr.shape[1]), np.float32)
        Xn_iva = X_neg_tr[mask_n_va] if n_n > 0 else np.empty((0, X_pos_tr.shape[1]), np.float32)

        if len(Xp_itr) == 0 or len(Xu_itr) == 0 or len(Xp_iva) == 0:
            print(f"      inner fold {ifold+1}/{N_INNER_FOLDS} [SKIP] vuoto"); continue

        n_P_i = len(Xp_itr); n_U_i = len(Xu_itr); n_N_i = len(Xn_itr)
        n_unl_total_i = n_U_i  # denominatore termine N (PNPU); ignorato in PU puro

        if PNPU:
            X_itr = np.vstack([Xp_itr, Xu_itr] + ([Xn_itr] if n_N_i > 0 else []))
            y_itr = np.concatenate([np.ones(n_P_i), np.zeros(n_U_i)]
                                   + ([np.full(n_N_i, 2.0)] if n_N_i > 0 else []))
        else:
            X_itr = np.vstack([Xp_itr, Xu_itr])
            y_itr = np.concatenate([np.ones(n_P_i), np.zeros(n_U_i)])

        rng2   = np.random.default_rng(RANDOM_SEED + ifold)
        n_es_i = max(5, int(0.10 * len(X_itr)))
        es_idx = rng2.choice(len(X_itr), size=n_es_i, replace=False)
        tr_idx = np.setdiff1d(np.arange(len(X_itr)), es_idx)

        y_es_bin = np.where(y_itr[es_idx] == 2, 0.0, y_itr[es_idx])

        n_P_tr_i = int((y_itr[tr_idx] == 1).sum())
        n_U_tr_i = int((y_itr[tr_idx] == 0).sum())

        for combo in combos:
            nl, md, lr = combo
            params = _build_params({"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": lr})
            try:
                model = train_lgbm(
                    X_itr[tr_idx], y_itr[tr_idx],
                    X_itr[es_idx], y_es_bin,
                    pi, n_P_tr_i, n_U_tr_i, n_unl_total_i,
                    params, features, cat_features,
                    n_rounds=N_INNER_ROUNDS, early_stop=None, verbose_eval=0,
                )
                sp = predict_proba(model, Xp_iva)
                su = predict_proba(model, Xu_iva)
                sn = predict_proba(model, Xn_iva) if len(Xn_iva) > 0 else np.array([])
                for k in K_GRID:
                    combo_scores[combo].append(lift_at_k(sp, sn, su, k=k))
            except Exception as e:
                print(f"      [WARN] combo {combo}: {e}")

        best_so_far = max(combos, key=lambda c: np.nanmean(combo_scores[c]) if combo_scores[c] else -np.inf)
        print(f"      inner {ifold+1}/{N_INNER_FOLDS}"
              f"  P={n_P_i}  U={n_U_i}  N={n_N_i}"
              f"  best={best_so_far}  ({int(time.time()-t_if)}s)")

    best_combo = max(combos, key=lambda c: np.nanmean(combo_scores[c]) if combo_scores[c] else -np.inf)
    nl, md, lr = best_combo
    return {"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": lr}


def run_cv(df: pd.DataFrame, features: list, cat_features: list) -> pd.DataFrame:
    rng          = np.random.default_rng(RANDOM_SEED)
    folds_to_run = [0] if TEST_MODE else list(range(N_OUTER_FOLDS))
    pi_to_run    = [0.02] if TEST_MODE else PI_GRID

    rows = []

    for outer_k in folds_to_run:
        print(f"\n[{_ts()}] -- Outer fold {outer_k+1}/{N_OUTER_FOLDS} --")
        fd = split_fold(df, features, outer_k)
        X_pos_tr, X_neg_tr, X_unl_tr = fd["X_pos_tr"], fd["X_neg_tr"], fd["X_unl_tr"]
        X_pos_va, X_neg_va, X_unl_va = fd["X_pos_va"], fd["X_neg_va"], fd["X_unl_va"]

        print(f"  Train: {len(X_pos_tr)} P | {len(X_neg_tr)} N | {len(X_unl_tr)} U")
        print(f"  Val:   {len(X_pos_va)} P | {len(X_neg_va)} N | {len(X_unl_va)} U")

        if len(X_pos_va) == 0 or len(X_unl_va) == 0:
            print("  [SKIP] fold di val vuoto."); continue

        n_unl_total = len(X_unl_tr)  # denominatore termine N (PNPU)

        if PNPU:
            X_all_tr = np.vstack([X_pos_tr, X_unl_tr, X_neg_tr])
            y_all_tr = np.concatenate([
                np.ones(len(X_pos_tr)),
                np.zeros(len(X_unl_tr)),
                np.full(len(X_neg_tr), 2.0),
            ])
        else:
            X_all_tr = np.vstack([X_pos_tr, X_unl_tr])
            y_all_tr = np.concatenate([np.ones(len(X_pos_tr)), np.zeros(len(X_unl_tr))])

        rng_es = np.random.default_rng(RANDOM_SEED + 100 + outer_k)
        n_es   = max(10, int(ES_FRAC * len(X_all_tr)))
        es_idx = rng_es.choice(len(X_all_tr), size=n_es, replace=False)
        tr_idx = np.setdiff1d(np.arange(len(X_all_tr)), es_idx)

        X_es = X_all_tr[es_idx]; y_es = y_all_tr[es_idx]
        y_es_bin = np.where(y_es == 2, 0.0, y_es)  # safe per entrambe le modalità

        X_tr = X_all_tr[tr_idx]; y_tr = y_all_tr[tr_idx]
        n_P_tr = int((y_tr == 1).sum())
        n_U_tr = int((y_tr == 0).sum())
        n_N_tr = int((y_tr == 2).sum())
        print(f"  ES: {n_es} | Train eff.: {len(X_tr)} ({n_P_tr} P, {n_U_tr} U, {n_N_tr} N)")

        row = {"fold": outer_k}

        for pi in pi_to_run:
            t_pi = time.time()
            n_combo = len(LEAVES_GRID) * len(MIN_DATA_GRID) * len(LR_GRID)
            print(f"\n  [pi={pi}] inner CV: {N_INNER_FOLDS} fold x {n_combo} combo...")

            best_params = _inner_cv(X_pos_tr, X_unl_tr, X_neg_tr, pi, rng, features, cat_features)
            print(f"  => {best_params}  ({int(time.time()-t_pi)}s)")

            params = _build_params(best_params)
            model  = train_lgbm(
                X_tr, y_tr, X_es, y_es_bin,
                pi, n_P_tr, n_U_tr, n_unl_total,
                params, features, cat_features,
                N_ROUNDS_MAX, early_stop=EARLY_STOP_ROUNDS, verbose_eval=0,
            )
            n_best = model.best_iteration if model.best_iteration > 0 else N_ROUNDS_MAX
            print(f"  best_iter={n_best}  ({int(time.time()-t_pi)}s)")

            sp = predict_proba(model, X_pos_va)
            sn = predict_proba(model, X_neg_va)
            su = predict_proba(model, X_unl_va)
            m  = eval_all(sp, sn, su)

            for k in K_GRID:
                row[f"lift{k}_pi{pi}"] = lift_at_k(sp, sn, su, k=k)
            row[f"best_iter_pi{pi}"]   = n_best
            row[f"best_leaves_pi{pi}"] = best_params["num_leaves"]
            row[f"best_lr_pi{pi}"]     = best_params["learning_rate"]
            row[f"prauc_pi{pi}"]       = m.get("pr_auc",  np.nan)
            row[f"rocauc_pi{pi}"]      = m.get("roc_auc", np.nan)
            print(f"  lift@1%={m['lift@1%']:.3f}  pr_auc={m.get('pr_auc',float('nan')):.3f}"
                  f"  roc_auc={m.get('roc_auc',float('nan')):.3f}  ({int(time.time()-t_pi)}s)")

        rows.append(row)

    return pd.DataFrame(rows)


def _ts(): return datetime.now().strftime("%H:%M:%S")


def retrain_and_save(df: pd.DataFrame, features: list, cat_features: list,
                     pi: float, params: dict):
    label = df[LABEL_COL]
    fold  = df[FOLD_COL].to_numpy(dtype=float, na_value=np.nan)
    is_p  = label.eq(1).fillna(False).values
    is_n  = label.eq(0).fillna(False).values
    is_u  = label.isna().values & ~np.isnan(fold)

    X = df[features].to_numpy(dtype=np.float32)
    X_pos = X[is_p]; X_unl = X[is_u]; X_neg = X[is_n]
    n_unl_total = len(X_unl)

    if PNPU:
        X_all = np.vstack([X_pos, X_unl, X_neg])
        y_all = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_unl)),
                                np.full(len(X_neg), 2.0)])
    else:
        X_all = np.vstack([X_pos, X_unl])
        y_all = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_unl))])

    rng_es = np.random.default_rng(RANDOM_SEED + 999)
    n_es   = max(10, int(ES_FRAC * len(X_all)))
    es_idx = rng_es.choice(len(X_all), size=n_es, replace=False)
    tr_idx = np.setdiff1d(np.arange(len(X_all)), es_idx)

    y_es_bin = np.where(y_all[es_idx] == 2, 0.0, y_all[es_idx])
    n_P_tr   = int((y_all[tr_idx] == 1).sum())
    n_U_tr   = int((y_all[tr_idx] == 0).sum())

    variant = "pnpu" if PNPU else "pu"
    print(f"  Retrain {variant.upper()}: {len(X_pos)} P | {len(X_unl)} U | {len(X_neg)} N  (ES: {n_es})")
    model = train_lgbm(
        X_all[tr_idx], y_all[tr_idx],
        X_all[es_idx], y_es_bin,
        pi, n_P_tr, n_U_tr, n_unl_total,
        _build_params(params), features, cat_features,
        N_ROUNDS_MAX, early_stop=EARLY_STOP_ROUNDS, verbose_eval=50,
    )
    out = Path(OUT_DIR) / f"lgbm_M{MODEL_NUMBER}_{variant}_pi{pi}.txt"
    model.save_model(str(out))
    print(f"  [SAVED] {out}")


def main():
    t0 = time.time()
    n_combo = len(LEAVES_GRID) * len(MIN_DATA_GRID) * len(LR_GRID)
    variant = "PNPU" if PNPU else "PU puro"
    print("=" * 65)
    print(f"  nnPU + LightGBM [{variant}] | M{MODEL_NUMBER} | TEST_MODE={TEST_MODE}")
    print(f"  pi grid: {PI_GRID if not TEST_MODE else '[0.02]'}")
    print(f"  combos: {n_combo} | inner folds: {N_INNER_FOLDS}")
    print(f"  N_INNER_ROUNDS={N_INNER_ROUNDS} N_ROUNDS_MAX={N_ROUNDS_MAX} ES={EARLY_STOP_ROUNDS}")
    print("=" * 65)

    df           = load_data()
    features     = get_features(df)
    cat_features = [f for f in _CAT_FEATURES if f in features]
    print(f"  Feature totali: {len(features)}  di cui categoriali: {len(cat_features)}")

    if RETRAIN_FINAL:
        if PI_FINAL is None or PARAMS_FINAL is None:
            raise ValueError("Imposta PI_FINAL e PARAMS_FINAL.")
        retrain_and_save(df, features, cat_features, PI_FINAL, PARAMS_FINAL)
    else:
        results = run_cv(df, features, cat_features)
        print("\n  OOF lift@1%")
        print(results.to_string(index=False))
        variant_tag = "pnpu" if PNPU else "pu"
        out = Path(OUT_DIR) / f"lgbm_M{MODEL_NUMBER}_{variant_tag}_fold_metrics.csv"
        results.to_csv(out, index=False)
        print(f"\n  [SAVED] {out}")

    s = int(time.time() - t0)
    print(f"\n  Completato in {s//60}m {s%60}s.")


if __name__ == "__main__":
    main()
