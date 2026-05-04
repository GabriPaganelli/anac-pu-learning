"""
Bagging PU + LightGBM
Base learner: LightGBM con loss BCE standard (objective=binary).
Dati: nativi (NA e categoriali gestiti nativamente da LightGBM).

Varianti:
  PNPU=True : N certi fissi in ogni bootstrap (sempre presenti, mai campionati).
  PNPU=False: N certi nel pool U, campionati con la stessa probabilità degli U.

Nessun peso esplicito sugli unlabeled: la correzione per π viene
dal rapporto implicito s/|U| nel campionamento bootstrap (Mordelet & Vert 2014).

Inner CV (analogo a nnpu_lgbm): grid num_leaves × N_ROUNDS, B_INNER=20 fits per combo,
metrica lift@1% su P+N dell'inner validation. Seleziona best combo per ogni outer fold.
"""

MODEL     = 1       # 1=M1, 2=M2, 3=M3
PNPU      = True    # True=PNPU, False=PU puro
TEST_MODE = False   # True = run rapido per verifica e stima tempi

B         = 150     if not TEST_MODE else 5
S         = 15_000  if not TEST_MODE else 300
RANDOM_SEED = 42
KS          = (0.01, 0.02, 0.05)

# Inner CV
N_INNER_FOLDS = 3
B_INNER       = 20  if not TEST_MODE else 3   # fits per combo nell'inner search
LEAVES_GRID   = [15, 31, 63]                  if not TEST_MODE else [15, 31]
ROUNDS_GRID   = [50, 100, 200]                if not TEST_MODE else [50]

# Parametri LightGBM fissi (invarianti alla grid) — da nnpu_lgbm
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

from pathlib import Path as _P
BASE_PATH = str(_P(__file__).resolve().parents[2] / "anac" / "output" / "parquet" / "model" / "nativi")
OUT_DIR   = str(_P(__file__).resolve().parent / "results")

LABEL_COL    = "label"
FOLD_COL     = "fold"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]

import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.pu_metrics import lift_at_k, eval_all

_CAT_FEATURES: list = []

def load_data() -> pd.DataFrame:
    global _CAT_FEATURES

    path = Path(BASE_PATH) / f"M{MODEL}.parquet"
    df   = pd.read_parquet(path)
    df   = df[df[FOLD_COL].notna()].reset_index(drop=True)

    drop_set = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    cat_cols = [c for c in df.columns
                if c not in drop_set and df[c].dtype.kind not in "iufb"]
    _CAT_FEATURES = cat_cols

    for col in cat_cols:
        df[col] = (df[col].astype("category")
                          .cat.codes
                          .replace(-1, np.nan)
                          .astype("float32"))

    if TEST_MODE:
        is_labeled = df[LABEL_COL].notna()
        is_unl     = df[LABEL_COL].isna()
        unl_sample = (df[is_unl]
                      .groupby(FOLD_COL, group_keys=False)
                      .apply(lambda g: g.sample(min(2_000, len(g)),
                                                 random_state=42)))
        df = pd.concat([df[is_labeled], unl_sample]).reset_index(drop=True)

    if cat_cols:
        print(f"  {len(cat_cols)} colonne categoriali convertite: {cat_cols}")

    print(f"  Dati M{MODEL}: {len(df)} righe, "
          f"{df[LABEL_COL].eq(1).sum()} P, "
          f"{df[LABEL_COL].eq(0).sum()} N, "
          f"{df[LABEL_COL].isna().sum()} U")
    return df


def get_features(df: pd.DataFrame) -> list:
    drop_set = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    return [c for c in df.columns
            if c not in drop_set and df[c].dtype.kind in "iufb"]


def make_dataset(X: np.ndarray, y: np.ndarray, features: list,
                 reference: lgb.Dataset = None) -> lgb.Dataset:
    cat_idx = [features.index(f) for f in _CAT_FEATURES if f in features]
    return lgb.Dataset(
        X, label=y,
        feature_name=features,
        categorical_feature=cat_idx if cat_idx else "auto",
        reference=reference,
        free_raw_data=False,
    )


def bagging_scores(X_P, X_N, X_pool, X_score,
                   features, params, n_rounds, b_iters, rng):
    """
    Esegue b_iters fits di bagging PU e restituisce score medi su X_score.
    X_pool: pool di campionamento negativo (U oppure U∪N per PU puro).
    X_N: negativi fissi (aggiunti sempre se PNPU, ignorati se PU puro).
    """
    s       = min(S, len(X_pool))
    replace = s == len(X_pool)
    acc     = np.zeros(len(X_score))

    for _ in range(b_iters):
        idx   = rng.choice(len(X_pool), size=s, replace=replace)
        X_neg = X_pool[idx]

        if PNPU and len(X_N) > 0:
            X_tr = np.vstack([X_P, X_N, X_neg])
            y_tr = np.concatenate([np.ones(len(X_P)),
                                   np.zeros(len(X_N)),
                                   np.zeros(s)])
        else:
            X_tr = np.vstack([X_P, X_neg])
            y_tr = np.concatenate([np.ones(len(X_P)), np.zeros(s)])

        p        = dict(params)
        p["seed"] = int(rng.randint(0, 99999))
        model    = lgb.train(p, make_dataset(X_tr, y_tr, features),
                             num_boost_round=n_rounds)
        acc     += model.predict(X_score)

    return acc / b_iters


def inner_cv(X_P, X_N, X_U, features, rng_inner):
    """
    Seleziona best (num_leaves, N_ROUNDS) via inner CV.
    Split: P e N in N_INNER_FOLDS fold stratificati; U usato interamente.
    Metrica: lift@1% su P+N dell'inner validation.
    """
    grid = list(itertools.product(LEAVES_GRID, ROUNDS_GRID))
    combo_scores = {c: [] for c in grid}

    # Costruisce fold assignments per P e N
    n_p, n_n = len(X_P), len(X_N)
    folds_p  = np.arange(n_p) % N_INNER_FOLDS
    folds_n  = np.arange(n_n) % N_INNER_FOLDS
    rng_inner.shuffle(folds_p)
    rng_inner.shuffle(folds_n)

    for ifold in range(N_INNER_FOLDS):
        # Split P e N
        X_P_itr = X_P[folds_p != ifold];  X_P_iva = X_P[folds_p == ifold]
        X_N_itr = X_N[folds_n != ifold];  X_N_iva = X_N[folds_n == ifold]

        # Pool: U completo in training (non diviso — abbastanza grande)
        if PNPU:
            X_pool_itr = X_U
        else:
            X_pool_itr = np.vstack([X_U, X_N_itr]) if len(X_N_itr) > 0 else X_U

        # Validation set per lift: P_iva e N_iva
        X_val = np.vstack([X_P_iva, X_N_iva])
        y_val = np.concatenate([np.ones(len(X_P_iva)), np.zeros(len(X_N_iva))])

        for (nl, nr) in grid:
            params_c = dict(LGBM_FIXED, num_leaves=nl)
            scores_val = bagging_scores(
                X_P_itr, X_N_itr, X_pool_itr, X_val,
                features, params_c, nr, B_INNER, rng_inner
            )
            lft = lift_at_k(scores_val[y_val == 1],
                            scores_val[y_val == 0],
                            np.array([]),   # nessun U in validation
                            k=0.01)
            combo_scores[(nl, nr)].append(lft if not np.isnan(lft) else 0.0)

    # Media lift@1% per combo
    mean_scores = {c: np.mean(v) for c, v in combo_scores.items()}
    best        = max(mean_scores, key=mean_scores.get)

    print(f"  Inner CV — best: num_leaves={best[0]}, N_ROUNDS={best[1]}"
          f"  (lift@1%={mean_scores[best]:.3f})")
    for c, s in sorted(mean_scores.items(), key=lambda x: -x[1]):
        print(f"    leaves={c[0]:3d}  rounds={c[1]:3d}  lift@1%={s:.3f}")

    return best


def run_bagging_lgbm(dati: pd.DataFrame) -> dict:

    rng          = check_random_state(RANDOM_SEED)
    features     = get_features(dati)
    score_raw    = np.full(len(dati), np.nan)
    fold_metrics = {}

    folds     = sorted(dati[FOLD_COL].dropna().unique().astype(int))
    fold_np   = dati[FOLD_COL].values.astype(float)
    label_all = dati[LABEL_COL].values.astype(float)
    X_full    = dati[features].to_numpy(dtype=np.float32)

    for fold_test in folds:

        print(f"\nFold test: {fold_test}  ({'PNPU' if PNPU else 'PU puro'})")

        tr_mask = (fold_np != fold_test) & ~np.isnan(fold_np)
        te_mask = (fold_np == fold_test)

        X_P    = X_full[(label_all == 1) & tr_mask]
        X_N    = X_full[(label_all == 0) & tr_mask]
        X_U    = X_full[np.isnan(label_all) & tr_mask]
        X_test = X_full[te_mask]

        X_pool = X_U if PNPU else (
            np.vstack([X_U, X_N]) if len(X_N) > 0 else X_U)

        _s = min(S, len(X_pool))
        print(f"  P={len(X_P)}, N={len(X_N)}, U={len(X_U)}, "
              f"pool={len(X_pool)}, s={_s}, replace={_s == len(X_pool)}")

        best_leaves, best_rounds = inner_cv(
            X_P, X_N, X_U, features,
            rng_inner=check_random_state(rng.randint(0, 99999))
        )
        params_outer = dict(LGBM_FIXED, num_leaves=best_leaves)

        print(f"  Bagging outer: B={B}, num_leaves={best_leaves}, "
              f"N_ROUNDS={best_rounds}")
        scores_test = bagging_scores(
            X_P, X_N, X_pool, X_test,
            features, params_outer, best_rounds, B, rng
        )
        score_raw[te_mask] = scores_test

        label_te = label_all[te_mask]

        m = eval_all(
            scores_pos=scores_test[label_te == 1],
            scores_neg=scores_test[label_te == 0],
            scores_unl=scores_test[np.isnan(label_te)],
            ks=KS,
        )
        fold_metrics[fold_test] = {"fold": fold_test, **m}
        print(f"  Metriche fold {fold_test}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    print("\nMetriche finali (tutti i fold)")
    final = eval_all(
        scores_pos=score_raw[label_all == 1],
        scores_neg=score_raw[label_all == 0],
        scores_unl=score_raw[np.isnan(label_all)],
        ks=KS,
    )
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")

    variant = "pnpu" if PNPU else "pu"
    prefix  = f"bagging_lgbm_M{MODEL}_{variant}"
    out     = Path(OUT_DIR)

    pd.DataFrame({"score_raw": score_raw}).to_csv(
        out / f"{prefix}_scores.csv", index=False)
    pd.DataFrame([final]).to_csv(
        out / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(list(fold_metrics.values())).to_csv(
        out / f"{prefix}_fold_metrics.csv", index=False)

    print(f"\nFile salvati: {prefix}_*.csv")
    return {"score_raw": score_raw, "fold_metrics": fold_metrics, "final_metrics": final}


if __name__ == "__main__":
    dati = load_data()
    run_bagging_lgbm(dati)
