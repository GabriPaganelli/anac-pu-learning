"""
Biased Learning -- LightGBM                                           [biased/lgbm.py]
U trattato come classe negativa con peso ridotto W_UNLABELED.

Pesi (biased learning):
  PNPU=True  (PNPU): P -> 1.0, N certi -> 1.0,           U -> W_UNLABELED
  PNPU=False (PU):   P -> 1.0, N certi -> W_UNLABELED,   U -> W_UNLABELED

Dataset: nativi (LightGBM gestisce NA continui nativamente).
Iperparametri tunati: num_leaves, learning_rate, min_data_in_leaf, feature_fraction.
Selezione: inner validation con early stopping su binary logloss;
           scelta finale in base a lift@1% sull'inner val.
Metrica primaria: Lift@1% su P vs (N + U) nel fold di validazione.
"""

MODEL_NUMBER   = 3       # 1=M1, 2=M2, 3=M3
PNPU           = True    # True: PNPU (N certi peso 1); False: PU puro (N certi peso W_UNLABELED)

from pathlib import Path as _P
BASE_PATH = str(_P(__file__).resolve().parents[2] / "anac" / "output" / "parquet" / "model" / "nativi")
OUT_DIR   = str(_P(__file__).resolve().parent / "results")

LABEL_COL    = "label"
FOLD_COL     = "fold"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]

N_OUTER_FOLDS      = 4
INNER_VAL_FRAC     = 0.25    # frazione del training riservata all'inner validation
EARLY_STOPPING_ROUNDS = 50   # rounds senza miglioramento (BCE stabile, nnPU usava 200)
N_ROUNDS_INNER     = 1000    # max rounds nell'inner fit (early stopping lo riduce)
N_ROUNDS_OUTER     = None    # se None: usa best_iteration dell'inner fit selezionato

W_UNLABELED = 0.025          # peso degli U (e dei N in PU puro)

# LightGBM -- parametri fissi (non tunati)
#
# lambda_l2=0.5: mantenuto per coerenza con nnPU LightGBM e lieve beneficio
#   di regolarizzazione. Ha senso anche per binary BCE standard.
#
# min_sum_hessian_in_leaf: NON impostato (usa default LightGBM = 1e-3).
#   Nel nnPU serviva 1e-5 perché i gradienti normalizzati producono hessiane
#   O(1/n_P) ≈ 1e-5. Per binary BCE standard le hessiane sono p*(1-p)*w_i,
#   che con w_unlabeled=0.025 dà min ≈ 0.006 >> 1e-3. Il default è adeguato.
#
# feature_fraction=0.8: fisso.
#   Non tunato come in nnPU (default 1.0), ma valore diverso e giustificato.
#
# Griglia allineata a nnPU LightGBM: stesse 27 combinazioni per confronto diretto.
LGB_FIXED = {
    "objective":        "binary",
    "verbosity":        -1,
    "seed":             42,
    "subsample":        0.8,     # row subsampling
    "subsample_freq":   1,
    "feature_fraction": 1.0,     # allineato a nnPU LightGBM (default LightGBM)
    "lambda_l2":        0.5,     # regularizzazione L2 (come nnPU LightGBM)
}

# Griglia iperparametri: allineata a nnPU LightGBM (3 x 3 x 3 = 27 combinazioni)
# EARLY_STOP_ROUNDS ridotto a 50 rispetto ai 200 del nnPU: la binary BCE converge
# più regolarmente della loss nnPU, non serve attesa così lunga.
NUM_LEAVES_GRID       = [15, 31, 63]
LEARNING_RATE_GRID    = [0.01, 0.05, 0.1]
MIN_DATA_IN_LEAF_GRID = [10, 20, 50]

RANDOM_SEED = 42


import sys
import time
import itertools
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.pu_metrics import lift_at_k, eval_all
from metrics.preprocessing import encode_categoricals


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _eta(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return "?"
    rem = elapsed_s / done * (total - done)
    m, s = divmod(int(rem), 60)
    return f"{m}m{s:02d}s"


def compute_weights(label: pd.Series) -> np.ndarray:
    """
    PNPU=True:  P -> 1.0, N certi -> 1.0,           U -> W_UNLABELED
    PNPU=False: P -> 1.0, N certi -> W_UNLABELED,   U -> W_UNLABELED
    """
    w = np.full(len(label), W_UNLABELED, dtype=float)
    w[label.eq(1).fillna(False).values] = 1.0
    if PNPU:
        w[label.eq(0).fillna(False).values] = 1.0
    return w


def get_group_scores(scores: np.ndarray, label: pd.Series):
    """Separa i punteggi in (P, N, U) per il calcolo delle metriche."""
    return (
        scores[label.eq(1).fillna(False).values],
        scores[label.eq(0).fillna(False).values],
        scores[label.isna().values],
    )


def _build_param_grid() -> list:
    combos = list(itertools.product(
        NUM_LEAVES_GRID,
        LEARNING_RATE_GRID,
        MIN_DATA_IN_LEAF_GRID,
    ))
    return [
        {
            **LGB_FIXED,
            "num_leaves":       nl,
            "learning_rate":    lr,
            "min_data_in_leaf": md,
        }
        for nl, lr, md in combos
    ]


def _param_label(p: dict) -> str:
    return (f"nl={p['num_leaves']}_lr={p['learning_rate']}_"
            f"md={p['min_data_in_leaf']}_ff={p['feature_fraction']}")


def load_data() -> pd.DataFrame:
    path = Path(BASE_PATH) / f"M{MODEL_NUMBER}.parquet"
    print(f"[{_ts()}] Caricamento {path.name}...")
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    table = pq.read_table(str(path), filters=pc.field(FOLD_COL).is_valid())
    df = table.to_pandas()
    print(f"  {len(df):,} righe con fold assegnato")
    # encode solo le colonne categoriche (object/category) -> LightGBM gestisce NA continui
    df = encode_categoricals(df, extra_drop=COLS_TO_DROP)
    print(f"  {df.shape[1]} colonne dopo encoding")
    return df


def get_features(df: pd.DataFrame) -> list:
    drop = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "iufb"]


def select_params(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    label_tr: pd.Series,
    param_grid: list,
    rng: np.random.Generator,
) -> tuple:
    """
    Divide il training in inner_train e inner_val (INNER_VAL_FRAC).
    Per ogni combinazione: allena LightGBM con early stopping su binary logloss.
    Seleziona i parametri con miglior lift@1% sull'inner val.
    Ritorna (best_params, best_n_rounds).
    """
    n = len(X_tr)
    n_val   = max(1, int(INNER_VAL_FRAC * n))
    perm    = rng.permutation(n)
    idx_iva = perm[:n_val]
    idx_itr = perm[n_val:]

    X_itr, X_iva = X_tr[idx_itr], X_tr[idx_iva]
    y_itr, y_iva = y_tr[idx_itr], y_tr[idx_iva]
    w_itr, w_iva = w_tr[idx_itr], w_tr[idx_iva]
    label_iva    = label_tr.iloc[idx_iva].reset_index(drop=True)

    n_p_iva = int(label_iva.eq(1).sum())
    if n_p_iva == 0:
        default = param_grid[0]
        print(f"    [WARN] inner val senza positivi -- uso default params")
        return default, N_ROUNDS_INNER // 2

    dtrain = lgb.Dataset(X_itr, label=y_itr, weight=w_itr, free_raw_data=False)
    dval   = lgb.Dataset(X_iva, label=y_iva, weight=w_iva,
                         reference=dtrain, free_raw_data=False)

    print(f"    inner_train={len(X_itr):,}  inner_val={len(X_iva):,}"
          f"  (P={n_p_iva}, N={int(label_iva.eq(0).sum())}, U={int(label_iva.isna().sum())})")
    print(f"    Grid: {len(param_grid)} combinazioni in corso...")

    best_params, best_lift, best_n_rounds = param_grid[0], -np.inf, N_ROUNDS_INNER // 2
    t0 = time.time()

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(-1),
    ]

    for j, params in enumerate(param_grid):
        t_m = time.time()
        model = lgb.train(
            params,
            dtrain,
            num_boost_round = N_ROUNDS_INNER,
            valid_sets      = [dval],
            callbacks       = callbacks,
        )
        n_iter     = model.best_iteration
        pred_iva   = model.predict(X_iva)
        sp, sn, su = get_group_scores(pred_iva, label_iva)
        lft        = lift_at_k(sp, sn, su, k=0.01)

        if (j + 1) % 9 == 0 or j == 0 or j == len(param_grid) - 1:
            print(f"    [{j+1:2d}/{len(param_grid)}] {_param_label(params)}"
                  f"  iters={n_iter}  lift@1%={lft:.3f}"
                  f"  ({int(time.time()-t_m)}s -- {int(time.time()-t0)}s totali)")

        if lft > best_lift:
            best_lift, best_params, best_n_rounds = lft, params, n_iter

    print(f"    => best: {_param_label(best_params)}"
          f"  n_rounds={best_n_rounds}  (inner lift@1%={best_lift:.3f})")
    return best_params, best_n_rounds


def run_cv(df: pd.DataFrame, features: list) -> pd.DataFrame:
    param_grid = _build_param_grid()
    rng        = np.random.default_rng(RANDOM_SEED)
    X_all      = df[features].to_numpy(dtype=np.float64)
    label      = df[LABEL_COL].reset_index(drop=True)
    fold       = df[FOLD_COL].to_numpy(dtype=float)
    y_all      = label.eq(1).fillna(False).astype(int).to_numpy()
    w_all      = compute_weights(label)

    rows = []
    t0   = time.time()

    for k in range(N_OUTER_FOLDS):
        print(f"\n[{_ts()}] ====== Outer fold {k+1}/{N_OUTER_FOLDS} ======")

        mask_tr = fold != k
        mask_va = fold == k

        X_tr     = X_all[mask_tr];  X_va     = X_all[mask_va]
        y_tr     = y_all[mask_tr]
        w_tr     = w_all[mask_tr]
        label_tr = label[mask_tr].reset_index(drop=True)
        label_va = label[mask_va].reset_index(drop=True)

        n_p_tr = int(label_tr.eq(1).sum())
        n_n_tr = int(label_tr.eq(0).sum())
        n_u_tr = int(label_tr.isna().sum())
        n_p_va = int(label_va.eq(1).sum())

        print(f"  Train: {n_p_tr} P | {n_n_tr} N | {n_u_tr} U")
        print(f"  Val:   {n_p_va} P | {int(label_va.eq(0).sum())} N | {int(label_va.isna().sum())} U")

        if n_p_va == 0:
            print("  [SKIP] nessun positivo in validation.")
            continue

        # -- Inner validation: selezione iperparametri --------------------------
        print(f"  [{_ts()}] Inner validation ({len(param_grid)} combinazioni)...")
        best_params, best_n_rounds = select_params(
            X_tr, y_tr, w_tr, label_tr, param_grid, rng
        )
        n_rounds_outer = N_ROUNDS_OUTER if N_ROUNDS_OUTER is not None else best_n_rounds

        # -- Fit GBM completo sul training fold --------------------------------
        print(f"  [{_ts()}] Alleno GBM finale "
              f"({_param_label(best_params)}, n_rounds={n_rounds_outer})...")
        t_fit   = time.time()
        dtrain  = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=False)
        model   = lgb.train(
            best_params,
            dtrain,
            num_boost_round = n_rounds_outer,
            callbacks       = [lgb.log_evaluation(-1)],
        )
        print(f"  Fit completato in {int(time.time()-t_fit)}s")

        # -- Valutazione su outer val -------------------------------------------
        pred       = model.predict(X_va)
        sp, sn, su = get_group_scores(pred, label_va)
        metrics    = eval_all(sp, sn, su)

        elapsed = int(time.time() - t0)
        eta_str = _eta(elapsed, k + 1, N_OUTER_FOLDS)
        print(f"  lift@1%={metrics['lift@1%']:.3f}  "
              f"pr_auc={metrics['pr_auc']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
        print(f"  [{_ts()}] Fold {k+1} completato | "
              f"trascorsi: {elapsed//60}m{elapsed%60:02d}s | ETA restante: {eta_str}")

        rows.append({
            "fold":         k,
            "best_params":  _param_label(best_params),
            "n_rounds":     n_rounds_outer,
            "lift_1pct":    metrics["lift@1%"],
            "lift_2pct":    metrics.get("lift@2%", np.nan),
            "lift_5pct":    metrics.get("lift@5%", np.nan),
            "pr_auc":       metrics["pr_auc"],
            "roc_auc":      metrics["roc_auc"],
        })

    return pd.DataFrame(rows)


def main():
    t0      = time.time()
    variant = "pnpu" if PNPU else "pu"

    print(f"  Biased GBM  |  M{MODEL_NUMBER}  |  variante={variant.upper()}")
    print(f"  W_UNLABELED={W_UNLABELED}  |  subsample={LGB_FIXED['subsample']}")
    n_combo = len(NUM_LEAVES_GRID)*len(LEARNING_RATE_GRID)*len(MIN_DATA_IN_LEAF_GRID)
    print(f"  Grid: {len(NUM_LEAVES_GRID)}x{len(LEARNING_RATE_GRID)}x"
          f"{len(MIN_DATA_IN_LEAF_GRID)} = {n_combo} combinazioni"
          f"  (feature_fraction={LGB_FIXED['feature_fraction']}, lambda_l2={LGB_FIXED['lambda_l2']})")
    print(f"  Early stopping: {EARLY_STOPPING_ROUNDS} rounds | max inner: {N_ROUNDS_INNER}")

    df       = load_data()
    features = get_features(df)
    print(f"  Feature selezionate: {len(features)}")

    results = run_cv(df, features)

    print("\n  RISULTATI OOF -- lift@1% per fold")
    print(results.to_string(index=False))
    print(f"\n  Lift@1% medio: {results['lift_1pct'].mean():.3f} "
          f"+/- {results['lift_1pct'].std():.3f}")

    out_csv = Path(OUT_DIR) / f"lgbm_M{MODEL_NUMBER}_{variant}_fold_metrics.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n  [SAVED] {out_csv}")

    s = int(time.time() - t0)
    print(f"  Completato in {s//60}m{s%60:02d}s.")


if __name__ == "__main__":
    main()
