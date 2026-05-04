"""
Biased Learning -- Random Forest                                       [biased/rf.py]
U trattato come classe negativa con peso ridotto W_UNLABELED.

Pesi (biased learning):
  PNPU=True  (PNPU): P -> 1.0, N certi -> 1.0,           U -> W_UNLABELED
  PNPU=False (PU):   P -> 1.0, N certi -> W_UNLABELED,   U -> W_UNLABELED

Dataset: preprocessed (no NA -- obbligatorio per sklearn).
Iperparametro tunato: max_features (frazione di feature per split).
Selezione: inner validation set (INNER_VAL_FRAC del training).
Metrica primaria: Lift@1% su P vs (N + U) nel fold di validazione.
"""

MODEL_NUMBER   = 3       # 1=M1, 2=M2, 3=M3
PNPU           = True    # True: PNPU (N certi peso 1); False: PU puro (N certi peso W_UNLABELED)

from pathlib import Path as _P
BASE_PATH = str(_P(__file__).resolve().parents[2] / "anac" / "output" / "parquet" / "model" / "preprocessed")
OUT_DIR   = str(_P(__file__).resolve().parent / "results")

LABEL_COL    = "label"
FOLD_COL     = "fold"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]

N_OUTER_FOLDS  = 4
INNER_VAL_FRAC = 0.25      # frazione del training riservata all'inner validation

W_UNLABELED    = 0.025     # peso degli U (e dei N in PU puro)

# RF -- iperparametri fissi
N_ESTIMATORS        = 500  # alberi nel fit finale (outer)
N_ESTIMATORS_INNER  = 150  # alberi nell'inner validation (velocità)
MAX_DEPTH           = 8    # profondità moderata
MIN_SAMPLES_LEAF    = 5    # regularizzazione sulle foglie

# Griglia per max_features (frazione delle feature campionate per ogni split)
MAX_FEATURES_GRID = [0.2, 0.3, 0.4, 0.5]

RANDOM_SEED = 42


import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

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


def load_data() -> pd.DataFrame:
    path = Path(BASE_PATH) / f"M{MODEL_NUMBER}.parquet"
    print(f"[{_ts()}] Caricamento {path.name}...")
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    table = pq.read_table(str(path), filters=pc.field(FOLD_COL).is_valid())
    df = table.to_pandas()
    print(f"  {len(df):,} righe con fold assegnato")
    df = encode_categoricals(df, extra_drop=COLS_TO_DROP)
    print(f"  {df.shape[1]} colonne dopo encoding")
    return df


def get_features(df: pd.DataFrame) -> list:
    drop = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "iufb"]


def select_max_features(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    label_tr: pd.Series,
    rng: np.random.Generator,
) -> float:
    """
    Divide il training set in inner_train e inner_val (INNER_VAL_FRAC).
    Allena RF con N_ESTIMATORS_INNER alberi per ogni valore di MAX_FEATURES_GRID.
    Ritorna il max_features che massimizza lift@1% sull'inner val.
    """
    n = len(X_tr)
    n_val = max(1, int(INNER_VAL_FRAC * n))
    perm    = rng.permutation(n)
    idx_iva = perm[:n_val]
    idx_itr = perm[n_val:]

    X_itr, X_iva = X_tr[idx_itr], X_tr[idx_iva]
    y_itr        = y_tr[idx_itr]
    w_itr        = w_tr[idx_itr]
    label_iva    = label_tr.iloc[idx_iva].reset_index(drop=True)

    n_p_iva = int(label_iva.eq(1).sum())
    if n_p_iva == 0:
        default = MAX_FEATURES_GRID[1]
        print(f"    [WARN] inner val senza positivi -- uso default max_features={default}")
        return default

    print(f"    inner_train={len(X_itr):,}  inner_val={len(X_iva):,}"
          f"  (P={n_p_iva}, N={int(label_iva.eq(0).sum())}, U={int(label_iva.isna().sum())})")

    best_mf, best_lift = MAX_FEATURES_GRID[0], -np.inf
    t0 = time.time()

    for mf in MAX_FEATURES_GRID:
        t_mf = time.time()
        rf = RandomForestClassifier(
            n_estimators     = N_ESTIMATORS_INNER,
            max_features     = mf,
            max_depth        = MAX_DEPTH,
            min_samples_leaf = MIN_SAMPLES_LEAF,
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        )
        rf.fit(X_itr, y_itr, sample_weight=w_itr)
        proba     = rf.predict_proba(X_iva)[:, 1]
        sp, sn, su = get_group_scores(proba, label_iva)
        lft       = lift_at_k(sp, sn, su, k=0.01)
        print(f"      max_features={mf:.1f}  lift@1%={lft:.3f}  "
              f"({int(time.time()-t_mf)}s -- {int(time.time()-t0)}s totali)")
        if lft > best_lift:
            best_lift, best_mf = lft, mf

    print(f"    => max_features*={best_mf}  (inner lift@1%={best_lift:.3f})")
    return best_mf


def run_cv(df: pd.DataFrame, features: list) -> pd.DataFrame:
    rng   = np.random.default_rng(RANDOM_SEED)
    X_all = df[features].to_numpy(dtype=float)
    label = df[LABEL_COL].reset_index(drop=True)
    fold  = df[FOLD_COL].to_numpy(dtype=float)
    y_all = label.eq(1).fillna(False).astype(int).to_numpy()
    w_all = compute_weights(label)

    rows = []
    t0   = time.time()

    for k in range(N_OUTER_FOLDS):
        print(f"\n[{_ts()}] ====== Outer fold {k+1}/{N_OUTER_FOLDS} ======")

        mask_tr = fold != k    # NaN != k -> True (già esclusi dal parquet)
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

        # -- Inner validation: selezione max_features --------------------------
        print(f"  [{_ts()}] Inner validation per selezione max_features...")
        best_mf = select_max_features(X_tr, y_tr, w_tr, label_tr, rng)

        # -- Fit RF completo sul training fold ---------------------------------
        print(f"  [{_ts()}] Alleno RF finale "
              f"(n_estimators={N_ESTIMATORS}, max_features={best_mf})...")
        t_fit = time.time()
        rf = RandomForestClassifier(
            n_estimators     = N_ESTIMATORS,
            max_features     = best_mf,
            max_depth        = MAX_DEPTH,
            min_samples_leaf = MIN_SAMPLES_LEAF,
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        )
        rf.fit(X_tr, y_tr, sample_weight=w_tr)
        print(f"  Fit completato in {int(time.time()-t_fit)}s")

        # -- Valutazione su outer val -------------------------------------------
        proba      = rf.predict_proba(X_va)[:, 1]
        sp, sn, su = get_group_scores(proba, label_va)
        metrics    = eval_all(sp, sn, su)

        elapsed = int(time.time() - t0)
        eta_str = _eta(elapsed, k + 1, N_OUTER_FOLDS)
        print(f"  lift@1%={metrics['lift@1%']:.3f}  "
              f"pr_auc={metrics['pr_auc']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
        print(f"  [{_ts()}] Fold {k+1} completato | "
              f"trascorsi: {elapsed//60}m{elapsed%60:02d}s | ETA restante: {eta_str}")

        rows.append({
            "fold":         k,
            "max_features": best_mf,
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

    print(f"  Biased RF  |  M{MODEL_NUMBER}  |  variante={variant.upper()}")
    print(f"  n_estimators={N_ESTIMATORS}  max_depth={MAX_DEPTH}  "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}")
    print(f"  W_UNLABELED={W_UNLABELED}  |  max_features grid={MAX_FEATURES_GRID}")

    df       = load_data()
    features = get_features(df)
    print(f"  Feature selezionate: {len(features)}")

    results = run_cv(df, features)

    print("\n  RISULTATI OOF -- lift@1% per fold")
    print(results.to_string(index=False))
    print(f"\n  Lift@1% medio: {results['lift_1pct'].mean():.3f} "
          f"+/- {results['lift_1pct'].std():.3f}")

    out_csv = Path(OUT_DIR) / f"rf_M{MODEL_NUMBER}_{variant}_fold_metrics.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n  [SAVED] {out_csv}")

    s = int(time.time() - t0)
    print(f"  Completato in {s//60}m{s%60:02d}s.")


if __name__ == "__main__":
    main()
