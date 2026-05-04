"""
Biased Learning -- SVM con kernel RBF                                [biased/svm.py]
U trattato come classe negativa con peso ridotto W_UNLABELED.

Pesi (biased learning):
  PNPU=True  (PNPU): P -> 1.0, N certi -> 1.0,           U -> W_UNLABELED
  PNPU=False (PU):   P -> 1.0, N certi -> W_UNLABELED,   U -> W_UNLABELED

Dataset: preprocessed (no NA -- obbligatorio per sklearn).
Kernel: RBF esatto (SVC). Fattibile: training ~9-32k righe per fold,
        ~1-15 secondi per fit su M1-M3.
Iperparametri tunati: C, gamma.
Selezione: inner validation set (INNER_VAL_FRAC del training).
Scoring: decision_function (ranking, non probabilità).
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

W_UNLABELED = 0.025        # peso degli U (e dei N in PU puro)

# Griglia SVC con kernel RBF (esatto -- fattibile su questi dataset)
# M1: ~32k righe -> ~10s/fit | M2: ~19k -> ~5s | M3: ~9k -> ~2s
RBF_C_GRID     = [0.1, 1.0, 10.0]
RBF_GAMMA_GRID = ["scale", 0.01, 0.1]   # "scale" = 1/(n_feat*var(X))

RANDOM_SEED = 42


import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

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
    return [
        {"C": c, "gamma": g}
        for c in RBF_C_GRID
        for g in RBF_GAMMA_GRID
    ]   # 3 x 3 = 9 combinazioni


def _make_model(params: dict) -> SVC:
    return SVC(kernel="rbf", C=params["C"], gamma=params["gamma"],
               random_state=RANDOM_SEED)


def _param_label(params: dict) -> str:
    return f"rbf_C{params['C']}_g{params['gamma']}"


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


def select_params(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    label_tr: pd.Series,
    param_grid: list,
    rng: np.random.Generator,
) -> dict:
    """
    Divide il training in inner_train e inner_val (INNER_VAL_FRAC).
    StandardScaler fittato solo su inner_train.
    Ritorna i parametri con miglior lift@1% sull'inner val.
    """
    n = len(X_tr)
    n_val   = max(1, int(INNER_VAL_FRAC * n))
    perm    = rng.permutation(n)
    idx_iva = perm[:n_val]
    idx_itr = perm[n_val:]

    X_itr, X_iva = X_tr[idx_itr], X_tr[idx_iva]
    y_itr        = y_tr[idx_itr]
    w_itr        = w_tr[idx_itr]
    label_iva    = label_tr.iloc[idx_iva].reset_index(drop=True)

    n_p_iva = int(label_iva.eq(1).sum())
    if n_p_iva == 0:
        print(f"    [WARN] inner val senza positivi -- uso default {_param_label(param_grid[0])}")
        return param_grid[0]

    scaler  = StandardScaler().fit(X_itr)
    X_itr_s = scaler.transform(X_itr)
    X_iva_s = scaler.transform(X_iva)

    print(f"    inner_train={len(X_itr):,}  inner_val={len(X_iva):,}"
          f"  (P={n_p_iva}, N={int(label_iva.eq(0).sum())}, U={int(label_iva.isna().sum())})")

    best_params, best_lift = param_grid[0], -np.inf
    t0 = time.time()

    for params in param_grid:
        t_m = time.time()
        model = _make_model(params)
        model.fit(X_itr_s, y_itr, sample_weight=w_itr)
        scores     = model.decision_function(X_iva_s)
        sp, sn, su = get_group_scores(scores, label_iva)
        lft        = lift_at_k(sp, sn, su, k=0.01)
        lbl        = _param_label(params)
        print(f"      {lbl:<30}  lift@1%={lft:.3f}  "
              f"({int(time.time()-t_m)}s -- {int(time.time()-t0)}s tot.)")
        if lft > best_lift:
            best_lift, best_params = lft, params

    print(f"    => best={_param_label(best_params)}  (inner lift@1%={best_lift:.3f})")
    return best_params


def run_cv(df: pd.DataFrame, features: list) -> pd.DataFrame:
    param_grid = _build_param_grid()
    rng        = np.random.default_rng(RANDOM_SEED)
    X_all      = df[features].to_numpy(dtype=float)
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

        n_p_va = int(label_va.eq(1).sum())
        print(f"  Train: {int(label_tr.eq(1).sum())} P | {int(label_tr.eq(0).sum())} N | {int(label_tr.isna().sum())} U")
        print(f"  Val:   {n_p_va} P | {int(label_va.eq(0).sum())} N | {int(label_va.isna().sum())} U")

        if n_p_va == 0:
            print("  [SKIP] nessun positivo in validation.")
            continue

        # -- Inner validation: selezione iperparametri -------------------------
        print(f"  [{_ts()}] Inner validation su {len(param_grid)} config RBF...")
        best_params = select_params(X_tr, y_tr, w_tr, label_tr, param_grid, rng)

        # -- Scaler finale fittato su tutto il training -------------------------
        print(f"  [{_ts()}] Scaling e fit finale ({_param_label(best_params)})...")
        t_fit  = time.time()
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = _make_model(best_params)
        model.fit(X_tr_s, y_tr, sample_weight=w_tr)
        print(f"  Fit completato in {int(time.time()-t_fit)}s")

        # -- Valutazione su outer val -------------------------------------------
        scores     = model.decision_function(X_va_s)
        sp, sn, su = get_group_scores(scores, label_va)
        metrics    = eval_all(sp, sn, su)

        elapsed = int(time.time() - t0)
        eta_str = _eta(elapsed, k + 1, N_OUTER_FOLDS)
        print(f"  lift@1%={metrics['lift@1%']:.3f}  "
              f"pr_auc={metrics['pr_auc']:.3f}  roc_auc={metrics['roc_auc']:.3f}")
        print(f"  [{_ts()}] Fold {k+1} completato | "
              f"trascorsi: {elapsed//60}m{elapsed%60:02d}s | ETA restante: {eta_str}")

        rows.append({
            "fold":       k,
            "best_model": _param_label(best_params),
            "lift_1pct":  metrics["lift@1%"],
            "lift_2pct":  metrics.get("lift@2%", np.nan),
            "lift_5pct":  metrics.get("lift@5%", np.nan),
            "pr_auc":     metrics["pr_auc"],
            "roc_auc":    metrics["roc_auc"],
        })

    return pd.DataFrame(rows)


def main():
    t0      = time.time()
    variant = "pnpu" if PNPU else "pu"

    print(f"  Biased SVM  |  M{MODEL_NUMBER}  |  variante={variant.upper()}")
    print(f"  RBF C grid: {RBF_C_GRID}  gamma grid: {RBF_GAMMA_GRID}")
    print(f"  Grid totale: {len(RBF_C_GRID)}x{len(RBF_GAMMA_GRID)}={len(RBF_C_GRID)*len(RBF_GAMMA_GRID)} config (esatto)")
    print(f"  W_UNLABELED={W_UNLABELED}")

    df       = load_data()
    features = get_features(df)
    print(f"  Feature selezionate: {len(features)}")

    results = run_cv(df, features)

    print("\n  RISULTATI OOF -- lift@1% per fold")
    print(results.to_string(index=False))
    print(f"\n  Lift@1% medio: {results['lift_1pct'].mean():.3f} "
          f"+/- {results['lift_1pct'].std():.3f}")

    out_csv = Path(OUT_DIR) / f"svm_M{MODEL_NUMBER}_{variant}_fold_metrics.csv"
    results.to_csv(out_csv, index=False)
    print(f"\n  [SAVED] {out_csv}")

    s = int(time.time() - t0)
    print(f"  Completato in {s//60}m{s%60:02d}s.")


if __name__ == "__main__":
    main()
