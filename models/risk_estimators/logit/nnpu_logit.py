"""
nnPU Risk Estimator + Regressione Logistica con Elastic Net
PNPU=True : loss estesa con termine N certi, normalizzato per n_unl_total.
PNPU=False: loss nnPU pura; N certi esclusi dal training, usati solo in val.

Ottimizzazione: SGD con cosine LR decay.
Regularizzazione: Elastic Net (alpha=0.5 fisso, lambda da inner CV).
Dataset: preprocessed (no NA). Label: 1=P, 0=N, NaN=U.
"""

MODEL_NUMBER   = 3          # 1=M1, 2=M2, 3=M3
PNPU           = True       # True = PNPU (N certi nel training); False = nnPU puro
TEST_MODE      = False      # True → smoke test rapido (1 fold, pochi dati, 3 epoche)
RETRAIN_FINAL  = False      # True → riallena su tutto il dataset con (pi*, lambda*)
PI_FINAL       = None
LAMBDA_FINAL   = None

from pathlib import Path as _P
BASE_PATH = str(_P(__file__).resolve().parents[3] / "anac" / "output" / "parquet" / "model" / "preprocessed")
OUT_DIR   = str(_P(__file__).resolve().parent / "results")

LABEL_COL    = "label"
FOLD_COL     = "fold"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]

N_OUTER_FOLDS  = 4
N_INNER_FOLDS  = 5   # re-split fresco sul training set per evitare correlazioni con fold outer

PI_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]

EXPLORATORY_MODE = False
LAMBDA_GRID = (
    [1e-3, 1e-2, 1e-1]              if TEST_MODE       else
    [1e-4, 1e-3, 1e-2, 1e-1, 1e0]   if EXPLORATORY_MODE else
    list(__import__('numpy').logspace(-5, 1, 30))
)

ELASTIC_NET_ALPHA = 0.5   # fisso

# U_PER_FOLD usato solo in TEST_MODE per _make_temp_folds.
U_PER_FOLD   = 200

BATCH_SIZE_U = 64    if TEST_MODE else 1_024
N_EPOCHS     = 3     if TEST_MODE else 50
LR_INIT      = 0.01

RANDOM_SEED  = 42


import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from metrics.pu_metrics import eval_all, lift_at_k
from metrics.preprocessing import encode_categoricals


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente stabile."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _sig_loss(f: np.ndarray, y: float) -> np.ndarray:
    """Sigmoid loss: log(1 + exp(-y*f)),  y ∈ {+1, -1}."""
    return np.log1p(np.exp(-y * f))


def _nnpu_grad_pnpu(w, b, X_pos, X_unl, pi, X_neg, n_unl_total):
    """
    Gradiente PNPU: R_PNPU = pi*R_P(+1) + max(0, R_U(-1) - pi*R_P(-1)) + (1-pi)*R_N(-1)

    Il termine N è normalizzato per n_unl_total (dimensione totale del pool U),
    non per len(X_neg): ogni esempio N contribuisce ~1/n_unl_total per batch,
    stesso ordine degli U, evitando che il termine N domini il gradiente.
    """
    f_p = X_pos @ w + b
    f_u = X_unl @ w + b
    f_n = X_neg @ w + b

    R_pos_neg = _sig_loss(f_p, -1.0).mean()
    R_unl_neg = _sig_loss(f_u, -1.0).mean()

    gf_p_plus = -_sigmoid(-f_p)
    dw = pi * (X_pos.T @ gf_p_plus) / len(X_pos)
    db = pi * gf_p_plus.mean()

    if R_unl_neg - pi * R_pos_neg >= 0.0:
        gf_u = _sigmoid(f_u)
        gf_p_neg = _sigmoid(f_p)
        dw += (X_unl.T @ gf_u) / len(X_unl) - pi * (X_pos.T @ gf_p_neg) / len(X_pos)
        db += gf_u.mean() - pi * gf_p_neg.mean()

    gf_n = _sigmoid(f_n)
    dw += (1.0 - pi) * (X_neg.T @ gf_n) / n_unl_total
    db += (1.0 - pi) * gf_n.sum() / n_unl_total

    return dw, db


def _nnpu_grad_pu(w, b, X_pos, X_unl, pi):
    """
    Gradiente nnPU puro: R_nnPU = pi*R_P(+1) + max(0, R_U(-1) - pi*R_P(-1))

    Gradiente P per-campione (non collassato a -pi/n_P costante): la versione
    collassata elimina il segnale per-campione e impedisce l'apprendimento.
    """
    f_p = X_pos @ w + b
    f_u = X_unl @ w + b

    R_pos_neg = _sig_loss(f_p, -1.0).mean()
    R_unl_neg = _sig_loss(f_u, -1.0).mean()

    gf_p_plus = -_sigmoid(-f_p)
    dw = pi * (X_pos.T @ gf_p_plus) / len(X_pos)
    db = pi * gf_p_plus.mean()

    if R_unl_neg - pi * R_pos_neg >= 0.0:
        gf_u = _sigmoid(f_u)
        gf_p_neg = _sigmoid(f_p)
        dw += (X_unl.T @ gf_u) / len(X_unl) - pi * (X_pos.T @ gf_p_neg) / len(X_pos)
        db += gf_u.mean() - pi * gf_p_neg.mean()

    return dw, db


def _en_grad(w, lam):
    """Gradiente Elastic Net: lam·(alpha·sign(w) + (1-alpha)·w)."""
    return lam * (ELASTIC_NET_ALPHA * np.sign(w) + (1.0 - ELASTIC_NET_ALPHA) * w)


def train(X_pos, X_unl, pi, lam, rng, X_neg=None, verbose=False) -> tuple:
    """
    SGD con cosine LR decay su nnPU (o PNPU) + Elastic Net.
    X_neg obbligatorio quando PNPU=True.
    Restituisce (w, b).
    """
    if PNPU and (X_neg is None or len(X_neg) == 0):
        raise ValueError("X_neg obbligatorio per PNPU.")

    n_feat = X_pos.shape[1]
    w = np.zeros(n_feat)
    b = 0.0

    n_unl      = len(X_unl)
    n_steps    = N_EPOCHS * max(1, n_unl // BATCH_SIZE_U)
    step       = 0
    print_every = max(1, N_EPOCHS // 5)
    t_train    = time.time()

    n_unl_total = n_unl  # denominatore termine N in PNPU

    for epoch in range(N_EPOCHS):
        perm = rng.permutation(n_unl)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n_unl, BATCH_SIZE_U):
            batch_u = X_unl[perm[i : i + BATCH_SIZE_U]]
            if len(batch_u) == 0:
                continue

            lr = 0.5 * LR_INIT * (1.0 + np.cos(np.pi * step / max(n_steps, 1)))
            step += 1

            if PNPU:
                dw, db = _nnpu_grad_pnpu(w, b, X_pos, batch_u, pi, X_neg, n_unl_total)
            else:
                dw, db = _nnpu_grad_pu(w, b, X_pos, batch_u, pi)

            f_p = X_pos @ w + b
            f_u = batch_u @ w + b
            rp_pos   = np.mean(_sig_loss(f_p, +1))
            ru_neg   = np.mean(_sig_loss(f_u, -1))
            rp_neg   = np.mean(_sig_loss(f_p, -1))
            neg_term = ru_neg - pi * rp_neg
            epoch_loss += pi * rp_pos + max(0.0, neg_term)
            n_batches  += 1

            dw += _en_grad(w, lam)
            w  -= lr * dw
            b  -= lr * db

        if verbose and (epoch + 1) % print_every == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed  = int(time.time() - t_train)
            print(f"    epoch {epoch+1:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  lr={lr:.2e}  ({elapsed}s)")

    return w, b


def predict(X, w, b) -> np.ndarray:
    return _sigmoid(X @ w + b)


def load_data() -> pd.DataFrame:
    path = Path(BASE_PATH) / f"M{MODEL_NUMBER}.parquet"
    print(f"  Caricamento {path.name}...")
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
    df = encode_categoricals(df, extra_drop=COLS_TO_DROP)
    print(f"  {df.shape[1]} colonne dopo encoding")

    if FOLD_COL not in df.columns:
        if TEST_MODE:
            print(f"  [WARN] Colonna '{FOLD_COL}' assente -- creo fold casuali (solo TEST_MODE).")
            df = _make_temp_folds(df)
        else:
            raise ValueError(f"Colonna '{FOLD_COL}' mancante nel parquet.")
    return df


def _make_temp_folds(df: pd.DataFrame) -> pd.DataFrame:
    rng   = np.random.default_rng(RANDOM_SEED)
    label = df[LABEL_COL]
    is_p  = label.eq(1).fillna(False).values
    is_n  = label.eq(0).fillna(False).values
    is_u  = label.isna().values

    fold = np.full(len(df), -1, dtype=int)
    for mask in [is_p, is_n]:
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % N_OUTER_FOLDS

    u_idx = np.where(is_u)[0]
    n_u   = min(len(u_idx), U_PER_FOLD * N_OUTER_FOLDS)
    chosen = rng.choice(u_idx, size=n_u, replace=False)
    rng.shuffle(chosen)
    for j, i in enumerate(chosen):
        fold[i] = j % N_OUTER_FOLDS

    df = df.copy()
    df[FOLD_COL] = fold
    return df


def get_features(df: pd.DataFrame) -> list:
    drop = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    return [c for c in df.columns if c not in drop and df[c].dtype.kind in "iufb"]


def split_fold(df, features, outer_k):
    label   = df[LABEL_COL]
    fold_np = df[FOLD_COL].to_numpy(dtype=float, na_value=np.nan)

    is_p = label.eq(1).fillna(False).to_numpy(dtype=bool)
    is_n = label.eq(0).fillna(False).to_numpy(dtype=bool)
    is_u = label.isna().to_numpy(dtype=bool) & (fold_np >= 0)

    tr = fold_np != outer_k
    va = fold_np == outer_k

    X = df[features].to_numpy(dtype=float)

    return (X[is_p & tr], X[is_n & tr], X[is_u & tr],
            X[is_p & va], X[is_n & va], X[is_u & va])


def run_cv(df: pd.DataFrame, features: list) -> pd.DataFrame:
    rng          = np.random.default_rng(RANDOM_SEED)
    folds_to_run = [0] if TEST_MODE else list(range(N_OUTER_FOLDS))
    pi_to_run    = [0.02] if TEST_MODE else PI_GRID

    rows = []

    for outer_k in folds_to_run:
        print(f"\n[{_ts()}] -- Outer fold {outer_k+1}/{N_OUTER_FOLDS} --")
        (X_pos_tr, X_neg_tr, X_unl_tr,
         X_pos_va, X_neg_va, X_unl_va) = split_fold(df, features, outer_k)

        print(f"  Train: {len(X_pos_tr)} P | {len(X_neg_tr)} N | {len(X_unl_tr)} U")
        print(f"  Val:   {len(X_pos_va)} P | {len(X_neg_va)} N | {len(X_unl_va)} U")

        if len(X_pos_va) == 0 or len(X_unl_va) == 0:
            print("  [SKIP] fold vuoto.")
            continue

        t_scale = time.time()
        scaler = StandardScaler().fit(np.vstack([X_pos_tr, X_neg_tr, X_unl_tr]))
        X_pos_tr_s = scaler.transform(X_pos_tr)
        X_neg_tr_s = scaler.transform(X_neg_tr)
        X_unl_tr_s = scaler.transform(X_unl_tr)
        X_pos_va_s = scaler.transform(X_pos_va)
        X_neg_va_s = scaler.transform(X_neg_va)
        X_unl_va_s = scaler.transform(X_unl_va)
        print(f"  Scaling fatto in {int(time.time()-t_scale)}s")

        # Re-split fresco per inner CV (indipendente dai fold outer)
        n_p_tr = len(X_pos_tr_s)
        n_u_tr = len(X_unl_tr_s)
        n_n_tr = len(X_neg_tr_s)
        inner_fold_p = rng.permutation(n_p_tr) % N_INNER_FOLDS
        inner_fold_u = rng.permutation(n_u_tr) % N_INNER_FOLDS
        inner_fold_n = rng.permutation(n_n_tr) % N_INNER_FOLDS

        row = {"fold": outer_k}

        for pi in pi_to_run:
            t_pi = time.time()
            print(f"\n  [pi={pi}] inner CV su {N_INNER_FOLDS} fold x {len(LAMBDA_GRID)} lambda...")

            lam_scores = {lam: [] for lam in LAMBDA_GRID}

            for inner_val_k in range(N_INNER_FOLDS):
                t_ifold = time.time()
                mask_itr_p = inner_fold_p != inner_val_k
                mask_iva_p = inner_fold_p == inner_val_k
                mask_itr_u = inner_fold_u != inner_val_k
                mask_iva_u = inner_fold_u == inner_val_k
                mask_itr_n = inner_fold_n != inner_val_k
                mask_iva_n = inner_fold_n == inner_val_k

                X_p_itr = X_pos_tr_s[mask_itr_p]
                X_p_iva = X_pos_tr_s[mask_iva_p]
                X_u_itr = X_unl_tr_s[mask_itr_u]
                X_u_iva = X_unl_tr_s[mask_iva_u]
                X_n_itr = X_neg_tr_s[mask_itr_n]
                X_n_iva = X_neg_tr_s[mask_iva_n]

                if len(X_p_itr) == 0 or len(X_u_itr) == 0 or len(X_p_iva) == 0:
                    print(f"    inner fold {inner_val_k+1}/{N_INNER_FOLDS} [SKIP] vuoto")
                    continue

                for lam in LAMBDA_GRID:
                    w, b   = train(X_p_itr, X_u_itr, pi, lam, rng,
                                   X_neg=X_n_itr if PNPU else None, verbose=False)
                    sp_iva = predict(X_p_iva, w, b)
                    su_iva = predict(X_u_iva, w, b) if len(X_u_iva) > 0 else np.array([])
                    sn_iva = predict(X_n_iva, w, b) if len(X_n_iva) > 0 else np.array([])
                    lam_scores[lam].append(_lift1(sp_iva, sn_iva, su_iva))

                best_so_far = max(LAMBDA_GRID,
                                  key=lambda l: np.nanmean(lam_scores[l])
                                  if lam_scores[l] else -np.inf)
                print(f"    inner fold {inner_val_k+1}/{N_INNER_FOLDS}"
                      f"  P={len(X_p_itr)}+{len(X_p_iva)}"
                      f"  U={len(X_u_itr)}+{len(X_u_iva)}"
                      f"  best_lam={best_so_far:.2e}"
                      f"  ({int(time.time()-t_ifold)}s)")

            lam_star = max(LAMBDA_GRID,
                           key=lambda l: np.nanmean(lam_scores[l])
                           if lam_scores[l] else -np.inf)
            print(f"  => lam*={lam_star:.2e}  (inner CV: {int(time.time()-t_pi)}s)")

            print(f"  Outer train con lam*={lam_star:.2e}, pi={pi}...")
            w, b = train(X_pos_tr_s, X_unl_tr_s, pi, lam_star, rng,
                         X_neg=X_neg_tr_s if PNPU else None, verbose=True)

            sp = predict(X_pos_va_s, w, b)
            sn = predict(X_neg_va_s, w, b)
            su = predict(X_unl_va_s, w, b)
            metrics = eval_all(sp, sn, su)

            row[f"lift1_pi{pi}"]  = metrics["lift@1%"]
            row[f"lam_pi{pi}"]    = lam_star
            row[f"prauc_pi{pi}"]  = metrics.get("pr_auc",  np.nan)
            row[f"rocauc_pi{pi}"] = metrics.get("roc_auc", np.nan)
            elapsed = int(time.time() - t_pi)
            print(f"  lift@1%={metrics['lift@1%']:.3f}  pr_auc={metrics.get('pr_auc', float('nan')):.3f}"
                  f"  lam*={lam_star:.2e}  ({elapsed}s)")

        rows.append(row)

    return pd.DataFrame(rows)


def _lift1(sp, sn, su):
    if len(sp) == 0:
        return np.nan
    return lift_at_k(sp,
                     sn if len(sn) > 0 else np.array([]),
                     su if len(su) > 0 else np.array([]),
                     k=0.01)


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def retrain_and_save(df: pd.DataFrame, features: list, pi: float, lam: float):
    rng   = np.random.default_rng(RANDOM_SEED)
    label = df[LABEL_COL]
    fold  = df[FOLD_COL].values
    is_p  = label.eq(1).fillna(False).values
    is_n  = label.eq(0).fillna(False).values
    is_u  = label.isna().values & (fold >= 0)

    X = df[features].to_numpy(dtype=float)
    X_pos = X[is_p]; X_neg = X[is_n]; X_unl = X[is_u]

    scaler = StandardScaler().fit(np.vstack([X_pos, X_neg, X_unl]))
    variant = "pnpu" if PNPU else "pu"
    print(f"  Retrain {variant.upper()}: {len(X_pos)} P | {len(X_unl)} U | {len(X_neg)} N")
    w, b = train(scaler.transform(X_pos), scaler.transform(X_unl), pi, lam, rng,
                 X_neg=scaler.transform(X_neg) if PNPU else None, verbose=True)

    out = Path(OUT_DIR) / f"logit_M{MODEL_NUMBER}_{variant}_pi{pi}_lam{lam:.2e}.npz"
    np.savez(out, w=w, b=np.array([b]),
             scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             features=np.array(features))
    print(f"  [SAVED] {out}")


def main():
    t0 = time.time()
    variant = "PNPU" if PNPU else "PU puro"
    print("=" * 60)
    print(f"  nnPU + Logit [{variant}]  |  M{MODEL_NUMBER}  |  TEST_MODE={TEST_MODE}")
    print(f"  pi grid: {PI_GRID if not TEST_MODE else '[0.02]'}")
    print(f"  lambda grid: {len(LAMBDA_GRID)} values  |  inner folds: {N_INNER_FOLDS}")
    print("=" * 60)

    df       = load_data()
    features = get_features(df)
    print(f"  Feature: {len(features)}")

    if RETRAIN_FINAL:
        print(f"\n[RETRAIN] pi={PI_FINAL}  lambda={LAMBDA_FINAL}")
        retrain_and_save(df, features, PI_FINAL, LAMBDA_FINAL)
    else:
        results = run_cv(df, features)

        print("  MATRICE OOF - lift@1% per fold x pi")
        print(results.to_string(index=False))

        variant_tag = "pnpu" if PNPU else "pu"
        out_csv = Path(OUT_DIR) / f"logit_M{MODEL_NUMBER}_{variant_tag}_fold_metrics.csv"
        results.to_csv(out_csv, index=False)
        print(f"\n  [SAVED] {out_csv}")

    s = int(time.time() - t0)
    print(f"\n  Completato in {s//60}m {s%60}s.")


if __name__ == "__main__":
    main()
