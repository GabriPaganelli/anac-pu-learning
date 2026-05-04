"""
nnPU Risk Estimator + Rete Neurale (MLP)
PNPU=True : loss estesa con termine N certi, normalizzato per n_unl_total.
PNPU=False: loss nnPU pura; N certi esclusi dal training, usati solo in val.

Architettura: MLP con BatchNorm + ReLU + Dropout(0.3).
  DEEP=False → [128, 64];  DEEP=True → [128, 128, 64].
Ottimizzatore: AdamW con CosineAnnealingLR (T_max=N_EPOCHS).
Iperparametro: weight_decay (WD_GRID, selezionato via inner CV su lift@1%).
Dataset: preprocessed (no NA). Label: 1=P, 0=N, NaN=U.
"""

MODEL_NUMBER   = 1          # 1=M1, 2=M2, 3=M3
PNPU           = True       # True = PNPU (N certi nel training); False = nnPU puro
TEST_MODE      = False      # True → smoke test rapido
RETRAIN_FINAL  = False      # True → riallena su tutto il dataset con (pi*, wd*)
PI_FINAL       = None
WD_FINAL       = None

from pathlib import Path as _P
BASE_PATH = str(_P(__file__).resolve().parents[3] / "anac" / "output" / "parquet" / "model" / "preprocessed")
OUT_DIR   = str(_P(__file__).resolve().parent / "results")

LABEL_COL    = "label"
FOLD_COL     = "fold"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]

N_OUTER_FOLDS  = 4
N_INNER_FOLDS  = 3     # budget ridotto rispetto a logit
N_INNER_EPOCHS = 40    # epoche inner CV (vs N_EPOCHS outer)

PI_GRID = [0.02]

import numpy as _np
WD_GRID = (
    [1e-4, 1e-2]                      if TEST_MODE else
    list(_np.logspace(-5, 1, 10))
)

# Architettura
DEEP    = False   # True → [128, 128, 64];  False → [128, 64]
DROPOUT = 0.3

# Ottimizzazione
LR_INIT      = 1e-3
BATCH_SIZE_U = 64   if TEST_MODE else 512
N_EPOCHS     = 3    if TEST_MODE else 150
INFER_BATCH  = 4_096

U_PER_FOLD   = 200   # solo TEST_MODE
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

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  [device] Apple MPS (Metal)")
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  [device] CUDA -- {torch.cuda.get_device_name(0)}")
        return dev
    print("  [device] CPU")
    return torch.device("cpu")


class NNModel(nn.Module):
    """
    MLP con BatchNorm + ReLU + Dropout per ogni hidden layer.
    Output: logit scalare (no sigmoid — usare BCEWithLogitsLoss).
    """
    def __init__(self, n_feat: int):
        super().__init__()
        hidden = [128, 128, 64] if DEEP else [128, 64]
        layers = []
        in_dim = n_feat
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def nnpu_risk_pnpu(logits_pos: torch.Tensor,
                   logits_unl: torch.Tensor,
                   logits_neg: torch.Tensor,
                   pi: float,
                   n_unl_total: int) -> torch.Tensor:
    """
    PNPU: R_PNPU = pi*R_P(+) + max(0, R_U(-) - pi*R_P(-)) + (1-pi)*R_N(-)

    Termine N normalizzato per n_unl_total (dimensione pool U del fold):
    ogni esempio N contribuisce ~1/n_unl_total per step, stesso ordine degli U.
    Quando neg_term < 0: solo P e N attivi (termine U invertito).
    """
    ones    = torch.ones_like(logits_pos)
    zeros_p = torch.zeros_like(logits_pos)
    zeros_u = torch.zeros_like(logits_unl)
    zeros_n = torch.zeros_like(logits_neg)

    r_pos_pos = F.binary_cross_entropy_with_logits(logits_pos, ones)
    r_pos_neg = F.binary_cross_entropy_with_logits(logits_pos, zeros_p)
    r_unl_neg = F.binary_cross_entropy_with_logits(logits_unl, zeros_u)

    r_neg_sum = F.binary_cross_entropy_with_logits(logits_neg, zeros_n, reduction="sum")
    n_term    = (1.0 - pi) * r_neg_sum / n_unl_total

    neg_term = r_unl_neg - pi * r_pos_neg

    if neg_term.item() >= 0.0:
        return pi * r_pos_pos + neg_term + n_term
    else:
        return pi * r_pos_pos + n_term


def nnpu_risk_pu(logits_pos: torch.Tensor,
                 logits_unl: torch.Tensor,
                 pi: float) -> torch.Tensor:
    """
    nnPU puro (Kiryo 2017): R_nnPU = pi*R_P(+) + max(0, R_U(-) - pi*R_P(-))

    Quando neg_term < 0: inversione gradiente, usa solo pi*R_P(+).
    """
    ones    = torch.ones_like(logits_pos)
    zeros_p = torch.zeros_like(logits_pos)
    zeros_u = torch.zeros_like(logits_unl)

    r_pos_pos = F.binary_cross_entropy_with_logits(logits_pos, ones)
    r_pos_neg = F.binary_cross_entropy_with_logits(logits_pos, zeros_p)
    r_unl_neg = F.binary_cross_entropy_with_logits(logits_unl, zeros_u)

    neg_term = r_unl_neg - pi * r_pos_neg

    if neg_term.item() >= 0.0:
        return pi * r_pos_pos + neg_term
    else:
        return pi * r_pos_pos


def train_nn(X_pos: np.ndarray,
             X_unl: np.ndarray,
             pi: float,
             wd: float,
             device: torch.device,
             rng: np.random.Generator,
             n_epochs: int,
             X_neg: np.ndarray = None,
             verbose: bool = False) -> NNModel:
    """
    Allena NNModel con PNPU o nnPU loss secondo flag PNPU.
    X_neg obbligatorio quando PNPU=True.
    Batch: tutti i P (+N in PNPU) + BATCH_SIZE_U unlabeled per step.
    Restituisce il modello in eval mode.
    """
    if PNPU and (X_neg is None or len(X_neg) == 0):
        raise ValueError("X_neg obbligatorio per PNPU.")

    n_feat      = X_pos.shape[1]
    n_unl_total = len(X_unl)

    model = NNModel(n_feat).to(device)
    model.train()

    T_pos = torch.tensor(X_pos, dtype=torch.float32, device=device)
    if PNPU:
        T_neg = torch.tensor(X_neg, dtype=torch.float32, device=device)

    opt   = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=0.0)

    n_unl       = len(X_unl)
    print_every = max(1, n_epochs // 5)
    t_train     = time.time()

    for epoch in range(n_epochs):
        perm = rng.permutation(n_unl)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n_unl, BATCH_SIZE_U):
            batch_idx = perm[i : i + BATCH_SIZE_U]
            if len(batch_idx) == 0:
                continue

            T_unl = torch.tensor(X_unl[batch_idx], dtype=torch.float32, device=device)

            opt.zero_grad()
            logits_pos = model(T_pos)
            logits_unl = model(T_unl)

            if PNPU:
                logits_neg = model(T_neg)
                loss = nnpu_risk_pnpu(logits_pos, logits_unl, logits_neg, pi, n_unl_total)
            else:
                loss = nnpu_risk_pu(logits_pos, logits_unl, pi)

            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()

        if verbose and (epoch + 1) % print_every == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            lr_now   = sched.get_last_lr()[0]
            elapsed  = int(time.time() - t_train)
            print(f"    epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}  ({elapsed}s)")

    model.eval()
    return model


def predict_nn(model: NNModel, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Score probabilistici (sigmoid del logit). Batched per memoria."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), INFER_BATCH):
            xb = torch.tensor(X[i : i + INFER_BATCH], dtype=torch.float32, device=device)
            parts.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(parts) if parts else np.array([])


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

    u_idx  = np.where(is_u)[0]
    n_u    = min(len(u_idx), U_PER_FOLD * N_OUTER_FOLDS)
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


def _lift1(sp, sn, su):
    if len(sp) == 0:
        return np.nan
    return lift_at_k(sp,
                     sn if len(sn) > 0 else np.array([]),
                     su if len(su) > 0 else np.array([]),
                     k=0.01)


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def run_cv(df: pd.DataFrame, features: list, device: torch.device) -> pd.DataFrame:
    """
    Outer CV. Iperparametro selezionato via inner CV: weight_decay (WD_GRID).
    PNPU=True: N certi inclusi nel training con termine gradiente separato.
    """
    rng          = np.random.default_rng(RANDOM_SEED)
    folds_to_run = [0]     if TEST_MODE else list(range(N_OUTER_FOLDS))
    pi_to_run    = [0.02]  if TEST_MODE else PI_GRID
    rows         = []

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
        scaler  = StandardScaler().fit(np.vstack([X_pos_tr, X_neg_tr, X_unl_tr]))
        X_pos_tr_s = scaler.transform(X_pos_tr)
        X_neg_tr_s = scaler.transform(X_neg_tr)
        X_unl_tr_s = scaler.transform(X_unl_tr)
        X_pos_va_s = scaler.transform(X_pos_va)
        X_neg_va_s = scaler.transform(X_neg_va)
        X_unl_va_s = scaler.transform(X_unl_va)
        print(f"  Scaling in {int(time.time()-t_scale)}s")

        n_p_tr = len(X_pos_tr_s)
        n_u_tr = len(X_unl_tr_s)
        n_n_tr = len(X_neg_tr_s)
        inner_fold_p = rng.permutation(n_p_tr) % N_INNER_FOLDS
        inner_fold_u = rng.permutation(n_u_tr) % N_INNER_FOLDS
        inner_fold_n = rng.permutation(n_n_tr) % N_INNER_FOLDS

        row = {"fold": outer_k}

        for pi in pi_to_run:
            t_pi = time.time()
            print(f"\n  [pi={pi}] inner CV su {N_INNER_FOLDS} fold x {len(WD_GRID)} wd ...")

            wd_scores = {wd: [] for wd in WD_GRID}

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
                if PNPU and len(X_n_itr) == 0:
                    print(f"    inner fold {inner_val_k+1}/{N_INNER_FOLDS} [SKIP] N vuoto")
                    continue

                for wd in WD_GRID:
                    model = train_nn(X_p_itr, X_u_itr, pi, wd, device, rng,
                                     n_epochs=N_INNER_EPOCHS,
                                     X_neg=X_n_itr if PNPU else None,
                                     verbose=False)
                    sp_iva = predict_nn(model, X_p_iva, device)
                    su_iva = predict_nn(model, X_u_iva, device) if len(X_u_iva) > 0 else np.array([])
                    sn_iva = predict_nn(model, X_n_iva, device) if len(X_n_iva) > 0 else np.array([])
                    wd_scores[wd].append(_lift1(sp_iva, sn_iva, su_iva))
                    del model

                best_so_far = max(WD_GRID,
                                  key=lambda w: np.nanmean(wd_scores[w])
                                  if wd_scores[w] else -np.inf)
                print(f"    inner fold {inner_val_k+1}/{N_INNER_FOLDS}"
                      f"  P={len(X_p_itr)}+{len(X_p_iva)}"
                      f"  U={len(X_u_itr)}+{len(X_u_iva)}"
                      f"  N={len(X_n_itr)}+{len(X_n_iva)}"
                      f"  best_wd={best_so_far:.2e}"
                      f"  ({int(time.time()-t_ifold)}s)")

            wd_star = max(WD_GRID,
                          key=lambda w: np.nanmean(wd_scores[w])
                          if wd_scores[w] else -np.inf)
            print(f"  => wd*={wd_star:.2e}  (inner CV: {int(time.time()-t_pi)}s)")

            print(f"  Outer train con wd*={wd_star:.2e}, pi={pi}...")
            model = train_nn(X_pos_tr_s, X_unl_tr_s, pi, wd_star, device, rng,
                             n_epochs=N_EPOCHS,
                             X_neg=X_neg_tr_s if PNPU else None,
                             verbose=True)

            sp = predict_nn(model, X_pos_va_s, device)
            sn = predict_nn(model, X_neg_va_s, device)
            su = predict_nn(model, X_unl_va_s, device)
            metrics = eval_all(sp, sn, su)
            del model

            row[f"lift1_pi{pi}"]  = metrics["lift@1%"]
            row[f"wd_pi{pi}"]     = wd_star
            row[f"prauc_pi{pi}"]  = metrics.get("pr_auc",  np.nan)
            row[f"rocauc_pi{pi}"] = metrics.get("roc_auc", np.nan)
            elapsed = int(time.time() - t_pi)
            print(f"  lift@1%={metrics['lift@1%']:.3f}"
                  f"  pr_auc={metrics.get('pr_auc', float('nan')):.3f}"
                  f"  wd*={wd_star:.2e}  ({elapsed}s)")

        rows.append(row)

    return pd.DataFrame(rows)


def retrain_and_save(df: pd.DataFrame, features: list,
                     pi: float, wd: float, device: torch.device):
    rng   = np.random.default_rng(RANDOM_SEED)
    label = df[LABEL_COL]
    fold  = df[FOLD_COL].values
    is_p  = label.eq(1).fillna(False).values
    is_n  = label.eq(0).fillna(False).values
    is_u  = label.isna().values & (fold >= 0)

    X     = df[features].to_numpy(dtype=float)
    X_pos = X[is_p]; X_neg = X[is_n]; X_unl = X[is_u]

    scaler = StandardScaler().fit(np.vstack([X_pos, X_neg, X_unl]))
    variant = "pnpu" if PNPU else "pu"
    print(f"  Retrain {variant.upper()}: {len(X_pos)} P | {len(X_neg)} N | {len(X_unl)} U")
    model = train_nn(scaler.transform(X_pos), scaler.transform(X_unl),
                     pi, wd, device, rng, n_epochs=N_EPOCHS,
                     X_neg=scaler.transform(X_neg) if PNPU else None,
                     verbose=True)

    out = Path(OUT_DIR) / f"nn_M{MODEL_NUMBER}_{variant}_pi{pi}_wd{wd:.2e}.pt"
    torch.save({
        "model_state":  model.state_dict(),
        "scaler_mean":  scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features":     features,
        "pi":           pi,
        "wd":           wd,
        "deep":         DEEP,
        "dropout":      DROPOUT,
    }, out)
    print(f"  [SAVED] {out}")


def main():
    t0 = time.time()
    variant = "PNPU" if PNPU else "PU puro"
    print("=" * 60)
    print(f"  nnPU + NN [{variant}]  |  M{MODEL_NUMBER}  |  TEST_MODE={TEST_MODE}")
    print(f"  DEEP={DEEP}  DROPOUT={DROPOUT}  BATCH_U={BATCH_SIZE_U}  EPOCHS={N_EPOCHS}")
    print(f"  pi grid: {PI_GRID if not TEST_MODE else '[0.02]'}")
    print(f"  wd grid: {len(WD_GRID)} values  |  inner folds: {N_INNER_FOLDS}  inner epochs: {N_INNER_EPOCHS}")
    print("=" * 60)

    device   = get_device()
    df       = load_data()
    features = get_features(df)
    print(f"  Feature: {len(features)}")

    if RETRAIN_FINAL:
        print(f"\n[RETRAIN] pi={PI_FINAL}  wd={WD_FINAL}")
        retrain_and_save(df, features, PI_FINAL, WD_FINAL, device)
    else:
        results = run_cv(df, features, device)

        print("  MATRICE OOF - lift@1% per fold x pi")
        print(results.to_string(index=False))

        variant_tag = "pnpu" if PNPU else "pu"
        out_csv = Path(OUT_DIR) / f"nn_M{MODEL_NUMBER}_{variant_tag}_fold_metrics.csv"
        results.to_csv(out_csv, index=False)
        print(f"\n  [SAVED] {out_csv}")

    s = int(time.time() - t0)
    print(f"\n  Completato in {s//60}m {s%60}s.")


if __name__ == "__main__":
    main()
