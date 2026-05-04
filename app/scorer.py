"""
app/scorer.py — Training + scoring wrapper per il risk scoring on-the-fly.

API pubblica:
  load_stage_data(stage, active_features, data_dir) → pd.DataFrame
  encode_contract(values_dict, active_features, encodings) → np.ndarray
  compute_u_size(n_P, pi) → int
  run_scoring(df, active_features, contract_vec, cat_names, stage,
              progress_cb, return_model)
      → dict(rank_pct, n_contracts, n_P, n_N, n_U_train, best_iter)
  run_scoring_batch(df, active_features, contract_vecs, cat_names, stage)
      → list[dict]   # un dict per contratto, stesso schema di run_scoring
  run_bootstrap(df, active_features, contract_vec, cat_names, stage,
                n_boot, n_workers, progress_cb)
      → (rank_low, rank_high)   # percentile 5° e 95° della distribuzione bootstrap

Design note: il rank_pct è calcolato sui raw score (senza Platt).
Il rango è invariante a trasformazioni monotone.

Bootstrap: parallelizzato via ThreadPoolExecutor. LightGBM rilascia il GIL
durante il training C++, quindi il threading produce vera concorrenza senza
oversubscription quando num_threads=1 per worker.
Percentile 5°–95° della distribuzione bootstrap sul campione di riferimento fisso.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from re_lgbm import fit_predict, PI_BY_MODEL


GAMMA          = 1.0
U_FLOOR        = 3_000
U_CAP          = 50_000
BOOTSTRAP_SEED = 2028



def load_stage_data(stage: int, active_features: list[str],
                    data_dir: Path) -> pd.DataFrame:
    """Carica M{stage}_ready.parquet con solo le colonne necessarie."""
    path = data_dir / f"M{stage}_ready.parquet"
    cols = ["label"] + active_features
    available = pq.read_schema(str(path)).names
    cols_ok   = [c for c in cols if c in available]
    missing   = [c for c in active_features if c not in available]
    if missing:
        print(f"  [WARN] Colonne non nel parquet M{stage}: {missing}")
    return pd.read_parquet(path, columns=cols_ok)



def compute_u_size(n_P: int, pi: float = 0.02) -> int:
    """n_U = |P| / π (Sakai et al. 2017), bounded in [U_FLOOR, U_CAP]."""
    return int(min(U_CAP, max(U_FLOOR, round(n_P / pi))))



def encode_contract(
    values_dict: dict,
    active_features: list[str],
    encodings: dict,
) -> np.ndarray:
    """Converte il dizionario {feature: valore} in un vettore float32.

    Per le colonne categoriali (presenti in encodings):
      - valore → indice nella lista sorted_unique_values → float32 code
      - valore sconosciuto o float('nan') → NaN (LightGBM missing nativo)
    Per le colonne numeriche:
      - float('nan') → np.nan (segnale informativo esplicito)
      - passaggio diretto a float32
    Precondizione: active_features contiene solo feature con values_dict[f] is not None
    (garantito da _build_active_features).
    """
    vec = np.full(len(active_features), np.nan, dtype=np.float32)
    for i, feat in enumerate(active_features):
        val = values_dict.get(feat, None)
        if feat in encodings:
            try:
                vec[i] = float(encodings[feat].index(str(val)))
            except ValueError:
                vec[i] = np.nan
        else:
            try:
                vec[i] = float(val)
            except (TypeError, ValueError):
                vec[i] = np.nan
    return vec



def _build_training_arrays(
    df: pd.DataFrame,
    active_features: list[str],
    u_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = df["label"].to_numpy(dtype=float, na_value=np.nan)
    X     = df[active_features].to_numpy(dtype=np.float32)

    mask_P = label == 1
    mask_N = label == 0
    mask_U = np.isnan(label)

    X_P = X[mask_P]
    X_N = X[mask_N]

    u_idx = np.where(mask_U)[0]
    if len(u_idx) > u_size:
        u_idx = rng.choice(u_idx, size=u_size, replace=False)
    X_U = X[u_idx]

    return X_P, X_N, X_U


def _cat_idx(cat_names: list[str], active_features: list[str]) -> list[int]:
    return [active_features.index(c) for c in cat_names if c in active_features]



def run_scoring(
    df: pd.DataFrame,
    active_features: list[str],
    contract_vec: np.ndarray,
    cat_names: list[str],
    stage: int,
    progress_cb: Optional[Callable[[str], None]] = None,
    return_model: bool = False,
) -> dict:
    """Allena il modello e calcola rank_pct del contratto target."""
    def _cb(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

    pi    = PI_BY_MODEL[stage]
    rng   = np.random.default_rng(42)

    label = df["label"].to_numpy(dtype=float, na_value=np.nan)
    n_P   = int((label == 1).sum())
    n_N   = int((label == 0).sum())
    n_U_available = int(np.isnan(label).sum())
    u_size = compute_u_size(n_P, pi)
    u_size = min(u_size, n_U_available)

    missing_in_df = [f for f in active_features if f not in df.columns]
    if missing_in_df:
        raise ValueError(
            f"Feature richieste non presenti nel parquet M{stage}: {missing_in_df}. "
            "Eseguire setup.py per rigenerare i dati."
        )

    _cb(f"Training M{stage}: {n_P} P, {n_N} N, {u_size} U  (pi={pi}, gamma={GAMMA})")

    X_P, X_N, X_U = _build_training_arrays(df, active_features, u_size, rng)
    cat_i = _cat_idx(cat_names, active_features)

    _cb("Training e scoring distribuzione di riferimento...")
    X_all   = df[active_features].to_numpy(dtype=np.float32)
    X_te    = contract_vec.reshape(1, -1)
    X_score = np.vstack([X_te, X_all])

    result_fit = fit_predict(
        X_P, X_N, X_U, X_score,
        gamma=GAMMA, pi=pi,
        features=active_features, cat_idx=cat_i if cat_i else None,
        rng=rng,
        return_model=return_model,
    )
    if return_model:
        scores, best_iter, booster = result_fit
    else:
        scores, best_iter = result_fit
        booster = None

    score_target = scores[0]
    scores_all   = scores[1:]
    rank_pct = float(np.mean(scores_all < score_target) * 100)

    _cb(f"rank_pct = {rank_pct:.1f}%  (best_iter={best_iter})")

    out = dict(
        rank_pct    = rank_pct,
        n_contracts = len(df),
        n_P         = n_P,
        n_N         = n_N,
        n_U_train   = u_size,
        best_iter   = best_iter,
        score_raw   = float(score_target),
    )
    if return_model:
        out["booster"] = booster
    return out



def run_scoring_batch(
    df: pd.DataFrame,
    active_features: list[str],
    contract_vecs: list[np.ndarray],
    cat_names: list[str],
    stage: int,
) -> list[dict]:
    """Allena il modello una volta e calcola rank_pct per ciascun contratto.

    Tutti i contratti in contract_vecs devono avere lo stesso active_features.
    Speedup rispetto a N chiamate a run_scoring: training eseguito una sola volta,
    tutti i vettori scorati in un'unica predict call.
    """
    pi  = PI_BY_MODEL[stage]
    rng = np.random.default_rng(42)

    label = df["label"].to_numpy(dtype=float, na_value=np.nan)
    n_P   = int((label == 1).sum())
    n_N   = int((label == 0).sum())
    n_U_available = int(np.isnan(label).sum())
    u_size = min(compute_u_size(n_P, pi), n_U_available)

    missing_in_df = [f for f in active_features if f not in df.columns]
    if missing_in_df:
        raise ValueError(
            f"Feature richieste non presenti nel parquet M{stage}: {missing_in_df}. "
            "Eseguire setup.py per rigenerare i dati."
        )

    X_P, X_N, X_U = _build_training_arrays(df, active_features, u_size, rng)
    cat_i = _cat_idx(cat_names, active_features)

    X_all     = df[active_features].to_numpy(dtype=np.float32)
    n_targets = len(contract_vecs)
    X_targets = np.vstack([v.reshape(1, -1) for v in contract_vecs])
    X_score   = np.vstack([X_targets, X_all])   # (n_targets + N_ref) × F

    scores, best_iter = fit_predict(
        X_P, X_N, X_U, X_score,
        gamma=GAMMA, pi=pi,
        features=active_features, cat_idx=cat_i if cat_i else None,
        rng=rng,
    )

    scores_targets = scores[:n_targets]
    scores_ref     = scores[n_targets:]

    return [
        dict(
            rank_pct    = float(np.mean(scores_ref < s) * 100),
            n_contracts = len(df),
            n_P         = n_P,
            n_N         = n_N,
            n_U_train   = u_size,
            best_iter   = best_iter,
        )
        for s in scores_targets
    ]



def _bootstrap_single(
    b: int,
    rng_seed: int,
    X: np.ndarray,
    X_score: np.ndarray,          # np.vstack([X_te, X_ref]) — pre-calcolato
    idx_P: np.ndarray,
    idx_N: np.ndarray,
    idx_U: np.ndarray,
    u_size: int,
    gamma: float,
    pi: float,
    active_features: list,
    cat_i: list,
    lgbm_threads: int,
) -> float:
    """Singola iterazione bootstrap — eseguibile in thread parallelo."""
    rng_b = np.random.default_rng(rng_seed + b)

    idx_P_boot = rng_b.choice(idx_P, size=len(idx_P), replace=True)
    idx_N_boot = rng_b.choice(idx_N, size=len(idx_N), replace=True)
    idx_U_boot = rng_b.choice(idx_U, size=u_size,     replace=True)

    X_P_b = X[idx_P_boot]
    X_N_b = X[idx_N_boot]
    X_U_b = X[idx_U_boot]

    scores_b, _ = fit_predict(
        X_P_b, X_N_b, X_U_b, X_score,
        gamma=gamma, pi=pi,
        features=active_features, cat_idx=cat_i if cat_i else None,
        rng=rng_b,
        num_threads=lgbm_threads,
    )
    score_target_b = scores_b[0]
    scores_ref_b   = scores_b[1:]
    return float(np.mean(scores_ref_b < score_target_b) * 100)


def run_bootstrap(
    df: pd.DataFrame,
    active_features: list[str],
    contract_vec: np.ndarray,
    cat_names: list[str],
    stage: int,
    n_boot: int = 50,
    n_workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> tuple[float, float]:
    """Bootstrap CI sul rank_pct del contratto target.

    Per ogni iterazione b:
      1. Ricampiona P, N, U con rimpiazzo
      2. Allena il modello sul campione ricampionato
      3. Calcola il rank del contratto contro tutta la popolazione di df (P+N+U)

    La popolazione di riferimento per il rank è identica a quella di run_scoring
    (tutte le righe di df), così il CI è coerente col punto stimato.

    Parallelizzazione: ThreadPoolExecutor con num_threads=1 per LightGBM.
    LightGBM rilascia il GIL durante il training C++, quindi i thread
    girano in vera concorrenza senza pickle overhead (a differenza dei processi).
    n_workers=None → auto-detect da os.cpu_count().

    Restituisce (percentile_5, percentile_95) del rank_pct bootstrap.
    """
    pi    = PI_BY_MODEL[stage]
    label = df["label"].to_numpy(dtype=float, na_value=np.nan)
    X     = df[active_features].to_numpy(dtype=np.float32)
    cat_i = _cat_idx(cat_names, active_features)

    idx_P = np.where(label == 1)[0]
    idx_N = np.where(label == 0)[0]
    idx_U = np.where(np.isnan(label))[0]

    n_P    = len(idx_P)
    u_size = min(compute_u_size(n_P, pi), len(idx_U))

    X_te = contract_vec.reshape(1, -1)

    # X_score è costante per tutte le iterazioni: target + intera popolazione df
    # (stesso riferimento di run_scoring → CI coerente col punto stimato)
    X_score_boot = np.vstack([X_te, X])

    cpu_count       = os.cpu_count() or 1
    n_workers_actual = min(n_boot, n_workers if n_workers is not None else cpu_count)
    lgbm_threads    = max(1, cpu_count // n_workers_actual)

    ranks: list[float] = []

    if n_workers_actual <= 1:
        # Path sequenziale: progress callback per iterazione
        for b in range(n_boot):
            ranks.append(_bootstrap_single(
                b, BOOTSTRAP_SEED, X, X_score_boot,
                idx_P, idx_N, idx_U, u_size,
                GAMMA, pi, active_features, cat_i, lgbm_threads,
            ))
            if progress_cb:
                progress_cb(b + 1, n_boot)
    else:
        # Path parallelo: ThreadPoolExecutor, progress via as_completed (main thread)
        with ThreadPoolExecutor(max_workers=n_workers_actual) as executor:
            futures = {
                executor.submit(
                    _bootstrap_single,
                    b, BOOTSTRAP_SEED, X, X_score_boot,
                    idx_P, idx_N, idx_U, u_size,
                    GAMMA, pi, active_features, cat_i, lgbm_threads,
                ): b
                for b in range(n_boot)
            }
            for fut in as_completed(futures):
                ranks.append(fut.result())
                if progress_cb:
                    progress_cb(len(ranks), n_boot)

    ranks_arr = np.array(ranks)
    return float(np.percentile(ranks_arr, 5)), float(np.percentile(ranks_arr, 95))



def run_shap(
    booster,
    contract_vec: np.ndarray,
    active_features: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Contributi SHAP (TreeSHAP built-in LightGBM) per il contratto target."""
    X = contract_vec.reshape(1, -1).astype(np.float32)
    contrib = booster.predict(X, pred_contrib=True)   # (1, F+1)
    sv = contrib[0, :-1]                               # (F,)
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    return [(active_features[i], float(sv[i])) for i in order]
