"""
17_assign_folds.py
Pipeline step: assegna la colonna 'fold' (0-3 oppure NaN) ai parquet
preprocessed e nativi di M1, M2, M3.

Strategia labeled (P+N) — propagazione bottom-up M3 -> M2 -> M1:
  1. M3: assegnazione stratificata 4-fold su P e N separatamente.
  2. M2: fold ereditati da M3 via join CIG; CIG labeled esclusivi di M2
     ricevono assegnazione stratificata casuale.
  3. M1: fold ereditati da M2; CIG labeled esclusivi di M1 assegnati casualmente.
  => M3 (dataset piu piccolo) determina la base del bilanciamento.

Strategia U:
  Ogni dataset campiona U_PER_FOLD * N_FOLDS unlabeled (resto NaN).
  Assegnazione round-robin sul campione casuale.
    M1: 10_000 per fold  |  M2: 6_000 per fold  |  M3: 3_000 per fold

Seed search (in-memory):
  Carica i parquet una sola volta, poi testa piu seed in RAM.
  Sceglie il seed con minima max_deviation dei decili per feature continua
  tra i fold U. Soglia: 4%. Feature continue = nunique >= 20 tra gli U.

Output:
  Sovrascrive 'fold' in tutti e 6 i parquet (preprocessed + nativi).
  Native riceve la stessa mappatura CIG -> fold del preprocessed.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE       = Path(__file__).parent.parent
PREP_DIR   = BASE / "output" / "parquet" / "model" / "preprocessed"
NATIVE_DIR = BASE / "output" / "parquet" / "model" / "nativi"

LABEL_COL = "label"
FOLD_COL  = "fold"
CIG_COL   = "cig"

N_FOLDS    = 4
U_PER_FOLD = {1: 10_000, 2: 6_000, 3: 3_000}

N_DECILES      = 10
MAX_DEV_OK     = 0.10   # |fold_mean - global_mean| / global_std < 0.10 per ogni feature
MAX_SEED_TRIES = 30
SEED_START     = 42

def stratified_fold_assign(cig_arr: np.ndarray, label_arr: np.ndarray,
                            rng: np.random.Generator) -> dict:
    """Assegna fold 0..N_FOLDS-1 stratificato per label. Ritorna CIG -> fold."""
    mapping = {}
    for lv in [1, 0]:
        idx = cig_arr[label_arr == lv]
        perm = rng.permutation(len(idx))
        for j, pos in enumerate(perm):
            mapping[idx[pos]] = j % N_FOLDS
    return mapping


def build_cig_map(dfs: dict, rng: np.random.Generator) -> dict:
    """
    Propagazione bottom-up M3 -> M2 -> M1.
    dfs: {1: df_M1, 2: df_M2, 3: df_M3} (solo colonne cig e label).
    """
    global_map = {}
    for mn in [3, 2, 1]:
        df = dfs[mn]
        labeled = df[df[LABEL_COL].notna()]
        new = labeled[~labeled[CIG_COL].isin(global_map)]
        if len(new):
            new_map = stratified_fold_assign(
                new[CIG_COL].to_numpy(), new[LABEL_COL].to_numpy(dtype=float), rng)
            global_map.update(new_map)
    return global_map


def assign_u_folds(df: pd.DataFrame, u_per_fold: int,
                   rng: np.random.Generator) -> pd.Series:
    """Campiona U_PER_FOLD * N_FOLDS unlabeled, assegna fold round-robin. Resto NaN."""
    is_u    = df[LABEL_COL].isna()
    u_idx   = df.index[is_u].to_numpy()
    n_samp  = min(len(u_idx), u_per_fold * N_FOLDS)
    chosen  = rng.choice(u_idx, size=n_samp, replace=False)
    rng.shuffle(chosen)
    fold = pd.Series(pd.NA, index=df.index, dtype="Int8")
    fold[chosen] = (np.arange(n_samp, dtype="int8") % N_FOLDS)
    return fold


def make_fold_series(df: pd.DataFrame, cig_map: dict, u_per_fold: int,
                     rng: np.random.Generator) -> pd.Series:
    """Combina labeled fold (da cig_map) + U fold (campionati)."""
    fold = assign_u_folds(df, u_per_fold, rng)
    is_labeled = df[LABEL_COL].notna()
    fold[is_labeled] = df.loc[is_labeled, CIG_COL].map(cig_map).astype("Int8")
    return fold


def u_fold_maxdev(df: pd.DataFrame, fold: pd.Series) -> float:
    """
    Per ogni feature continua (nuniq>=20 tra gli U assegnati), calcola
    max_k |mean_fold_k - mean_global| / std_global.
    Valori bassi (<0.1) indicano distribuzione uniforme tra fold.
    Robusto a skewness e ties (non usa decili).
    """
    is_u_assigned = df[LABEL_COL].isna() & fold.notna()
    if is_u_assigned.sum() < N_FOLDS * 10:
        return 0.0

    cont_cols = [c for c in df.columns
                 if df[c].dtype.kind in "iuf"
                 and c not in (LABEL_COL, FOLD_COL)
                 and df.loc[is_u_assigned, c].nunique() >= 20]
    if not cont_cols:
        return 0.0

    max_dev = 0.0
    for col in cont_cols:
        vals = df.loc[is_u_assigned, col].dropna()
        g_mean = vals.mean()
        g_std  = vals.std()
        if g_std == 0:
            continue
        for k in range(N_FOLDS):
            mask_k = is_u_assigned & (fold == k)
            fv = df.loc[mask_k, col].dropna()
            if len(fv) == 0:
                continue
            dev = abs(fv.mean() - g_mean) / g_std
            if dev > max_dev:
                max_dev = dev

    return max_dev


def main():
    print("Caricamento parquet (una volta)...")
    dfs_prep   = {mn: pd.read_parquet(PREP_DIR   / f"M{mn}.parquet") for mn in [1, 2, 3]}
    dfs_native = {mn: pd.read_parquet(NATIVE_DIR / f"M{mn}.parquet") for mn in [1, 2, 3]}
    print("  Caricati.")

    # Subset leggero per build_cig_map (solo cig + label)
    dfs_light = {mn: dfs_prep[mn][[CIG_COL, LABEL_COL]] for mn in [1, 2, 3]}

    best_seed   = SEED_START
    best_maxdev = np.inf
    best_map    = None
    best_folds  = None

    print("Seed search...")
    for attempt in range(MAX_SEED_TRIES):
        seed = SEED_START + attempt
        rng  = np.random.default_rng(seed)
        cig_map = build_cig_map(dfs_light, rng)

        folds_this  = {}
        maxdevs     = []
        for mn in [1, 2, 3]:
            rng_u = np.random.default_rng(seed + mn * 1000)
            f = make_fold_series(dfs_prep[mn], cig_map, U_PER_FOLD[mn], rng_u)
            folds_this[mn] = f
            maxdevs.append(u_fold_maxdev(dfs_prep[mn], f))

        worst = max(maxdevs)
        marker = " *" if worst < best_maxdev else ""
        print(f"  seed={seed}  maxdev=[{maxdevs[0]:.3f},{maxdevs[1]:.3f},{maxdevs[2]:.3f}]"
              f"  worst={worst:.3f}{marker}")

        if worst < best_maxdev:
            best_maxdev = worst
            best_seed   = seed
            best_map    = cig_map
            best_folds  = folds_this

        if worst <= MAX_DEV_OK:
            print(f"=> seed={seed} accettato (maxdev={worst:.3f})")
            break
    else:
        print(f"=> Nessun seed sotto soglia. Uso seed={best_seed} (maxdev={best_maxdev:.3f})")

    # Rigenera U con best_seed (labeled map e folds gia pronti)
    folds = best_folds
    cig_map = best_map

    # Salva
    print("\nSalvataggio...")
    for mn in [1, 2, 3]:
        fold = folds[mn]
        lab  = dfs_prep[mn][LABEL_COL]

        p_counts = [int((fold[lab == 1] == k).sum()) for k in range(N_FOLDS)]
        n_counts = [int((fold[lab == 0] == k).sum()) for k in range(N_FOLDS)]
        u_tot    = int(fold[lab.isna()].notna().sum())
        md       = u_fold_maxdev(dfs_prep[mn], fold)
        print(f"  M{mn}  P={p_counts}  N={n_counts}  U={u_tot}  maxdev={md:.3f}")

        # preprocessed
        dfs_prep[mn][FOLD_COL] = fold
        dfs_prep[mn].to_parquet(PREP_DIR / f"M{mn}.parquet", index=False)

        # native: stessa mappatura CIG -> fold
        cig_to_fold = dict(zip(dfs_prep[mn][CIG_COL].values, fold.values))
        dfs_native[mn][FOLD_COL] = dfs_native[mn][CIG_COL].map(cig_to_fold).astype("Int8")
        dfs_native[mn].to_parquet(NATIVE_DIR / f"M{mn}.parquet", index=False)

        print(f"    Salvati M{mn} preprocessed + native")

    print("\nDone.")


if __name__ == "__main__":
    main()
