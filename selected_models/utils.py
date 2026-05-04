"""
Utilities condivise per i modelli finalisti — PNU Mixing (Roadmap §5).

Responsabilità:
  - Caricamento parquet (nativi e preprocessed)
  - Reshuffle stratificato dei fold (seed separato dal benchmark originale)
  - Applicazione del fold-map per CIG al parquet preprocessed (PUET)
  - Split fold → array numpy
  - Wrapper metriche
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

_ROOT             = Path(__file__).resolve().parents[1]
NATIVI_PATH       = str(_ROOT / "anac" / "output" / "parquet" / "model" / "nativi")
PREPROCESSED_PATH = str(_ROOT / "anac" / "output" / "parquet" / "model" / "preprocessed")

# Aggiunge la root del progetto al path per importare metrics.*
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics.pu_metrics import eval_all  # noqa: E402

LABEL_COL    = "label"
FOLD_COL     = "fold"
CIG_COL      = "cig"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]
KS           = (0.01, 0.02, 0.05)

# Seed del reshuffle: diverso dal seed originale del benchmark (42 / 17)
# per garantire indipendenza dalla struttura fold usata per la HP selection.
RESHUFFLE_SEED = 2025



def _convert_categoricals_nativi(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Converte colonne object/category in float32 integer codes (LightGBM native).
    NA in categoriali → NaN (gestite nativamente da LightGBM).

    Ritorna (df_convertito, cat_feature_names) dove cat_feature_names è la lista
    dei nomi delle colonne originariamente categoriali (prima della conversione).
    Serve per passare categorical_feature a lgb.Dataset dopo la conversione.
    """
    drop_set = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    cat_cols = [c for c in df.columns
                if c not in drop_set and df[c].dtype.kind not in "iufb"]
    for col in cat_cols:
        df[col] = (df[col].astype("category")
                          .cat.codes
                          .replace(-1, np.nan)
                          .astype("float32"))
    return df, cat_cols   # return FUORI dal loop


def load_nativi_raw(model_number: int) -> tuple[pd.DataFrame, list]:
    """Carica parquet nativi mantenendo la colonna CIG (serve per il fold-map).
    Filtra solo le righe con fold assegnato. Converte categoriali.

    Ritorna (df, cat_feature_names):
      df               — DataFrame con fold assegnato e categoriali convertite
      cat_feature_names — nomi originali delle colonne categoriali (per lgb.Dataset)
    """
    path = Path(NATIVI_PATH) / f"M{model_number}.parquet"
    df = pd.read_parquet(path)
    df = df[df[FOLD_COL].notna()].reset_index(drop=True)
    df, cat_names = _convert_categoricals_nativi(df)
    print(f"  [nativi M{model_number}] {len(df):,} righe — "
          f"{df[LABEL_COL].eq(1).sum()} P, "
          f"{df[LABEL_COL].eq(0).sum()} N, "
          f"{df[LABEL_COL].isna().sum()} U | "
          f"{len(cat_names)} categoriali: {cat_names}")
    return df, cat_names


def load_preprocessed_raw(model_number: int) -> pd.DataFrame:
    """Carica parquet preprocessed mantenendo la colonna CIG."""
    path = Path(PREPROCESSED_PATH) / f"M{model_number}.parquet"
    df = pd.read_parquet(path)
    df = df[df[FOLD_COL].notna()].reset_index(drop=True)
    print(f"  [preprocessed M{model_number}] {len(df):,} righe — "
          f"{df[LABEL_COL].eq(1).sum()} P, "
          f"{df[LABEL_COL].eq(0).sum()} N, "
          f"{df[LABEL_COL].isna().sum()} U")
    return df



def compute_fold_map(df_nativi: pd.DataFrame, seed: int = RESHUFFLE_SEED) -> pd.Series:
    """Calcola una nuova assegnazione fold stratificata sulle righe di df_nativi.

    Strati: P (label=1) → 1, N (label=0) → 0, U (label=NA) → 2.
    StratifiedKFold garantisce stessi conteggi P/N/U per fold (±1 per arrotondamento).

    Il DataFrame viene sortato per CIG prima dello split: garantisce che la stessa
    chiamata su nativi e preprocessed (stesso seed, stessi CIG) produca fold identici
    anche se le righe sono in ordine diverso nei due parquet.

    Ritorna una pd.Series con indice = CIG e valori = nuovo fold (0-3).
    Il seed è separato da quello del benchmark (42/17) per indipendenza.

    Assumzione: CIG è unico per riga nel dataset filtrato (fold != NA).
    """
    df_sorted = df_nativi.sort_values(CIG_COL).reset_index(drop=True)
    label     = df_sorted[LABEL_COL].values
    stratum   = np.where(label == 1, 1, np.where(label == 0, 0, 2))

    skf      = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    new_fold = np.empty(len(df_sorted), dtype=float)
    for fold_i, (_, test_idx) in enumerate(skf.split(np.zeros(len(df_sorted)), stratum)):
        new_fold[test_idx] = float(fold_i)

    fold_map = pd.Series(new_fold, index=df_sorted[CIG_COL].values, name=FOLD_COL)

    # Diagnostica: verifica che i conteggi P/N/U siano bilanciati tra fold
    _check_fold_balance(label, new_fold)
    return fold_map


def _check_fold_balance(label: np.ndarray, new_fold: np.ndarray) -> None:
    """Stampa i conteggi P/N/U per fold come verifica del reshuffle."""
    for f in range(4):
        mask = new_fold == f
        n_p  = int((label[mask] == 1).sum())
        n_n  = int((label[mask] == 0).sum())
        n_u  = int(np.isnan(label[mask]).sum())
        print(f"    Fold {f}: {n_p} P, {n_n} N, {n_u} U")


def compute_fold_map_hierarchical(seed: int = RESHUFFLE_SEED) -> pd.Series:
    """Calcola fold_map gerarchico: M3 prima (più vincolato), poi M2-only, poi M1-only.

    Motivazione:
      M3 ⊆ M2 ⊆ M1 per i CIG unlabeled. Calcolare il fold_map da M1 e proiettarlo su
      M3 rischia di produrre fold sbilanciati in M3 se i CIG di M3 si concentrano in
      un sottoinsieme del range ordinato dei CIG di M1 (es. contratti recenti con CIG
      alti, o post-aggiudicazione con maggiore completezza dei dati).
      La strategia M3→M2→M1 garantisce bilanciamento locale in ogni dataset.

    Nota: P e N certi NON sono identici tra M1/M2/M3 — ogni dataset può contenere
    contratti con dati disponibili solo per quella fase (ante/durante/post).
    La strategia gerarchica bilancia P, N e U correttamente in ciascun dataset.

    Logica:
      1. M3: StratifiedKFold su TUTTI i CIG di M3 (P, N, U_M3) → fold bilanciati in M3.
      2. M2: StratifiedKFold sui CIG di M2 non ancora assegnati (U_M2 escluso M3,
             piu eventuali P/N presenti in M2 ma assenti in M3).
      3. M1: StratifiedKFold sui CIG di M1 non ancora assegnati (residui M1-only).

    Caricamento:
      Legge solo le colonne [CIG, label, fold] dai parquet (slim load).
      Viene sempre chiamata su tutti e tre i dataset indipendentemente da --datasets,
      per garantire un fold_map coerente e riproducibile.

    Ritorna una pd.Series con indice=CIG e valori=fold (0-3).
    """
    fold_map = pd.Series(dtype=float, name=FOLD_COL)

    for mn in [3, 2, 1]:
        path = Path(NATIVI_PATH) / f"M{mn}.parquet"
        df   = pd.read_parquet(path, columns=[CIG_COL, LABEL_COL, FOLD_COL])
        df   = df[df[FOLD_COL].notna()].reset_index(drop=True)
        df   = df[[CIG_COL, LABEL_COL]]

        df_sorted = df.sort_values(CIG_COL).reset_index(drop=True)

        # Solo CIG non ancora nel fold_map (nuovi rispetto ai dataset precedenti)
        is_new = ~df_sorted[CIG_COL].isin(fold_map.index)
        df_new = df_sorted[is_new].reset_index(drop=True)

        n_new = len(df_new)
        if n_new == 0:
            print(f"  [fold_map M{mn}] tutti i CIG già assegnati da dataset più piccoli")
            continue

        label_new = df_new[LABEL_COL].values.astype(float)
        stratum   = np.where(label_new == 1, 1, np.where(label_new == 0, 0, 2))

        skf      = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        new_fold = np.empty(n_new, dtype=float)
        for fold_i, (_, test_idx) in enumerate(skf.split(np.zeros(n_new), stratum)):
            new_fold[test_idx] = float(fold_i)

        new_map  = pd.Series(new_fold, index=df_new[CIG_COL].values, name=FOLD_COL)
        fold_map = pd.concat([fold_map, new_map])

        n_p = int((label_new == 1).sum())
        n_n = int((label_new == 0).sum())
        n_u = int(np.isnan(label_new).sum())
        print(f"  [fold_map M{mn}] {n_new:,} CIG nuovi "
              f"({n_p} P, {n_n} N, {n_u} U) — bilanciamento:")
        _check_fold_balance(label_new, new_fold)

    print(f"  [fold_map] Totale CIG assegnati: {len(fold_map):,}")
    return fold_map


def apply_fold_map(df: pd.DataFrame, fold_map: pd.Series) -> pd.DataFrame:
    """Sostituisce la colonna fold con i valori di fold_map (join su CIG).
    Righe con CIG non trovato nel map vengono eliminate (non dovrebbe succedere).
    """
    df = df.copy()
    df[FOLD_COL] = df[CIG_COL].map(fold_map)
    missing = df[FOLD_COL].isna().sum()
    if missing > 0:
        print(f"  [WARN] {missing} righe con CIG non trovato nel fold_map — eliminate")
    df = df[df[FOLD_COL].notna()].reset_index(drop=True)
    return df



def get_features(df: pd.DataFrame) -> list:
    """Tutte le colonne numeriche/float/bool dopo drop delle colonne amministrative."""
    drop_set = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    return [c for c in df.columns
            if c not in drop_set and df[c].dtype.kind in "iufb"]


def cat_names_to_idx(cat_feature_names: list, features: list) -> list:
    """Converte nomi delle colonne categoriali in indici 0-based nella feature matrix.

    Necessario per lgb.Dataset(categorical_feature=cat_idx) quando X è passato
    come numpy array (senza feature_name). Se features include feature_name,
    LightGBM accetta sia indici int che nomi stringa — qui usiamo indici.
    """
    return [features.index(c) for c in cat_feature_names if c in features]


def split_fold(df: pd.DataFrame, features: list, fold_test: int) -> dict:
    """Restituisce dizionario di array numpy per training e test del fold indicato.

    Chiavi: X_P_tr, X_N_tr, X_U_tr, X_te, label_te
    X_te contiene TUTTE le righe del fold di test (P, N, U).
    label_te: 1=P, 0=N, NaN=U.
    """
    label = df[LABEL_COL].values.astype(float)
    fold  = df[FOLD_COL].values.astype(float)
    X     = df[features].to_numpy(dtype=np.float32)

    tr = fold != fold_test
    te = fold == fold_test

    return dict(
        X_P_tr   = X[(label == 1) & tr],
        X_N_tr   = X[(label == 0) & tr],
        X_U_tr   = X[np.isnan(label) & tr],
        X_te     = X[te],
        label_te = label[te],
    )



def compute_fold_metrics(scores_te: np.ndarray, label_te: np.ndarray,
                         ks: tuple = KS) -> dict:
    """Calcola Lift@k%, PR AUC, ROC AUC per un fold dato score e label."""
    sp = scores_te[label_te == 1]
    sn = scores_te[label_te == 0]
    su = scores_te[np.isnan(label_te)]
    return eval_all(sp, sn, su, ks=ks)
