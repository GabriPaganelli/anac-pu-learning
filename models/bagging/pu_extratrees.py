"""
PUET — PU Extra-Trees Bagging
Base learner: ExtraTreesClassifier (sklearn).
Dati: preprocessed (no NA, categoriali label-encoded).

Varianti:
  PNPU=True : N certi fissi in ogni bootstrap (sempre presenti, mai campionati).
  PNPU=False: N certi nel pool U, campionati con la stessa probabilità degli U.

Nessun peso esplicito sugli unlabeled: la correzione per π viene
dal rapporto implicito s/|U| nel campionamento bootstrap (Mordelet & Vert 2014).
"""

MODEL     = 1       # 1=M1, 2=M2, 3=M3
PNPU      = True    # True=PNPU, False=PU puro
TEST_MODE = False   # True = run rapido (B=5, S=300, ntree=3) per verifica e tempi

B                = 150  if not TEST_MODE else 5
S                = 15_000 if not TEST_MODE else 300
NTREE_ET         = 50   if not TEST_MODE else 3
MAX_FEATURES     = 1.0   # tutte le feature: randomizzazione solo sulle soglie (Geurts 2006)
MIN_SAMPLES_LEAF = 1
RANDOM_SEED      = 42
KS               = (0.01, 0.02, 0.05)

from pathlib import Path as _P
BASE_PATH = str(_P(__file__).resolve().parents[2] / "anac" / "output" / "parquet" / "model" / "preprocessed")
OUT_DIR   = str(_P(__file__).resolve().parent / "results")

LABEL_COL    = "label"
FOLD_COL     = "fold"
EXCLUDE_COLS = ["cig", "label", "fold"]

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.pu_metrics import eval_all

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode globale: fit su tutto il dataset prima del loop, mapping consistente tra fold."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le   = LabelEncoder()
        mask = df[col].notna()
        df[col] = df[col].astype(object)
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
    return df


def load_data() -> pd.DataFrame:
    path = Path(BASE_PATH) / f"M{MODEL}.parquet"
    df   = pd.read_parquet(path)
    df   = df[df[FOLD_COL].notna()].reset_index(drop=True)
    if TEST_MODE:
        # Tieni tutti i P e N (pochi), campiona 2k U per fold
        is_labeled = df[LABEL_COL].notna()
        is_unl     = df[LABEL_COL].isna()
        unl_sample = (df[is_unl]
                      .groupby(FOLD_COL, group_keys=False)
                      .apply(lambda g: g.sample(min(2_000, len(g)),
                                                 random_state=RANDOM_SEED)))
        df = pd.concat([df[is_labeled], unl_sample]).reset_index(drop=True)
    print(f"  Dati M{MODEL}: {len(df)} righe, "
          f"{df[LABEL_COL].eq(1).sum()} P, "
          f"{df[LABEL_COL].eq(0).sum()} N, "
          f"{df[LABEL_COL].isna().sum()} U")
    return df


def run_puet(dati: pd.DataFrame) -> dict:

    rng          = check_random_state(RANDOM_SEED)
    score_raw    = np.full(len(dati), np.nan)
    fold_metrics = {}

    # Encode globale: mapping consistente per tutti i fold
    feat_data = encode_categoricals(dati.drop(columns=EXCLUDE_COLS, errors="ignore"))
    label_col = dati[LABEL_COL].values.astype(float)
    fold_col  = dati[FOLD_COL].values

    folds = sorted(dati[FOLD_COL].dropna().unique().astype(int))

    for fold_test in folds:

        print(f"\nFold test: {fold_test}  ({'PNPU' if PNPU else 'PU puro'})")

        train_mask = fold_col != fold_test
        test_mask  = fold_col == fold_test

        label_tr = label_col[train_mask]

        X_full = feat_data.values
        X_P    = X_full[train_mask & (label_col == 1)]
        X_N    = X_full[train_mask & (label_col == 0)]
        X_U    = X_full[train_mask & np.isnan(label_col)]
        X_test = X_full[test_mask]

        # Pool di campionamento negativo
        if PNPU:
            X_pool = X_U
        else:
            X_pool = np.vstack([X_U, X_N]) if len(X_N) > 0 else X_U

        s_fold  = min(S, len(X_pool))
        replace = s_fold == len(X_pool)   # bootstrap con rimpiazzo se pool esaurito
        if replace:
            print(f"  Nota: pool ({len(X_pool)}) < S ({S}), campionamento con rimpiazzo")

        print(f"  P={len(X_P)}, N={len(X_N)}, U={len(X_U)}, "
              f"pool={len(X_pool)}, s={s_fold}, replace={replace}")

        scores_test = np.zeros(len(X_test))

        for b in range(B):
            if (b + 1) % 40 == 0:
                print(f"  modello {b + 1}/{B}")

            idx  = rng.choice(len(X_pool), size=s_fold, replace=replace)
            X_neg = X_pool[idx]

            if PNPU and len(X_N) > 0:
                X_tr = np.vstack([X_P, X_N, X_neg])
                y_tr = np.concatenate([
                    np.ones(len(X_P)),
                    np.zeros(len(X_N)),
                    np.zeros(s_fold),
                ])
            else:
                X_tr = np.vstack([X_P, X_neg])
                y_tr = np.concatenate([
                    np.ones(len(X_P)),
                    np.zeros(s_fold),
                ])

            et = ExtraTreesClassifier(
                n_estimators     = NTREE_ET,
                max_features     = MAX_FEATURES,
                min_samples_leaf = MIN_SAMPLES_LEAF,
                bootstrap        = False,
                n_jobs           = -1,
                random_state     = rng.randint(0, 99999),
            )
            et.fit(X_tr, y_tr)
            scores_test += et.predict_proba(X_test)[:, 1]

        scores_test /= B
        score_raw[test_mask] = scores_test

        # Metriche fold
        label_te = label_col[test_mask]
        is_pos   = label_te == 1
        is_neg   = label_te == 0
        is_unl   = np.isnan(label_te)

        m = eval_all(
            scores_pos=scores_test[is_pos],
            scores_neg=scores_test[is_neg],
            scores_unl=scores_test[is_unl],
            ks=KS,
        )
        fold_metrics[fold_test] = {"fold": fold_test, **m}
        print(f"  Metriche fold {fold_test}:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    print("\nMetriche finali (tutti i fold)")

    final      = eval_all(
        scores_pos=score_raw[label_col == 1],
        scores_neg=score_raw[label_col == 0],
        scores_unl=score_raw[np.isnan(label_col)],
        ks=KS,
    )
    for k, v in final.items():
        print(f"  {k}: {v:.4f}")

    # Salvataggio
    variant = "pnpu" if PNPU else "pu"
    prefix  = f"puet_M{MODEL}_{variant}"
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
    run_puet(dati)
