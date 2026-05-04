"""
make_qq_ranks.py — QQ plot del ranking degli score tra modelli finalisti.

Confronta rank_pct (re_lgbm vs lgb_supervised) e (re_lgbm vs bagging_lgbm)
su contratti U *fuori dal training* per dataset M1, M2, M3.

Strategia campionamento (uniforme per tutti i dataset):
  Carica nativi M{n} senza filtro fold, prende U con fold=NaN (non in training).
  Il nativi M1 contiene tutti i 9.47M contratti — stessa struttura di M2/M3.

Output: qq_rank_comparison_M{n}.png/.pdf in questa cartella.
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.special import expit as sigmoid
from scipy.stats import spearmanr, rankdata

HERE    = Path(__file__).resolve().parent
ROOT    = HERE.parents[1]
EX_POST = HERE.parent
RESULTS = EX_POST / "results"

for _p in (str(ROOT), str(EX_POST)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selected_models.utils import (
    NATIVI_PATH, LABEL_COL, FOLD_COL, CIG_COL,
    _convert_categoricals_nativi, COLS_TO_DROP,
)
from utils_expost import BaggingLGBWrapper, load_model

SAMPLE_N   = 4_000
RNG_SEED   = 42

MODELS_QQ = ["re_lgbm", "lgb_supervised", "bagging_lgbm"]


# Preprocessing

def _learn_cat_map(df_train: pd.DataFrame, cat_names: list) -> dict:
    """Apprende mapping categoriale dal df training (stesso ordine del training)."""
    return {
        col: sorted(df_train[col].dropna().unique())
        for col in cat_names
        if col in df_train.columns
    }


def _apply_cat_map(df: pd.DataFrame, cat_map: dict) -> pd.DataFrame:
    """Applica mapping categoriale: valore -> codice intero (sconosciuto -> NaN)."""
    df = df.copy()
    for col, categories in cat_map.items():
        if col in df.columns:
            code_map = {v: float(i) for i, v in enumerate(categories)}
            df[col] = df[col].map(code_map).astype("float32")
    return df


# Caricamento dati training (per CIG e cat_map)

def load_training_info(mn: int) -> tuple[set, list, list, dict]:
    """Carica nativi training, ritorna (training_cigs, features, cat_names, cat_map).

    Usato solo per sapere quali CIG erano in training e per il cat_map.
    Non serve caricare tutto il df: load parquet leggero.
    """
    path = Path(NATIVI_PATH) / f"M{mn}.parquet"
    # Carica solo righe con fold assegnato (= training)
    df = pd.read_parquet(path)
    df_train = df[df[FOLD_COL].notna()].reset_index(drop=True)

    # cat_map sul training originale (prima di convertire)
    drop_set  = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    cat_names = [c for c in df_train.columns
                 if c not in drop_set and df_train[c].dtype.kind not in "iufb"]
    cat_map   = _learn_cat_map(df_train, cat_names)

    # Converti per ricavare la feature list
    df_train, _ = _convert_categoricals_nativi(df_train)
    features    = [c for c in df_train.columns
                   if c not in set(COLS_TO_DROP + [LABEL_COL, FOLD_COL, "esito",
                                                    "anno_pubblicazione", "regione"])
                   and df_train[c].dtype.kind in "iufb"]

    training_cigs = set(df_train[CIG_COL].values)
    print(f"  [M{mn}] Training CIG: {len(training_cigs):,}  features: {len(features)}")
    return training_cigs, features, cat_names, cat_map


# Campionamento U fuori training

def sample_u_outside_training(mn: int, cat_map: dict, features: list,
                               n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Campiona n contratti U con fold=NaN dal nativi M{mn} (fuori training).

    Stessa logica per M1, M2, M3: il nativi M1 contiene tutti i 9.47M contratti
    con la colonna fold assegnata solo ai 42k usati in training.
    """
    rng  = np.random.default_rng(seed)
    path = Path(NATIVI_PATH) / f"M{mn}.parquet"
    df   = pd.read_parquet(path)

    mask = df[LABEL_COL].isna() & df[FOLD_COL].isna()
    df_u = df[mask].copy()
    print(f"  [M{mn}] U fuori training (fold=NaN): {len(df_u):,}")

    chosen_idx = rng.choice(len(df_u), size=min(n, len(df_u)), replace=False)
    df_sample  = df_u.iloc[chosen_idx].copy()

    df_sample = _apply_cat_map(df_sample, cat_map)
    missing = [f for f in features if f not in df_sample.columns]
    for c in missing:
        df_sample[c] = np.nan
    X    = df_sample[features].to_numpy(dtype=np.float32)
    cigs = df_sample[CIG_COL].values
    print(f"  [M{mn}] X_sample shape={X.shape}")
    return X, cigs


# Scoring

def score_model(model_name: str, mn: int, X: np.ndarray) -> np.ndarray:
    """Carica modello finale e produce raw scores."""
    if model_name == "re_lgbm" and mn == 3:
        folder = RESULTS / "re_lgbm_M3_g100"
    else:
        folder = RESULTS / f"{model_name}_M{mn}"

    model = load_model(folder, "final_model", model_name)

    if model_name == "re_lgbm":
        raw = sigmoid(model.predict(X, raw_score=True))
    elif model_name == "lgb_supervised":
        raw = model.predict(X)
    else:  # bagging_lgbm
        raw = model.predict(X)
    return raw.astype(np.float32)


def apply_platt(model_name: str, mn: int, raw: np.ndarray) -> np.ndarray:
    """Applica Platt scaling globale dai parametri salvati."""
    if model_name == "re_lgbm" and mn == 3:
        folder = RESULTS / "re_lgbm_M3_g100"
    else:
        folder = RESULTS / f"{model_name}_M{mn}"

    platt_path = folder / "calibration" / "platt_params.csv"
    if not platt_path.exists():
        print(f"  [Platt] non trovato per {model_name} M{mn}, uso raw")
        return raw

    df_p   = pd.read_csv(platt_path)
    row    = df_p[df_p["fold"] == "global"].iloc[0]
    a, b   = float(row["a"]), float(row["b"])
    return sigmoid(a * raw + b).astype(np.float32)


# Plot

def qq_panel(ax, x: np.ndarray, y: np.ndarray,
             xlabel: str, ylabel: str, title: str, n_label: int = 12):
    """Panel QQ rank con coloring per distanza dalla diagonale."""
    delta = np.abs(x - y)
    vmax  = max(delta.max(), 1.0)
    norm  = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap  = plt.cm.RdYlGn_r

    ax.scatter(x, y, c=delta, cmap=cmap, norm=norm,
               s=7, alpha=0.50, linewidths=0, rasterized=True)

    ax.plot([0, 100], [0, 100], color="black", lw=1.3, ls="--", zorder=4)
    for band, ls, alpha in [(10, ":", 0.55), (20, "-.", 0.35)]:
        ax.plot([0, 100], [band, 100 + band], color="steelblue",
                lw=0.8, ls=ls, alpha=alpha)
        ax.plot([0, 100], [-band, 100 - band], color="steelblue",
                lw=0.8, ls=ls, alpha=alpha)

    # Cerchi neri sui punti piu discordanti
    top_idx = np.argsort(delta)[-n_label:]
    ax.scatter(x[top_idx], y[top_idx],
               s=45, facecolors="none", edgecolors="black", lw=0.8, zorder=5)

    rho, _ = spearmanr(x, y)
    mae    = delta.mean()
    p20    = (delta > 20).mean() * 100

    ax.text(0.04, 0.94,
            f"Spearman rho = {rho:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.text(0.04, 0.85,
            f"MAE = {mae:.1f}pp   |delta|>20pp: {p20:.1f}%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.tick_params(labelsize=8)

    return mcolors.Normalize(vmin=0, vmax=vmax), cmap


def make_figure(mn: int,
                x_re: np.ndarray, y_lgb: np.ndarray, y_bag: np.ndarray,
                n_sample: int):
    """Genera figura 1x2 per dataset M{mn}."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        f"Coerenza del ranking tra modelli - Dataset M{mn}\n"
        f"(n={n_sample:,} contratti U fuori training, rank_pct su campione)",
        fontsize=12, y=1.02,
    )

    norm1, cmap1 = qq_panel(
        axes[0], x_re, y_lgb,
        xlabel="re_lgbm  rank_pct",
        ylabel="lgb_supervised  rank_pct",
        title="re_lgbm  vs  lgb_supervised",
    )
    norm2, cmap2 = qq_panel(
        axes[1], x_re, y_bag,
        xlabel="re_lgbm  rank_pct",
        ylabel="bagging_lgbm  rank_pct",
        title="re_lgbm  vs  bagging_lgbm",
    )

    # Colorbar condivisa (delta rank)
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    sm  = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("|delta rank| (pp)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.text(
        0.5, -0.03,
        "Linea tratteggiata = accordo perfetto.  "
        "Bande blu = +-10pp e +-20pp.  "
        "Cerchi neri = 12 contratti piu discordanti.",
        ha="center", fontsize=8, color="gray",
    )

    fig.tight_layout(rect=[0, 0, 0.91, 1])
    for ext in ("png", "pdf"):
        out = HERE / f"qq_rank_comparison_M{mn}.{ext}"
        try:
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Salvato: {out}")
        except PermissionError:
            print(f"  [SKIP] {out} e' aperto in un altro programma — chiudilo e riesegui")
    plt.close(fig)


# Main

def run_mn(mn: int):
    print(f"\n== M{mn} ==================================================")

    # 1. Info training per tutti e 3 i modelli (features identiche per M{mn})
    # Usiamo re_lgbm come riferimento per features/cat_map (stesso tipo "nativi")
    training_cigs, features, cat_names, cat_map = load_training_info(mn)

    # 2. Campiona U fuori training (stessa logica per M1/M2/M3: nativi + fold=NaN)
    X_sample, sample_cigs = sample_u_outside_training(
        mn, cat_map, features, SAMPLE_N, RNG_SEED + mn)

    n_sample = len(sample_cigs)
    if n_sample < 50:
        print(f"  Troppo pochi campioni ({n_sample}), skip M{mn}.")
        return

    # 3. Scoratura con i 3 modelli
    print(f"  Scoratura re_lgbm M{mn}...")
    raw_re  = score_model("re_lgbm",       mn, X_sample)
    print(f"  Scoratura lgb_supervised M{mn}...")
    raw_lgb = score_model("lgb_supervised", mn, X_sample)
    print(f"  Scoratura bagging_lgbm M{mn}...")
    raw_bag = score_model("bagging_lgbm",  mn, X_sample)

    # 4. Platt calibration
    cal_re  = apply_platt("re_lgbm",       mn, raw_re)
    cal_lgb = apply_platt("lgb_supervised", mn, raw_lgb)
    cal_bag = apply_platt("bagging_lgbm",  mn, raw_bag)

    # 5. Rank percentile nel campione (0-100)
    rank_re  = rankdata(cal_re,  method="average") / n_sample * 100
    rank_lgb = rankdata(cal_lgb, method="average") / n_sample * 100
    rank_bag = rankdata(cal_bag, method="average") / n_sample * 100

    print(f"  re_lgbm  : p50={np.percentile(cal_re,50):.4f}  "
          f"p99={np.percentile(cal_re,99):.4f}")
    print(f"  lgb_sup  : p50={np.percentile(cal_lgb,50):.4f}  "
          f"p99={np.percentile(cal_lgb,99):.4f}")
    print(f"  bagging  : p50={np.percentile(cal_bag,50):.4f}  "
          f"p99={np.percentile(cal_bag,99):.4f}")

    # 6. Figura
    make_figure(mn, rank_re, rank_lgb, rank_bag, n_sample)


if __name__ == "__main__":
    for mn in (1, 2, 3):
        run_mn(mn)
    print("\nDone.")
