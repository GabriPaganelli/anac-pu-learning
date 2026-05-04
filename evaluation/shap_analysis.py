"""
shap_analysis.py — SHAP per i 4 modelli finalisti.
Usa il modello finale addestrato su TUTTI i dati (già salvato da calibration.py).
SHAP si applica agli score grezzi (pre-Platt): si interpreta il modello,
non la probabilità calibrata. Platt è una trasformazione monotona che non
altera i ranking né il segno dei contributi, solo la scala.

Per Bagging LGB e PUET l'ensemble non ha un unico Booster.
SHAP viene calcolato mediando i valori di n_shap_sample modelli individuali:
  - BaggingLGB: TreeSHAP su ciascun LGB booster, poi media
  - PUET: TreeSHAP su ciascun ExtraTrees bag, poi media
Convergenza rapida: già a 20-30 campioni i ranking globali si stabilizzano.

Output in results/{model}_{Mn}/shap/:
  shap_values_final.npy        — (n_rows, n_features) valori SHAP
  feature_names.json           — lista ordinata delle feature
  feature_importance.csv       — mean |SHAP| per feature, ordinata
  plots/bar_top20.{png,pdf}
  plots/beeswarm_top20.{png,pdf}
  plots/waterfall_{caso}.{png,pdf}  — 4 osservazioni chiave
  plots/pdp_top3.{png,pdf}
"""

import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit as sigmoid

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shap

from utils_expost import (
    setup_out_dir, load_model, model_exists, save_model,
    load_gamma_star, load_data, model_predict,
    PREPARE_FNS, FINAL_FIT_FNS,
    BaggingLGBWrapper, PUETWrapper,
    ALL_MODELS, ALL_DATASETS,
    LABEL_COL, FOLD_COL, CIG_COL, CALIB_SEED,
)
from selected_models.utils import compute_fold_map_hierarchical

TOP_K_BAR      = 20
TOP_K_BEESWARM = 20
TOP_K_PDP      = 3
N_SHAP_ROWS    = 10_000   # righe da campionare per SHAP (globale + beeswarm)
SHAP_ROWS_SEED = 17



def compute_shap_values(model, X: np.ndarray, model_name: str,
                        n_shap_sample: int = 50) -> tuple[np.ndarray, float]:
    """Calcola valori SHAP e base_value per il modello dato.

    Strategia per tipo di modello:
      lgb_supervised / re_lgbm : TreeSHAP sul Booster (esatto, veloce)
      bagging_lgbm              : media TreeSHAP su n_shap_sample boosters
      puet                      : media TreeSHAP su n_shap_sample bags ET

    Ritorna (shap_values: (n, d), base_value: float).
    SHAP values in spazio score grezzo (log-odds / margine).
    """
    print(f"  [SHAP] Calcolo per {model_name} su {X.shape[0]} osservazioni...")

    if model_name in ("lgb_supervised", "re_lgbm"):
        ex   = shap.TreeExplainer(model)
        sv   = ex.shap_values(X)
        bv   = float(ex.expected_value)
        if isinstance(sv, list):
            sv = sv[1]
        print(f"  [SHAP] shape={sv.shape}  base_value={bv:.4f}")
        return sv, bv

    if model_name == "bagging_lgbm":
        assert isinstance(model, BaggingLGBWrapper)
        n_s    = min(n_shap_sample, len(model.boosters))
        all_sv = []
        all_bv = []
        for i, b in enumerate(model.boosters[:n_s]):
            ex = shap.TreeExplainer(b)
            sv = ex.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]
            all_sv.append(sv)
            all_bv.append(float(ex.expected_value))
            if (i + 1) % 10 == 0:
                print(f"    booster {i+1}/{n_s}")
        shap_mean = np.mean(all_sv, axis=0)
        bv_mean   = float(np.mean(all_bv))
        print(f"  [SHAP] Bagging: {n_s} boosters mediati. "
              f"shape={shap_mean.shape}  base_value={bv_mean:.4f}")
        return shap_mean, bv_mean

    if model_name == "puet":
        assert isinstance(model, PUETWrapper)
        n_s    = min(n_shap_sample, len(model.estimators))
        all_sv = []
        all_bv = []
        for i, et in enumerate(model.estimators[:n_s]):
            ex = shap.TreeExplainer(et)
            sv = ex.shap_values(X)
            # ExtraTreesClassifier può restituire liste o array 3D (n_samples, n_features, n_classes)
            if isinstance(sv, list):
                sv_pos = sv[1]  # lista [sv_class0, sv_class1]
            elif hasattr(sv, 'ndim') and sv.ndim == 3:
                sv_pos = sv[:, :, 1]  # array 3D → classe positiva (indice 1)
            else:
                sv_pos = sv  # fallback: assumiamo sia 2D
            all_sv.append(sv_pos)
            ev = ex.expected_value
            all_bv.append(float(ev[1]) if hasattr(ev, "__len__") else float(ev))
            if (i + 1) % 5 == 0:
                print(f"    bag {i+1}/{n_s}")
        shap_mean = np.mean(all_sv, axis=0)
        bv_mean   = float(np.mean(all_bv))
        # ExtraTreesClassifier TreeSHAP può restituire 3D (n, d, 2)
        # Assicurati sempre 2D estraendo classe positiva se necessario
        if shap_mean.ndim == 3:
            shap_mean = shap_mean[:, :, 1]
        print(f"  [SHAP] PUET: {n_s} bags mediati. "
              f"shape={shap_mean.shape}  base_value={bv_mean:.4f}")
        return shap_mean, bv_mean

    raise ValueError(f"model_name sconosciuto: {model_name}")



def plot_bar(shap_vals: np.ndarray, feature_names: list,
             model_name: str, model_number: int,
             out_dir: Path, top_k: int = TOP_K_BAR) -> pd.DataFrame:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df_imp = (pd.DataFrame({"feature": feature_names, "importance": mean_abs})
              .sort_values("importance", ascending=False)
              .head(top_k)
              .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.35)))
    ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1],
            color="#2E86AB", edgecolor="white")
    ax.set_xlabel("Importanza SHAP media (|SHAP|)", fontsize=10)
    ax.set_title(
        f"SHAP importanza globale — {model_name} M{model_number}\n"
        f"Top {top_k} feature (score grezzo pre-Platt)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "shap" / "plots" / f"bar_top{top_k}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    return df_imp



def plot_beeswarm(shap_vals: np.ndarray, base_val: float,
                  X: np.ndarray, feature_names: list,
                  model_name: str, model_number: int,
                  out_dir: Path, top_k: int = TOP_K_BEESWARM) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_k]
    sv_top   = shap_vals[:, top_idx]
    X_top    = X[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]

    exp = shap.Explanation(
        values        = sv_top,
        base_values   = np.full(len(sv_top), base_val),
        data          = X_top,
        feature_names = names_top,
    )
    fig, ax = plt.subplots(figsize=(10, max(5, top_k * 0.4)))
    shap.plots.beeswarm(exp, max_display=top_k, show=False)
    plt.title(
        f"SHAP beeswarm — {model_name} M{model_number}  "
        f"(top {top_k}, score grezzo)", fontsize=10)
    fig = plt.gcf()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "shap" / "plots" / f"beeswarm_top{top_k}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)



def plot_waterfall(shap_vals: np.ndarray, base_val: float,
                   X: np.ndarray, feature_names: list,
                   scores: np.ndarray, label: np.ndarray,
                   model_name: str, model_number: int,
                   out_dir: Path) -> None:
    """4 osservazioni chiave:
      - score massimo (tra tutti)
      - positivo difficile (P con score più basso)
      - falso positivo estremo (N con score più alto)
      - caso mediano (score più vicino alla mediana)
    """
    is_pos = label == 1.0
    is_neg = label == 0.0

    if not is_pos.any():
        print("  [SHAP waterfall] Nessun positivo: salto.")
        return
    if not is_neg.any():
        print("  [SHAP waterfall] Nessun negativo: salto.")
        return

    mediana = float(np.nanmedian(scores))
    casi = {
        "score_massimo":          int(np.argmax(scores)),
        "positivo_difficile":     int(np.argmin(np.where(is_pos, scores, np.inf))),
        "falso_positivo_estremo": int(np.argmax(np.where(is_neg, scores, -np.inf))),
        "caso_mediano":           int(np.argmin(np.abs(scores - mediana))),
    }
    titoli = {
        "score_massimo":          "Score massimo",
        "positivo_difficile":     "Positivo con score più basso",
        "falso_positivo_estremo": "Negativo con score più alto",
        "caso_mediano":           "Osservazione mediana",
    }

    for nome, idx in casi.items():
        exp_i = shap.Explanation(
            values        = shap_vals[idx],
            base_values   = base_val,
            data          = X[idx],
            feature_names = feature_names,
        )
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(exp_i, max_display=15, show=False)
        plt.title(
            f"{titoli[nome]}  (score={scores[idx]:.4f})\n"
            f"{model_name} M{model_number} — SHAP in spazio score grezzo",
            fontsize=10)
        fig = plt.gcf()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                out_dir / "shap" / "plots" / f"waterfall_{nome}.{ext}",
                bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"    waterfall_{nome}: idx={idx}  score={scores[idx]:.4f}")



def _compute_pdp_manual(predict_fn, X: np.ndarray, feature_idx: int,
                        n_grid: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """PDP 1-D via media marginale (implementazione manuale, senza sklearn).

    Evita problemi di compatibilità con PartialDependenceDisplay >= 1.4/1.6.
    Campiona la griglia tra il 5° e il 95° percentile della feature.
    """
    vals = X[:, feature_idx]
    lo, hi = np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)
    if lo == hi:                      # feature costante nel campione
        lo = vals.min(); hi = vals.max()
    grid = np.linspace(lo, hi, n_grid)
    avg  = np.empty(n_grid)
    for i, v in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feature_idx] = v
        avg[i] = predict_fn(X_mod).mean()
    return grid, avg


def plot_pdp(model, X_df: pd.DataFrame, shap_vals: np.ndarray,
             model_name: str, model_number: int,
             out_dir: Path, top_k: int = TOP_K_PDP) -> None:
    """PDP top-k feature via media marginale manuale (no sklearn wrapper).

    Calcola E[f(x) | x_j = v] variando v su una griglia [p5, p95] della feature.
    Usa al massimo 2000 righe per velocità.
    """
    mean_abs  = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:top_k]
    top_names = [X_df.columns[i] for i in top_idx]

    n_pdp = min(2000, len(X_df))
    X_pdp = X_df.iloc[:n_pdp].to_numpy(dtype=np.float32)

    def predict_fn(X):
        return model_predict(model, X, model_name)

    fig, axes = plt.subplots(1, top_k, figsize=(5 * top_k, 4))
    if top_k == 1:
        axes = [axes]

    for ax, feat_idx, name in zip(axes, top_idx, top_names):
        grid, avg = _compute_pdp_manual(predict_fn, X_pdp, feat_idx)
        ax.plot(grid, avg, color="#2E86AB", lw=2)
        ax.fill_between(grid, avg, alpha=0.15, color="#2E86AB")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("P(frode) media marginale", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"PDP top-{top_k} feature — {model_name} M{model_number}\n"
        f"(media marginale, score grezzo pre-Platt)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / "shap" / "plots" / f"pdp_top{top_k}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [SHAP PDP] Salvato top-{top_k}: {top_names}")



def run_shap(model_name: str, model_number: int, gamma: float,
             n_shap_sample: int = 50,
             n_shap_rows: int = N_SHAP_ROWS,
             force_retrain: bool = False) -> dict:
    """SHAP completo per un modello × dataset.

    Carica il modello finale da calibration.py (o lo raddestra se non trovato).
    SHAP viene calcolato su P+N (righe labeled, qualche migliaio) che hanno
    etichetta nota: semanticamente corretto per capire cosa discrimina P da N.
    Se P+N > n_shap_rows scatta il subsampling (salvaguardia, raramente attivo).
    I full-data scores (scores_final.csv) usano tutte le righe incluso U.
    """
    print(f"  SHAP  {model_name}  M{model_number}  gamma={gamma}")
    print(f"  n_shap_sample={n_shap_sample}  n_shap_rows={n_shap_rows:,}")

    out_dir = setup_out_dir(model_name, model_number)

    print(f"\n  Fold map (seed={CALIB_SEED}, stesso di calibration)...")
    fold_map = compute_fold_map_hierarchical(seed=CALIB_SEED)
    print("  Caricamento dati...")
    df, features, cat_idx = load_data(model_name, model_number, fold_map)
    label_all = df[LABEL_COL].values.astype(float)
    X_all     = df[features].to_numpy(dtype=np.float32)

    n_P = int((label_all == 1).sum())
    n_N = int((label_all == 0).sum())
    n_U = int(np.isnan(label_all).sum())
    print(f"  Dati totali: {len(df):,} ({n_P} P, {n_N} N, {n_U} U)")

    # Solo P+N hanno fold assegnato: sono i labeled, poche migliaia.
    # SHAP su P+N è semanticamente corretto (discrimina positivi da negativi).
    # U (non labeled, potenzialmente centinaia di migliaia) è escluso da SHAP
    # ma incluso in scores_final per le predizioni di deploy.
    mask_labeled  = ~np.isnan(label_all)
    X_labeled     = X_all[mask_labeled]
    label_labeled = label_all[mask_labeled]
    n_labeled     = len(X_labeled)
    print(f"  Righe labeled per SHAP (P+N): {n_labeled:,} "
          f"({n_P} P, {n_N} N)")

    # Subsample solo se P+N supera la soglia (salvaguardia)
    idx_path = out_dir / "shap" / "shap_sample_idx.npy"

    if not force_retrain and idx_path.exists():
        shap_idx = np.load(str(idx_path))
        print(f"  Indici campione SHAP caricati: {len(shap_idx):,} righe")
    else:
        if n_labeled <= n_shap_rows:
            shap_idx = np.arange(n_labeled)
            print(f"  SHAP su tutte le righe labeled: {n_labeled:,} "
                  f"(< soglia {n_shap_rows:,})")
        else:
            rng_shap = np.random.default_rng(SHAP_ROWS_SEED)
            shap_idx = np.sort(
                rng_shap.choice(n_labeled, size=n_shap_rows, replace=False))
            print(f"  SHAP subsample: {n_shap_rows:,}/{n_labeled:,} righe "
                  f"(seed={SHAP_ROWS_SEED})")
        np.save(str(idx_path), shap_idx)

    X_shap     = X_labeled[shap_idx]
    label_shap = label_labeled[shap_idx]
    X_shap_df  = pd.DataFrame(X_shap, columns=features)

    shap_path = out_dir / "shap" / "shap_values_final.npy"

    if not force_retrain and model_exists(out_dir, "final_model", model_name):
        print("\n  Carico modello finale esistente...")
        model = load_model(out_dir, "final_model", model_name)
    else:
        print("\n  Modello finale non trovato — raddestra...")
        state = PREPARE_FNS[model_name](df, features, cat_idx, gamma, model_number)
        model = FINAL_FIT_FNS[model_name](
            df, features, cat_idx, gamma, model_number, state)
        save_model(model, out_dir, "final_model", model_name)

    if not shap_path.exists() or force_retrain:
        shap_vals, base_val = compute_shap_values(
            model, X_shap, model_name, n_shap_sample=n_shap_sample)
        np.save(str(shap_path), shap_vals)
        print(f"  SHAP values salvati: {shap_path}  shape={shap_vals.shape}")
    else:
        print(f"\n  SHAP values trovati: {shap_path} — carico.")
        shap_vals = np.load(str(shap_path))
        # Ricalcola base_value dal modello (serve per waterfall/beeswarm)
        _, base_val = compute_shap_values(
            model, X_shap[:5], model_name, n_shap_sample=1)

    print("\n  Calcolo scores su tutti i dati (per scores_final.csv)...")
    scores_all = model_predict(model, X_all, model_name)

    scores_shap = model_predict(model, X_shap, model_name)

    feat_path = out_dir / "shap" / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    df_imp = (pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))
    df_imp.to_csv(out_dir / "shap" / "feature_importance.csv", index=False)
    print(f"\n  Top-5 feature: {df_imp['feature'].head(5).tolist()}")

    print("\n  Bar chart importanza globale...")
    plot_bar(shap_vals, features, model_name, model_number, out_dir)

    print("  Beeswarm plot...")
    plot_beeswarm(shap_vals, base_val, X_shap, features,
                  model_name, model_number, out_dir)

    print("  Waterfall plot (4 osservazioni chiave)...")
    plot_waterfall(shap_vals, base_val, X_shap, features,
                   scores_shap, label_shap, model_name, model_number, out_dir)

    print("  PDP top-3 feature...")
    plot_pdp(model, X_shap_df, shap_vals, model_name, model_number, out_dir)

    print(f"\n  OK SHAP completato. Output in: {out_dir / 'shap'}")
    return {
        "top_features": df_imp["feature"].head(10).tolist(),
        "out_dir":      out_dir,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP analysis per i 4 finalisti PU")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument(
        "--datasets", nargs="+", type=int, default=ALL_DATASETS,
        choices=ALL_DATASETS)
    parser.add_argument(
        "--gamma-config", type=Path,
        default=_HERE / "gamma_star.json")
    parser.add_argument(
        "--n-shap-sample", type=int, default=50,
        help="Numero di modelli da campionare per SHAP (Bagging/PUET)")
    parser.add_argument(
        "--n-shap-rows", type=int, default=N_SHAP_ROWS,
        help=f"Righe da campionare per SHAP (default {N_SHAP_ROWS:,})")
    parser.add_argument(
        "--force-retrain", action="store_true")
    args = parser.parse_args()

    gamma_star = load_gamma_star(args.gamma_config)

    for mn in args.datasets:
        for mod in args.models:
            gamma = gamma_star[mod][mn]
            run_shap(mod, mn, gamma,
                     n_shap_sample=args.n_shap_sample,
                     n_shap_rows=args.n_shap_rows,
                     force_retrain=args.force_retrain)
