"""
bootstrap_ci.py — Bootstrap CI per RE LightGBM nnPU.
Modello: re_lgbm  (γ* da gamma_star.json, π=0.02 fisso per tutti i dataset)

Per ogni dataset M1/M2/M3:
  - Campiona N_SCORE contratti unlabeled dal mega dataset (fuori da U_train, fissi)
  - B=200 fit di RE LGB su campioni bootstrap di (P + N + U_train) con rimpiazzo
  - Ogni bootstrap: score U_score + score labeled (per metriche)
  - Platt globale fisso (da calibration.py) applicato ai raw scores di U_score
  - Metriche (lift@1%, PR-AUC, ROC-AUC) calcolate su raw scores (rank-based)

Output in results/re_lgbm_M{n}/bootstrap/:
  scores_ci.csv   — cig, score_mean, score_lower, score_upper  (probabilità calibrate)
  metrics_ci.csv  — metrica, mean, lower, upper                (raw scores, rank-based)
  u_score_cigs.npy — CIG campionati (per riproducibilità)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from scipy.special import expit as sigmoid

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selected_models.utils import (
    compute_fold_map_hierarchical,
    NATIVI_PATH, LABEL_COL, FOLD_COL, CIG_COL,
)
from selected_models.re_lgbm import fit_predict
from utils_expost import load_data, load_gamma_star, CALIB_SEED, OUT_ROOT
from metrics.pu_metrics import eval_all

MEGA_PATH   = _ROOT / "anac" / "output" / "parquet" / "bando_cig_all.parquet"
GAMMA_PATH  = _HERE / "gamma_star.json"

PI_FIXED    = 0.02      # override PI_BY_MODEL — uniformità paper
BOOT_SEED   = 2028      # seed bootstrap (diverso da CALIB_SEED/CONF_SEED)
N_SCORE_DEF = {1: 10_000, 2: 10_000, 3: 3_000}   # U_score per dataset
N_BOOT_DEF  = 200


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _result_folder(model_number: int, gamma: float) -> Path:
    """Restituisce la cartella risultati corretta per re_lgbm.

    M1/M2 usano il nome standard (re_lgbm_M1, re_lgbm_M2).
    M3 ha cartelle con suffix gamma (re_lgbm_M3_g033, re_lgbm_M3_g100).
    Se esiste la cartella suffissata la usa; altrimenti fallback a nome standard.
    """
    gamma_str = f"{gamma:.2f}".replace(".", "")
    suffixed  = OUT_ROOT / f"re_lgbm_M{model_number}_g{gamma_str}"
    standard  = OUT_ROOT / f"re_lgbm_M{model_number}"
    if suffixed.exists():
        return suffixed
    return standard



def sample_u_score(model_number: int, training_cigs: set,
                   features: list, cat_names: list,
                   df_nativi: pd.DataFrame,
                   n_score: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Campiona n_score contratti unlabeled dal mega dataset (fuori da U_train).

    Strategia encoding categoriali:
      Il modello è stato addestrato su codici categoria derivati dal nativi parquet.
      Per garantire la stessa codifica sulle righe del mega dataset, si apprende
      il mapping category→code dal nativi di training e lo si applica al mega.
      Categorie sconosciute (non viste in training) → NaN, gestite native da LGB.

    Ritorna (X_score: np.ndarray, score_cigs: np.ndarray).
    """
    rng = np.random.default_rng(seed)

    cat_categories = {}
    for col in cat_names:
        if col in df_nativi.columns:
            # Stesso ordine usato da _convert_categoricals_nativi (sorted unique)
            unique_vals = sorted(df_nativi[col].dropna().unique())
            cat_categories[col] = unique_vals

    print(f"  [U_score M{model_number}] Carico CIG mega dataset...")
    mega_cig = pq.read_table(str(MEGA_PATH), columns=[CIG_COL]).to_pandas()[CIG_COL]
    non_train_mask = ~mega_cig.isin(training_cigs)
    non_train_cigs = mega_cig[non_train_mask].values
    print(f"  [U_score M{model_number}] {len(non_train_cigs):,} CIG fuori training → "
          f"campiono {n_score:,}")

    if len(non_train_cigs) < n_score:
        raise ValueError(f"Mega dataset: solo {len(non_train_cigs)} CIG disponibili "
                         f"fuori training, richiesti {n_score}")

    sampled_cigs = rng.choice(non_train_cigs, size=n_score, replace=False)

    print(f"  [U_score M{model_number}] Carico feature per {n_score:,} righe...")
    cols_to_load = [CIG_COL] + features
    # Carica tutte le colonne rilevanti per le righe campionate
    # (pyarrow non supporta filtro per lista CIG efficiente → load subset di cols
    #  e filtra in memoria; feasible perché carichiamo solo ~30 cols su 9.5M righe)
    df_mega = pd.read_parquet(str(MEGA_PATH), columns=cols_to_load)
    df_score = df_mega[df_mega[CIG_COL].isin(set(sampled_cigs))].copy()
    del df_mega

    for col, categories in cat_categories.items():
        if col in df_score.columns:
            # Mappa: valore → codice (stesso ordine del nativi training)
            code_map = {v: i for i, v in enumerate(categories)}
            df_score[col] = (df_score[col]
                             .map(code_map)          # unknown → NaN
                             .astype("float32"))

    X_score = df_score[features].to_numpy(dtype=np.float32)
    cigs    = df_score[CIG_COL].values

    print(f"  [U_score M{model_number}] X_score shape={X_score.shape}")
    return X_score, cigs



def run_bootstrap(model_number: int, gamma: float,
                  n_boot: int = N_BOOT_DEF,
                  n_score: int | None = None,
                  seed: int = BOOT_SEED) -> None:
    """Esegue bootstrap CI per re_lgbm su dataset M{model_number}.

    Salva results/re_lgbm_M{model_number}/bootstrap/{scores_ci,metrics_ci}.csv
    """
    if n_score is None:
        n_score = N_SCORE_DEF[model_number]

    res_folder = _result_folder(model_number, gamma)
    out_dir    = res_folder / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_ci_path  = out_dir / "scores_ci.csv"
    metrics_ci_path = out_dir / "metrics_ci.csv"
    cigs_cache      = out_dir / "u_score_cigs.npy"

    print(f"  [{_ts()}] Bootstrap CI — re_lgbm  M{model_number}  gamma={gamma}  pi={PI_FIXED}")
    print(f"  B={n_boot}  n_score={n_score:,}  seed={seed}")
    print(f"  Output: {out_dir}")

    print(f"\n  Fold map (seed={CALIB_SEED})...")
    fold_map = compute_fold_map_hierarchical(seed=CALIB_SEED)
    print("  Caricamento dati training...")
    df, features, cat_idx = load_data("re_lgbm", model_number, fold_map)

    label = df[LABEL_COL].values.astype(float)
    X_all = df[features].to_numpy(dtype=np.float32)

    # Matrici per bootstrap
    X_P = X_all[label == 1];   n_P = len(X_P)
    X_N = X_all[label == 0];   n_N = len(X_N)
    X_U = X_all[np.isnan(label)]

    # Labeled per valutazione metriche
    X_labeled    = np.vstack([X_P, X_N])
    label_labeled = np.concatenate([np.ones(n_P), np.zeros(n_N)])

    training_cigs = set(df[CIG_COL].values)
    print(f"  Training: {n_P} P, {n_N} N, {len(X_U):,} U")

    # cat_idx è la lista degli indici; per il nome occorre la lista originale
    # Ri-deriva i cat_names dal nativi parquet (i col con dtype float32
    # che originariamente erano object/category vengono tracciati in cat_idx)
    df_nativi_raw_full = pd.read_parquet(
        Path(NATIVI_PATH) / f"M{model_number}.parquet",
        columns=None)          # carica tutto per avere i dtype originali
    cat_names_orig = [c for c in df_nativi_raw_full.columns
                      if c not in {LABEL_COL, FOLD_COL, CIG_COL,
                                   "esito", "anno_pubblicazione", "regione"}
                      and df_nativi_raw_full[c].dtype.kind not in "iufb"]
    del df_nativi_raw_full

    if cigs_cache.exists() and scores_ci_path.exists():
        print("\n  Bootstrap già completato. Elimina i file per rigenerare.")
        return

    if cigs_cache.exists():
        print(f"\n  Carico U_score CIG dalla cache: {cigs_cache}")
        score_cigs = np.load(str(cigs_cache), allow_pickle=True)
        # Ricarica X_score dal mega dataset con gli stessi CIG
        cols_to_load = [CIG_COL] + features
        df_mega = pd.read_parquet(str(MEGA_PATH), columns=cols_to_load)
        df_score_rows = df_mega[df_mega[CIG_COL].isin(set(score_cigs))].copy()
        del df_mega
        X_score = df_score_rows[features].to_numpy(dtype=np.float32)
        print(f"  X_score shape={X_score.shape}")
    else:
        X_score, score_cigs = sample_u_score(
            model_number, training_cigs, features, cat_names_orig,
            df, n_score, seed)
        np.save(str(cigs_cache), score_cigs)

    platt_path = res_folder / "calibration" / "platt_params.csv"
    if platt_path.exists():
        platt_df  = pd.read_csv(platt_path)
        platt_row = platt_df[platt_df["fold"] == "global"].iloc[0]
        a_platt   = float(platt_row["a"])
        b_platt   = float(platt_row["b"])
        print(f"\n  Platt globale: a={a_platt:.4f}  b={b_platt:.4f}")
    else:
        print(f"\n  [WARN] Platt params non trovati in {platt_path}.")
        print("    Calibrazione non applicata agli score CI (raw scores).")
        a_platt, b_platt = 1.0, 0.0   # identità

    # Prediciamo tutto in una chiamata per efficienza
    n_labeled = len(X_labeled)
    X_te_all  = np.vstack([X_labeled, X_score])

    rng = np.random.default_rng(seed)

    boot_scores_score   = np.empty((n_boot, len(X_score)),   dtype=np.float32)
    boot_scores_labeled = np.empty((n_boot, n_labeled),      dtype=np.float32)
    boot_iters          = np.empty(n_boot, dtype=int)

    t_boot_start = datetime.now()
    print(f"\n  [{_ts()}] Avvio bootstrap (B={n_boot})...")
    for b in range(n_boot):
        t_b = datetime.now()

        # Bootstrap con rimpiazzo: campiona separatamente P, N, U
        idx_P = rng.integers(0, n_P,      size=n_P)
        idx_N = rng.integers(0, n_N,      size=n_N)
        idx_U = rng.integers(0, len(X_U), size=len(X_U))

        X_P_b = X_P[idx_P]
        X_N_b = X_N[idx_N]
        X_U_b = X_U[idx_U]

        scores_te, best_iter = fit_predict(
            X_P_b, X_N_b, X_U_b, X_te_all,
            gamma    = gamma,
            pi       = PI_FIXED,     # override PI_BY_MODEL
            features = features,
            cat_idx  = cat_idx,
            rng      = np.random.default_rng(seed + b + 1),
        )

        boot_scores_labeled[b] = scores_te[:n_labeled]
        boot_scores_score[b]   = scores_te[n_labeled:]
        boot_iters[b]           = best_iter

        if (b + 1) % 10 == 0 or b == 0:
            elapsed_tot = (datetime.now() - t_boot_start).total_seconds()
            elapsed_b   = (datetime.now() - t_b).total_seconds()
            done        = b + 1
            eta_min     = (elapsed_tot / done) * (n_boot - done) / 60
            mean_iter   = boot_iters[:done].mean()
            print(f"  [{_ts()}] boot {done:3d}/{n_boot}  "
                  f"iter={best_iter}  mean_iter={mean_iter:.0f}  "
                  f"({elapsed_b:.1f}s/boot)  ETA ~{eta_min:.0f} min")

    elapsed_tot = (datetime.now() - t_boot_start).total_seconds()
    print(f"\n  [{_ts()}] Bootstrap completato in {elapsed_tot/60:.1f} min. "
          f"best_iter: mean={boot_iters.mean():.0f}  SD={boot_iters.std():.0f}")

    # raw score bootstrap → applica Platt → percentili
    boot_calib = sigmoid(a_platt * boot_scores_score + b_platt)

    scores_mean  = boot_calib.mean(axis=0)
    scores_lower = np.percentile(boot_calib, 2.5, axis=0)
    scores_upper = np.percentile(boot_calib, 97.5, axis=0)

    # Il Platt e calibrato su P+N con r = n_P/(n_P+n_N) >> pi.
    # Correzione logit: logit(p_corr) = logit(p_platt) + log(pi/(1-pi)) - log(r/(1-r))
    # Produce probabilita interpretabili nella popolazione reale (pi=0.02).
    r_labeled = n_P / (n_P + n_N)
    prior_shift = (np.log(PI_FIXED / (1 - PI_FIXED))
                   - np.log(r_labeled / (1 - r_labeled)))

    def _prior_correct(p: np.ndarray) -> np.ndarray:
        p_clip = np.clip(p, 1e-7, 1 - 1e-7)
        return sigmoid(np.log(p_clip / (1 - p_clip)) + prior_shift)

    pc_mean  = _prior_correct(scores_mean)
    pc_lower = _prior_correct(scores_lower)
    pc_upper = _prior_correct(scores_upper)

    print(f"\n  Prior correction: r_labeled={r_labeled:.3f}  pi={PI_FIXED}  "
          f"logit_shift={prior_shift:.3f}")
    print(f"  Score Platt medio (U): {scores_mean.mean():.4f}  "
          f"Score prior-corrected medio: {pc_mean.mean():.4f}  "
          f"(atteso ~{PI_FIXED:.3f})")

    df_scores_ci = pd.DataFrame({
        CIG_COL:          score_cigs,
        "score_mean":     scores_mean.round(6),
        "score_lower":    scores_lower.round(6),
        "score_upper":    scores_upper.round(6),
        "pc_mean":        pc_mean.round(6),
        "pc_lower":       pc_lower.round(6),
        "pc_upper":       pc_upper.round(6),
        "lift_vs_prior":  (pc_mean / PI_FIXED).round(4),
        "percentile_rank": (pd.Series(scores_mean).rank(pct=True) * 100).round(2).values,
    }).sort_values("score_mean", ascending=False).reset_index(drop=True)

    df_scores_ci.to_csv(scores_ci_path, index=False)
    print(f"\n  Scores CI salvati: {scores_ci_path}")
    top5_cols = [CIG_COL, "pc_mean", "pc_lower", "pc_upper", "lift_vs_prior", "percentile_rank"]
    print(f"    Top-5 (prior-corrected):\n{df_scores_ci[top5_cols].head().to_string(index=False)}")

    metrics_rows = []
    label_P = label_labeled == 1
    label_N = label_labeled == 0

    for b in range(n_boot):
        s = boot_scores_labeled[b]
        # eval_all richiede (scores_pos, scores_neg, scores_unl)
        # Non abbiamo U labeled → passiamo array vuoto
        m = eval_all(s[label_P], s[label_N], np.array([]))
        metrics_rows.append(m)

    df_metrics = pd.DataFrame(metrics_rows)

    metrics_ci_rows = []
    for col in df_metrics.columns:
        vals = df_metrics[col].dropna().values
        metrics_ci_rows.append({
            "metrica": col,
            "mean":    float(np.mean(vals)),
            "lower":   float(np.percentile(vals, 2.5)),
            "upper":   float(np.percentile(vals, 97.5)),
            "sd":      float(np.std(vals)),
        })

    df_metrics_ci = pd.DataFrame(metrics_ci_rows)
    df_metrics_ci.to_csv(metrics_ci_path, index=False)
    print(f"\n  Metrics CI salvati: {metrics_ci_path}")
    print(df_metrics_ci.to_string(index=False))

    print(f"\n  OK Bootstrap M{model_number} completato. Output: {out_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CI — re_lgbm")
    parser.add_argument("--datasets",   nargs="+", type=int,
                        default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--n-boot",     type=int, default=N_BOOT_DEF)
    parser.add_argument("--n-score",    type=int, default=None,
                        help="Righe U_score (default: 10k M1/M2, 3k M3)")
    parser.add_argument("--seed",       type=int, default=BOOT_SEED)
    parser.add_argument("--gamma-config", type=Path, default=GAMMA_PATH)
    args = parser.parse_args()

    gamma_star = load_gamma_star(args.gamma_config)

    for mn in args.datasets:
        gamma = gamma_star["re_lgbm"][mn]
        n_sc  = args.n_score if args.n_score else N_SCORE_DEF[mn]
        run_bootstrap(model_number=mn, gamma=gamma,
                      n_boot=args.n_boot, n_score=n_sc, seed=args.seed)
