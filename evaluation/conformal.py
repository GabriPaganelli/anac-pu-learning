"""
conformal.py — Split conformal class-conditional (Mondrian) per i 4 finalisti.
Seed fold map: CONF_SEED = 2027 (indipendente da calibration seed 2026).

I modelli vengono ri-addestrati da zero con la nuova struttura di fold (seed=2027).
Gli score raw vengono trasformati col Global Platt di calibration.py (caricato
da results/{model}_{Mn}/calibration/platt_params.csv).

Garanzie:
  - Copertura class-conditional (Mondrian): P(y ∈ set | y=k) ≥ 1−α
    garantita per P e N certi (scambiabilità dentro la stessa classe).
  - Per U: i p-value conformi sono RANKING robusti, non probabilità
    calibrate; la garanzia formale NON vale (selection bias P+N ≠ U).

Per ogni fold k:
  1. Addestra modello su fold ≠ k
  2. Applica Global Platt agli score raw → score calibrati
  3. Split 50/50 dei labeled del fold k:
       conformal_cal (50%) → soglie Mondrian q1, q0
       conformal_test (50%) → validazione copertura empirica
  4. Prediction set per tutti gli U del fold k
  5. P-value conforme verso classe 1 per U (ranking frodi)

Output in results/{model}_{Mn}/conformal/:
  thresholds.csv     — q1, q0, n1_cal, n0_cal per fold
  coverage.csv       — copertura per classe per fold + media OOF
  prediction_sets.csv — cig, label, set, ampiezza, pvalue_classe1
  plots/set_dist.{png,pdf}     — distribuzione dei set su U per fold
  plots/pvalue_hist.{png,pdf}  — istogramma p-value conformi su U
"""

import sys
import argparse
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

from utils_expost import (
    CONF_SEED,
    setup_out_dir, load_gamma_star, load_data, split_fold, model_predict,
    PREPARE_FNS, OOF_FNS,
    ALL_MODELS, ALL_DATASETS,
    LABEL_COL, FOLD_COL, CIG_COL,
)
from selected_models.utils import compute_fold_map_hierarchical

ALPHA       = 0.10    # errore target: copertura attesa ≥ 90%
CAL_FRAC    = 0.50    # 50% labeled del fold k per calibrazione conformal
SPLIT_SEED  = 42      # seed per il 50/50 split dei labeled (fisso, non fold)



def _nonconformity(prob1: np.ndarray, y: np.ndarray) -> np.ndarray:
    """s(x, y) = 1 − p̂(y|x).  Assume prob1 ∈ [0,1], y ∈ {0,1}."""
    return np.where(y == 1, 1.0 - prob1, prob1)



def _quantile_conforme(s_k: np.ndarray, alpha: float) -> float:
    """Quantile con correzione finite-sample: ⌈(n+1)(1−α)⌉ / n."""
    n = len(s_k)
    if n < 20:
        warnings.warn(
            f"n_cal={n}: quantile conforme instabile (< 20 campioni).",
            stacklevel=3)
    lvl = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(s_k, lvl, method="lower"))


def calibra_mondrian(prob1_cal: np.ndarray, y_cal: np.ndarray,
                     alpha: float = ALPHA) -> dict:
    """Soglie separate per classe 0 e classe 1 (split conformal Mondrian)."""
    s_cal = _nonconformity(prob1_cal, y_cal)
    s1    = s_cal[y_cal == 1]
    s0    = s_cal[y_cal == 0]
    return {
        "q1": _quantile_conforme(s1, alpha),
        "q0": _quantile_conforme(s0, alpha),
        "n1": int((y_cal == 1).sum()),
        "n0": int((y_cal == 0).sum()),
        "alpha": alpha,
    }



def predici_set(prob1: np.ndarray, cal: dict) -> pd.DataFrame:
    """Prediction set Mondrian per ciascuna osservazione.

    Set possibili: {1}, {0}, {0,1} (ambiguo), {} (vuoto, raro).
    """
    s1   = 1.0 - prob1
    s0   = prob1
    inc1 = s1 <= cal["q1"]
    inc0 = s0 <= cal["q0"]

    etichetta = np.where(
        inc1 & ~inc0, "{1}",
        np.where(~inc1 & inc0, "{0}",
                 np.where(inc1 & inc0, "{0,1}", "{}")))

    return pd.DataFrame({
        "prob1_cal": prob1,
        "include_0": inc0,
        "include_1": inc1,
        "set":       etichetta,
        "ampiezza":  inc0.astype(int) + inc1.astype(int),
    })



def pvalue_conforme_classe1(prob1_new: np.ndarray,
                             prob1_cal: np.ndarray,
                             y_cal: np.ndarray) -> np.ndarray:
    """p_i = (1 + #{s_cal_1 ≥ s_new_i}) / (n_cal_1 + 1).

    Utile per ranking degli U verso la classe positiva (rischio frode).
    Uniforme se il modello non ha segnale; concentrata a valori bassi se sì.
    """
    s_cal1  = _nonconformity(prob1_cal[y_cal == 1],
                             np.ones((y_cal == 1).sum(), dtype=int))
    s_new1  = _nonconformity(prob1_new,
                             np.ones(len(prob1_new), dtype=int))
    s_sorted = np.sort(s_cal1)
    n_k      = len(s_sorted)
    counts   = n_k - np.searchsorted(s_sorted, s_new1, side="left")
    return (1 + counts) / (n_k + 1)



def valida_copertura(sets_df: pd.DataFrame, y_true: np.ndarray) -> dict:
    y   = np.asarray(y_true, dtype=int)
    i1  = sets_df["include_1"].to_numpy()
    i0  = sets_df["include_0"].to_numpy()
    amp = sets_df["ampiezza"].to_numpy()

    cop_glob = float(np.mean(((y == 1) & i1) | ((y == 0) & i0)))
    cop_1    = float(i1[y == 1].mean()) if (y == 1).any() else np.nan
    cop_0    = float(i0[y == 0].mean()) if (y == 0).any() else np.nan

    return {
        "copertura_marginale": cop_glob,
        "copertura_classe1":   cop_1,
        "copertura_classe0":   cop_0,
        "ampiezza_media":      float(amp.mean()),
        "frac_singleton":      float((amp == 1).mean()),
        "frac_ambiguo":        float((amp == 2).mean()),
        "frac_vuoto":          float((amp == 0).mean()),
    }



def _load_global_platt(out_dir: Path) -> tuple[float, float]:
    """Legge a_global e b_global da platt_params.csv."""
    path = out_dir / "calibration" / "platt_params.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Platt params non trovati: {path}\n"
            "Esegui prima calibration.py per questo modello/dataset.")
    df = pd.read_csv(path)
    row = df[df["fold"] == "global"]
    if row.empty:
        raise ValueError("Riga 'global' non trovata in platt_params.csv.")
    return float(row["a"].iloc[0]), float(row["b"].iloc[0])



def _plot_set_distribution(sets_u_per_fold: list, model_name: str,
                           model_number: int, out_dir: Path) -> None:
    """Distribuzione dei prediction set su U per fold."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    categories = ["{0}", "{1}", "{0,1}", "{}"]
    colors     = ["steelblue", "tomato", "orange", "gray"]

    for fold_id, (ax, df_u) in enumerate(zip(axes, sets_u_per_fold)):
        counts = [int((df_u["set"] == c).sum()) for c in categories]
        total  = len(df_u)
        fracs  = [c / total for c in counts]
        bars   = ax.bar(categories, fracs, color=colors, edgecolor="white")
        ax.set_title(f"Fold {fold_id}", fontsize=9)
        ax.set_xlabel("Set predetto", fontsize=8)
        if fold_id == 0:
            ax.set_ylabel("Frazione U", fontsize=8)
        ax.bar_label(bars, labels=[f"{f:.2%}" for f in fracs],
                     fontsize=7, padding=2)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Distribuzione prediction set su U — {model_name} M{model_number}",
        fontsize=10)
    fig.tight_layout()
    plot_dir = out_dir / "conformal" / "plots"
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"set_dist.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_pvalue_hist(pv_list: list, fold_ids: list,
                      model_name: str, model_number: int,
                      out_dir: Path) -> None:
    """Istogramma p-value conformi verso classe 1 per gli U."""
    fig, axes = plt.subplots(1, len(pv_list), figsize=(4 * len(pv_list), 4),
                             sharey=False)
    if len(pv_list) == 1:
        axes = [axes]

    for ax, pv, fid in zip(axes, pv_list, fold_ids):
        ax.hist(pv, bins=20, range=(0, 1), color="steelblue",
                edgecolor="white", density=True)
        ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.5, label="Uniforme")
        ax.set_title(f"Fold {fid}", fontsize=9)
        ax.set_xlabel("p-value conforme (classe 1)", fontsize=8)
        ax.set_ylabel("Densità", fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"P-value conformi su U (classe 1) — {model_name} M{model_number}\n"
        "Uniforme = nessun segnale; concentrata vicino a 0 = segnale forte",
        fontsize=9)
    fig.tight_layout()
    plot_dir = out_dir / "conformal" / "plots"
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"pvalue_hist.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)



def run_conformal(model_name: str, model_number: int, gamma: float,
                  alpha: float = ALPHA) -> dict:
    """Conformal inference Mondrian per un modello × dataset.

    Presuppone che calibration.py sia già stato eseguito
    (legge Global Platt da platt_params.csv).
    """
    print(f"  CONFORMAL  {model_name}  M{model_number}  gamma={gamma}  alpha={alpha}")
    print(f"  Fold seed: {CONF_SEED}")

    out_dir = setup_out_dir(model_name, model_number)

    a_pl, b_pl = _load_global_platt(out_dir)
    print(f"\n  Global Platt caricato: a={a_pl:.4f}  b={b_pl:.4f}")

    print(f"\n  Fold map gerarchico (seed={CONF_SEED})...")
    fold_map = compute_fold_map_hierarchical(seed=CONF_SEED)

    print("  Caricamento dati...")
    df, features, cat_idx = load_data(model_name, model_number, fold_map)
    label_all = df[LABEL_COL].values.astype(float)
    fold_all  = df[FOLD_COL].values.astype(float)
    cig_all   = df[CIG_COL].values

    state = PREPARE_FNS[model_name](df, features, cat_idx, gamma, model_number)

    threshold_rows = []
    coverage_rows  = []
    sets_u_per_fold = []   # per il plot
    pv_u_per_fold   = []   # per il plot

    # Accumulatori per prediction_sets.csv (solo labeled e U)
    all_set_records = []

    rng_split = np.random.default_rng(SPLIT_SEED)

    for fold_test in range(4):
        print(f"\n  ── Fold {fold_test} ──────────────────────────────────────")
        sp = split_fold(df, features, fold_test)
        n_pn_te = int(np.isfinite(sp["label_te"]).sum())
        n_u_te  = int(np.isnan(sp["label_te"]).sum())
        print(f"    Train: {len(sp['X_P_tr'])} P  {len(sp['X_N_tr'])} N  "
              f"{len(sp['X_U_tr'])} U")
        print(f"    Test labeled: {n_pn_te}  U: {n_u_te}  — fitting...")

        # Score raw su TUTTI i record del fold k
        sc_te_raw = OOF_FNS[model_name](
            sp, gamma, model_number, features, cat_idx, state)

        # Applica Global Platt
        sc_te_cal = sigmoid(a_pl * sc_te_raw + b_pl)

        # Indici locali (nel fold k) per P+N e U
        lab_te   = sp["label_te"]
        mask_pn  = np.isfinite(lab_te)
        mask_u   = np.isnan(lab_te)
        idx_glob = sp["idx_te"]

        sc_pn_cal = sc_te_cal[mask_pn]
        y_pn      = lab_te[mask_pn].astype(int)
        sc_u_cal  = sc_te_cal[mask_u]

        # 50/50 split dei labeled per conformal cal / test
        n_pn  = len(sc_pn_cal)
        perm  = rng_split.permutation(n_pn)
        n_cal = max(1, int(np.floor(CAL_FRAC * n_pn)))
        idx_cal_loc = perm[:n_cal]
        idx_tst_loc = perm[n_cal:]

        sc_cal_c  = sc_pn_cal[idx_cal_loc]; y_cal_c = y_pn[idx_cal_loc]
        sc_test_c = sc_pn_cal[idx_tst_loc]; y_test_c = y_pn[idx_tst_loc]

        n1_cal = int((y_cal_c == 1).sum()); n0_cal = int((y_cal_c == 0).sum())
        n1_tst = int((y_test_c == 1).sum()); n0_tst = int((y_test_c == 0).sum())
        print(f"    Conf cal: {n1_cal} P + {n0_cal} N  |  "
              f"Test: {n1_tst} P + {n0_tst} N")

        # Calibrazione Mondrian
        cal = calibra_mondrian(sc_cal_c, y_cal_c, alpha)
        print(f"    q1={cal['q1']:.4f} (n1={cal['n1']})  "
              f"q0={cal['q0']:.4f} (n0={cal['n0']})")
        threshold_rows.append({
            "fold": fold_test,
            "q1": cal["q1"], "q0": cal["q0"],
            "n1_cal": cal["n1"], "n0_cal": cal["n0"],
            "alpha": alpha,
        })

        # Prediction set sul test labeled
        set_test = predici_set(sc_test_c, cal)
        cov = valida_copertura(set_test, y_test_c)
        print(f"    Copertura: classe1={cov['copertura_classe1']:.2f}  "
              f"classe0={cov['copertura_classe0']:.2f}  "
              f"amp_media={cov['ampiezza_media']:.2f}")
        coverage_rows.append({"fold": fold_test, **cov,
                               "n1_test": n1_tst, "n0_test": n0_tst})

        # Prediction set su U
        set_u = predici_set(sc_u_cal, cal)
        pv_u  = pvalue_conforme_classe1(sc_u_cal, sc_cal_c, y_cal_c)
        print(f"    Set U: {{1}}={int((set_u['set']=='{1}').sum())}  "
              f"{{0}}={int((set_u['set']=='{0}').sum())}  "
              f"{{0,1}}={int((set_u['set']=='{0,1}').sum())}  "
              f"{{}}={int((set_u['set']=='{}').sum())}")

        sets_u_per_fold.append(set_u)
        pv_u_per_fold.append(pv_u)

        # Accumula records per il CSV finale
        # — test labeled
        idx_pn_glob  = idx_glob[mask_pn]
        idx_tst_glob = idx_pn_glob[idx_tst_loc]
        for i, gi in enumerate(idx_tst_glob):
            all_set_records.append({
                "cig":             cig_all[gi],
                "fold":            fold_test,
                "label":           y_test_c[i],
                "split":           "labeled_test",
                "prob1_cal":       float(set_test["prob1_cal"].iloc[i]),
                "set":             set_test["set"].iloc[i],
                "ampiezza":        int(set_test["ampiezza"].iloc[i]),
                "pvalue_classe1":  np.nan,
            })
        # — U
        idx_u_glob = idx_glob[mask_u]
        for i, gi in enumerate(idx_u_glob):
            all_set_records.append({
                "cig":             cig_all[gi],
                "fold":            fold_test,
                "label":           np.nan,
                "split":           "U",
                "prob1_cal":       float(set_u["prob1_cal"].iloc[i]),
                "set":             set_u["set"].iloc[i],
                "ampiezza":        int(set_u["ampiezza"].iloc[i]),
                "pvalue_classe1":  float(pv_u[i]),
            })

    df_cov = pd.DataFrame(coverage_rows)
    mean_row = df_cov.drop(columns=["fold", "n1_test", "n0_test"]).mean().to_dict()
    mean_row.update({"fold": "mean_oof",
                     "n1_test": df_cov["n1_test"].sum(),
                     "n0_test": df_cov["n0_test"].sum()})
    df_cov = pd.concat([df_cov, pd.DataFrame([mean_row])], ignore_index=True)

    print(f"\n  Copertura media OOF:")
    print(f"    classe1={mean_row['copertura_classe1']:.3f}  "
          f"classe0={mean_row['copertura_classe0']:.3f}  "
          f"amp={mean_row['ampiezza_media']:.3f}")

    if mean_row["copertura_classe1"] < (1 - alpha - 0.05):
        print(f"  [WARN]  Copertura classe1 < {1-alpha-0.05:.2f}: "
              "n_cal_classe1 probabilmente troppo basso.")

    pd.DataFrame(threshold_rows).to_csv(
        out_dir / "conformal" / "thresholds.csv", index=False)
    df_cov.to_csv(out_dir / "conformal" / "coverage.csv", index=False)
    pd.DataFrame(all_set_records).to_csv(
        out_dir / "conformal" / "prediction_sets.csv", index=False)

    _plot_set_distribution(sets_u_per_fold, model_name, model_number, out_dir)
    _plot_pvalue_hist(pv_u_per_fold, list(range(4)),
                      model_name, model_number, out_dir)

    print(f"\n  OK Conformal completato. Output in: {out_dir / 'conformal'}")
    return {"coverage_mean": mean_row, "out_dir": out_dir}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conformal prediction (Mondrian) per i 4 finalisti PU")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument(
        "--datasets", nargs="+", type=int, default=ALL_DATASETS,
        choices=ALL_DATASETS)
    parser.add_argument(
        "--gamma-config", type=Path,
        default=_HERE / "gamma_star.json")
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"Livello di errore target (default {ALPHA})")
    args = parser.parse_args()

    gamma_star = load_gamma_star(args.gamma_config)

    for mn in args.datasets:
        for mod in args.models:
            gamma = gamma_star[mod][mn]
            run_conformal(mod, mn, gamma, alpha=args.alpha)
