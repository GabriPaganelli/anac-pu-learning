"""
calibration.py — Platt scaling per i 4 modelli finalisti.
Seed fold map: CALIB_SEED = 2026 (indipendente da mixing seed 2025).

Per ogni modello × dataset:
  1. OOF 4-fold (fold_map seed=2026)
     → scores raw su TUTTI i record del fold test (P + N + U)
  2. Platt per-fold: fit σ(a·s + b) sui labeled (P+N) di ciascun fold
  3. Platt global: fit unico su tutti gli OOF labeled concatenati
  4. Confronto: mean±SD dei parametri per-fold vs global
  5. Modello finale su tutti i dati → scores_final raw + calibrated (global)
  6. Diagnostiche: ECE, Brier per fold e overall
  7. Reliability plot (raw / per-fold-mean / global)

Output in results/{model}_{Mn}/:
  calibration/platt_params.csv        — a, b, SE per fold + riga "mean" + riga "global"
  calibration/ece_brier.csv           — ECE e Brier per fold e overall
  calibration/plots/reliability.{png,pdf}
  scores/scores_oof.csv               — cig, fold, label, score_raw,
                                        score_calib_pf (mean params),
                                        score_calib_gl (global params)
  scores/scores_final.csv             — cig, label, score_raw_final,
                                        score_calib_final (global params)
  models/final_model.{txt|pkl}        — modello finale serializzato
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
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils_expost import (
    CALIB_SEED,
    setup_out_dir, save_model, load_model, model_exists,
    load_gamma_star, load_data, split_fold, model_predict,
    PREPARE_FNS, OOF_FNS, FINAL_FIT_FNS,
    ALL_MODELS, ALL_DATASETS,
    LABEL_COL, FOLD_COL, CIG_COL,
)
from selected_models.utils import compute_fold_map_hierarchical



def _fit_platt(scores: np.ndarray, y: np.ndarray) -> tuple:
    """Fit σ(a·s + b) tramite LogisticRegression senza regolarizzazione.

    Ritorna (lr_model, a, b, se_a, se_b).
    se_a / se_b = errore standard dal Fisher information (Hessian^{-1}).
    """
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000, fit_intercept=True)
    lr.fit(scores.reshape(-1, 1), y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    se_a, se_b = _platt_se(scores, y, a, b)
    return lr, a, b, se_a, se_b


def _platt_se(scores: np.ndarray, y: np.ndarray,
              a: float, b: float) -> tuple[float, float]:
    """SE dei parametri Platt via Fisher information: Cov = (X^T W X)^{-1}."""
    p   = sigmoid(a * scores + b)
    w   = p * (1.0 - p)
    Xm  = np.column_stack([scores, np.ones(len(scores))])
    H   = (Xm * w[:, None]).T @ Xm   # (2,2) Hessian del log-lik
    try:
        cov  = np.linalg.inv(H)
        se_a = float(np.sqrt(max(cov[0, 0], 0.0)))
        se_b = float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        se_a = se_b = np.nan
    return se_a, se_b


def apply_platt(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    """Applica la trasformazione Platt: σ(a·s + b)."""
    return sigmoid(a * scores + b)



def compute_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (bin equal-width)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(probs) * abs(y[mask].mean() - probs[mask].mean())
    return float(ece)


def _calibration_metrics(scores: np.ndarray, y: np.ndarray,
                          a: float, b: float) -> dict:
    sc_cal = apply_platt(scores, a, b)
    return {
        "ece_raw":       compute_ece(scores, y),
        "ece_calib":     compute_ece(sc_cal,  y),
        "brier_raw":     float(brier_score_loss(y, scores)),
        "brier_calib":   float(brier_score_loss(y, sc_cal)),
    }



def _plot_reliability(sc_raw, sc_pf, sc_gl, y,
                      model_name, model_number, out_dir):
    """Reliability diagram: raw / per-fold-mean / global Platt."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    configs = [
        (axes[0], sc_raw, "Raw scores",          "steelblue"),
        (axes[1], sc_pf,  "Platt per-fold (mean)", "darkorange"),
        (axes[2], sc_gl,  "Platt global",         "seagreen"),
    ]
    for ax, sc, title, color in configs:
        try:
            frac_pos, mean_pred = calibration_curve(
                y, sc, n_bins=10, strategy="uniform")
        except ValueError:
            ax.set_title(f"{title}\n(dati insufficienti)")
            continue

        ece   = compute_ece(sc, y)
        brier = float(brier_score_loss(y, sc))

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfetta")
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=1.5, label=title)
        ax.set_xlabel("Probabilità predetta media", fontsize=9)
        ax.set_ylabel("Frazione di positivi", fontsize=9)
        ax.set_title(f"{title}\nECE={ece:.4f}  Brier={brier:.4f}", fontsize=9)
        ax.legend(fontsize=8)
        # Nota calibrazione su contratti investigati
        ax.text(
            0.02, 0.97,
            "Calibrato su P+N (contratti investigati)\n≠ popolazione generale degli appalti.",
            transform=ax.transAxes, fontsize=6.5, va="top", alpha=0.65,
        )
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Reliability diagram — {model_name}  M{model_number}", fontsize=11)
    fig.tight_layout()
    plot_dir = out_dir / "calibration" / "plots"
    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"reliability.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Reliability plot salvato in {plot_dir}")



def run_calibration(model_name: str, model_number: int, gamma: float,
                    force_retrain: bool = False) -> dict:
    """Calibrazione Platt completa per un modello × dataset.

    Ritorna dict con i parametri Platt global — usato da conformal.py.
    """
    print(f"  CALIBRAZIONE  {model_name}  M{model_number}  gamma={gamma}")
    print(f"  Fold seed: {CALIB_SEED}")

    out_dir = setup_out_dir(model_name, model_number)

    print(f"\n  Fold map gerarchico (seed={CALIB_SEED})...")
    fold_map = compute_fold_map_hierarchical(seed=CALIB_SEED)

    print("  Caricamento dati...")
    df, features, cat_idx = load_data(model_name, model_number, fold_map)
    label_all = df[LABEL_COL].values.astype(float)
    fold_all  = df[FOLD_COL].values.astype(float)
    cig_all   = df[CIG_COL].values

    state = PREPARE_FNS[model_name](df, features, cat_idx, gamma, model_number)

    scores_oof_raw    = np.full(len(df), np.nan)
    platt_fold_rows   = []

    for fold_test in range(4):
        print(f"\n  ── Fold {fold_test} ──────────────────────────────────────")
        sp = split_fold(df, features, fold_test)
        print(f"    Train: {len(sp['X_P_tr'])} P  {len(sp['X_N_tr'])} N  "
              f"{len(sp['X_U_tr'])} U")
        print(f"    Test:  {int((sp['label_te']==1).sum())} P  "
              f"{int((sp['label_te']==0).sum())} N  "
              f"{int(np.isnan(sp['label_te']).sum())} U  — fitting...")

        sc_te = OOF_FNS[model_name](
            sp, gamma, model_number, features, cat_idx, state)
        scores_oof_raw[sp["idx_te"]] = sc_te

        # Platt su P+N del fold k
        lab_te  = sp["label_te"]
        mask_pn = np.isfinite(lab_te)
        sc_pn   = sc_te[mask_pn]
        y_pn    = lab_te[mask_pn].astype(int)
        n_P_te  = int((y_pn == 1).sum())
        n_N_te  = int((y_pn == 0).sum())

        print(f"    Platt fit: {n_P_te} P + {n_N_te} N ...")
        _, a_k, b_k, se_a_k, se_b_k = _fit_platt(sc_pn, y_pn)
        print(f"    a={a_k:.4f} ± {se_a_k:.4f}   b={b_k:.4f} ± {se_b_k:.4f}")

        platt_fold_rows.append({
            "fold": fold_test,
            "a": a_k, "b": b_k, "se_a": se_a_k, "se_b": se_b_k,
            "n_P": n_P_te, "n_N": n_N_te,
            "note": "per-fold",
        })

    mask_lab  = np.isfinite(label_all)
    sc_lab    = scores_oof_raw[mask_lab]
    y_lab     = label_all[mask_lab].astype(int)
    n_P_all   = int((y_lab == 1).sum())
    n_N_all   = int((y_lab == 0).sum())

    print(f"\n  Global Platt: {n_P_all} P + {n_N_all} N OOF totali ...")
    _, a_gl, b_gl, se_a_gl, se_b_gl = _fit_platt(sc_lab, y_lab)
    print(f"  a_global={a_gl:.4f} ± {se_a_gl:.4f}   "
          f"b_global={b_gl:.4f} ± {se_b_gl:.4f}")

    df_pf  = pd.DataFrame(platt_fold_rows)
    a_mean = float(df_pf["a"].mean())
    b_mean = float(df_pf["b"].mean())
    a_sd   = float(df_pf["a"].std(ddof=1))
    b_sd   = float(df_pf["b"].std(ddof=1))
    print(f"\n  Mean fold:  a={a_mean:.4f} (SD={a_sd:.4f})  "
          f"b={b_mean:.4f} (SD={b_sd:.4f})")

    if a_sd > 0.5 or b_sd > 0.5:
        print("  [WARN]  SD cross-fold elevata: considera di usare solo il Global Platt.")

    scores_oof_calib_pf = apply_platt(scores_oof_raw, a_mean, b_mean)
    scores_oof_calib_gl = apply_platt(scores_oof_raw, a_gl,   b_gl)

    if force_retrain or not model_exists(out_dir, "final_model", model_name):
        print("\n  Modello finale (tutti i dati)...")
        final_model = FINAL_FIT_FNS[model_name](
            df, features, cat_idx, gamma, model_number, state)
        save_model(final_model, out_dir, "final_model", model_name)
        print(f"  Salvato in {out_dir / 'models'}")
    else:
        print("\n  Carico modello finale esistente...")
        final_model = load_model(out_dir, "final_model", model_name)

    X_all             = df[features].to_numpy(dtype=np.float32)
    scores_final_raw  = model_predict(final_model, X_all, model_name)
    scores_final_cal  = apply_platt(scores_final_raw, a_gl, b_gl)

    metrics_rows = []
    for row in platt_fold_rows:
        k = row["fold"]
        idx_te = np.where(fold_all == k)[0]
        m_pn   = np.isfinite(label_all[idx_te])
        if m_pn.sum() == 0:
            continue
        sc_k = scores_oof_raw[idx_te][m_pn]
        y_k  = label_all[idx_te][m_pn].astype(int)
        m    = _calibration_metrics(sc_k, y_k, a_gl, b_gl)
        metrics_rows.append({
            "fold": k, "n_P": row["n_P"], "n_N": row["n_N"], **m})

    # overall
    m_overall = _calibration_metrics(sc_lab, y_lab, a_gl, b_gl)
    metrics_rows.append({
        "fold": "overall", "n_P": n_P_all, "n_N": n_N_all, **m_overall})
    print(f"\n  ECE overall:  raw={m_overall['ece_raw']:.4f}  "
          f"calib={m_overall['ece_calib']:.4f}")
    print(f"  Brier overall: raw={m_overall['brier_raw']:.4f}  "
          f"calib={m_overall['brier_calib']:.4f}")


    # platt_params.csv
    extra_rows = [
        {"fold": "mean",   "a": a_mean, "b": b_mean,
         "se_a": a_sd,    "se_b": b_sd,
         "n_P": None, "n_N": None,
         "note": "cross-fold mean (se_a/se_b = cross-fold SD)"},
        {"fold": "global", "a": a_gl,   "b": b_gl,
         "se_a": se_a_gl, "se_b": se_b_gl,
         "n_P": n_P_all, "n_N": n_N_all,
         "note": "global fit su tutti gli OOF labeled"},
    ]
    df_platt = pd.concat(
        [df_pf, pd.DataFrame(extra_rows)], ignore_index=True)
    df_platt.to_csv(out_dir / "calibration" / "platt_params.csv", index=False)

    # ece_brier.csv
    pd.DataFrame(metrics_rows).to_csv(
        out_dir / "calibration" / "ece_brier.csv", index=False)

    # scores_oof.csv  (tutti i record con fold assegnato, P+N+U)
    pd.DataFrame({
        "cig":            cig_all,
        "fold":           fold_all,
        "label":          label_all,
        "score_raw":      scores_oof_raw,
        "score_calib_pf": scores_oof_calib_pf,
        "score_calib_gl": scores_oof_calib_gl,
    }).to_csv(out_dir / "scores" / "scores_oof.csv", index=False)

    # scores_final.csv  (modello addestrato su tutti i dati)
    pd.DataFrame({
        "cig":               cig_all,
        "label":             label_all,
        "score_raw_final":   scores_final_raw,
        "score_calib_final": scores_final_cal,
    }).to_csv(out_dir / "scores" / "scores_final.csv", index=False)

    # Reliability plot
    _plot_reliability(
        sc_lab,
        scores_oof_calib_pf[mask_lab],
        scores_oof_calib_gl[mask_lab],
        y_lab, model_name, model_number, out_dir,
    )

    print(f"\n  OK Calibrazione completata. Output in: {out_dir}")

    return {
        "a_global":    a_gl,
        "b_global":    b_gl,
        "se_a_global": se_a_gl,
        "se_b_global": se_b_gl,
        "a_mean":      a_mean,
        "b_mean":      b_mean,
        "a_sd":        a_sd,
        "b_sd":        b_sd,
        "out_dir":     out_dir,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Platt calibration per i 4 modelli finalisti PU")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS)
    parser.add_argument(
        "--datasets", nargs="+", type=int, default=ALL_DATASETS,
        choices=ALL_DATASETS)
    parser.add_argument(
        "--gamma-config", type=Path,
        default=_HERE / "gamma_star.json",
        help="Path a gamma_star.json")
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Raddestra il modello finale anche se già salvato")
    args = parser.parse_args()

    gamma_star = load_gamma_star(args.gamma_config)

    for mn in args.datasets:
        for mod in args.models:
            gamma = gamma_star[mod][mn]
            run_calibration(mod, mn, gamma, force_retrain=args.force_retrain)
