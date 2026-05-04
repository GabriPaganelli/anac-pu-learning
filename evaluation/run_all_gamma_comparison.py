"""
run_all_gamma_comparison.py — Esegui ex_post su un modello × dataset con molteplici γ.

Lancia calibration → conformal → SHAP per ogni γ specificato su un dato modello/dataset.
Default: re_lgbm M3 con γ={0.33, 1.0}.

Output: results/{model}_{Mn}_g{gamma_str}/ per ogni γ, senza sovrascritture.
ATTENZIONE: non usare run_all.py --models re_lgbm --datasets 3 separatamente,
perché creerebbe results/re_lgbm_M3/ come copia non etichettata.

Uso:
  python run_all_gamma_comparison.py                              # re_lgbm M3, γ=0.33 e 1.0
  python run_all_gamma_comparison.py --gammas 0.33 0.5 1.0       # γ custom
  python run_all_gamma_comparison.py --model bagging_lgbm --dataset 2 --gammas 0.66 1.0
  python run_all_gamma_comparison.py --gammas 0.33 1.0 --steps calibration shap
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from shutil import rmtree

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils_expost import load_gamma_star, ALL_MODELS, ALL_DATASETS
from calibration   import run_calibration
from conformal     import run_conformal
from shap_analysis import run_shap

GAMMA_JSON = _HERE / "gamma_star.json"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def format_gamma_suffix(gamma: float) -> str:
    """g0.33 o g1.0"""
    return f"g{gamma:.2f}".replace(".", "")


def run_gamma_series(
    gammas: list,
    steps: list,
    model_name: str,
    model_number: int,
    alpha: float,
    n_shap_sample: int,
    n_shap_rows: int,
    force_retrain: bool,
) -> None:
    """Esegue calibration → conformal → SHAP per ogni γ su model_name × model_number.

    Output salvato in results/{model_name}_M{model_number}_g{gamma:.2f}/.
    """
    print(f"  gamma-Comparison: {model_name} M{model_number}  |  "
          f"gamma={gammas}  ({_ts()})")

    key = f"{model_name}_M{model_number}"

    for gamma in gammas:
        gamma_suffix = format_gamma_suffix(gamma)
        out_prefix = f"{key}_{gamma_suffix}"

        print(f"  {out_prefix}  ({_ts()})")

        t0 = datetime.now()
        results = {}

        if "calibration" in steps:
            print(f"\n  [{_ts()}] Step 1/3: calibrazione...")
            res_cal = run_calibration(
                model_name, model_number, gamma, force_retrain=force_retrain)
            results["calibration"] = res_cal
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  OK Calibrazione  ({elapsed:.0f}s)")

        if "conformal" in steps:
            t1 = datetime.now()
            print(f"\n  [{_ts()}] Step 2/3: conformal...")
            res_conf = run_conformal(model_name, model_number, gamma, alpha=alpha)
            results["conformal"] = res_conf
            elapsed = (datetime.now() - t1).total_seconds()
            cov = res_conf["coverage_mean"]
            print(f"  OK Conformal  ({elapsed:.0f}s)  "
                  f"copertura classe1={cov.get('copertura_classe1', float('nan')):.3f}")

        if "shap" in steps:
            t2 = datetime.now()
            print(f"\n  [{_ts()}] Step 3/3: SHAP...")
            res_shap = run_shap(
                model_name, model_number, gamma,
                n_shap_sample=n_shap_sample,
                n_shap_rows=n_shap_rows,
                force_retrain=force_retrain)
            results["shap"] = res_shap
            elapsed = (datetime.now() - t2).total_seconds()
            print(f"  OK SHAP  ({elapsed:.0f}s)")

        total = (datetime.now() - t0).total_seconds()
        print(f"\n  OK {out_prefix} completato in {total/60:.1f} min")

        # Rinomina cartelle da {model}_{Mn} a {model}_{Mn}_g{gamma}
        # (le funzioni salvano con il nome base, noi rinominiamo dopo)
        if "calibration" in steps:
            old_dir = _HERE / "results" / key
            new_dir = _HERE / "results" / out_prefix
            if old_dir.exists() and old_dir != new_dir:
                if new_dir.exists():
                    rmtree(new_dir)
                old_dir.rename(new_dir)
                print(f"  [RINOMINATO] {key} → {out_prefix}")

    print(f"  Tutto completato  ({_ts()})")
    print(f"  Output: {_HERE / 'results'}")

    # Riepilogo rapido
    print("  Riepilogo risultati per gamma:")
    print(f"  {'gamma':<8}  {'ECE raw':>8}  {'ECE calib':>9}  {'Copertura cl1':>13}")
    for gamma in gammas:
        gamma_suffix = format_gamma_suffix(gamma)
        out_prefix = f"{key}_{gamma_suffix}"
        out_dir = _HERE / "results" / out_prefix

        if not out_dir.exists():
            continue

        try:
            df_m = pd.read_csv(out_dir / "calibration" / "ece_brier.csv")
            row  = df_m[df_m["fold"] == "overall"].iloc[0]
            print(f"  {gamma:<8.2f}  {row['ece_raw']:>8.4f}  "
                  f"{row['ece_calib']:>9.4f}  ", end="")

            df_c = pd.read_csv(out_dir / "conformal" / "coverage.csv")
            row_c = df_c[df_c["fold"] == "mean_oof"].iloc[0]
            print(f"{row_c.get('copertura_classe1', float('nan')):>13.3f}")
        except Exception:
            print(f"  {gamma:<8.2f}  [errore lettura]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Confronto γ per un modello × dataset")
    parser.add_argument(
        "--model", default="re_lgbm", choices=ALL_MODELS,
        help="Modello da analizzare (default: re_lgbm)")
    parser.add_argument(
        "--dataset", type=int, default=3, choices=ALL_DATASETS,
        help="Dataset (1=M1, 2=M2, 3=M3 — default: 3)")
    parser.add_argument(
        "--gammas", nargs="+", type=float, default=[0.33, 1.0],
        help="Valori γ da testare (default: 0.33 1.0)")
    parser.add_argument(
        "--steps", nargs="+", default=["calibration", "conformal", "shap"],
        choices=["calibration", "conformal", "shap"],
        help="Step da eseguire")
    parser.add_argument(
        "--alpha", type=float, default=0.10,
        help="Livello di errore per conformal (default 0.10)")
    parser.add_argument(
        "--n-shap-sample", type=int, default=50,
        help="Boosters da campionare per SHAP")
    parser.add_argument(
        "--n-shap-rows", type=int, default=10_000,
        help="Righe per SHAP")
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Riaddestrare modelli anche se salvati")
    args = parser.parse_args()

    run_gamma_series(
        gammas        = args.gammas,
        steps         = args.steps,
        model_name    = args.model,
        model_number  = args.dataset,
        alpha         = args.alpha,
        n_shap_sample = args.n_shap_sample,
        n_shap_rows   = args.n_shap_rows,
        force_retrain = args.force_retrain,
    )
