"""
run_all.py — Wrapper ex-post: calibrazione → conformal → SHAP.
Esegue i 3 step in sequenza per ogni combinazione modello × dataset.

Ordine obbligatorio:
  1. calibration.py  (seed=2026): Platt OOF + modello finale
  2. conformal.py    (seed=2027): Mondrian su score Platt-calibrati
  3. shap_analysis.py           : SHAP sul modello finale

conformal.py dipende dall'output di calibration.py (platt_params.csv).
shap_analysis.py usa il modello finale salvato da calibration.py.

Uso:
  python run_all.py                              # tutto
  python run_all.py --models lgbm_supervised      # solo un modello
  python run_all.py --datasets 3                 # solo M3
  python run_all.py --steps calibration shap     # salta conformal
  python run_all.py --n-shap-sample 100          # più boosters per SHAP
  python run_all.py --n-shap-rows 5000           # righe per SHAP (default 10_000)
  python run_all.py --force-retrain              # raddestra anche se già salvato
  python run_all.py --alpha 0.05                 # conformal con α=0.05
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils_expost import load_gamma_star, ALL_MODELS, ALL_DATASETS
from calibration   import run_calibration
from conformal     import run_conformal
from shap_analysis import run_shap

ALL_STEPS = ["calibration", "conformal", "shap"]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main(models: list, datasets: list, steps: list,
         gamma_config: Path,
         alpha: float,
         n_shap_sample: int,
         n_shap_rows: int,
         force_retrain: bool) -> None:

    gamma_star = load_gamma_star(gamma_config)

    print(f"  Ex-post analysis — PU Learning  ({_ts()})")
    print(f"  Modelli:  {models}")
    print(f"  Dataset:  M{datasets}")
    print(f"  Step:     {steps}")
    print(f"  gamma config: {gamma_config}")
    print(f"  SHAP rows: {n_shap_rows:,}")

    results = {}

    for mn in datasets:
        for mod in models:
            gamma = gamma_star[mod][mn]
            key   = f"{mod}_M{mn}"

            print(f"  {key}  gamma={gamma}  ({_ts()})")

            results[key] = {}
            t0 = datetime.now()

            if "calibration" in steps:
                print(f"\n  [{_ts()}] Step 1/3: calibrazione...")
                res_cal = run_calibration(
                    mod, mn, gamma, force_retrain=force_retrain)
                results[key]["calibration"] = res_cal
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"  OK Calibrazione  ({elapsed:.0f}s)  "
                      f"ECE global: raw→calib via platt_params.csv")

            if "conformal" in steps:
                t1 = datetime.now()
                print(f"\n  [{_ts()}] Step 2/3: conformal...")
                res_conf = run_conformal(mod, mn, gamma, alpha=alpha)
                results[key]["conformal"] = res_conf
                elapsed = (datetime.now() - t1).total_seconds()
                cov = res_conf["coverage_mean"]
                print(f"  OK Conformal  ({elapsed:.0f}s)  "
                      f"copertura classe1={cov.get('copertura_classe1', float('nan')):.3f}  "
                      f"classe0={cov.get('copertura_classe0', float('nan')):.3f}")

            if "shap" in steps:
                t2 = datetime.now()
                print(f"\n  [{_ts()}] Step 3/3: SHAP "
                      f"(n_sample={n_shap_sample}, n_rows={n_shap_rows:,})...")
                res_shap = run_shap(
                    mod, mn, gamma,
                    n_shap_sample=n_shap_sample,
                    n_shap_rows=n_shap_rows,
                    force_retrain=force_retrain)
                results[key]["shap"] = res_shap
                elapsed = (datetime.now() - t2).total_seconds()
                print(f"  OK SHAP  ({elapsed:.0f}s)  "
                      f"top feature: {res_shap['top_features'][:3]}")

            total = (datetime.now() - t0).total_seconds()
            print(f"\n  OK {key} completato in {total/60:.1f} min")

    print(f"  Tutto completato  ({_ts()})")
    print(f"  Output root: {_HERE / 'results'}")

    # Riepilogo rapido
    if "calibration" in steps and results:
        print("\n  Riepilogo ECE global Platt:")
        print(f"  {'Modello':<22} {'M':>2}  {'ECE raw':>8}  {'ECE calib':>9}")
        for key, res in results.items():
            if "calibration" not in res:
                continue
            out_dir = res["calibration"]["out_dir"]
            try:
                df_m = pd.read_csv(out_dir / "calibration" / "ece_brier.csv")
                row  = df_m[df_m["fold"] == "overall"].iloc[0]
                mod_name, mn_str = key.rsplit("_M", 1)
                print(f"  {mod_name:<22} {mn_str:>2}  "
                      f"{row['ece_raw']:>8.4f}  {row['ece_calib']:>9.4f}")
            except Exception:
                pass

    if "conformal" in steps and results:
        print("\n  Riepilogo copertura conformal (mean OOF):")
        print(f"  {'Modello':<22} {'M':>2}  "
              f"{'cop cl1':>8}  {'cop cl0':>8}  {'amp med':>8}")
        for key, res in results.items():
            if "conformal" not in res:
                continue
            cov = res["conformal"]["coverage_mean"]
            mod_name, mn_str = key.rsplit("_M", 1)
            print(f"  {mod_name:<22} {mn_str:>2}  "
                  f"{cov.get('copertura_classe1', float('nan')):>8.3f}  "
                  f"{cov.get('copertura_classe0', float('nan')):>8.3f}  "
                  f"{cov.get('ampiezza_media', float('nan')):>8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ex-post analysis completa per i 4 finalisti PU")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS,
        help="Modelli da processare")
    parser.add_argument(
        "--datasets", nargs="+", type=int, default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Dataset (1=M1, 2=M2, 3=M3)")
    parser.add_argument(
        "--steps", nargs="+", default=ALL_STEPS, choices=ALL_STEPS,
        help="Step da eseguire (default: tutti)")
    parser.add_argument(
        "--gamma-config", type=Path,
        default=_HERE / "gamma_star.json",
        help="Path a gamma_star.json")
    parser.add_argument(
        "--alpha", type=float, default=0.10,
        help="Livello di errore per conformal (default 0.10)")
    parser.add_argument(
        "--n-shap-sample", type=int, default=50,
        help="Boosters/bag da campionare per SHAP ensemble (default 50)")
    parser.add_argument(
        "--n-shap-rows", type=int, default=10_000,
        help="Righe da campionare per SHAP importance/beeswarm (default 10_000)")
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Raddestra modelli anche se già salvati")
    args = parser.parse_args()

    main(
        models        = args.models,
        datasets      = args.datasets,
        steps         = args.steps,
        gamma_config  = args.gamma_config,
        alpha         = args.alpha,
        n_shap_sample = args.n_shap_sample,
        n_shap_rows   = args.n_shap_rows,
        force_retrain = args.force_retrain,
    )
