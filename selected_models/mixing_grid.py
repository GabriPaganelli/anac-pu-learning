"""
PNU Mixing — Wrapper di valutazione su griglia γ (Roadmap §5).

Esegue i 4 modelli finalisti su tutti i dataset (M1, M2, M3) per ogni
valore di γ nella griglia. Salva i risultati in results/.

Struttura output:
  results/mixing_{model}_{Mn}_fold_metrics.csv  — lift/auc per fold × γ
  results/mixing_{model}_{Mn}_summary.csv       — media ± SD per γ

Reshuffle fold (gerarchico M3→M2→M1):
  I fold vengono reshuffati una sola volta prima del loop γ.
  Strategia gerarchica: M3 (più vincolato) viene reshuffato per primo,
  poi i CIG aggiuntivi di M2, poi quelli aggiuntivi di M1.
  Questo garantisce che ogni dataset abbia fold bilanciati localmente
  (non solo M1 che è il superset). Stessi P/N/U per fold dell'originale,
  assegnazioni diverse → indipendenza dalla HP selection precedente.

  Il fold_map viene sempre calcolato su tutti e tre i dataset (slim load
  di sole 3 colonne), indipendentemente da --datasets, per consistenza.

Griglia γ:
  Default:     [0, 0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0]
  LGB supervisionato: [0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0]
  (γ=0 escluso: senza N certi il classificatore binario non ha negativi)

Modelli:
  lgb_supervised  — LGB P vs N certi (EM-Hard iter=0, Python rewrite)
  bagging_lgbm    — Bagging LightGBM PNPU
  re_lgbm         — RE LightGBM (nnPU loss, PNPU)
  puet            — PU Extra Trees (usa preprocessed; PNPU)

Uso:
  python mixing_grid.py                        # tutti i modelli, M1..M3
  python mixing_grid.py --models re_lgbm puet  # solo due modelli
  python mixing_grid.py --datasets 3           # solo M3
  python mixing_grid.py --gamma 0.5 1.0        # solo due valori di γ
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Root del progetto
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from selected_models.utils import (
    load_nativi_raw, load_preprocessed_raw,
    compute_fold_map_hierarchical, apply_fold_map,
    get_features, cat_names_to_idx,
    KS, RESHUFFLE_SEED,
)
import selected_models.lgb_supervised as lgb_sup
import selected_models.bagging_lgbm   as bag_lgb
import selected_models.re_lgbm        as re_lgb
import selected_models.puet           as puet_mod

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

GAMMA_DEFAULT    = [0.0, 0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0]
GAMMA_SUPERVISED = [0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0]
DATASETS         = [1, 2, 3]
ALL_MODELS       = ["lgb_supervised", "bagging_lgbm", "re_lgbm", "puet"]



def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _summary_from_folds(fold_rows: list) -> dict:
    """Calcola media ± SD delle metriche per un γ dato."""
    df = pd.DataFrame(fold_rows)
    summary = {"gamma": df["gamma"].iloc[0]}
    lift_cols = [f"lift@{k*100:g}%" for k in KS]
    for col in lift_cols + ["pr_auc", "roc_auc"]:
        if col in df.columns:
            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_sd"]   = df[col].std(ddof=1)
    return summary


def _eta_str(t_loop_start: datetime, done: int, total: int) -> str:
    """Stima ETA in base al tempo medio per γ completati."""
    if done == 0:
        return ""
    elapsed = (datetime.now() - t_loop_start).total_seconds()
    avg     = elapsed / done
    rem     = total - done
    if rem <= 0:
        return "ultimo γ"
    mins = avg * rem / 60
    return f"ETA ~{mins:.0f} min ({rem} γ rimanenti)"


def _save_results(model_name: str, model_number: int,
                  all_fold_rows: list, all_summary_rows: list) -> None:
    prefix = f"mixing_{model_name}_M{model_number}"
    pd.DataFrame(all_fold_rows).to_csv(
        OUT_DIR / f"{prefix}_fold_metrics.csv", index=False)
    pd.DataFrame(all_summary_rows).to_csv(
        OUT_DIR / f"{prefix}_summary.csv", index=False)
    print(f"  [SAVED] {prefix}_*.csv")



def run_lgb_supervised(df_nativi: pd.DataFrame, model_number: int,
                       gamma_grid: list, cat_names: list) -> tuple[list, list]:
    """LGB supervisionato P vs N certi.
    Tuning globale UNA volta, poi sweep γ.
    """
    features = get_features(df_nativi)
    cat_idx  = cat_names_to_idx(cat_names, features) or None

    label = df_nativi["label"].values.astype(float)
    X     = df_nativi[features].to_numpy(dtype=np.float32)
    X_P_all = X[label == 1]
    X_N_all = X[label == 0]

    print(f"\n  [{_ts()}] Tuning globale LGB supervised M{model_number}...")
    best_params = lgb_sup.global_tune(X_P_all, X_N_all, cat_idx=cat_idx)

    all_fold_rows, all_summary_rows = [], []
    t_loop = datetime.now()

    for i, gamma in enumerate(gamma_grid):
        t_g = datetime.now()
        print(f"\n  [{_ts()}] lgb_supervised M{model_number} gamma={gamma:.2f}  "
              f"({i+1}/{len(gamma_grid)})  {_eta_str(t_loop, i, len(gamma_grid))}")
        _, fold_metrics = lgb_sup.run_oof(
            df_nativi, gamma, best_params, features, cat_idx)
        fold_rows = list(fold_metrics.values())
        all_fold_rows.extend(fold_rows)
        all_summary_rows.append(_summary_from_folds(fold_rows))
        elapsed_g = (datetime.now() - t_g).total_seconds()
        row = all_summary_rows[-1]
        print(f"  OK gamma={gamma:.2f}  lift@1%={row.get('lift@1%_mean', float('nan')):.2f}x"
              f"  PR={row.get('pr_auc_mean', float('nan')):.3f}  ({elapsed_g:.0f}s)")

    return all_fold_rows, all_summary_rows


def run_bagging_lgbm(df_nativi: pd.DataFrame, model_number: int,
                     gamma_grid: list, cat_names: list) -> tuple[list, list]:
    """Bagging LGB PNPU con γ-weighted N certi."""
    features = get_features(df_nativi)
    cat_idx  = cat_names_to_idx(cat_names, features) or None

    all_fold_rows, all_summary_rows = [], []
    t_loop = datetime.now()

    for i, gamma in enumerate(gamma_grid):
        t_g = datetime.now()
        print(f"\n  [{_ts()}] bagging_lgbm M{model_number} gamma={gamma:.2f}  "
              f"({i+1}/{len(gamma_grid)})  {_eta_str(t_loop, i, len(gamma_grid))}")
        _, fold_metrics = bag_lgb.run_oof(
            df_nativi, gamma, features, cat_idx)
        fold_rows = list(fold_metrics.values())
        all_fold_rows.extend(fold_rows)
        all_summary_rows.append(_summary_from_folds(fold_rows))
        elapsed_g = (datetime.now() - t_g).total_seconds()
        row = all_summary_rows[-1]
        print(f"  OK gamma={gamma:.2f}  lift@1%={row.get('lift@1%_mean', float('nan')):.2f}x"
              f"  PR={row.get('pr_auc_mean', float('nan')):.3f}  ({elapsed_g:.0f}s)")

    return all_fold_rows, all_summary_rows


def run_re_lgbm(df_nativi: pd.DataFrame, model_number: int,
                gamma_grid: list, cat_names: list) -> tuple[list, list]:
    """RE LGB (nnPU loss) con γ scalato sul termine gradiente N certi."""
    features = get_features(df_nativi)
    cat_idx  = cat_names_to_idx(cat_names, features) or None

    all_fold_rows, all_summary_rows = [], []
    t_loop = datetime.now()

    for i, gamma in enumerate(gamma_grid):
        t_g = datetime.now()
        print(f"\n  [{_ts()}] re_lgbm M{model_number} gamma={gamma:.2f}  "
              f"({i+1}/{len(gamma_grid)})  {_eta_str(t_loop, i, len(gamma_grid))}"
              f"  (pi={re_lgb.PI_BY_MODEL[model_number]})")
        _, fold_metrics = re_lgb.run_oof(
            df_nativi, gamma, model_number, features, cat_idx)
        fold_rows = list(fold_metrics.values())
        all_fold_rows.extend(fold_rows)
        all_summary_rows.append(_summary_from_folds(fold_rows))
        elapsed_g = (datetime.now() - t_g).total_seconds()
        row = all_summary_rows[-1]
        print(f"  OK gamma={gamma:.2f}  lift@1%={row.get('lift@1%_mean', float('nan')):.2f}x"
              f"  PR={row.get('pr_auc_mean', float('nan')):.3f}  ({elapsed_g:.0f}s)")

    return all_fold_rows, all_summary_rows


def run_puet(df_preprocessed: pd.DataFrame, _model_number: int,
             gamma_grid: list) -> tuple[list, list]:
    """PUET con γ-weighted N certi. Usa preprocessed (no NA)."""
    features = get_features(df_preprocessed)

    all_fold_rows, all_summary_rows = [], []
    t_loop = datetime.now()

    for i, gamma in enumerate(gamma_grid):
        t_g = datetime.now()
        print(f"\n  [{_ts()}] puet M{_model_number} gamma={gamma:.2f}  "
              f"({i+1}/{len(gamma_grid)})  {_eta_str(t_loop, i, len(gamma_grid))}")
        _, fold_metrics = puet_mod.run_oof(df_preprocessed, gamma, features)
        fold_rows = list(fold_metrics.values())
        all_fold_rows.extend(fold_rows)
        all_summary_rows.append(_summary_from_folds(fold_rows))
        elapsed_g = (datetime.now() - t_g).total_seconds()
        row = all_summary_rows[-1]
        print(f"  OK gamma={gamma:.2f}  lift@1%={row.get('lift@1%_mean', float('nan')):.2f}x"
              f"  PR={row.get('pr_auc_mean', float('nan')):.3f}  ({elapsed_g:.0f}s)")

    return all_fold_rows, all_summary_rows



MODEL_RUNNERS = {
    "lgb_supervised": (run_lgb_supervised, "nativi",       GAMMA_SUPERVISED),
    "bagging_lgbm":   (run_bagging_lgbm,   "nativi",       GAMMA_DEFAULT),
    "re_lgbm":        (run_re_lgbm,        "nativi",       GAMMA_DEFAULT),
    "puet":           (run_puet,           "preprocessed", GAMMA_DEFAULT),
}


def main(models: list, datasets: list, gamma_override: list = None) -> None:
    print(f"  PNU Mixing — Griglia gamma")
    print(f"  Modelli: {models}")
    print(f"  Dataset: M{datasets}")
    print(f"  Reshuffle seed: {RESHUFFLE_SEED}")

    print(f"\n  [{_ts()}] Calcolo fold_map gerarchico M3->M2->M1 (seed={RESHUFFLE_SEED})...")
    fold_map = compute_fold_map_hierarchical(seed=RESHUFFLE_SEED)

    for model_number in datasets:
        print(f"\n  Dataset M{model_number}")

        print(f"\n  [{_ts()}] Caricamento nativi M{model_number}...")
        df_nativi, cat_names = load_nativi_raw(model_number)
        df_nativi = apply_fold_map(df_nativi, fold_map)

        df_preprocessed = None
        if any(MODEL_RUNNERS[m][1] == "preprocessed" for m in models):
            print(f"  [{_ts()}] Caricamento preprocessed M{model_number}...")
            df_preprocessed = apply_fold_map(
                load_preprocessed_raw(model_number), fold_map)

        for model_name in models:
            runner_fn, data_type, default_grid = MODEL_RUNNERS[model_name]
            gamma_grid = gamma_override if gamma_override else default_grid

            print(f"\n  [{_ts()}] {model_name.upper()} — M{model_number}")
            print(f"  gamma grid ({len(gamma_grid)} valori): {gamma_grid}")

            df = df_nativi if data_type == "nativi" else df_preprocessed

            t0 = datetime.now()
            extra = {"cat_names": cat_names} if data_type == "nativi" else {}
            fold_rows, summary_rows = runner_fn(df, model_number, gamma_grid, **extra)
            elapsed = (datetime.now() - t0).total_seconds()

            _save_results(model_name, model_number, fold_rows, summary_rows)

            print(f"\n  [{_ts()}] {model_name} M{model_number} — riepilogo "
                  f"({elapsed/60:.1f} min)")
            print(f"  {'gamma':>6}  {'lift@1% mean':>12}  {'SD':>6}  {'PR AUC mean':>11}")
            for row in summary_rows:
                print(f"  {row['gamma']:>6.2f}  "
                      f"{row.get('lift@1%_mean', float('nan')):>10.2f}  "
                      f"{row.get('lift@1%_sd',   float('nan')):>7.2f}  "
                      f"{row.get('pr_auc_mean',  float('nan')):>9.3f}")

    print(f"\n  Completato. Output in: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PNU Mixing grid search")
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Modelli da eseguire (default: tutti)")
    parser.add_argument(
        "--datasets", nargs="+", type=int, default=DATASETS,
        choices=[1, 2, 3],
        help="Dataset da usare (1=M1, 2=M2, 3=M3; default: tutti)")
    parser.add_argument(
        "--gamma", nargs="+", type=float, default=None,
        help="Override griglia γ (default: grid specifica per modello)")
    args = parser.parse_args()

    main(models=args.models, datasets=args.datasets, gamma_override=args.gamma)
