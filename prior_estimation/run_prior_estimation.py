# Dipendenze non-standard:
#   pip install git+https://github.com/unmtransinfo/PULSNAR.git
#
# KMPE.py: vendored from http://web.eecs.umich.edu/~cscott/code.html#kmpe
#   via github.com/dimonenka/DEDPUL (MIT License)

MODEL_NUMBER         = 3        # 1=M1, 2=M2, 3=M3
TEST_MODE            = False    # True = smoke test rapido
N_RUNS_PULSNAR       = 30
MAX_UNL_EN_BLANCHARD = 200_000
MAX_UNL_KM2          = 1_500
N_RUNS_KM2           = 10
MAX_UNL_PULSNAR      = 50_000

import pathlib as _pathlib
_ROOT                  = _pathlib.Path(__file__).resolve().parents[1]
BASE_PATH_NATIVE       = _ROOT / "anac" / "output" / "parquet" / "model" / "nativi"
BASE_PATH_PREPROCESSED = _ROOT / "anac" / "output" / "parquet" / "model" / "preprocessed"
RESULTS_DIR            = _pathlib.Path(__file__).resolve().parent / "results"

COLS_TO_DROP = ['cig', 'esito', 'anno_pubblicazione', 'regione', 'fold']
LABEL_COL    = 'label'
RANDOM_STATE = 42


import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from KMPE import km
    KM2_AVAILABLE = True
    print("[OK]   KM2 disponibile.")
except ImportError:
    KM2_AVAILABLE = False
    print("[SKIP] KM2 non disponibile - KMPE.py non trovato.")

try:
    import PULSNAR
    PULSNAR_AVAILABLE = True
    print("[OK]   PULSNAR disponibile.")
except ImportError:
    PULSNAR_AVAILABLE = False
    print("[SKIP] PULSNAR non disponibile - "
          "pip install git+https://github.com/unmtransinfo/PULSNAR.git")

print()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 60}m {s % 60}s"


def load_dataset(model_number: int, preprocessed: bool = False) -> pd.DataFrame:
    base  = BASE_PATH_PREPROCESSED if preprocessed else BASE_PATH_NATIVE
    path  = base / f"M{model_number}.parquet"
    label = "preprocessed" if preprocessed else "nativo"
    print(f"  [{_ts()}] Caricamento dataset {label}: {path.name}")
    df = pd.read_parquet(path)
    print(f"             {df.shape[0]:,} righe x {df.shape[1]} colonne")
    return df


def prepare_pu_labels(df: pd.DataFrame):
    label  = df[LABEL_COL]
    is_pos = label.eq(1).fillna(False).values
    is_neg = label.eq(0).fillna(False).values
    is_unl = label.isna().values
    y_pu   = is_pos.astype(int)
    print(f"  Positivi certi   : {is_pos.sum():>8,}")
    print(f"  Negativi certi   : {is_neg.sum():>8,}")
    print(f"  Unlabeled        : {is_unl.sum():>8,}")
    print(f"  Totale righe     : {len(df):>8,}")
    return y_pu, is_pos, is_neg, is_unl


def drop_excluded_cols(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=present + [LABEL_COL], errors='ignore')


def subsample_df(df, is_pos, is_neg, is_unl, max_unl, rng):
    unl_idx  = np.where(is_unl)[0]
    pos_idx  = np.where(is_pos)[0]
    neg_idx  = np.where(is_neg)[0]
    n_sample = min(max_unl, len(unl_idx))
    sampled  = rng.choice(unl_idx, size=n_sample, replace=False)
    keep     = np.sort(np.concatenate([pos_idx, neg_idx, sampled]))
    print(f"  Subsample: {n_sample:,} unlabeled + {len(pos_idx)} pos "
          f"+ {len(neg_idx)} neg = {len(keep):,} righe")
    return df.iloc[keep].reset_index(drop=True), is_pos[keep], is_neg[keep], is_unl[keep]


def load_preprocessed_full(model_number: int):
    """
    Carica l'intero dataset preprocessed come numpy float, senza subsampling.
    Condiviso tra KM2 e PULSNAR per evitare letture parquet multiple.
    """
    path   = BASE_PATH_PREPROCESSED / f"M{model_number}.parquet"
    print(f"  [{_ts()}] Caricamento preprocessed: {path.name}")
    df     = pd.read_parquet(path)
    is_pos = df[LABEL_COL].eq(1).fillna(False).values
    y_pu   = is_pos.astype(int)
    X      = (drop_excluded_cols(df)
              .select_dtypes(include='number')
              .to_numpy(dtype=float))
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_med = np.nanmedian(X, axis=0)
        X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
        print(f"  [WARN] {nan_mask.sum():,} NA imputati con mediana colonna.")
    print(f"  {X.shape[0]:,} righe x {X.shape[1]} feature  "
          f"| P={y_pu.sum():,}  non-P={(y_pu == 0).sum():,}")
    return X, y_pu


def fit_lgbm_pu_scorer(X: pd.DataFrame, y_pu: np.ndarray,
                       random_state: int = RANDOM_STATE) -> np.ndarray:
    X = X.copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, verbose=-1, n_jobs=-1,
    )
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_predict(clf, X, y_pu, cv=cv, method='predict_proba')[:, 1]
    return scores


def elkan_noto_estimate(scores: np.ndarray, is_pos: np.ndarray) -> float:
    """alpha_hat = E[g(x) | x in P]  (Elkan & Noto 2008, sotto SCAR)."""
    return float(scores[is_pos].mean())


def blanchard_quantile_estimate(scores: np.ndarray, y_pu: np.ndarray) -> float:
    """pi_hat = inf_t [ F_U(S>=t) / F_P(S>=t) ]  (Blanchard et al. 2010)."""
    pos_scores = scores[y_pu == 1]
    unl_scores = scores[y_pu == 0]
    thresholds = np.percentile(scores, np.linspace(5, 95, 200))
    ratios = []
    for t in thresholds:
        f_p = np.mean(pos_scores >= t)
        if f_p > 1e-6:
            ratios.append(np.mean(unl_scores >= t) / f_p)
    return float(np.min(ratios)) if ratios else np.nan


def bootstrap_scores_ci(estimator_fn, scores, mask, n_boot, random_state):
    rng   = np.random.default_rng(random_state)
    n     = len(scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(estimator_fn(scores[idx], mask[idx]))
        except Exception:
            pass
    if len(boots) < 2:
        return np.nan, np.nan
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def _km2_single(X_full: np.ndarray, y_pu_full: np.ndarray,
                max_unl: int, random_state: int) -> float:
    rng      = np.random.default_rng(random_state)
    pos_idx  = np.where(y_pu_full == 1)[0]
    unl_idx  = np.where(y_pu_full == 0)[0]
    n_sample = min(max_unl, len(unl_idx))
    sampled  = rng.choice(unl_idx, size=n_sample, replace=False)
    X_pos    = X_full[pos_idx]
    X_unl    = X_full[sampled]
    print(f"    X_pos: {X_pos.shape} | X_unl: {X_unl.shape}")
    try:
        return float(km(X_unl, X_pos))   # mixture=unlabeled, component=positives
    except Exception as e:
        print(f"    [WARN] KM2 fallito: {e}")
        return np.nan


def _pulsnar_single(X_full: np.ndarray, y_pu_full: np.ndarray,
                    max_unl: int, random_state: int) -> float:
    rng      = np.random.default_rng(random_state)
    pos_idx  = np.where(y_pu_full == 1)[0]
    unl_idx  = np.where(y_pu_full == 0)[0]
    n_sample = min(max_unl, len(unl_idx))
    sampled  = rng.choice(unl_idx, size=n_sample, replace=False)
    keep     = np.sort(np.concatenate([pos_idx, sampled]))
    # StandardScaler necessario: GMM interno e' sensibile alle scale delle feature
    X_sub    = StandardScaler().fit_transform(X_full[keep])
    y_sub    = y_pu_full[keep]
    rec_ids  = np.arange(len(y_sub))
    print(f"    X_sub: {X_sub.shape} | pos={y_sub.sum()} | non-P={(y_sub == 0).sum()}")
    try:
        pls = PULSNAR.PULSNARClassifier(
            scar=False, csrdata=False, classifier='xgboost',
            bin_method='scott', bw_method='hist',
            lowerbw=0.01, upperbw=0.5, optim='local',
            calibration=True, calibration_data='PU',
            calibration_method='isotonic', calibration_n_bins=100,
            smooth_isotonic=False, classification_metrics=False,
            n_iterations=1, kfold=5, kflips=1,
        )
        res = pls.pulsnar(X_sub, y_sub, rec_list=rec_ids)
        return float(res['estimated_alpha'])
    except Exception as e:
        print(f"    [WARN] PULSNAR fallito: {e}")
        return np.nan


def multisample_ci(estimate_fn, n_runs: int, base_seed: int = RANDOM_STATE) -> tuple:
    """
    Esegue estimate_fn n_runs volte con seed base_seed + i*1000.
    Punto stima = mediana di tutti i run; CI = [2.5%, 97.5%].
    Con n_runs=1 restituisce (valore, nan, nan).
    """
    vals = []
    for i in range(n_runs):
        seed = base_seed + i * 1_000
        t0   = time.time()
        print(f"  [{_ts()}] Run {i+1}/{n_runs} (seed={seed})...")
        v = estimate_fn(random_state=seed)
        print(f"             -> alpha = {v:.4f}  ({_elapsed(t0)})" if not np.isnan(v)
              else f"             -> fallito ({_elapsed(t0)})")
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), np.nan, np.nan
    med = float(np.median(vals))
    return med, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sensitivity_analysis(scores: np.ndarray, y_pu: np.ndarray,
                          alpha_hat: float) -> pd.DataFrame:
    results = []
    for m in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        alpha_t    = np.clip(alpha_hat * m, 1e-6, 0.9999)
        scores_adj = np.clip(scores * alpha_t / max(alpha_hat, 1e-6), 0, 1)
        est        = blanchard_quantile_estimate(scores_adj, y_pu)
        results.append({'mult': m, 'alpha_ipotetico': round(alpha_t, 4),
                        'stima_quantile': round(est, 4) if not np.isnan(est) else np.nan})
    return pd.DataFrame(results)


def print_results_table(results: dict):
    W = 70
    print("  RIEPILOGO STIME DELLA PRIOR DI CLASSE (alpha)")
    print(f"  Modello M{MODEL_NUMBER} | EN/Blanchard -> nativo | KM2/PULSNAR -> preprocessed")
    print("=" * W)
    print(f"  {'Metodo':<32} {'alpha':>10} {'CI 2.5%':>10} {'CI 97.5%':>10}")
    print("-" * W)
    alphas_valid = []
    for method, vals in results.items():
        alpha   = vals.get('alpha', np.nan)
        lower   = vals.get('ci_lower', np.nan)
        upper   = vals.get('ci_upper', np.nan)
        skipped = vals.get('skipped', False)
        note    = " [SKIP]" if skipped else ""
        a_s = f"{alpha:.4f}" if not np.isnan(alpha) else ("-" if skipped else "N/A")
        l_s = f"{lower:.4f}" if not np.isnan(lower) else "-"
        u_s = f"{upper:.4f}" if not np.isnan(upper) else "-"
        print(f"  {method:<32} {a_s:>10} {l_s:>10} {u_s:>10}{note}")
        if not np.isnan(alpha) and not skipped:
            alphas_valid.append(alpha)
    print("-" * W)
    if alphas_valid:
        media = float(np.mean(alphas_valid))
        print(f"\n  Media metodi disponibili : {media:.4f}")
        if len(alphas_valid) > 1:
            stdev = float(np.std(alphas_valid))
            print(f"  Dev. standard            : {stdev:.4f}")
            if stdev > 0.05:
                print("  [ATTENZIONE] Alta variabilita' tra metodi (sigma > 0.05).")
                print("  -> Usare il valore piu' alto (approccio conservativo).")
            else:
                print("  [OK] Le stime convergono.")
        alpha_op = max(alphas_valid)
        print(f"\n  VALORE OPERATIVO CONSIGLIATO : alpha = {alpha_op:.4f}")
        print("  (massimo tra i metodi validi - conservativo per ridurre FN)")
    print("=" * W + "\n")


def save_prior_estimates(results: dict, run_dt: str, elapsed: str,
                         n_pos: int, n_neg: int, n_unl: int, cfg: dict):
    _notes = {
        'Elkan-Noto (LightGBM)': 'stima c  - bias SAR -> tende a sovrastimare',
        'Blanchard Quantili':    'stima c  - lower bound teorico (piu\' robusto a SAR)',
        'KM2 (DEDPUL)':         'stima pi - verifica CI: se degenere, non affidabile',
        'PULSNAR':              'stima pi - instabile con <500 positivi',
    }
    out_path = RESULTS_DIR / f"prior_estimates_M{MODEL_NUMBER}.txt"
    W = 72
    lines = []
    lines.append("=" * W)
    lines.append("  STIMA DELLA PRIOR DI CLASSE - PU Learning su Appalti Pubblici")
    lines.append(f"  Data: {run_dt}  |  Modello: M{MODEL_NUMBER}  |  TEST_MODE: {TEST_MODE}")
    lines.append("=" * W)
    lines.append(f"  Positivi certi : {n_pos:>10,}")
    lines.append(f"  Negativi certi : {n_neg:>10,}")
    lines.append(f"  Unlabeled      : {n_unl:>10,}")
    lines.append(f"  Totale         : {n_pos + n_neg + n_unl:>10,}")
    lines.append(f"  Runtime        :  {elapsed}")
    lines.append(f"  Config EN/Blanchard : max_unl={cfg['lgbm_max_unl']:,}  bootstrap={cfg['n_bootstrap']}")
    lines.append(f"  Config KM2          : max_unl={cfg['km2_max_unl']:,}  runs={cfg['n_km2_runs']}")
    lines.append(f"  Config PULSNAR      : max_unl={cfg['pulsnar_max_unl']:,}  runs={cfg['n_pulsnar_runs']}")
    lines.append("-" * W)
    lines.append(f"  {'Metodo':<30} {'alpha':>7} {'CI 2.5%':>9} {'CI 97.5%':>9}")
    lines.append("-" * W)
    alphas_valid = []
    for method, vals in results.items():
        alpha   = vals.get('alpha', np.nan)
        lower   = vals.get('ci_lower', np.nan)
        upper   = vals.get('ci_upper', np.nan)
        skipped = vals.get('skipped', False)
        a_s = f"{alpha:.4f}" if not np.isnan(alpha) else ("-" if skipped else "N/A")
        l_s = f"{lower:.4f}" if not np.isnan(lower) else "-"
        u_s = f"{upper:.4f}" if not np.isnan(upper) else "-"
        flag = ""
        if not skipped and not np.isnan(alpha) and not np.isnan(lower) and not np.isnan(upper):
            if lower == upper:
                flag = "  [!] CI degenere"
            elif alpha > upper + 1e-6:
                flag = "  [!] alpha > CI upper"
        if skipped:
            flag = "  [SKIP]"
        lines.append(f"  {method:<30} {a_s:>7} {l_s:>9} {u_s:>9}{flag}")
        if not np.isnan(alpha) and not skipped:
            alphas_valid.append(alpha)
    lines.append("-" * W)
    if alphas_valid:
        media = float(np.mean(alphas_valid))
        stdev = float(np.std(alphas_valid))
        lines.append(f"  Media  : {media:.4f}   Dev.std : {stdev:.4f}")
        alpha_op = max(alphas_valid)
        lines.append(f"  VALORE OPERATIVO: alpha = {alpha_op:.4f}  (max tra validi - conservativo)")
    lines.append("-" * W)
    lines.append("  Cosa stima ogni metodo:")
    for method, note in _notes.items():
        lines.append(f"    {method:<30}  {note}")
    lines.append("=" * W)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  [SAVED] {out_path}")


def main():
    t_total = time.time()
    rng     = np.random.default_rng(RANDOM_STATE)

    lgbm_max_unl    = 5_000 if TEST_MODE else MAX_UNL_EN_BLANCHARD
    km2_max_unl     = 500   if TEST_MODE else MAX_UNL_KM2
    pulsnar_max_unl = 5_000 if TEST_MODE else MAX_UNL_PULSNAR
    n_km2_runs      = 1     if TEST_MODE else N_RUNS_KM2
    n_pulsnar_runs  = 1     if TEST_MODE else N_RUNS_PULSNAR
    n_bootstrap     = 1     if TEST_MODE else 200

    W = 70
    print("  PIPELINE STIMA PRIOR - PU Learning su Appalti Pubblici")
    print(f"  [{_ts()}] Modello: M{MODEL_NUMBER}  |  TEST_MODE: {TEST_MODE}")
    print(f"  EN/Blanchard : max_unl={lgbm_max_unl:,}  bootstrap={n_bootstrap}")
    print(f"  KM2          : max_unl={km2_max_unl:,}  runs={n_km2_runs}")
    print(f"  PULSNAR      : max_unl={pulsnar_max_unl:,}  runs={n_pulsnar_runs}")

    results = {}

    print(f"[{_ts()}] -- STEP 1/4: caricamento dataset nativo --")
    t0     = time.time()
    df_nat = load_dataset(MODEL_NUMBER, preprocessed=False)
    y_pu_full, is_pos_full, is_neg_full, is_unl_full = prepare_pu_labels(df_nat)

    # Subsample righe prima di copiare colonne (evita OOM su dataset da milioni di righe)
    if lgbm_max_unl is not None:
        df_nat, is_pos_s, is_neg_s, is_unl_s = subsample_df(
            df_nat, is_pos_full, is_neg_full, is_unl_full, lgbm_max_unl, rng)
        y_pu_s = is_pos_s.astype(int)
    else:
        is_pos_s, is_neg_s, is_unl_s, y_pu_s = (
            is_pos_full, is_neg_full, is_unl_full, y_pu_full)

    X = drop_excluded_cols(df_nat)
    print(f"  Feature usate: {X.shape[1]} colonne  |  {_elapsed(t0)}")

    print(f"\n[{_ts()}] -- STEP 2/4: addestramento LightGBM (5-fold CV) --")
    t0     = time.time()
    scores = fit_lgbm_pu_scorer(X, y_pu_s)
    print(f"  Score calcolati.  ({_elapsed(t0)})")
    print(f"  Score pos - media: {scores[is_pos_s].mean():.4f}  std: {scores[is_pos_s].std():.4f}")
    print(f"  Score unl - media: {scores[is_unl_s].mean():.4f}  std: {scores[is_unl_s].std():.4f}")

    print(f"\n[{_ts()}] -- Elkan-Noto --")
    t0       = time.time()
    en_alpha = elkan_noto_estimate(scores, is_pos_s)
    print(f"  alpha_hat = {en_alpha:.4f}")
    if n_bootstrap > 1:
        print(f"  Bootstrap CI ({n_bootstrap} iter su scores)...")
        en_lo, en_hi = bootstrap_scores_ci(
            elkan_noto_estimate, scores, is_pos_s, n_bootstrap, RANDOM_STATE)
        print(f"  CI 95%: [{en_lo:.4f}, {en_hi:.4f}]  ({_elapsed(t0)})")
    else:
        en_lo, en_hi = np.nan, np.nan
        print("  CI: saltato (TEST_MODE).")
    results['Elkan-Noto (LightGBM)'] = {
        'alpha': en_alpha, 'ci_lower': en_lo, 'ci_upper': en_hi}

    print(f"\n[{_ts()}] -- Blanchard Quantili --")
    t0       = time.time()
    bl_alpha = blanchard_quantile_estimate(scores, y_pu_s)
    print(f"  alpha_hat = {bl_alpha:.4f}")
    if n_bootstrap > 1:
        print(f"  Bootstrap CI ({n_bootstrap} iter su scores)...")
        bl_lo, bl_hi = bootstrap_scores_ci(
            blanchard_quantile_estimate, scores, y_pu_s, n_bootstrap, RANDOM_STATE)
        print(f"  CI 95%: [{bl_lo:.4f}, {bl_hi:.4f}]  ({_elapsed(t0)})")
    else:
        bl_lo, bl_hi = np.nan, np.nan
        print("  CI: saltato (TEST_MODE).")
    results['Blanchard Quantili'] = {
        'alpha': bl_alpha, 'ci_lower': bl_lo, 'ci_upper': bl_hi}

    print(f"\n[{_ts()}] -- STEP 3/4: KM2 --")
    if KM2_AVAILABLE:
        t0 = time.time()
        print(f"  Caricamento dataset preprocessed...")
        X_prep, y_pu_prep = load_preprocessed_full(MODEL_NUMBER)

        def _km2_run(random_state):
            return _km2_single(X_prep, y_pu_prep, km2_max_unl, random_state)

        km2_med, km2_lo, km2_hi = multisample_ci(_km2_run, n_km2_runs, RANDOM_STATE)
        lo_s = f"{km2_lo:.4f}" if not np.isnan(km2_lo) else "-"
        hi_s = f"{km2_hi:.4f}" if not np.isnan(km2_hi) else "-"
        if not np.isnan(km2_med):
            print(f"  mediana = {km2_med:.4f}  CI 95%: [{lo_s}, {hi_s}]  ({_elapsed(t0)})")
        else:
            print(f"  Stima non riuscita.  ({_elapsed(t0)})")
        results['KM2 (DEDPUL)'] = {
            'alpha': km2_med, 'ci_lower': km2_lo, 'ci_upper': km2_hi}
    else:
        X_prep = y_pu_prep = None
        results['KM2 (DEDPUL)'] = {
            'alpha': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'skipped': True}

    print(f"\n[{_ts()}] -- STEP 4/4: PULSNAR --")
    if PULSNAR_AVAILABLE:
        t0 = time.time()
        # Riutilizza i dati preprocessed se già caricati da KM2
        if X_prep is None:
            print(f"  Caricamento dataset preprocessed...")
            X_prep, y_pu_prep = load_preprocessed_full(MODEL_NUMBER)

        def _pulsnar_run(random_state):
            return _pulsnar_single(X_prep, y_pu_prep, pulsnar_max_unl, random_state)

        pn_med, pn_lo, pn_hi = multisample_ci(_pulsnar_run, n_pulsnar_runs, RANDOM_STATE)
        lo_s = f"{pn_lo:.4f}" if not np.isnan(pn_lo) else "-"
        hi_s = f"{pn_hi:.4f}" if not np.isnan(pn_hi) else "-"
        if not np.isnan(pn_med):
            print(f"  mediana = {pn_med:.4f}  CI 95%: [{lo_s}, {hi_s}]  ({_elapsed(t0)})")
        else:
            print(f"  Stima non riuscita.  ({_elapsed(t0)})")
        results['PULSNAR'] = {
            'alpha': pn_med, 'ci_lower': pn_lo, 'ci_upper': pn_hi}
        # PULSNAR scrive i suoi file diagnostici nella cwd: consolidali in results/
        import shutil
        RESULTS_DIR.mkdir(exist_ok=True)
        for _f in ("alpha_estimates.tsv", "predictions.tsv",
                   "bic_vs_cluster_count.png", "model_imp_features.pkl"):
            if _pathlib.Path(_f).is_file():
                shutil.move(_f, RESULTS_DIR / _f)
    else:
        results['PULSNAR'] = {
            'alpha': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'skipped': True}

    print_results_table(results)
    save_prior_estimates(
        results,
        run_dt=datetime.now().strftime("%Y-%m-%d %H:%M"),
        elapsed=_elapsed(t_total),
        n_pos=int(is_pos_full.sum()),
        n_neg=int(is_neg_full.sum()),
        n_unl=int(is_unl_full.sum()),
        cfg=dict(
            lgbm_max_unl=lgbm_max_unl, n_bootstrap=n_bootstrap,
            km2_max_unl=km2_max_unl, n_km2_runs=n_km2_runs,
            pulsnar_max_unl=pulsnar_max_unl, n_pulsnar_runs=n_pulsnar_runs,
        ),
    )

    if not TEST_MODE and not np.isnan(en_alpha):
        print(f"[{_ts()}] -- SENSITIVITY ANALYSIS --")
        sens_df = sensitivity_analysis(scores, y_pu_s, en_alpha)
        print(sens_df.to_string(index=False))
        r = sens_df['stima_quantile'].max() - sens_df['stima_quantile'].min()
        print(f"\n  Range quantile al variare di alpha: {r:.4f}")
        print("  [OK]" if r < 0.05
              else "  [ATTENZIONE] Modello sensibile alla prior (range >= 0.05).")

    print(f"\n[{_ts()}] Pipeline completata in {_elapsed(t_total)}.\n")
    return results


if __name__ == "__main__":
    results = main()
