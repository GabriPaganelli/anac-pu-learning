"""
Utilities condivise per l'analisi ex-post dei 4 modelli finalisti.

Responsabilità:
  - Costanti (seed, path, data type per modello)
  - Setup struttura directory output
  - Save/load modelli (LGB .txt, ensemble .pkl)
  - Caricamento dati + applicazione fold map
  - Split fold
  - Wrapper BaggingLGBWrapper e PUETWrapper (ensemble finale + SHAP)
  - Dispatch OOF: prepare / oof_fit_score / final_fit per ogni modello
  - Lettura gamma_star.json
"""

import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.utils import check_random_state
from sklearn.ensemble import ExtraTreesClassifier
from scipy.special import expit as sigmoid

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from selected_models.utils import (
    compute_fold_map_hierarchical, apply_fold_map,
    load_nativi_raw, load_preprocessed_raw,
    get_features, cat_names_to_idx,
    LABEL_COL, FOLD_COL, COLS_TO_DROP, CIG_COL,
)
import selected_models.lgbm_supervised as lgb_sup
import selected_models.bagging_lgbm   as bag_lgb
import selected_models.re_lgbm        as re_lgb
import selected_models.puet           as puet_mod

CALIB_SEED = 2026   # fold map per Platt calibration
CONF_SEED  = 2027   # fold map per conformal inference

DATA_TYPES = {
    "lgbm_supervised": "nativi",
    "bagging_lgbm":   "nativi",
    "re_lgbm":        "nativi",
    "puet":           "preprocessed",
}

ALL_MODELS   = ["lgbm_supervised", "bagging_lgbm", "re_lgbm", "puet"]
ALL_DATASETS = [1, 2, 3]

OUT_ROOT = _HERE / "results"


def setup_out_dir(model_name: str, model_number: int) -> Path:
    d = OUT_ROOT / f"{model_name}_M{model_number}"
    for sub in (
        "models", "scores",
        "calibration", "calibration/plots",
        "conformal", "conformal/plots",
        "shap", "shap/plots",
    ):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d



def save_model(model, out_dir: Path, name: str, model_name: str) -> None:
    if model_name in ("lgbm_supervised", "re_lgbm"):
        model.save_model(str(out_dir / "models" / f"{name}.txt"))
    else:
        joblib.dump(model, out_dir / "models" / f"{name}.pkl", compress=3)


def load_model(out_dir: Path, name: str, model_name: str):
    if model_name in ("lgbm_supervised", "re_lgbm"):
        return lgb.Booster(model_file=str(out_dir / "models" / f"{name}.txt"))
    return joblib.load(out_dir / "models" / f"{name}.pkl")


def model_exists(out_dir: Path, name: str, model_name: str) -> bool:
    if model_name in ("lgbm_supervised", "re_lgbm"):
        return (out_dir / "models" / f"{name}.txt").exists()
    return (out_dir / "models" / f"{name}.pkl").exists()



def load_gamma_star(path: Path) -> dict:
    """Legge gamma_star.json → {model_name: {dataset_int: gamma_float}}.

    Le chiavi che iniziano con '_' (es. '_note') vengono ignorate.
    """
    with open(path) as f:
        raw = json.load(f)
    return {
        m: {int(k): float(v) for k, v in ms.items()}
        for m, ms in raw.items()
        if not m.startswith("_") and isinstance(ms, dict)
    }



def load_data(model_name: str, model_number: int,
              fold_map: pd.Series) -> tuple[pd.DataFrame, list, list]:
    """Carica df corretto (nativi/preprocessed), applica fold_map.

    Ritorna (df, features, cat_idx).
    """
    if DATA_TYPES[model_name] == "nativi":
        df, cat_names = load_nativi_raw(model_number)
        df = apply_fold_map(df, fold_map)
        features = get_features(df)
        cat_idx  = cat_names_to_idx(cat_names, features) or None
    else:
        df = load_preprocessed_raw(model_number)
        df = apply_fold_map(df, fold_map)
        exclude  = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
        features = [c for c in df.columns
                    if c not in exclude and df[c].dtype.kind in "iufb"]
        cat_idx  = None
    return df, features, cat_idx



def split_fold(df: pd.DataFrame, features: list, fold_test: int) -> dict:
    """Restituisce dict con X_P_tr, X_N_tr, X_U_tr, X_te, label_te, idx_te."""
    label = df[LABEL_COL].values.astype(float)
    fold  = df[FOLD_COL].values.astype(float)
    X     = df[features].to_numpy(dtype=np.float32)
    tr    = fold != fold_test
    te    = fold == fold_test
    return dict(
        X_P_tr   = X[(label == 1) & tr],
        X_N_tr   = X[(label == 0) & tr],
        X_U_tr   = X[np.isnan(label) & tr],
        X_te     = X[te],
        label_te = label[te],
        idx_te   = np.where(te)[0],
    )



def model_predict(model, X: np.ndarray, model_name: str) -> np.ndarray:
    """Restituisce probabilità in [0,1] per qualsiasi tipo di modello."""
    if model_name == "re_lgbm":
        return sigmoid(model.predict(X, raw_score=True))
    if isinstance(model, (BaggingLGBWrapper, PUETWrapper)):
        return model.predict(X)
    # lgb.Booster binario o lgbm_supervised
    return model.predict(X)



class BaggingLGBWrapper:
    """Ensemble di B LGB boosters — predict(), predict_proba(), shap_values_sample()."""

    def __init__(self, boosters: list):
        self.boosters = boosters

    @classmethod
    def from_scratch(cls, X_P, X_N, X_U, gamma, cat_idx, rng,
                     n_boosters: int = bag_lgb.B) -> "BaggingLGBWrapper":
        params = dict(bag_lgb.LGBM_FIXED, num_leaves=bag_lgb.NUM_LEAVES)
        boosters = []
        for b in range(n_boosters):
            s   = min(bag_lgb.S, len(X_U))
            idx = rng.choice(len(X_U), size=s, replace=(s == len(X_U)))
            X_neg = X_U[idx]
            if gamma > 0 and len(X_N) > 0:
                X_tr = np.vstack([X_P, X_N, X_neg])
                y_tr = np.concatenate([np.ones(len(X_P)),
                                       np.zeros(len(X_N)), np.zeros(s)])
                w_tr = np.concatenate([np.ones(len(X_P)),
                                       np.full(len(X_N), gamma), np.ones(s)])
            else:
                X_tr = np.vstack([X_P, X_neg])
                y_tr = np.concatenate([np.ones(len(X_P)), np.zeros(s)])
                w_tr = None
            p = dict(params); p["seed"] = int(rng.randint(0, 99999))
            m = lgb.train(
                p,
                bag_lgb._make_dataset(X_tr, y_tr, w_tr, cat_idx),
                num_boost_round=bag_lgb.N_ROUNDS,
                callbacks=[lgb.log_evaluation(period=0)],
            )
            boosters.append(m)
            if (b + 1) % 50 == 0:
                print(f"      [BaggingLGB] bag {b+1}/{n_boosters}")
        return cls(boosters)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([b.predict(X) for b in self.boosters], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1 - p, p])

    def shap_values_sample(self, X: np.ndarray,
                           n_sample: int = 50) -> np.ndarray:
        """SHAP values mediati su n_sample boosters (TreeSHAP esatto per ciascuno)."""
        import shap
        sample = self.boosters[:n_sample]
        all_sv = []
        for i, b in enumerate(sample):
            ex = shap.TreeExplainer(b)
            sv = ex.shap_values(X)
            all_sv.append(sv)
            if (i + 1) % 10 == 0:
                print(f"      [SHAP Bagging] booster {i+1}/{n_sample}")
        return np.mean(all_sv, axis=0)


class PUETWrapper:
    """Ensemble di B ExtraTreesClassifier — predict(), predict_proba(), shap_values_sample()."""

    def __init__(self, estimators: list):
        self.estimators = estimators

    @classmethod
    def from_scratch(cls, X_P, X_N, X_U, gamma, rng,
                     n_bags: int = puet_mod.B) -> "PUETWrapper":
        estimators = []
        s       = min(puet_mod.S, len(X_U))
        replace = s == len(X_U)
        for b in range(n_bags):
            idx   = rng.choice(len(X_U), size=s, replace=replace)
            X_neg = X_U[idx]
            if gamma > 0 and len(X_N) > 0:
                X_tr = np.vstack([X_P, X_N, X_neg])
                y_tr = np.concatenate([np.ones(len(X_P)),
                                       np.zeros(len(X_N)), np.zeros(s)])
                w_tr = np.concatenate([np.ones(len(X_P)),
                                       np.full(len(X_N), gamma), np.ones(s)])
            else:
                X_tr = np.vstack([X_P, X_neg])
                y_tr = np.concatenate([np.ones(len(X_P)), np.zeros(s)])
                w_tr = None
            et = ExtraTreesClassifier(
                n_estimators     = puet_mod.NTREE_ET,
                max_features     = puet_mod.MAX_FEATURES,
                min_samples_leaf = puet_mod.MIN_SAMPLES_LEAF,
                bootstrap        = False,
                n_jobs           = -1,
                random_state     = rng.randint(0, 99999),
            )
            et.fit(X_tr, y_tr, sample_weight=w_tr)
            estimators.append(et)
            if (b + 1) % 50 == 0:
                print(f"      [PUET] bag {b+1}/{n_bags}")
        return cls(estimators)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([et.predict_proba(X)[:, 1] for et in self.estimators], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1 - p, p])

    def shap_values_sample(self, X: np.ndarray,
                           n_sample: int = 20) -> np.ndarray:
        """SHAP values mediati su n_sample bag (TreeSHAP per ExtraTrees).

        ExtraTrees con MAX_FEATURES=1.0 e bootstrap=False: TreeExplainer
        funziona ma può essere lento per ensemble grandi.
        """
        import shap
        sample = self.estimators[:n_sample]
        all_sv = []
        for i, et in enumerate(sample):
            ex = shap.TreeExplainer(et)
            sv = ex.shap_values(X)
            # ExtraTrees restituisce lista [sv_cls0, sv_cls1]
            sv_pos = sv[1] if isinstance(sv, list) else sv
            all_sv.append(sv_pos)
            if (i + 1) % 5 == 0:
                print(f"      [SHAP PUET] bag {i+1}/{n_sample}")
        return np.mean(all_sv, axis=0)


#
# prepare(df, features, cat_idx, gamma, model_number) -> state dict
#   Chiamato UNA volta prima del loop OOF.
#   Esegue global_tune (lgbm_supervised) o inizializza RNG.
#
# oof_fit_score(split, gamma, model_number, features, cat_idx, state) -> scores_te
#   Chiamato PER FOLD. Restituisce scores su TUTTI i record del fold test (P+N+U).
#
# final_fit(df, features, cat_idx, gamma, model_number, state) -> model
#   Addestra il modello finale su TUTTI i dati. Ritorna oggetto salvabile.


def _lgb_sup_prepare(df, features, cat_idx, gamma, model_number):
    label = df[LABEL_COL].values.astype(float)
    X     = df[features].to_numpy(dtype=np.float32)
    print("  [lgbm_supervised] Global tune...")
    best_params = lgb_sup.global_tune(X[label == 1], X[label == 0], cat_idx=cat_idx)
    return {"best_params": best_params}


def _lgb_sup_oof_fit_score(split, gamma, model_number, features, cat_idx, state):
    return lgb_sup.fit_predict(
        split["X_P_tr"], split["X_N_tr"], split["X_te"],
        gamma=gamma, best_params=state["best_params"], cat_idx=cat_idx,
    )


def _lgb_sup_final_fit(df, features, cat_idx, gamma, model_number, state):
    label = df[LABEL_COL].values.astype(float)
    X     = df[features].to_numpy(dtype=np.float32)
    X_P   = X[label == 1]; X_N = X[label == 0]
    X_tr  = np.vstack([X_P, X_N])
    y_tr  = np.concatenate([np.ones(len(X_P)), np.zeros(len(X_N))])
    w_tr  = np.concatenate([np.ones(len(X_P)), np.full(len(X_N), gamma)])
    params = dict(lgb_sup.LGBM_FIXED, **state["best_params"])
    cat    = cat_idx or "auto"
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                         categorical_feature=cat, free_raw_data=False)
    # N_ROUNDS_MAX senza early stopping: con tutti i dati non c'è holdout ES.
    # Regolarizzazione (lambda_l2=0.5, feature/bagging_fraction) previene overfitting.
    return lgb.train(
        params, dtrain,
        num_boost_round=lgb_sup.N_ROUNDS_MAX,
        callbacks=[lgb.log_evaluation(period=0)],
    )



def _bag_lgb_prepare(df, features, cat_idx, gamma, model_number):
    return {"rng": check_random_state(bag_lgb.RANDOM_SEED)}


def _bag_lgb_oof_fit_score(split, gamma, model_number, features, cat_idx, state):
    return bag_lgb.fit_predict(
        split["X_P_tr"], split["X_N_tr"], split["X_U_tr"], split["X_te"],
        gamma=gamma, rng=state["rng"], cat_idx=cat_idx,
    )


def _bag_lgb_final_fit(df, features, cat_idx, gamma, model_number, state):
    label = df[LABEL_COL].values.astype(float)
    X     = df[features].to_numpy(dtype=np.float32)
    rng   = check_random_state(bag_lgb.RANDOM_SEED + 9999)
    return BaggingLGBWrapper.from_scratch(
        X[label == 1], X[label == 0], X[np.isnan(label)],
        gamma=gamma, cat_idx=cat_idx, rng=rng,
    )



def _re_lgb_prepare(df, features, cat_idx, gamma, model_number):
    return {"best_iters": []}


def _re_lgb_oof_fit_score(split, gamma, model_number, features, cat_idx, state):
    pi = re_lgb.PI_BY_MODEL[model_number]
    scores, best_iter = re_lgb.fit_predict(
        split["X_P_tr"], split["X_N_tr"], split["X_U_tr"], split["X_te"],
        gamma=gamma, pi=pi, features=features, cat_idx=cat_idx,
    )
    state["best_iters"].append(best_iter)
    return scores


def _re_lgb_final_fit(df, features, cat_idx, gamma, model_number, state):
    pi    = re_lgb.PI_BY_MODEL[model_number]
    label = df[LABEL_COL].values.astype(float)
    X     = df[features].to_numpy(dtype=np.float32)
    X_P   = X[label == 1]; X_N = X[label == 0]; X_U = X[np.isnan(label)]
    n_P   = len(X_P); n_U = len(X_U); n_N = len(X_N)

    X_all = np.vstack([X_P, X_U, X_N])
    y_all = np.concatenate([np.ones(n_P), np.zeros(n_U), np.full(n_N, 2.0)])

    fobj   = re_lgb.make_pnpu_objective(pi, gamma, n_P, n_U, n_U)
    params = dict(
        re_lgb.LGBM_FIXED,
        num_leaves       = re_lgb.NUM_LEAVES,
        min_data_in_leaf = re_lgb.MIN_DATA_IN_LEAF,
        learning_rate    = re_lgb.LEARNING_RATE,
        objective        = fobj,
    )
    # n_rounds dalla media dei best_iter OOF; fallback a N_ROUNDS_MAX
    n_rounds = (int(np.mean(state["best_iters"]))
                if state["best_iters"] else re_lgb.N_ROUNDS_MAX)
    cat    = cat_idx if cat_idx else "auto"
    dtrain = lgb.Dataset(X_all, label=y_all, feature_name=features,
                         categorical_feature=cat, free_raw_data=False)
    return lgb.train(
        params, dtrain,
        num_boost_round=n_rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )



def _puet_prepare(df, features, cat_idx, gamma, model_number):
    return {"rng": check_random_state(puet_mod.RANDOM_SEED)}


def _puet_oof_fit_score(split, gamma, model_number, features, cat_idx, state):
    return puet_mod.fit_predict(
        split["X_P_tr"], split["X_N_tr"], split["X_U_tr"], split["X_te"],
        gamma=gamma, rng=state["rng"],
    )


def _puet_final_fit(df, features, cat_idx, gamma, model_number, state):
    label = df[LABEL_COL].values.astype(float)
    X     = df[features].to_numpy(dtype=np.float32)
    rng   = check_random_state(puet_mod.RANDOM_SEED + 9999)
    return PUETWrapper.from_scratch(
        X[label == 1], X[label == 0], X[np.isnan(label)],
        gamma=gamma, rng=rng,
    )



PREPARE_FNS = {
    "lgbm_supervised": _lgb_sup_prepare,
    "bagging_lgbm":   _bag_lgb_prepare,
    "re_lgbm":        _re_lgb_prepare,
    "puet":           _puet_prepare,
}

OOF_FNS = {
    "lgbm_supervised": _lgb_sup_oof_fit_score,
    "bagging_lgbm":   _bag_lgb_oof_fit_score,
    "re_lgbm":        _re_lgb_oof_fit_score,
    "puet":           _puet_oof_fit_score,
}

FINAL_FIT_FNS = {
    "lgbm_supervised": _lgb_sup_final_fit,
    "bagging_lgbm":   _bag_lgb_final_fit,
    "re_lgbm":        _re_lgb_final_fit,
    "puet":           _puet_final_fit,
}
