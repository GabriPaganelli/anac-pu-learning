# evaluation — Pipeline ex-post per i 4 modelli finalisti PU
# *evaluation — Ex-post pipeline for the 4 finalist PU models*

---

## Italiano

Calibrazione Platt, conformal prediction (Mondrian) e analisi SHAP applicati ai
4 modelli finalisti (`lgb_supervised`, `bagging_lgbm`, `re_lgbm`, `puet`) sui 3
dataset temporali M1/M2/M3. Dipende da `selected_models/` e `metrics/` nella root.

---

## Struttura

```
evaluation/
|-- utils_expost.py              # Costanti, I/O, wrappers BaggingLGB/PUET, dispatch OOF
|-- calibration.py               # Platt scaling OOF 4-fold + fit globale
|-- conformal.py                 # Split conformal Mondrian (legge Platt da calibration)
|-- shap_analysis.py             # SHAP sul modello finale (legge modello da calibration)
|-- bootstrap_ci.py              # Bootstrap CI su re_lgbm (qualsiasi dataset)
|-- run_all.py                   # Entry point: calibration -> conformal -> SHAP
|-- run_all_gamma_comparison.py  # Sensitivity su griglia gamma per un modello x dataset
|-- gamma_star.json              # gamma* canonici per ogni modello x dataset
`-- plots_exploration/
    `-- make_qq_ranks.py         # QQ plot del ranking tra modelli su contratti U
```

---

## Dipendenze e ordine obbligatorio

```
calibration.py
    |
    +-- conformal.py       (legge platt_params.csv)
    |
    `-- shap_analysis.py   (legge final_model.{txt|pkl})

bootstrap_ci.py            (legge platt_params.csv, indipendente da conformal/shap)
```

`conformal.py` e `shap_analysis.py` presuppongono che `calibration.py` sia gia
stato eseguito per la stessa combinazione modello x dataset.
`bootstrap_ci.py` e indipendente dagli altri step ma usa l'output di `calibration.py`
se disponibile (Platt globale per calibrare gli score bootstrap).

---

## Utilizzo

### Pipeline completa (tutti i modelli x tutti i dataset)

```bash
python evaluation/run_all.py
```

### Opzioni principali di run_all.py

```bash
# Solo un modello o un dataset
python evaluation/run_all.py --models re_lgbm --datasets 3

# Solo alcuni step
python evaluation/run_all.py --steps calibration shap

# Livello di errore conformal diverso da 0.10
python evaluation/run_all.py --alpha 0.05

# Piu boosters per SHAP ensemble (default 50)
python evaluation/run_all.py --n-shap-sample 100

# Riaddestrare anche se il modello e gia salvato
python evaluation/run_all.py --force-retrain
```

### Sensitivity su griglia gamma

```bash
# re_lgbm M3 con gamma=0.33 e 1.0 (default)
python evaluation/run_all_gamma_comparison.py

# Modello e dataset arbitrari
python evaluation/run_all_gamma_comparison.py --model bagging_lgbm --dataset 2 --gammas 0.33 0.66 1.0

# Solo calibration e SHAP, con alpha=0.05
python evaluation/run_all_gamma_comparison.py --steps calibration shap --alpha 0.05
```

### Bootstrap CI (re_lgbm, qualsiasi dataset)

```bash
# Tutti i dataset con parametri default (B=200, n_score dal dizionario interno)
python evaluation/bootstrap_ci.py

# Solo M3 con n_boot ridotto
python evaluation/bootstrap_ci.py --datasets 3 --n-boot 50
```

---

## Output

Tutto viene salvato sotto `evaluation/results/{model}_{Mn}/`.
Per `run_all_gamma_comparison.py` le cartelle hanno il suffisso gamma:
`results/{model}_{Mn}_g{gamma}/`.

```
results/
`-- {model}_{Mn}/
    |-- models/
    |   `-- final_model.{txt|pkl}      # Modello addestrato su tutti i dati (.pkl via DVC: `dvc pull`)
    |-- scores/
    |   |-- scores_oof.csv             # cig, fold, label, score_raw, score_calib_pf, score_calib_gl
    |   `-- scores_final.csv           # cig, label, score_raw_final, score_calib_final
    |-- calibration/
    |   |-- platt_params.csv           # a, b, SE per fold + righe "mean" e "global"
    |   |-- ece_brier.csv              # ECE e Brier per fold e overall
    |   `-- plots/reliability.{png,pdf}
    |-- conformal/
    |   |-- thresholds.csv             # q1, q0, n1_cal, n0_cal per fold
    |   |-- coverage.csv               # copertura per classe per fold + riga "mean_oof"
    |   |-- prediction_sets.csv        # cig, label, set, ampiezza, pvalue_classe1
    |   `-- plots/{set_dist,pvalue_hist}.{png,pdf}
    |-- shap/
    |   |-- shap_values_final.npy      # (n_rows, n_features)
    |   |-- feature_names.json
    |   |-- feature_importance.csv     # mean |SHAP| per feature, ordinata
    |   `-- plots/{bar_top20,beeswarm_top20,waterfall_*,pdp_top3}.{png,pdf}
    `-- bootstrap/                     # solo re_lgbm
        |-- scores_ci.csv              # cig, score_mean/lower/upper, pc_mean/lower/upper
        |-- metrics_ci.csv             # metrica, mean, lower, upper, sd
        `-- u_score_cigs.npy           # CIG campionati (cache per riproducibilita)
```

---

## gamma_star.json

Contiene i `gamma*` canonici usati da `run_all.py` e `bootstrap_ci.py`.
Le chiavi numeriche `"1"`, `"2"`, `"3"` corrispondono a M1, M2, M3.

```json
{
  "lgb_supervised": {"1": 1.0, "2": 1.0, "3": 1.0},
  "bagging_lgbm":   {"1": 0.66, "2": 0.66, "3": 0.66},
  "re_lgbm":        {"1": 1.0, "2": 1.0, "3": 1.0},
  "puet":           {"1": 0.0, "2": 0.0, "3": 0.0}
}
```

Chiavi che iniziano con `_` (es. `_note`, `_rationale`) sono ignorate dal codice.
Per cambiare `gamma*` basta modificare i valori numerici: non e necessario
toccare nessun altro file.

---

## plots_exploration/

`make_qq_ranks.py` confronta il ranking degli score tra modelli (re_lgbm vs
lgb_supervised, re_lgbm vs bagging_lgbm) su contratti U fuori dal training.
Richiede che i modelli finali siano gia stati addestrati e salvati da `calibration.py`.
Output: `plots_exploration/qq_rank_comparison_M{n}.{png,pdf}` per M1, M2, M3.

```bash
python evaluation/plots_exploration/make_qq_ranks.py
```

---

## Prerequisiti

```bash
pip install pandas numpy scipy scikit-learn lightgbm joblib pyarrow matplotlib shap
```

`shap` viene importato in modo lazy in `utils_expost.py` (solo nelle funzioni che
lo usano effettivamente) per evitare il costo di startup quando si importa
`utils_expost` da script che non usano SHAP.

---

## English

Platt scaling, conformal prediction (Mondrian), and SHAP analysis applied to the
4 finalist models (`lgb_supervised`, `bagging_lgbm`, `re_lgbm`, `puet`) on the
3 temporal datasets M1/M2/M3. Depends on `selected_models/` and `metrics/` at the root.

---

## Structure

```
evaluation/
|-- utils_expost.py              # Constants, I/O, BaggingLGB/PUET wrappers, OOF dispatch
|-- calibration.py               # Platt scaling OOF 4-fold + global fit
|-- conformal.py                 # Split conformal Mondrian (reads Platt from calibration)
|-- shap_analysis.py             # SHAP on final model (reads model from calibration)
|-- bootstrap_ci.py              # Bootstrap CI for re_lgbm (any dataset)
|-- run_all.py                   # Entry point: calibration -> conformal -> SHAP
|-- run_all_gamma_comparison.py  # Sensitivity over gamma grid for one model x dataset
|-- gamma_star.json              # Canonical gamma* for each model x dataset
`-- plots_exploration/
    `-- make_qq_ranks.py         # QQ plot of cross-model ranking on U contracts
```

---

## Dependencies and required order

```
calibration.py
    |
    +-- conformal.py       (reads platt_params.csv)
    |
    `-- shap_analysis.py   (reads final_model.{txt|pkl})

bootstrap_ci.py            (reads platt_params.csv, independent of conformal/shap)
```

`conformal.py` and `shap_analysis.py` assume `calibration.py` has already been run
for the same model x dataset combination.
`bootstrap_ci.py` is independent of the other steps but uses `calibration.py` output
if available (global Platt to calibrate bootstrap scores).

---

## Usage

### Full pipeline (all models x all datasets)

```bash
python evaluation/run_all.py
```

### Main run_all.py options

```bash
# Single model or dataset
python evaluation/run_all.py --models re_lgbm --datasets 3

# Selected steps only
python evaluation/run_all.py --steps calibration shap

# Different conformal error level
python evaluation/run_all.py --alpha 0.05

# More boosters for SHAP ensemble (default 50)
python evaluation/run_all.py --n-shap-sample 100

# Retrain even if model is already saved
python evaluation/run_all.py --force-retrain
```

### Sensitivity over gamma grid

```bash
# re_lgbm M3 with gamma=0.33 and 1.0 (default)
python evaluation/run_all_gamma_comparison.py

# Arbitrary model and dataset
python evaluation/run_all_gamma_comparison.py --model bagging_lgbm --dataset 2 --gammas 0.33 0.66 1.0

# Only calibration and SHAP, with alpha=0.05
python evaluation/run_all_gamma_comparison.py --steps calibration shap --alpha 0.05
```

### Bootstrap CI (re_lgbm, any dataset)

```bash
# All datasets with default parameters (B=200, n_score from internal dict)
python evaluation/bootstrap_ci.py

# Only M3 with reduced n_boot
python evaluation/bootstrap_ci.py --datasets 3 --n-boot 50
```

---

## Output

Everything is saved under `evaluation/results/{model}_{Mn}/`.
For `run_all_gamma_comparison.py` folders include a gamma suffix:
`results/{model}_{Mn}_g{gamma}/`.

```
results/
`-- {model}_{Mn}/
    |-- models/
    |   `-- final_model.{txt|pkl}      # Model trained on all data (.pkl via DVC: `dvc pull`)
    |-- scores/
    |   |-- scores_oof.csv             # cig, fold, label, score_raw, score_calib_pf, score_calib_gl
    |   `-- scores_final.csv           # cig, label, score_raw_final, score_calib_final
    |-- calibration/
    |   |-- platt_params.csv           # a, b, SE per fold + "mean" and "global" rows
    |   |-- ece_brier.csv              # ECE and Brier per fold and overall
    |   `-- plots/reliability.{png,pdf}
    |-- conformal/
    |   |-- thresholds.csv             # q1, q0, n1_cal, n0_cal per fold
    |   |-- coverage.csv               # per-class coverage per fold + "mean_oof" row
    |   |-- prediction_sets.csv        # cig, label, set, width, pvalue_classe1
    |   `-- plots/{set_dist,pvalue_hist}.{png,pdf}
    |-- shap/
    |   |-- shap_values_final.npy      # (n_rows, n_features)
    |   |-- feature_names.json
    |   |-- feature_importance.csv     # mean |SHAP| per feature, sorted
    |   `-- plots/{bar_top20,beeswarm_top20,waterfall_*,pdp_top3}.{png,pdf}
    `-- bootstrap/                     # re_lgbm only
        |-- scores_ci.csv              # cig, score_mean/lower/upper, pc_mean/lower/upper
        |-- metrics_ci.csv             # metric, mean, lower, upper, sd
        `-- u_score_cigs.npy           # sampled CIGs (cache for reproducibility)
```

---

## gamma_star.json

Contains the canonical `gamma*` values used by `run_all.py` and `bootstrap_ci.py`.
Numeric keys `"1"`, `"2"`, `"3"` correspond to M1, M2, M3.

```json
{
  "lgb_supervised": {"1": 1.0, "2": 1.0, "3": 1.0},
  "bagging_lgbm":   {"1": 0.66, "2": 0.66, "3": 0.66},
  "re_lgbm":        {"1": 1.0, "2": 1.0, "3": 1.0},
  "puet":           {"1": 0.0, "2": 0.0, "3": 0.0}
}
```

Keys starting with `_` (e.g. `_note`, `_rationale`) are ignored by the code.
To change `gamma*`, just edit the numeric values — no other file needs to be touched.

---

## plots_exploration/

`make_qq_ranks.py` compares the score rankings across models (re_lgbm vs
lgb_supervised, re_lgbm vs bagging_lgbm) on U contracts outside the training set.
Requires that final models have already been trained and saved by `calibration.py`.
Output: `plots_exploration/qq_rank_comparison_M{n}.{png,pdf}` for M1, M2, M3.

```bash
python evaluation/plots_exploration/make_qq_ranks.py
```

---

## Prerequisites

```bash
pip install pandas numpy scipy scikit-learn lightgbm joblib pyarrow matplotlib shap
```

`shap` is imported lazily in `utils_expost.py` (only inside functions that actually
use it) to avoid the startup cost (~3-4s) when importing `utils_expost` from scripts
that do not need SHAP.
