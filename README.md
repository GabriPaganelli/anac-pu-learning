# APPalti — PU Learning per la rilevazione di frodi negli appalti pubblici italiani
# *APPalti — PU Learning for fraud detection in Italian public procurement*

---

## Italiano

Davide Bortoletto, Giorgia Caicchiolo, Gabriele Paganelli, Gianmarco Rosa  
*Laboratorio di Statistica con le Aziende — Gruppo "L'Ipotesi Alternativa"*

---

## Il problema

Solo una frazione esigua degli appalti pubblici italiani risulta associata a procedimenti giudiziari conclusi. La grande maggioranza è priva di etichetta: un appalto senza sentenza può essere genuinamente regolare oppure semplicemente impunito. Trattare i non etichettati come negativi certi introdurrebbe un bias sistematico: è il regime del **Positive-Unlabeled (PU) learning**.

Questo repository implementa una pipeline completa che, partendo dai microdati ANAC (Banca Dati Nazionale dei Contratti Pubblici, 2008–2025), costruisce un sistema di scoring del rischio corruttivo su ~9,5 milioni di appalti.

---

## Dataset

Tre modelli temporali annidati (M3 ⊂ M2 ⊂ M1), corrispondenti a tre orizzonti nel ciclo di vita dell'appalto. Le etichette positive derivano da sentenze TAR di accoglimento e da chiusure ANAC per reato o antimafia; le negative da ricorsi rigettati con sentenza definitiva.

| Dataset | Bandi | Feature | P (condannati) | N (scagionati) | U per fold |
|---------|------:|--------:|---------------:|---------------:|-----------:|
| M1 — ex ante | 9 468 795 | 24 | 781 | 1 817 | 10 650 |
| M2 — durante | 3 469 880 | 40 | 496 | 1 045 | 6 386 |
| M3 — ex post | 1 129 233 | 60 | 210 | 280 | 3 122 |

Prior stimata: **π̂ = 0.02** (PULSNAR, regime SAR).

---

## Struttura del repository

```
anac/                    Dati e pipeline di preprocessing
eda/                     Analisi esplorativa (R)
prior_estimation/        Stima della prior π
models/                  Benchmark completo di modelli PU
selected_models/         4 finalisti con PNU Mixing (griglia γ)
evaluation/              Calibrazione, conformal prediction, SHAP, bootstrap CI
app/                     Applicazione Streamlit (APPalti)
metrics/                 Metriche condivise (lift@k%, PR AUC)
docs/                    Working paper, poster, demo dell'app
requirements.txt         Dipendenze Python
```

---

## `anac/` — Dati e feature engineering

```
anac/
├── 01_sentenze/     Scarica le sentenze TAR/CdS da OpenGA → cig_condannati.csv, cig_scagionati.csv
├── 02_pipeline/     17 script: CSV ANAC + dati ISTAT → bando_cig_all.parquet → M1/M2/M3
├── data/
│   ├── raw/         CSV ANAC originali (gestiti da DVC)
│   ├── territorial/ Dati contestuali provinciali ISTAT/MEF
│   └── lookup/      Tabelle di lookup
├── labels/          cig_condannati.csv · cig_scagionati.csv
├── output/parquet/  bando_cig_all.parquet · model/nativi/ · model/preprocessed/
├── utils/           Download dati territoriali, generazione dizionario variabili
└── variables/       Selezione variabili (xlsx) · dizionario dati (xlsx)
```

Ogni dataset è disponibile in due versioni: **nativi** (per LightGBM/XGBoost) e **preprocessed** (per Logistica/SVM, con discretizzazione log-quantile e encoding categoriale). La colonna `fold` (0–3) permette cross-validation stratificata con propagazione gerarchica M3→M2→M1.

Fonti dati: bandi e contratti da [ANAC BDNCP](https://dati.anticorruzione.it/), sentenze da [OpenGA](https://openga.giustizia-amministrativa.it/), dati territoriali da ISTAT SDMX e MEF.

---

## `eda/` — Analisi esplorativa

Script R unico (`eda.R`) che produce 18 grafici, dalla panoramica descrittiva agli indicatori forensi (Benford, bunching alle soglie del Codice Appalti, Corruption Risk Indicators di Fazekas et al., bid-rigging OECD) fino alla proiezione UMAP con distanza di Gower su ~12k contratti per modello.

```bash
Rscript eda/eda.R   # ~20–40 min; dipendenze R: arrow, ggplot2, uwot, cluster, ...
```

---

## `prior_estimation/` — Stima della prior π

Quattro stimatori implementati: **Elkan-Noto**, **Blanchard Quantili**, **KM2** (DEDPUL/KMPE, vendored) e **PULSNAR** (richiede R via `rpy2`). Il valore operativo adottato è quello di PULSNAR, l'unico che non risulta distorto. Si arrotonda il valore per eccesso (approccio conservativo).

```bash
python prior_estimation/run_prior_estimation.py   # tempi: 1–7h secondo il metodo
```

PULSNAR richiede R con i pacchetti `mclust` e `e1071` e l'installazione separata del pacchetto Python:
```bash
pip install git+https://github.com/unmtransinfo/PULSNAR.git
pip install rpy2 xgboost catboost
```

---

## `models/` — Benchmark modelli PU

Quattro famiglie, ciascuna disponibile per M1/M2/M3 e nelle varianti PU e PNPU (con termine correttivo per i negativi certi):

| Famiglia | Strategia | Base learner |
|----------|-----------|--------------|
| `risk_estimators/` | nnPU loss corretta per π | LightGBM, Logistica, MLP |
| `bagging/` | PU Bagging (Mordelet & Vert 2014) | LightGBM, ExtraTrees |
| `biased/` | Unlabeled come negativi pesati | LightGBM, Logistica, RF, SVM |
| `em_like/` | Iterativo EM-like | LightGBM (soft e hard) |

Ogni script ha una sezione `CONFIGURAZIONE` in testa con `MODEL_NUMBER`, `PNPU` e `TEST_MODE`.

---

## `selected_models/` — PNU Mixing e selezione γ

I 4 modelli finalisti vengono valutati su una griglia γ ∈ [0, 1] che controlla il peso dei negativi certi nel training (PNU Mixing, Sakai et al. ICML 2017). γ* viene selezionato via Lift@1% OOF.

```bash
python selected_models/mixing_grid.py              # sweep completo
python selected_models/mixing_grid.py --models re_lgbm --datasets 3
python selected_models/plot_mixing.py              # grafici lift vs γ
```

| Modello | γ* | Lift@1% M1 | Lift@1% M2 | Lift@1% M3 | PR AUC M3 |
|---------|---:|----------:|----------:|----------:|----------:|
| LightGBM P vs N (`lgb_supervised`) | 1.0 | 20.95 | 13.49 | 17.86 | 0.885 ± 0.027 |
| Bagging LightGBM | 0.66 | 17.80 | 18.15 | 21.43 | 0.591 ± 0.078 |
| RE LightGBM (nnPU) | 1.0 | 17.41 | 17.14 | 21.90 | 0.556 ± 0.063 |
| PUET | 0.0 | 15.75 | 15.73 | 14.76 | 0.591 ± 0.021 |

Il **RE LightGBM** è il modello di riferimento: fondazione teorica nnPU, stabilità del lift, ECE < 0.06 dopo Platt scaling, coerenza SHAP con i meccanismi noti della corruzione. LightGBM P vs N ha prestazioni ottime su alcune metriche, ma su altre è molto distorto (come prevedibile).

---

## `evaluation/` — Analisi ex-post

Pipeline ex-post applicata ai 4 finalisti su M1/M2/M3: calibrazione Platt (4-fold OOF + fit globale), conformal prediction Mondrian (split, α=0.10), analisi SHAP con TreeSHAP, bootstrap CI (solo RE LightGBM).

```bash
python evaluation/run_all.py                          # pipeline completa
python evaluation/run_all.py --models re_lgbm --datasets 3 --steps calibration shap
python evaluation/bootstrap_ci.py --datasets 3
python evaluation/run_all_gamma_comparison.py --gammas 0.33 1.0   # sensitivity γ
```

I risultati vengono salvati in `evaluation/results/{modello}_{Mn}/` (calibrazione, conformal, SHAP, bootstrap).

---

## `app/` — APPalti (Streamlit)

Applicazione web per lo scoring del rischio corruttivo su singoli appalti o batch CSV. Addestra RE LightGBM on-the-fly sulla distribuzione storica e restituisce il rango percentile del contratto rispetto ai ~9,5 milioni di appalti ANAC.

```bash
python app/setup.py        # una tantum: pre-encoding dei parquet nativi → app/data/
streamlit run app/app.py
```

Funzionalità: scoring singolo · bootstrap CI (5°–95° percentile del rank) · SHAP per-feature · modalità batch CSV.

---

## Dati (DVC)

I file pesanti (CSV ANAC, parquet, modelli addestrati) sono versionati con **DVC** su Dagshub. Per scaricarli dopo aver clonato il repository:

```bash
pip install dvc dvc-dagshub
dvc pull
```

I CSV ANAC originali sono anche scaricabili direttamente da [dati.anticorruzione.it](https://dati.anticorruzione.it/).

---

## Riproduzione dei risultati

```bash
# 1. Dati territoriali ISTAT/MEF
python anac/utils/download_contesto.py

# 2. Etichette da sentenze TAR/CdS
python anac/01_sentenze/run_pipeline.py

# 3. Feature engineering
python anac/02_pipeline/run_pipeline.py

# 4. Analisi esplorativa (opzionale)
Rscript eda/eda.R

# 5. Stima prior
python prior_estimation/run_prior_estimation.py

# 6. PNU Mixing sui 4 finalisti
python selected_models/mixing_grid.py

# 7. Analisi ex-post
python evaluation/run_all.py
python evaluation/bootstrap_ci.py

# 8. App
python app/setup.py && streamlit run app/app.py
```

---

## Dipendenze

```bash
pip install -r requirements.txt
```

Per `prior_estimation` e `eda`: **R** con `mclust`, `e1071`, `arrow`, `ggplot2`, `uwot`, `cluster` e altri (vedere i rispettivi README).

---

## Licenza

[GNU AGPL v3](LICENSE)

---

## Riferimenti

- Kiryo R., Niu G., du Plessis M.C., Sugiyama M. (2017). *Positive-Unlabeled Learning with Non-Negative Risk Estimator.* NeurIPS.
- Sakai T., du Plessis M.C., Niu G., Sugiyama M. (2017). *Semi-supervised classification based on classification from positive and unlabeled data.* ICML.
- Mordelet F., Vert J.-P. (2014). *A bagging SVM to learn from positive and unlabeled examples.* Pattern Recognition Letters.
- Wilton R. et al. (2022). *PUET: Positive-Unlabeled Learning with Extra Trees.*
- Kumar A. et al. (2023). *PULSNAR: Positive Unlabeled Learning Selected Not At Random.* arXiv:2306.03383.
- Fazekas I. et al. (2016). *Corruption risks in public procurement.* Government Transparency Institute.

---

## English

Davide Bortoletto, Giorgia Caicchiolo, Gabriele Paganelli, Gianmarco Rosa  
*Laboratorio di Statistica con le Aziende — Gruppo "L'Ipotesi Alternativa"*

> Bortoletto D., Caicchiolo G., Paganelli G., Rosa G. (2025). *Fraud detection in Italian public procurement: a Positive-Unlabeled Learning approach.* Working paper, Laboratorio di Statistica con le Aziende.

---

## The problem

Only a small fraction of Italian public contracts is linked to concluded judicial proceedings. The vast majority is unlabelled: a contract without a ruling may be genuinely compliant or simply unpunished. Treating unlabelled examples as certain negatives introduces a systematic bias — this is the **Positive-Unlabeled (PU) learning** regime.

This repository implements a complete pipeline that, starting from ANAC microdata (National Public Contracts Database, 2008–2025), builds a corruption risk scoring system over ~9.5 million contracts.

---

## Dataset

Three temporally nested models (M3 ⊂ M2 ⊂ M1), corresponding to three horizons in the contract lifecycle. Positive labels come from TAR administrative court rulings in favour of the appellant and ANAC closures for crime or anti-mafia provisions; negative labels from definitively rejected appeals.

| Dataset | Contracts | Features | P (condemned) | N (cleared) | U per fold |
|---------|----------:|---------:|--------------:|------------:|-----------:|
| M1 — ex ante | 9 468 795 | 24 | 781 | 1 817 | 10 650 |
| M2 — during | 3 469 880 | 40 | 496 | 1 045 | 6 386 |
| M3 — ex post | 1 129 233 | 60 | 210 | 280 | 3 122 |

Estimated prior: **π̂ = 0.02** (PULSNAR, SAR regime).

---

## Repository structure

```
anac/                    Data and preprocessing pipeline
eda/                     Exploratory analysis (R)
prior_estimation/        Prior π estimation
models/                  Full PU model benchmark
selected_models/         4 finalists with PNU Mixing (γ grid)
evaluation/              Calibration, conformal prediction, SHAP, bootstrap CI
app/                     Streamlit application (APPalti)
metrics/                 Shared metrics (lift@k%, PR AUC)
docs/                    Working paper, poster, app demo
requirements.txt         Python dependencies
```

---

## `anac/` — Data and feature engineering

```
anac/
├── 01_sentenze/     Downloads TAR/CdS rulings from OpenGA → cig_condannati.csv, cig_scagionati.csv
├── 02_pipeline/     17 scripts: ANAC CSVs + ISTAT data → bando_cig_all.parquet → M1/M2/M3
├── data/
│   ├── raw/         Raw ANAC CSV files (managed by DVC)
│   ├── territorial/ Provincial context data from ISTAT/MEF
│   └── lookup/      Lookup tables
├── labels/          cig_condannati.csv · cig_scagionati.csv
├── output/parquet/  bando_cig_all.parquet · model/nativi/ · model/preprocessed/
├── utils/           Territorial data download, data dictionary generation
└── variables/       Variable selection (xlsx) · data dictionary (xlsx)
```

Each model is available in two variants: **native** (for LightGBM/XGBoost, handles NA natively) and **preprocessed** (for Logistic Regression/SVM, with log-quantile discretisation and categorical encoding). The `fold` column (0–3) enables stratified cross-validation with hierarchical propagation M3→M2→M1.

Data sources: contracts from [ANAC BDNCP](https://dati.anticorruzione.it/), court rulings from [OpenGA](https://openga.giustizia-amministrativa.it/), territorial data from ISTAT SDMX and MEF.

---

## `eda/` — Exploratory data analysis

Single R script (`eda.R`) producing 18 plots: descriptive overview, forensic indicators (Benford's law, bunching at Procurement Code thresholds, Fazekas et al. Corruption Risk Indicators, OECD bid-rigging), and UMAP projection with Gower distance on ~12k contracts per model.

```bash
Rscript eda/eda.R   # ~20–40 min; R dependencies: arrow, ggplot2, uwot, cluster, ...
```

---

## `prior_estimation/` — Prior π estimation

Four estimators: **Elkan-Noto**, **Blanchard Quantile**, **KM2** (DEDPUL/KMPE, vendored), and **PULSNAR** (requires R via `rpy2`). The operational value adopted is that of PULSNAR, the only unbiased method. The value is rounded up (conservative approach).

```bash
python prior_estimation/run_prior_estimation.py   # runtime: 1–7h depending on method
```

PULSNAR additionally requires R with `mclust` and `e1071`, plus:
```bash
pip install git+https://github.com/unmtransinfo/PULSNAR.git
pip install rpy2 xgboost catboost
```

---

## `models/` — PU model benchmark

Four families, each available for M1/M2/M3 and in PU and PNPU variants (with corrective term for confirmed negatives):

| Family | Strategy | Base learner |
|--------|----------|--------------|
| `risk_estimators/` | nnPU π-corrected loss | LightGBM, Logistic regression, MLP PyTorch |
| `bagging/` | PU Bagging (Mordelet & Vert 2014) | LightGBM, ExtraTrees |
| `biased/` | Unlabeled as down-weighted negatives | LightGBM, Logistic, RF, SVM |
| `em_like/` | Iterative EM-like | R + LightGBM (soft and hard) |

Each script has a `CONFIGURAZIONE` section at the top with `MODEL_NUMBER`, `PNPU`, and `TEST_MODE`.

---

## `selected_models/` — PNU Mixing and γ selection

The 4 finalist models are evaluated on a grid γ ∈ [0, 1] controlling the weight of confirmed negatives during training (PNU Mixing, Sakai et al. ICML 2017). γ* is selected via OOF Lift@1%.

```bash
python selected_models/mixing_grid.py              # full sweep
python selected_models/mixing_grid.py --models re_lgbm --datasets 3
python selected_models/plot_mixing.py              # lift vs γ plots
```

| Model | γ* | Lift@1% M1 | Lift@1% M2 | Lift@1% M3 | PR AUC M3 |
|-------|---:|----------:|----------:|----------:|----------:|
| LightGBM P vs N (`lgb_supervised`) | 1.0 | 20.95 | 13.49 | 17.86 | 0.885 ± 0.027 |
| Bagging LightGBM | 0.66 | 17.80 | 18.15 | 21.43 | 0.591 ± 0.078 |
| RE LightGBM (nnPU) | 1.0 | 17.41 | 17.14 | 21.90 | 0.556 ± 0.063 |
| PUET | 0.0 | 15.75 | 15.73 | 14.76 | 0.591 ± 0.021 |

**RE LightGBM** is the reference model: nnPU theoretical foundation, stable lift across datasets, ECE < 0.06 after Platt scaling, SHAP importances consistent with known corruption mechanisms.

---

## `evaluation/` — Ex-post analysis

Ex-post pipeline applied to the 4 finalists on M1/M2/M3: Platt calibration (4-fold OOF + global fit), Mondrian split conformal prediction (α=0.10), SHAP with TreeSHAP, bootstrap CI (RE LightGBM only).

```bash
python evaluation/run_all.py                          # full pipeline
python evaluation/run_all.py --models re_lgbm --datasets 3 --steps calibration shap
python evaluation/bootstrap_ci.py --datasets 3
python evaluation/run_all_gamma_comparison.py --gammas 0.33 1.0   # γ sensitivity
```

Results are saved under `evaluation/results/{model}_{Mn}/` (calibration, conformal, SHAP, bootstrap).

---

## `app/` — APPalti (Streamlit)

Web application for corruption risk scoring on individual contracts or batch CSV files. Trains RE LightGBM on-the-fly on the historical distribution and returns the percentile rank of the contract relative to the ~9.5 million ANAC contracts.

```bash
python app/setup.py        # one-time: pre-encode native parquets → app/data/
streamlit run app/app.py
```

Features: single scoring · bootstrap CI (5th–95th percentile of the rank) · per-feature SHAP · batch CSV mode.

---

## Data (DVC)

Large files (ANAC CSVs, parquets, trained models) are versioned with **DVC** on Dagshub. To download them after cloning:

```bash
pip install dvc dvc-dagshub
dvc pull
```

The raw ANAC CSVs are also directly downloadable from [dati.anticorruzione.it](https://dati.anticorruzione.it/).

---

## Reproducing the results

```bash
# 1. Territorial data ISTAT/MEF
python anac/utils/download_contesto.py

# 2. Labels from TAR/CdS court rulings
python anac/01_sentenze/run_pipeline.py

# 3. Feature engineering
python anac/02_pipeline/run_pipeline.py

# 4. Exploratory analysis (optional)
Rscript eda/eda.R

# 5. Prior estimation
python prior_estimation/run_prior_estimation.py

# 6. PNU Mixing on the 4 finalists
python selected_models/mixing_grid.py

# 7. Ex-post analysis
python evaluation/run_all.py
python evaluation/bootstrap_ci.py

# 8. App
python app/setup.py && streamlit run app/app.py
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

For `prior_estimation` and `eda`: **R** with `mclust`, `e1071`, `arrow`, `ggplot2`, `uwot`, `cluster`, and others (see individual READMEs).

---

## License

[GNU AGPL v3](LICENSE)

---

## References

- Kiryo R., Niu G., du Plessis M.C., Sugiyama M. (2017). *Positive-Unlabeled Learning with Non-Negative Risk Estimator.* NeurIPS.
- Sakai T., du Plessis M.C., Niu G., Sugiyama M. (2017). *Semi-supervised classification based on classification from positive and unlabeled data.* ICML.
- Mordelet F., Vert J.-P. (2014). *A bagging SVM to learn from positive and unlabeled examples.* Pattern Recognition Letters.
- Wilton R. et al. (2022). *PUET: Positive-Unlabeled Learning with Extra Trees.*
- Kumar A. et al. (2023). *PULSNAR: Positive Unlabeled Learning Selected Not At Random.* arXiv:2306.03383.
- Fazekas I. et al. (2016). *Corruption risks in public procurement.* Government Transparency Institute.
