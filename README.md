# APPalti — PU Learning per la rilevazione di frodi negli appalti pubblici italiani

Davide Bortoletto, Giorgia Caicchiolo, Gabriele Paganelli, Gianmarco Rosa  
*Laboratorio di Statistica con le Aziende — Gruppo "L'Ipotesi Alternativa"*

> Bortoletto D., Caicchiolo G., Paganelli G., Rosa G. (2025). *Rilevazione di frodi negli appalti pubblici italiani: un approccio di Positive-Unlabeled Learning.* Working paper, Laboratorio di Statistica con le Aziende.

---

## Il problema

Solo una frazione esigua degli appalti pubblici italiani risulta associata a procedimenti giudiziari conclusi. La grande maggioranza è priva di etichetta: un appalto senza sentenza può essere genuinamente regolare oppure semplicemente impunito. Trattare i non etichettati come negativi certi introdurrebbe un bias sistematico — è il regime del **Positive-Unlabeled (PU) learning**.

Questo repository implementa una pipeline completa che, partendo dai microdati ANAC (Banca Dati Nazionale dei Contratti Pubblici, 2008–2025), costruisce un sistema di scoring del rischio corruttivo su ~9,5 milioni di appalti.

---

## Dataset

| Dataset | Bandi | Feature | P (condannati) | N (scagionati) | U per fold |
|---------|------:|--------:|---------------:|---------------:|-----------:|
| M1 (ex ante) | 9 468 795 | 24 | 781 | 1 817 | 10 650 |
| M2 (durante) | 3 469 880 | 40 | 496 | 1 045 | 6 386 |
| M3 (ex post) | 1 129 233 | 60 | 210 | 280 | 3 122 |

I tre dataset sono annidati (M3 ⊂ M2 ⊂ M1) e corrispondono a tre orizzonti temporali nel ciclo di vita dell'appalto. Le etichette positive derivano da sentenze TAR di accoglimento e da chiusure ANAC per reato o antimafia; le negative da ricorsi rigettati con sentenza definitiva.

Prior stimata: **π̂ = 0.02** (PULSNAR, regime SAR — Selected At Random).

---

## Pipeline

```
anac/01_sentenze/        TAR/CdS → cig_condannati.csv, cig_scagionati.csv
        ↓
anac/02_pipeline/        17 step: CSV ANAC + ISTAT → M1/M2/M3 parquet (6 file)
        ↓
prior_estimation/        PULSNAR → π̂ = 0.02
        ↓
models/                  Benchmark: Biased (Logit/SVM/RF/GBM), nnPU, EM-like, Bagging
        ↓
selected_models/         4 finalisti × griglia γ (PNU Mixing) → γ* per modello
        ↓
evaluation/              Calibrazione Platt · Conformal · SHAP · Bootstrap CI
        ↓
app/                     Streamlit: scoring on-the-fly, bootstrap CI, SHAP, batch CSV
```

---

## Modelli finalisti

PNU Mixing (Sakai et al. 2017) applicato su griglia γ ∈ [0, 1]. γ* selezionato via Lift@1% OOF.

| Modello | γ* | Lift@1% M1 | Lift@1% M2 | Lift@1% M3 | PR AUC M3 |
|---------|---:|----------:|----------:|----------:|----------:|
| LightGBM P vs N (lgb_supervised) | 1.0 | 20.95 | 13.49 | 17.86 | 0.885 ± 0.027 |
| Bagging LightGBM | 0.66 | 17.80 | 18.15 | 21.43 | 0.591 ± 0.078 |
| RE LightGBM (nnPU) | 1.0 | 17.41 | 17.14 | 21.90 | 0.556 ± 0.063 |
| PUET | 0.0 | 15.75 | 15.73 | 14.76 | 0.591 ± 0.021 |

Il **RE LightGBM** è il modello di riferimento: fondazione teorica nnPU, stabilità lift, calibrazione ECE < 0.06 dopo Platt scaling, coerenza SHAP con i meccanismi noti della corruzione.

---

## Struttura del repository

```
anac/
├── 01_sentenze/         Download e parsing sentenze OpenGA → etichette
├── 02_pipeline/         Feature engineering ANAC (17 script) → parquet M1/M2/M3
└── utils/               Download dati contestuali ISTAT, dizionario variabili
prior_estimation/        Stimatori prior: PULSNAR (SAR), KM2, Elkan-Noto, Blanchard
metrics/                 Metriche condivise: lift@k%, PR AUC, preprocessing OOF
models/                  Benchmark completo: biased, nnPU, two-step EM, bagging
selected_models/         4 finalisti + PNU Mixing grid + γ* + grafici lift vs γ
evaluation/              Ex-post: calibrazione · conformal · SHAP · bootstrap CI
app/                     Streamlit app (APPalti): scoring singolo e batch
eda/                     Analisi esplorativa: UMAP, distribuzioni, correlazioni
requirements.txt         Dipendenze Python
```

---

## Riproduzione dei risultati

L'ordine di esecuzione riflette le dipendenze tra i moduli.

```bash
# 1. Etichette da sentenze TAR/CdS
python anac/01_sentenze/run_pipeline.py

# 2. Feature engineering (legge data/raw/ e labels/)
python anac/02_pipeline/run_pipeline.py

# 3. Stima prior (richiede R + PULSNAR installato)
python prior_estimation/run_prior_estimation.py

# 4. Benchmark modelli (opzionale — i risultati sono già in selected_models/results/)
python selected_models/mixing_grid.py

# 5. Ex-post analysis (calibration → conformal → SHAP)
python evaluation/run_all.py

# 6. Bootstrap CI (solo re_lgbm)
python evaluation/bootstrap_ci.py
```

Per riprodurre la sensitivity γ su re_lgbm M3:

```bash
python evaluation/run_all_gamma_comparison.py --gammas 0.33 1.0
```

---

## App

```bash
# Setup una tantum (pre-encoding parquet nativi)
python app/setup.py

# Avvio
streamlit run app/app.py
```

L'app addestra il modello on-the-fly sulla distribuzione storica e restituisce il rango percentile del contratto inserito rispetto ai ~9,5 milioni di appalti ANAC. Supporta modalità singola e batch CSV.

---

## Dati

I file pesanti (parquet ANAC, modelli addestrati) non sono versionati su git. La cartella `app/data/` contiene i parquet pre-processati necessari all'app; i parquet sorgente vanno scaricati manualmente da [dati.anticorruzione.it](https://dati.anticorruzione.it/) e posizionati in `anac/data/raw/`.

---

## Dipendenze

```bash
pip install -r requirements.txt
```

PULSNAR richiede inoltre **R** con i pacchetti `mclust` e `e1071`. Per l'installazione vedere la documentazione di [PULSNAR](https://github.com/unmtransinfo/PULSNAR).

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

---

## English version below

# APPalti — PU Learning for fraud detection in Italian public procurement

Davide Bortoletto, Giorgia Caicchiolo, Gabriele Paganelli, Gianmarco Rosa  
*Laboratorio di Statistica con le Aziende — Gruppo "L'Ipotesi Alternativa"*

> Bortoletto D., Caicchiolo G., Paganelli G., Rosa G. (2025). *Fraud detection in Italian public procurement: a Positive-Unlabeled Learning approach.* Working paper, Laboratorio di Statistica con le Aziende.

---

## The problem

Only a small fraction of Italian public contracts (appalti) is linked to concluded judicial proceedings. The vast majority is unlabelled: a contract without a ruling may be genuinely compliant or simply unpunished. Treating unlabelled examples as certain negatives introduces a systematic bias — this is the **Positive-Unlabeled (PU) learning** regime.

This repository implements a complete pipeline that, starting from ANAC microdata (National Public Contracts Database, 2008–2025), builds a corruption risk scoring system over ~9.5 million contracts.

---

## Dataset

| Dataset | Contracts | Features | P (condemned) | N (cleared) | U per fold |
|---------|----------:|---------:|--------------:|------------:|-----------:|
| M1 (ex ante) | 9 468 795 | 24 | 781 | 1 817 | 10 650 |
| M2 (durante) | 3 469 880 | 40 | 496 | 1 045 | 6 386 |
| M3 (ex post) | 1 129 233 | 60 | 210 | 280 | 3 122 |

The three datasets are nested (M3 ⊂ M2 ⊂ M1) and correspond to three temporal horizons in the contract lifecycle. Positive labels come from TAR (Regional Administrative Court) rulings in favour of the appellant and ANAC closures for crime or anti-mafia provisions; negative labels from definitively rejected appeals.

Estimated prior: **π̂ = 0.02** (PULSNAR, SAR — Selected At Random regime).

---

## Pipeline

```
anac/01_sentenze/        TAR/CdS rulings → cig_condannati.csv, cig_scagionati.csv
        ↓
anac/02_pipeline/        17 steps: ANAC + ISTAT CSVs → M1/M2/M3 parquets (6 files)
        ↓
prior_estimation/        PULSNAR → π̂ = 0.02
        ↓
models/                  Benchmark: Biased (Logit/SVM/RF/GBM), nnPU, EM-like, Bagging
        ↓
selected_models/         4 finalists × γ grid (PNU Mixing) → γ* per model
        ↓
evaluation/              Platt calibration · Conformal prediction · SHAP · Bootstrap CI
        ↓
app/                     Streamlit: on-the-fly scoring, bootstrap CI, SHAP, batch CSV
```

---

## Finalist models

PNU Mixing (Sakai et al. 2017) applied over a grid γ ∈ [0, 1]. γ* selected via OOF Lift@1%.

| Model | γ* | Lift@1% M1 | Lift@1% M2 | Lift@1% M3 | PR AUC M3 |
|-------|---:|----------:|----------:|----------:|----------:|
| LightGBM P vs N (lgb_supervised) | 1.0 | 20.95 | 13.49 | 17.86 | 0.885 ± 0.027 |
| Bagging LightGBM | 0.66 | 17.80 | 18.15 | 21.43 | 0.591 ± 0.078 |
| RE LightGBM (nnPU) | 1.0 | 17.41 | 17.14 | 21.90 | 0.556 ± 0.063 |
| PUET | 0.0 | 15.75 | 15.73 | 14.76 | 0.591 ± 0.021 |

**RE LightGBM** is the reference model: nnPU theoretical foundation, stable lift, ECE < 0.06 after Platt scaling, SHAP importances consistent with known corruption mechanisms.

---

## Repository structure

```
anac/
├── 01_sentenze/         OpenGA ruling download and parsing → labels
├── 02_pipeline/         ANAC feature engineering (17 scripts) → M1/M2/M3 parquets
└── utils/               ISTAT contextual data download, variable dictionary
prior_estimation/        Prior estimators: PULSNAR (SAR), KM2, Elkan-Noto, Blanchard
metrics/                 Shared metrics: lift@k%, PR AUC, OOF preprocessing
models/                  Full benchmark: biased, nnPU, two-step EM, bagging
selected_models/         4 finalists + PNU Mixing grid + γ* + lift-vs-γ plots
evaluation/              Ex-post: calibration · conformal · SHAP · bootstrap CI
app/                     Streamlit app (APPalti): single and batch scoring
eda/                     Exploratory analysis: UMAP, distributions, correlations
requirements.txt         Python dependencies
```

---

## Reproducing the results

The execution order follows module dependencies.

```bash
# 1. Labels from TAR/CdS rulings
python anac/01_sentenze/run_pipeline.py

# 2. Feature engineering (reads data/raw/ and labels/)
python anac/02_pipeline/run_pipeline.py

# 3. Prior estimation (requires R + PULSNAR installed)
python prior_estimation/run_prior_estimation.py

# 4. Model benchmark (optional — results already in selected_models/results/)
python selected_models/mixing_grid.py

# 5. Ex-post analysis (calibration → conformal → SHAP)
python evaluation/run_all.py

# 6. Bootstrap CI (re_lgbm only)
python evaluation/bootstrap_ci.py
```

To reproduce the γ sensitivity analysis for re_lgbm M3:

```bash
python evaluation/run_all_gamma_comparison.py --gammas 0.33 1.0
```

---

## App

```bash
# One-time setup (pre-encode native parquets)
python app/setup.py

# Launch
streamlit run app/app.py
```

The app trains the model on-the-fly on the historical distribution and returns the percentile rank of the submitted contract relative to the ~9.5 million ANAC contracts. Supports single-contract and batch CSV modes.

---

## Data

Large files (ANAC parquets, trained models) are not versioned. The `app/data/` folder contains the pre-processed parquets needed by the app; source parquets must be downloaded manually from [dati.anticorruzione.it](https://dati.anticorruzione.it/) and placed in `anac/data/raw/`.

---

## Dependencies

```bash
pip install -r requirements.txt
```

PULSNAR additionally requires **R** with the `mclust` and `e1071` packages. See the [PULSNAR](https://github.com/unmtransinfo/PULSNAR) documentation for installation.

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
