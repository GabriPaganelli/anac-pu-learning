# selected_models — PNU Mixing (Roadmap §5)
# *selected_models — PNU Mixing (Roadmap §5)*

---

## Italiano

Quattro modelli finalisti valutati su una griglia di γ ∈ [0, 1] che controlla il peso degli **N certi** nel training (PNU Mixing, Sakai et al. ICML 2017).

γ=0 → PU puro (N certi assenti); γ=1 → PNPU originale (N certi a peso pieno); γ intermedio → interpolazione continua.

---

## Struttura

```
selected_models/
├── mixing_grid.py       # Entry point — sweep γ su tutti i modelli e dataset
├── plot_mixing.py       # Grafico lift@1% vs γ (legge results/, salva PNG/PDF)
├── utils.py             # Caricamento dati, reshuffle fold, metriche condivise
├── lgb_supervised.py    # LGB P vs N certi (EM-Hard iter=0)
├── bagging_lgbm.py      # Bagging LightGBM (PNPU)
├── re_lgbm.py           # RE LightGBM (nnPU loss, PNPU)
├── puet.py              # PU Extra Trees (preprocessed, PNPU)
└── results/             # CSV fold_metrics e summary per modello × dataset × γ
```

---

## Modelli

| File | Base learner | Dataset | γ applicato a |
|------|-------------|---------|---------------|
| `lgb_supervised.py` | LightGBM BCE | nativi | sample_weight N certi (γ=0 escluso: no negativi) |
| `bagging_lgbm.py` | LightGBM BCE (bagging) | nativi | sample_weight N certi per bag |
| `re_lgbm.py` | LightGBM nnPU (custom fobj) | nativi | coefficiente termine gradiente N certi |
| `puet.py` | ExtraTreesClassifier (bagging) | preprocessed | sample_weight N certi per bag |

---

## Rimescolamento dei fold

I fold vengono rimescolati **una sola volta** prima del loop γ, con strategia **gerarchica M3→M2→M1**: M3 (dataset più piccolo e più vincolato) viene rimescolato per primo, poi i CIG aggiuntivi di M2, poi quelli di M1. Garantisce fold bilanciati localmente per P, N e U in ciascun dataset (seed=2025, indipendente dal benchmark originale).

---

## Utilizzo

```bash
# Tutti i modelli, tutti i dataset (M1–M3)
python selected_models/mixing_grid.py

# Solo due modelli
python selected_models/mixing_grid.py --models re_lgbm puet

# Solo M3
python selected_models/mixing_grid.py --datasets 3

# Override griglia γ
python selected_models/mixing_grid.py --gamma 0.0 0.5 1.0

# Grafici lift@1% vs γ (legge results/ esistente)
python selected_models/plot_mixing.py
```

---

## Griglia γ di default

| Modelli | Griglia γ |
|---------|-----------|
| `bagging_lgbm`, `re_lgbm`, `puet` | 0, 0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0 |
| `lgb_supervised` | 0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0 |

---

## Output

```
results/mixing_{modello}_M{n}_fold_metrics.csv   # lift/auc per fold × γ
results/mixing_{modello}_M{n}_summary.csv        # media ± SD per γ
results/mixing_lift_vs_gamma.{png,pdf}           # grafici (da plot_mixing.py)
```

Metriche: **lift@1%**, lift@2%, lift@5%, PR-AUC, ROC-AUC.

---

## γ_best per modello (seed=2025)

| Modello | M1 γ_best | M2 γ_best | M3 γ_best |
|---------|-----------|-----------|-----------|
| lgb_supervised | — (alta varianza) | — (alta varianza) | — (alta varianza) |
| re_lgbm | 0.9 | 1.0 | 0.33 (lift) / 1.0 (PR AUC) |
| bagging_lgbm | 0.66 | 0.80 | 0.33–0.66 |
| puet | 0.0 | 0.0 | 0.0 (lift) / 0.05 (PR AUC) |

---

## `plot_mixing.py`

Legge i CSV di `results/` e genera un pannello 2×2 con errorbar (lift@1% ± SD vs γ) per M1/M2/M3. Se esiste `evaluation/gamma_star.json`, sovrappone linee verticali rosse con i γ* scelti manualmente.

---

## Prerequisiti

```bash
pip install lightgbm scikit-learn pandas numpy pyarrow scipy matplotlib
```

---

## English

Four finalist models evaluated on a grid of γ ∈ [0, 1] that controls the weight of **certain negatives (N certi)** during training (PNU Mixing, Sakai et al. ICML 2017).

γ=0 → pure PU (certain negatives absent); γ=1 → original PNPU (certain negatives at full weight); intermediate γ → continuous interpolation.

---

## Structure

```
selected_models/
├── mixing_grid.py       # Entry point — γ sweep over all models and datasets
├── plot_mixing.py       # Lift@1% vs γ plot (reads results/, saves PNG/PDF)
├── utils.py             # Data loading, fold reshuffle, shared metrics
├── lgb_supervised.py    # LGB P vs certain N (EM-Hard iter=0)
├── bagging_lgbm.py      # Bagging LightGBM (PNPU)
├── re_lgbm.py           # RE LightGBM (nnPU loss, PNPU)
├── puet.py              # PU Extra Trees (preprocessed, PNPU)
└── results/             # fold_metrics and summary CSVs per model × dataset × γ
```

---

## Models

| File | Base learner | Dataset | γ applied to |
|------|-------------|---------|--------------|
| `lgb_supervised.py` | LightGBM BCE | nativi | sample_weight of certain N (γ=0 excluded: no negatives) |
| `bagging_lgbm.py` | LightGBM BCE (bagging) | nativi | sample_weight of certain N per bag |
| `re_lgbm.py` | LightGBM nnPU (custom fobj) | nativi | coefficient of certain N gradient term |
| `puet.py` | ExtraTreesClassifier (bagging) | preprocessed | sample_weight of certain N per bag |

---

## Fold reshuffle

Folds are reshuffled **once** before the γ loop, using a **hierarchical M3→M2→M1 strategy**: M3 (the smallest and most constrained dataset) is reshuffled first, then additional CIGs (tender identifiers) from M2, then those from M1. This guarantees locally balanced folds for P, N, and U in each dataset (seed=2025, independent from the original benchmark).

---

## Usage

```bash
# All models, all datasets (M1–M3)
python selected_models/mixing_grid.py

# Only two models
python selected_models/mixing_grid.py --models re_lgbm puet

# Only M3
python selected_models/mixing_grid.py --datasets 3

# Override γ grid
python selected_models/mixing_grid.py --gamma 0.0 0.5 1.0

# Lift@1% vs γ plots (reads existing results/)
python selected_models/plot_mixing.py
```

---

## Default γ grid

| Models | γ grid |
|--------|--------|
| `bagging_lgbm`, `re_lgbm`, `puet` | 0, 0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0 |
| `lgb_supervised` | 0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0 |

---

## Output

```
results/mixing_{model}_M{n}_fold_metrics.csv   # lift/auc per fold × γ
results/mixing_{model}_M{n}_summary.csv        # mean ± SD per γ
results/mixing_lift_vs_gamma.{png,pdf}         # plots (from plot_mixing.py)
```

Metrics: **lift@1%**, lift@2%, lift@5%, PR-AUC, ROC-AUC.

---

## γ_best per model (seed=2025)

| Model | M1 γ_best | M2 γ_best | M3 γ_best |
|-------|-----------|-----------|-----------|
| lgb_supervised | — (high variance) | — (high variance) | — (high variance) |
| re_lgbm | 0.9 | 1.0 | 0.33 (lift) / 1.0 (PR AUC) |
| bagging_lgbm | 0.66 | 0.80 | 0.33–0.66 |
| puet | 0.0 | 0.0 | 0.0 (lift) / 0.05 (PR AUC) |

---

## `plot_mixing.py`

Reads CSVs from `results/` and generates a 2×2 panel with errorbars (lift@1% ± SD vs γ) for M1/M2/M3. If `evaluation/gamma_star.json` exists, overlays red vertical lines with the manually chosen γ* values.

---

## Prerequisites

```bash
pip install lightgbm scikit-learn pandas numpy pyarrow scipy matplotlib
```
