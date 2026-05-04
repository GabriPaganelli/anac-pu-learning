# PNU Mixing — Riepilogo risultati griglia γ
**Data:** 2026-04-23  
**Seed reshuffle fold:** 2025 (M3→M2→M1 gerarchico)  
**Griglia γ:** {0, 0.05, 0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 0.9, 1.0}  
**Metrica primaria:** Lift@1% (media 4 fold OOF) | **Secondaria:** PR AUC  

---

## Nota metodologica

γ=0 → comportamento PU puro (N certi assenti dal training)  
γ=1 → comportamento PNPU originale (N certi a peso pieno)  
γ intermedio → mixing continuo secondo Sakai et al. ICML 2017  

**Fold:** reshuffled (seed=2025, indipendente dal benchmark). Stratificazione gerarchica
M3→M2→M1 per garantire bilanciamento P/N/U in ogni dataset (P/N NON identici
tra dataset: M3 ha 210P/280N, M2 ne aggiunge 286P/765N, M1 ne aggiunge 285P/772N).

**lgb_supervised:** γ=0 escluso (no negativi → classificatore degenere).

---

## 1. lgb_supervised (LGB P vs N certi, EM-Hard iter=0)

**Osservazione generale:** alta varianza tra fold (pochi labeled, nessun U in training).
Nessun trend γ affidabile su lift@1%; PR AUC stabile intorno a 0.58–0.61 su M1,
0.69–0.71 su M2, 0.86–0.88 su M3. Benchmark confermato nell'ordine di grandezza.

### M1
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.1 | **9.21** | 2.64 | 0.570 |
| 0.5 | 7.80 | 2.23 | **0.603** |
| 1.0 | 8.18 | 3.07 | 0.582 |

γ_best lift: 0.1 (marginale, non affidabile). γ_best PR AUC: 0.5.

### M2
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.1 | 10.13 | **12.34** | 0.696 |
| 0.6 | **14.94** | 13.49 | 0.679 |
| 1.0 | 4.25 | 3.33 | **0.710** |

⚠️ SD altissima — risultati inaffidabili su M2 per lift@1%. PR AUC piatto ~0.70 per tutti i γ.

### M3
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.1 | **20.86** | **20.08** | 0.859 |
| 0.4 | 9.56 | 4.76 | **0.883** |
| 1.0 | 5.23 | 5.01 | 0.873 |

⚠️ γ=0.1 outlier da un fold estremo (SD=20). γ_best affidabile PR AUC: 0.4.

**Conclusione lgb_supervised:** nessun γ_best stabile. Modello troppo rumoroso con
pochi labeled e nessun U. Per il modello finale: mantenere γ=1 (baseline storica).

---

## 2. RE LightGBM (nnPU, γ sul termine gradiente N certi)

**Osservazione generale:** pattern γ chiari e stabili. SD controllata.
M1/M2: più N certi → meglio (γ→1 ottimale). M3: curva a campana con picco a γ=0.33.

### M1 — trend monotono crescente
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.0 | 14.59 | 2.08 | 0.279 |
| 0.33 | 15.37 | 1.79 | 0.313 |
| 0.66 | 16.00 | 1.67 | 0.341 |
| **0.9** | **17.42** | 3.34 | 0.354 |
| 1.0 | 17.29 | 1.75 | **0.354** |

γ_best: **0.9** (lift), **0.9–1.0** (PR AUC). Benchmark γ=1: 17.41× ✓

### M2 — trend monotono crescente
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.0 | 13.91 | 0.95 | 0.288 |
| 0.5 | 16.74 | 2.54 | 0.373 |
| **1.0** | **18.76** | 3.92 | **0.397** |

γ_best: **1.0** (lift e PR AUC). Benchmark γ=1: 17.14× (nuovo fold leggermente superiore).

### M3 — curva a campana, picco a γ=0.33
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.0 | 20.00 | 1.77 | 0.458 |
| **0.33** | **22.37** | 2.24 | 0.523 |
| 0.5 | 21.43 | 2.81 | 0.534 |
| 1.0 | 19.05 | 3.50 | **0.547** |

γ_best lift: **0.33**. γ_best PR AUC: **1.0** (tensione: N certi aiutano il ranking globale
ma introducono rumore nel top-1%). Benchmark γ=0.03 originale: 21.90× ✓ (coerente).

**Conclusione RE LGB:**
- M1: γ_best ≈ 0.9
- M2: γ_best = 1.0
- M3: γ_best ≈ 0.33 (lift), 1.0 (PR AUC) — scegliere in base all'obiettivo

---

## 3. Bagging LightGBM (γ come sample_weight su N certi per bag)

**Osservazione generale:** pattern a campana su tutti i dataset — γ intermedio ottimale
per lift@1%. PR AUC monotonicamente crescente con γ. Pattern opposto a RE LGB su M1/M2.

### M1 — picco a γ=0.66
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.0 | 14.85 | 2.69 | 0.279 |
| 0.33 | 17.41 | 2.31 | 0.340 |
| **0.66** | **18.05** | 2.59 | 0.372 |
| 1.0 | 16.90 | 3.03 | **0.396** |

γ_best lift: **0.66**. Benchmark γ=1: 17.80× (γ=0.66 supera di +0.25×).

### M2 — picco a γ=0.80
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.0 | 11.89 | 0.74 | 0.287 |
| 0.5 | 16.33 | 2.04 | 0.375 |
| **0.8** | **16.54** | 1.48 | 0.400 |
| 1.0 | 16.13 | 1.52 | **0.413** |

γ_best lift: **0.8**. Benchmark γ=1: 18.15× (fold diversi, qui 16.54×).

### M3 — plateau a γ=0.33–0.66
| γ | lift@1% μ | ±SD | PR AUC μ |
|---|-----------|-----|----------|
| 0.0 | 20.49 | 3.38 | 0.440 |
| **0.33** | **22.38** | 2.83 | 0.525 |
| **0.66** | **22.38** | 2.83 | 0.561 |
| 1.0 | 21.42 | 3.15 | **0.583** |

γ_best lift: **0.33–0.66** (plateau). Benchmark γ=1: 21.43× (γ_best supera di +0.95×).

**Conclusione Bagging LGB:**
- M1: γ_best ≈ 0.66
- M2: γ_best ≈ 0.80
- M3: γ_best ≈ 0.33–0.66
- N certi a peso pieno (γ=1) sovrappesa il segnale giudiziario nel bagging → lieve degradazione lift

---

## 4. PUET (PU Extra Trees, γ come sample_weight su N certi per bag)

**Osservazione generale:** pattern radicalmente diverso dagli altri tre modelli.
γ=0 (PU puro) dà il miglior lift@1% su tutti i dataset; a γ=0.05 c'è un salto
brusco di PR AUC e ROC AUC (+0.06 e +0.11 su M1), poi plateau quasi piatto fino a γ=1.
I N certi causano una transizione di fase nel decision boundary di ExtraTrees — non un
miglioramento graduale — che migliora la calibrazione globale ma riduce la precisione
al top assoluto.

### M1 — γ=0 best lift, salto brusco PR AUC a γ=0.05
| γ | lift@1% μ | ±SD | PR AUC μ | ROC AUC μ |
|---|-----------|-----|----------|-----------|
| **0.0** | **13.57** | 1.24 | 0.277 | 0.416 |
| 0.05 | 10.88 | 1.41 | **0.337** | **0.530** |
| 0.33 | 11.14 | 1.36 | 0.337 | 0.530 |
| 1.0 | 10.63 | 1.55 | 0.337 | 0.531 |

γ_best lift: **0.0**. γ_best PR AUC / ROC AUC: **≥0.05** (plateau).
Benchmark γ=1: 15.75× (lift più basso con nuovi fold; struttura confermata).

### M2 — stesso pattern
| γ | lift@1% μ | ±SD | PR AUC μ | ROC AUC μ |
|---|-----------|-----|----------|-----------|
| **0.0** | **11.29** | 2.89 | 0.276 | 0.354 |
| 0.05 | 8.27 | 1.58 | **0.335** | **0.458** |
| 1.0 | 8.47 | 1.44 | 0.334 | 0.460 |

γ_best lift: **0.0**. γ_best PR AUC / ROC AUC: **≥0.05** (plateau).

### M3 — stesso pattern, valori discretizzati (pool < S)
| γ | lift@1% μ | ±SD | PR AUC μ | ROC AUC μ |
|---|-----------|-----|----------|-----------|
| **0.0** | **8.60** | 3.39 | 0.382 | 0.362 |
| 0.05 | 7.62 | 0.08 | **0.439** | **0.425** |
| 1.0 | 7.62 | 0.08 | 0.434 | 0.420 |

⚠️ Valori γ>0 molto discretizzati (pool=9k < S=15k → campionamento con rimpiazzo;
test fold piccolo → lift@1% può valere solo multipli di 100/n_P_test).
γ_best lift: **0.0** (ma SD alta). γ_best PR AUC: **0.05**.

**Conclusione PUET:** unico modello dove γ_best=0 domina lift@1%. La spiegazione:
ExtraTrees senza N certi impara a separare P dalla grande nuvola U; aggiungere N certi
(anche pochi, γ=0.05) ridefinisce il confine separando P da N-certi — migliorando
enormemente il ranking globale ma riducendo la precisione nella coda estrema.
Implicazione operativa: se la lista di ispezione è fissa al top-1%, usa PUET γ=0;
se la soglia è variabile, usa PUET γ≥0.05 (o qualsiasi valore — la differenza è minima).

---

## 5. Sintesi γ_best per modello e dataset

| Modello | M1 γ_best | M2 γ_best | M3 γ_best | Pattern |
|---------|-----------|-----------|-----------|---------|
| lgb_supervised | — (rumore) | — (rumore) | — (rumore) | nessun trend stabile |
| re_lgbm | **0.9** | **1.0** | **0.33** (lift) / 1.0 (PR) | monotono tranne M3 |
| bagging_lgbm | **0.66** | **0.80** | **0.33–0.66** | campana su tutti |
| puet | **0.0** | **0.0** | **0.0** (lift) / 0.05 (PR) | inversione: PU puro vince |

### Lift@1% a γ_best vs γ=1.0

| Modello | M1: γ_best / Δ vs γ=1 | M2: γ_best / Δ vs γ=1 | M3: γ_best / Δ vs γ=1 |
|---------|----------------------|----------------------|----------------------|
| re_lgbm | 0.9 / +0.13× | 1.0 / 0 | 0.33 / +3.32× |
| bagging_lgbm | 0.66 / +1.15× | 0.80 / +0.41× | 0.33 / +0.96× |
| puet | **0.0 / +2.94×** | **0.0 / +2.82×** | **0.0 / +0.98×** |

---

## 6. Interpretazione trasversale

**RE LGB:** N certi migliorano sempre (loss nnPU usa N come correzione al gradiente —
più segnale supervisato = meglio). Unica eccezione: M3 lift con γ > 0.33.

**Bagging LGB:** N certi a peso pieno sovrapesano il segnale giudiziario in ogni bag,
riducendo la diversità del bootstrap. Ottimale con γ intermedio (0.33–0.80).

**PUET:** pattern opposto a tutti. N certi ridefiniscono il confine separando P da N-certi
invece che P da U → migliora il ranking globale (PR AUC, ROC AUC) ma riduce la
precisione al top. Trasizione di fase a γ=0.05, poi plateau: la quantità di N certi
conta poco, la presenza sì.

**PR AUC vs lift@1%:** tensione sistematica in tutti i modelli. PR AUC preferisce
sempre γ alto (o qualsiasi γ>0 per PUET); lift@1% preferisce γ intermedio o zero.
Implicazione pratica: la scelta di γ dipende dall'obiettivo operativo.

| Obiettivo | γ consigliato | Modello preferito |
|-----------|--------------|-------------------|
| Lista breve fissa (top-1%) | γ_best < 1 o 0 | Bagging (γ~0.5) o PUET (γ=0) |
| Ranking globale robusto | γ = 1 | RE LGB o PUET (γ≥0.05) |
| Bilanciamento | γ intermedio | Bagging o RE LGB |

**M3 vs M1/M2:** su M3 il vantaggio di γ_best vs γ=1 è minore per RE LGB e Bagging.
M3 è il dataset più informativo (feature esecutive): il modello discrimina bene anche
senza ottimizzare γ. Su M1 (meno feature) il γ_best conta di più.
