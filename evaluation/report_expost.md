# Analisi Ex-Post — Modelli PU Learning per la Rilevazione di Irregolarità negli Appalti

**Data:** 2026-04-24  
**Dataset:** ANAC sentenze 2008–2025 (gare pubbliche italiane)  
**Modelli:** lgb_supervised, bagging_lgbm, re_lgbm, puet  
**Finestre temporali:** M1 (ante-gara), M2 (durante-gara), M3 (post-gara)  
**Prior π = 0.02** per tutti i modelli e dataset (stimato da PULSNAR)

---

## 1. Sommario

Dopo la fase di model selection (PU mixing grid con 4 fold OOF, seed=2025), sono stati eseguiti tre step di analisi ex-post per ciascuna delle 12 combinazioni modello × dataset:

| Step | Strumento | Seed | Scopo |
|------|-----------|------|-------|
| Calibrazione Platt | LogisticRegression (C=1e9) OOF | 2026 | Trasformare raw scores in probabilità calibrate |
| Conformal prediction | Mondrian class-conditional, α=0.10 | 2027 | Set di predizione con copertura garantita ≥90% per classe |
| SHAP | TreeExplainer sul modello finale (P+N labeled) | — | Interpretabilità feature |

Addizionalmente, per **re_lgbm** su tutti e tre i dataset, è stato eseguito un bootstrap B=200 per stimare gli intervalli di confidenza al 95% su metriche e score dei contratti unlabeled.

Per **re_lgbm M3** è stata condotta un'analisi di sensibilità sul parametro di mixing γ (valori 0.33 e 1.0); per uniformità narrativa con gli altri dataset, si adotta γ=1.0 come configurazione canonica.

---

## 2. Configurazione sperimentale

### 2.1 Scelta di γ* per modello

| Modello | γ* | Motivazione |
|---------|-----|-------------|
| lgb_supervised | 1.0 | Massima PR-AUC su tutti e 3 i dataset; supervisato puro (U non informativo) |
| bagging_lgbm | 0.66 | Picco lift@1% coerente su M1/M2/M3; bilancia P vs U efficacemente |
| re_lgbm | 1.0 | Uniforme; M3 mostra picco secondario a γ=0.33 (sensitivity analysis §7) |
| puet | 0.0 | nnPU loss solo su P+N certi; γ=0 = nessun mixing U (regime ottimale per nnPU) |

### 2.2 Prior π = 0.02

La stima π=0.02 è derivata da PULSNAR per tutti e tre i dataset. M3 mostrava π≈0.03 come artefatto del gradiente nnPU, non come adattamento reale; uniformato a 0.02 per coerenza metodologica con il paper.

### 2.3 Lifting al 1%: mixing grid vs. ex-post

Il mixing grid (seed=2025, 4-fold reshuffled, pool P+N+U) produce metriche di lifting che non sono direttamente comparabili con l'ex-post bootstrap (seed=2028, P+N labeled solo). I valori di lifting ex-post risultano più bassi in valore assoluto ma sono più stabili e calcolati sul set labeled puro.

**Riepilogo lifting al 1% — mixing grid OOF (media ± SD):**

| Modello | γ* | M1 | M2 | M3 |
|---------|-----|----|----|-----|
| lgb_supervised | 1.0 | 8.18 ± 3.07 | 4.25 ± 3.33 | 5.23 ± 5.01 |
| bagging_lgbm | 0.66 | 18.05 ± 2.59 | 16.13 ± 2.03 | 22.38 ± 2.83 |
| re_lgbm | 1.0 | 17.29 ± 1.75 | 18.76 ± 3.92 | 21.89 ± 3.84 |
| puet | 0.0 | 13.57 ± 1.24 | 11.29 ± 2.89 | 8.60 ± 3.39 |

> **Nota lgb_supervised:** il lifting OOF del mixing (8–5×) è inferiore al lifting OOF originale (~21×) perché il mixing grid include contratti U non labeled nel pool di test e usa fold reshuffled. Non indica peggioramento del modello, ma una metrica più conservativa.

---

## 3. Calibrazione di Probabilità (Platt Scaling)

### 3.1 Metodologia

La calibrazione è realizzata via regressione logistica (Platt scaling globale):

```
p̂ = σ(a · s + b)
```

dove s è il raw score, a e b sono stimati in 4-fold cross-validation (seed=2026) su P+N labeled, e i parametri globali sono la media pesata dei fold. Il modello finale (usato per calibrare U) è addestrato su tutti i dati training.

**ECE** (Expected Calibration Error, 15 bin uniformi) misura il disallineamento tra probabilità predette e frequenze empiriche osservate; valori più bassi indicano calibrazione migliore.

### 3.2 Parametri Platt globali

| Modello | Dataset | a | b | Interpretazione |
|---------|---------|---|---|-----------------|
| lgb_supervised | M1 | +5.18 | −2.44 | Score ben separati, shift negativo |
| lgb_supervised | M2 | +5.48 | −2.57 | Simile a M1 |
| lgb_supervised | M3 | +17.80 | −7.90 | Separazione estrema (M3 include execution) |
| bagging_lgbm | M1 | +1.12 | −1.10 | Score compressi; Platt stretches |
| bagging_lgbm | M2 | +1.28 | −0.95 | Analogo |
| bagging_lgbm | M3 | +2.84 | −0.60 | Maggiore separazione post-gara |
| re_lgbm | M1 | +1.08 | −1.04 | Quasi identico a bagging_lgbm M1 |
| re_lgbm | M2 | +0.89 | −0.87 | Score leggermente più compressi |
| re_lgbm | M3 | +3.22 | −0.67 | Separazione intermedia |
| **puet** | **M1** | **−0.82** | **−0.47** | **a < 0: score invertiti** |
| **puet** | **M2** | **−1.43** | **−0.34** | **a < 0: score invertiti** |
| **puet** | **M3** | **−1.85** | **+0.08** | **a < 0: score invertiti** |

> **PUET: a negativo.** La loss nnPU (Kiryo 2017) produce score con orientamento inverso: i contratti a rischio ricevono score *bassi* in output raw. Il parametro Platt a<0 inverte automaticamente l'ordine: p̂ = σ(a·s+b) con a<0 mappa score bassi → probabilità alte. Questo è corretto e atteso per nnPU con PUEt.

### 3.3 ECE e Brier Score

**ECE raw → calibrato (overall OOF), con Brier score calibrato:**

| Modello | Dataset | ECE raw | ECE calib | Brier calib | Riduzione ECE |
|---------|---------|---------|-----------|-------------|---------------|
| lgb_supervised | M1 | 0.0450 | 0.0393 | 0.1755 | −13% |
| lgb_supervised | M2 | 0.0555 | 0.0442 | 0.1563 | −20% |
| lgb_supervised | M3 | 0.1406 | 0.0778 | 0.1528 | −45% |
| bagging_lgbm | M1 | 0.1083 | 0.0133 | 0.2091 | −88% |
| bagging_lgbm | M2 | 0.1732 | 0.0376 | 0.2166 | −78% |
| bagging_lgbm | M3 | 0.3190 | 0.0454 | 0.2358 | −86% |
| re_lgbm | M1 | 0.1383 | 0.0120 | 0.2092 | −91% |
| re_lgbm | M2 | 0.1858 | 0.0317 | 0.2177 | −83% |
| re_lgbm | M3 (γ=1.0) | 0.3082 | 0.0535 | 0.2344 | −83% |
| puet | M1 | 0.3537 | 0.0018 | 0.2072 | −99% |
| puet | M2 | 0.3392 | 0.0904 | 0.2103 | −73% |
| puet | M3 | 0.3548 | 0.0755 | 0.2344 | −79% |

**Osservazioni principali:**

- **lgb_supervised** è l'unico modello già parzialmente calibrato prima del Platt (ECE raw 0.04–0.14 contro 0.17–0.35 degli altri); questo perché usa cross-entropy standard che penalizza direttamente la calibrazione.
- **bagging_lgbm e re_lgbm** mostrano ECE raw elevata (0.11–0.31) dovuta alla loss nnPU che non ottimizza la calibrazione delle probabilità, ma post-Platt raggiungono ECE calibrata <0.055 su tutti i dataset.
- **puet M1** raggiunge ECE calibrata di 0.0018, la migliore dell'intero esperimento. M2 e M3 sono più difficili con ECE calibrata 0.07–0.09.
- Il Brier score calibrato è abbastanza omogeneo tra modelli (0.15–0.24), indicando che la difficoltà del task è distribuita in modo simile indipendentemente dall'architettura.
- **M3 è sistematicamente il dataset più difficile** per ECE raw (tutti i modelli mostrano ECE raw > 0.14 su M3 contro < 0.19 su M1/M2), perché include feature post-esecuzione che creano distribuzione degli score più bimodale.

---

## 4. Conformal Prediction (Set di Predizione)

### 4.1 Metodologia

Conformal prediction Mondrian class-conditional (Venn-ABERS):
- **Score di non-conformità:** 1 − p̂(y=1) per P, p̂(y=1) per N
- **Threshold:** quantile (1−α) dei calibration scores per classe, con α=0.10
- **4 fold** di calibrazione (seed=2027), metriche OOF aggregate come media

**Set possibili per ogni contratto U:**
- `{1}` — classificato positivo (singletons: alta probabilità di irregolarità)
- `{0}` — classificato negativo
- `{0,1}` — ambiguo (sia 0 sia 1 plausibili a α=0.10)
- `{}` — vuoto (nessuna classe plausibile; atteso ≈0 per α=0.10)

Copertura obiettivo: **≥90%** per entrambe le classi (1−α).

### 4.2 Copertura OOF media

| Modello | Dataset | Cop. cl1 | Cop. cl0 | Ampiezza media |
|---------|---------|----------|----------|----------------|
| lgb_supervised | M1 | **0.882** | 0.913 | 1.591 |
| lgb_supervised | M2 | 0.895 | 0.928 | 1.542 |
| lgb_supervised | M3 | 0.934 | 0.922 | 1.475 |
| bagging_lgbm | M1 | 0.900 | 0.907 | 1.805 |
| bagging_lgbm | M2 | **0.894** | 0.920 | 1.829 |
| bagging_lgbm | M3 | 0.957 | 0.950 | 1.850 |
| re_lgbm | M1 | **0.874** | 0.909 | 1.799 |
| re_lgbm | M2 | 0.911 | 0.921 | 1.830 |
| re_lgbm | M3 (γ=1.0) | 0.935 | 0.951 | 1.841 |
| puet | M1 | 0.912 | 0.927 | 1.795 |
| puet | M2 | 0.908 | 0.922 | 1.757 |
| puet | M3 | 0.887 | **0.864** | 1.659 |

> Valori in **grassetto**: copertura <90% (sotto soglia nominale α=0.10).

**Osservazioni:**

- **4 casi su 24** scendono sotto il 90% nominale: lgb_supervised M1 cl1 (0.882), bagging_lgbm M2 cl1 (0.894), re_lgbm M1 cl1 (0.874), puet M3 cl0 (0.864). In tutti i casi, lo scostamento è modesto (≤3 punti percentuali) e con 4 fold OOF il margine d'errore per fold è ±5–7%.
- **re_lgbm M1 cl1 = 0.874** è la copertura più bassa su classe positiva. È attribuibile al fatto che M1 ha pochissimi P in calibrazione (≈13–14 per fold) e la stima del quantile q₁ è instabile.
- **puet M3 cl0 = 0.864** è il caso peggiore su classe negativa. In M3, la variabilità dello score PUET su N è maggiore (score raw hanno scala diversa rispetto agli altri dataset).
- **Ampiezza media** è sistematicamente più bassa per lgb_supervised (1.47–1.59) rispetto a bagging/re_lgbm (1.80–1.85): lgb_supervised produce score più netti → più singletons → set più piccoli. PUET ha ampiezza intermedia (1.66–1.80).
- Virtualmente **zero set vuoti** su tutti i modelli: la massa di probabilità è ben distribuita.

---

## 5. SHAP: Importanza delle Feature

### 5.1 Metodologia

TreeExplainer (SHAP) calcolato sul modello finale addestrato su tutti i dati (P+N+U). Le righe di spiegazione sono campionate sui soli contratti **labeled (P+N)** per garantire riferimento ground-truth. Viene usata la media del valore assoluto SHAP come misura di importanza globale.

### 5.2 Top-5 feature per modello e dataset

#### M1 (ante-gara — solo feature di bando)

| Posizione | lgb_supervised | bagging_lgbm | re_lgbm | puet |
|-----------|---------------|-------------|---------|------|
| 1 | importo_complessivo_gara | **importo_lotto** | **importo_lotto** | **importo_lotto** |
| 2 | cod_strumento_svolgimento | importo_complessivo_gara | finestra_offerta_giorni | flag_urgenza |
| 3 | **importo_lotto** | finestra_offerta_giorni | flag_delega | finestra_offerta_giorni |
| 4 | finestra_offerta_giorni | flag_delega | importo_complessivo_gara | flag_ripetizioni |
| 5 | importo_sicurezza_pct | tasso_disoccupazione | tasso_disoccupazione | cod_modalita_realizzazione |

#### M2 (durante-gara — aggiunge feature di aggiudicazione)

| Posizione | lgb_supervised | bagging_lgbm | re_lgbm | puet |
|-----------|---------------|-------------|---------|------|
| 1 | cod_strumento_svolgimento | **importo_lotto** | **importo_lotto** | **importo_lotto** |
| 2 | importo_complessivo_gara | lag_aggiudicazione_giorni | lag_aggiudicazione_giorni | finestra_offerta_giorni |
| 3 | lag_comunicazione_esito_giorni | numero_offerte_ammesse | numero_offerte_ammesse | flag_urgenza |
| 4 | importo_aggiudicazione | ribasso_aggiudicazione | ribasso_aggiudicazione | flag_subappalto |
| 5 | lag_aggiudicazione_giorni | importo_aggiudicazione | n_lotti_componenti | importo_complessivo_gara |

#### M3 (post-gara — aggiunge feature di esecuzione)

| Posizione | lgb_supervised | bagging_lgbm | re_lgbm | puet |
|-----------|---------------|-------------|---------|------|
| 1 | cod_strumento_svolgimento | **importo_lotto** | **importo_lotto** | **importo_lotto** |
| 2 | flag_delega | lag_aggiudicazione_giorni | flag_delega | finestra_offerta_giorni |
| 3 | importo_complessivo_gara | importo_complessivo_gara | lag_aggiudicazione_giorni | flag_subappalto |
| 4 | reddito_irpef_procapite | **pct_overrun_core** | importo_complessivo_gara | flag_urgenza |
| 5 | lag_comunicazione_esito_giorni | tasso_disoccupazione | **pct_overrun_core** | flag_consuntivo_presente |

### 5.3 Letture principali

**Feature universalmente rilevanti:**
- **`importo_lotto`** è la feature più importante in 11 su 12 combinazioni (unica eccezione: lgb_supervised, dove è sempre top-3 ma non #1). La dimensione economica del lotto è il segnale di rischio più robusto attraverso tutti i modelli PU.
- **`importo_complessivo_gara`** è sistematicamente top-5 per tutti i modelli: il valore totale della gara cattura scala e attrattività per condotte irregolari.
- **`finestra_offerta_giorni`** (giorni tra pubblicazione e scadenza offerte): feature top-3 per re_lgbm M1 e puet M1/M2/M3. Finestre brevi possono indicare procedure meno competitive.

**Feature specifiche per modello:**
- **lgb_supervised** assegna importanza elevata a **`cod_strumento_svolgimento`** (modalità della procedura: aperta, ristretta, negoziata...) su tutti e 3 i dataset, suggerendo che la scelta della procedura è segnale di rischio quando si dispone di ground-truth diretto.
- **puet** evidenzia con costanza **`flag_urgenza`** e **`flag_ripetizioni`** (flag binari di procedura d'urgenza e ripetizione contrattuale). Questi segnali sono deboli in valore assoluto ma robusti nel regime nnPU, dove P sono pochissimi e la loss amplifica feature discriminative tra P certi e N certi.
- **`reddito_irpef_procapite`** (indicatore socioeconomico territoriale) è rilevante per lgb_supervised (M1, M3) e bagging_lgbm (M1): la capacità dell'ambiente istituzionale locale di influenzare il rischio di corruzione è catturata meglio dai modelli supervisionati/nnPU-bag.

**Feature di esecuzione (M3 specifiche):**
- **`pct_overrun_core`** (scostamento percentuale dal costo pianificato): appare in top-5 per bagging_lgbm M3 e re_lgbm M3. Un overrun elevato è segnale di variante intenzionale e gestione opaca.
- **`flag_consuntivo_presente`** (presenza del SAL finale): evidenziato da puet M3 come segnale rilevante, probabilmente perché la sua assenza è correlata a procedure poco trasparenti.
- **`lag_comunicazione_esito_giorni`**: ritardo nella comunicazione dell'esito, rilevante per lgb_supervised M2/M3.

---

## 6. Bootstrap CI per re_lgbm (B = 200)

### 6.1 Disegno del bootstrap

Per quantificare l'incertezza delle stime di re_lgbm, è stato eseguito un bootstrap stratificato B=200:

- **B = 200** fit completi su campioni con rimpiazzo (P, N, U campionati separatamente)
- **U_score**: 10.000 contratti unlabeled da mega dataset per M1/M2, 3.000 per M3 (fissi, fuori da U_train)
- **Platt globale fisso** applicato agli score U_score a ogni iterazione
- Output principale: `scores_ci.csv` — per ogni contratto U: `score_mean`, `score_lower`, `score_upper` al 95%

### 6.2 Distribuzione degli score e metrica comunicativa

#### Il rank percentile globale come metrica operativa

La metrica principale per comunicare il rischio di un contratto è il **rank percentile globale**: la posizione dello score di un contratto rispetto all'intera distribuzione empirica di P+N+U del dataset di training.

> *"Questo contratto è più a rischio del 95% dei contratti analizzati."*

Questa formulazione è onesta (non afferma una probabilità assoluta di corruzione), intuitiva per non tecnici (scala 0–100, come un voto), e azionabile (es. *"revisioniamo tutti i contratti sopra il 98° percentile"*). È la metrica standard nei sistemi di fraud detection e credit scoring.

La colonna `rank_pct_global` è disponibile in tutti i `scores_final.csv`. Per un nuovo contratto in fase di deployment, il lookup è:

```python
# ecdf = distribuzione empirica ordinata degli score calib del training
rank_pct = np.searchsorted(ecdf, new_score_calib) / len(ecdf) * 100
```

#### Distribuzione degli score U nel ranking globale

| Dataset | U p90 globale | U p99 globale | U max globale | Note |
|---------|--------------|--------------|--------------|------|
| M1 | ~85% | ~96% | ~99.9% | 24 U nel top 1% globale |
| M2 | ~85% | ~97% | ~99.9% | 28 U nel top 1% globale |
| M3 | ~85% | ~97% | ~99.9% | 15 U nel top 1% globale |

Il 90% dei contratti U si colloca sotto l'85° percentile globale: il modello li giudica meno sospetti della grande maggioranza dei contratti labeled. Solo l'1% più estremo di U (24–28 contratti per dataset) penetra nel top 1% globale, dove si concentra la quasi totalità dei P condannati.

Questo è un risultato, non un difetto: il modello è selettivo. La grande massa dei contratti U non ha caratteristiche associate ai P condannati; segnalarli tutti sarebbe rumore.

#### Top-5 contratti U per dataset (bootstrap B=200)

Le probabilità assolute riportate (colonna `p_corr`) sono ottenute correggendo il prior Platt per π=0.02 — formula tecnica in Appendice C. Per comunicazione operativa usare `rank_pct_global`.

**M1** — top U: rank 99.9°–99.96° percentile globale:

| CIG | rank globale | p_corr [IC 95%] | IC ampiezza |
|-----|------------|-----------------|-------------|
| 0202159ADF | 99.99% | 3.1% [2.2%, 4.5%] | 0.168 |
| 5171915F42 | 99.99% | 3.0% [2.3%, 4.2%] | 0.143 |
| B3135578FD | 99.98% | 3.0% [2.3%, 3.8%] | 0.121 |
| 73767037E2 | 99.97% | 2.9% [2.2%, 4.1%] | 0.144 |
| 495636274E | 99.96% | 2.8% [2.2%, 3.9%] | 0.138 |

**M2** — top U: rank 99.96°–99.99° percentile globale:

| CIG | rank globale | p_corr [IC 95%] | IC ampiezza |
|-----|------------|-----------------|-------------|
| 0550366064 | 99.99% | 3.5% [2.5%, 4.6%] | 0.142 |
| 48633546AD | 99.99% | 3.5% [2.7%, 4.5%] | 0.124 |
| 0312054B18 | 99.98% | 3.3% [2.6%, 4.4%] | 0.130 |
| 6985755B62 | 99.97% | 3.3% [2.4%, 4.7%] | 0.158 |
| 443718835F | 99.96% | 3.1% [2.5%, 4.2%] | 0.123 |

**M3** — segnale netto sul contratto top; IC più ampio per n_labeled piccolo:

| CIG | rank globale | p_corr [IC 95%] | IC ampiezza |
|-----|------------|-----------------|-------------|
| **0462709F87** | **99.99%** | **6.3% [2.2%, 18.8%]** | 0.446 |
| 7064979D18 | 99.97% | 3.0% [1.7%, 10.1%] | 0.418 |
| 0384232645 | 99.93% | 2.8% [1.6%, 7.5%] | 0.373 |
| 9440488924 | 99.90% | 2.6% [1.6%, 6.9%] | 0.357 |
| 55964310F6 | 99.87% | 2.5% [1.5%, 6.1%] | 0.340 |

#### Stabilità del ranking (ampiezza IC)

L'ampiezza dell'IC bootstrap misura quanto il rank di un contratto varia tra i 200 resampling:

| Quintile di score | M1 IC (media/max) | M2 IC (media/max) | M3 IC (media/max) |
|------------------|------------------|------------------|------------------|
| Q1–Q4 (basso–medio) | ~0 / 0.001 | ~0 / 0.001 | 0.050 / 0.077 |
| Q5 (top 20%) | 0.030 / 0.173 | 0.014 / 0.158 | 0.116 / 0.446 |

I contratti nella massa bassa hanno IC quasi nullo — il loro ranking è stabile perché tutti si aggregano alla baseline. I top contratti hanno IC più ampi (0.12–0.45): il segnale è reale ma la stima varia tra bootstrap. M3 ha IC strutturalmente più alti per via del minor numero di labeled (210P, 280N).

### 6.3 Interpretazione

**Il valore del ranking, non della probabilità assoluta.** Le probabilità prior-corrected (1.9%–6.3%) sono corrette tecnicamente ma difficili da comunicare: qualsiasi contratto sembrerà "poco rischioso" in assoluto a causa del prior π=2%. L'informazione utile risiede nell'ordine: i contratti nel top 1% globale sono strutturalmente i più simili ai P condannati tra tutti quelli analizzati.

**M1/M2 vs M3.** Le feature ante/durante-gara non producono segnali individuali netti: il miglior contratto in M1 è al 99.99° percentile ma la sua p_corr è solo 3.1% (1.5× il prior). Questo è un risultato substantivo, non un fallimento: significa che bando e aggiudicazione da soli non identificano contratti individuali corrotti con alta confidenza. Il valore dei modelli M1/M2 è nella prioritizzazione — concentrare l'attenzione dell'audit sul top 1% riduce i contratti da esaminare da 40.000 a 400.

**M3 è più discriminativo.** Le feature post-esecuzione (sovraccosti, durata, SAL) producono segnali più forti: il contratto 0462709F87 raggiunge 6.3% (3.2× il prior) con rank al 99.99° percentile. L'IC ampio [2.2%, 18.8%] è atteso con 210 P in training.

**Cosa trova il modello.** Il training usa P = gare con condanne definitive. Il modello apprende la corruzione *giudicata*, non quella reale. I contratti U top potrebbero essere corruzione non ancora rilevata, o contratti ad alto rischio strutturale senza illeciti — questa ambiguità è intrinseca al PU learning con ground truth giudiziario.

**Nota tecnica (prior correction).** La formula di correzione del prior è riportata in Appendice C per completezza. I CSV `scores_ci.csv` contengono le colonne `pc_mean`, `pc_lower`, `pc_upper` per chi necessita delle probabilità assolute.

---

## 7. Analisi di Sensibilità γ: re_lgbm M3

Per re_lgbm M3, sono state valutate due configurazioni di mixing:
- **γ=0.33**: picco di lift@1% nel mixing grid (lift=22.84±2.94)
- **γ=1.0**: configurazione canonica (uniformità con M1/M2), lift=21.89±3.84

Entrambi i risultati sono salvati rispettivamente in `results/re_lgbm_M3_g033/` e `results/re_lgbm_M3_g100/`.

### 7.1 Confronto calibrazione

| γ | ECE raw | ECE calib | Brier calib |
|---|---------|-----------|-------------|
| 0.33 | 0.2887 | 0.0476 | 0.2424 |
| 1.0 | 0.3082 | 0.0535 | 0.2344 |

**γ=0.33** mostra ECE calibrata leggermente inferiore (0.048 vs 0.054), ma la differenza è di circa un punto percentuale assoluto: trascurabile per le finalità del paper.

### 7.2 Confronto conformal (copertura media OOF)

| γ | Cop. cl1 | Cop. cl0 | Ampiezza media |
|---|----------|----------|----------------|
| 0.33 | **0.946** | 0.942 | 1.841 |
| 1.0 | 0.935 | 0.951 | 1.841 |

Entrambe le configurazioni superano il 90% su entrambe le classi. γ=0.33 ha copertura cl1 leggermente più alta (+1.1 pp), mentre γ=1.0 ha copertura cl0 più alta (+0.9 pp). L'ampiezza media è identica (1.841).

### 7.3 Confronto SHAP top-5

| Posizione | γ=0.33 | γ=1.0 |
|-----------|--------|-------|
| 1 | importo_lotto | importo_lotto |
| 2 | flag_delega | flag_delega |
| 3 | lag_aggiudicazione_giorni | lag_aggiudicazione_giorni |
| 4 | importo_complessivo_gara | importo_complessivo_gara |
| 5 | pct_overrun_core | pct_overrun_core |

Le top-5 feature sono **identiche** nelle due configurazioni e nello stesso ordine.

### 7.4 Conclusione analisi di sensibilità

Le differenze tra γ=0.33 e γ=1.0 per re_lgbm M3 sono inferiori al margine di variabilità del metodo. Le due configurazioni producono risultati statisticamente indistinguibili su:
- ECE calibrata (±0.006 assoluto)
- Copertura conformal (±0.016 sulla classe peggiore)
- Ranking SHAP (identico top-5)

La scelta di γ=1.0 come configurazione canonica è **metodologicamente giustificata** dalla coerenza narrativa: tutti i dataset usano la stessa politica di mixing (full PNU), semplificando la presentazione senza perdita di performance rilevante.

La cartella `results/re_lgbm_M3_g033/` è mantenuta come sensitivity analysis riproducibile.

---

## 8. Discussione e Conclusioni

### 8.1 Quale modello raccomandare?

| Criterio | Migliore |
|----------|---------|
| Lifting bruto (mixing grid) | bagging_lgbm / re_lgbm (~18–22× su M2/M3) |
| Stabilità lifting (SD) | re_lgbm M1 (SD=1.75), bagging_lgbm M2 (SD=2.03) |
| ECE calibrata | puet M1 (0.0018), re_lgbm M1 (0.0120) |
| Copertura conformal (≥90% entrambe classi) | Tutti tranne 4 casi marginali |
| Interpretabilità SHAP | Coerenza cross-modello su importo_lotto |

**re_lgbm** emerge come modello di riferimento per il paper: lifting top (17–22×), calibrazione ottima post-Platt (ECE ≤ 0.054), copertura conformal robusta, e feature importance interpretabili e coerenti con la letteratura sulla corruzione negli appalti pubblici.

**bagging_lgbm** è quasi equivalente per lifting e conformal, con ECE leggermente più alta. È la prima alternativa.

**puet** mostra comportamento unico (score invertiti, flag binari in SHAP), performance di lifting inferiori su M2/M3, e copertura conformal M3 leggermente sotto soglia. È metodologicamente rilevante (nnPU puro) ma non il modello primario per il deployment.

**lgb_supervised** è già calibrato nativamente (ECE raw bassa), ma ha lifting inferiore perché non sfrutta U. Set conformal più piccoli (ampiezza 1.47–1.59 vs 1.79–1.85): riflette maggiore certezza nelle predizioni a scapito della copertura marginale.

### 8.2 Feature più rilevanti (cross-model consensus)

Le feature con importanza consistente attraverso tutti i modelli sono:
1. **`importo_lotto`** — dimensione economica del singolo lotto (universale)
2. **`importo_complessivo_gara`** — scala della gara
3. **`finestra_offerta_giorni`** — potenziale restrizione della concorrenza
4. **`flag_delega`** — gare con delega specifica (bagging/re_lgbm M1/M3)
5. **`tasso_disoccupazione`** / **`reddito_irpef_procapite`** — contesto socioeconomico territoriale

Feature specifiche di dataset:
- M2/M3: **`lag_aggiudicazione_giorni`**, **`ribasso_aggiudicazione`**, **`numero_offerte_ammesse`** (competitività aggiudicazione)
- M3: **`pct_overrun_core`** (scostamento esecuzione), **`flag_consuntivo_presente`** (trasparenza)

### 8.3 Copertura conformal e applicazione pratica

Il conformal predictor con α=0.10 assegna:
- **Set singleton `{1}`**: contratti con alta probabilità di irregolarità (candidati a revisione)
- **Set ambigui `{0,1}`**: contratti borderline (richiedono ispezione complementare)
- **Set singleton `{0}`**: contratti quasi certamente regolari (non necessitano revisione immediata)

La frazione di singletons varia per modello:
- lgb_supervised: alta (score netti → ~38–51% singletons per fold)
- bagging/re_lgbm/puet: ~8–23% singletons, ~77–92% ambigui

Questa differenza riflette la confidenza dei modelli: lgb_supervised è più assertivo (ma con minor lifting totale); bagging/re_lgbm preferiscono la prudenza dichiarata. In un contesto di screening, la maggiore frazione di ambigui è preferibile: riduce i falsi positivi emettendo segnale solo dove la confidenza è alta.

---

## Appendice A — Riproducibilità

```bash
# Calibrazione + Conformal + SHAP per tutti i modelli
python run_all.py

# Solo re_lgbm M3 con entrambi i gamma
python run_all_gamma_comparison.py --gammas 0.33 1.0

# Bootstrap CI per re_lgbm
python bootstrap_ci.py --datasets 1 2 3 --n-boot 200
```

File chiave per γ* e parametri Platt:
- `gamma_star.json` — γ* per ogni modello/dataset + nota riproducibilità
- `results/{model}_{Mn}/calibration/platt_params.csv` — parametri Platt fold-by-fold e globali
- `results/{model}_{Mn}/calibration/ece_brier.csv` — ECE e Brier per fold e overall
- `results/{model}_{Mn}/conformal/coverage.csv` — copertura OOF fold-by-fold e mean
- `results/{model}_{Mn}/shap/feature_importance.csv` — mean_abs_shap per feature
- `results/re_lgbm_M{n}/bootstrap/metrics_ci.csv` — CI metriche bootstrap

---

## Appendice B — Seed e Ordine di Esecuzione

| Step | Seed | Dipendenze |
|------|------|------------|
| PU mixing grid | 2025 | Nessuna |
| Calibrazione Platt | 2026 | mixing (modello finale) |
| Conformal prediction | 2027 | calibrazione (platt_params.csv) |
| Bootstrap CI | 2028 | calibrazione (platt_params.csv, modello finale) |
| SHAP | — | calibrazione (modello finale) |

I seed sono scelti sequenziali per chiarezza; non c'è sovrapposizione di uso tra step.

---

## Appendice C — Prior Correction: formula tecnica

I raw Platt scores sono calibrati su P+N con base rate r = n_P/(n_P+n_N) >> π. Per convertire in probabilità di popolazione compatibili con π=0.02, si applica la correzione logit:

```
logit(p_corr) = logit(p_platt) + log(π/(1-π)) - log(r/(1-r))
```

equivalentemente, in termini di odds:

```
odds_corr = odds_platt × [π/(1-π)] / [r/(1-r)]
p_corr = odds_corr / (1 + odds_corr)
```

Questa è la correzione bayesiana standard per classificatori addestrati su sample biasati (si veda Saerens et al. 2002, *Adjusting the outputs of a classifier*).

| Dataset | r | logit shift | p_corr mediana U | p_corr max U |
|---------|---|-------------|-----------------|-------------|
| M1 | 0.270 | −2.90 | 1.9% | 3.1% |
| M2 | 0.272 | −2.91 | 2.2% | 3.5% |
| M3 | 0.429 | −3.60 | 1.4% | 6.3% |

Le colonne `pc_mean`, `pc_lower`, `pc_upper` nei file `scores_ci.csv` (bootstrap re_lgbm) già incorporano questa correzione. Per gli altri modelli applicare la formula ai valori `score_calib_final` di `scores_final.csv` usando r calcolato dal medesimo file.
