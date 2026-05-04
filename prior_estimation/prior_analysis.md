# Analisi della Prior di Classe — PU Learning su Appalti Pubblici
*Questo file consolida e sostituisce `results.md` e `note_prior.md`.*
*Run di produzione del 2026-04-12 (M1, M2) e 2026-05-02 (M3, rieseguito con script unificato: 30 run PULSNAR, mediana come punto stima). File di dettaglio per modello: `prior_estimates_M{1,2,3}.txt`.*

---

## 1. Cosa stima ogni metodo — versione corretta

| Metodo | Quantità stimata | Simbolo | Definizione | Assunzione |
|---|---|---|---|---|
| **Elkan-Noto** | Label frequency | **c** | P(S=1 \| Y=1) — prob. di essere etichettato dato che sei positivo | SCAR |
| **Blanchard** | Class prior | **π** | P(Y=1) — proporzione di positivi nel pool degli unlabeled | SCAR (o weaker) |
| **KM2 (DEDPUL)** | Mixing proportion | **π** | P(Y=1 \| S=0) ≈ π·(1−c)/(1−π·c) ≈ π | SCAR |
| **PULSNAR** | Class prior | **π** | P(Y=1 \| S=0) ≈ π, modellato con GMM+XGBoost | SAR (SNAR) |

**Correzione rispetto alle note precedenti:** Blanchard et al. (2010 JMLR) stimano la
proporzione di mescolanza di f₊ in f_U — ossia π, non c. Questo è confermato sia dal
paper originale che dal framework di Ramaswamy et al. (2016) che usa lo stesso principio.
Solo Elkan-Noto stima c.

**Relazioni fondamentali:**
```
π = n_pos / (N_tot × c)             ← ricava π da EN/c
c = n_pos / (N_tot × π)             ← ricava c da Blanchard/KM2/PULSNAR
P(Y=1|S=0) = π(1−c)/(1−πc) ≈ π    ← approssimazione valida per πc ≪ 1
```

---

## 2. Risultati della produzione run

### 2a. Dati dei modelli

| Modello | Positivi | Negativi | Unlabeled | Totale | Runtime |
|---|---|---|---|---|---|
| **M1** (ex-ante) | 781 | 1,817 | 9,466,197 | 9,468,795 | 27m 9s |
| **M2** (durante) | 496 | 1,045 | 3,467,282 | 3,469,880 | 52m 47s |
| **M3** (ex-post) | 210 | 280 | 1,128,743 | 1,129,233 | 15m 10s |

Config: EN/Blanchard max_unl=200,000 boot=200 · KM2 max_unl=1,500 runs=10 · PULSNAR max_unl=50,000 runs=10

### 2b. Stime grezze e cosa rappresentano

| Metodo | Stima | M1 | CI M1 | M2 | CI M2 | M3 | CI M3 |
|---|---|---|---|---|---|---|---|
| **Elkan-Noto** | **c** | 0.1166 | [0.1017, 0.1346] | 0.0701 | [0.0531, 0.0874] | 0.0568 | [0.0303, 0.0862] |
| **Blanchard** | **π** | 0.0633 | [0.0609, 0.0658] | 0.0863 | [0.0803, 0.0928] | 0.1084 | [0.0945, 0.1235] |
| **KM2** | **π** | 0.1886 | [0.1702, 0.2193] ⚠ | 0.2839 | [0.2977, 0.3236] ⚠ | 0.4108 | [0.3912, 0.4360] ⚠ |
| **PULSNAR** | **π** | 0.0222 | [0.0103, 0.0168] ⚠ | 0.0164 | [0.0098, 0.0260] | 0.0208 | [0.0025, 0.0316] |

⚠ = anomalia nel CI (vedi §4)

### 2c. Conversioni interne

Dall'unico metodo che stima c (EN), si ricava π. Dagli altri (Blanchard, PULSNAR),
si ricava c. Entrambe le direzioni sono utili per verificare la coerenza interna.

**π implicita da EN** (tramite π = n_pos / (N_tot × c_EN)):

| Modello | c_EN | π implicita da EN | Contratti corrotti impliciti |
|---|---|---|---|
| M1 | 0.1166 | **0.07%** | ~6,600 su 9.47M |
| M2 | 0.0701 | **0.32%** | ~11,100 su 3.47M |
| M3 | 0.0568 | **0.33%** | ~3,700 su 1.13M |

**c implicita da PULSNAR** (tramite c = n_pos / (N_tot × π_PULSNAR)):

| Modello | π_PULSNAR | c implicita da PULSNAR | Contratti corrotti impliciti |
|---|---|---|---|
| M1 | 0.0222 | **0.35%** | ~210,000 su 9.47M |
| M2 | 0.0164 | **1.37%** | ~56,900 su 3.47M |
| M3 | 0.0208 | **0.89%** | ~23,500 su 1.13M |

---

## 3. Confronto con la letteratura

| Fonte | Anno | π stimata | Definizione | Dataset |
|---|---|---|---|---|
| Conzo et al. (J. Public Econ.) | 2020 | ~1% | Infiltrazione mafiosa accertata | 68K contratti, comuni IT 2012–2017 |
| Decarolis & Giorgiantonio (BI QEF 544) | 2022 | **2%** convictions, **15%** investigated | Imprese con manager sotto indagine | Lavori stradali municipali |
| Coviello, Guglielmo, Spagnolo (NBER 28209) | 2020 | **2%** convictions, **17%** investigated | Dati AISI su imprese indagate | Edifici pubblici e strade |
| RAND Europe | 2016 | **~10%** | Probabilità bid rigging (screening) | EU-wide |

**Range credibile per π nell'intero dataset ANAC** (lavori + forniture + servizi):
- **Lower bound** (solo condanne): 1–2%
- **Stima centrale**: 2–5%
- **Upper bound** (bid rigging incluso): 5–15%

*Nota:* la letteratura si concentra sui lavori pubblici (settore a più alto rischio). Forniture e servizi hanno tassi più bassi → π media ANAC sarà inferiore ai valori sopra.

**Confronto diretto con i metodi:**

| Metodo | π stimata (range M1–M3) | vs. letteratura |
|---|---|---|
| EN (π implicita) | 0.07–0.33% | **Troppo bassa** — SAR gonfia c di 10–100×, comprimendo π implicita |
| **Blanchard** | 6.3–10.8% | **Al limite superiore** — entro range RAND (bid rigging); sopra range convictions |
| KM2 | 18.9–41.1% | **Implausibile** — ratio pos/unl distorto (33–52% vs 0.008% reale); scartare |
| **PULSNAR** | 1.6–2.2% | **Nel range centrale** — coerente con letteratura convictions (1–2%) |

---

## 4. Valutazione critica di ogni metodo

### Elkan-Noto
**Cosa stima:** c = P(S=1|Y=1).

**Problema — violazione SCAR (SAR):** i 781 positivi non sono un campione casuale
dei contratti corrotti. Sono filtrati tre volte: condanna definitiva → citazione CIG →
match ANAC. Contratti ad alto valore, edilizia, sanità, aree ad alta criminalità
organizzata sono sistematicamente sovrarappresentati. EN interpreta questa selettività
come "segnale forte" e sovrastima c. La c implicita di 0.07% per π (troppo bassa)
ne è la conferma: EN dice "su 100 contratti corrotti, 11.7 vengono condannati" → implica
solo 6.600 corrotti su 9.4M contratti, completamente implausibile.

**Uso corretto:** EN è utile come stima di c per la **calibrazione degli score**
(P(Y=1|x) = g(x)/c, dove c = c_EN). Non va usato per derivare π.

**Trend M1→M2→M3:** c_EN decresce (0.117 → 0.070 → 0.057). Con più feature
discriminative (durante, ex-post), LightGBM separa meglio P da U, abbassando lo
score medio dei positivi → c_EN scende. Questo è un artefatto del miglioramento del
classificatore, non un cambiamento della label frequency vera.

### Blanchard Quantili
**Cosa stima:** π = P(Y=1), come infimum del rapporto F_U(s≥t)/F_P(s≥t) al variare
della soglia t sugli score. Fornisce un lower bound di π sotto SCAR.

**Trend M1→M2→M3:** π_Blanchard cresce (0.063 → 0.086 → 0.108). Questo è anomalo:
ci si aspetterebbe che π sia stabile o leggermente variabile tra modelli che descrivono
lo stesso fenomeno. La crescita riflette probabilmente che con più feature discriminative
(M3 ha feature ex-post) il LightGBM separa meglio P da U in termini di score, allargando
il gap tra le CDF e alzando il minimo del rapporto. Non necessariamente significa che
"più contratti corrotti" nei dati ex-post.

**Attendibilità:** CI molto stretti (±0.2–0.7pp) indicano stabilità interna del
bootstrap sugli score. La stima di π ≈ 6–11% è nel range della letteratura (bid rigging
+ indagini), ma al limite superiore. Come lower bound teorico tende a essere conservativo.

**Anomalia M2/M3:** Blanchard > EN per M2 e M3. Sotto SCAR vale EN ≥ π ≥ Blanchard.
Il sorpasso segnala violazione SAR profonda: EN è stato compresso dall'apprendimento
LightGBM al punto da cadere sotto il lower bound di Blanchard. Questo non invalida
Blanchard (che non richiede EN > Blanchard), ma conferma che EN non stima c in modo
affidabile per M2/M3.

**Bug sensitivity analysis:** range sempre 0.0000. Blanchard è rank-invariante
(usa soglie sul ranking degli score): scalare i punteggi non cambia il ranking → la
sensitivity analysis attuale non è informativa. Il flag RUN_SENSITIVITY può restare
ma il risultato non va interpretato.

### KM2 (DEDPUL/KMPE)
**Cosa stima:** proporzione di f_+ nell'unlabeled pool tramite kernel mean embeddings
(Ramaswamy et al. 2016, ICML). Corrisponde a P(Y=1|S=0) ≈ π.

**Run corretti (argomenti km(X_unl, X_pos), 2026-04-13):** M1=0.1886 [0.1702, 0.2193],
M2=0.2839 [0.2977, 0.3236] ⚠ (punto stima < CI lower), M3=0.4108 [0.3912, 0.4360].
I CI non sono più degeneri ma i valori sono implausibili: π=19–41% implicherebbe
milioni di contratti corrotti. Il trend crescente M1→M3 è anch'esso sospetto.
Nota aggiuntiva: M2 ha solo 496 positivi nel preprocessed (vs 781 nel nativo) —
285 positivi etichettati eliminati perché le feature "durante" sono NaN.

**Problema strutturale confermato — ratio distorto:** con 1500 unlabeled e
781/496/210 positivi, il ratio nel subsample è 33–52%, contro ≈ 0.008–0.02% reale.
Il fix degli argomenti ha corretto il bug ma non il problema di fondo. Aumentare
il subsample a valori significativi è proibitivo: O(N³).

**Conclusione:** KM2 non produce stime affidabili in questo setup. Scartare.

### PULSNAR
**Cosa stima:** α = P(Y=1|S=0) ≈ π tramite GMM sulle feature preprocessed + XGBoost.
Progettato per SAR (SNAR), non richiede SCAR — vantaggio teorico significativo.

**M1 — bug CI:** α_hat=0.0222 (seed=42) > CI upper (0.0168). Il seed=42 non è incluso
nel campionamento CI (CI usa semi 43, 1043, ...). Non è un bug del codice ma del
design: il punto stima non è un campione dal CI. Da notare, non da usare come
valore operativo senza cautela.

**M3 — CI molto ampio:** run CI: 0.0023→0.0360 (fattore 15×). Con 210 positivi
su 26 feature il GMM interno è instabile. Il numero di cluster varia tra 2 e 3 a
seconda del seed.

**M2 — caso migliore:** α_hat=0.0164 ∈ [0.0098, 0.0260]. Con 781 positivi il
GMM è più stabile. Il CI è ragionevole (±8pp).

**Stima di π:** 1.6–2.2% — il range più coerente con la letteratura tra tutti i metodi.

---

## 5. Sintesi per la stima operativa

### Cosa usare per π (input per nnPU/uPU downstream)

Il confronto dei metodi indica:

| | π stimata | Affidabilità | Coerenza letteratura |
|---|---|---|---|
| Blanchard | 6.3–10.8% | Media (CI stabili ma SAR può distorcere) | Limite superiore (bid rigging range) |
| PULSNAR | 1.6–2.2% | Media (CI ampi in M3) | **Range centrale** (conviction-based) |
| KM2 | 18.9–41.1% | Bassa (ratio distorto, punto stima M2 < CI lower) | Implausibile — scartare |
| EN (implicita) | 0.07–0.33% | Bassa (SAR gonfia c) | Troppo bassa |

**Raccomandazione:** usare PULSNAR come stima puntuale di riferimento (π ≈ 0.02)
e coprire il range con sensitivity analysis.

**Griglia sensitivity consigliata:** `π ∈ {0.005, 0.01, 0.02, 0.03, 0.05}`
- 0.005 (0.5%) — lower bound per dataset ANAC misto (bassa incidenza forniture/servizi)
- 0.01 (1%) — lower bound letteratura convictions
- 0.02 (2%) — stima centrale PULSNAR e letteratura
- 0.03 (3%) — stima centrale superiore
- 0.05 (5%) — limite superiore convictions (coerente con RAND per lavori pubblici)

*Questi 5 valori coprono tutto il range credibile; la sensitivity nel PU downstream
dirà quali π producono modelli stabili vs instabili.*

### Cosa usare per c (calibrazione score EN)

Se si vuole calibrare gli score post-training con P(Y=1|x) = g(x)/c, usare la c di
Blanchard ricavata inversamente da π_PULSNAR (più stabile):

| Modello | π_PULSNAR | c implicita da PULSNAR |
|---|---|---|
| M1 | 0.022 | **0.35%** |
| M2 | 0.016 | **1.37%** |
| M3 | 0.021 | **0.89%** |

In alternativa: c da EN come upper bound (stima aggressiva), c da inversione PULSNAR
come stima centrale, letteratura come prior esterna.

---

## 6. Problemi tecnici identificati

| Problema | Dove | Impatto | Azione |
|---|---|---|---|
| KM2 argomenti invertiti | `km(X_pos, X_unl)` invece di `km(X_unl, X_pos)` | Stima proporzione sbagliata (f_U in f_+ invece di f_+ in f_U) | Bug — irrilevante perché KM2 già scartato per CI degenere |
| KM2 valori implausibili | M1=18.9%, M2=28.4%, M3=41.1% dopo fix argomenti | Ratio pos/unl nel subsample 33–52% vs 0.01% reale | Strutturale — scartare KM2 |
| KM2 point ≠ CI lower | M2 (point=0.2839 < CI lower=0.2977) | seed=42 non incluso nel campionamento CI | Strutturale |
| PULSNAR α_hat > CI upper | M1 | Seed=42 non incluso nel campionamento CI | Minore — annotato |
| Blanchard > EN | M2, M3 | Inversione anomala — SAR severo | Strutturale (dataset) |
| Sensitivity analysis range=0 | Tutti | Non informativa (rank-invariante) | Bug — disabilitare o riscrivere |
| EN classifica Blanchard come "stima c" | note_prior.md (vecchio) | Interpretazione sbagliata delle stime | **Corretto in questo file** |

---

## 7. Conviene fare re-run con numerosità diverse?

**No, per i seguenti motivi:**

| Metodo | Problema | Re-run utile? |
|---|---|---|
| EN/Blanchard | Già a 200K unlabeled (20× più del necessario per stabilità) — la fonte di bias è SAR, non la numerosità | No |
| KM2 | Degeneracy algoritmica a 1,500 unlabeled. A 50K: O(N³) ≈ 50K³/10⁹ ≈ giorni. A 10K: ancora ore. | No |
| PULSNAR | CI ampio in M3 (210 positivi) è un limite strutturale del dataset, non del subsample | No per M1/M2, discutibile per M3 |

**Alternativa praticabile per KM2 (se si vuole un secondo estimatore di π robusto):**
applicare KM2 sugli score 1D prodotti da LightGBM, non sulle feature grezze. Con score
monodimensionali il kernel QP è O(N) anziché O(N³) e si può usare l'intero dataset. Non
è implementato nello script attuale ma è la strada giusta se si vuole un confronto con
PULSNAR su π.

---

## 8. Cosa passare al modello PU downstream

| Parametro | Valore | Fonte |
|---|---|---|
| **π** (per nnPU, uPU, LightGBM+nnPU) | **sensitivity: {0.005, 0.01, 0.02, 0.03, 0.05}** — punto centrale 0.02 | PULSNAR + letteratura |
| **c** (per calibrazione EN degli score) | M1=0.35%, M2=1.37%, M3=0.89% | Inversione PULSNAR |
| **c alternativo** (conservative upper) | M1=11.7%, M2=7.0%, M3=5.7% | Elkan-Noto (SAR-biased) |

---

## 9. Riferimenti

- Elkan & Noto (2008). *Learning classifiers from only positive and unlabeled data.* KDD 2008.
- Blanchard, Lee & Scott (2010). *Semi-supervised novelty detection.* JMLR 11:2973–3009.
- Ramaswamy, Scott & Tewari (2016). *Mixture proportion estimation via kernel mean embeddings.* ICML 2016.
- Kumar & Lambert (2023). *PULSNAR: PU Learning to Sample Not All Rejected.* arXiv:2303.08269 / PeerJ CS 2024.
- Decarolis & Giorgiantonio (2022). *Corruption red flags in public procurement.* Bank of Italy QEF No. 544.
- Coviello, Guglielmo, Spagnolo (2020). *Rules, Discretion, and Corruption in Procurement.* NBER WP 28209.
- Conzo et al. (2020). *Mafia infiltration.* Journal of Public Economics. https://doi.org/10.1016/j.jpubeco.2020.104267
- RAND Europe (2016). *The Cost of Non-Europe in the area of Organised Crime and Corruption.*
