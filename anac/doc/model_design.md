# Design dei 3 Modelli PU Learning
**Data**: 2026-04-08  
**Parquet sorgente**: `output/parquet/bando_cig_all.parquet` (66 col × 9,468,795 righe)

---

## Principi guida

### Definizione temporale
| Modello | Finestra temporale | Domanda di ricerca |
|---|---|---|
| **M1 — Ex ante** | Al momento della pubblicazione del bando | Qual è il rischio di corruzione prima che chiunque partecipi? |
| **M2 — Durante** | Dopo l'aggiudicazione, prima dell'esecuzione | Cosa aggiunge sapere chi ha vinto e come? |
| **M3 — Ex post** | Dopo la firma del contratto e l'esecuzione | Quanto segnale emerge dall'intera vita dell'appalto? |

### Architettura nested
Ogni modello è un **superset** del precedente:  
M1 ⊂ M2 ⊂ M3 (feature set)

### Native NA vs non-native
- **Modelli nativi** (XGBoost, LightGBM): stesso row filtering per fase, ma **senza discretizzazione** delle feature continue. Il parquet grezzo `bando_cig_all.parquet` è la fonte.
- **Modelli non-nativi** (logistica, SVM): usano il parquet discretizzato `bando_cig_model.parquet` (da costruire). Stesso row filtering.
- **Importante**: "aggiudicazioni=NA" NON è segnale informativo per i modelli nativi — è censura temporale (contratto non ancora aggiudicato), non pattern di corruzione. Il row filtering per fase risolve questo problema.

---

## Row Selection

### M1 — Ex ante
**Tutti i CIG del parquet**: 9,468,795 righe

Tutti i bandi sono comparabili al momento della pubblicazione, indipendentemente da come è andata a finire. Include anche i 1,014 labeled con `esito=None` (label da OpenGA, non da ANAC aggiudicazioni).

### M2 — Durante
**Filtro**: `esito = 'AGGIUDICATA'` → **3,690,097 CIG**

Richiede che il bando abbia un vincitore registrato in ANAC: solo così le feature di aggiudicazione sono disponibili e confrontabili tra labeled e unlabeled.

- Labeled disponibili: **1,541** (496 COND / 1,045 SCAG)
- Persi rispetto a M1: 1,057 labeled (285 COND / 772 SCAG) con `esito=None`
- **Motivazione esclusione**: includere i 1,014 labeled con aggiudicazioni=NA creerebbe un'asimmetria MNAR — i condannati avrebbero NA strutturale mentre l'unlabeled avrebbe NA lifecycle. Il modello imparebbe il tipo di NA, non il rischio di corruzione.

### M3 — Ex post (F4)
**Filtro**: `esito = 'AGGIUDICATA'` AND (`lag_stipula non-null` OR `n_sal > 0` OR `n_varianti > 0` OR `n_sospensioni > 0`) → **1,129,233 CIG**

Richiede che il contratto sia stato firmato (stipula = obbligo legale di esecuzione) **oppure** che abbia dati di monitoraggio (evidenza diretta di esecuzione avvenuta). È il filtro più principiato: la firma è il punto di non ritorno amministrativo.

- Labeled disponibili: **490** (210 COND / 280 SCAG)
- Shrinkage accettabile: il rapporto COND/SCAG rimane bilanciato (~43/57)
- I 1,014 labeled con esito=None non passano F4 in nessun caso (hanno al massimo 12 stipule e 6 SAL) → impatto trascurabile (+13 se inclusi)

---

## Feature Assignment

### Ciclo di vita dell'appalto

```
PROGETTAZIONE    PUBBLICAZIONE    OFFERTE    AGGIUDICAZIONE    STIPULA    ESECUZIONE    COLLAUDO
      │               │              │              │              │            │             │
      ▼               ▼              ▼              ▼              ▼            ▼             ▼
──────●───────────────●──────────────●──────────────●─────────────●────────────●─────────────●──
      
      │← QE BASE_ASTA │←─────────── M1 (ante) ─────►│
                                                      │←────── M2 (durante) ────►│
                                                                                  │←── M3 ────►│
```

---

### M1 — Ex ante (24 feature)
*Informazioni note al momento della pubblicazione del bando*

| Feature | Fonte | Note |
|---|---|---|
| `importo_complessivo_gara` | BANDO CIG | Valore totale della gara |
| `importo_lotto` | BANDO CIG | Valore del lotto |
| `n_lotti_componenti` | BANDO CIG | Numero lotti (14% null → imputa 1) |
| `oggetto_principale_contratto` | BANDO CIG | LAVORI/SERVIZI/FORNITURE |
| `modalita_realizzazione_macro` | BANDO CIG | 6 classi: APPALTO/ECONOMIA/ACCORDO_QUADRO/CONCESSIONE/PPP/ALTRO (aggr. da 23 codici ANAC) |
| `sezione_regionale` | BANDO CIG | Sezione ANAC |
| `cod_strumento_svolgimento` | BANDO CIG | Strumento (76.8% null → categoria) |
| `flag_urgenza` | BANDO CIG | Flag urgenza |
| `cod_motivo_urgenza` | BANDO CIG | Motivo urgenza (89.5% null → categoria) |
| `flag_delega` | BANDO CIG | Contratto delegato (91.7% null → 3ª cat) |
| `finestra_offerta_giorni` | BANDO CIG | Giorni dalla pubblicazione alla scadenza offerte |
| `lag_perfezionamento_giorni` | BANDO CIG | Giorni al perfezionamento del bando |
| `flag_accordo_quadro` | BANDO CIG | Flag accordo quadro |
| `flag_ripetizioni` | BANDO CIG | Flag ripetizioni previste |
| `settore_speciale` | BANDO CIG | Flag settore speciale |
| `flag_appalto_riservato` | BANDO CIG | Flag appalto riservato |
| `tipo_scelta_4cls` | BANDO CIG + lookup | APERTA/RISTRETTA/NEGOZIATA/AFFIDAMENTO_DIRETTO |
| `cpv_macro_categoria` | BANDO CIG + CPV | LAVORI/SERVIZI/FORNITURE/IT/ING_PROF/SANITA |
| `importo_sicurezza_pct` | BANDO CIG / QE BASE_ASTA | Quota sicurezza / importo lotto (73.9% null) |
| `pct_riserva_base` | QE BASE_ASTA | Somme a disposizione / sum_core (72% null). QE BASE_ASTA compilato in fase di progettazione → ante |
| `natura_giuridica_SA` | STAZIONE APPALTANTE | Tipo ente appaltante |
| `tasso_disoccupazione` | ISTAT (t-1) | Provinciale, min(anno_pub−1, 2024) (35% null) |
| `reddito_irpef_procapite` | MEF (t-1) | Provinciale, min(anno_pub−1, 2024) (35% null) |
| `tasso_omicidi_100k` | ISTAT BES (t-1) | Provinciale, min(anno_pub−1, 2024) (35% null) |

> **Ambiguità risolta**: `pct_riserva_base` e `importo_sicurezza_pct` classificati come ANTE perché il loro contenuto informativo (pianificazione finanziaria di progetto, quota sicurezza) è stabilito prima della pubblicazione del bando, anche se ANAC può riceverli in momenti diversi.

> **Non-feature (metadata)**: `anno_pubblicazione` e `regione` restano in `bando_cig_all.parquet` per analisi descrittive ma sono escluse da M1_FEAT (`COLS_TO_DROP`).

> **Rimossa**: `tipo_lavorazione_macro` — non costruita nei parquet model (ridondante con `oggetto_principale_contratto` per i modelli; rimane in `bando_cig_all.parquet` per analisi).

---

### M2 — Durante (40 feature = 24 ante + 16 aggiudicazioni)
*Aggiunge informazioni su partecipanti e esito della gara*

| Feature | Fonte | Note |
|---|---|---|
| `numero_offerte_ammesse` | AGGIUDICAZIONI | N offerte ammesse (73.7% null nel parquet totale, 0% in M2) |
| `numero_offerte_escluse` | AGGIUDICAZIONI | N offerte escluse |
| `num_imprese_offerenti` | AGGIUDICAZIONI | N imprese che hanno offerto |
| `pct_offerte_escluse` | AGGIUDICAZIONI | % offerte escluse |
| `ribasso_aggiudicazione` | AGGIUDICAZIONI | Ribasso % del vincitore |
| `ribasso_spread` | AGGIUDICAZIONI | Differenza tra ribasso max e vincitore |
| `flag_vince_minimo` | AGGIUDICAZIONI | Il vincitore ha offerto il minimo? (0/1/MISSING) |
| `flag_progettazione_esterna` | AGGIUDICAZIONI | Progettazione esterna? (0/1/MISSING) |
| `lag_aggiudicazione_giorni` | AGGIUDICAZIONI | Giorni dalla pubblicazione all'aggiudicazione |
| `lag_comunicazione_esito_giorni` | AGGIUDICAZIONI | Giorni dalla pubblicazione alla comunicazione esito |
| ~~`delta_coerenza_pct`~~ | ~~AGGIUDICAZIONI~~ | ~~Droppata: segnale spurio da importo_sicurezza_pct. Vedi SC-03.~~ |
| `asta_elettronica` | AGGIUDICAZIONI | Flag asta elettronica |
| `flag_scomputo` | AGGIUDICAZIONI | Flag scomputo oneri sicurezza |
| `flag_proc_accelerata` | AGGIUDICAZIONI | Flag procedura accelerata |
| `importo_aggiudicazione` | AGGIUDICAZIONI | Importo aggiudicato |
| `tipo_soggetto_agg` | AGGIUDICATARI | SINGOLA/ATI/CONSORZIO/SA/GEIE/ALTRO |
| `flag_subappalto` | AGGIUDICAZIONI + SUBAPPALTI | OR tra flag bando e presenza in subappalti.csv |

> **Nota**: Le feature di aggiudicazioni hanno alta null% nel parquet totale (60-74%) perché il 61% dei CIG non ha `esito=AGGIUDICATA`. In M2, filtrato a AGGIUDICATA, la null% crolla drasticamente — queste feature sono quasi complete nel subset M2.

---

### M3 — Ex post (60 feature nativi / 57 preprocessed = 40 durante + 20/17 esecuzione)
*Aggiunge tutta la vita esecutiva del contratto*

| Feature | Fonte | Note |
|---|---|---|
| `lag_stipula_aggiudicazione_giorni` | AVVIO CONTRATTO | Giorni tra aggiudicazione e firma contratto |
| `durata_pianificata_giorni` | AVVIO CONTRATTO | Durata prevista in giorni |
| `consegna_frazionata` | AVVIO CONTRATTO | 0/1/MISSING (3ª categoria approvata) |
| `consegna_sotto_riserva` | AVVIO CONTRATTO | 0/1/MISSING (3ª categoria approvata) |
| `n_varianti` | VARIANTI | N varianti approvate (0 se assente) |
| `flag_variante_sostanziale` | VARIANTI | 0/1 (0 se nessuna variante) |
| `flag_variante_oltre_termine` | VARIANTI | 0/1 (0 se nessuna variante) |
| `pct_overrun_variante` | VARIANTI | Importo varianti / importo aggiudicazione (**solo nativi**) |
| `pct_vita_prima_variante` | VARIANTI | Delta prima variante / durata pianificata (**solo nativi**) |
| `n_sospensioni` | SOSPENSIONI | N sospensioni (0 se assente) |
| `durata_totale_sospensioni_gg` | SOSPENSIONI | Giorni totali sospesi (0 se assente) |
| `flag_sospensione` | SOSPENSIONI | 0/1 |
| `flag_sosp_giudiziaria` | SOSPENSIONI | 0/1 — sospensione per ordine giudiziario |
| `pct_durata_sospesa` | SOSPENSIONI / AVVIO | durata_sosp / durata_pianificata |
| `n_sal` | SAL | N stati di avanzamento (0 se assente) |
| `flag_in_ritardo` | SAL | 0/1/MISSING — NaN se nessun SAL (3ª cat) |
| `flag_proroga` | SAL | 0/1 |
| `pct_overrun_core` | QE CONSUNTIVO | Costo finale / costo iniziale (93.6% null) |
| `pct_riserva_consumata` | QE CONSUNTIVO | Somme disposizione consumate (93.6% null) |
| `flag_consuntivo_presente` | QE CONSUNTIVO | 1 se CONSUNTIVO presente in QE — presente in `model/preprocessed/M3.parquet` |
| `esito_collaudo` | COLLAUDO | POSITIVO/NEGATIVO/MISSING (93.5% null) |

> **Solo nativi**: `pct_overrun_variante` (97.8% null) e `pct_vita_prima_variante` (98.3% null) hanno analoghe binarie già in M3 (`flag_variante_sostanziale`, `flag_variante_oltre_termine`). Incluse solo per modelli con NA nativo.

---

## Riepilogo

| | M1 Ante | M2 Durante | M3 Ex post |
|---|---|---|---|
| **CIG totali** | 9,468,795 | 3,690,097 | 1,129,233 |
| **Labeled** | 2,598 | 1,541 | 490 |
| **COND / SCAG** | 781 / 1,817 | 496 / 1,045 | 210 / 280 |
| **Feature** | 24 | 40 | 60 nativi / 57 preprocessed |
| **DoF one-hot (prep)** | 75 | 136 | 169 |
| **Filtro righe** | nessuno | esito=AGGIUDICATA | AGGIUDICATA + (stipula OR monitoring) |
| **Parquet sorgente** | output/parquet/bando_cig_all | id. (filtered) | id. (filtered) |
| **Parquet model** | output/parquet/model/nativi\|preprocessed | id. | id. |

> **Nota feature count**: `anno_pubblicazione` e `regione` sono colonne di metadata/analisi
> nel parquet sorgente, **non** incluse nei feature set di modello. Per includerle,
> aggiungile a `M1_FEAT` in `pipeline/16_build_model_datasets.py`.
>
> **M3 nativi vs preprocessed**: i nativi (XGBoost/LightGBM) includono 4 variabili continue
> ad alta null% (`pct_overrun_variante`, `pct_vita_prima_variante`, `pct_overrun_core`,
> `pct_riserva_consumata`). I preprocessed sostituiscono queste con `flag_consuntivo_presente`
> (presenza del CONSUNTIVO in QE). Delta: 60 − 57 = 3 (4 nativi-only − 1 preprocessed-only).

---

## Stato pipeline (2026-04-09)

1. ~~Creare `flag_consuntivo_presente`~~ — **Fatto**: presente in `model/preprocessed/M3.parquet` (derivata da `pct_overrun_core.notna()`)
2. **Build model datasets**: completato — 6 parquet in `output/parquet/model/`
3. I 3 subset si ottengono filtrando a runtime — non servono file separati per fase
