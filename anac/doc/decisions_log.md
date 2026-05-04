# Decisions Log — Progetto Appalti ANAC

Traccia le decisioni metodologiche significative prese durante la costruzione del dataset.
Le decisioni a livello di singola variabile stanno in `variables/variable_selection.xlsx`.
Questo file documenta scelte di design, filtri, aggregazioni e gestione delle anomalie.

Aggiornato progressivamente ad ogni dataset processato.

---

## Struttura generale del progetto

- **Fonte dati**: ANAC BDNCP, 19 dataset, CIG come chiave primaria
- **Label positivi** (label=1): 787 CIG — fonte: `fine_contratto_ANAC` (motivo=REATI ACCERTATI o RECESSO CODICE ANTIMAFIA) + `sentenza_TAR` (ACCOGLIE)
- **Label negativi** (label=0): 1843 CIG — fonte: CdS, altri tribunali, esiti non corruzione
- **Unlabeled**: ~9.57M CIG
- **Framework ML**: PU Learning a 3 modelli (ante / durante / ex post)
- **Formato output**: Parquet colonnare (`output/parquet/bando_cig_all.parquet`)

---

## Label construction

### CIG non joinati con ANAC
~164 condannati e ~474 scagionati non hanno dati in nessun dataset ANAC.
**Decisione**: non rimossi dai file label — semplicemente non joinano col mega dataset.
Documentati in `labels/cig_condannati_senza_dati.csv` e `labels/cig_scagionati_senza_dati.csv`.
**Motivi assenza**: SMARTCIG pre-2024, formato CIG anomalo (pre-SIMOG, < 10 caratteri), procedure semplificate non tracciate in BDNCP.

### Terminologia
- I CIG "positivi" sono **condannati** (reato accertato da sentenza o da ANAC), non semplicemente sospetti.
- I CIG "negativi" sono **scagionati** (procedimento concluso senza condanna per corruzione).

---

## Decisioni cross-dataset

### Filtro esito gara (Opzione A)
**Problema**: 6 condannati hanno `esito` = DESERTA / NON_AGGIUDICATA / IN_CORSO in BANDO CIG — artefatto di aggiornamento ANAC (sentenza TAR emessa mentre ANAC non aveva ancora aggiornato il campo).
**Bias potenziale**: tenerli solo in P ma non in U crea selection bias nel PU learning.
**Decisione**: rimuovere simmetricamente da P e U i CIG con:
  - `esito ∈ {DESERTA, NON_AGGIUDICATA, IN_CORSO}`
  - AND `cod_esito` in AGGIUDICAZIONI ∉ {1=efficace, 5=sospesa per contenzioso}
**Impatto**: -6 condannati (0.8%), -26 scagionati (1.4%), -~100k unlabeled (1%).
**Categorie mantenute**: AGGIUDICATA, None (non compilato — tipico affidamenti diretti), ANNULLATA (esito legittimo post-sentenza TAR), SENZA_ESITO (assimilato a None).

### Riaggiudicazioni
**Logica**: riaggiudicazione = secondo contratto dopo rescissione del primo per reati accertati. La corruzione sta sempre nel primo contratto (dimostrato per costruzione del labeling: fine_contratto ha motivo=REATI ACCERTATI sul primo contratto; TAR ACCOGLIE annulla la prima aggiudicazione).
**Identificazione**: `id_aggiudicazione` in `fine-contratto.csv` identifica esattamente quale aggiudicazione fu terminata → join diretto su AGGIUDICAZIONI.
**Decisione**: per ogni CIG con più righe in AGGIUDICAZIONI, tenere la riga identificata da fine-contratto (condannati) o la prima per data senza `cod_modo_riaggiudicazione` (tutti gli altri).
**Feature leaky non costruite**: `flag_riaggiudicazione`, `n_aggiudicazioni` — rivelano che la corruzione è già stata scoperta.

### PU Learning — definizione dei set
- **P** = label=1 (condannati)
- **U** = label=0 + label=NaN (scagionati + unlabeled insieme)
- I label=0 NON sono usati come negativi espliciti: nel framework PU, vanno nel pool U.
- Filtro pre-training: solo `esito = AGGIUDICATA | None | ANNULLATA | SENZA_ESITO`.

### Modelli ante / durante / ex post
- **Ante**: features note al momento della pubblicazione bando (BANDO CIG). `esito` non incluso (non noto).
- **Durante**: features di esecuzione contratto (AGGIUDICAZIONI, AVVIO CONTRATTO, VARIANTI, ...). `esito` non incluso (trivialmente AGGIUDICATA).
- **Ex post**: features post-completamento (COLLAUDO, QUADRO ECONOMICO, ...). `esito` non incluso come feature ma usato come filtro dataset.
- CIG con esito DESERTA/REVOCATO esclusi da tutti e tre (nessun contratto → out of scope).

---

## Dataset: BANDO CIG

### Deduplicazione multi-CPV
ANAC registra una riga per CPV per contratti multi-CPV. `flag_prevalente=1` identifica la lavorazione principale.
**Decisione**: tenere solo `flag_prevalente=1`, droppare la colonna.

### Colonne droppate
- `durata_prevista`: alta correlazione con lag calcolati, poca affidabilità
- `luogo_istat`: usata per join territoriale, poi droppata (regione derivata)
- `flag_prevalente`: dopo deduplicazione, sempre 1

### Feature ingegnerizzate
| Feature | Logica |
|---|---|
| `finestra_offerta_giorni` | `data_scadenza_offerta - data_pubblicazione` |
| `lag_perfezionamento_giorni` | `data_perfezionamento - data_pubblicazione` |
| `lag_comunicazione_esito_giorni` | `data_comunicazione_esito - data_pubblicazione` |
| `importo_sicurezza_pct` | `importo_sicurezza / importo_lotto` |
| `tipo_scelta_4cls` | mapping 35 codici → 4 classi (APERTA/RISTRETTA/NEGOZIATA/AFFIDAMENTO_DIRETTO) |
| `cpv_macro_categoria` | prime 2 cifre CPV → 6 macro-categorie (LAVORI/ING_PROF/SANITA/IT/SERVIZI/FORNITURE) |
| `sezione_regionale` | 3 classi: CENTRALE / REGIONALE / NaN |
| `modalita_realizzazione_macro` | 23 codici ANAC → 6 classi (APPALTO/ECONOMIA/ACCORDO_QUADRO/CONCESSIONE/PPP/ALTRO) |

### Gestione date overflow
Valori estremi (es. 1703, 2229) causano overflow in pandas. **Decisione**: clamp a range 1990-2030 → `NaT` fuori range.

### Join territoriale
`luogo_istat` (codice comune) → primi 3 digit dopo zfill(6) → codice provincia → join su `territoriale/contesto_province.csv` per anno `min(anno_pubblicazione − 1, 2024)` (t-1, capped all'ultimo anno disponibile).
Feature aggiunte: `regione`, `tasso_disoccupazione`, `reddito_irpef_procapite`, `tasso_omicidi_100k`.
**NA residui**: ~40% per CIG numerici storici (luogo_istat non disponibile), ~1% per B-prefix 2024-2025.

### Prefissi CIG
- Numerici (0-9): CIG storici standard
- A-prefix (2023+): nuovo formato CIG post D.Lgs. 36/2023 — trattati come standard
- B-prefix (2024-2025): nuovo formato ANAC 2024, qualità dati uguale o migliore degli storici — trattati come standard
- Tutti i prefissi mantenuti: il mega dataset è già il filtro (se un CIG è nel parquet, è dentro)

---

## Dataset: AGGIUDICAZIONI

### Selezione riga per CIG multi-aggiudicazione
Vedi sezione "Riaggiudicazioni" sopra.

### Feature ingegnerizzate
| Feature | Logica |
|---|---|
| `pct_offerte_escluse` | `escluse / max(num_imprese_offerenti, ammesse+escluse)` |
| `ribasso_spread` | `massimo_ribasso - ribasso_aggiudicazione` (NaN se non asta al ribasso) |
| `flag_vince_minimo` | `ribasso_aggiudicazione == minimo_ribasso` |
| `flag_progettazione_esterna` | `cig_prog_esterna.notna()` |
| `lag_aggiudicazione_giorni` | `data_aggiudicazione_definitiva - data_pubblicazione` |
| `delta_coerenza_pct` | `|importo_aggiudicazione - importo_lotto*(1-ribasso/100)| / importo_lotto` (SC-03 riformulato) |

### Colonne mantenute per SC, non come feature
- `num_imprese_offerenti`: solo SC-OFFERTE
- `cod_esito`: solo SC-16 e filtro fase

### Colonne da droppare alla fine della pipeline
- `data_aggiudicazione_definitiva`, `data_comunicazione_esito`: usate per lag, poi drop

### Boolean NaN → 0
`asta_elettronica`, `flag_scomputo`, `flag_proc_accelerata`: NaN implicano assenza della caratteristica.

---

## Dataset: STAZIONE APPALTANTE

### Feature prodotta
| Feature | Logica |
|---|---|
| `natura_giuridica_SA` | 8 categorie: PA_ENTE / PA_ISTITUZIONALE / SANITARIO / EPE / SOCIETA / NON_PROFIT / CONSORZIO / ALTRO |

### Join key
Join doppio: `codice_ausa` normalizzato (primary) + `cf_amministrazione_appaltante` (fallback).
Normalizzazione necessaria: parquet ha ausa come float-string (`165817.0`), file SA ha leading zeros (`0000165817`).
Fix: strip `.0` + lstrip(`0`). Coverage finale: 99.9% CIG classificati.

### Colonne non incluse
- `regione_sa`: non inclusa — segnale poco pulito (mix SA nazionali e locali)
- `flag_sa_fuori_regione`: non inclusa — dominata da enti nazionali (Consip ecc.), non informativa come proxy di anomalia
- `citta_codice`: assente dal file SA
- `natura_giuridica_codice`: già droppata in filter_columns.py

### Grouping natura_giuridica (8 categorie)
Valutazione fatta a livello CIG (non SA) per identificare la coverage reale:

| Categoria | SA | CIG | % |
|---|---|---|---|
| PA_ENTE | 24.8k | 5.0M | 52.8% |
| SOCIETA | 8.4k | 1.71M | 18.1% |
| PA_ISTITUZIONALE | 3.2k | 1.06M | 11.2% |
| SANITARIO | 355 | 967k | 10.2% |
| EPE | 1.1k | 302k | 3.2% |
| NON_PROFIT | 5.4k | 290k | 3.1% |
| CONSORZIO | 2.1k | 72k | 0.8% |
| ALTRO | 1.6k | 49k | 0.5% |

SANITARIO (ospedali, ASL, previdenza): solo 355 SA ma 10% dei CIG — settore ad alta esposizione corruttiva.
EPE (enti pubblici economici, utilities): separato da PA_ENTE per dinamiche di procurement diverse.
COOPERATIVA (623 SA, 4.7k CIG = 0.05%): merge in SOCIETA, copertura trascurabile.
ECCLESIASTICO (319 SA, 9k CIG = 0.1%): merge in NON_PROFIT, troppo sparso.

### SC-10
codice_ausa: 99.63% match dopo normalizzazione. Residuo 0.37% (34.9k CIG) irrecuperabile — accettabile.

---

## Dataset: PARTECIPANTI

Nessuna feature aggiunta al parquet. Dataset strutturalmente sparso (128k CIG su 9.4M, 1.4% copertura).

### SC-PARTECIPANTI — n_partecipanti vs num_imprese_offerenti
Eseguito su 67.8k CIG confrontabili (partecipanti ∩ aggiudicazioni).

| Casistica | N | % | Interpretazione |
|---|---|---|---|
| Delta = 0 | 21.655 | 31.9% | Coerenza perfetta |
| Delta > 0 | 24.520 | 36.2% | Partecipanti senza offerta (procedure 2 fasi, ritiri) — non errore |
| Delta < 0 | 21.652 | 31.9% | File partecipanti incompleto (solo 11.3% spiegato da ATI) |

**Decisione**: nessuna azione. `num_imprese_offerenti` da AGGIUDICAZIONI è la fonte autorevole per il conteggio offerenti (più completa, 26% coverage vs 1.4%). Il 79% dei CIG confrontabili ha delta ≤ 2.

---

## Dataset: AGGIUDICATARI

### Feature prodotta
| Feature | Logica |
|---|---|
| `tipo_soggetto_agg` | Raggruppamento semantico: SINGOLA / ATI / CONSORZIO / SA / GEIE / ALTRO |

### Colonne usate internamente, non nel parquet
- `ruolo`: usato solo per identificare la riga MANDATARIA nei CIG multi-riga (ATI), poi droppato
- `codice_fiscale`: non entra nel parquet. Dataset aziende (SOA, CERVED) non disponibili; SC-07 eseguito su raw files

### Aggregazione
Dataset multi-riga per CIG (1.13 righe/CIG in media). Aggregazione a CIG level:
- CIG con 1 riga: diretta
- CIG con >1 riga: si prende la riga con `ruolo = MANDATARIA` (se presente), altrimenti la prima riga

### Grouping tipo_soggetto
| Gruppo | Valori inclusi |
|---|---|
| SINGOLA | IMPRESA SINGOLA…, IMPRESA, DITTA INDIVIDUALE, MONOSOGGETTIVO |
| ATI | ATI (RAGGRUPPAMENTI TEMPORANEI…), RTI |
| CONSORZIO | CONSORZIO (tutte le varianti: cooperative, artigiane, stabili) |
| SA | STAZIONE APPALTANTE (concessioni/in-house, ~3% dei CIG) |
| GEIE | GEIE |
| ALTRO | NaN + resto |

### SC-07 — CF aggiudicatario ∈ PARTECIPANTI
**Risultato**: 15.8% fail (24k righe su 154k confrontabili), 3.8% CIG con almeno un aggiudicatario fuori dai partecipanti.
**Severità**: bassa. Copertura partecipanti sparsa (2.7% dei CIG), fail concentrati su MANDANTI di ATI (registrati come singola entità ATI nei partecipanti, non come singole imprese). Nessuna azione necessaria.

---

## AVVIO CONTRATTO

### Dataset
- File: `avvio-contratto.csv` — 1,925,829 righe, 1,922,826 CIG unici (20.3% del parquet)
- CIG con >1 riga: 2,904; gestiti con la stessa logica di AGGIUDICAZIONI (id_aggiudicazione da fine-contratto per i condannati, prima riga altrimenti)

### Feature prodotte

| Feature | Tipo | Copertura (parquet) | Note |
|---------|------|---------------------|------|
| `lag_stipula_aggiudicazione_giorni` | float | 10.2% | data_stipula - data_aggiudicazione_definitiva; neg->NaN (5.2%, causa SC-02) |
| `durata_pianificata_giorni` | float | 12.7% | data_termine_contrattuale - data_inizio_effettiva; neg->NaN (0.45%) |
| `consegna_frazionata` | Int8 nullable | 20.3% | 0=1.55M, 1=83k, NA=292k — NaN mantenuto come categoria |
| `consegna_sotto_riserva` | Int8 nullable | 20.3% | 0=1.38M, 1=252k, NA=292k — NaN mantenuto come categoria |
| date grezze (4) | date | var. | tenute per lags futuri; da droppare a fine pipeline |

### Coppie lag scartate
Valutate tutte le coppie tra le 5 date disponibili (AGG, STIP, INIZIO, CONS, TERM):
- **STIP->INIZIO**, **STIP->CONS**, **INIZIO->CONS**: mediana=0d (eventi quasi contemporanei; nessun valore informativo)
- **STIP->TERM**, **AGG->TERM**: r=0.967 con INIZIO->TERM (ridondante)
- **AGG->CONS**: troppi negativi (>10%), copertura bassa
Tenute solo le 2 coppie con migliore trade-off copertura/ridondanza/negativi.

### Decisioni chiave
- `consegna_frazionata` e `consegna_sotto_riserva`: NaN trattato come categoria distinta (non imputato a 0), da codificare come terza classe nel modello.
- Lag negativi su `lag_stipula_aggiudicazione_giorni` (5.2%): stessa causa di SC-02 (contratti caricati retroattivamente). Già HANDLED.
- `data_aggiudicazione_definitiva` presa dal parquet (non presente in avvio-contratto.csv), poi droppata dalla join temporanea.

### Sanity Checks (SC-06)
- **SC-06a** (inizio_effettiva <= termine_contrattuale): 6,107 violazioni (0.45%) → OK, soglia <1%. Le violazioni coincidono esattamente con i negativi di `durata_pianificata_giorni` (già NaN).
- **SC-06b** (aggiudicazione <= stipula): 52,928 violazioni (5.20%) → HANDLED. Stessa causa SC-02. Il lag è già impostato a NaN per i negativi.

---

## Variabili territoriali

### Fonti
- `tasso_disoccupazione`: ISTAT SDMX (flow 151_1193 per 2007-2020, flow 151_914 per 2020-2024)
- `reddito_irpef_procapite`: MEF, aggregato da comuni a province
- `tasso_omicidi_100k`: ISTAT BES 2025, Indicatori per provincia

### Gestione NA e province
- Province nuove (2009+): backfill da provincia madre (MB←MI, FM←AP, BT←BA)
- Sud Sardegna (111): proxy da Cagliari (92) per anni pre-2016 e omicidi (BES 2025 non ha SU)
- BZ (21) e TN (22): tasso disoccupazione da NUTS2 (ISTAT non ha dati province)
- Ex-province sarde (104-107, abolite 2016): rimosse
- Backward fill omicidi: max 10 anni dal primo valore disponibile (tasso stabile anno-anno)
- Forward fill: sì per tutti; backward fill: solo omicidi con limite 10 anni

---

## Sanity checks

I check sono definiti nel foglio `SANITY_CHECKS` di `variables/variable_selection.xlsx`.
I risultati dell'esecuzione (n violazioni, severity, decisione) vanno documentati qui man mano che vengono eseguiti.

### SC eseguiti

| SC | Violazioni | Severity | Decisione |
|----|-----------|----------|-----------|
| SC-01 | 0 | — | OK, nessun duplicato CIG |
| SC-02 | ~18.5% aggiudicazioni con `lag < 0` | ⚠️ WARNING | `lag_aggiudicazione_giorni = NaN` quando negativo. Causa: ANAC ha caricato retroattivamente contratti pre-2008; la `data_pubblicazione` è la data di inserimento in ANAC (es. 2009), la `data_aggiudicazione_definitiva` è quella reale (es. 2007). Il lag negativo non è interpretabile, non è un errore sul dato. Nessun drop di righe. |
| SC-03 | — | INFO | `delta_coerenza_pct` **DROPPATA**. Segnale apparente (COND mediana 0.019 vs SCAG 0.086) quasi certamente spiegato da `importo_sicurezza_pct` già presente: la formula non deduce gli oneri di sicurezza (non soggetti a ribasso), quindi il delta cresce meccanicamente al crescere di `importo_sicurezza`. Misura di coerenza aritmetica tra tabelle ANAC, non segnale economico sull'appalto. Droppata da parquet (66 col) e da piano NA. |
| SC-05 | ~0.07% (≈6.600 righe) | ✅ bassa gravità | `importo_sicurezza` e `importo_sicurezza_pct` → NaN dove `importo_sicurezza > importo_lotto`. **Condizionale**: fix applicato solo se la colonna ha già NA esistenti (per non alterare dtype). Fisicamente impossibile: arrotondamenti o errori di inserimento. |
| SC-08 | 0 | — | OK, nessun CIG in entrambe le liste |
| SC-13 | 0 condannati con esito anomalo | — | OK post-filtro Opzione A |
| SC-16 | — | INFO | Distribuzione cod_esito stampata a log |
| SC-OFFERTE | ~5.1% delta > 2 | ✅ bassa gravità | Nessuna azione. Delta legittimo: procedure a due fasi, offerte ritirate, criteri di conteggio diversi tra SA. Il denominatore `max()` in `pct_offerte_escluse` è già robusto. |

### SC in attesa di dataset futuri
- SC-04: FONTI DI FINANZIAMENTO (non ancora joinato)
- SC-06: AVVIO CONTRATTO eseguito (SC-06a OK, SC-06b HANDLED); FINE CONTRATTO ancora da processare
- SC-09: BANDO CIG numero_lotti vs count AGGIUDICAZIONI (richiede aggregazione)
- SC-10: codice_ausa BANDO CIG vs STAZIONE APPALTANTE — eseguito; 99.63% match dopo normalizzazione float-string
- SC-11, SC-12: QUADRO ECONOMICO (non ancora joinato)

---

## VARIANTI

### Dataset
- File: `varianti.csv` — 321,294 righe, 220,976 CIG unici (2.3% del parquet)
- Colonne: `cod_motivo_variante`, `data_approvazione_variante`, `cig`, `id_aggiudicazione`
- No `importo_variante` nel file → `pct_overrun_variante` non computabile

### Feature prodotte

| Feature | Tipo | Note |
|---------|------|------|
| `n_varianti` | int (0 se assente) | mediana=1, p95=3, max=538 |
| `flag_variante_sostanziale` | int 0/1 (0 se assente) | 48% dei CIG con varianti; condannati 4.7% vs scagionati 0.9% (5x) |
| ~~`delta_prima_variante_giorni`~~ | ~~float, NaN se assente~~ | ~~Superseded da `pct_vita_prima_variante` (vedi sezione "VARIANTI decisioni post-discussione"). Il parquet contiene la versione normalizzata.~~ |

### Decisioni chiave
- Assenza dal file = 0 varianti (non NaN): semanticamente informativo, distinto da "dato mancante"
- `flag_variante_sostanziale`: esclude codici 17/18/19/22/96/98/99 (amministrativi/leciti); include tutto il resto (errrori progetto, lavori supplementari, miglioramenti, sostituzione contraente, ecc.)
- Codici 96/98/99 esclusi perché 98="non sostanziali" (esplicito), 99="non specificato", 96=revisione prezzi contrattuale
- Date anomale clampate [2000, 2040] prima del calcolo delta (overflow int64 su valori outlier)

### Sanity Checks
- **SC-VAR-1** (delta < 0): 945 (0.4%) → NaN, OK
- **SC-VAR-2** (condannati vs scagionati): flag_sost 4.7% vs 0.9% — segnale forte

---

## SOSPENSIONI

Script: `ANAC/build_sospensioni.py` (v2 — rivisto dopo analisi motivi)
Parquet dopo: 9,468,795 x 72 colonne

### Dataset
- File: `sospensioni.csv` — 227,747 righe, 126,325 CIG unici (1.3% del parquet)
- Colonne: `cig`, `data_sospensione`, `data_ripresa`, `descrizione_motivo`, `id_aggiudicazione`
- 7 categorie di motivo; data_ripresa mancante 7.7% (sospensioni ancora aperte: escluse da durata)

### Feature prodotte (v2)

| Feature | Tipo | Note |
|---|---|---|
| `n_sospensioni` | int (0 se assente) | mediana=1, p95=4, max=126 |
| `durata_totale_sospensioni_gg` | float (**0** se assente) | mediana=92d, p95=499d; solo righe con data_ripresa |
| `flag_sospensione` | int 0/1 | 1 se n_sospensioni > 0; condannati 5.9% vs scagionati 0.7% (8x) |
| `flag_sosp_giudiziaria` | int 0/1 (0 se assente) | 534 CIG; unico flag qualitativo (intervento diretto autorita') |
| `pct_durata_sospesa` | float | durata_totale / durata_pianificata; **0** se no sosp; NaN se durata mancante; cap 10; 99.8% non-null |

### Decisione sui motivi (v2)
Analisi dati mostra che tutti i 7 motivi hanno segnale (4x-18x). In particolare, i motivi
classificati a priori come "innocui" (forza maggiore 17x, interferenze tecniche 18x) hanno
segnale comparabile o superiore ai "sospetti". Due interpretazioni: pretesto procedurale, o
confounding da dimensione appalto.

**Scelta**: opzione 1 — flag binario unico (`flag_sospensione`). Distinzione qualitativa solo
per `flag_sosp_giudiziaria` (unico motivo con implicazione legale diretta esplicita).
`flag_sosp_sospetta` droppato (grouping a priori non confermato dai dati).

### pct_durata_sospesa
- Numeratore: durata_totale_sospensioni_gg (0 se assente)
- Denominatore: durata_pianificata_giorni (da AVVIO CONTRATTO)
- pct > 1 atteso: le sospensioni estendono il contratto oltre la durata pianificata originale (39K CIG)
- pct > 10 → NaN (1,267 casi estremi)
- Coverage 99.8%: quasi tutto il parquet ha 0 (no sospensioni); NaN solo 21,515 CIG (sosp. presente ma durata mancante)

---

## VARIANTI (decisioni post-discussione)

### n_varianti — cap a 20
Cap scelto dopo analisi: >20 varianti = 83 CIG (0.04%), di cui 25% batch anomali (stessa data, stesso codice in un giorno solo). Nessun breakpoint naturale nella distribuzione; cap a 20 elimina i casi palesemente artefattuali.

### pct_overrun_variante (da QUADRO ECONOMICO — righe VARIANTE)
- `importo_variante_totale` = somma importi su tutte le colonne per righe descrizione_evento=VARIANTE
- `pct_overrun_variante` = importo_variante_totale / importo_aggiudicazione; valori <0 o >10 → NaN
- Copertura: 2.2% parquet; condannati mediana 1.78 vs scagionati 1.45
- **NOTA**: quando si processa QUADRO ECONOMICO per intero, calcolare anche `pct_overrun_totale` = importo_CONSUNTIVO / importo_BASE_ASTA — più robusto perché usa i totali definitivi, non la somma per componente. 627k CIG hanno entrambi i valori nel parquet.

### Variabili timing da aggiungere (in attesa conferma)
- `pct_vita_prima_variante` = delta_prima_variante / durata_pianificata (continua, NaN se assente)
- `flag_variante_oltre_termine` = 1 se pct_vita > 1 (NaN → 0); condannati ~4.6% vs scagionati ~1%

---

## QUADRO ECONOMICO

Script: `ANAC/build_quadro_economico.py`
Parquet dopo: 9,468,795 x 71 colonne

### Struttura fonte
`quadro-economico.csv` — 5.5M righe, 4.6M CIG unici.
Ogni CIG puo' avere piu' righe per `descrizione_evento` (BASE_ASTA, CONSUNTIVO, VARIANTE, ...).
Aggregazione: `groupby('cig').sum(min_count=1)` — NaN solo se tutte le righe sono NaN.

### 6 colonne core
`importo_lavori`, `importo_forniture`, `importo_servizi`, `importo_progettazione`,
`importo_sicurezza`, `ulteriori_oneri_non_soggetti_ribasso`
Sommate per ottenere `sum_core` = totale costi vivi dell'appalto (escluse somme a disposizione).

### somme_a_disposizione
Riserva contingency della SA — separata dalle 6 colonne core. Misura quanto margine
la SA si e' tenuta prima (BASE_ASTA) e quanto ne ha consumato (CONSUNTIVO).

### Feature prodotte
| Feature | Formula | Evento | Coverage |
|---|---|---|---|
| `pct_riserva_base` | somme_disp(BA) / sum_core(BA) | ante | 2.65M non-null (28%) |
| `pct_overrun_core` | sum_core(CO) / sum_core(BA) | ex post | 603K non-null (6.4%) |
| `pct_riserva_consumata` | somme_disp(CO) / sum_core(BA) | ex post | 604K non-null (6.4%) |

Valori <0 o >10 → NaN (cap+NaN policy).

### Signal sui labeled
- `pct_riserva_base`: CONDANNATI mediana=0.219 (n=454) vs SCAGIONATI mediana=0.085 (n=960) → 2.6x. Segnale forte ante.
- `pct_overrun_core`: CONDANNATI 73 / SCAGIONATI 14 sample — ex post, troppo piccolo per concludere.
- `pct_riserva_consumata`: stesso problema sample.

### importo_sicurezza fallback
Dove BANDO CIG aveva NaN in `importo_sicurezza`, riempito da QE BASE_ASTA.
SC-11 conferma 95.8% match (tolleranza 1%) tra le due fonti → fallback sicuro.
Riempiti +593,047 valori.

### SC eseguiti
- **SC-09**: n_lotti_componenti vs count aggiudicazioni — 13.7% discrepanza su 3.4M confrontabili. Solo informativo; n_lotti_componenti viene da BANDO, count_agg da AGGIUDICAZIONI — disallineamenti attesi (lotti deserte, riaggiudicazioni, ecc.).
- **SC-11**: importo_sicurezza QE vs BANDO — 95.8% match → coerenza alta.
- **SC-12**: importo_lotto vs sum_6col BASE_ASTA — 78.9% match (tol 5%) → informativo; differenza plausibile (importo_lotto include IVA, QE al netto).

### Colonne droppate
- `importo_variante_totale`: gia' sostituita da `pct_overrun_variante` (relativa). Rimossa in questo step.

---

## FONTI DI FINANZIAMENTO

**DROPPATO.** Coverage 1.6%, natura quasi-binaria (96% dei CIG ha una sola fonte al 100%),
sample labeled 67/89, bias di scoperta sui fondi comunitari non gestibile in modellazione.
Nessuna feature estratta, nessuna modifica al parquet.

---

## SUBAPPALTI

Script: `ANAC/build_subappalti.py`
Parquet dopo: 9,468,795 x 76 colonne

### Feature prodotta
`flag_subappalto` (int 0/1) — logica OR su due fonti:
- `flag_subappalto=True` da AGGIUDICAZIONI: subappalto autorizzato/ammesso nel bando (482K CIG)
- Presenza in `subappalti.csv`: subappalto effettivamente registrato (121K CIG)
- Overlap: 103K CIG; union: 500K CIG → 494K nel parquet (5.2%)

### Decisioni
- `subappalti.csv` non produce feature proprie: coverage 1.2%, segnale debole (4.2% vs 3.2%)
- `flag_subappalto` da AGGIUDICAZIONI non era nel parquet (non processato in build_aggiudicazioni)
- OR combinato da' coverage 5.2% e segnale piu' robusto
- cod_categoria e cod_cpv da subappalti.csv droppati (troppo sparsi, non discriminativi)
- Subappalti irregolari/non dichiarati non tracciati in ANAC: segnale volutamente conservativo

### Segnale
CONDANNATI 21.0% vs SCAGIONATI 17.0% (1.2x) — moderato ma coerente con la letteratura
(subappalto come meccanismo di trasferimento proventi illeciti verso imprese collegate).

---

## LAVORAZIONI

Script: `ANAC/build_lavorazioni.py`
Parquet dopo: 9,468,795 x 77 colonne

### Dataset
- File: `lavorazioni.csv` — 1,008,701 righe, 893,992 CIG unici, 2 colonne (cig, cod_tipo_lavorazione)
- Coverage nel parquet: 9.1% — esclusivamente LAVORI (65.5% dei LAVORI ce l'ha; SERVIZI/FORNITURE 0%)
- tipo_lavorazione testuale assente dal CSV (contrariamente a quanto documentato in Excel)

### Lookup cod_tipo_lavorazione
Fonte: dati.anticorruzione.it + github.com/anticorruzione/npa/tipologiaLavoro.json
- 1-5: modalita' acquisizione (ACQUISTO/LEASING/NOLEGGIO/RISCATTO/MISTO) — anomali per LAVORI
- 6-7: COSTRUZIONE, DEMOLIZIONE
- 8-10: RECUPERO, RISTRUTTURAZIONE, RESTAURO
- 11,14: MANUTENZIONE generica e combinata — deprecati 2014 (contratti storici)
- 12-13: MANUTENZIONE ORDINARIA, MANUTENZIONE STRAORDINARIA

### Feature prodotta: tipo_lavorazione_macro (DROPPATA)
Raggruppamento in 3 macro-classi + NaN:
- COSTRUZIONE (cod 6-7): opera nuova o demolizione
- RISANAMENTO (cod 8-10): recupero, ristrutturazione, restauro
- MANUTENZIONE (cod 11-14): qualsiasi manutenzione
- NaN: assente dal dataset oppure cod 1-5 (ACQUISIZIONE, anomala, 0 labeled)

**Rimossa dai feature set M1/M2/M3** (2026-04-12): non-null solo per contratti LAVORI (9.1%
delle righe). Ridondante con `oggetto_principale_contratto` per il modello — il segnale
COSTRUZIONE/MANUTENZIONE/RISANAMENTO e' interamente nested in `oggetto=LAVORI`.
Non ridondante con `cpv_macro_categoria` (i valori non-null si distribuiscono su tutti i CPV).
Rimane in `bando_cig_all.parquet` per analisi descrittive.

### Segnale (pre-drop, per riferimento)
- COSTRUZIONE: COND 9.9% vs SCAG 5.1% (1.9x)
- RISANAMENTO: COND 5.1% vs SCAG 2.7% (1.9x)
- MANUTENZIONE: COND 13.7% vs SCAG 6.1% (2.2x) — ratio piu alto, semanticamente sensato
  (manutenzione difficile da verificare ex-post: lavori fantasma, gonfiamento prezzi)
- Nota: parte del segnale e' confounding con tipo contratto LAVORI

### NOTA: pct_overrun_totale (alias pct_overrun_core)
Il `pct_overrun_core` calcolato qui (sum_core_CONSUNTIVO / sum_core_BASE_ASTA) e' equivalente
al `pct_overrun_totale` menzionato in precedenza. Copertura 6.4% — stessa dell'ex-post da QE.

---

## STATI DI AVANZAMENTO (SAL)

Script: `ANAC/build_sal.py`
Parquet dopo: 9,468,795 x 75 colonne

### Dataset
- File: `stati-avanzamento.csv` — 1,274,801 righe, 343,824 CIG unici
- Coverage nel parquet: 3.2% (300K CIG)
- 6 colonne: cig, progressivo_sal, flag_ritardo, n_giorni_scostamento, giorni_proroga, id_aggiudicazione

### SAL e' istituto specifico dei LAVORI
LAVORI 11.4% coverage, SERVIZI 2.4%, FORNITURE 1.4%.
Assenza di SAL non significa "nessun ritardo": significa che il contratto non viene monitorato con questo strumento (tipico per forniture e servizi).

### Feature prodotte

| Feature | Tipo | Assenza |
|---|---|---|
| `n_sal` | int (0 se assente) | 0 |
| `flag_in_ritardo` | Int8 0/1/NaN | NaN (nessun SAL) |
| `flag_proroga` | int 0/1 | 0 |

### Decisioni
- `n_giorni_scostamento` DROPPATO: mediana=p95=0, max=318k (errori di inserimento), coperto da `flag_in_ritardo`
- `giorni_proroga_totale` non calcolato: 97% zeri, coperto da `flag_proroga`
- `pct_sal_in_ritardo` non calcolata: autocorrelazione temporale la rende equivalente al flag binario
- `flag_in_ritardo` usa NaN (non 0) per assenza SAL: distingue "mai in ritardo" da "nessun SAL registrato"

### Segnale
- `flag_in_ritardo`: CONDANNATI 14.7% vs SCAGIONATI 7.7% (calcolato su CIG con SAL, n=95/104) — 1.9x
- `flag_proroga`: CONDANNATI 1.7% vs SCAGIONATI 0.2% — 8.5x
- SC: 28.5% CIG con proroga e' anche in ritardo vs 2.2% senza (coerenza interna)
- Nota: il paradosso "scagionati piu' in ritardo" visto in analisi preliminare era un artefatto del calcolo pct sul full labeled set (includeva zeros); scomparso con flag binario calcolato correttamente.

---

## COLLAUDO

Script: `pipeline/15_build_collaudo.py`
Parquet dopo: 9,468,795 x 67 colonne (step 15 nella pipeline corrente)

### Dataset
- File: `collaudo.csv` — colonne cig, esito_collaudo (+ altre non usate)
- Coverage nel parquet: 6.5% (CIG con almeno un record collaudo)

### Feature prodotta
`esito_collaudo` (categoriale, NaN se assente) — preso as-is dalla fonte, drop_duplicates('cig').
Assenza = NaN (93.5%): non significa "collaudo ok", significa "non registrato in ANAC".

### Decisione
Join diretto senza aggregazione. Coverage troppo bassa per features derivate.
Tenuto come categoriale grezzo per non perdere informazione sulla distribuzione degli esiti.

---

## CATEGORIE DPCM

**DROPPATO.** Dataset `categorie-dpcm-aggregazione.csv` non processato.
Coverage e segnale insufficienti per giustificare l'aggiunta di feature.
Nessuna modifica al parquet.

---

## CATEGORIE OPERA

**DROPPATO.** Dataset `categorie-opera.csv` — 11.4M righe, codici SOA OG/OS per categoria prevalente.

### Analisi svolta
- Lookup completo costruito in `ANAC/lookup/categorie_opera.csv` (fonti: github.com/anticorruzione/npa/categoria.json + Allegato A DPR 207/2010)
- 7 macro_gruppo testati: LAVORI_INFRASTRUTTURE (1.46x), LAVORI_EDILIZIA (1.31x), LAVORI_SPECIALIZZATI (1.29x), LAVORI_IMPIANTI (1.25x), ALTRO (1.00x), SERVIZI (0.94x), FORNITURE (0.77x)
- Merge con `cpv_macro_categoria` (gia' nel parquet) valutato in 3 opzioni

### Decisione: usa cpv_macro_categoria as-is
Il merge SOA+CPV produceva valore solo se si mantenevano i 4 sub-tipi LAVORI.
Collassando tutti i LAVORI in categoria unica, la variabile unificata e' sostanzialmente
identica a `cpv_macro_categoria` (gia' presente, 6 valori, 100% coverage).

`cpv_macro_categoria` rimane la variabile di macro-categorizzazione contratto.
Nessuna nuova colonna aggiunta al parquet.

### Conflitti CPV vs SOA (nota per report)
420K CIG (4.4%): CPV=LAVORI ma SOA=FB/FS/AA — enti con sistema qualificazione proprio (AA)
o potenziali errori di classificazione. Segnale 1.13x vs LAVORI puri 1.35x.
Non aggiunto come flag (marginal value troppo basso).

---

## FINE CONTRATTO

**DROPPATO — leakage grave.**

### Dataset
- File: `fine-contratto.csv` — 1,070,254 righe, 1,068,186 CIG unici
- Colonne: cig, cod_motivo_risoluzione, motivo_risoluzione, cod_motivo_interruzione_anticipata,
  motivo_interruzione_anticipata, data_conclusione_anticipata, data_effettiva_ultimazione,
  id_aggiudicazione, giorni_proroga (100% null)

### Problema: leakage dalla derivazione delle label
Il file e' stato usato per derivare alcune label (interruzioni per ragioni criminali).
Questo crea correlazione artificiale tra presenza nel file e label=1:
- CONDANNATI: 32.8% presenti nel file (n=256)
- SCAGIONATI:  2.6% presenti nel file (n=48)
- Ratio solo per presenza: 12.4x

Qualsiasi feature estratta dal file (inclusa data_effettiva_ultimazione per calcolo ritardi)
eredita questo bias. Il modello imparerebbe "presente in fine-contratto → condannato"
invece della relazione causale sottostante.

### Decisione
Drop totale. Nessuna feature estratta, nessuna modifica al parquet.
`ritardo_ultimazione_giorni` non calcolabile in modo pulito da questa fonte.

> **Nota pipeline**: nonostante il drop come fonte di feature, `fine-contratto.csv` è ancora letto da `05_build_aggiudicazioni.py` per la logica di selezione riga nelle riaggiudicazioni (vedi sezione "Riaggiudicazioni"): l'`id_aggiudicazione` presente in fine-contratto identifica quale aggiudicazione fu terminata per reati, permettendo di selezionare la riga corretta in AGGIUDICAZIONI per i condannati. Il file non viene rimosso dal pipeline.

---

## REVIEW FINALE E DROP COLONNE

### Fix dtype applicati al parquet (pre-drop)
- `flag_variante_sostanziale`: bug NaN/0 — 9.25M CIG con n_varianti=0 avevano NaN invece di 0;
  96K CIG con varianti tutte cod=99 (non specificato, escluso) avevano NaN invece di 0.
  Fix: fillna(0) → int8. Coerente con flag_variante_oltre_termine (gia' corretto).
- `asta_elettronica`, `flag_scomputo`, `flag_proc_accelerata`: float64 con 62.5% NaN →
  NaN semanticamente = 0 (CIG non in aggiudicazioni = no procedura elettronica/scomputo/accelerata).
  Fix: fillna(0) → int8.

### Colonne droppate dal parquet (78 → 66 colonne)

| Colonna | Motivo drop |
|---|---|
| `stato` | Zero varianza: 9,468,793 ATTIVO + 2 CANCELLATO |
| `data_aggiudicazione_definitiva` | Data grezza — lag gia' computato in `lag_aggiudicazione_giorni` |
| `data_comunicazione_esito` | Data grezza — lag gia' computato in `lag_comunicazione_esito_giorni` |
| `data_stipula_contratto` | Data grezza — lag gia' computato in `lag_stipula_aggiudicazione_giorni` |
| `data_inizio_effettiva` | Data grezza — gia' usata per `durata_pianificata_giorni` |
| `data_termine_contrattuale` | Data grezza — gia' usata per `durata_pianificata_giorni` |
| `data_verbale_consegna_definitiva` | Data grezza — nessun lag mai computato, nessun valore residuo |
| `cod_esito` | Versione codificata di `esito`, ridondante |
| `codice_ausa` | Join key SA gia' servito; identificatore SA non necessario nel parquet finale |
| `cf_amministrazione_appaltante` | Alta cardinalita' (45K valori), non usabile come feature diretta; droppato per ora |
| `importo_sicurezza` | Raw amount (32.4% non-null, segnale 1.28x) superseded da `importo_sicurezza_pct` (segnale 3.79x) |
| `delta_coerenza_pct` | Misura di coerenza aritmetica tra tabelle ANAC (bando vs aggiudicazioni). Segnale spurio: guidato da `importo_sicurezza_pct` (formula non detrae gli oneri fissi dal ribasso). Nessun segnale economico autonomo. |

### Colonne mantenute con nota
- `esito`: presente solo in `bando_cig_all.parquet` come selettore righe per i 3 modelli (ante/durante/post).
  Non usato come feature. AGGIUDICATA = modelli 2/3, ANNULLATA = modello 1.
  Non inclusa nei 6 parquet model (nativi/preprocessed M1/M2/M3) — esclusa a valle della selezione righe in `pipeline/16_build_model_datasets.py`.
- `anno_pubblicazione`, `regione`: colonne di metadata/analisi. Non incluse nei feature set M1/M2/M3.
  Utili per analisi descrittive e slice-by-slice evaluation. Per includerle come feature, aggiungere a `M1_FEAT` in `pipeline/15_build_model_datasets.py`.
- `data_pubblicazione`: colonna ausiliaria temporanea. Aggiunta da `04_build_bando_cig.py` per permettere a `05_build_aggiudicazioni.py` di calcolare `lag_aggiudicazione_giorni` e `lag_comunicazione_esito_giorni`. Eliminata dal parquet sorgente in `15_build_model_datasets.py` (non presente nei parquet model). Non è una feature; conteggio colonne del parquet sorgente la esclude (66 col).

### NA come livello esplicito nei parquet preprocessed
Tutti i NA nei parquet `model/preprocessed/` sono convertiti in un livello categoriale
esplicito `"MISSING"` — non NA. Questo vale per ogni colonna feature non numerica:

- **Categoriali con NA piccolo (<1%)**: imputate con moda su M1 — `tipo_scelta_4cls`
  (AFFIDAMENTO_DIRETTO), `oggetto_principale_contratto` (FORNITURE), `natura_giuridica_SA`
  (PA_ENTE). Livello MISSING non creato per queste colonne.
- **Categoriali nominali stringa** (`sezione_regionale`, `esito_collaudo`,
  `tipo_soggetto_agg`, flag 3-categoria, bins discretizzati): `fillna('MISSING')`.
- **Codici interi nominali** (`cod_strumento_svolgimento`, `cod_motivo_urgenza`):
  convertiti da `Int64` a `category` stringa, NA → `"MISSING"`.
- **`label`**: unica colonna con NA residui (by design — unlabeled = NA per PU learning).

Implementazione: sweep finale in `apply_preprocessing()` in `pipeline/16_build_model_datasets.py`.
R e Python vedono `"MISSING"` come livello categoriale, non come NA.

### Segnali anomali documentati (non richiedono azione)
- `pct_overrun_core`, `pct_riserva_consumata`: segnale apparentemente invertito (COND < SCAG)
  ma sample troppo piccolo (n=73/14). Probabile bias da appalti interrotti prima del completamento
  (CONSUNTIVO parziale → overrun apparentemente basso). Tenere con caveat.
- `esito_collaudo=POSITIVO` ratio=2.73x: selection bias — i condannati completano piu' spesso
  l'iter burocratico formale (collaudo anche su lavori corrotti).
- `delta_coerenza_pct`: droppata — segnale spurio da `importo_sicurezza_pct` (documentato SC-03 aggiornato).
- `numero_offerte_ammesse` ↑COND: coerente con cartello — offerte fasulle per simulare competizione.
