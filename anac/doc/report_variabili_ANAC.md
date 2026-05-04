# Report variabili ANAC — Analisi qualitativa per PU Learning sulla corruzione negli appalti pubblici

**Versione**: 3.1
**Data**: 2026-03-27
**Scope**: 25 dataset ANAC BDNCP, 2019–2025

---

## Premessa metodologica

Questo report integra l'analisi statistica dell'Excel con un'interpretazione qualitativa variabile per variabile. Per ogni entry vengono discussi: il significato nel ciclo d'appalto, come la corruzione si manifesta attraverso quella variabile, e la proposta di trasformazione per il modello PU learning.

Struttura del report:
- **PARTE 0**: Decisioni operative confermate (riepilogo esecutivo)
- **PARTE 1**: Analisi per-variabile (entries R-XX organizzati per dataset)
- **PARTE 2**: Feature engineering sistematico — flag, lag, rapporti
- **PARTE 3**: Problemi metodologici (NA complessi, leakage, aggregazioni 1:N)
- **PARTE 4**: CPV — proposta di raggruppamento
- **PARTE 5**: Domande aperte residue

---

## PARTE 0 — Decisioni operative confermate

Riepilogo di tutte le decisioni prese. Queste **non sono più in discussione** e sono già riflesse nell'Excel.

### 0.1 Tipo di modello
- **Modello prospettico** (predittivo su nuovi contratti). Conseguenza: tutti i dati ex post (collaudo, fine contratto, SAL avanzati) sono leakage se usati come feature dirette.
- **Eccezione**: dati ex post aggregati storicamente (es. tasso medio di varianti dei contratti precedenti dello stesso appaltatore) non sono leakage per il contratto corrente.

### 0.2 Dataset da scartare interamente

| Dataset | Motivazione |
|---|---|
| FINE CONTRATTO | Usato parzialmente per costruire i label. Drop completo per leakage circolare. |
| CENTRO DI COSTO | Ridondante rispetto a STAZIONE APPALTANTE, nessun valore predittivo. |

### 0.3 Variabili testo (denominazioni) → DROP

Tutte le denominazioni in testo libero vengono droppate. Il join si fa tramite codice fiscale o codice AUSA.

| Variabile droppata | Dataset | Chiave di join alternativa |
|---|---|---|
| denominazione aggiudicatario | AGGIUDICATARI | codice_fiscale → SOA |
| denominazione amministrazione | BANDO CIG | codice_ausa |
| oggetto_gara / oggetto_lotto | BANDO CIG | — (NLP out of scope) |
| DENOMINAZIONE_SA_DELEGANTE/DELEGATA | BANDO CIG | CF_SA_DELEGANTE |
| denominazione partecipante | PARTECIPANTI | dataset usato solo per aggregati |
| denominazione SA | STAZIONE APPALTANTE | codice_ausa / codice_fiscale |
| denominazione subappaltatore | SUBAPPALTI | dataset usato per aggregati |
| denominazione_sal | SAL | — |

### 0.4 Codice vs testo ridondanti → tieni codice, droppa testo

| Coppia | Tieni | Droppa |
|---|---|---|
| cod_tipo_scelta_contraente + testo | codice | tipo_scelta_contraente |
| cod_modalita_realizzazione + testo | codice | modalita_realizzazione |
| COD_MODALITA_INDIZIONE_* + testo | codice | testi ridondanti |
| COD_STRUMENTO_SVOLGIMENTO + testo | codice | STRUMENTO_SVOLGIMENTO |
| COD_MOTIVO_URGENZA + testo | codice | MOTIVO_URGENZA |
| COD_ESITO (BANDO) + ESITO testo | codice | testo ESITO |
| cod_tipo_lavorazione + testo | codice | tipo_lavorazione |
| cod_cpv + descrizione_cpv | cod_cpv | descrizione_cpv (tutti i dataset) |
| categoria SOA + desc_categoria | categoria | desc_categoria |

### 0.5 Decisioni su variabili specifiche

| Variabile | Decisione | Motivazione |
|---|---|---|
| anno_aggiudicazione | **Non usare** (nemmeno come derivata) | Confounding temporale; effetti normativi da gestire in architettura |
| COLLAUDO: 7 date di collaudo | DROP (leakage prospettico) | Tutte ex post rispetto all'aggiudicazione |
| COLLAUDO: RISERVE + CONTENZIOSO | TRANSFORM (ratio) | Segnale contrattuale forte, calcolato ex post ma usabile retrospettivamente |
| VARIANTI: cod_motivo_variante | TRANSFORM | Feature di overrun, decisione confermata |
| CIG_COLLEGAMENTO | TRANSFORM → `flag_ha_cig_collegato` (bool) | Network features scartate per dipendenza tra osservazioni |
| STRUMENTO_SVOLGIMENTO | TRANSFORM — due opzioni da testare | Vedi § 0.6 |
| dettaglio_evento (QUAD.EC.) | DROP | Label ad alta cardinalità, non predittiva |
| FUNZIONI_DELEGATE | DROP | 99.5% NA, coperto da FLAG_DELEGA |
| FONTI FINANZIAMENTO | TRANSFORM → ratio | Convertire tutti gli importi in % sul totale |
| data_guce | TRANSFORM → `flag_pubblicato_guce` | Proxy "sopra soglia EU, alta sorveglianza" |
| cf_impresa (SOA) | KEEP — chiave join | Collegamento SOA ↔ AGGIUDICATARI |
| codice_fiscale (AGGIUDICATARI) | KEEP — chiave join | Collegamento verso SOA e dati esterni |
| codice_fiscale (STAZIONE APPALTANTE) | KEEP — chiave join | Collegamento CIG ↔ CF_PA |

### 0.6 Strumento_svolgimento — due opzioni da confrontare

Il COD_STRUMENTO_SVOLGIMENTO ha 97.6% NA nel campione mensile. Sul dataset completo la copertura è maggiore. **Testare entrambe** le codifiche in cross-validation:

- **Opzione A** — categoriale a 5 classi: MePA, Consip, Accordo Quadro, gara elettronica, altro
- **Opzione B** — bool separati: `flag_mepa`, `flag_consip`, `flag_accordo_quadro`, `flag_gara_online`

Non è possibile scegliere a priori: dipende dal segnale empirico. Registrare le performance di entrambe nel log sperimentale.

### 0.7 Label bias — vincolo accettato

I label positivi derivano da sentenze TAR e da FINE CONTRATTO. Questo introduce un bias strutturale: il modello impara "contratti corrotti che sono stati anche formalmente contestati o rilevati", non "tutti i contratti corrotti". Conseguenze:
- Le feature più rilevanti saranno quelle visibili nelle contestazioni formali (varianti, riserve, overrun documentato).
- Il modello è conservativo lato positivo: sarà sottostimato rispetto alla corruzione effettiva.
- Da documentare come **limitation intrinseca**, non come errore di ingegnerizzazione.

---

## PARTE 1 — Analisi per-variabile

---

### AGGIUDICAZIONI *(4.9M righe — dataset centrale del modello)*

---

#### [R-01] data_aggiudicazione_definitiva
**NA**: 0.9% | **Tipo**: date | **Azione**: TRANSFORM → `lag_bando_aggiudicazione_gg`

Data in cui la SA formalizza l'aggiudicazione definitiva. Da sola non dice molto, ma il delta con `data_pubblicazione` (BANDO CIG) misura la durata dell'iter di gara. Un lag anomalmente basso in procedure formalmente competitive (es. procedura aperta con lag di 3 giorni) può indicare che l'esito era pre-determinato: la legge impone almeno 35 giorni di termine per le offerte, quindi se l'aggiudicazione arriva in pochi giorni dalla pubblicazione, i tempi di verifica e valutazione non sono stati rispettati. Valori negativi (aggiudicazione formalmente prima della pubblicazione) esistono per procedure registrate a posteriori.

**Proposta**: calcolare `lag_bando_aggiudicazione_gg`. Usare come anchor per altri lag (stipula, avvio lavori).

---

#### [R-02] criterio_aggiudicazione
**NA**: 66.3% | **Tipo**: cat | **Azione**: DROP

Come discusso: 2–3 categorie reali, tutte centrate sul prezzo. Il 66% di NA e la quasi-assenza di varianza reale la rendono non informativa. DROP.

---

#### [R-03] data_comunicazione_esito
**NA**: 0.1% | **Tipo**: date | **Azione**: DROP (ridondante)

Coincide quasi sempre con `data_aggiudicazione_definitiva` (stessa fonte, spesso stesso giorno). Drop o usare come backup per eventuali NA di R-01.

---

#### [R-04] massimo_ribasso
**NA**: 77.6% | **Tipo**: num | **Azione**: KEEP — usare in combinazione con R-05

Il ribasso dell'offerta più alta tra quelle ammesse. Da solo ha limitata utilità, ma combinato con `minimo_ribasso` permette di calcolare il **ribasso_spread** = massimo − minimo. In una gara sana le offerte si disperdono lungo un intervallo; uno spread prossimo a zero tra molti offerenti è un segnale classico di accordo di cartello (coordinamento sul prezzo). Calcolabile solo dove `num_imprese_offerenti > 2`.

---

#### [R-05] minimo_ribasso
**NA**: 79.8% | **Tipo**: num | **Azione**: KEEP — vedi R-04

Vedi sopra. Feature derivata: `ribasso_spread = massimo_ribasso − minimo_ribasso`.

---

#### [R-06] / [R-07] COD_PRESTAZIONI_COMPRESE / PRESTAZIONI_COMPRESE
**NA**: 61.2% | **Tipo**: cat | **Azione**: TRANSFORM → `flag_design_build`

Indica se l'aggiudicatario esegue soltanto i lavori o anche li progetta. Tre livelli:
- **SOLA ESECUZIONE**: progetto già approvato, l'impresa esegue. Rischio standard.
- **PROGETTAZIONE ED ESECUZIONE (su progetto definitivo)**: l'impresa affina un progetto già sviluppato. Rischio medio.
- **PROGETTAZIONE ED ESECUZIONE (su progetto preliminare)**: design & build completo. **Rischio alto**: lo stesso soggetto controlla sia la progettazione sia l'esecuzione. Può scrivere capitolati tecnici che avvantaggiano se stesso nelle varianti successive, e la SA ha scarsa capacità di verifica indipendente.

**Proposta**: `flag_design_build = 1` se PRESTAZIONI_COMPRESE ≠ 'SOLA ESECUZIONE'. Oppure variabile ordinale (1/2/3).

---

#### [R-08] / [R-09] / [R-10] CIG_PROG_ESTERNA / DATA_INCARICO_PROG / DATA_CONS_PROG
**NA**: 97–98% | **Azione**: TRANSFORM → `flag_progettazione_esterna` + `lag_prog_esterna_gg`

Quando non null, la progettazione è affidata a un professionista esterno. La progettazione esterna introduce un attore aggiuntivo nella catena corruttiva: il progettista può avere rapporti con l'aggiudicatario e scrivere capitolati tecnici su misura (bando "cucito"). `DATA_CONS_PROG − DATA_INCARICO_PROG` = durata della progettazione: tempi brevissimi su progetti complessi indicano che il progetto era pronto prima dell'incarico formale.

**Proposta**: `flag_progettazione_esterna = 1` se CIG_PROG_ESTERNA not null e ≠ '0000000000'. Calcolare `lag_prog_esterna_gg`.

---

#### [R-11] / [R-12] COD_MODO_RIAGGIUDICAZIONE / MODO_RIAGGIUDICAZIONE
**NA**: 99.9% | **Azione**: TRANSFORM → `flag_riaggiudicazione`

La riaggiudicazione avviene dopo che il contratto con il primo aggiudicatario è caduto per revoca, risoluzione o inadempimento. Evento rarissimo (0.1% dei CIG) ma molto informativo: le sentenze nei positivi spesso riguardano contratti con storie di riaggiudicazione. La sola presenza dell'evento è il segnale; il codice specifico della causa è secondario.

**Proposta**: `flag_riaggiudicazione = 1` se COD_MODO_RIAGGIUDICAZIONE not null.

---

#### Note variabili AGGIUDICAZIONI senza R-code

- **esito** (0% NA, 11 valori): Oltre ad AGGIUDICATA ci sono 'NON AGGIUDICATA', 'ANNULLATA/REVOCATA DOPO APERTURA BUSTE', 'DESERTA'. Queste anomalie meritano `flag_esito_anomalo = esito ≠ 'AGGIUDICATA'`.
- **numero_offerte_ammesse / num_imprese_offerenti / invitati / richiedenti** (47–50% NA): NA = affidamento diretto, nessuna gara. Calcolare `tasso_esclusione = offerte_escluse / offerenti` e `tasso_partecipazione = offerenti / invitati`. Alta esclusione in gare negoziate = possibile pre-selezione dei "perdenti".
- **ribasso_aggiudicazione** (1% NA): Feature principale. Ribasso ≤ 0 su gare non-affidamento-diretto è anomalo → `flag_ribasso_zero`. Calcolare `ribasso_vs_mediana_cpv` (scostamento dal ribasso mediano per stessa categoria CPV e tipo procedura).
- **asta_elettronica** (10% NA, bool): Le aste elettroniche impediscono accordi informali al tavolo → `flag_asta_elettronica`.
- **FLAG_PROC_ACCELERATA** (14% NA, bool): Procedura d'urgenza. Urgenza non giustificata = meccanismo classico per bypassare controlli. Già bool, tenere direttamente.
- **N_MANIF_INTERESSE** (49% NA): Solo per procedure negoziate. Basso numero = bando che scoraggiava la partecipazione, o aggiudicatario già noto.
- **FLAG_SCOMPUTO** (36% NA, bool): Scomputo manodopera dal ribasso. Rilevante per lavori.

---

### BANDO CIG *(~2M CIG/anno nel periodo completo — anagrafica dell'appalto)*

---

#### [R-19] cig_accordo_quadro
**NA**: 75.8% | **Tipo**: id | **Azione**: TRANSFORM → `flag_accordo_quadro`

Quando non null, questo CIG è un appalto specifico discendente da un accordo quadro pre-esistente. L'accordo quadro è uno strumento lecito di aggregazione della domanda, ma crea rischi specifici: il fornitore è già stato selezionato in una gara "madre" e i singoli appalti discendenti possono avere zero competizione, specialmente nella modalità "senza successivo confronto competitivo". Volumi elevati possono essere concentrati su un unico fornitore senza gara per singolo acquisto. Il campo si sovrappone con `modalita_realizzazione` ('CONTRATTO D'APPALTO DISCENDENTE DA AQ SENZA CONFRONTO COMPETITIVO'): usare il campo più popolato tra i due.

**Proposta**: `flag_accordo_quadro = 1` se cig_accordo_quadro not null.

---

#### [R-20] data_pubblicazione
**NA**: 0% | **Tipo**: date | **Azione**: TRANSFORM (anchor per lag)

Anchor temporale dell'appalto. Non ha segnale diretto, ma è il punto di riferimento per quasi tutti i lag del modello: `lag_bando_aggiudicazione`, `finestra_offerta`, `lag_guri`. Usare anche come feature grezza: anno e trimestre di pubblicazione catturano trend normativi (pre/post D.Lgs. 50/2016, pre/post D.Lgs. 36/2023) e cicli politici/elettorali.

---

#### [R-21] data_scadenza_offerta
**NA**: 1.5% | **Tipo**: date | **Azione**: TRANSFORM → `finestra_offerta_gg`

`finestra_offerta_gg = data_scadenza_offerta − data_pubblicazione`. Una finestra breve ostacola i concorrenti non "avvisati" in anticipo. La normativa UE fissa minimali (30–35 giorni per procedure aperte sopra soglia); finestre molto più corte dove non strettamente obbligatorio possono essere barriere intenzionali all'entrata. Correlato con `tipo_scelta_contraente`: ha senso solo per procedure competitive.

---

#### [R-22] DATA_ULTIMO_PERFEZIONAMENTO
**NA**: 0% | **Tipo**: date | **Azione**: TRANSFORM → `lag_perfezionamento_gg`

`lag_perfezionamento_gg = DATA_ULTIMO_PERFEZIONAMENTO − data_pubblicazione`. Misura quanto tardi il bando è stato modificato dopo la pubblicazione. Modifiche tardive (es. cambio requisiti a pochi giorni dalla scadenza) avvantaggiano chi era già informato della modifica in anticipo.

---

#### [R-23] / [R-24] COD_MODALITA_INDIZIONE_SPECIALI / MODALITA_INDIZIONE_SPECIALI
**NA**: 99.9% | **Tipo**: cat | **Azione**: TRANSFORM → categoriale 4 classi rischio

Per i settori speciali (utilities: acqua, gas, energia, trasporti). Aggregare in 4 classi: 1=aperta, 2=ristretta, 3=negoziata con bando, 4=negoziata senza bando. Progressione 1→4 = rischio corruzione crescente. MODALITA_INDIZIONE_SPECIALI (testo) → DROP.

---

#### [R-25] / [R-26] COD_MODALITA_INDIZIONE_SERVIZI / MODALITA_INDIZIONE_SERVIZI
**NA**: 100% nel campione | **Azione**: TRANSFORM → categoriale 4 classi (stessa logica di R-23)

Verificare copertura nel dataset completo. MODALITA_INDIZIONE_SERVIZI (testo) → DROP.

---

#### [R-27] DURATA_PREVISTA
**NA**: 96.4% | **Tipo**: num (giorni) | **Azione**: TRANSFORM → numerico continuo (log1p se skewed)

Durata pianificata in giorni al bando — stima ex ante. **Non binarizzare**: la distribuzione continua è informativamente più ricca. Applicare `log1p` se asimmetrica. Confrontare con `durata_pianificata_gg` da AVVIO CONTRATTO per misurare le rinegoziazioni post-aggiudicazione. 96.4% NA nel campione: verificare copertura nel dataset completo.

---

#### [R-28] / [R-29] COD_STRUMENTO_SVOLGIMENTO / STRUMENTO_SVOLGIMENTO
**NA**: 97.6% | **Tipo**: cat | **Azione**: TRANSFORM (due opzioni da testare) — DROP testo

Strumento di gara: MePA, Consip, Accordo Quadro, gara telematica, ecc.

Profilo di rischio per classe:
- **MePA / Consip**: catalogo standardizzato → SA con minima discrezionalità. Effetto protettivo.
- **Accordo Quadro**: SA vincolata a un fornitore in anticipo → zero competizione sui contratti applicativi. Rischio alto.
- **Gara telematica / SATER**: piattaforma digitale → alta tracciabilità. Rischio basso.

Due opzioni da testare in cross-validation (vedi § 0.6). STRUMENTO_SVOLGIMENTO (testo) → DROP.

---

#### [R-30] FUNZIONI_DELEGATE
**NA**: 99.5% | **Tipo**: cat | **Azione**: DROP

99.5% NA. Il tipo di funzione delegata è già catturato da FLAG_DELEGA. Drop.

---

#### [R-31] / [R-32] CF_SA_DELEGATA / DENOMINAZIONE_SA_DELEGATA
**NA**: 99.2% | **Azione**: DROP entrambe

CF_SA_DELEGATA → DROP (99.2% NA). DENOMINAZIONE_SA_DELEGATA → DROP (testo + 99.2% NA). La delega è già catturata da FLAG_DELEGA (r91) e CF_SA_DELEGANTE (KEEP).

---

#### [R-33] IMPORTO_SICUREZZA
**NA**: 66.5% | **Tipo**: num | **Azione**: KEEP → calcolare `importo_sicurezza_pct`

Costi per la sicurezza dei lavoratori (D.Lgs. 81/2008). Obbligatori per lavori, assenti o nulli per servizi e forniture. **Segnale critico per lavori**: `importo_sicurezza_pct = IMPORTO_SICUREZZA / importo_lotto`. Dove oggetto = LAVORI e sicurezza_pct < 1%, c'è una sottostima intenzionale dei costi di sicurezza: meccanismo per gonfiare il ribasso nominale mantenendo i margini reali (l'impresa non risparmia davvero sul prezzo, risparmia sulla sicurezza).

---

#### [R-34] TIPO_APPALTO_RISERVATO
**NA**: 87.6% | **Tipo**: cat | **Azione**: TRANSFORM → `flag_appalto_riservato`

Tre valori: 'LA PARTECIPAZIONE NON E RISERVATA' (default), riservato a laboratori protetti/cooperative B, riservato a organizzazioni di pubblica utilità. `flag_appalto_riservato = 1` se ≠ non riservato. Gli appalti riservati hanno per definizione meno competizione — non è corruzione di per sé, ma è un contesto a competizione ridotta che può mascherare favoritismi.

---

#### [R-35] CUI_PROGRAMMA
**NA**: 97.4% | **Tipo**: text | **Azione**: DROP

Codice Unico di Intervento del programma triennale. Testo libero non strutturato, nessun valore ML diretto. DROP. (Nota: la *presenza* del CUI indica che l'appalto è programmato nel piano triennale; `flag_ha_cui` potrebbe valere i pochi byte.)

---

#### [R-36] / [R-37] COD_IPOTESI_COLLEGAMENTO / IPOTESI_COLLEGAMENTO
**NA**: 88.9% | **Tipo**: cat | **Azione**: TRANSFORM → `flag_gara_deserta_precedente` + `flag_ripetizione_lavori`

Valori rilevanti:
- *Procedura a seguito di gara annullata o deserta*: il CIG precedente non ha prodotto aggiudicazione. Meccanismo noto: un primo bando viene costruito per fallire (requisiti impossibili o bando poco visibile), il secondo — meno competitivo — va al soggetto desiderato. `flag_gara_deserta_precedente = 1`.
- *Ripetizione di lavori o servizi analoghi* (art. 63 co. 5): permette di affidare senza gara a chi ha già eseguito lavori analoghi per la stessa SA. `flag_ripetizione_lavori = 1`.

---

#### [R-38] CIG_COLLEGAMENTO
**NA**: 100% con rarissimi esempi storici | **Azione**: DROP o `flag_ha_cig_collegamento`

Il CIG del bando precedente collegato. Copertura insufficiente nel range 2019–2025 per uso diretto. DROP, o al limite bool `flag_ha_cig_collegamento`.

---

#### [R-39] DATA_COMUNICAZIONE_ESITO (BANDO CIG)
**NA**: 10.5% | **Tipo**: date | **Azione**: TRANSFORM

Presente anche in AGGIUDICAZIONI (R-03). Dove manca in AGGIUDICAZIONI può essere recuperata qui. `lag_pubblicazione_esito_gg = DATA_COMUNICAZIONE_ESITO − data_pubblicazione`.

---

#### Note variabili BANDO CIG senza R-code

- **tipo_scelta_contraente** (0% NA, 21 valori): La variabile più importante del dataset. Definisce il regime competitivo: AFFIDAMENTO DIRETTO (< 40k, nessuna gara), PROCEDURA NEGOZIATA SENZA PUBBLICAZIONE (SA sceglie chi invitare — massimo rischio), PROCEDURA APERTA (massima competizione). In Italia ~70% dei CIG è affidamento diretto. Trattare come ordinale rispetto al livello di competitività indotto.
- **importo_lotto** (0% NA): Feature numerica principale. Calcolare `log(importo_lotto)` per normalizzare la distribuzione asimmetrica. Verificare se si posiziona appena sotto le soglie normative: `flag_soglia_40k = 35k < importo < 42k`, `flag_soglia_150k = 140k < importo < 155k` — segnale di parcellizzazione intenzionale per abbassare il livello di procedura obbligatorio.
- **n_lotti_componenti**: `flag_multilotto = n_lotti > 1`. Gare con molti lotti possono suddividere un grande contratto in parti più gestibili ma anche ripartire i "diritti" di aggiudicazione tra imprese in accordo.
- **settore** (SETTORI ORDINARI / SPECIALI): `flag_settori_speciali`. Le utility (acqua, gas, energia, trasporti) hanno regole diverse (D.Lgs. 50/2016 Parte III) e spesso procedure meno aperte.
- **FLAG_URGENZA + MOTIVO_URGENZA** (0% NA): MOTIVO_URGENZA distingue l'urgenza reale (calamità, COVID) da quella "costruita" per bypassa re le procedure ordinarie. Già bool FLAG_URGENZA.
- **FLAG_PNRR_PNC** (0% NA, bool): Appalti PNRR hanno volumi enormi e tempi compressi. Feature binaria rilevante come moderatore.
- **modalita_realizzazione** (12 valori): Include 'CONTRATTO D'APPALTO DISCENDENTE DA AQ SENZA CONFRONTO COMPETITIVO' — caso più opaco dell'accordo quadro. Complementa `flag_accordo_quadro`.
- **FLAG_PREV_RIPETIZIONI** (97.6% NA): La clausola di ripetizione permette di affidare senza gara lavori analoghi allo stesso soggetto. Segnale concettualmente rilevante; verificare copertura nel dataset completo.
- **luogo_istat / provincia**: Feature geografica. Alcune province hanno storicamente tassi più alti di irregolarità. Usare come categorica.

---

### AVVIO CONTRATTO *(1.9M righe — date di avvio esecuzione)*

---

#### [R-13] data_stipula_contratto
**NA**: 38% | **Tipo**: date | **Azione**: TRANSFORM → `lag_aggiudicazione_stipula_gg`

`lag_aggiudicazione_stipula_gg = data_stipula − data_aggiudicazione_definitiva`. Misura il tempo tra la decisione formale e la firma del contratto. Un lag molto breve (1–2 giorni) su contratti che per legge richiedono uno stand-still period (10–15 giorni, Direttiva Ricorsi 2007/66/CE) indica che il contratto era già pronto prima dell'aggiudicazione formale — possibile accordo preventivo. Lag molto lungo (mesi) indica contenziosi o blocchi burocratici.

---

#### [R-14] data_esecutivita_contratto
**NA**: 79.1% | **Tipo**: date | **Azione**: TRANSFORM → `lag_esecutivita_stipula_gg`

Il contratto diventa "esecutivo" dopo i controlli di legittimità (visto Corte dei Conti, approvazione dell'organo deliberante). `lag_esecutivita_stipula_gg = data_esecutivita − data_stipula`. Lag molto breve su contratti grandi = controlli saltati o formalizzati a posteriori. Lag molto lungo = blocchi burocratici o contenziosi.

---

#### [R-15] data_termine_contrattuale
**NA**: 16.4% | **Tipo**: date | **Azione**: TRANSFORM → `durata_pianificata_gg`

`durata_pianificata_gg = data_termine − data_inizio_effettiva`. Base per il calcolo dell'overrun. Confrontare con `DURATA_PREVISTA` (R-27) dal bando: se la durata contrattuale supera significativamente quella di bando, c'è stata una rinegoziazione post-aggiudicazione.

---

#### [R-16] data_verbale_consegna_definitiva
**NA**: 23.8% | **Tipo**: date | **Azione**: TRANSFORM → `lag_consegna_inizio_gg`

`lag_consegna_inizio_gg = data_verbale_consegna_definitiva − data_inizio_effettiva`. La consegna del cantiere dovrebbe avvenire prima o contestualmente all'inizio: un lag positivo (consegna dopo l'avvio dichiarato) è un'irregolarità procedurale o indica ritardi della SA nel liberare il sito.

---

#### [R-17] data_inizio_effettiva
**NA**: 28.1% | **Tipo**: date | **Azione**: TRANSFORM → `lag_stipula_inizio_gg`

`lag_stipula_inizio_gg = data_inizio_effettiva − data_stipula_contratto`. Ritardi nell'avvio sono spesso collegati a mancanza del progetto esecutivo approvato o di permessi — elementi che in teoria dovrebbero essere pronti prima dell'aggiudicazione.

---

#### [R-18] data_verbale_prima_consegna
**NA**: 87.3% | **Tipo**: date | **Azione**: TRANSFORM (solo se `consegna_frazionata=1`)

Prima consegna parziale del cantiere. `lag_prima_seconda_consegna_gg = data_verbale_consegna_definitiva − data_verbale_prima_consegna`.

---

#### [R-89] DATA_APPR_PROG_ESE
**NA**: 93.6% | **Tipo**: date | **Azione**: TRANSFORM → `lag_prog_ese_aggiudicazione_gg`

Approvazione del progetto esecutivo da parte della SA. Per grandi opere questo dovrebbe avvenire prima dell'inizio lavori. `lag_prog_ese_aggiudicazione_gg = DATA_APPR_PROG_ESE − data_aggiudicazione_definitiva`. Se il lag è negativo rispetto a `data_inizio_effettiva` (progetto approvato dopo l'avvio dei lavori): si lavora senza progetto definitivo — irregolarità documentabile e segnale di fretta sospetta.

---

#### Note AVVIO CONTRATTO senza R-code

- **consegna_sotto_riserva** (15% NA, bool): Segnale forte. L'appaltatore accetta il cantiere "sotto riserva" = protesta formalmente fin dal primo giorno. Tattica per prepararsi a richiedere varianti e compensi extra. `flag_consegna_riserva`.
- **consegna_frazionata** (15% NA, bool): Normale per grandi opere, introduce complessità nel monitoraggio.

---

### FINE CONTRATTO *(1.07M righe — ⚠️ DROP COMPLETO)*

> **Decisione definitiva: tutto il dataset FINE CONTRATTO va droppato.**
>
> Questo dataset è stato usato parzialmente per costruire i label positivi (il campo `motivo_risoluzione = 'REATI ACCERTATI'` contribuisce all'identificazione dei casi corrotti). Usare qualsiasi variabile da questo dataset come feature predittiva crea leakage circolare: il modello imparebbe a predire la corruzione usando informazioni che provengono dalla stessa fonte che ha generato i label.
>
> Anche le feature apparentemente innocue (`data_effettiva_ultimazione` per il time overrun, `flag_risoluzione_anticipata` per interruzioni non-criminali) vengono droppate in via precauzionale: la dipendenza tra dataset-label e dataset-feature è troppo stretta per distinguere in modo affidabile le porzioni sicure.
>
> **Nota di archivio** — le feature che avrebbero potuto essere derivate da questo dataset se non ci fosse il problema del leakage:

| Feature proposta | Calcolo | Perché era utile |
|---|---|---|
| `overrun_gg` | data_effettiva_ultimazione − data_termine | Time overrun (segnale forte in letteratura) |
| `pct_overrun` | overrun_gg / durata_pianificata | Normalizzazione per durata contratto |
| `flag_risoluzione_anticipata` | cod_motivo_risoluzione non null (escluso REATI) | Contratto problematico non criminale |
| `flag_interruzione_anticipata` | cod_motivo_interruzione non null | Recesso SA o appaltatore |

Se in future iterazioni si decide di separare nettamente la porzione dei label costruita su FINE CONTRATTO da quella costruita su sentenze TAR, alcune di queste feature potrebbero essere recuperate.

---

### COLLAUDO *(648k righe)*

Il dataset COLLAUDO è **ex post** rispetto all'aggiudicazione. In un modello prospettico, le variabili di processo (date di collaudo, esito) sono leakage. I tre campi di conflittualità contrattuale (riserve, contenzioso) vengono invece mantenuti perché misurano fenomeni strutturali del rapporto SA-appaltatore.

---

#### [R-43] cig
KEEP (chiave join). Solo ~34% dei CIG di aggiudicazioni ha un record di collaudo. `flag_nessun_collaudo = 1` se CIG assente dal dataset — segnale in sé (collaudo non registrato o non avvenuto).

---

#### [R-44] data_delibera — **DROP**
>96% NA. Eliminata.

#### [R-45] data_cert_collaudo — **DROP (leakage prospettico)**
Attestazione di collaudo: avviene dopo la consegna dell'opera. Ex post, leakage.

#### [R-46] esito_collaudo — **DROP (leakage prospettico)**
Esito del collaudo: ex post, leakage.

#### [R-47]/[R-48]/[R-49]/[R-50] data_inizio_oper / data_regolare_esec / data_nomina_coll / data_collaudo_stat — **DROP (leakage prospettico)**
Tutte date ex post. Eliminate.

---

#### [R-51] id_aggiudicazione
KEEP (join).

---

#### [R-52] / [R-53] RISERVE_AVANZATE / RISERVE_DEFINITE
**NA**: 10.9% | **Tipo**: num | **Azione**: TRANSFORM

> NA = 0 riserve, non dato mancante. Imputare NA = 0.

Le riserve sono contestazioni formali dell'appaltatore sui compensi dovuti. Il meccanismo corruttivo: l'appaltatore avanza riserve gonfiandone l'ammontare sapendo che la SA è disposta ad accettarle (compenso informale concordato in anticipo).

**Feature derivate**:
- `flag_riserve` = 1 se RISERVE_AVANZATE > 0
- `ratio_riserve_su_importo` = RISERVE_AVANZATE / importo_aggiudicazione
- `ratio_riserve_definite_su_avanzate` = RISERVE_DEFINITE / RISERVE_AVANZATE — misura la "vittoria" dell'appaltatore nelle contestazioni. Se > 1 è anomalo (ammesse più riserve di quelle formalmente avanzate = errore dati o accordo opaco).

---

#### [R-54] IMPORTO_CONTENZ_RISOLTO
**NA**: 10.9% | **Tipo**: num | **Azione**: TRANSFORM

> NA = 0, imputare come tale.

**Feature derivata**: `ratio_contenzioso` = IMPORTO_CONTENZ_RISOLTO / importo_aggiudicazione. Valori elevati indicano contratto molto litigioso o SA molto compiacente. Da interpretare insieme a `ratio_riserve_definite_su_avanzate`.

---

### VARIANTI *(303k righe — modifiche al contratto in corso d'opera)*

---

#### [R-81] id_variante
DROP (id tecnico).

---

#### [R-82] cod_motivo_variante
**NA**: 0% | **Tipo**: cat (26 codici) | **Azione**: TRANSFORM

La variante modifica il contratto dopo l'aggiudicazione. Codici rilevanti:
- "Cause impreviste e imprevedibili" (art. 106 D.Lgs. 50/2016): Lecito, ma cronicamente abusato. Se il 90% delle varianti di un CIG è "imprevedibile", il progetto originale era lacunoso.
- "Lavori supplementari non inclusi nell'appalto iniziale": Gonfiamento post-aggiudicazione — si vince con prezzo basso, poi si recupera con extra lavori.
- "Sopravvenute disposizioni legislative": Generalmente legittimo.

Il numero di varianti per CIG è uno dei segnali più citati in letteratura (Olken, 2007; Golden & Picci, 2005).

**Feature derivate** (aggregare per CIG):
- `n_varianti` = numero totale varianti
- `pct_overrun_variante` = sum(importo_variante) / importo_aggiudicazione ← **feature centrale**
- `flag_variante_imprevista` = 1 se cod_motivo ∈ {cause impreviste, lavori supplementari}
- `flag_variante_precoce` = 1 se prima variante entro 30gg da data_inizio_effettiva

---

#### [R-83] motivo_variante
**NA**: 0% | **Azione**: DROP (93k valori unici = testo libero)

`cod_motivo_variante` è sufficiente.

---

#### [R-84] data_approvazione_variante
**NA**: 0.9% | **Tipo**: date | **Azione**: TRANSFORM → lag

`lag_variante_da_aggiudicazione_gg = min(data_approvazione_variante) − data_aggiudicazione`. Una variante approvata pochissimi giorni dopo l'aggiudicazione è molto sospetta: il progetto era già noto come incompleto, o fu redatto intenzionalmente lacunoso per permettere successivi arricchimenti dell'aggiudicatario.

---

#### [R-85] / [R-86] cig / id_aggiudicazione
KEEP (chiavi di join).

---

#### [R-87] CIG_PROROGA
**NA**: 99.4% | **Azione**: DROP (testo libero, copertura insufficiente)

---

#### [R-88] DATA_ATTO_AGGIUNTIVO
**NA**: 49.6% | **Tipo**: date | **Azione**: ASK (leakage)

`lag_atto_su_variante_gg = DATA_ATTO_AGGIUNTIVO − data_approvazione_variante`. Lag brevissimo = l'accordo di modifica era già pronto prima dell'approvazione formale.

---

### SOSPENSIONI *(385k righe — interruzioni dei lavori)*

---

#### [R-71] / [R-72] data_sospensione / data_ripresa
**NA**: 0% / 7.7% | **Tipo**: date | **Azione**: TRANSFORM

`durata_sospensione_gg = data_ripresa − data_sospensione`. NA su data_ripresa = sospensione ancora in corso. Aggregare per CIG: `n_sospensioni`, `durata_sospensioni_totale_gg`, `pct_tempo_sospeso = durata_sospensioni_totale / durata_pianificata_gg`.

**Motivo sospensione** (solo 7 valori — quasi una categorica):
- *'REDAZIONE DI VARIANTI IN CORSO DI ESECUZIONE'*: si ferma il cantiere per preparare una variante = meccanismo per bloccare e rinegoziare. `flag_sospensione_per_variante = 1`.
- *'SOSPENSIONE DISPOSTA DALL'AUTORITÀ GIUDIZIARIA'*: cantiere fermato dalla magistratura = indagini in corso. `flag_sospensione_ag = 1` — segnale fortissimo.
- *'MANCANZA DI PROVVEDIMENTI AUTORIZZATIVI'*: i permessi non erano pronti all'avvio. `flag_sospensione_no_autorizzativi = 1`.
- *'INTERFERENZE DI NATURA TECNICA'*: progettazione carente.
- *'AVVERSE CONDIZIONI CLIMATICHE'*: lecito ma può essere gonfiato.

---

### STATI DI AVANZAMENTO / SAL *(1.84M righe — pagamenti progressivi)*

---

#### [R-73] data_emissione_sal
**NA**: 0% | **Tipo**: date | **Azione**: TRANSFORM

Aggregare per CIG: `n_sal`, `data_primo_sal`, `data_ultimo_sal`, `avg_giorni_tra_sal = (data_ultimo − data_primo) / (n_sal − 1)`. Cadenza irregolare dei SAL (troppo ravvicinati o troppo radi rispetto alla durata contrattuale) può indicare anomalie nella gestione della liquidità.

---

#### [R-74] DATA_CERT_PAGAMENTO
**NA**: 40.2% | **Tipo**: date | **Azione**: TRANSFORM → `lag_cert_pagamento_gg`

`lag_cert_pagamento_gg = DATA_CERT_PAGAMENTO − data_emissione_sal`. Il Codice Appalti prevede certificazione entro 7 giorni dall'emissione del SAL. Ritardi sistematici indicano problemi di liquidità della SA; anticipi anomali rispetto alla norma possono indicare trattamento preferenziale dell'impresa.

---

#### Note SAL senza R-code

- **flag_ritardo** (0% NA, 3 valori IN LINEA/IN RITARDO/IN ANTICIPO): `pct_sal_in_ritardo = n_sal_ritardo / n_sal_totali`. Più del 50% in ritardo = contratto sistematicamente problematico.
- **n_giorni_scostamento** (12% NA): Scostamento in giorni dal programma lavori. Media e massimo per CIG.
- **GIORNI_PROROGA** (12.4% NA — **non 100% NA!**): Proroghe concesse per singolo SAL. `giorni_proroga_tot = sum(GIORNI_PROROGA)` per CIG — questo è il campo corretto (quello in FINE CONTRATTO è 100% NA).
- **importo_sal + progressivo_sal**: `n_sal_totali = max(progressivo_sal)`. `avg_importo_per_sal = importo_aggiudicazione / n_sal`. `delta_sal_cert = importo_sal − IMPORTO_CERT_PAGAMENTO` = differenza tra lavori dichiarati e importo certificato (trattenute, rivalse).

---

### QUADRO ECONOMICO *(4.58M righe — valori economici per evento)*

---

#### [R-68] data
**NA**: 94.5% | **Tipo**: date | **Azione**: TRANSFORM

Presente solo per eventi VARIANTE e CONSUNTIVO. `lag_consuntivo_ultimazione_gg = data(CONSUNTIVO) − data_effettiva_ultimazione`.

---

#### [R-69] importo_progettazione
**NA**: 65.5% | **Tipo**: num | **Azione**: KEEP → calcolare `pct_progettazione`

Solo per lavori con progettazione. `pct_progettazione = importo_progettazione / (importo_lavori + importo_progettazione)`. Proporzioni anomalmente alte della progettazione rispetto all'esecuzione possono indicare gonfiamento dei costi professionali (più difficili da verificare rispetto ai materiali).

---

#### [R-70] ulteriori_oneri_non_soggetti_ribasso
**NA**: 80.1% | **Tipo**: num | **Azione**: KEEP → calcolare `pct_oneri_non_ribasso`

`pct_oneri_non_ribasso = ulteriori_oneri / importo_totale`. Più alta è questa percentuale, meno del contratto è soggetto alla competizione di prezzo: l'impresa non risparmia davvero, il ribasso è cosmetico.

---

#### Feature chiave dal QUADRO ECONOMICO

Confronto BASE_ASTA vs CONSUNTIVO per stesso CIG:
- `delta_importo_lavori_pct = (CONSUNTIVO.importo_lavori − BASE_ASTA.importo_lavori) / BASE_ASTA.importo_lavori`
- `delta_importo_totale_pct = (CONSUNTIVO totale − BASE_ASTA totale) / BASE_ASTA totale`

Questi delta misurano il cost overrun economico — tra i segnali più forti e documentati in letteratura sulla corruzione negli appalti (Olken, 2007; Fazekas & Tóth, 2016).

---

### SUBAPPALTI *(330k righe)*

---

#### [R-77] cf_subappaltante
**NA**: 66.1% | **Azione**: DROP come feature diretta

Non usabile come feature ML diretta, ma utile per costruire feature di rete: `flag_subappalto_circolare = 1` se il subappaltatore appare come aggiudicatario diretto di altri CIG con la stessa SA (possibile circolazione fittizia di risorse).

---

#### [R-78] data_autorizzazione
**NA**: 6.6% | **Tipo**: date | **Azione**: TRANSFORM → `lag_autorizzazione_subappalto_gg`

`lag = data_autorizzazione − data_stipula_contratto`. Se negativo = subappalto "autorizzato" prima della firma del contratto principale (anomalia). Se molto positivo = lavori iniziati senza autorizzazione, poi regolarizzati. Aggregare: `n_subappalti`.

---

#### [R-79] / [R-80] cod_cpv / descrizione_cpv (SUBAPPALTI)
**NA**: 0.5% | **Azione**: TRANSFORM (grouping CPV)

Il CPV del subappalto. Confrontare con il CPV del contratto principale: `flag_cpv_divergente = cpv2(subappalto) ≠ cpv2(contratto_principale)`. Una forte divergenza (es. contratto di servizi con subappalto di costruzioni) può indicare utilizzo improprio della struttura contrattuale.

---

### STAZIONE APPALTANTE *(46k SA — anagrafica committenti)*

---

#### [R-75] data_inizio
**NA**: 0% | **Tipo**: date | **Azione**: TRANSFORM → `eta_sa_anni`

`eta_sa_anni = data_aggiudicazione − data_inizio`. SA molto giovani (< 1 anno dalla fondazione) che aggiudicano contratti grandi sono anomalie statistiche. SA cessate che continuano ad aggiudicare sono irregolarità formali: `flag_sa_cessata = stato = 'CESSATO'`.

---

#### [R-76] data_fine
**NA**: 0% | **Azione**: DROP

Tutti i valori sono '2099-12-31' (placeholder per "nessuna data di fine"). Cardinality = 1 con valore placeholder. DROP.

---

#### Note STAZIONE APPALTANTE senza R-code

- **natura_giuridica_codice** (102 valori): Raggruppare in ~8 macro: PA centrale, comune/provincia/regione, ASL/ospedale, università, ente pubblico economico, società mista/controllata, altro. Le società a controllo pubblico hanno spesso governance più opaca rispetto alle PA tradizionali.
- **flag_inHouse / flag_partecipata** (card=1 nel campione Gen 2025): Tutti False nel campione mensile. **Non droppare**: nel dataset completo questi flag sono certamente popolati (in-house companies esistono). La loro importanza è alta: appalti a società in-house possono essere usati per aggirare le norme sulla concorrenza.
- **soggetto_estero** (card=1, tutti False): Tutte le SA italiane sono non-estere. DROP.
- **provincia_codice**: Feature geografica aggregata. Confrontare con la provincia dell'aggiudicatario per rilevare concentrazioni locali.

---

### AGGIUDICATARI *(5.4M righe)*

---

#### Note AGGIUDICATARI

- **ruolo** (45% NA, 19 valori): NA = monosoggettivo. MANDATARIA/MANDANTE = ATI. Le ATI sono spesso usate per suddividere i "diritti" di aggiudicazione tra imprese in accordo di alternanza. `flag_ati = 1` se ruolo IN (MANDATARIA, MANDANTE, CONSORZIATA).
- **tipo_soggetto** (18 valori): Raggruppare in: singola / ATI-RTI / consorzio stabile / altri.
- **denominazione**: Il valore 'IMPRESA INESISTENTE' appare nel dataset — utile per quality check, non come feature.
- Aggregare per CIG: `n_aggiudicatari`, `flag_ati`, `flag_consorzio`.

---

### PARTECIPANTI *(copertura ~128k CIG unici)*

---

#### Note PARTECIPANTI

- Copertura molto parziale (128k vs 4.9M aggiudicazioni). Usare solo dove disponibile.
- Aggregare per CIG: `n_partecipanti`, `pct_ati_partecipanti`.
- Il partecipante perdente che poi vince altri CIG con la stessa SA = possibile accordo di alternanza. Richiede network analysis.
- `tipo_soggetto` ha 82 valori unici (vs 18 in aggiudicatari): normalizzare alle stesse categorie.

---

### FONTI DI FINANZIAMENTO *(200k righe)*

---

#### Note FONTI DI FINANZIAMENTO

Tutti i campi sono importi in euro (0 quando non applicabile, 0% NA). Feature chiave:
- `totale_fondi = sum(tutti i campi)`
- `pct_fondi_eu = entrate_comunitarie / totale_fondi`: esposizione a fondi UE (più controllo ma anche più pressione a spendere).
- `pct_fondi_privati = (apporto_capitali_privati + entrate_vincolate_privati) / totale_fondi`
- `pct_fondi_bilancio_sa = fondi_bilancio_sa / totale_fondi`
- `flag_trasferimento_immobili = trasferimento_immobili > 0` (finanziamento tramite vendita di beni pubblici — meccanismo specifico da monitorare)

---

### CATEGORIE DPCM AGGREGAZIONE

---

#### Note CATEGORIE DPCM

- `cod_categoria_merceologica_dpcm_aggregazione` (26 valori): Categorie soggette a obbligo di acquisto centralizzato (Consip, ARCA, ecc.).
- `cod_deroga_soggetto_aggregatore` (7 valori, 15% NA): Motivazione della deroga all'obbligo. Deroga senza giustificazione adeguata è un'irregolarità procedurale: `flag_deroga_aggregazione = cod_deroga not null`.

---

### INDICATORI PNRR/PNC

---

#### [R-62] quota_femminile
**NA**: 94.7% | **Azione**: KEEP

Percentuale di lavoratrici richiesta contrattualmente (solo PNRR). 0% = deroga applicata. Rilevante solo combinata con `FLAG_PNRR_PNC`. `flag_quota_genere_rispettata = flag_quote = 'S'`.

---

#### [R-63] quota_giovanile
**NA**: 95.2% | **Azione**: ASK

Simile a R-62. Copertura bassa anche tra i soli CIG PNRR.

---

#### [R-64] / [R-65] cod_mot_deroga / mot_deroga
**NA**: 79.6% | **Azione**: KEEP

Motivo per cui le quote non si applicano. `flag_deroga_pnrr_generica = mot_deroga = 'ALTRO'` (deroga senza motivazione specifica).

---

### ATTESTAZIONI SOA *(500k+ righe — qualificazioni imprese di costruzione)*

---

#### [R-92] data_autorizzazione (della SOA)
Caratteristica dell'ente SOA, non dell'impresa. Poco rilevante come feature per il singolo contratto.

---

#### [R-93] / [R-94] / [R-95] / [R-96] date emissione e scadenza SOA
**Azione**: TRANSFORM → `lag_validita_soa_gg` + `flag_soa_in_scadenza`

`lag_validita_soa_gg = data_scadenza_finale − data_aggiudicazione_definitiva`. Se negativo = l'impresa aveva la SOA scaduta al momento dell'aggiudicazione (irregolarità verificabile). Se < 180 giorni = `flag_soa_in_scadenza` (rischio che scada prima della fine del contratto). Le SOA durano normalmente 5 anni.

---

#### [R-97] alla_data_del
**Azione**: TRANSFORM (gestione temporalità del join)

Data dello snapshot SOA. Usare per filtrare: tenere solo attestazioni con `data_emissione ≤ data_aggiudicazione ≤ data_scadenza_finale`.

---

#### [R-98] enteRilcertQualita
**NA**: 14.5% | **Tipo**: text (8276 valori) | **Azione**: TRANSFORM (grouping)

Raggruppare nei principali enti (Bureau Veritas, RINA, IMQ, DNV, TÜV, Certiquality → top 10; resto → 'altri').

---

#### [R-99] / [R-100] certificazioneDiQualitaScadenza / data_effettuazione_verifica
**Azione**: TRANSFORM → `flag_cert_qualita_scaduta`

`flag_cert_qualita_scaduta = certificazioneDiQualitaScadenza < data_aggiudicazione`. Impresa che vince con certificazione scaduta = irregolarità verificabile e documentabile.

---

#### Note ATTESTAZIONI SOA

- **cod_categoria + classifica** (I–VIII, corrispondente a fasce di importo): La classifica deve corrispondere all'importo del contratto (Classifica I max ~258k€, Classifica VIII > 15M€). `flag_classifica_inadeguata = importo_contratto > importo_massimo_classifica_soa`. Join complesso: cf_impresa × data_aggiudicazione × categoria_opera.
- **fase_attestato**: Filtrare su 'PUBBLICA' per attestazioni correnti.

---

### PUBBLICAZIONI *(date di pubblicazione su gazzette e albi)*

---

#### [R-101] data_creazione
**NA**: 3% | **Tipo**: date | **Azione**: TRANSFORM

Data di creazione del record ANAC. `lag_guri_creazione_gg = data_guri − data_creazione`.

---

#### [R-102] data_albo
**NA**: 99.8% | **Azione**: DROP

Albo Pretorio quasi mai registrato nel range 2019–2025.

---

#### [R-103] data_guri
**NA**: 84.1% | **Tipo**: date | **Azione**: TRANSFORM → `lag_guri_bando_gg`

GURI = Gazzetta Ufficiale della Repubblica Italiana. `lag_guri_bando_gg = data_guri − data_pubblicazione`. Ritardi nella pubblicazione ufficiale dopo il bando possono indicare procedura non regolare. NA = non pubblicato su GURI (normale per importi sotto soglia).

---

#### [R-104] data_guce
**NA**: 98.7% | **Tipo**: date | **Azione**: ASK

GUCE = Gazzetta Ufficiale UE. Obbligatoria sopra soglie EU. `lag_guce_bando_gg = data_guce − data_pubblicazione`. Solo per grandi contratti.

---

#### [R-105] data_bore
**NA**: 100% con rarissimi esempi | **Azione**: DROP

Bollettino Regionale. DROP.

---

#### [R-66] / [R-67] SCADENZA_INVITO / DATA_LETTERA_INVITO
**NA**: 98.3–98.7% | **Tipo**: date | **Azione**: ASK

Solo per procedure negoziate con invito formale. `finestra_risposta_invito_gg = SCADENZA_INVITO − DATA_LETTERA_INVITO`. Finestre molto brevi = barriera de facto per chi non era già "pronto".

---

### MISURE PREMIALI *(4073 CIG)*

La presenza stessa nel dataset è informativa: `flag_misura_premiale = 1` indica che la SA ha previsto incentivi formali per la puntualità (art. 113-bis D.Lgs. 50/2016) — il che implica un rischio percepito di ritardo elevato già al momento del bando.

---

### CUP *(dataset di join CIG ↔ codice investimento)*

Il CUP non è una feature ML ma permette di linkare l'appalto al programma di investimento pubblico. `flag_ha_cup = 1` se CIG presente. Appalti con CUP fanno parte di programmi monitorati su OpenCUP, con maggiore tracciabilità.

---

## PARTE 2 — Feature engineering: proposte sistematiche

### 2.1 Nuovi bool flag (non ancora nell'Excel)

| Flag | Fonte | Logica |
|---|---|---|
| `flag_design_build` | AGGIUDICAZIONI | PRESTAZIONI_COMPRESE ≠ 'SOLA ESECUZIONE' |
| `flag_eprocurement` | BANDO CIG | STRUMENTO_SVOLGIMENTO contains TELEMATICA o CATALOGO |
| `flag_appalto_riservato` | BANDO CIG | TIPO_APPALTO_RISERVATO ≠ 'LA PARTECIPAZIONE NON E RISERVATA' |
| `flag_ripetizione` | BANDO CIG | FLAG_PREV_RIPETIZIONI = 1 |
| `flag_gara_deserta_precedente` | BANDO CIG | IPOTESI_COLLEGAMENTO contains DESERTA o ANNULLATA |
| `flag_esito_anomalo` | AGGIUDICAZIONI | esito ≠ 'AGGIUDICATA' |
| `flag_ribasso_zero` | AGGIUDICAZIONI | ribasso_aggiudicazione ≤ 0 |
| `flag_asta_elettronica` | AGGIUDICAZIONI | asta_elettronica = 1 |
| `flag_ati_aggiudicatario` | AGGIUDICATARI | ruolo IN (MANDATARIA, MANDANTE) o tipo contiene ATI |
| `flag_consegna_riserva` | AVVIO CONTRATTO | consegna_sotto_riserva = 1 |
| `flag_risoluzione_anticipata` | FINE CONTRATTO | cod_motivo_risoluzione not null AND motivo ≠ REATI ACCERTATI |
| `flag_risoluzione_per_reati` | FINE CONTRATTO | motivo = 'REATI ACCERTATI' — *usare come label extra, non feature* |
| `flag_interruzione_anticipata` | FINE CONTRATTO | cod_motivo_interruzione_anticipata not null |
| `flag_collaudo_negativo` | COLLAUDO | esito_collaudo = 'NEGATIVO' |
| `flag_nessun_collaudo` | COLLAUDO | CIG assente dal dataset |
| `flag_riserve` | COLLAUDO | RISERVE_AVANZATE > 0 |
| `flag_variante_precoce` | VARIANTI + AVVIO | prima variante entro 30gg da data_inizio_effettiva |
| `flag_variante_imprevista` | VARIANTI | cod_motivo in {cause impreviste, lavori supplementari} |
| `flag_sospensione_per_variante` | SOSPENSIONI | descrizione_motivo = REDAZIONE DI VARIANTI |
| `flag_sospensione_ag` | SOSPENSIONI | descrizione_motivo = SOSPENSIONE DISPOSTA DALL'A.G. |
| `flag_sospensione_no_autorizzativi` | SOSPENSIONI | descrizione_motivo = MANCANZA DI PROVVEDIMENTI AUTORIZZATIVI |
| `flag_sal_in_ritardo` | SAL | any flag_ritardo = IN RITARDO |
| `flag_sal_con_proroga` | SAL | any GIORNI_PROROGA > 0 |
| `flag_fondi_eu` | FONTI | entrate_comunitarie > 0 |
| `flag_fondi_privati` | FONTI | apporto_capitali_privati > 0 |
| `flag_trasferimento_immobili` | FONTI | trasferimento_immobili > 0 |
| `flag_deroga_aggregazione` | CATEGORIE DPCM | cod_deroga not null |
| `flag_soa_in_scadenza` | SOA | data_scadenza_finale − data_aggiudicazione < 180gg |
| `flag_cert_qualita_scaduta` | SOA | certificazioneDiQualitaScadenza < data_aggiudicazione |
| `flag_classifica_inadeguata` | SOA | importo_contratto > importo_massimo_classifica |
| `flag_ha_cup` | CUP | CIG presente nel dataset |
| `flag_misura_premiale` | MISURE PREMIALI | CIG presente nel dataset |
| `flag_multilotto` | BANDO CIG | n_lotti_componenti > 1 |
| `flag_settori_speciali` | BANDO CIG | settore = SETTORI SPECIALI |
| `flag_soglia_40k` | BANDO CIG | 35k < importo_lotto < 42k |
| `flag_soglia_150k` | BANDO CIG | 140k < importo_lotto < 155k |
| `flag_cpv_divergente_subappalto` | SUBAPPALTI + CIG | cpv2(subappalto) ≠ cpv2(contratto) |
| `flag_deroga_pnrr_generica` | INDICATORI PNRR | mot_deroga = ALTRO |
| `flag_sa_cessata` | STAZ. APPALTANTE | stato = CESSATO |
| `flag_subappalto_circolare` | SUBAPPALTI + AGGIUDICATARI | subappaltante compare come aggiudicatario diretto con stessa SA |

---

### 2.2 Lag e durate proposti

| Feature | Calcolo | Fonti |
|---|---|---|
| `lag_bando_aggiudicazione_gg` | data_aggiudicazione − data_pubblicazione | AGG + CIG |
| `finestra_offerta_gg` | data_scadenza_offerta − data_pubblicazione | CIG |
| `lag_perfezionamento_gg` | DATA_ULTIMO_PERFEZIONAMENTO − data_pubblicazione | CIG |
| `lag_aggiudicazione_stipula_gg` | data_stipula − data_aggiudicazione | AVVIO + AGG |
| `lag_stipula_esecutivita_gg` | data_esecutivita − data_stipula | AVVIO |
| `lag_stipula_inizio_gg` | data_inizio_effettiva − data_stipula | AVVIO |
| `lag_consegna_inizio_gg` | data_verbale_consegna_definitiva − data_inizio_effettiva | AVVIO |
| `lag_prog_esterna_gg` | DATA_CONS_PROG − DATA_INCARICO_PROG | AGG |
| `lag_prog_ese_aggiudicazione_gg` | DATA_APPR_PROG_ESE − data_aggiudicazione | AVVIO + AGG |
| `durata_pianificata_gg` | data_termine − data_inizio_effettiva | AVVIO |
| `overrun_gg` | data_effettiva_ultimazione − data_termine | FINE + AVVIO |
| `pct_overrun` | overrun_gg / durata_pianificata_gg | FINE + AVVIO |
| `lag_conclusione_anticipata_gg` | data_conclusione_anticipata − data_inizio_effettiva | FINE |
| `lag_guri_bando_gg` | data_guri − data_pubblicazione | PUB + CIG |
| `lag_cert_collaudo_ultimazione_gg` | data_cert_collaudo − data_effettiva_ultimazione | COLLAUDO + FINE |
| `lag_nomina_coll_aggiudicazione_gg` | data_nomina_coll − data_aggiudicazione | COLLAUDO + AGG |
| `lag_variante_da_inizio_gg` | min(data_approvazione_variante) − data_inizio_effettiva | VARIANTI + AVVIO |
| `lag_atto_su_variante_gg` | DATA_ATTO_AGGIUNTIVO − data_approvazione_variante | VARIANTI |
| `durata_sospensioni_totale_gg` | sum(data_ripresa − data_sospensione) per CIG | SOSPENSIONI |
| `pct_tempo_sospeso` | durata_sospensioni_totale / durata_pianificata | SOSP + AVVIO |
| `avg_giorni_tra_sal` | (data_ultimo_sal − data_primo_sal) / (n_sal − 1) | SAL |
| `lag_cert_pagamento_gg` | avg(DATA_CERT_PAGAMENTO − data_emissione_sal) | SAL |
| `giorni_proroga_tot` | sum(GIORNI_PROROGA) per CIG | SAL |
| `lag_autorizzazione_subappalto_gg` | data_autorizzazione − data_stipula | SUBAPPALTI + AVVIO |
| `eta_sa_anni` | data_aggiudicazione − data_inizio(SA) | SA + AGG |
| `lag_soa_da_aggiudicazione_gg` | data_scadenza_finale(SOA) − data_aggiudicazione | SOA + AGG |

---

### 2.3 Rapporti e tassi proposti

| Feature | Calcolo | Fonti | Segnale |
|---|---|---|---|
| `ribasso_spread` | massimo_ribasso − minimo_ribasso | AGG | coordinamento offerte |
| `tasso_esclusione` | offerte_escluse / num_imprese_offerenti | AGG | selezione forzata |
| `tasso_partecipazione` | num_imprese_offerenti / num_imprese_invitate | AGG | risposta agli inviti |
| `scostamento_importo_pct` | (importo_aggiudicazione − importo_lotto) / importo_lotto | AGG + CIG | anomalie di prezzo |
| `importo_sicurezza_pct` | IMPORTO_SICUREZZA / importo_lotto | CIG | sicurezza lavori (costruzioni) |
| `pct_oneri_non_ribasso` | ulteriori_oneri / importo_totale | QUAD.EC. | quota non competitiva |
| `delta_importo_lavori_pct` | (CONSUNTIVO − BASE_ASTA) / BASE_ASTA | QUAD.EC. | cost overrun economico |
| `pct_riserve` | IMPORTO_CONTENZ_RISOLTO / importo_aggiudicazione | COLLAUDO | litigiosità contrattuale |
| `riserve_accettate_ratio` | RISERVE_DEFINITE / RISERVE_AVANZATE | COLLAUDO | cedimento SA a pressioni |
| `pct_sal_in_ritardo` | n_sal_ritardo / n_sal_totali | SAL | tasso ritardi esecuzione |
| `n_varianti_per_anno` | n_varianti / (durata_pianificata_gg / 365) | VARIANTI + AVVIO | frequenza modifiche contratto |
| `pct_fondi_eu` | fondi_comunitari / totale_fondi | FONTI | esposizione UE |
| `pct_fondi_privati` | (privati + capitali_privati) / totale_fondi | FONTI | co-finanziamento privato |
| `avg_importo_per_sal` | importo_aggiudicazione / n_sal | AGG + SAL | granularità pagamenti |
| `log_importo_lotto` | log(importo_lotto) | CIG | normalizzazione distribuzione |
| `ribasso_vs_mediana_cpv` | ribasso − median(ribasso per cpv2 + tipo_scelta) | AGG | anomalia rispetto al settore |
| `pct_importo_varianti` | importo_variante_tot / importo_aggiudicazione | QUAD.EC. | peso varianti su contratto |
| `pct_progettazione` | importo_progettazione / importo_totale | QUAD.EC. | peso costi progettuali |
| `n_aggiudicazioni_sa_anno` | count(CIG) per SA per anno | CIG | volume contrattuale SA (feature SA-level) |
| `delta_sal_cert` | avg(importo_sal − IMPORTO_CERT_PAGAMENTO) | SAL | trattenute/rivalse sistematiche |

---

## PARTE 3 — Problemi metodologici

### 3.1 NA complessi (NA ≠ dato mancante)

Queste variabili hanno NA che significa "evento non occorso", non "dato non rilevato". Per queste, imputare NA = 0 anziché usare imputation statistica:

| Variabile | % NA | Significato di NA |
|---|---|---|
| RISERVE_AVANZATE / RISERVE_DEFINITE | 10.9% | 0 riserve avanzate |
| massimo_ribasso / minimo_ribasso | 77–80% | offerta unica, spread incalcolabile |
| numero_offerte_* / num_imprese_* | 47–50% | affidamento diretto, nessuna gara |
| data_esecutivita_contratto | 79% | non registrato per tutti i tipi |
| data_verbale_prima_consegna | 87% | consegna non frazionata |
| GIORNI_PROROGA (SAL) | 12% | nessuna proroga per quel SAL |
| quota_femminile / quota_giovanile | 94–95% | CIG non PNRR |
| data_conclusione_anticipata | 98% | contratto non terminato anticipatamente |
| importo sicurezza | 66% | contratto non di lavori (forniture/servizi) |

---

### 3.2 Leakage temporale

Classificazione rischio per dataset rispetto al momento del bando:

| Dataset | Rischio | Decisione | Note |
|---|---|---|---|
| BANDO CIG | **Basso** | ✅ Usare | Ex ante al bando |
| AGGIUDICAZIONI | Basso-Medio | ✅ Usare | Post-gara, pre-esecuzione |
| AVVIO CONTRATTO | Basso | ✅ Usare | Inizio esecuzione |
| FONTI DI FINANZIAMENTO | **Basso** | ✅ Usare (→ ratio) | Registrate al bando |
| CATEGORIE DPCM | **Basso** | ✅ Usare | Al bando |
| STAZIONE APPALTANTE | **Basso** | ✅ Usare (+ join esterno) | Anagrafica statica |
| ATTESTAZIONI SOA | Basso-Medio | ✅ Usare (time-aware) | Snapshot storico — filtrare per data validità SOA ≥ data_aggiudicazione |
| AGGIUDICATARI / PARTECIPANTI | Basso-Medio | ✅ Aggregare per CIG | Post-gara, pre-esecuzione |
| SOSPENSIONI | Medio | ✅ Usare (aggregati) | Durante esecuzione |
| SAL | Medio | ✅ Usare (aggregati) | Durante esecuzione, cumulativi |
| SUBAPPALTI | Medio | ✅ Usare (aggregati) | Durante esecuzione |
| VARIANTI | **Alto** | ✅ Usare (pct_overrun, n_varianti) | Ex post aggiudicazione, accettato per scoring retrospettivo |
| QUADRO ECONOMICO CONSUNTIVO | **Alto** | ⚠️ Solo feature aggregate | Ex post — usare importi base d'asta, non consuntivi |
| FINE CONTRATTO | **⛔ DROP** | ❌ Drop completo | Usato per costruire label → leakage circolare |
| COLLAUDO (date) | **Alto** | ❌ Drop date | Leakage prospettico |
| COLLAUDO (riserve) | Medio | ✅ Usare (→ ratio) | Segnale contrattuale, accettato |

**Punto di osservazione scelto**: modello prospettico con dati disponibili fino a contratto in esecuzione. Feature ex post (COLLAUDO date, FINE CONTRATTO) sono escluse. Feature aggregate storiche dello stesso appaltatore/SA non sono leakage per il contratto corrente e sono incoraggiate.

---

### 3.3 Aggregazioni 1:N necessarie prima del join

| Dataset | Aggregazioni chiave |
|---|---|
| AGGIUDICATARI | n_aggiudicatari, flag_ati, tipo_soggetto_cat |
| PARTECIPANTI | n_partecipanti, pct_ati_partecipanti |
| VARIANTI | n_varianti, flag_variante_precoce, flag_variante_imprevista, n_varianti_per_anno |
| SOSPENSIONI | n_sospensioni, durata_totale, flag_sospensione_ag, flag_sosp_per_variante |
| SAL | n_sal, pct_sal_ritardo, giorni_proroga_tot, avg_giorni_tra_sal, lag_cert_pagamento_gg |
| SUBAPPALTI | n_subappalti, flag_cpv_divergente, lag_autorizzazione_subappalto_gg |
| LAVORAZIONI | n_lavorazioni |
| CATEGORIE OPERA | n_categorie_opera |
| FONTI DI FINANZIAMENTO | pct_fondi_eu, pct_fondi_privati, flag_trasferimento_immobili |
| QUADRO ECONOMICO | delta_importo_lavori_pct (BASE_ASTA → CONSUNTIVO), pct_oneri_non_ribasso |
| ATTESTAZIONI SOA | classifica_max, flag_soa_in_scadenza, flag_cert_qualita_scaduta |
| PUBBLICAZIONI | flag_ha_guri, lag_guri_bando_gg |

---

### 3.4 SmartCIG

Il 69% dei CIG "normali" nel 2024 ha importo < 40k€. I CIG SmartCIG (contratti diretti < 40k€) non sono nel BDNCP principale. Implicazioni:
- Il modello addestrato sui soli dati ANAC non copre la fascia di valore più bassa.
- Verificare quante etichette positive (labels/cig_condannati.csv) hanno importo < 40k€: se molte, c'è un gap di copertura rilevante.
- Includere SmartCIG aumenterebbe il dataset di ~20M CIG/anno ma richiederebbe gestione di una struttura dati separata.

---

### 3.5 Variabili ridondanti

Regola generale: **tenere il codice, droppare il testo**. Eccezione: se il testo aggiunge granularità non catturata dal codice (es. PRESTAZIONI_COMPRESE ha 4 valori testuali contro 3 codici → tenere il codice comunque per stabilità).

| Coppia ridondante | Tenere | Droppare |
|---|---|---|
| cod_tipo_scelta_contraente + tipo_scelta_contraente | codice | testo |
| COD_PRESTAZIONI_COMPRESE + PRESTAZIONI_COMPRESE | codice | testo |
| cod_esito + esito (AGG) | codice | testo |
| COD_ESITO + ESITO (BANDO) | codice | testo |
| cod_motivo_variante + motivo_variante | codice | testo (93k valori) |
| cod_cpv + descrizione_cpv (tutti i dataset) | codice | testo |
| cod_modalita_realizzazione + modalita_realizzazione | codice | testo |
| COD_MODALITA_INDIZIONE_* + testi | codici | testi |
| COD_STRUMENTO_SVOLGIMENTO + STRUMENTO | codice | testo |
| COD_MOTIVO_URGENZA + MOTIVO_URGENZA | codice | testo |
| natura_giuridica_codice + descrizione | codice | descrizione |
| provincia_codice + provincia_nome | codice | nome |
| cod_tipo_lavorazione + tipo_lavorazione | codice | testo |
| categoria SOA + desc_categoria | categoria | desc_categoria |

---

## PARTE 4 — CPV: proposta di raggruppamento

Basato su 112k CIG del campione Gen 2025 (0% NA):

| Macro-categoria | CPV prime 2 cifre | % campione | Rischio corruzione |
|---|---|---|---|
| SANITA_FARMACI | 33, 85 | 25.3% | **Alto** (cartelli farmaceutici, dispositivi medici) |
| LAVORI_COSTRUZIONI | 44, 45 | 10.1% | **Alto** (settore classico) |
| IT_INFORMATICA | 30, 48, 72 | 9.8% | **Medio-alto** (difficile valutazione tecnica) |
| SERVIZI_PROFESSIONALI | 66, 71, 73, 74, 79 | 15% | **Medio-alto** (alta discrezionalità) |
| MANUTENZIONE | 50 | 4.3% | **Medio** (parcellizzazione tipica) |
| FORNITURE_BENI | 09, 14–43 non sopra | ~15% | **Basso-medio** |
| SERVIZI_VARI | 55, 60, 63, 64, 75–98 | ~20% | Variabile |

Proposta operativa: usare `cod_cpv[:2]` come feature categorica (46 valori reali), raggruppabile nelle 7 macro-categorie sopra.

**Feature di interazione consigliata**: `cpv_macro × tipo_procedura` — cattura abbinamenti strutturalmente ad alto rischio. Esempi di coppie pericolose:
- SERVIZI_PROFESSIONALI + negoziata senza bando → alta discrezionalità + bassa competizione
- IT_INFORMATICA + affidamento diretto → difficile valutazione tecnica + nessuna gara
- LAVORI_COSTRUZIONI + affidamento diretto → settore ad alto volume + nessuna competizione

Implementazione: feature di interazione categoriale (concat delle due classi) o encoding come dummy pairs.

---

## PARTE 5 — Domande aperte residue

*(Le domande D-01, D-02, D-05, D-08 sono state risolte nella sessione corrente. Rimangono aperte:)*

**D-03 — Copertura temporale labeled data**: Verificare quanti dei 951 CIG positivi hanno dati ANAC disponibili prima di scegliere il range temporale. Molti potrebbero essere pre-2015 (BDNCP ha copertura sistematica dal ~2012, ma la qualità aumenta dal 2015).

**D-04 — Scagionati come negativi certi o unlabeled**: I 2318 CIG scagionati da sentenza corte possono essere trattati come negativi certi nel PU framework, oppure come unlabeled "rumorosi". La scelta impatta la stima del prior P(positivo) e la loss function del learner. Raccomandazione: trattarli come negativi certi in un primo run, poi verificare la sensibilità.

**D-06 — Attestazioni SOA**: Il join è time-aware (SOA valida alla data di aggiudicazione × cf_impresa × classifica richiesta). Complessità implementativa alta. Confermare se il guadagno predittivo giustifica il costo. Feature target: `flag_soa_scaduta`, `flag_classifica_inadeguata`.

**D-07 — Feature storiche dell'appaltatore**: Includere feature aggregate storiche (es. `tasso_medio_varianti_cf_storico` = media di pct_overrun_variante nei contratti precedenti dello stesso CF, negli ultimi 3 anni)? Non sono leakage per il contratto corrente ma aumentano significativamente la complessità del pipeline. Alta potenza predittiva attesa (appaltatori "recidivi").

**D-09 — NAl trattamento QUADRO ECONOMICO**: Per i contratti di tipo LAVORI, NA in voci standard del QE (importo_lavori, importo_sicurezza) è dato mancante o importo zero? Raccomandazione: incrociare con oggetto_principale_contratto. Se LAVORI e NA → dato mancante (segnale di qualità dati). Se SERVIZI/FORNITURE e NA → 0 concettuale.

---

*Fine report v3.1 — aggiornato 2026-03-27*
