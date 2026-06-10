# 01_sentenze — Pipeline etichette da sentenze TAR/CdS
# *01_sentenze — Label pipeline from TAR/CdS court rulings*

---

## Italiano

Scarica e processa le sentenze del Tribunale Amministrativo Regionale (TAR) e del Consiglio di Stato (CdS) dal portale [OpenGA](https://openga.giustizia-amministrativa.it), identifica i Codici Identificativi Gara (CIG) coinvolti in contenziosi e produce le etichette per il PU learning.

---

## Logica di etichettatura

Un CIG riceve **label=1** (condannato) se:
- Il ricorrente ha **vinto** davanti al TAR (sentenza favorevole), **e**
- Il Consiglio di Stato non ha successivamente ribaltato la decisione (dominanza CdS).

Oppure se la gara risulta chiusa per:
- **REATI ACCERTATI** (`cod_motivo_risoluzione = 4` in fine-contratto ANAC)
- **CODICE ANTIMAFIA** (`cod_motivo_interruzione_anticipata = 6`)

Un CIG riceve **label=0** (scagionato) se:
- Il ricorrente ha **perso** davanti al TAR (ricorso rigettato/inammissibile), **e**
- Il CIG non compare mai tra i condannati (non-contraddizione).

---

## Esecuzione

### Prerequisiti

```bash
pip install -r requirements.txt   # dalla root del repository
```

### Esecuzione completa

```bash
python run_pipeline.py
```

Esegue tutti e 9 i passi in sequenza:

| Step | Descrizione |
|------|-------------|
| 1    | Estrazione — scopre e scarica tutti i CSV da OpenGA |
| 2    | Pulizia — normalizza colonne, valida CIG, costruisce UID |
| 3    | Deduplicazione — risolve UID duplicati per dataset |
| 4    | Consolidamento — unisce Ordinanze + Sentenze (Sentenza prevale) |
| 5    | Join — inner join esiti con i Ricorsi via UID |
| 6    | Dominanza CdS — rimuove CIG dove il CdS ha ribaltato la vittoria TAR |
| 7    | Salvataggio — produce `cig_condannati.csv` + report diagnostico |
| 8    | Lista scagionati — produce `cig_scagionati.csv` |
| 9    | Build labels — copia i file finali in `labels/` |

### Opzioni

```bash
python run_pipeline.py --skip-download   # Usa file raw già scaricati (riprende da step 2)
python run_pipeline.py --no-cache        # Forza il re-download anche se i file esistono
python run_pipeline.py --dry-run         # Scopre gli URL senza scaricare
python run_pipeline.py --log-level DEBUG # Log verboso
```

### Solo lista scagionati (se i condannati sono già aggiornati)

```bash
python build_zero_list.py
```

### Solo aggiornamento labels/

```bash
python build_labels.py
```

---

## Struttura cartelle

```
01_sentenze/
├── run_pipeline.py        # Entry point completo (step 1-9)
├── build_zero_list.py     # Produce cig_scagionati.csv da esiti sfavorevoli
├── build_labels.py        # Combina TAR + fine-contratto → labels/
├── config.py              # Configurazione centralizzata (URL, path, filtri esito)
├── pipeline/              # Moduli della pipeline
│   ├── extractor.py       # Download CSV da CKAN OpenGA
│   ├── cleaner.py         # Normalizzazione, validazione CIG, costruzione UID
│   ├── deduplicator.py    # Risoluzione duplicati per UID
│   ├── consolidator.py    # Unione Ordinanze + Sentenze
│   ├── joiner.py          # Join esiti con Ricorsi
│   ├── cds_precedence.py  # Precedenza esiti CdS su TAR
│   └── reporter.py        # Salvataggio output e report diagnostico
├── utils/
│   ├── log_setup.py       # Configurazione logging
│   └── csv_loader.py      # Caricamento e concatenazione CSV
├── data/
│   ├── raw/               # CSV scaricati da OpenGA (via DVC: `dvc pull`)
│   ├── interim/           # Parquet intermedi (via DVC: `dvc pull`)
│   └── output/            # Output della pipeline (versionato in git)
│       ├── cig_condannati.csv      # CIG con sentenza favorevole (TAR)
│       ├── cig_scagionati.csv      # CIG con ricorso rigettato
│       └── diagnostic_report.json  # Statistiche della pipeline
└── logs/                  # Non versionato (gitignore)
    ├── pipeline.log
    └── decision_log.csv
```

---

## Output finale

I file prodotti in `labels/` sono quelli usati dalla `02_pipeline`:

| File | Contenuto | Colonne |
|------|-----------|---------|
| `labels/cig_condannati.csv` | CIG label=1 | CIG, fonte, motivo |
| `labels/cig_scagionati.csv` | CIG label=0 | CIG, fonte, motivo |

La colonna `fonte` distingue l'origine dell'etichetta:
- `sentenza_TAR`: verdetto TAR (vittoria o sconfitta del ricorrente)
- `fine_contratto`: chiusura per reato o antimafia registrata in ANAC (solo condannati)
- `ordinanza_TAR`: ordinanza cautelare sfavorevole (solo scagionati)

---

## Note metodologiche

**UID (Unique Identifier)**: ogni ricorso è identificato dalla tripletta normalizzata `(NOME_SEDE, ANNO_DEPOSITO_RICORSO, NUMERO_RICORSO)`. Il join tra esiti e ricorsi avviene solo su match esatto — nessun fuzzy matching.

**Dominanza CdS**: se il Consiglio di Stato ha rigettato l'appello del ricorrente (o dichiarato l'appello inammissibile/improcedibile), la vittoria TAR viene annullata e il CIG rimosso dalla lista condannati.

**Non-contraddizione**: un CIG non può essere contemporaneamente in condannati e scagionati. In caso di conflitto, il CIG rimane tra i condannati e viene rimosso dagli scagionati.

---

## English

Downloads and processes rulings from the Regional Administrative Court (TAR) and Council of State (CdS) from the [OpenGA](https://openga.giustizia-amministrativa.it) portal, identifies the tender identifiers (CIG — Codice Identificativo Gara) involved in litigation, and produces labels for PU learning.

---

## Labelling logic

A CIG receives **label=1** (condemned) if:
- The appellant **won** before the TAR (favourable ruling), **and**
- The Council of State did not subsequently overturn the decision (CdS dominance).

Or if the tender was closed due to:
- **ASCERTAINED CRIMES** (`cod_motivo_risoluzione = 4` in ANAC end-of-contract data)
- **ANTI-MAFIA CODE** (`cod_motivo_interruzione_anticipata = 6`)

A CIG receives **label=0** (cleared) if:
- The appellant **lost** before the TAR (appeal rejected/inadmissible), **and**
- The CIG never appears among the condemned (non-contradiction).

---

## Execution

### Prerequisites

```bash
pip install -r requirements.txt   # from the repository root
```

Access to `openga.giustizia-amministrativa.it` is required.

### Full run

```bash
python run_pipeline.py
```

Executes all 9 steps in sequence:

| Step | Description |
|------|-------------|
| 1    | Extraction — discovers and downloads all CSVs from OpenGA |
| 2    | Cleaning — normalises columns, validates CIGs, builds UIDs |
| 3    | Deduplication — resolves duplicate UIDs per dataset |
| 4    | Consolidation — merges Ordinances + Rulings (Ruling takes precedence) |
| 5    | Join — inner join of outcomes with Appeals via UID |
| 6    | CdS dominance — removes CIGs where CdS overturned the TAR win |
| 7    | Save — produces `cig_condannati.csv` + diagnostic report |
| 8    | Cleared list — produces `cig_scagionati.csv` |
| 9    | Build labels — copies final files to `labels/` |

### Options

```bash
python run_pipeline.py --skip-download   # Use already-downloaded raw files (resumes from step 2)
python run_pipeline.py --no-cache        # Force re-download even if files exist
python run_pipeline.py --dry-run         # Discover URLs without downloading
python run_pipeline.py --log-level DEBUG # Verbose logging
```

### Cleared list only (if condemned list is already up to date)

```bash
python build_zero_list.py
```

### Labels update only

```bash
python build_labels.py
```

---

## Folder structure

```
01_sentenze/
├── run_pipeline.py        # Full entry point (steps 1–9)
├── build_zero_list.py     # Produces cig_scagionati.csv from unfavourable outcomes
├── build_labels.py        # Combines TAR + end-of-contract → labels/
├── config.py              # Centralised configuration (URLs, paths, outcome filters)
├── pipeline/              # Pipeline modules
│   ├── extractor.py       # CSV download from CKAN OpenGA
│   ├── cleaner.py         # Normalisation, CIG validation, UID construction
│   ├── deduplicator.py    # Duplicate resolution by UID
│   ├── consolidator.py    # Merge Ordinances + Rulings
│   ├── joiner.py          # Join outcomes with Appeals
│   ├── cds_precedence.py  # CdS precedence over TAR outcomes
│   └── reporter.py        # Output saving and diagnostic report
├── utils/
│   ├── log_setup.py       # Logging configuration
│   └── csv_loader.py      # CSV loading and concatenation
├── data/
│   ├── raw/               # CSVs downloaded from OpenGA (via DVC: `dvc pull`)
│   ├── interim/           # Intermediate parquets (via DVC: `dvc pull`)
│   └── output/            # Pipeline output (versioned in git)
│       ├── cig_condannati.csv      # CIGs with favourable TAR ruling
│       ├── cig_scagionati.csv      # CIGs with rejected appeal
│       └── diagnostic_report.json  # Pipeline statistics
└── logs/                  # Not versioned (gitignore)
    ├── pipeline.log
    └── decision_log.csv
```

---

## Final output

Files produced in `labels/` are those consumed by `02_pipeline`:

| File | Content | Columns |
|------|---------|---------|
| `labels/cig_condannati.csv` | CIG label=1 | CIG, fonte, motivo |
| `labels/cig_scagionati.csv` | CIG label=0 | CIG, fonte, motivo |

The `fonte` column distinguishes the label origin:
- `sentenza_TAR`: TAR ruling (claimant win or loss)
- `fine_contratto`: closure for crime or anti-mafia code recorded in ANAC (condannati only)
- `ordinanza_TAR`: unfavourable interim order (scagionati only)

---

## Methodological notes

**UID (Unique Identifier)**: each appeal is identified by the normalised triple `(NOME_SEDE, ANNO_DEPOSITO_RICORSO, NUMERO_RICORSO)`. The join between outcomes and appeals uses exact match only — no fuzzy matching.

**CdS dominance**: if the Council of State rejected the appellant's appeal (or declared it inadmissible/improcedable), the TAR win is annulled and the CIG removed from the condemned list.

**Non-contradiction**: a CIG cannot simultaneously appear in both the condemned and cleared lists. In case of conflict, the CIG remains among the condemned and is removed from the cleared list.
