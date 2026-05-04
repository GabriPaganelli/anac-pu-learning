"""
config.py — Configurazione centralizzata per la pipeline OpenGA CIG.
Modificare i valori qui; non usare valori fissi negli script della pipeline.
"""

from pathlib import Path

# Cartelle
BASE_DIR    = Path(__file__).parent
RAW_DIR     = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
OUTPUT_DIR  = BASE_DIR / "data" / "output"
LOG_DIR     = BASE_DIR / "logs"
LABELS_DIR  = BASE_DIR.parent / "labels"

for _d in (RAW_DIR, INTERIM_DIR, OUTPUT_DIR, LOG_DIR, LABELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Portale OpenGA
BASE_URL     = "https://openga.giustizia-amministrativa.it"
CKAN_API_URL = f"{BASE_URL}/api/3/action"

# Slug dei gruppi CKAN
GROUP_RICORSI   = "ricorsi-pervenuti"
GROUP_ORDINANZE = "ordinanze"
GROUP_SENTENZE  = "sentenze"

# Scarica solo i pacchetti Ricorsi che contengono righe CIG per appalto.
# I pacchetti "tipologia" sono statistiche aggregate — non utili per la pipeline.
RICORSI_PACKAGE_FILTER = "appalto"  # substring match sul nome (case-insensitive)

# Download
CACHE_ENABLED        = True   # salta il re-download se il file esiste e la dimensione corrisponde
MAX_DOWNLOAD_WORKERS = 8
DOWNLOAD_TIMEOUT     = 60     # secondi per request

# Validazione CIG
CIG_EXPECTED_LENGTH = 10

# Valori non validi da rimuovere prima della validazione.
# Euristica: compaiono nei dati reali come segnaposto "nessun CIG disponibile".
CIG_PLACEHOLDER_VALUES: set[str] = {
    "", "0000000000", "N.D.", "ND", "N/D", "NA", "NULL", "NONE",
    "-", "--", "---", "N.A.", "//", "/",
}

# Separatori che possono unire più CIG in una singola cella
CIG_SEPARATORS = [",", ";", "\n", "\r\n", " - ", " / "]

# Esiti favorevoli (il ricorrente HA VINTO — candidato per cig_condannati)
# Tutte le stringhe di esito vengono normalizzate prima del confronto:
#   strip, lower, rimozione accenti, apostrofo → spazio, compressione spazi.
# Vedi pipeline/cleaner.py::_norm_esito() per la logica di normalizzazione.
FAVORABLE_SENTENZE: set[str] = {
    "accoglie",
    "accoglie il ricorso",
    "accoglie sui motivi aggiunti",
    "accoglie parzialmente",
    "accolto",
    "accolto parzialmente",
    "accolto parzialmente nei termini in motivazione",
    "dichiara nullita",
    "dichiara la nullita",
    "annulla",
    "annulla in parte",
}

# Per le Ordinanze: solo "accoglie" (sospensiva cautelare concessa).
# Non si usa matching più ampio per le ordinanze interlocutorie.
FAVORABLE_ORDINANZE: set[str] = {
    "accoglie",
    "accoglie il ricorso",
    "accoglie parzialmente",
    "accoglie parzialmente il ricorso",
}

# Esiti sfavorevoli (il ricorrente HA PERSO — candidato per cig_scagionati)
UNFAVORABLE_SENTENZE: set[str] = {
    "respinge",
    "respinge il ricorso",
    "respinge sui motivi aggiunti",
    "respinge nel merito",
    "respinto",
    "respinto nel merito",
    "dichiara inammissibile",
    "dichiara inammissibile il ricorso",
    "dichiara irricevibile",
    "dichiara irricevibile il ricorso",
    "dichiara improcedibile",
    "dichiara improcedibile il ricorso",
    "dichiara estinto",
    "dichiara estinto il ricorso",
    "dichiara perenzione",
    "da in perenzione",
    "perenzione",
}

UNFAVORABLE_ORDINANZE: set[str] = {
    "respinge",
    "respinge il ricorso",
    "respinge la domanda cautelare",
    "respinge la domanda",
    "rigetta",
    "rigetta la domanda cautelare",
}

# Dominanza Consiglio di Stato
# Esiti CdS che annullano una vittoria TAR → il CIG esce dalla lista condannati.
CDS_REMOVE_OUTCOMES: set[str] = {
    "respinge",
    "respinge il ricorso",
    "respinge l appello",
    "dichiara improcedibile",
    "dichiara improcedibile l appello",
    "dichiara inammissibile",
    "dichiara inammissibile l appello",
}

# Esiti CdS genuinamente ambigui → il CIG rimane in lista ma viene loggato.
CDS_AMBIGUOUS_OUTCOMES: set[str] = {
    "annulla con rinvio",
    "rimette al tar",
    "sospende il giudizio",
}

# Alias di colonne
# Mappa nomi non standard trovati in alcune versioni CSV → nome canonico.
COLUMN_ALIASES: dict[str, str] = {
    "ANNO_DEPOSITO":     "ANNO_DEPOSITO_RICORSO",
    "ANNO_DEP_RICORSO":  "ANNO_DEPOSITO_RICORSO",
    "DATA_DEPOSITO":     "DATA_DEPOSITO_RICORSO",
    "COD_CIG":           "CODICE_CIG",
    "CIG":               "CODICE_CIG",
    "NUM_RICORSO":       "NUMERO_RICORSO",
    "NR_RICORSO":        "NUMERO_RICORSO",
    "ESITO":             "ESITO_PROVVEDIMENTO",
    "ESITO_PROV":        "ESITO_PROVVEDIMENTO",
    "DATA_PUB":          "DATA_PUBBLICAZIONE",
    "DATA_PUBBL":        "DATA_PUBBLICAZIONE",
    "DATA_PUBBLICAZ":    "DATA_PUBBLICAZIONE",
    "NUM_PROVVEDIMENTO": "NUMERO_PROVVEDIMENTO",
    "NR_PROVVEDIMENTO":  "NUMERO_PROVVEDIMENTO",
}

# File intermedi
INTERIM_RICORSI   = INTERIM_DIR / "ricorsi_clean.parquet"
INTERIM_ORDINANZE = INTERIM_DIR / "ordinanze_clean.parquet"
INTERIM_SENTENZE  = INTERIM_DIR / "sentenze_clean.parquet"
INTERIM_OUTCOMES  = INTERIM_DIR / "outcomes_consolidated.parquet"
INTERIM_JOINED    = INTERIM_DIR / "joined_tar.parquet"
INTERIM_CDS       = INTERIM_DIR / "cds_outcomes.parquet"

# File di output
OUTPUT_CIG_CONDANNATI = OUTPUT_DIR / "cig_condannati.csv"
OUTPUT_CIG_SCAGIONATI = OUTPUT_DIR / "cig_scagionati.csv"
OUTPUT_DIAGNOSTIC     = OUTPUT_DIR / "diagnostic_report.json"

# File di log
LOG_DECISIONS = LOG_DIR / "decision_log.csv"
LOG_PIPELINE  = LOG_DIR / "pipeline.log"
