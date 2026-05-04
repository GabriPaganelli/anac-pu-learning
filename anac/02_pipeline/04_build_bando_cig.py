"""
04_build_bando_cig.py
Crea il parquet base da tutti i file CIG annuali (2008-2025).

Operazioni:
  1. Concat di tutti i cig_YYYY.csv (dopo 01_filter_cig_annual + 02_filter_columns)
  2. Deduplicazione multi-CPV (flag_prevalente = 1)
  3. Feature engineering BANDO CIG (lag, ratios, mappings)
  4. Join label (condannati/scagionati)
  5. Join variabili territoriali (regione, tasso_disoccupazione, ...)
  6. Riordino colonne: cig, label, esito, anno_pubblicazione, regione, [features], data_pubblicazione
  7. Scrittura output/parquet/bando_cig_all.parquet

NOTA: data_pubblicazione (Timestamp) è mantenuta nel parquet come colonna ausiliaria
      perché build_aggiudicazioni.py la usa per calcolare lag_aggiudicazione_giorni.
      Viene droppata da 15_build_model_datasets.py alla fine della pipeline.

Decisioni documentate: doc/decisions_log.md §BANDO CIG, §Label construction,
                       §Variabili territoriali
"""

import sys
import os
import glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(BASE, "data", "raw")
DATA_TERR  = os.path.join(BASE, "data", "territorial")
LABELS_DIR = os.path.join(BASE, "labels")
OUT        = os.path.join(BASE, "output", "parquet", "bando_cig_all.parquet")

DATE_MIN = pd.Timestamp("1990-01-01")
DATE_MAX = pd.Timestamp("2030-12-31")


def parse_date(series):
    """Parse date string con clamping a [1990, 2030] per evitare overflow."""
    s = pd.to_datetime(series, errors="coerce")
    s = s.where(s >= DATE_MIN, pd.NaT)
    s = s.where(s <= DATE_MAX, pd.NaT)
    return s


print("STEP 1: Carica CIG annuali")

files = sorted(glob.glob(os.path.join(DATA_RAW, "cig_*.csv")))
print(f"  File trovati: {len(files)}")
frames = []
for f in files:
    try:
        df = pd.read_csv(f, sep=";", dtype=str, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        frames.append(df)
    except Exception as e:
        print(f"  ⚠️  {os.path.basename(f)}: {e}")

cig = pd.concat(frames, ignore_index=True)
print(f"  Concat totale: {cig.shape[0]:,} righe × {cig.shape[1]} colonne")

print("\nSTEP 2: Deduplicazione multi-CPV (flag_prevalente=1)")

if "flag_prevalente" in cig.columns:
    before = len(cig)
    cig = cig[cig["flag_prevalente"].astype(str).str.strip() == "1"].copy()
    print(f"  Droppate {before - len(cig):,} righe multi-CPV")
else:
    print("  ⚠️  flag_prevalente non trovato — skip deduplicazione")

print(f"  Righe dopo dedup: {len(cig):,}")
assert cig["cig"].duplicated().sum() == 0, "CIG duplicati dopo deduplicazione!"
print("  ✓ Nessun CIG duplicato")

# Drop flag_prevalente (sempre 1 dopo dedup)
cig.drop(columns=["flag_prevalente"], errors="ignore", inplace=True)

print("\nSTEP 3: Parse date (clamp 1990-2030)")

for col in ["data_pubblicazione", "data_scadenza_offerta",
            "data_ultimo_perfezionamento", "data_comunicazione_esito"]:
    if col in cig.columns:
        cig[col] = parse_date(cig[col])

print("\nSTEP 4: Feature engineering — lag e ratios")

dp = cig["data_pubblicazione"]

cig["finestra_offerta_giorni"] = (
    (cig["data_scadenza_offerta"] - dp).dt.days
    .where(cig["data_scadenza_offerta"].notna() & dp.notna())
    .astype("float64")
)

if "data_ultimo_perfezionamento" in cig.columns:
    cig["lag_perfezionamento_giorni"] = (
        (cig["data_ultimo_perfezionamento"] - dp).dt.days
        .where(cig["data_ultimo_perfezionamento"].notna() & dp.notna())
        .astype("float64")
    )

# lag_comunicazione_esito_giorni (feature M2 calcolata qui perché usa data_pubblicazione)
if "data_comunicazione_esito" in cig.columns:
    cig["lag_comunicazione_esito_giorni"] = (
        (cig["data_comunicazione_esito"] - dp).dt.days
        .where(cig["data_comunicazione_esito"].notna() & dp.notna())
        .astype("float64")
    )

for col in ["importo_lotto", "importo_sicurezza", "importo_complessivo_gara"]:
    if col in cig.columns:
        cig[col] = pd.to_numeric(cig[col].str.replace(",", "."), errors="coerce")

cig["importo_sicurezza_pct"] = np.where(
    cig["importo_lotto"].gt(0) & cig["importo_sicurezza"].notna(),
    cig["importo_sicurezza"] / cig["importo_lotto"],
    np.nan,
)
cig["importo_sicurezza_pct"] = cig["importo_sicurezza_pct"].astype("float64")

# SC-05: importo_sicurezza > importo_lotto → NaN
if cig["importo_sicurezza_pct"].notna().any():
    bad_mask = cig["importo_sicurezza"] > cig["importo_lotto"]
    n_bad = bad_mask.sum()
    if n_bad > 0:
        cig.loc[bad_mask, "importo_sicurezza_pct"] = np.nan
        cig.loc[bad_mask, "importo_sicurezza"]     = np.nan
        print(f"  SC-05: {n_bad:,} righe con importo_sicurezza > importo_lotto → NaN")

if "anno_pubblicazione" in cig.columns:
    cig["anno_pubblicazione"] = pd.to_numeric(cig["anno_pubblicazione"], errors="coerce").astype("Int64")

cig["n_lotti_componenti"] = pd.to_numeric(cig["n_lotti_componenti"], errors="coerce").astype("Int64")

# cod_modalita_realizzazione → modalita_realizzazione_macro (6 classi semantiche)
cig["cod_modalita_realizzazione"] = pd.to_numeric(
    cig["cod_modalita_realizzazione"], errors="coerce"
).astype("Int64")

_MODALITA_MACRO_MAP = {
    1: "APPALTO",
    7: "ECONOMIA",
    2: "ACCORDO_QUADRO",  9: "ACCORDO_QUADRO", 11: "ACCORDO_QUADRO",
    17: "ACCORDO_QUADRO", 18: "ACCORDO_QUADRO",
    3: "CONCESSIONE",     4: "CONCESSIONE",    20: "CONCESSIONE",    21: "CONCESSIONE",
    5: "PPP",             6: "PPP",             8: "PPP",
    12: "PPP",           13: "PPP",
    10: "ALTRO",         14: "ALTRO",          15: "ALTRO",
    16: "ALTRO",         19: "ALTRO",          90: "ALTRO",         999: "ALTRO",
}
cig["modalita_realizzazione_macro"] = (
    cig["cod_modalita_realizzazione"]
    .map(_MODALITA_MACRO_MAP)
    .astype("category")
)

cig["cod_strumento_svolgimento"] = pd.to_numeric(
    cig["cod_strumento_svolgimento"], errors="coerce"
).astype("Int64")

cig["cod_motivo_urgenza"] = pd.to_numeric(
    cig["cod_motivo_urgenza"], errors="coerce"
).astype("Int64")

# flag_urgenza: 0/1, NaN → 0
cig["flag_urgenza"] = (
    pd.to_numeric(cig["flag_urgenza"], errors="coerce")
    .fillna(0).astype("Int64")
)

# flag_delega: mantieni NaN (MNAR: assenza = non delegato)
cig["flag_delega"] = pd.to_numeric(cig["flag_delega"], errors="coerce").astype("Int64")

print("\nSTEP 5: Flag binari")

cig["flag_accordo_quadro"] = (
    cig["cig_accordo_quadro"].notna().astype("int8")
)

# flag_ripetizioni
cig["flag_ripetizioni"] = (
    pd.to_numeric(cig["flag_prev_ripetizioni"], errors="coerce")
    .fillna(0).clip(0, 1).astype("int8")
)

cig["settore_speciale"] = (
    cig["settore"].str.upper().str.contains("SPECIALI", na=False)
    .astype("int8")
)

# flag_appalto_riservato: 1 se compilato (riferimento legge), 0 se NaN
cig["flag_appalto_riservato"] = cig["tipo_appalto_riservato"].notna().astype("int8")

print("\nSTEP 6: Mappings categoriali")

# sezione_regionale: CENTRALE / REGIONALE / NaN
def map_sezione(val):
    if pd.isna(val):
        return None
    v = str(val).upper()
    if "CENTRALE" in v:
        return "CENTRALE"
    if "REGIONALE" in v or "SEZIONE" in v:
        return "REGIONALE"
    return None

cig["sezione_regionale"] = cig["sezione_regionale"].map(map_sezione)

# tipo_scelta_4cls: 35 codici → 4 classi (derivata empiricamente dal parquet originale)
TIPO_SCELTA_MAP = {
    "1":   "APERTA",
    "5":   "NEGOZIATA",    # dialogo competitivo
    "7":   "AFFIDAMENTO_DIRETTO",  # sistema dinamico acquisizione
    "8":   "AFFIDAMENTO_DIRETTO",  # cottimo fiduciario
    "12":  "RISTRETTA",
    "14":  "APERTA",
    "15":  "RISTRETTA",    # procedura negoziata (old code)
    "16":  "NEGOZIATA",    # accordo quadro
    "2":   "RISTRETTA",
    "22":  "AFFIDAMENTO_DIRETTO",
    "23":  "AFFIDAMENTO_DIRETTO",
    "24":  "AFFIDAMENTO_DIRETTO",
    "25":  "AFFIDAMENTO_DIRETTO",
    "26":  "RISTRETTA",
    "27":  "APERTA",
    "28":  "APERTA",
    "29":  "APERTA",
    "3":   "NEGOZIATA",
    "30":  "APERTA",
    "32":  "RISTRETTA",
    "33":  "RISTRETTA",
    "34":  "AFFIDAMENTO_DIRETTO",
    "35":  "AFFIDAMENTO_DIRETTO",
    "36":  "AFFIDAMENTO_DIRETTO",
    "37":  "AFFIDAMENTO_DIRETTO",
    "38":  "AFFIDAMENTO_DIRETTO",
    "4":   "NEGOZIATA",
    "40":  "NEGOZIATA",
    "6":   "AFFIDAMENTO_DIRETTO",
    "97":  "RISTRETTA",
    "98":  "AFFIDAMENTO_DIRETTO",
    "99":  "AFFIDAMENTO_DIRETTO",
    "114": "AFFIDAMENTO_DIRETTO",
    # 9 (pubblico incanto), 10 (licitazione privata): codici storici
    "9":   "APERTA",
    "10":  "RISTRETTA",
    "11":  "RISTRETTA",
    "13":  "AFFIDAMENTO_DIRETTO",
    "17":  None,           # non classificato → NaN
    "18":  "AFFIDAMENTO_DIRETTO",
    "19":  "AFFIDAMENTO_DIRETTO",
    "20":  "AFFIDAMENTO_DIRETTO",
    "21":  "AFFIDAMENTO_DIRETTO",
    # 999, 122 → NaN
}
cig["tipo_scelta_4cls"] = (
    cig["cod_tipo_scelta_contraente"].astype(str).str.strip()
    .map(TIPO_SCELTA_MAP)
)
print(f"  tipo_scelta_4cls: {cig['tipo_scelta_4cls'].value_counts(dropna=False).to_dict()}")

# cpv_macro_categoria: prime 2 cifre CPV → 6 macro-categorie
CPV_MAP = {
    "30": "IT", "48": "IT", "72": "IT",
    "33": "SANITA", "85": "SANITA",
    "44": "LAVORI", "45": "LAVORI",
    "71": "ING_PROF", "73": "ING_PROF", "79": "ING_PROF",
    "50": "SERVIZI", "51": "SERVIZI", "55": "SERVIZI", "60": "SERVIZI",
    "63": "SERVIZI", "64": "SERVIZI", "65": "SERVIZI", "66": "SERVIZI",
    "70": "SERVIZI", "75": "SERVIZI", "76": "SERVIZI", "77": "SERVIZI",
    "80": "SERVIZI", "90": "SERVIZI", "92": "SERVIZI", "98": "SERVIZI",
}

def map_cpv(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    prefix = str(val).strip()[:2]
    return CPV_MAP.get(prefix, "FORNITURE")

cig["cpv_macro_categoria"] = cig["cod_cpv"].map(map_cpv)
print(f"  cpv_macro_categoria: {cig['cpv_macro_categoria'].value_counts(dropna=False).to_dict()}")

# oggetto_principale_contratto: standardize
if "oggetto_principale_contratto" not in cig.columns:
    cig["oggetto_principale_contratto"] = None

print("\nSTEP 7: Normalizzazione esito")

def normalize_esito(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    v = str(val).strip().upper()
    if v == "AGGIUDICATA":
        return "AGGIUDICATA"
    if "ANNULLAT" in v or "REVOCAT" in v:
        return "ANNULLATA"
    if "DESERTA" in v:
        return "DESERTA"
    if "IN_CORSO" in v or v == "IN CORSO":
        return "IN_CORSO"
    if "SENZA ESITO" in v or "NON AGGIUDICATA" in v or "NON_AGGIUDICATA" in v:
        return "SENZA_ESITO"
    # Formato B-prefix 2024-2025: descrizioni testuali estese
    if "NESSUN VINCITORE" in v and "CHIUSA" in v:
        return "SENZA_ESITO"
    if "ANCORA IN CORSO" in v or "NON ANCORA" in v:
        return "IN_CORSO"
    if "PROPOSTA" in v:
        return "PROPOSTA"
    return v  # fallback: mantieni valore originale normalizzato

cig["esito"] = cig["esito"].map(normalize_esito)
print(f"  esito distribution: {cig['esito'].value_counts(dropna=False).head(8).to_dict()}")

print("\nSTEP 8: Label join")

condannati = pd.read_csv(os.path.join(LABELS_DIR, "cig_condannati.csv"), dtype=str)
scagionati  = pd.read_csv(os.path.join(LABELS_DIR, "cig_scagionati.csv"),  dtype=str)

cig_condannati = set(condannati["CIG"].str.strip())
cig_scagionati  = set(scagionati["CIG"].str.strip())

label_map = {c: pd.array([1], dtype="Int64")[0] for c in cig_condannati}
label_map.update({c: pd.array([0], dtype="Int64")[0] for c in cig_scagionati})

cig["label"] = cig["cig"].map(label_map).astype("Int64")
print(f"  label=1: {(cig['label']==1).sum():,} condannati joinati")
print(f"  label=0: {(cig['label']==0).sum():,} scagionati joinati")
print(f"  label=NA: {cig['label'].isna().sum():,} unlabeled")

print("\nSTEP 9: Join variabili territoriali")

terr = pd.read_csv(os.path.join(DATA_TERR, "contesto_province.csv"))
terr["codice_provincia"] = terr["codice_provincia"].astype(int)
terr["anno"] = terr["anno"].astype(int)

def luogo_to_prov(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    s = str(val).strip()
    # Prefissi non-standard (A-, B-, lettere) → non mappabili
    if not s.replace(".", "").replace(",", "").isdigit():
        return None
    try:
        return int(str(int(float(s))).zfill(6)[:3])
    except Exception:
        return None

cig["_cod_prov"] = cig["luogo_istat"].map(luogo_to_prov)

# Anno join: min(anno_pub − 1, 2024) → usa il dato dell'anno precedente la pubblicazione,
# capped all'ultimo anno disponibile in contesto_province.csv (2024).
cig["_anno_join"] = (
    (cig["anno_pubblicazione"].fillna(2025).astype(int) - 1).clip(upper=2024)
).clip(lower=2007)

terr_join = terr.set_index(["codice_provincia", "anno"])

def lookup_terr(row):
    prov = row["_cod_prov"]
    anno = row["_anno_join"]
    if prov is None or pd.isna(prov):
        return pd.Series({"regione": None, "tasso_disoccupazione": None,
                          "reddito_irpef_procapite": None, "tasso_omicidi_100k": None})
    key = (int(prov), int(anno))
    if key in terr_join.index:
        r = terr_join.loc[key]
        return pd.Series({
            "regione":               r["nome_provincia"],  # placeholder: sarà sostituito sotto
            "tasso_disoccupazione":  r["tasso_disoccupazione"],
            "reddito_irpef_procapite": r["reddito_irpef_procapite"],
            "tasso_omicidi_100k":    r["tasso_omicidi_100k"],
        })
    return pd.Series({"regione": None, "tasso_disoccupazione": None,
                      "reddito_irpef_procapite": None, "tasso_omicidi_100k": None})

# Aggiungi regione da provincia
# Mappa codice_provincia → nome_regione usando una lookup standard
PROV_TO_REGIONE = {
    1: "Piemonte", 2: "Piemonte", 3: "Piemonte", 4: "Piemonte",
    5: "Piemonte", 6: "Piemonte", 7: "Piemonte", 8: "Piemonte",
    9: "Valle d'Aosta",
    10: "Liguria", 11: "Liguria", 12: "Liguria", 13: "Liguria",
    14: "Lombardia", 15: "Lombardia", 16: "Lombardia", 17: "Lombardia",
    18: "Lombardia", 19: "Lombardia", 20: "Lombardia", 21: "Trentino-Alto Adige",
    22: "Trentino-Alto Adige", 23: "Veneto", 24: "Veneto", 25: "Veneto",
    26: "Veneto", 27: "Veneto", 28: "Veneto", 29: "Veneto",
    30: "Friuli-Venezia Giulia", 31: "Friuli-Venezia Giulia",
    32: "Friuli-Venezia Giulia", 33: "Friuli-Venezia Giulia",
    34: "Emilia-Romagna", 35: "Emilia-Romagna", 36: "Emilia-Romagna",
    37: "Emilia-Romagna", 38: "Emilia-Romagna", 39: "Emilia-Romagna",
    40: "Emilia-Romagna", 41: "Emilia-Romagna",
    42: "Toscana", 43: "Toscana", 44: "Toscana", 45: "Toscana",
    46: "Toscana", 47: "Toscana", 48: "Toscana", 49: "Toscana",
    50: "Toscana", 51: "Toscana",
    52: "Umbria", 53: "Umbria",
    54: "Marche", 55: "Marche", 56: "Marche", 57: "Marche", 109: "Marche",
    58: "Lazio", 59: "Lazio", 60: "Lazio", 61: "Lazio", 62: "Lazio",
    63: "Abruzzo", 64: "Abruzzo", 65: "Abruzzo", 66: "Abruzzo",
    67: "Molise", 68: "Molise",
    69: "Campania", 70: "Campania", 71: "Campania", 72: "Campania",
    73: "Campania",
    74: "Puglia", 75: "Puglia", 76: "Puglia", 77: "Puglia",
    78: "Puglia", 110: "Puglia",
    79: "Basilicata", 80: "Basilicata",
    81: "Calabria", 82: "Calabria", 83: "Calabria", 84: "Calabria",
    85: "Calabria",
    86: "Sicilia", 87: "Sicilia", 88: "Sicilia", 89: "Sicilia",
    90: "Sicilia", 91: "Sicilia", 92: "Sicilia", 93: "Sicilia",
    94: "Sicilia",
    95: "Sardegna", 96: "Sardegna", 97: "Sardegna", 98: "Sardegna",
    111: "Sardegna",  # Sud Sardegna
}

cig_for_join = cig[["cig", "_cod_prov", "_anno_join"]].copy()
cig_for_join["_cod_prov"] = pd.to_numeric(cig_for_join["_cod_prov"], errors="coerce")

merged_terr = cig_for_join.merge(
    terr.rename(columns={"codice_provincia": "_cod_prov", "anno": "_anno_join"}),
    on=["_cod_prov", "_anno_join"],
    how="left"
)
merged_terr = merged_terr.drop_duplicates("cig")

merged_terr["regione"] = merged_terr["_cod_prov"].map(PROV_TO_REGIONE)

cig = cig.merge(
    merged_terr[["cig", "regione", "tasso_disoccupazione",
                 "reddito_irpef_procapite", "tasso_omicidi_100k"]],
    on="cig", how="left"
)

terr_coverage = cig["tasso_disoccupazione"].notna().mean()
print(f"  Coverage territoriale: {terr_coverage:.1%}")
print(f"  NA regione: {cig['regione'].isna().mean():.1%}")

print("\nSTEP 10: Drop colonne intermedie")

DROP_COLS = [
    "cig_accordo_quadro",      # sostituita da flag_accordo_quadro
    "flag_prev_ripetizioni",   # sostituita da flag_ripetizioni
    "tipo_appalto_riservato",  # sostituita da flag_appalto_riservato
    "settore",                 # sostituita da settore_speciale
    "data_scadenza_offerta",   # usata per finestra_offerta_giorni
    "data_ultimo_perfezionamento",  # usata per lag_perfezionamento_giorni
    "data_comunicazione_esito",     # usata per lag_comunicazione_esito_giorni
    "importo_sicurezza",       # sostituita da importo_sicurezza_pct
    "stato",                   # zero varianza: quasi tutto ATTIVO
    "luogo_istat",             # usata per join territoriale, poi droppata
    "provincia",               # codice provincia usato nel join, sostituito da regione
    "durata_prevista",         # alta correlazione con lag calcolati, poca affidabilità
    "cod_cpv",                 # sostituita da cpv_macro_categoria
    "cod_tipo_scelta_contraente",  # sostituita da tipo_scelta_4cls
    "_cod_prov", "_anno_join",     # colonne ausiliarie join
]
cig.drop(columns=[c for c in DROP_COLS if c in cig.columns], inplace=True)

print("\nSTEP 11: Riordino colonne")

FRONT_COLS = ["cig", "label", "esito", "anno_pubblicazione", "regione"]
# data_pubblicazione in fondo (ausiliaria per step 05)
AUX_COLS   = ["data_pubblicazione"] if "data_pubblicazione" in cig.columns else []

remaining = [c for c in cig.columns
             if c not in FRONT_COLS and c not in AUX_COLS]
final_order = FRONT_COLS + remaining + AUX_COLS
cig = cig[[c for c in final_order if c in cig.columns]]

print(f"  Colonne finali: {len(cig.columns)}")
print(f"  Prime 6: {list(cig.columns[:6])}")

print("\nSTEP 12: Scrittura parquet")
print(f"  Shape: {cig.shape[0]:,} × {cig.shape[1]}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cig.to_parquet(OUT, index=False)
print(f"  ✓ Scritto: {OUT}")
print("\nDone.")
