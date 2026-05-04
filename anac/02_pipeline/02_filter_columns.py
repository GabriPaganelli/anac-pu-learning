"""
filter_columns.py — Filtra le colonne dei dataset ANAC secondo v33.

Per ogni CSV in DATA_DIR:
  1. Legge l'header
  2. Interseca con le colonne da tenere (case-insensitive)
  3. Riscrive il file con solo le colonne utili
  4. Logga eventuali colonne v33 non trovate nel CSV

File grandi (>200MB): scrittura a chunk per evitare OOM.
"""

import sys, os, shutil
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import os as _os
_BASE    = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_RAW = _os.path.join(_BASE, "data", "raw")

DATA_DIR = _os.path.join(_BASE, "data", "raw")
CHUNK_SIZE = 500_000
LARGE_THRESHOLD_MB = 200

# Colonne da tenere per dataset (da v33, esclusi DROP espliciti)
# Chiave = nome file CSV (lowercase), valore = lista colonne v33 (case come in v33)

KEEP_COLS = {
    "cig_2025.csv": [
        "cig","data_pubblicazione","data_scadenza_offerta",
        "DATA_ULTIMO_PERFEZIONAMENTO","DATA_COMUNICAZIONE_ESITO",
        "importo_complessivo_gara","importo_lotto","IMPORTO_SICUREZZA",
        "oggetto_principale_contratto","provincia","sezione_regionale",
        "TIPO_APPALTO_RISERVATO","ESITO","stato","settore",
        "anno_pubblicazione","flag_prevalente","FLAG_URGENZA","FLAG_DELEGA",
        "FLAG_PREV_RIPETIZIONI","cig_accordo_quadro","n_lotti_componenti",
        "luogo_istat","cod_tipo_scelta_contraente","cod_modalita_realizzazione",
        "codice_ausa","cf_amministrazione_appaltante","cod_cpv",
        "DURATA_PREVISTA","COD_STRUMENTO_SVOLGIMENTO","COD_MOTIVO_URGENZA",
        "COD_ESITO",
    ],
    "bandi-cig-modalita-realizzazione.csv": [
        "modalita-realizzazione_denominazione","modalita-realizzazione_codice",
    ],
    "bandi-cig-tipo-scelta-contraente.csv": [
        "tipo-scelta-contraente_denominazione","tipo-scelta-contraente_codice",
    ],
    "aggiudicazioni.csv": [
        "cig","id_aggiudicazione","data_aggiudicazione_definitiva",
        "data_comunicazione_esito","DATA_INCARICO_PROG","DATA_CONS_PROG",
        "importo_aggiudicazione","esito","PRESTAZIONI_COMPRESE",
        "MODO_RIAGGIUDICAZIONE","flag_subappalto","asta_elettronica",
        "FLAG_SCOMPUTO","FLAG_PROC_ACCELERATA","numero_offerte_ammesse",
        "numero_offerte_escluse","ribasso_aggiudicazione","num_imprese_offerenti",
        "cod_esito","num_imprese_richiedenti","num_imprese_invitate",
        "massimo_ribasso","minimo_ribasso","COD_PRESTAZIONI_COMPRESE",
        "CIG_PROG_ESTERNA","COD_MODO_RIAGGIUDICAZIONE","N_MANIF_INTERESSE",
    ],
    "aggiudicatari.csv": [
        "cig","codice_fiscale","id_aggiudicazione","ruolo","tipo_soggetto",
    ],
    "partecipanti.csv": [
        "cig","codice_fiscale","ruolo",
    ],
    "stazioni-appaltanti.csv": [
        # natura_giuridica_codice → DROP esplicito in v33
        "codice_fiscale","codice_ausa",
        "natura_giuridica_descrizione","provincia_codice","citta_codice",
    ],
    "avvio-contratto.csv": [
        "cig","id_aggiudicazione","data_stipula_contratto",
        "data_termine_contrattuale","data_verbale_consegna_definitiva",
        "data_inizio_effettiva","consegna_frazionata","consegna_sotto_riserva",
    ],
    "varianti.csv": [
        "cig","id_aggiudicazione","data_approvazione_variante",
        "cod_motivo_variante",
    ],
    "sospensioni.csv": [
        "cig","id_aggiudicazione","data_sospensione","data_ripresa",
        "descrizione_motivo",
    ],
    "stati-avanzamento.csv": [
        "cig","id_aggiudicazione","flag_ritardo","n_giorni_scostamento",
        "progressivo_sal","GIORNI_PROROGA",
    ],
    "quadro-economico.csv": [
        "cig","id_aggiudicazione","importo_sicurezza","importo_forniture",
        "importo_lavori","importo_progettazione","importo_servizi",
        "descrizione_evento","somme_a_disposizione",
        "ulteriori_oneri_non_soggetti_ribasso",
    ],
    "fonti-finanziamento.csv": [
        "cig","id_aggiudicazione",
        "entrate_con_dest_vincolata_pubblica_nazionale_regionale",
        "trasferimento_di_immobili_ex_art53_c6_dlgs_n163_06_economia_su_stanziamenti_non_vincolati",
        "fondi_di_bilancio_dellamministrazione_competente",
        "mutuo",
        "entrate_con_dest_vincolata_pubblica_nazionale_centrale",
        "entrate_con_dest_vincolata_pubblica_nazionale_locale",
        "altro",
        "entrate_con_dest_vincolata_pubblica_nazionale_altri",
        "apporto_di_capitali_privati",
        "sfruttamento_economico_e_funzionale_del_bene",
        "fondi_di_bilancio_della_stazione_appaltante",
        "entrate_con_dest_vincolata_pubblica_comunitaria",
        "entrate_con_dest_vincolata_privati",
    ],
    "subappalti.csv": [
        "cig","data_autorizzazione","cod_categoria","cod_cpv",
    ],
    "lavorazioni.csv": [
        "cig","tipo_lavorazione","cod_tipo_lavorazione",
    ],
    "collaudo.csv": [
        "cig","id_aggiudicazione","data_cert_collaudo","data_inizio_oper",
        "data_regolare_esec","data_nomina_coll","IMPORTO_CONTENZ_RISOLTO",
        "esito_collaudo","RISERVE_AVANZATE","RISERVE_DEFINITE",
    ],
    "misurepremiali-pnrrpnc.csv": [
        "cig",  # cod_mis_premiale → DROP esplicito
    ],
    "categorie-dpcm-aggregazione.csv": [
        "cig","categoria_merceologica_dpcm_aggregazione",
        "deroga_dpcm_soggetto_aggregatore",
        "cod_categoria_merceologica_dpcm_aggregazione",
        "cod_deroga_soggetto_aggregatore",
    ],
    "categorie-opera.csv": [
        "cig","id_categoria","descrizione","cod_tipo_categoria",
        "descrizione_tipo_categoria",
    ],
}


def resolve_cols(actual_header: list[str], wanted: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Risolve i nomi effettivi delle colonne da tenere con match case-insensitive.
    Ritorna: (cols_to_use, matched_v33, missing_in_csv)
    """
    actual_lower = {c.lower(): c for c in actual_header}
    cols_to_use, matched, missing = [], [], []
    for w in wanted:
        key = w.lower()
        if key in actual_lower:
            cols_to_use.append(actual_lower[key])
            matched.append(w)
        else:
            missing.append(w)
    return cols_to_use, matched, missing


def filter_file(fname: str, wanted: list[str]) -> dict:
    path = os.path.join(DATA_DIR, fname)
    size_mb = os.path.getsize(path) / (1024 ** 2)

    # Leggi header
    header_df = pd.read_csv(path, sep=";", nrows=0, low_memory=False)
    actual_header = header_df.columns.tolist()

    cols_to_use, _, missing = resolve_cols(actual_header, wanted)

    cols_dropped = [c for c in actual_header if c.lower() not in {w.lower() for w in wanted}]

    if not cols_to_use:
        return {"file": fname, "status": "ERROR", "msg": "Nessuna colonna trovata"}

    tmp_path = path + ".tmp"

    try:
        if size_mb > LARGE_THRESHOLD_MB:
            # Scrittura a chunk
            reader = pd.read_csv(
                path, sep=";", usecols=cols_to_use,
                chunksize=CHUNK_SIZE, low_memory=False
            )
            first = True
            for chunk in reader:
                chunk.to_csv(tmp_path, mode="a" if not first else "w",
                             header=first, index=False, sep=";")
                first = False
        else:
            df = pd.read_csv(path, sep=";", usecols=cols_to_use, low_memory=False)
            df.to_csv(tmp_path, index=False, sep=";")

        shutil.move(tmp_path, path)

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"file": fname, "status": "ERROR", "msg": str(e)}

    return {
        "file": fname,
        "status": "OK",
        "size_mb": round(size_mb, 1),
        "kept": len(cols_to_use),
        "dropped_csv": len(cols_dropped),
        "missing_v33": missing,
        "dropped_names": cols_dropped,
    }



files = sorted(KEEP_COLS.keys())
total = len(files)

for i, fname in enumerate(files, 1):
    pct = round(i / total * 100)
    print(f"[{pct:3d}%] ({i}/{total}) {fname} ...", flush=True)
    result = filter_file(fname, KEEP_COLS[fname])
    if result["status"] == "OK":
        r = result
        dropped_str = f"{r['dropped_csv']} colonne droppate"
        missing_str = f" | ATTENZIONE colonne v33 non trovate: {r['missing_v33']}" if r["missing_v33"] else ""
        print(f"       ✓ {r['size_mb']}MB → tenute {r['kept']} col, {dropped_str}{missing_str}")
    else:
        print(f"       ✗ ERRORE: {result['msg']}")

print("\nDone.")
