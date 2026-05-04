"""
filter_cig_annual.py
Applica ai file cig_YYYY.csv (2008-2024) la stessa selezione colonne di cig_2025.csv:
  - Converte separatore , → ;
  - Droppa mese_pubblicazione, IPOTESI_COLLEGAMENTO, CIG_COLLEGAMENTO
  - Mantiene ESITO (descrizione testuale — schema pre-2025)
  - NB: COD_ESITO non esiste in questi file, ESITO è il campo canonico per gli anni storici
"""
import sys, os, shutil
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

import os as _os
_BASE      = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_RAW  = _os.path.join(_BASE, "data", "raw")
_LABELS    = _os.path.join(_BASE, "labels")
_MEGA      = _os.path.join(_BASE, "output", "parquet", "bando_cig_all.parquet")
_SC_LOG    = _os.path.join(_BASE, "output", "sc_results.txt")

DATA = _DATA_RAW

# Colonne v33 BANDO CIG (versione anni storici: ESITO al posto di COD_ESITO)
KEEP = {
    "cig", "data_pubblicazione", "data_scadenza_offerta",
    "data_ultimo_perfezionamento", "data_comunicazione_esito",
    "importo_complessivo_gara", "importo_lotto", "importo_sicurezza",
    "oggetto_principale_contratto", "provincia", "sezione_regionale",
    "tipo_appalto_riservato", "esito", "stato", "settore",
    "anno_pubblicazione", "flag_prevalente", "flag_urgenza", "flag_delega",
    "flag_prev_ripetizioni", "cig_accordo_quadro", "n_lotti_componenti",
    "luogo_istat", "cod_tipo_scelta_contraente", "cod_modalita_realizzazione",
    "codice_ausa", "cf_amministrazione_appaltante", "cod_cpv",
    "durata_prevista", "cod_strumento_svolgimento", "cod_motivo_urgenza",
}
# NB: COD_ESITO non incluso — nei file 2008-2024 non esiste

years = list(range(2008, 2025))
total = len(years)

for i, y in enumerate(years, 1):
    fname = f"cig_{y}.csv"
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"[{i}/{total}] {fname}: NON TROVATO — skip")
        continue

    size_mb = os.path.getsize(path) / 1024**2

    # Leggi header per capire quali colonne ci sono
    hdr = pd.read_csv(path, sep=",", nrows=0, low_memory=False)
    actual_cols = hdr.columns.tolist()
    actual_lower = {c.lower(): c for c in actual_cols}

    # Selezione case-insensitive
    use_cols = [actual_lower[k] for k in KEEP if k in actual_lower]
    dropped = [c for c in actual_cols if c.lower() not in KEEP]

    # Leggi e riscrivi con sep=;
    tmp = path + ".tmp"
    reader = pd.read_csv(path, sep=",", usecols=use_cols,
                         chunksize=300_000, low_memory=False)
    first = True
    for chunk in reader:
        chunk.to_csv(tmp, mode="a" if not first else "w",
                     header=first, index=False, sep=";")
        first = False
    shutil.move(tmp, path)

    new_mb = os.path.getsize(path) / 1024**2
    pct = round(i / total * 100)
    print(f"[{pct:3d}%] ({i}/{total}) {fname}: {size_mb:.0f}MB → {new_mb:.0f}MB | "
          f"kept {len(use_cols)} col | dropped {len(dropped)}: {dropped}")

print("\nDone.")
