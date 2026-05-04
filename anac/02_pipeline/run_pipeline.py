"""
run_pipeline.py
Entry point per la pipeline completa.

Esegue gli step in ordine, uno alla volta.
Per un run parziale: commentare le righe che non servono.

Prerequisiti:
  - data/raw/           : file CSV ANAC (scaricati da 01_inventory.py)
  - data/territorial/   : contesto_province.csv (da utils/download_contesto.py)
  - labels/             : cig_condannati.csv, cig_scagionati.csv
  - data/lookup/        : lavorazioni_tipo.csv, categorie_opera.csv (manuali)

Output finale:
  - output/parquet/bando_cig_all.parquet   (parquet sorgente, 66+ colonne)
  - output/parquet/model/nativi/           M1/M2/M3 per XGBoost/LightGBM
  - output/parquet/model/preprocessed/    M1/M2/M3 per logistica/SVM
"""

import subprocess
import sys
import os
import time

BASE        = os.path.dirname(os.path.abspath(__file__))
PYTHON      = sys.executable


def run(script, description):
    path = os.path.join(BASE, script)
    print(f"\n▶  {script}  —  {description}")
    t0 = time.time()
    result = subprocess.run([PYTHON, path], cwd=os.path.dirname(BASE))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌  {script} FALLITO (exit {result.returncode}) dopo {elapsed:.0f}s")
        sys.exit(result.returncode)
    print(f"\n✓  {script} completato in {elapsed:.0f}s")


run("01_filter_cig_annual.py",
    "Normalizza i CIG annuali 2008-2024 (separatore, selezione colonne)")

run("02_filter_columns.py",
    "Filtra le colonne di tutti i dataset ANAC secondo variable_selection.xlsx")

run("03_build_lookups.py",
    "Genera i file di lookup codice→descrizione in data/lookup/")

run("04_build_bando_cig.py",
    "Crea bando_cig_all.parquet da CIG annuali + label join + territorial join")

run("05_build_aggiudicazioni.py",
    "Aggiunge feature aggiudicazione + filtro Opzione A")

run("06_build_aggiudicatari.py",
    "Aggiunge tipo_soggetto_agg (SINGOLA/ATI/CONSORZIO/...)")

run("07_build_stazione_appaltante.py",
    "Aggiunge natura_giuridica_SA (8 categorie)")

run("08_build_quadro_economico.py",
    "Aggiunge pct_riserva_base, pct_overrun_core, pct_riserva_consumata"
    " + fallback importo_sicurezza_pct da QE BASE_ASTA")

run("09_build_avvio_contratto.py",
    "Aggiunge lag_stipula, durata_pianificata, consegna_frazionata/sotto_riserva")

run("10_build_varianti.py",
    "Aggiunge n_varianti, flag_variante_sostanziale, pct_overrun_variante, ...")

run("11_build_sospensioni.py",
    "Aggiunge n_sospensioni, flag_sospensione, pct_durata_sospesa, ...")

run("12_build_sal.py",
    "Aggiunge n_sal, flag_in_ritardo, flag_proroga")

run("13_build_subappalti.py",
    "Aggiunge flag_subappalto (OR tra aggiudicazioni e subappalti.csv)")

run("14_build_lavorazioni.py",
    "Aggiunge tipo_lavorazione_macro (COSTRUZIONE/RISANAMENTO/MANUTENZIONE)")

run("15_build_collaudo.py",
    "Aggiunge esito_collaudo (POSITIVO/NEGATIVO; 6.5% coverage, NaN = non registrato)")

run("16_build_model_datasets.py",
    "Produce M1/M2/M3 nativi e preprocessed; droppa data_pubblicazione dal sorgente")

run("17_assign_folds.py",
    "Assegna la colonna 'fold' (0-3) ai 6 parquet model (stratificato P+N, round-robin U)")

print("\nPipeline completata.")
