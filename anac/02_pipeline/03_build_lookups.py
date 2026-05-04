"""
03_build_lookups.py
Genera (o rigenera) i file di lookup in data/lookup/.

Due fonti:
  Fase 1 — Lookup estratti dai CSV raw (codice + descrizione presenti nel file).
            Le colonne descrizione vengono estratte qui; non servono nel parquet.
  Fase 2 — Lookup arricchiti manualmente (lavorazioni_tipo.csv, categorie_opera.csv)
            e lookup da file ANAC supplementari (categorie-dpcm, categorie-opera).

I file in data/lookup/ vengono sovrascritti solo se la fonte esiste.
Se i file esistono già e la fonte non è cambiata, è sicuro saltare questo script.

NOTE: lavorazioni_tipo.csv e categorie_opera.csv in data/lookup/ contengono
      colonne aggiuntive (macro_gruppo, note) aggiunte manualmente — NON vengono
      sovrascritte da questo script.
"""

import sys
import os
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW  = os.path.join(BASE, "data", "raw")
LOOKUP    = os.path.join(BASE, "data", "lookup")
TERR      = os.path.join(BASE, "data", "territorial")

os.makedirs(LOOKUP, exist_ok=True)


def read_csv(filename, **kwargs):
    path = os.path.join(DATA_RAW, filename)
    if not os.path.exists(path):
        print(f"  ⚠️  File non trovato: {filename} — skip")
        return None
    return pd.read_csv(path, sep=";", dtype=str, low_memory=False, **kwargs)


def extract_lookup(df, cod_col, desc_col, out_name, sep=";"):
    """Estrae coppie uniche (cod, desc) e le salva in data/lookup/."""
    if df is None or cod_col not in df.columns or desc_col not in df.columns:
        print(f"  ⚠️  Colonne {cod_col}/{desc_col} non trovate — skip {out_name}")
        return
    lk = (df[[cod_col, desc_col]]
          .drop_duplicates()
          .dropna(subset=[cod_col])
          .sort_values(cod_col)
          .reset_index(drop=True))
    out_path = os.path.join(LOOKUP, out_name)
    lk.to_csv(out_path, sep=sep, index=False)
    print(f"  ✓  {out_name}: {len(lk)} valori")


print("FASE 1a: Lookup da BANDO CIG")

cig = read_csv("cig_2023.csv")
if cig is not None:
    cig.columns = [c.lower() for c in cig.columns]
    extract_lookup(cig, "cod_esito",               "esito",
                   "lookup_cig_COD_ESITO.csv")
    extract_lookup(cig, "cod_motivo_urgenza",       "motivo_urgenza",
                   "lookup_cig_COD_MOTIVO_URGENZA.csv")
    extract_lookup(cig, "cod_strumento_svolgimento","strumento_svolgimento",
                   "lookup_cig_COD_STRUMENTO_SVOLGIMENTO.csv")

print("FASE 1b: Lookup da AGGIUDICAZIONI")

agg = read_csv("aggiudicazioni.csv")
if agg is not None:
    agg.columns = [c.lower() for c in agg.columns]
    extract_lookup(agg, "cod_esito",                "esito",
                   "lookup_aggiudicazioni_cod_esito.csv")
    extract_lookup(agg, "cod_modo_riaggiudicazione", "modo_riaggiudicazione",
                   "lookup_aggiudicazioni_COD_MODO_RIAGGIUDICAZIONE.csv")
    extract_lookup(agg, "cod_prestazioni_comprese",  "prestazioni_comprese",
                   "lookup_aggiudicazioni_COD_PRESTAZIONI_COMPRESE.csv")

print("FASE 1c: Lookup da VARIANTI")

var = read_csv("varianti.csv")
if var is not None:
    var.columns = [c.lower() for c in var.columns]
    # desc più comune per codice (colonna motivo_variante non sempre presente)
    if "cod_motivo_variante" in var.columns and "descrizione_motivo" in var.columns:
        extract_lookup(var, "cod_motivo_variante", "descrizione_motivo",
                       "lookup_varianti_cod_motivo_variante.csv")

print("FASE 1d: Lookup da SUBAPPALTI")

sub = read_csv("subappalti.csv")
if sub is not None:
    sub.columns = [c.lower() for c in sub.columns]
    if "cod_categoria" in sub.columns and "descrizione_categoria" in sub.columns:
        extract_lookup(sub, "cod_categoria", "descrizione_categoria",
                       "lookup_subappalti_cod_categoria.csv")

print("FASE 1e: Lookup da CATEGORIE OPERA")

cat_opera = read_csv("categorie-opera.csv")
if cat_opera is not None:
    cat_opera.columns = [c.lower() for c in cat_opera.columns]
    if "id_categoria" in cat_opera.columns and "tipo_categoria" in cat_opera.columns:
        extract_lookup(cat_opera, "id_categoria", "tipo_categoria",
                       "lookup_categorie_opera_id_categoria.csv")
    if "cod_tipo_categoria" in cat_opera.columns and "tipo_categoria" in cat_opera.columns:
        extract_lookup(cat_opera, "cod_tipo_categoria", "tipo_categoria",
                       "lookup_categorie_opera_cod_tipo_categoria.csv")

print("FASE 1f: Lookup da CATEGORIE DPCM")

cat_dpcm = read_csv("categorie-dpcm-aggregazione.csv")
if cat_dpcm is not None:
    cat_dpcm.columns = [c.lower() for c in cat_dpcm.columns]
    if "cod_categoria_merceologica" in cat_dpcm.columns and "desc_categoria_merceologica" in cat_dpcm.columns:
        extract_lookup(cat_dpcm, "cod_categoria_merceologica", "desc_categoria_merceologica",
                       "lookup_categorie_dpcm_cod_categoria_merceologica.csv")
    if "cod_deroga" in cat_dpcm.columns and "desc_deroga" in cat_dpcm.columns:
        extract_lookup(cat_dpcm, "cod_deroga", "desc_deroga",
                       "lookup_categorie_dpcm_cod_deroga.csv")

print("FASE 1g: Lookup province")

terr_path = os.path.join(TERR, "contesto_province.csv")
if os.path.exists(terr_path):
    terr = pd.read_csv(terr_path, dtype=str, low_memory=False)
    terr.columns = [c.lower() for c in terr.columns]
    if "codice_provincia" in terr.columns and "nome_provincia" in terr.columns:
        lk = (terr[["codice_provincia", "nome_provincia"]]
              .drop_duplicates()
              .dropna()
              .sort_values("codice_provincia")
              .reset_index(drop=True))
        out_path = os.path.join(LOOKUP, "lookup_province_codice_nome.csv")
        lk.to_csv(out_path, index=False)
        print(f"  ✓  lookup_province_codice_nome.csv: {len(lk)} province")
else:
    print("  ⚠️  contesto_province.csv non trovato — eseguire prima utils/download_contesto.py")

print("FASE 2: Verifica lookup manuali (non sovrascritti)")

manual = ["lavorazioni_tipo.csv", "categorie_opera.csv"]
for f in manual:
    path = os.path.join(LOOKUP, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✓  {f}: {len(df)} righe — non modificato")
    else:
        print(f"  ⚠️  {f}: NON TROVATO — deve essere ricreato manualmente.")
        print(f"       Vedere doc/decisions_log.md (sezioni VARIANTI e CATEGORIE OPERA) per la logica.")

print("\nDone. Lookup in data/lookup/:")
for f in sorted(os.listdir(LOOKUP)):
    print(f"  {f}")
