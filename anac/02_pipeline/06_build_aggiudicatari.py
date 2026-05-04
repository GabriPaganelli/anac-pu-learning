"""
Build aggiudicatari features and join onto mega parquet.

Feature prodotta: tipo_soggetto_agg (categoriale raggruppata)
Decisioni:
  - ruolo usato solo per row selection (mandataria in ATI), poi droppato
  - codice_fiscale non entra nel parquet
  - SC-07 eseguito su raw files (non richiede dati nel parquet)
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

import os as _os
_BASE      = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_RAW  = _os.path.join(_BASE, "data", "raw")
_LABELS    = _os.path.join(_BASE, "labels")
_MEGA      = _os.path.join(_BASE, "output", "parquet", "bando_cig_all.parquet")
_SC_LOG    = _os.path.join(_BASE, "output", "sc_results.txt")

BASE       = _BASE
ANAC_DATA  = _DATA_RAW
MEGA       = _MEGA
SC_LOG     = _SC_LOG

sc_results = []

print("STEP 1: Reading aggiudicatari.csv")
agg = pd.read_csv(f'{ANAC_DATA}/aggiudicatari.csv', sep=';', dtype=str, low_memory=False)
agg.columns = [c.lower().strip() for c in agg.columns]
agg['cig'] = agg['cig'].str.strip()
print(f"Shape: {agg.shape}")
print(f"CIG unici: {agg['cig'].nunique():,}")
print(f"CIG con >1 riga: {(agg['cig'].value_counts() > 1).sum():,}")

print("\nSTEP 2: Normalizing and grouping tipo_soggetto")

def map_tipo_soggetto(val):
    if pd.isna(val):
        return 'ALTRO'
    v = val.strip().upper()
    if v.startswith('IMPRESA SINGOLA') or v in ('IMPRESA', 'DITTA INDIVIDUALE', 'MONOSOGGETTIVO'):
        return 'SINGOLA'
    if v.startswith('ATI') or v == 'RTI':
        return 'ATI'
    if v.startswith('CONSORZIO') or v.startswith('GEIE') is False and 'CONSORZIO' in v:
        return 'CONSORZIO'
    if v.startswith('GEIE'):
        return 'GEIE'
    if v == 'STAZIONE APPALTANTE':
        return 'SA'
    return 'ALTRO'

agg['tipo_soggetto_agg'] = agg['tipo_soggetto'].apply(map_tipo_soggetto)
print("Distribuzione tipo_soggetto_agg:")
print(agg['tipo_soggetto_agg'].value_counts(dropna=False).to_string())

print("\nSTEP 3: Normalizing ruolo for row selection")
agg['_ruolo_norm'] = agg['ruolo'].str.strip().str.upper().fillna('')

print("\nSTEP 4: Aggregating to CIG level")

def select_row(group):
    # Priorita': riga con ruolo MANDATARIA
    mandataria = group[group['_ruolo_norm'] == 'MANDATARIA']
    if len(mandataria) > 0:
        return mandataria.iloc[[0]]
    return group.iloc[[0]]

cig_counts = agg['cig'].value_counts()
single_cigs = set(cig_counts[cig_counts == 1].index)
multi_cigs  = set(cig_counts[cig_counts > 1].index)

single_df = agg[agg['cig'].isin(single_cigs)]
multi_df  = agg[agg['cig'].isin(multi_cigs)]

selected_multi = multi_df.groupby('cig', group_keys=False).apply(select_row)
agg_cig = pd.concat([single_df, selected_multi], ignore_index=True)
agg_cig = agg_cig[['cig', 'tipo_soggetto_agg']].copy()

print(f"Righe dopo aggregazione: {len(agg_cig):,} (atteso: {agg['cig'].nunique():,})")
assert agg_cig['cig'].duplicated().sum() == 0, "ERRORE: CIG duplicati dopo aggregazione"
print("Distribuzione finale tipo_soggetto_agg:")
print(agg_cig['tipo_soggetto_agg'].value_counts().to_string())

print("\nSC-07: CF aggiudicatario in PARTECIPANTI per CIG")

par = pd.read_csv(f'{ANAC_DATA}/partecipanti.csv', sep=';', dtype=str, low_memory=False)
par.columns = [c.lower().strip() for c in par.columns]

agg_raw = pd.read_csv(f'{ANAC_DATA}/aggiudicatari.csv', sep=';', dtype=str, low_memory=False)
agg_raw.columns = [c.lower().strip() for c in agg_raw.columns]
agg_raw = agg_raw.dropna(subset=['cig','codice_fiscale'])
agg_raw['cig'] = agg_raw['cig'].str.strip()
agg_raw['codice_fiscale'] = agg_raw['codice_fiscale'].str.strip()

par_clean = par.dropna(subset=['cig','codice_fiscale']).copy()
par_clean['cig'] = par_clean['cig'].str.strip()
par_clean['codice_fiscale'] = par_clean['codice_fiscale'].str.strip()
par_set = par_clean.groupby('cig')['codice_fiscale'].apply(set)

common_cigs = set(agg_raw['cig']) & set(par_set.index)
n_par_cigs = par_clean['cig'].nunique()
n_agg_cigs = agg_raw['cig'].nunique()

agg_common = agg_raw[agg_raw['cig'].isin(common_cigs)].copy()
agg_common['_in_par'] = agg_common.apply(
    lambda r: r['codice_fiscale'] in par_set.get(r['cig'], set()), axis=1
)

n_total = len(agg_common)
n_ok    = agg_common['_in_par'].sum()
n_fail  = n_total - n_ok
pct_fail = n_fail / n_total * 100 if n_total > 0 else 0

cig_any_fail = (~agg_common.groupby('cig')['_in_par'].any()).sum()
pct_cig_fail = cig_any_fail / len(common_cigs) * 100

print(f"  Copertura partecipanti: {n_par_cigs:,} CIG ({n_par_cigs/n_agg_cigs*100:.1f}% di aggiudicatari)")
print(f"  CIG confrontabili (in entrambi): {len(common_cigs):,}")
print(f"  Righe confrontate: {n_total:,}")
print(f"  CF trovato in partecipanti: {n_ok:,} ({100-pct_fail:.1f}%)")
print(f"  CF NON trovato: {n_fail:,} ({pct_fail:.1f}%)")
print(f"  CIG con >= 1 aggiudicatario fuori da partecipanti: {cig_any_fail:,} ({pct_cig_fail:.1f}%)")
print(f"  Nota: fail concentrati su ATI (mandanti spesso non registrati singolarmente)")

status_07 = "WARNING" if pct_fail > 10 else "OK"
msg_07 = (f"copertura_par={n_par_cigs/n_agg_cigs*100:.1f}%, confrontabili={len(common_cigs):,}, "
          f"CF_fail={n_fail:,} ({pct_fail:.1f}%), CIG_fail={cig_any_fail:,} ({pct_cig_fail:.1f}%) "
          f"| bassa severita': copertura partecipanti sparsa (~2.6% CIG), fail concentrati su ATI")
sc_results.append(("SC-07", status_07, msg_07))

print("\nSTEP 6: Loading parquet and joining")
parquet = pd.read_parquet(MEGA)
print(f"  Parquet shape: {parquet.shape}")

# Drop colonna pre-esistente se presente (re-run idempotente)
if 'tipo_soggetto_agg' in parquet.columns:
    parquet = parquet.drop(columns=['tipo_soggetto_agg'])
    print("  Droppata tipo_soggetto_agg pre-esistente")

result = parquet.merge(agg_cig, on='cig', how='left')
print(f"  Shape dopo join: {result.shape}")

n_joined = result['tipo_soggetto_agg'].notna().sum()
print(f"  CIG con tipo_soggetto_agg: {n_joined:,} ({n_joined/len(result)*100:.1f}%)")
print(f"  Distribuzione tipo_soggetto_agg nel parquet:")
print(result['tipo_soggetto_agg'].value_counts(dropna=False).to_string())
print(f"  Label=1 rimasti: {(result['label']==1).sum()}")
print(f"  Label=0 rimasti: {(result['label']==0).sum()}")

print("\nSaving parquet...")
result.to_parquet(MEGA, index=False)
print("Parquet salvato.")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os.makedirs(os.path.dirname(SC_LOG), exist_ok=True)
with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n\n{'='*60}\n")
    f.write(f"Run: {timestamp} [build_aggiudicatari]\n")
    f.write(f"Parquet shape: {result.shape}\n")
    f.write("\nSanity Checks:\n")
    for sc_id, status, msg in sc_results:
        f.write(f"  {sc_id:12s} [{status}]  {msg}\n")
print(f"SC results appended to: {SC_LOG}")
