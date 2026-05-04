"""
Build sospensioni features and join onto mega parquet.

Feature prodotte:
  - n_sospensioni:                 conteggio sospensioni per CIG (0 se assente)
  - durata_totale_sospensioni_gg:  somma durate sospensioni complete (0 se assente)
  - flag_sospensione:              1 se almeno 1 sospensione (= n_sospensioni > 0)
  - flag_sosp_giudiziaria:         1 se almeno 1 sospensione per intervento autorita' giudiziaria
  - pct_durata_sospesa:            durata_totale / durata_pianificata_giorni (NaN se durata mancante, cap 10)

Note:
  - Assenza dal file = 0 sospensioni, 0 giorni sospesi (semanticamente informativo)
  - pct_durata_sospesa > 1 e' atteso: le sospensioni estendono il contratto oltre la durata pianificata originale
  - flag_sosp_sospetta rimosso: il grouping a priori non era confermato dai dati; tutti i motivi
    mostrano segnale comparabile. Si usa flag_sospensione (binario) + flag_sosp_giudiziaria (specifico).
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import os as _os
_BASE      = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_RAW  = _os.path.join(_BASE, "data", "raw")
_LABELS    = _os.path.join(_BASE, "labels")
_MEGA      = _os.path.join(_BASE, "output", "parquet", "bando_cig_all.parquet")
_SC_LOG    = _os.path.join(_BASE, "output", "sc_results.txt")

ANAC_DATA = _DATA_RAW
MEGA      = _MEGA
SC_LOG    = _SC_LOG

print("STEP 1: Reading sospensioni.csv")
df = pd.read_csv(f'{ANAC_DATA}/sospensioni.csv', sep=';', dtype=str, low_memory=False)
df.columns = [c.lower().strip() for c in df.columns]
df['cig'] = df['cig'].str.strip()
print(f"Shape: {df.shape} | CIG unici: {df['cig'].nunique():,}")

pq_cigs = set(pd.read_parquet(MEGA, columns=['cig'])['cig'])

print("\nSTEP 2: Parsing dates")
df['data_sospensione'] = pd.to_datetime(df['data_sospensione'], errors='coerce')
df['data_ripresa']     = pd.to_datetime(df['data_ripresa'], errors='coerce')
n_no_ripresa = df['data_ripresa'].isna().sum()
print(f"  data_ripresa NA: {n_no_ripresa:,} ({n_no_ripresa/len(df)*100:.1f}%) -- escluse da durata")

print("\nSTEP 3: Aggregation per CIG")
n_sosp = df.groupby('cig').size().rename('n_sospensioni')

# Durata solo righe complete (data_ripresa non-null)
complete = df.dropna(subset=['data_sospensione', 'data_ripresa']).copy()
complete['durata_gg'] = (complete['data_ripresa'] - complete['data_sospensione']).dt.days.clip(lower=0)
durata_tot = complete.groupby('cig')['durata_gg'].sum().rename('durata_totale_sospensioni_gg')

flag_giud = (df[df['descrizione_motivo'] == "INTERVENTO AUTORITA' GIUDIZIARIA"]
             .groupby('cig').size().gt(0).astype(int).rename('flag_sosp_giudiziaria'))

agg = pd.concat([n_sosp, durata_tot, flag_giud], axis=1).reset_index()
agg['n_sospensioni']          = agg['n_sospensioni'].fillna(0).astype(int)
agg['durata_totale_sospensioni_gg'] = agg['durata_totale_sospensioni_gg'].fillna(0)  # 0 se assente
agg['flag_sosp_giudiziaria']  = agg['flag_sosp_giudiziaria'].fillna(0).astype(int)
agg['flag_sospensione']       = (agg['n_sospensioni'] > 0).astype(int)

print(f"  n_sospensioni mediana: {agg['n_sospensioni'].median():.0f}, p95: {agg['n_sospensioni'].quantile(0.95):.0f}, max: {agg['n_sospensioni'].max()}")
print(f"  durata_totale mediana: {agg['durata_totale_sospensioni_gg'].median():.0f}d, p95: {agg['durata_totale_sospensioni_gg'].quantile(0.95):.0f}d")
print(f"  flag_sosp_giudiziaria=1: {agg['flag_sosp_giudiziaria'].sum():,} CIG")
print(f"  flag_sospensione=1:      {agg['flag_sospensione'].sum():,} CIG")

print("\nSTEP 4: pct_durata_sospesa = durata_totale / durata_pianificata_giorni")
pq_dur = pd.read_parquet(MEGA, columns=['cig', 'durata_pianificata_giorni'])
agg = agg.merge(pq_dur, on='cig', how='left')

pct = agg['durata_totale_sospensioni_gg'] / agg['durata_pianificata_giorni']
# cap > 10 -> NaN (coerente con altre ratio nel dataset)
agg['pct_durata_sospesa'] = pct.where((pct >= 0) & (pct <= 10), np.nan)
agg.drop(columns=['durata_pianificata_giorni'], inplace=True)

n_calcolabile = agg['pct_durata_sospesa'].notna().sum()
n_gt1 = (agg['pct_durata_sospesa'] > 1).sum()
print(f"  Non-null: {n_calcolabile:,} ({n_calcolabile/len(agg)*100:.1f}% dei CIG con sospensione)")
print(f"  pct > 1 (sosp > durata pianificata): {n_gt1:,} -- atteso (sospensioni estendono contratto)")
print(f"  pct > 10 (-> NaN): {(pct > 10).sum():,}")
print(f"  Mediana (non-null): {agg['pct_durata_sospesa'].median():.3f}")
print(f"  p95: {agg['pct_durata_sospesa'].quantile(.95):.3f}")

print("\nSANITY CHECKS")

pq_lbl = pd.read_parquet(MEGA, columns=['cig', 'label'])
check = agg[['cig', 'n_sospensioni', 'flag_sospensione', 'flag_sosp_giudiziaria',
             'durata_totale_sospensioni_gg', 'pct_durata_sospesa']].merge(pq_lbl, on='cig', how='right')

check['n_sospensioni']               = check['n_sospensioni'].fillna(0)
check['flag_sospensione']            = check['flag_sospensione'].fillna(0)
check['flag_sosp_giudiziaria']       = check['flag_sosp_giudiziaria'].fillna(0)
check['durata_totale_sospensioni_gg'] = check['durata_totale_sospensioni_gg'].fillna(0)

sc_lines = []
for lbl, name in [(1, 'CONDANNATI'), (0, 'SCAGIONATI')]:
    sub = check[check['label'] == lbl]
    line = (f"SC-SOSP | {name}: "
            f"flag_sosp={sub['flag_sospensione'].mean()*100:.1f}%, "
            f"n_sosp mediana={sub['n_sospensioni'].median():.1f}, "
            f"durata_med={sub['durata_totale_sospensioni_gg'].median():.0f}d, "
            f"pct_dur_med={sub['pct_durata_sospesa'].median():.3f}, "
            f"flag_giud={sub['flag_sosp_giudiziaria'].mean()*100:.1f}%")
    print(line)
    sc_lines.append(line)

# SC: CIG con sospensione ma durata_pianificata mancante
n_sosp_no_dur = (check['flag_sospensione'] == 1) & check['pct_durata_sospesa'].isna()
pct_missing = n_sosp_no_dur.sum() / (check['flag_sospensione'] == 1).sum() * 100
line_cov = f"SC-SOSP | CIG con sosp ma pct_durata mancante: {n_sosp_no_dur.sum():,} ({pct_missing:.1f}%)"
print(line_cov)
sc_lines.append(line_cov)

print("\nSTEP 6: Loading parquet and joining")
parquet = pd.read_parquet(MEGA)
print(f"  Parquet shape: {parquet.shape}")

# Drop tutte le colonne sospensioni pre-esistenti (idempotenza)
drop_cols = ['n_sospensioni', 'durata_totale_sospensioni_gg', 'flag_sosp_giudiziaria',
             'flag_sosp_sospetta', 'flag_sospensione', 'pct_durata_sospesa']
overlap = [c for c in drop_cols if c in parquet.columns]
if overlap:
    parquet = parquet.drop(columns=overlap)
    print(f"  Droppate {len(overlap)} colonne pre-esistenti: {overlap}")

join_cols = ['cig', 'n_sospensioni', 'durata_totale_sospensioni_gg',
             'flag_sosp_giudiziaria', 'flag_sospensione', 'pct_durata_sospesa']
result = parquet.merge(agg[join_cols], on='cig', how='left')

# Fillna: 0 per le colonne di conteggio/flag (assenza = 0)
result['n_sospensioni']               = result['n_sospensioni'].fillna(0).astype(int)
result['durata_totale_sospensioni_gg'] = result['durata_totale_sospensioni_gg'].fillna(0)
result['flag_sosp_giudiziaria']        = result['flag_sosp_giudiziaria'].fillna(0).astype(int)
result['flag_sospensione']             = result['flag_sospensione'].fillna(0).astype(int)
# pct_durata_sospesa: 0 se no sospensione (durata_totale=0), NaN se durata_pianificata mancante
result['pct_durata_sospesa'] = result['pct_durata_sospesa'].fillna(
    result['n_sospensioni'].map(lambda x: 0.0 if x == 0 else np.nan)
)

print(f"  Shape: {result.shape}")
print(f"  CIG con sospensioni: {(result['n_sospensioni']>0).sum():,} ({(result['n_sospensioni']>0).mean()*100:.1f}%)")
print(f"  pct_durata_sospesa non-null: {result['pct_durata_sospesa'].notna().sum():,} ({result['pct_durata_sospesa'].notna().mean()*100:.1f}%)")
print(f"  Label=1: {(result['label']==1).sum()} | Label=0: {(result['label']==0).sum()}")

result.to_parquet(MEGA, index=False)
print(f"\nParquet salvato. Shape: {result.shape}")

with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n[SOSPENSIONI v2]\n")
    for line in sc_lines:
        f.write(line + '\n')
print(f"SC results appended to: {SC_LOG}")
