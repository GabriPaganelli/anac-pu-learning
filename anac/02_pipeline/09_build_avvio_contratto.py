"""
Build avvio contratto features and join onto mega parquet.

Feature prodotte:
  - lag_stipula_aggiudicazione_giorni: data_stipula - data_aggiudicazione_definitiva
  - durata_pianificata_giorni:         data_termine - data_inizio_effettiva
  - consegna_frazionata:               bool nullable (0/1/NA)
  - consegna_sotto_riserva:            bool nullable (0/1/NA)
  - date grezze tenute per ora (drop alla fine pipeline)

Decisioni:
  - lag negativi -> NaN (stessa causa SC-02: contratti retroattivi)
  - NaN nei bool = categoria propria, non imputata a 0
  - coppie scartate (STIP->INIZIO, STIP->CONS, INIZIO->CONS, STIP->TERM, AGG->TERM ecc.)
    per mediana=0d, r>0.96, o troppi negativi (>10%)
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

ANAC_DATA  = _DATA_RAW
LABELS_DIR = _LABELS
MEGA       = _MEGA
SC_LOG     = _SC_LOG

sc_results = []

DATE_MIN = pd.Timestamp('1990-01-01')
DATE_MAX = pd.Timestamp('2030-12-31')

def parse_date(series):
    d = pd.to_datetime(series, errors='coerce')
    d = d.where(d >= DATE_MIN, pd.NaT)
    d = d.where(d <= DATE_MAX, pd.NaT)
    return d

print("STEP 1: Reading avvio-contratto.csv")
av = pd.read_csv(f'{ANAC_DATA}/avvio-contratto.csv', sep=';', dtype=str, low_memory=False)
av.columns = [c.lower().strip() for c in av.columns]
print(f"Shape: {av.shape}")
av['cig'] = av['cig'].str.strip()
print(f"CIG unici: {av['cig'].nunique():,}")
print(f"CIG con >1 riga: {(av['cig'].value_counts() > 1).sum():,}")

print("\nSTEP 2: Parsing dates")
date_cols = ['data_stipula_contratto', 'data_termine_contrattuale',
             'data_verbale_consegna_definitiva', 'data_inizio_effettiva']
for col in date_cols:
    av[col] = parse_date(av[col])
    print(f"  {col}: {av[col].notna().sum():,} non-null ({av[col].notna().mean()*100:.1f}%)")

print("\nSTEP 3: Boolean columns (NaN = categoria)")
for col in ['consegna_frazionata', 'consegna_sotto_riserva']:
    av[col] = pd.to_numeric(av[col], errors='coerce')
    av[col] = av[col].astype('Int8')  # nullable int: 0/1/NA
    vc = av[col].value_counts(dropna=False)
    print(f"  {col}: {vc.to_dict()}")

print("\nSTEP 4: Row selection for multi-row CIGs")
condannati_df = pd.read_csv(f'{LABELS_DIR}/cig_condannati.csv', dtype=str)
condannati_df.columns = [c.lower().strip() for c in condannati_df.columns]
condannati_set = set(condannati_df['cig'].str.strip())

fine_df = pd.read_csv(f'{ANAC_DATA}/fine-contratto.csv', sep=';', dtype=str, low_memory=False)
fine_df.columns = [c.lower().strip() for c in fine_df.columns]
fine_condannati = (fine_df[fine_df['cig'].isin(condannati_set) & fine_df['id_aggiudicazione'].notna()]
                   [['cig', 'id_aggiudicazione']].drop_duplicates('cig'))
cig_to_id_agg = dict(zip(fine_condannati['cig'], fine_condannati['id_aggiudicazione'].str.strip()))

cig_counts = av['cig'].value_counts()
multi_cigs  = set(cig_counts[cig_counts > 1].index)
single_cigs = set(cig_counts[cig_counts == 1].index)

def select_row(group):
    cig = group['cig'].iloc[0]
    if cig in condannati_set and cig in cig_to_id_agg:
        target = cig_to_id_agg[cig]
        match = group[group['id_aggiudicazione'].astype(str).str.strip() == target]
        if len(match) >= 1:
            return match.iloc[[0]]
    return group.iloc[[0]]

single_df = av[av['cig'].isin(single_cigs)]
multi_df  = av[av['cig'].isin(multi_cigs)]
selected_multi = multi_df.groupby('cig', group_keys=False).apply(select_row)
av = pd.concat([single_df, selected_multi], ignore_index=True)
print(f"  Shape dopo selezione: {av.shape} (atteso: {len(cig_counts):,} righe)")

print("\nSTEP 5: durata_pianificata_giorni = data_termine - data_inizio_effettiva")
dur = (av['data_termine_contrattuale'] - av['data_inizio_effettiva']).dt.days
n_neg = (dur < 0).sum()
av['durata_pianificata_giorni'] = dur.where(dur >= 0, np.nan)
print(f"  Non-null: {av['durata_pianificata_giorni'].notna().sum():,} ({av['durata_pianificata_giorni'].notna().mean()*100:.1f}%)")
print(f"  Negativi (-> NaN): {n_neg:,}")
print(f"  Mediana: {av['durata_pianificata_giorni'].median():.0f}d | p95: {av['durata_pianificata_giorni'].quantile(0.95):.0f}d")

print("\nSTEP 6: lag_stipula_aggiudicazione_giorni = data_stipula - data_aggiudicazione_definitiva")
pq_agg = pd.read_parquet(MEGA, columns=['cig', 'data_aggiudicazione_definitiva'])
av = av.merge(pq_agg, on='cig', how='left')
av['data_aggiudicazione_definitiva'] = parse_date(av['data_aggiudicazione_definitiva'])

lag_stip = (av['data_stipula_contratto'] - av['data_aggiudicazione_definitiva']).dt.days
n_neg_stip = (lag_stip < 0).sum()
av['lag_stipula_aggiudicazione_giorni'] = lag_stip.where(lag_stip >= 0, np.nan)
print(f"  Non-null: {av['lag_stipula_aggiudicazione_giorni'].notna().sum():,} ({av['lag_stipula_aggiudicazione_giorni'].notna().mean()*100:.1f}%)")
print(f"  Negativi (-> NaN): {n_neg_stip:,} ({n_neg_stip/lag_stip.notna().sum()*100:.1f}%)")
print(f"  Mediana: {av['lag_stipula_aggiudicazione_giorni'].median():.0f}d | p95: {av['lag_stipula_aggiudicazione_giorni'].quantile(0.95):.0f}d")
av.drop(columns=['data_aggiudicazione_definitiva'], inplace=True)

print("\nSANITY CHECKS")

# SC-06a: data_inizio_effettiva <= data_termine_contrattuale
mask_06a = av['data_inizio_effettiva'].notna() & av['data_termine_contrattuale'].notna()
viol_06a = (av.loc[mask_06a, 'data_inizio_effettiva'] > av.loc[mask_06a, 'data_termine_contrattuale']).sum()
pct_06a  = viol_06a / mask_06a.sum() * 100
print(f"\nSC-06a | data_inizio > data_termine: {viol_06a:,} ({pct_06a:.2f}%) su {mask_06a.sum():,} confrontabili")
status_06a = "OK" if pct_06a < 1 else "WARNING"
sc_results.append(("SC-06a", status_06a, f"inizio>termine={viol_06a:,} ({pct_06a:.2f}%)"))

# SC-06b: data_aggiudicazione <= data_stipula (stessa causa SC-02)
pq_agg2 = pd.read_parquet(MEGA, columns=['cig', 'data_aggiudicazione_definitiva'])
av_sc = av[['cig', 'data_stipula_contratto']].merge(pq_agg2, on='cig', how='left')
av_sc['data_aggiudicazione_definitiva'] = parse_date(av_sc['data_aggiudicazione_definitiva'])
mask_06b = av_sc['data_stipula_contratto'].notna() & av_sc['data_aggiudicazione_definitiva'].notna()
viol_06b  = (av_sc.loc[mask_06b, 'data_aggiudicazione_definitiva'] > av_sc.loc[mask_06b, 'data_stipula_contratto']).sum()
pct_06b   = viol_06b / mask_06b.sum() * 100
print(f"SC-06b | data_aggiudicazione > data_stipula: {viol_06b:,} ({pct_06b:.2f}%) su {mask_06b.sum():,} confrontabili")
print(f"        (stessa causa SC-02: contratti caricati retroattivamente - lag gia' impostato a NaN)")
status_06b = "HANDLED" if pct_06b < 10 else "WARNING"
sc_results.append(("SC-06b", status_06b,
    f"agg>stipula={viol_06b:,} ({pct_06b:.2f}%) | lag_stipula_aggiudicazione negativi -> NaN"))

print("\nSTEP 7: Loading parquet and joining")

keep = [
    'cig',
    'lag_stipula_aggiudicazione_giorni',
    'durata_pianificata_giorni',
    'consegna_frazionata',
    'consegna_sotto_riserva',
    'data_stipula_contratto',
    'data_inizio_effettiva',
    'data_verbale_consegna_definitiva',
    'data_termine_contrattuale',
]
av_join = av[[c for c in keep if c in av.columns]].copy()

parquet = pd.read_parquet(MEGA)
print(f"  Parquet shape: {parquet.shape}")

# Drop pre-existing columns
overlap = [c for c in av_join.columns if c != 'cig' and c in parquet.columns]
if overlap:
    parquet = parquet.drop(columns=overlap)
    print(f"  Droppate {len(overlap)} colonne pre-esistenti: {overlap}")

result = parquet.merge(av_join, on='cig', how='left')
print(f"  Shape dopo join: {result.shape}")
print(f"  CIG con dati avvio (lag_stipula non-null): {result['lag_stipula_aggiudicazione_giorni'].notna().sum():,} ({result['lag_stipula_aggiudicazione_giorni'].notna().mean()*100:.1f}%)")
print(f"  durata_pianificata non-null: {result['durata_pianificata_giorni'].notna().sum():,} ({result['durata_pianificata_giorni'].notna().mean()*100:.1f}%)")
print(f"  Label=1 rimasti: {(result['label']==1).sum()}")
print(f"  Label=0 rimasti: {(result['label']==0).sum()}")

print("\nSaving parquet...")
result.to_parquet(MEGA, index=False)
print(f"Parquet salvato. Shape: {result.shape}")

# SC log
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n\n{'='*60}\n")
    f.write(f"Run: {timestamp} [build_avvio_contratto]\n")
    f.write(f"Parquet shape: {result.shape}\n")
    f.write("\nSanity Checks:\n")
    for sc_id, status, msg in sc_results:
        f.write(f"  {sc_id:12s} [{status}]  {msg}\n")
print(f"SC results appended to: {SC_LOG}")
