"""
Build stati di avanzamento (SAL) features and join onto mega parquet.

Feature prodotte:
  - n_sal:          numero SAL registrati per CIG (0 se assente)
  - flag_in_ritardo: 1 se almeno 1 SAL IN RITARDO; 0 se SAL presente ma sempre IN LINEA/ANTICIPO;
                     NaN se nessun SAL (contratto non monitorato via SAL -- tipico forniture/servizi)
  - flag_proroga:   1 se almeno 1 SAL con giorni_proroga > 0 (0 se assente)

Note:
  - SAL e' istituto tipico dei LAVORI (11.4% coverage) vs forniture/servizi (1.4-2.4%)
  - n_giorni_scostamento escluso: mediana=p95=0, max=318k (errori), coperto da flag_in_ritardo
  - pct_sal_in_ritardo escluso: autocorrelazione temporale la rende quasi equivalente al flag
  - giorni_proroga_totale escluso: 97% zeri, coperto da flag_proroga
  - flag_in_ritardo: NaN != 0 -- distingue 'mai in ritardo' da 'nessun SAL'
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

print("STEP 1: Reading stati-avanzamento.csv")
df = pd.read_csv(f'{ANAC_DATA}/stati-avanzamento.csv', sep=';', dtype=str, low_memory=False)
df.columns = [c.lower().strip() for c in df.columns]
df['cig'] = df['cig'].str.strip()
df['progressivo_sal'] = pd.to_numeric(df['progressivo_sal'], errors='coerce')
df['giorni_proroga']  = pd.to_numeric(df['giorni_proroga'],  errors='coerce')
print(f"Shape: {df.shape} | CIG unici: {df['cig'].nunique():,}")

print("\nSTEP 2: Aggregation per CIG")

n_sal = df.groupby('cig').size().rename('n_sal')

# flag_in_ritardo: 1 se almeno 1 riga IN RITARDO, 0 altrimenti
# (NaN viene aggiunto in fase di join per CIG senza SAL)
flag_rit = (df[df['flag_ritardo'] == 'IN RITARDO']
            .groupby('cig').size().gt(0).astype('Int8').rename('flag_in_ritardo'))
# CIG con SAL ma nessun ritardo -> 0
all_cig_sal = pd.Series(1, index=n_sal.index)
flag_rit = flag_rit.reindex(all_cig_sal.index).fillna(pd.array([0], dtype='Int8')[0])

# flag_proroga: 1 se almeno 1 SAL con giorni_proroga > 0
flag_pror = (df[df['giorni_proroga'] > 0]
             .groupby('cig').size().gt(0).astype(int).rename('flag_proroga'))

agg = pd.concat([n_sal, flag_rit, flag_pror], axis=1).reset_index()
agg['flag_proroga'] = agg['flag_proroga'].fillna(0).astype(int)

print(f"  n_sal: mediana={agg['n_sal'].median():.0f}, p95={agg['n_sal'].quantile(.95):.0f}, max={agg['n_sal'].max()}")
print(f"  flag_in_ritardo=1: {(agg['flag_in_ritardo']==1).sum():,} CIG ({(agg['flag_in_ritardo']==1).mean()*100:.1f}%)")
print(f"  flag_in_ritardo=0: {(agg['flag_in_ritardo']==0).sum():,} CIG")
print(f"  flag_proroga=1:    {agg['flag_proroga'].sum():,} CIG ({agg['flag_proroga'].mean()*100:.1f}%)")

print("\nSANITY CHECKS")

pq_lbl = pd.read_parquet(MEGA, columns=['cig', 'label', 'oggetto_principale_contratto'])
check = agg.merge(pq_lbl, on='cig', how='right')
check['n_sal']       = check['n_sal'].fillna(0).astype(int)
check['flag_proroga'] = check['flag_proroga'].fillna(0).astype(int)
# flag_in_ritardo: NaN dove no SAL

sc_lines = []

# SC-SAL-1: signal su labeled
for lbl_val, name in [(1, 'CONDANNATI'), (0, 'SCAGIONATI')]:
    sub = check[check['label'] == lbl_val]
    n_ha_sal = (sub['n_sal'] > 0).sum()
    line = (f"SC-SAL | {name} (n={len(sub)}, con_sal={n_ha_sal}): "
            f"flag_ritardo={sub['flag_in_ritardo'].mean(skipna=True)*100:.1f}% (su {sub['flag_in_ritardo'].notna().sum()}), "
            f"flag_proroga={sub['flag_proroga'].mean()*100:.1f}%")
    print(line)
    sc_lines.append(line)

# SC-SAL-2: coverage per tipo contratto
print()
for tipo in ['LAVORI', 'SERVIZI', 'FORNITURE']:
    sub = pq_lbl[pq_lbl['oggetto_principale_contratto'] == tipo]
    pct = sub['cig'].isin(agg['cig']).mean() * 100
    line = f"SC-SAL | coverage {tipo}: {pct:.1f}%"
    print(line)
    sc_lines.append(line)

# SC-SAL-3: ritardo su CIG con proroga vs senza
con_pror   = agg[agg['flag_proroga'] == 1]['flag_in_ritardo']
senza_pror = agg[agg['flag_proroga'] == 0]['flag_in_ritardo']
line = (f"SC-SAL | ritardo su CIG con proroga: {con_pror.mean(skipna=True)*100:.1f}% "
        f"vs senza proroga: {senza_pror.mean(skipna=True)*100:.1f}%")
print(line)
sc_lines.append(line)

print("\nSTEP 4: Loading parquet and joining")
parquet = pd.read_parquet(MEGA)
print(f"  Parquet shape: {parquet.shape}")

overlap = [c for c in ['n_sal', 'flag_in_ritardo', 'flag_proroga'] if c in parquet.columns]
if overlap:
    parquet = parquet.drop(columns=overlap)
    print(f"  Droppate {len(overlap)} colonne pre-esistenti: {overlap}")

join_cols = ['cig', 'n_sal', 'flag_in_ritardo', 'flag_proroga']
result = parquet.merge(agg[join_cols], on='cig', how='left')

# n_sal e flag_proroga: 0 se assente
result['n_sal']        = result['n_sal'].fillna(0).astype(int)
result['flag_proroga'] = result['flag_proroga'].fillna(0).astype(int)
# flag_in_ritardo: NaN se nessun SAL (Int8 nullable)
result['flag_in_ritardo'] = result['flag_in_ritardo'].astype('Int8')

print(f"  Shape: {result.shape}")
print(f"  CIG con SAL (n_sal>0): {(result['n_sal']>0).sum():,} ({(result['n_sal']>0).mean()*100:.1f}%)")
print(f"  flag_in_ritardo=1: {(result['flag_in_ritardo']==1).sum():,}")
print(f"  flag_in_ritardo NaN: {result['flag_in_ritardo'].isna().sum():,} ({result['flag_in_ritardo'].isna().mean()*100:.1f}%)")
print(f"  flag_proroga=1: {(result['flag_proroga']==1).sum():,}")
print(f"  Label=1: {(result['label']==1).sum()} | Label=0: {(result['label']==0).sum()}")

result.to_parquet(MEGA, index=False)
print(f"\nParquet salvato. Shape: {result.shape}")

with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n[STATI DI AVANZAMENTO]\n")
    for line in sc_lines:
        f.write(line + '\n')
print(f"SC results appended to: {SC_LOG}")
