"""
Build flag_subappalto combinato e join sul mega parquet.

Logica OR:
  flag_subappalto = 1 se:
    - flag_subappalto='True' in AGGIUDICAZIONI (subappalto era autorizzato/ammesso nel bando)
    - OR cig presente in subappalti.csv (subappalto effettivamente registrato)
  0 altrimenti (nessuna delle due fonti lo indica)

Semantica: cattura sia i contratti dove il subappalto era previsto contrattualmente
sia quelli dove e' stato effettivamente rendicontato ad ANAC.
Dataset subappalti.csv non produce feature ulteriori (cod_categoria/cpv troppo sparsi,
n_subappalti segnale debole con 1.2% coverage).
"""
import pandas as pd
import numpy as np
import sys, os
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

print("STEP 1: Raccolta CIG da entrambe le fonti")

# Fonte 1: flag_subappalto=True da AGGIUDICAZIONI
agg = pd.read_csv(f'{ANAC_DATA}/aggiudicazioni.csv', sep=';', dtype=str,
                  usecols=['cig', 'flag_subappalto'], low_memory=False)
agg.columns = [c.lower().strip() for c in agg.columns]
agg['cig'] = agg['cig'].str.strip()
agg = agg.drop_duplicates('cig')
cig_agg_true = set(agg[agg['flag_subappalto'] == 'True']['cig'])
print(f"  flag_subappalto=True (AGGIUDICAZIONI): {len(cig_agg_true):,}")

# Fonte 2: presenza in subappalti.csv
sub = pd.read_csv(f'{ANAC_DATA}/subappalti.csv', sep=';', dtype=str,
                  usecols=['cig'], low_memory=False)
sub['cig'] = sub['cig'].str.strip()
cig_sub = set(sub['cig'].unique())
print(f"  CIG in subappalti.csv:                 {len(cig_sub):,}")
print(f"  Overlap:                               {len(cig_agg_true & cig_sub):,}")

cig_union = cig_agg_true | cig_sub
print(f"  UNION (OR):                            {len(cig_union):,}")

print("\nSANITY CHECKS")

pq_lbl = pd.read_parquet(MEGA, columns=['cig', 'label'])
pq_lbl['flag_subappalto'] = pq_lbl['cig'].isin(cig_union).astype(int)

in_parquet = len(cig_union & set(pq_lbl['cig']))
print(f"CIG union nel parquet: {in_parquet:,} ({in_parquet/len(pq_lbl)*100:.1f}%)")

sc_lines = []
lbl = pq_lbl[pq_lbl['label'].isin([0, 1])]
for lv, name in [(1, 'CONDANNATI'), (0, 'SCAGIONATI')]:
    sub_lbl = lbl[lbl['label'] == lv]
    pct = sub_lbl['flag_subappalto'].mean() * 100
    line = f"SC-SUB | {name}: flag_subappalto=1 -> {pct:.1f}% (n={len(sub_lbl)})"
    print(line)
    sc_lines.append(line)

# SC: solo da subappalti.csv (escluso aggiudicazioni)
only_sub = cig_sub - cig_agg_true
line = f"SC-SUB | CIG solo da subappalti.csv (non in flag agg): {len(only_sub):,}"
print(line)
sc_lines.append(line)

print("\nSTEP 3: Loading parquet and joining")
parquet = pd.read_parquet(MEGA)
print(f"  Parquet shape: {parquet.shape}")

if 'flag_subappalto' in parquet.columns:
    parquet = parquet.drop(columns=['flag_subappalto'])
    print("  Droppata colonna pre-esistente: flag_subappalto")

parquet['flag_subappalto'] = parquet['cig'].isin(cig_union).astype(int)

print(f"  Shape: {parquet.shape}")
print(f"  flag_subappalto=1: {parquet['flag_subappalto'].sum():,} ({parquet['flag_subappalto'].mean()*100:.1f}%)")
print(f"  Label=1: {(parquet['label']==1).sum()} | Label=0: {(parquet['label']==0).sum()}")

parquet.to_parquet(MEGA, index=False)
print(f"\nParquet salvato. Shape: {parquet.shape}")

with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n[SUBAPPALTI]\n")
    for line in sc_lines:
        f.write(line + '\n')
print(f"SC results appended to: {SC_LOG}")
