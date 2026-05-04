"""
Build tipo_lavorazione_macro e join sul mega parquet.

Lookup cod_tipo_lavorazione (fonte: dati.anticorruzione.it + github.com/anticorruzione/npa):
  1=ACQUISTO, 2=LEASING, 3=NOLEGGIO, 4=ACQUISTO_RISCATTO, 5=MISTO -> ACQUISIZIONE (anomali)
  6=COSTRUZIONE, 7=DEMOLIZIONE                                       -> COSTRUZIONE
  8=RECUPERO, 9=RISTRUTTURAZIONE, 10=RESTAURO                       -> RISANAMENTO
  11=MANUTENZIONE (dep.2014), 12=MAN.ORDINARIA,
  13=MAN.STRAORDINARIA, 14=MAN.ORD+STRAORD (dep.2014)               -> MANUTENZIONE

Feature prodotta:
  tipo_lavorazione_macro: categoriale COSTRUZIONE / RISANAMENTO / MANUTENZIONE
    NaN se CIG assente dal dataset o unica macro = ACQUISIZIONE (anomala per LAVORI)

Per CIG con piu lavorazioni: priorita COSTRUZIONE > RISANAMENTO > MANUTENZIONE > ACQUISIZIONE
(opera nuova domina sulla manutenzione come caratterizzazione principale del contratto)

Note:
  - Cod=11 (MANUTENZIONE generica) deprecato dal 2014: tutti contratti pre-2014, nessun
    artefatto semantico. Segnale 999x era interamente dovuto alla storicita' del codice.
  - Cod=1-5 (ACQUISIZIONE): 201 CIG nel parquet, 0 labeled. Trattati come NaN.
  - Coverage 9.1% esclusivamente su LAVORI (65.6% dei LAVORI ha lavorazioni registrate).
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

LOOKUP = {
    1:'ACQUISIZIONE', 2:'ACQUISIZIONE', 3:'ACQUISIZIONE',
    4:'ACQUISIZIONE', 5:'ACQUISIZIONE',
    6:'COSTRUZIONE',  7:'COSTRUZIONE',
    8:'RISANAMENTO',  9:'RISANAMENTO',  10:'RISANAMENTO',
    11:'MANUTENZIONE',12:'MANUTENZIONE',
    13:'MANUTENZIONE',14:'MANUTENZIONE'
}
PRIORITY = {'COSTRUZIONE':1, 'RISANAMENTO':2, 'MANUTENZIONE':3, 'ACQUISIZIONE':4}

print("STEP 1: Reading lavorazioni.csv")
df = pd.read_csv(f'{ANAC_DATA}/lavorazioni.csv', sep=';', dtype=str, low_memory=False)
df.columns = [c.lower().strip() for c in df.columns]
df['cig'] = df['cig'].str.strip()
df['cod'] = pd.to_numeric(df['cod_tipo_lavorazione'], errors='coerce')
df['macro'] = df['cod'].map(LOOKUP)
print(f"Shape: {df.shape} | CIG unici: {df['cig'].nunique():,}")

print("\nSTEP 2: Aggregazione per CIG (priorita COSTRUZIONE > RISANAMENTO > MANUTENZIONE)")
df['prio'] = df['macro'].map(PRIORITY).fillna(99)
best = (df.sort_values('prio')
          .groupby('cig')['macro']
          .first()
          .rename('tipo_lavorazione_macro'))

# ACQUISIZIONE -> NaN (anomala, 0 labeled)
best = best.where(best != 'ACQUISIZIONE', other=np.nan)

print(f"  Distribuzione:")
print(best.value_counts(dropna=False).to_string())

print("\nSANITY CHECKS")

pq_lbl = pd.read_parquet(MEGA, columns=['cig', 'label', 'oggetto_principale_contratto'])
check = pq_lbl.merge(best.reset_index(), on='cig', how='left')

# SC-LAV-1: coverage per tipo contratto
sc_lines = []
for tipo in ['LAVORI', 'SERVIZI', 'FORNITURE']:
    sub = check[check['oggetto_principale_contratto'] == tipo]
    pct = sub['tipo_lavorazione_macro'].notna().mean() * 100
    line = f"SC-LAV | coverage {tipo}: {pct:.1f}%"
    print(line); sc_lines.append(line)

# SC-LAV-2: signal per macro su labeled
print()
lbl = check[check['label'].isin([0, 1])]
tot_c = (lbl['label'] == 1).sum()
tot_s = (lbl['label'] == 0).sum()
for macro in ['COSTRUZIONE', 'RISANAMENTO', 'MANUTENZIONE']:
    cond = ((lbl['label']==1) & (lbl['tipo_lavorazione_macro']==macro)).sum()
    scag = ((lbl['label']==0) & (lbl['tipo_lavorazione_macro']==macro)).sum()
    ratio = (cond/tot_c) / (scag/tot_s) if scag > 0 else float('inf')
    line = (f"SC-LAV | {macro}: COND={cond} ({cond/tot_c*100:.1f}%) "
            f"SCAG={scag} ({scag/tot_s*100:.1f}%) ratio={ratio:.1f}x")
    print(line); sc_lines.append(line)

print("\nSTEP 4: Loading parquet and joining")
parquet = pd.read_parquet(MEGA)
print(f"  Parquet shape: {parquet.shape}")

if 'tipo_lavorazione_macro' in parquet.columns:
    parquet = parquet.drop(columns=['tipo_lavorazione_macro'])
    print("  Droppata colonna pre-esistente.")

result = parquet.merge(best.reset_index(), on='cig', how='left')
# Lascia NaN dove assente (non ha senso imputare un tipo di lavorazione)

print(f"  Shape: {result.shape}")
print(f"  tipo_lavorazione_macro non-null: {result['tipo_lavorazione_macro'].notna().sum():,} "
      f"({result['tipo_lavorazione_macro'].notna().mean()*100:.1f}%)")
print(f"  Distribuzione:")
print(result['tipo_lavorazione_macro'].value_counts(dropna=False).head(5).to_string())
print(f"  Label=1: {(result['label']==1).sum()} | Label=0: {(result['label']==0).sum()}")

result.to_parquet(MEGA, index=False)
print(f"\nParquet salvato. Shape: {result.shape}")

with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n[LAVORAZIONI]\n")
    for line in sc_lines:
        f.write(line + '\n')
print(f"SC results appended to: {SC_LOG}")
