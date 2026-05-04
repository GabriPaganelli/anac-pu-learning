"""
Build varianti features and join onto mega parquet.

Feature prodotte:
  - n_varianti:                  conteggio varianti per CIG (0 se assente)
  - flag_variante_sostanziale:   1 se almeno una variante con codice non-amministrativo (0/1, no NaN)
  - delta_prima_variante_giorni: min(data_approvazione_variante) - data_aggiudicazione_definitiva

Codici ESCLUSI da flag_variante_sostanziale (amministrativi/leciti):
  17 = previste nei documenti di gara
  18 = proroga tecnica
  19 = rinnovo previsto
  22 = revisione dei prezzi
  96 = revisione prezzi/opzioni (D.Lgs.50)
  98 = esplicitamente NON sostanziali
  99 = non specificato

Assenza dal file = 0 varianti (non NaN): l'assenza e' semanticamente informativa.
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

CODICI_NON_SOSTANZIALI = {17.0, 18.0, 19.0, 22.0, 96.0, 98.0, 99.0}

print("STEP 1: Reading varianti.csv")
df = pd.read_csv(f'{ANAC_DATA}/varianti.csv', sep=';', dtype=str, low_memory=False)
df.columns = [c.lower().strip() for c in df.columns]
df['cig'] = df['cig'].str.strip()
print(f"Shape: {df.shape}")

pq_cigs = set(pd.read_parquet(MEGA, columns=['cig'])['cig'])
print(f"CIG unici: {df['cig'].nunique():,}")

print("\nSTEP 2: Parsing")
df['cod'] = pd.to_numeric(df['cod_motivo_variante'], errors='coerce')
df['data_approvazione_variante'] = pd.to_datetime(df['data_approvazione_variante'], errors='coerce')

print("\nSTEP 3: Aggregation per CIG")

n_var = df.groupby('cig').size().rename('n_varianti')

flag_sost = (
    df[~df['cod'].isin(CODICI_NON_SOSTANZIALI)]
    .groupby('cig').size().gt(0).astype(int).rename('flag_variante_sostanziale')
)

first_date = df.groupby('cig')['data_approvazione_variante'].min().rename('_data_prima_variante')

agg = pd.concat([n_var, flag_sost, first_date], axis=1)
agg['flag_variante_sostanziale'] = agg['flag_variante_sostanziale'].fillna(0).astype(int)
agg = agg.reset_index()

print(f"  n_varianti mediana: {agg['n_varianti'].median():.0f}, p95: {agg['n_varianti'].quantile(0.95):.0f}, max: {agg['n_varianti'].max()}")
print(f"  flag_variante_sostanziale=1: {agg['flag_variante_sostanziale'].sum():,} CIG ({agg['flag_variante_sostanziale'].mean()*100:.1f}%)")

print("\nSTEP 4: delta_prima_variante_giorni")
pq_agg = pd.read_parquet(MEGA, columns=['cig', 'data_aggiudicazione_definitiva'])
agg = agg.merge(pq_agg, on='cig', how='left')
# Clamp date fuori range (evita overflow int64 su date anomale)
date_min, date_max = pd.Timestamp('2000-01-01'), pd.Timestamp('2040-01-01')
d1 = agg['_data_prima_variante'].clip(lower=date_min, upper=date_max)
d2 = agg['data_aggiudicazione_definitiva'].clip(lower=date_min, upper=date_max)
delta = (d1 - d2).dt.days
n_neg = (delta < 0).sum()
agg['delta_prima_variante_giorni'] = delta.where(delta >= 0, np.nan)
print(f"  Non-null: {agg['delta_prima_variante_giorni'].notna().sum():,}")
print(f"  Negativi (-> NaN): {n_neg:,}")
print(f"  Mediana: {agg['delta_prima_variante_giorni'].median():.0f}d | p95: {agg['delta_prima_variante_giorni'].quantile(0.95):.0f}d")
print(f"  Varianti entro 30gg aggiudicazione: {(agg['delta_prima_variante_giorni'] <= 30).sum():,}")

print("\nSANITY CHECKS")

# SC-VAR-1: delta negativo
sc_var1_n = n_neg
sc_var1_pct = n_neg / agg['delta_prima_variante_giorni'].notna().sum() * 100 if agg['delta_prima_variante_giorni'].notna().sum() else 0
print(f"SC-VAR-1 | delta_prima_variante < 0: {sc_var1_n} ({sc_var1_pct:.1f}%) -> NaN")

# SC-VAR-2: confronto n_varianti tra labeled
pq_full = pd.read_parquet(MEGA, columns=['cig','label'])
agg_check = agg[['cig','n_varianti','flag_variante_sostanziale']].merge(pq_full, on='cig', how='right')
agg_check['n_varianti'] = agg_check['n_varianti'].fillna(0)
agg_check['flag_variante_sostanziale'] = agg_check['flag_variante_sostanziale'].fillna(0)
for lbl, name in [(1,'CONDANNATI'),(0,'SCAGIONATI')]:
    sub = agg_check[agg_check['label']==lbl]
    print(f"SC-VAR-2 | {name}: n_varianti mediana={sub['n_varianti'].median():.1f}, flag_sost={sub['flag_variante_sostanziale'].mean()*100:.1f}%")

print("\nSTEP 6: Loading parquet and joining")
parquet = pd.read_parquet(MEGA)
print(f"  Parquet shape: {parquet.shape}")

# Rimuovi colonne pre-esistenti per idempotenza
overlap = [c for c in ['n_varianti','flag_variante_sostanziale','delta_prima_variante_giorni'] if c in parquet.columns]
if overlap:
    parquet = parquet.drop(columns=overlap)

join_cols = ['cig','n_varianti','flag_variante_sostanziale','delta_prima_variante_giorni']
result = parquet.merge(agg[join_cols], on='cig', how='left')

# Assenza = 0 varianti (non NaN)
result['n_varianti'] = result['n_varianti'].fillna(0).astype(int)
result['flag_variante_sostanziale'] = result['flag_variante_sostanziale'].fillna(0).astype(int)
# delta resta NaN dove non computabile

print(f"  Shape dopo join: {result.shape}")
print(f"  CIG con varianti (n>0): {(result['n_varianti']>0).sum():,} ({(result['n_varianti']>0).mean()*100:.1f}%)")
print(f"  Label=1 rimasti: {(result['label']==1).sum()}")
print(f"  Label=0 rimasti: {(result['label']==0).sum()}")

result.to_parquet(MEGA, index=False)
print(f"\nParquet salvato. Shape: {result.shape}")

# SC log
with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n[VARIANTI]\n")
    f.write(f"SC-VAR-1 | delta_prima_variante < 0: {sc_var1_n} ({sc_var1_pct:.1f}%) -> NaN\n")
    for lbl, name in [(1,'CONDANNATI'),(0,'SCAGIONATI')]:
        sub = agg_check[agg_check['label']==lbl]
        f.write(f"SC-VAR-2 | {name}: n_varianti mediana={sub['n_varianti'].median():.1f}, flag_sost={sub['flag_variante_sostanziale'].mean()*100:.1f}%\n")
print(f"SC results appended to: {SC_LOG}")
