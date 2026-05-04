"""
15_build_collaudo.py
Aggiunge esito_collaudo al mega parquet.

Dataset: collaudo.csv
  - 648,862 righe, 648,674 CIG unici (6.5% del parquet)
  - Colonne usate: cig, esito_collaudo
  - 185 CIG con >1 riga: kept='first' (ordine CSV — riproducibile con l'originale)
  - Valori: POSITIVO (98.6%), NEGATIVO (1.4%)

Nota leakage: NESSUNO. Il collaudo non entra nella costruzione delle label.
A differenza di fine-contratto.csv (droppato), questo dataset registra l'esito
tecnico dell'opera, non l'interruzione anticipata per reati.

Nota feature: NaN = non registrato (93.5% dei CIG) — NON equivale a collaudo negativo.
Trattato come categoria MISSING nel preprocessing.

Signal su labeled:
  POSITIVO: COND 8.8% vs SCAG 0.8% — ratio 2.7x (selection bias: vedi decisions_log)
"""

import pandas as pd
import os as _os

_BASE     = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_RAW = _os.path.join(_BASE, "data", "raw")
_MEGA     = _os.path.join(_BASE, "output", "parquet", "bando_cig_all.parquet")

COLLAUDO_CSV = _os.path.join(_DATA_RAW, "collaudo.csv")

print("BUILD COLLAUDO")

coll = pd.read_csv(COLLAUDO_CSV, sep=';', usecols=['cig', 'esito_collaudo'],
                   dtype={'cig': str})

# 185 CIG duplicati: keep='first' (riproducibile con l'originale)
before = len(coll)
coll = coll.drop_duplicates('cig', keep='first')
print(f"  Righe: {before:,} -> {len(coll):,} dopo dedup (CIG unici: {coll.cig.nunique():,})")
print(f"  esito_collaudo: {coll.esito_collaudo.value_counts().to_dict()}")

pq = pd.read_parquet(_MEGA)
print(f"\n  Parquet: {pq.shape[0]:,} righe × {pq.shape[1]} colonne")

# Rimuovi colonna precedente se presente (idempotenza)
if 'esito_collaudo' in pq.columns:
    pq = pq.drop(columns=['esito_collaudo'])
    print("  esito_collaudo precedente rimossa (idempotenza)")

pq = pq.merge(coll[['cig', 'esito_collaudo']], on='cig', how='left')

# Verifica
n_joined = pq['esito_collaudo'].notna().sum()
pct = n_joined / len(pq) * 100
print(f"\n  Join: {n_joined:,} CIG con esito_collaudo ({pct:.1f}%)")
print(f"  Distribuzione: {pq.esito_collaudo.value_counts(dropna=False).to_dict()}")

# Signal su labeled
labeled = pq[pq['label'].notna()]
if len(labeled) > 0:
    sig = labeled.groupby('label')['esito_collaudo'].value_counts(dropna=False)
    print(f"\n  Signal su labeled (n={len(labeled)}):")
    print(f"  {sig.to_dict()}")

print(f"\n  Colonne finali: {pq.shape[1]}")

pq.to_parquet(_MEGA, index=False)
print(f"\n  Salvato: {_MEGA}")
