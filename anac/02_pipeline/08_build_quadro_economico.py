"""
Build quadro economico features and join onto mega parquet.

Feature prodotte:
  - pct_riserva_base:      somme_disp(BASE_ASTA) / sum_6col(BASE_ASTA)     [ante, ~48% coverage]
  - pct_overrun_core:      sum_6col(CONSUNTIVO)  / sum_6col(BASE_ASTA)     [ex post, ~6.4%]
  - pct_riserva_consumata: somme_disp(CONSUNTIVO)/ sum_6col(BASE_ASTA)     [ex post, ~6.4%]

6 colonne core: importo_lavori, importo_forniture, importo_servizi,
                importo_progettazione, importo_sicurezza,
                ulteriori_oneri_non_soggetti_ribasso

Altre operazioni:
  - importo_sicurezza: fallback da QE BASE_ASTA dove BANDO CIG e' NaN
  - Drop importo_variante_totale (rimane pct_overrun_variante)
  - SC-09: n_lotti_componenti vs count aggiudicazioni per CIG
  - SC-11: importo_sicurezza QE vs BANDO CIG (informativo)
  - SC-12: importo_lotto vs sum_6col BASE_ASTA (informativo)
"""
import pandas as pd
import numpy as np

import os as _os
_BASE      = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA_RAW  = _os.path.join(_BASE, "data", "raw")
_LABELS    = _os.path.join(_BASE, "labels")
_MEGA      = _os.path.join(_BASE, "output", "parquet", "bando_cig_all.parquet")
_SC_LOG    = _os.path.join(_BASE, "output", "sc_results.txt")

ANAC_DATA = _DATA_RAW
MEGA      = _MEGA
SC_LOG    = _SC_LOG

CORE_COLS = ['importo_lavori','importo_forniture','importo_servizi',
             'importo_progettazione','importo_sicurezza',
             'ulteriori_oneri_non_soggetti_ribasso']

print("STEP 1: Loading quadro-economico.csv")
qe = pd.read_csv(f'{ANAC_DATA}/quadro-economico.csv', sep=';', dtype=str, low_memory=False)
qe['cig'] = qe['cig'].str.strip()
for c in CORE_COLS + ['somme_a_disposizione']:
    qe[c] = pd.to_numeric(qe[c], errors='coerce')
print(f"Shape: {qe.shape} | CIG unici: {qe['cig'].nunique():,}")

pq = pd.read_parquet(MEGA)
pq_cigs = set(pq['cig'])

ba = qe[qe['descrizione_evento'] == 'BASE_ASTA'].copy()
co = qe[qe['descrizione_evento'] == 'CONSUNTIVO'].copy()
print(f"BASE_ASTA: {len(ba):,} righe | CONSUNTIVO: {len(co):,} righe")

# Aggregazione per CIG (sum per multi-riga)
def agg_cig(df, cols):
    return df.groupby('cig')[cols].sum(min_count=1)  # NaN se tutte NaN

ba_agg = agg_cig(ba, CORE_COLS + ['somme_a_disposizione'])
co_agg = agg_cig(co, CORE_COLS + ['somme_a_disposizione'])

ba_agg['sum_core'] = ba_agg[CORE_COLS].sum(axis=1, skipna=True)
co_agg['sum_core'] = co_agg[CORE_COLS].sum(axis=1, skipna=True)

# STEP 2: SC-09
print("\nSC-09: n_lotti_componenti vs count aggiudicazioni per CIG")
agg_raw = pd.read_csv(f'{ANAC_DATA}/aggiudicazioni.csv', sep=';',
    usecols=['cig'], dtype=str, low_memory=False)
agg_raw['cig'] = agg_raw['cig'].str.strip()
count_agg = agg_raw.groupby('cig').size().rename('n_agg_raw')
check09 = pq[['cig','n_lotti_componenti']].merge(
    count_agg.reset_index(), on='cig', how='left')
check09['n_lotti_componenti'] = pd.to_numeric(check09['n_lotti_componenti'], errors='coerce')
confrontabili = check09.dropna(subset=['n_lotti_componenti','n_agg_raw'])
discrepanza = (confrontabili['n_lotti_componenti'] != confrontabili['n_agg_raw']).sum()
pct_disc = discrepanza / len(confrontabili) * 100
print(f"  Confrontabili: {len(confrontabili):,}")
print(f"  Discrepanze: {discrepanza:,} ({pct_disc:.1f}%)")
sc09_line = f"SC-09 | n_lotti vs count_agg: {discrepanza:,} discrepanze ({pct_disc:.1f}%) su {len(confrontabili):,} confrontabili"
print(f"  {sc09_line}")

# STEP 3: SC-11
print("\nSC-11: importo_sicurezza QE BASE_ASTA vs BANDO CIG")
ba_sic = ba_agg['importo_sicurezza'].rename('sic_qe')
check11 = pq[['cig','importo_sicurezza']].set_index('cig').join(ba_sic, how='inner')
check11 = check11.dropna()
check11['importo_sicurezza'] = pd.to_numeric(check11['importo_sicurezza'], errors='coerce')
check11 = check11.dropna()
tol = (check11['importo_sicurezza'] - check11['sic_qe']).abs()
match = (tol <= check11['importo_sicurezza'].abs() * 0.01).sum()  # tolleranza 1%
pct_match = match / len(check11) * 100
print(f"  Confrontabili: {len(check11):,}")
print(f"  Match (delta <1%): {match:,} ({pct_match:.1f}%)")
sc11_line = f"SC-11 | importo_sicurezza QE vs BANDO CIG: {pct_match:.1f}% match (tol 1%) su {len(check11):,} CIG - solo informativo"

# Applica fallback importo_sicurezza dove BANDO CIG e' NaN
n_nan_before = pq['importo_sicurezza'].isna().sum()
pq = pq.set_index('cig')
pq['importo_sicurezza'] = pq['importo_sicurezza'].fillna(ba_sic)
n_filled = n_nan_before - pq['importo_sicurezza'].isna().sum()
pq = pq.reset_index()
print(f"  importo_sicurezza fallback da QE: +{n_filled:,} valori")

# STEP 4: SC-12
print("\nSC-12: importo_lotto vs sum_6col BASE_ASTA (informativo)")
check12 = pq[['cig','importo_lotto']].set_index('cig').join(
    ba_agg['sum_core'].rename('sum_qe'), how='inner')
check12 = check12.dropna()
check12['importo_lotto'] = pd.to_numeric(check12['importo_lotto'], errors='coerce')
check12 = check12.dropna()
delta12 = (check12['importo_lotto'] - check12['sum_qe']).abs() / check12['importo_lotto'].abs()
match12 = (delta12 <= 0.05).sum()
pct12 = match12 / len(check12) * 100
print(f"  Confrontabili: {len(check12):,}")
print(f"  Match (delta <5%): {match12:,} ({pct12:.1f}%) - solo informativo")
sc12_line = f"SC-12 | importo_lotto vs sum QE BASE_ASTA: {pct12:.1f}% match (tol 5%) su {len(check12):,} - informativo"

# STEP 5: Feature
print("\nSTEP 5: Calcolo feature")

def safe_ratio(num, den, cap=10):
    r = num / den
    return r.where((r >= 0) & (r <= cap), np.nan)

# pct_riserva_base (ante)
pct_ris_base = safe_ratio(ba_agg['somme_a_disposizione'], ba_agg['sum_core'])
pct_ris_base = pct_ris_base[pct_ris_base.index.isin(pq_cigs)].rename('pct_riserva_base')
print(f"pct_riserva_base: {pct_ris_base.notna().sum():,} non-null | mediana={pct_ris_base.median():.3f} | p95={pct_ris_base.quantile(.95):.3f}")

# pct_overrun_core (ex post)
pct_overrun = safe_ratio(co_agg['sum_core'], ba_agg['sum_core'])
pct_overrun = pct_overrun[pct_overrun.index.isin(pq_cigs)].rename('pct_overrun_core')
print(f"pct_overrun_core: {pct_overrun.notna().sum():,} non-null | mediana={pct_overrun.median():.3f} | p95={pct_overrun.quantile(.95):.3f}")

# pct_riserva_consumata (ex post)
pct_ris_cons = safe_ratio(co_agg['somme_a_disposizione'], ba_agg['sum_core'])
pct_ris_cons = pct_ris_cons[pct_ris_cons.index.isin(pq_cigs)].rename('pct_riserva_consumata')
print(f"pct_riserva_consumata: {pct_ris_cons.notna().sum():,} non-null | mediana={pct_ris_cons.median():.3f} | p95={pct_ris_cons.quantile(.95):.3f}")

# STEP 6: SC custom labeled
print("\nSC custom: distribuzione labeled")
sc_custom = []
tmp = pq[['cig','label']].set_index('cig')
tmp['pct_riserva_base'] = pct_ris_base
tmp['pct_overrun_core'] = pct_overrun
tmp['pct_riserva_consumata'] = pct_ris_cons
for lbl, name in [(1,'CONDANNATI'),(0,'SCAGIONATI')]:
    sub = tmp[tmp['label']==lbl]
    for feat in ['pct_riserva_base','pct_overrun_core','pct_riserva_consumata']:
        s = sub[feat].dropna()
        if len(s):
            line = f"SC-QE | {name} {feat}: n={len(s)}, mediana={s.median():.3f}, mean={s.mean():.3f}"
            print(f"  {line}")
            sc_custom.append(line)

# STEP 7: Join + drop importo_variante_totale
print("\nSTEP 7: Join parquet")
overlap = [c for c in ['pct_riserva_base','pct_overrun_core','pct_riserva_consumata'] if c in pq.columns]
if overlap: pq = pq.drop(columns=overlap)
pq = pq.drop(columns=['importo_variante_totale'], errors='ignore')

pq = pq.set_index('cig')
pq['pct_riserva_base']      = pct_ris_base
pq['pct_overrun_core']      = pct_overrun
pq['pct_riserva_consumata'] = pct_ris_cons
pq = pq.reset_index()

print(f"  Shape: {pq.shape}")
print(f"  Label=1: {(pq['label']==1).sum()} | Label=0: {(pq['label']==0).sum()}")
pq.to_parquet(MEGA, index=False)
print("Parquet salvato.")

# SC log
with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write("\n[QUADRO ECONOMICO]\n")
    f.write(sc09_line + '\n')
    f.write(sc11_line + '\n')
    f.write(sc12_line + '\n')
    for line in sc_custom:
        f.write(line + '\n')
print(f"SC log aggiornato.")
