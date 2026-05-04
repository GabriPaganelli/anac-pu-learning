"""
Build stazione appaltante features and join onto mega parquet.

Feature prodotta: natura_giuridica_SA (8 categorie)
Join key: codice_ausa (normalizzato) con fallback su cf_amministrazione_appaltante
Decisioni:
  - regione_sa e flag_sa_fuori_regione non incluse
  - codice_ausa nel parquet e' float-string (es. '165817.0'), nel file SA ha leading zeros
    -> normalizzazione: strip '.0' + lstrip('0')
  - SC-10: codice_ausa coerenza BANDO CIG vs STAZIONE APPALTANTE
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

ANAC_DATA = _DATA_RAW
MEGA      = _MEGA
SC_LOG    = _SC_LOG

sc_results = []

print("STEP 1: Reading stazioni-appaltanti.csv")
sa = pd.read_csv(f'{ANAC_DATA}/stazioni-appaltanti.csv', sep=';', dtype=str, low_memory=False)
sa.columns = [c.lower().strip() for c in sa.columns]
print(f"Shape: {sa.shape}")
print(f"Columns: {list(sa.columns)}")

print("\nSTEP 2: Grouping natura_giuridica_descrizione")

def map_natura_giuridica(val):
    if pd.isna(val):
        return 'ALTRO'
    v = val.strip().upper()

    # SANITARIO: ospedali, ASL, enti di assistenza sanitaria
    if any(x in v for x in [
        'OSPEDALIERI', 'CURA, SOGGIORNO', 'DI PREVIDENZA E DI ASSIST',
        'ISTITUTO PUBBLICO DI ASSIST', 'CASSE MUTUE',
        'AZIENDA PUBBLICA DI SERVIZI ALLE PERSONE'
    ]):
        return 'SANITARIO'

    # EPE: enti pubblici economici (utilities, multiutility)
    if 'ENTI PUBBLICI ECONOMICI' in v:
        return 'EPE'

    # PA_ISTITUZIONALE: governi locali e centrali
    if any(x in v for x in [
        'AMMINISTRAZIONI PUBBLICHE', 'COMUNE', 'ORGANI ED AMMIN',
        'UNIONE DI COMUNI', 'CONSORZIO INTERCOMUNALE',
        'CONSORZIO DI CUI AL DLGS', 'CONSORZI AI SENSI', 'CONSORZIO AI SENSI',
        'CONSORZIO DI DIRITTO PUBBLICO'
    ]):
        return 'PA_ISTITUZIONALE'

    # PA_ENTE: enti pubblici vari, agenzie, istituti, scuole, ordini
    if any(x in v for x in [
        'ENTI PUBBLICI NON ECONOMICI', 'ALTRI ENTI ED ISTITUTI',
        'ALTRO ENTE PUBBLICO', 'AZIENDE REGIONALI', 'AZIENDA SPECIALE',
        'AZIENDE AUTONOME', 'ISTITUTO E SCUOLA PUBBLICA',
        'ORDINE E COLLEGIO', 'ASSOCIAZIONI TRA PROFESSIONISTI',
        'LAVORATORI AUTONOMI', 'LIBERO PROFESSIONISTA',
        'ENTE DIRITTO PUBBLICO', 'ENTE PARCO', 'ENTE AMBIENTALE',
        'ENTE MORALE', 'ENTE SOCIALE', 'ENTE IMPRESA',
        'ENTI ED ISTITUTI DI PREVIDENZA',
    ]):
        return 'PA_ENTE'
    if v in ('ENTE', 'ENTE ECCLESIASTICO'):
        return 'PA_ENTE'

    # NON_PROFIT: fondazioni, associazioni, enti ecclesiastici, opere pie
    if any(x in v for x in [
        'FONDAZION', 'ASSOCIAZION', 'ORGANIZZAZIONI SENZA',
        'OPERE PIE', 'ECCLESIASTIC', 'ISTITUTO RELIGIOSO',
        'ENTE RELIGIOSO', 'ENTE MORALE',
        'SOCIETA\' SPORTIVE DILETTANTISTICHE',
        'AZIENDA SPECIALE DI CUI AL DLGS 267'  # no, this is PA
    ]):
        return 'NON_PROFIT'

    # CONSORZIO: tutte le forme consortili
    if 'CONSORZIO' in v or ('CONSORZI' in v and 'CONSORZI AI SENSI' not in v):
        return 'CONSORZIO'

    # SOCIETA: societa' di capitali/persone, ditte, cooperative, GEIE
    if any(x in v for x in [
        'SOCIETA\'', 'SOCIETA ', 'SOCIET\ufffd',
        'DITTE INDIVIDUALI', 'IMPRESE INDIVIDUALI', 'CONDOMINI',
        'COOPERATIV', 'GRUPPI EUROPEI', 'GEIE',
        'CONTRATTO DI RETE', 'MUTUE ASSICURATRICI',
        'SOCIETA\' DI ARMAMENTO', 'SOCIETA\' DI FATTO',
    ]):
        return 'SOCIETA'

    return 'ALTRO'

sa['natura_giuridica_SA'] = sa['natura_giuridica_descrizione'].apply(map_natura_giuridica)
print("Distribuzione natura_giuridica_SA (SA):")
print(sa['natura_giuridica_SA'].value_counts(dropna=False).to_string())

print("\nSTEP 3: Normalizing codice_ausa")
# SA file: '0000165817' -> '165817'
sa['ausa_norm'] = sa['codice_ausa'].str.strip().str.lstrip('0')
# Duplicati su ausa_norm (prendi prima riga)
dups = sa['ausa_norm'].duplicated().sum()
print(f"  Duplicati ausa_norm nel file SA: {dups}")
sa_by_ausa = sa.drop_duplicates(subset='ausa_norm')[['ausa_norm','natura_giuridica_SA','codice_fiscale']].copy()
sa_by_cf   = sa.drop_duplicates(subset='codice_fiscale')[['codice_fiscale','natura_giuridica_SA']].copy()

print("\nSTEP 4: Loading parquet")
parquet = pd.read_parquet(MEGA)
print(f"  Shape: {parquet.shape}")

# Parquet ausa: '165817.0' -> '165817'
parquet['_ausa_norm'] = (
    parquet['codice_ausa']
    .astype(str)
    .str.strip()
    .str.replace(r'\.0$', '', regex=True)
    .str.lstrip('0')
    .replace('nan', np.nan)
    .replace('', np.nan)
)

print("\nSC-10: codice_ausa BANDO CIG vs STAZIONE APPALTANTE")
has_ausa = parquet['_ausa_norm'].notna()
match_ausa = parquet['_ausa_norm'].isin(sa_by_ausa['ausa_norm'].dropna())
n_has = has_ausa.sum()
n_match = (has_ausa & match_ausa).sum()
n_nomatch = (has_ausa & ~match_ausa).sum()
pct_match = n_match / n_has * 100 if n_has > 0 else 0
print(f"  CIG con codice_ausa non-null: {n_has:,}")
print(f"  Match in SA file: {n_match:,} ({pct_match:.2f}%)")
print(f"  No match: {n_nomatch:,} ({100-pct_match:.2f}%)")
status_10 = "OK" if pct_match > 99 else "WARNING"
msg_10 = (f"ausa_non_null={n_has:,}, match={n_match:,} ({pct_match:.2f}%), "
          f"no_match={n_nomatch:,} | fix: normalizzazione strip zeros + '.0'")
sc_results.append(("SC-10", status_10, msg_10))

print("\nSTEP 6: Joining")

# Drop pre-existing column
if 'natura_giuridica_SA' in parquet.columns:
    parquet = parquet.drop(columns=['natura_giuridica_SA'])

# Join via ausa
parquet = parquet.merge(
    sa_by_ausa[['ausa_norm','natura_giuridica_SA']].rename(columns={'natura_giuridica_SA': '_ng_ausa'}),
    left_on='_ausa_norm', right_on='ausa_norm', how='left'
).drop(columns=['ausa_norm'])

# Join via CF (fallback)
parquet = parquet.merge(
    sa_by_cf.rename(columns={'natura_giuridica_SA': '_ng_cf'}),
    left_on='cf_amministrazione_appaltante', right_on='codice_fiscale', how='left'
).drop(columns=['codice_fiscale'])

# Combine: ausa primo, poi CF
parquet['natura_giuridica_SA'] = parquet['_ng_ausa'].where(
    parquet['_ng_ausa'].notna(), parquet['_ng_cf']
)
parquet = parquet.drop(columns=['_ng_ausa', '_ng_cf', '_ausa_norm'])

n_filled = parquet['natura_giuridica_SA'].notna().sum()
print(f"  CIG con natura_giuridica_SA: {n_filled:,} ({n_filled/len(parquet)*100:.1f}%)")
print("  Distribuzione nel parquet:")
print(parquet['natura_giuridica_SA'].value_counts(dropna=False).to_string())
print(f"  Label=1 rimasti: {(parquet['label']==1).sum()}")
print(f"  Label=0 rimasti: {(parquet['label']==0).sum()}")

print("\nSaving parquet...")
parquet.to_parquet(MEGA, index=False)
print("Parquet salvato.")
print(f"Shape finale: {parquet.shape}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n\n{'='*60}\n")
    f.write(f"Run: {timestamp} [build_stazione_appaltante]\n")
    f.write(f"Parquet shape: {parquet.shape}\n")
    f.write("\nSanity Checks:\n")
    for sc_id, status, msg in sc_results:
        f.write(f"  {sc_id:12s} [{status}]  {msg}\n")
print(f"SC results appended to: {SC_LOG}")
