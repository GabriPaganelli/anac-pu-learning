"""
05_build_aggiudicazioni.py — Aggiunge feature aggiudicazione al parquet bando_cig_all e applica il filtro Opzione A
(rimuove CIG con esito DESERTA / NON_AGGIUDICATA / IN_CORSO senza override da cod_esito).
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.join(BASE, "data", "raw")
LABELS_DIR = os.path.join(BASE, "labels")
PARQUET    = os.path.join(BASE, "output", "parquet", "bando_cig_all.parquet")
SC_LOG     = os.path.join(BASE, "output", "sc_results.txt")

DATE_MIN = pd.Timestamp('1990-01-01')
DATE_MAX = pd.Timestamp('2030-12-31')

sc_results = []   # list of (id, status, message)

def parse_date_safe(series):
    s = pd.to_datetime(series, errors='coerce')
    s = s.where(s >= DATE_MIN, pd.NaT)
    s = s.where(s <= DATE_MAX, pd.NaT)
    return s

def parse_float(series):
    """Parse string with comma-as-decimal to float."""
    return (
        series.astype(str)
              .str.strip()
              .str.replace(',', '.', regex=False)
              .str.replace(r'\s+', '', regex=True)
        .pipe(pd.to_numeric, errors='coerce')
    )

print("STEP 1: Reading aggiudicazioni.csv")
agg = pd.read_csv(
    f'{DATA_RAW}/aggiudicazioni.csv',
    sep=';', dtype=str, low_memory=False
)
agg.columns = [c.lower().strip() for c in agg.columns]
print(f"Aggiudicazioni shape: {agg.shape}")
print(f"Columns: {list(agg.columns)}")

print("\nSTEP 2: Parsing numeric columns")
num_cols = [
    'numero_offerte_ammesse', 'numero_offerte_escluse', 'num_imprese_offerenti',
    'ribasso_aggiudicazione', 'massimo_ribasso', 'minimo_ribasso',
    'importo_aggiudicazione'
]
for col in num_cols:
    if col in agg.columns:
        agg[col] = parse_float(agg[col])
        print(f"  {col}: {agg[col].notna().sum():,} non-null")
    else:
        print(f"  WARN: column '{col}' not found")

print("\nSTEP 3: Row selection for multi-row CIGs")

# 3a. Load condannati and fine-contratto
condannati_df = pd.read_csv(f'{LABELS_DIR}/cig_condannati.csv', dtype=str)
condannati_df.columns = [c.lower().strip() for c in condannati_df.columns]
condannati_set = set(condannati_df['cig'].str.strip())

fine_df = pd.read_csv(
    f'{DATA_RAW}/fine-contratto.csv',
    sep=';', dtype=str, low_memory=False
)
fine_df.columns = [c.lower().strip() for c in fine_df.columns]
# For condannati with riaggiudicazione: map cig -> id_aggiudicazione
fine_condannati = fine_df[fine_df['cig'].isin(condannati_set) & fine_df['id_aggiudicazione'].notna()][['cig','id_aggiudicazione']].drop_duplicates('cig')
cig_to_id_agg = dict(zip(fine_condannati['cig'], fine_condannati['id_aggiudicazione'].str.strip()))
print(f"  Condannati con id_aggiudicazione in fine-contratto: {len(cig_to_id_agg)}")

# 3b. Temporary date parse for sorting
agg['_date_sort'] = parse_date_safe(agg['data_aggiudicazione_definitiva'])

cig_counts = agg['cig'].value_counts()
multi_cigs = set(cig_counts[cig_counts > 1].index)
single_cigs = set(cig_counts[cig_counts == 1].index)
print(f"  CIG con una sola riga: {len(single_cigs):,}")
print(f"  CIG con più righe: {len(multi_cigs):,}")

def select_row(group):
    cig = group['cig'].iloc[0]
    if len(group) == 1:
        return group.iloc[[0]]
    # condannato con id_aggiudicazione noto?
    if cig in condannati_set and cig in cig_to_id_agg:
        target_id = cig_to_id_agg[cig]
        match = group[group['id_aggiudicazione'].astype(str).str.strip() == target_id]
        if len(match) >= 1:
            return match.iloc[[0]]
    # altrimenti: prima riga per data senza riaggiudicazione
    no_riaggiud = group[
        group['cod_modo_riaggiudicazione'].isna() |
        (group['cod_modo_riaggiudicazione'].astype(str).str.strip() == '') |
        (group['cod_modo_riaggiudicazione'].astype(str).str.strip() == 'nan')
    ]
    if len(no_riaggiud) > 0:
        sorted_rows = no_riaggiud.sort_values('_date_sort', na_position='last')
        return sorted_rows.iloc[[0]]
    else:
        # tutte con riaggiudicazione: prima per data
        sorted_rows = group.sort_values('_date_sort', na_position='last')
        return sorted_rows.iloc[[0]]

multi_df   = agg[agg['cig'].isin(multi_cigs)]
single_df  = agg[agg['cig'].isin(single_cigs)]

selected_multi = multi_df.groupby('cig', group_keys=False).apply(select_row)
condannati_with_riaggiud_found = sum(
    1 for c in multi_cigs if c in condannati_set and c in cig_to_id_agg
)
print(f"  Condannati multi-riga con riaggiudicazione identificati: {condannati_with_riaggiud_found}")

agg = pd.concat([single_df, selected_multi], ignore_index=True)
agg.drop(columns=['_date_sort'], inplace=True)
print(f"  Shape finale dopo selezione: {agg.shape}")

print("\nSTEP 4: Parsing date columns")
agg['data_aggiudicazione_definitiva'] = parse_date_safe(agg['data_aggiudicazione_definitiva'])
agg['data_comunicazione_esito']       = parse_date_safe(agg['data_comunicazione_esito'])
print(f"  data_aggiudicazione_definitiva non-null: {agg['data_aggiudicazione_definitiva'].notna().sum():,}")
print(f"  data_comunicazione_esito non-null: {agg['data_comunicazione_esito'].notna().sum():,}")

print("\nSTEP 5: Boolean NaN -> 0")
for col in ['asta_elettronica', 'flag_scomputo', 'flag_proc_accelerata']:
    if col in agg.columns:
        agg[col] = pd.to_numeric(agg[col], errors='coerce').fillna(0).astype(int)
        print(f"  {col}: values = {agg[col].value_counts().to_dict()}")
    else:
        print(f"  WARN: '{col}' not found")

print("\nSTEP 6: Building features")

# pct_offerte_escluse
total_max = np.maximum(
    agg['num_imprese_offerenti'].fillna(0),
    agg['numero_offerte_ammesse'].fillna(0) + agg['numero_offerte_escluse'].fillna(0)
)
agg['pct_offerte_escluse'] = np.where(
    total_max > 0,
    agg['numero_offerte_escluse'] / total_max,
    np.nan
)

# ribasso_spread
agg['ribasso_spread'] = np.where(
    agg['massimo_ribasso'].notna() & agg['ribasso_aggiudicazione'].notna(),
    agg['massimo_ribasso'] - agg['ribasso_aggiudicazione'],
    np.nan
)

# flag_vince_minimo
agg['flag_vince_minimo'] = (
    agg['ribasso_aggiudicazione'].notna() &
    agg['minimo_ribasso'].notna() &
    (np.abs(agg['ribasso_aggiudicazione'] - agg['minimo_ribasso']) < 0.001)
).astype('boolean')

# flag_progettazione_esterna
if 'cig_prog_esterna' in agg.columns:
    agg['flag_progettazione_esterna'] = (
        agg['cig_prog_esterna'].notna() &
        (agg['cig_prog_esterna'].str.strip() != '')
    )
else:
    print("  WARN: cig_prog_esterna non trovata, flag_progettazione_esterna = False")
    agg['flag_progettazione_esterna'] = False
agg['flag_progettazione_esterna'] = agg['flag_progettazione_esterna'].astype('boolean')

print(f"  pct_offerte_escluse non-null: {agg['pct_offerte_escluse'].notna().sum():,}")
print(f"  ribasso_spread non-null: {agg['ribasso_spread'].notna().sum():,}")
print(f"  flag_vince_minimo True: {agg['flag_vince_minimo'].sum():,}")
print(f"  flag_progettazione_esterna True: {agg['flag_progettazione_esterna'].sum():,}")

print("\nSTEP 7: lag_aggiudicazione_giorni + delta_coerenza_pct")

# data_pubblicazione letta dal parquet (aggiunta da 04_build_bando_cig.py)
print("  Carico data_pubblicazione dal parquet...")
pub_df = pd.read_parquet(PARQUET, columns=['cig', 'data_pubblicazione'])
pub_df['data_pubblicazione'] = parse_date_safe(pub_df['data_pubblicazione'])
print(f"  data_pubblicazione loaded: {pub_df['data_pubblicazione'].notna().sum():,} non-null")

# Join data_pubblicazione to agg
agg = agg.merge(pub_df[['cig','data_pubblicazione']], on='cig', how='left')
lag = (agg['data_aggiudicazione_definitiva'] - agg['data_pubblicazione']).dt.days
n_lag_neg = (lag < 0).sum()
agg['lag_aggiudicazione_giorni'] = lag.where(lag >= 0, np.nan)  # SC-02: lag negativo = contratto ante-ANAC, non interpretabile
print(f"  lag_aggiudicazione_giorni non-null: {agg['lag_aggiudicazione_giorni'].notna().sum():,}")
print(f"  lag_aggiudicazione_giorni < 0 (-> NaN): {n_lag_neg:,}")

# delta_coerenza_pct — needs importo_lotto from parquet
parquet_lotto = pd.read_parquet(PARQUET, columns=['cig', 'importo_lotto'])
agg = agg.merge(parquet_lotto[['cig','importo_lotto']], on='cig', how='left')
agg['delta_coerenza_pct'] = np.where(
    agg['importo_lotto'].notna() & (agg['importo_lotto'] > 0) &
    agg['importo_aggiudicazione'].notna() & agg['ribasso_aggiudicazione'].notna(),
    np.abs(
        agg['importo_aggiudicazione'] -
        agg['importo_lotto'] * (1 - agg['ribasso_aggiudicazione'] / 100)
    ) / agg['importo_lotto'],
    np.nan
)
agg.drop(columns=['importo_lotto'], inplace=True)
print(f"  delta_coerenza_pct non-null: {agg['delta_coerenza_pct'].notna().sum():,}")

print("\nSTEP 8: Filtro esito")
parquet_esito = pd.read_parquet(PARQUET, columns=['cig', 'esito', 'label'])

BAD_ESITO   = {'DESERTA', 'NON_AGGIUDICATA', 'IN_CORSO'}
GOOD_COD_ESITO = {'1', '5'}

cigs_bad_esito = set(parquet_esito[parquet_esito['esito'].isin(BAD_ESITO)]['cig'])

agg_cod = agg[['cig', 'cod_esito']].copy()
agg_cod['cod_esito_clean'] = (
    agg_cod['cod_esito'].astype(str).str.strip().str.split('.').str[0]
)
cigs_override = set(agg_cod[agg_cod['cod_esito_clean'].isin(GOOD_COD_ESITO)]['cig'])
cigs_to_drop  = cigs_bad_esito - cigs_override

labels_affected = parquet_esito[parquet_esito['cig'].isin(cigs_to_drop)]
print(f"  CIGs con bad esito: {len(cigs_bad_esito):,}")
print(f"  Override da cod_esito accettabile: {len(cigs_override & cigs_bad_esito)}")
print(f"  CIGs da droppare: {len(cigs_to_drop):,}")
print(f"  Label persi: pos={(labels_affected['label']==1).sum()}, neg={(labels_affected['label']==0).sum()}")

print("\nSTEP 9: Selecting columns for join")
keep_cols = [
    'cig',
    'numero_offerte_ammesse',
    'pct_offerte_escluse',
    'ribasso_aggiudicazione',
    'ribasso_spread',
    'flag_vince_minimo',
    'flag_progettazione_esterna',
    'lag_aggiudicazione_giorni',
    'delta_coerenza_pct',
    'asta_elettronica',
    'flag_scomputo',
    'flag_proc_accelerata',
    'importo_aggiudicazione',
    'data_aggiudicazione_definitiva',
    'data_comunicazione_esito',
    'cod_esito',
    'num_imprese_offerenti',
    'numero_offerte_escluse',       # needed for SC-OFFERTE
]
# Only keep columns that actually exist
keep_cols = [c for c in keep_cols if c in agg.columns]
print(f"  Columns kept: {keep_cols}")
agg_to_join = agg[keep_cols].copy()

print("\nSTEP 10: Loading parquet and joining")
parquet_full = pd.read_parquet(PARQUET)
print(f"  Parquet prima del filtro esito: {parquet_full.shape}")

parquet_full = parquet_full[~parquet_full['cig'].isin(cigs_to_drop)]
print(f"  Parquet dopo filtro esito: {parquet_full.shape}")

# Drop colonne già presenti da run precedente per evitare suffissi _x/_y
overlap_cols = [c for c in keep_cols if c != 'cig' and c in parquet_full.columns]
if overlap_cols:
    parquet_full = parquet_full.drop(columns=overlap_cols)
    print(f"  Droppate {len(overlap_cols)} colonne pre-esistenti dal parquet: {overlap_cols}")

result = parquet_full.merge(agg_to_join, on='cig', how='left')

print(f"  Parquet dopo join: {result.shape}")
print(f"  CIGs con dati aggiudicazioni (importo_aggiudicazione notna): {result['importo_aggiudicazione'].notna().sum():,}")
print(f"  Label=1 rimasti: {(result['label']==1).sum()}")
print(f"  Label=0 rimasti: {(result['label']==0).sum()}")

# Bring data_pubblicazione into result (for SC-02)
if 'data_pubblicazione' not in result.columns:
    result = result.merge(pub_df[['cig','data_pubblicazione']], on='cig', how='left')

print("\nSANITY CHECKS")

# SC-01
n_dup = result['cig'].duplicated().sum()
status_01 = "OK" if n_dup == 0 else "ERROR"
msg_01 = f"SC-01 | CIG duplicati nel parquet: {n_dup}"
print(msg_01)
sc_results.append(("SC-01", status_01, msg_01))

# SC-02
if 'data_pubblicazione' in result.columns:
    mask_both = result['data_aggiudicazione_definitiva'].notna() & result['data_pubblicazione'].notna()
    violations_02 = result[
        mask_both &
        (result['data_aggiudicazione_definitiva'] < result['data_pubblicazione'])
    ]
    n_valid = result['data_aggiudicazione_definitiva'].notna().sum()
    pct_02 = len(violations_02) / n_valid * 100 if n_valid > 0 else 0
    msg_02 = f"SC-02 | Aggiudicazioni prima del bando: {len(violations_02):,} ({pct_02:.2f}%)"
    # SC-02: lag negativi = contratti caricati retroattivamente da ANAC, non errori.
    # Fix applicato: lag_aggiudicazione_giorni = NaN per questi casi.
    # Status HANDLED (non ERROR) perche' il problema e' noto e gestito.
    status_02 = "OK" if len(violations_02) == 0 else "HANDLED"
    print(msg_02)
    if len(violations_02) > 0:
        print(violations_02[['cig','data_aggiudicazione_definitiva','data_pubblicazione','label']].head(10).to_string())
    sc_results.append(("SC-02", status_02, msg_02))
else:
    print("SC-02 | SKIP: data_pubblicazione non disponibile nel result")
    sc_results.append(("SC-02", "SKIP", "data_pubblicazione non disponibile"))

# SC-03
has_delta = result['delta_coerenza_pct'].notna()
pct_has_delta = has_delta.mean() * 100
d20 = (result['delta_coerenza_pct'] > 0.20).sum()
d50 = (result['delta_coerenza_pct'] > 0.50).sum()
print(f"\nSC-03 | Righe con delta calcolabile: {has_delta.sum():,} ({pct_has_delta:.1f}%)")
print(f"SC-03 | delta > 20%: {d20:,} ({(result['delta_coerenza_pct'] > 0.20).mean()*100:.1f}%)")
print(f"SC-03 | delta > 50%: {d50:,}")
print(f"SC-03 | Distribuzione percentile delta:")
print(result['delta_coerenza_pct'].describe(percentiles=[.5,.75,.9,.95,.99]))
for lbl, name in [(1,'condannati'), (0,'scagionati')]:
    sub = result[result['label']==lbl]['delta_coerenza_pct'].dropna()
    if len(sub) > 0:
        print(f"  {name}: mediana={sub.median():.3f}, >20%={(sub>0.2).mean()*100:.1f}%")
    else:
        print(f"  {name}: no data")
status_03 = "OK" if pct_has_delta > 10 else "WARNING"
msg_03 = f"delta calcolabile={has_delta.sum():,} ({pct_has_delta:.1f}%), delta>20%={d20:,}, delta>50%={d50:,}"
sc_results.append(("SC-03", status_03, msg_03))

# SC-05
mask_05 = result['importo_sicurezza'].notna() & result['importo_lotto'].notna()
violations_05 = result[mask_05 & (result['importo_sicurezza'] > result['importo_lotto'])]
pct_05 = len(violations_05) / mask_05.sum() * 100 if mask_05.sum() > 0 else 0
n_pct100 = (result['importo_sicurezza_pct'] > 1).sum() if 'importo_sicurezza_pct' in result.columns else 0
print(f"\nSC-05 | importo_sicurezza > importo_lotto: {len(violations_05):,} ({pct_05:.2f}%)")
print(f"SC-05 | importo_sicurezza_pct > 100%: {n_pct100:,}")
status_05 = "OK" if pct_05 < 1 else "WARNING"
msg_05 = f"importo_sicurezza>importo_lotto={len(violations_05):,} ({pct_05:.2f}%), pct>100%={n_pct100:,}"
sc_results.append(("SC-05", status_05, msg_05))

# SC-05 FIX: NaN solo se la colonna ha già NA (evita alterazione del dtype)
if len(violations_05) > 0 and result['importo_sicurezza'].isna().any():
    viol_idx = violations_05.index
    result.loc[viol_idx, 'importo_sicurezza'] = np.nan
    if 'importo_sicurezza_pct' in result.columns and result['importo_sicurezza_pct'].isna().any():
        result.loc[viol_idx, 'importo_sicurezza_pct'] = np.nan
    print(f"  SC-05 FIX applicato: {len(viol_idx):,} righe -> importo_sicurezza = NaN")
elif len(violations_05) > 0:
    print(f"  SC-05 FIX SKIP: importo_sicurezza senza NA esistenti, skip per non alterare dtype")

# SC-08
condannati_set_labels = set(
    pd.read_csv(f'{LABELS_DIR}/cig_condannati.csv', dtype=str)
      .rename(columns=str.lower)['cig'].str.strip()
)
scagionati_set_labels = set(
    pd.read_csv(f'{LABELS_DIR}/cig_scagionati.csv', dtype=str)
      .rename(columns=str.lower)['cig'].str.strip()
)
overlap_08 = condannati_set_labels & scagionati_set_labels
print(f"\nSC-08 | CIG in entrambe le liste: {len(overlap_08)} (atteso: 0)")
if overlap_08:
    print("ERRORE CRITICO:", overlap_08)
status_08 = "OK" if len(overlap_08) == 0 else "ERROR"
sc_results.append(("SC-08", status_08, f"overlap={len(overlap_08)}"))

# SC-13
mask_agg_esito = result['esito'] == 'AGGIUDICATA'
missing_agg_data = result[mask_agg_esito & result['importo_aggiudicazione'].isna()]
pct_13a = len(missing_agg_data) / mask_agg_esito.sum() * 100 if mask_agg_esito.sum() > 0 else 0
print(f"\nSC-13a | AGGIUDICATA senza dati aggiudicazioni: {len(missing_agg_data):,} ({pct_13a:.1f}%)")

pos_bad_esito = result[(result['label']==1) & (result['esito'].isin(BAD_ESITO))]
print(f"SC-13c | Condannati con esito anomalo (residui dopo filtro): {len(pos_bad_esito)}")
status_13 = "OK" if len(pos_bad_esito) == 0 else "WARNING"
msg_13 = f"AGGIUDICATA senza dati={len(missing_agg_data):,} ({pct_13a:.1f}%), condannati esito anomalo={len(pos_bad_esito)}"
sc_results.append(("SC-13", status_13, msg_13))

# SC-OFFERTE
result['_total_calc'] = (
    result['numero_offerte_ammesse'].fillna(0) +
    result['numero_offerte_escluse'].fillna(0)
    if 'numero_offerte_escluse' in result.columns
    else result['numero_offerte_ammesse'].fillna(0)
)
has_both_off = result['num_imprese_offerenti'].notna() & (result['_total_calc'] > 0)
delta_offerte = np.abs(
    result.loc[has_both_off, 'num_imprese_offerenti'] -
    result.loc[has_both_off, '_total_calc']
)
pct_perfect = (delta_offerte == 0).mean() * 100 if len(delta_offerte) > 0 else 0
pct_gt2     = (delta_offerte > 2).mean() * 100  if len(delta_offerte) > 0 else 0
n_gt10 = (delta_offerte > 10).sum()
print(f"\nSC-OFFERTE | Righe confrontabili: {has_both_off.sum():,}")
print(f"SC-OFFERTE | Delta == 0 (perfetta coerenza): {(delta_offerte==0).sum():,} ({pct_perfect:.1f}%)")
print(f"SC-OFFERTE | Delta > 2: {(delta_offerte>2).sum():,} ({pct_gt2:.1f}%)")
print(f"SC-OFFERTE | Delta > 10: {n_gt10:,}")
status_off = "OK" if pct_perfect > 80 else "WARNING"
msg_off = f"confrontabili={has_both_off.sum():,}, perfect={pct_perfect:.1f}%, gt2={pct_gt2:.1f}%, gt10={n_gt10}"
sc_results.append(("SC-OFFERTE", status_off, msg_off))

# SC-16
print(f"\nSC-16 | Distribuzione cod_esito:")
print(result['cod_esito'].value_counts(dropna=False).head(10))
print(f"\nSC-16 | cod_esito per esito principale:")
print(result.groupby('esito')['cod_esito'].value_counts(dropna=False).head(20))
sc_results.append(("SC-16", "INFO", "cod_esito distribution printed"))

print("\nSaving parquet...")

# Drop data_pubblicazione from result (it came from a side-join, not parquet original)
if 'data_pubblicazione' in result.columns:
    result.drop(columns=['data_pubblicazione'], inplace=True)

# Drop helper column
if '_total_calc' in result.columns:
    result.drop(columns=['_total_calc'], inplace=True)

result.to_parquet(PARQUET, index=False)
print("Parquet salvato.")

print("\nRIEPILOGO FINALE")
print(f"Shape parquet finale: {result.shape}")
n_pos = (result['label']==1).sum()
n_neg = (result['label']==0).sum()
n_unlab = result['label'].isna().sum()
print(f"Label=1 (condannati): {n_pos:,}")
print(f"Label=0 (scagionati): {n_neg:,}")
print(f"Label=NaN (unlabeled): {n_unlab:,}")

new_cols = [c for c in result.columns if c in keep_cols and c != 'cig']
print(f"\nColonne aggiunte da AGGIUDICAZIONI ({len(new_cols)}):")
for c in new_cols:
    nn = result[c].notna().sum()
    print(f"  {c}: {nn:,} non-null ({nn/len(result)*100:.1f}%)")

print("\nSanity Check Results:")
for sc_id, status, msg in sc_results:
    print(f"  {sc_id:12s} [{status}]  {msg}")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os.makedirs(os.path.dirname(SC_LOG), exist_ok=True)
with open(SC_LOG, 'a', encoding='utf-8') as f:
    f.write(f"\n\n{'='*60}\n")
    f.write(f"Run: {timestamp}\n")
    f.write(f"Parquet shape: {result.shape}\n")
    f.write(f"Label=1: {n_pos}, Label=0: {n_neg}, Unlabeled: {n_unlab}\n")
    f.write("\nSanity Checks:\n")
    for sc_id, status, msg in sc_results:
        f.write(f"  {sc_id:12s} [{status}]  {msg}\n")
print(f"\nSC results appended to: {SC_LOG}")
