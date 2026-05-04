#!/usr/bin/env python3
"""
16_build_model_datasets.py — Costruisce 6 parquet model (3 nativi + 3 preprocessed) da bando_cig_all.parquet.

Struttura output:
  output/parquet/model/nativi/       M1.parquet  M2.parquet  M3.parquet
  output/parquet/model/preprocessed/ M1.parquet  M2.parquet  M3.parquet

Questo script e' anche responsabile di droppare data_pubblicazione
da bando_cig_all.parquet (colonna ausiliaria aggiunta da 04_build_bando_cig.py
per uso di 05_build_aggiudicazioni.py, non e' una feature di modello).

Nota: anno_pubblicazione e regione sono colonne di metadata/analisi,
NON incluse nei feature set. Per includerle aggiungile esplicitamente a M1_FEAT.
"""

import os
import numpy as np
import pandas as pd

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC      = os.path.join(BASE, "output", "parquet", "bando_cig_all.parquet")
OUT_NAT  = os.path.join(BASE, "output", "parquet", "model", "nativi")
OUT_PREP = os.path.join(BASE, "output", "parquet", "model", "preprocessed")

os.makedirs(OUT_NAT,  exist_ok=True)
os.makedirs(OUT_PREP, exist_ok=True)

ID_COLS = ['cig', 'label']

# anno_pubblicazione e regione sono metadata/analisi, non feature di modello.
# Per includerle nel training aggiungile esplicitamente qui sotto.
M1_FEAT = [
    'importo_complessivo_gara', 'importo_lotto', 'n_lotti_componenti',
    'oggetto_principale_contratto', 'modalita_realizzazione_macro', 'sezione_regionale',
    'cod_strumento_svolgimento', 'flag_urgenza', 'cod_motivo_urgenza',
    'flag_delega', 'finestra_offerta_giorni', 'lag_perfezionamento_giorni',
    'flag_accordo_quadro', 'flag_ripetizioni', 'settore_speciale', 'flag_appalto_riservato',
    'tipo_scelta_4cls', 'cpv_macro_categoria', 'importo_sicurezza_pct', 'pct_riserva_base',
    'natura_giuridica_SA',
    'tasso_disoccupazione', 'reddito_irpef_procapite', 'tasso_omicidi_100k',
]

M2_AGGIUDICAZIONI = [
    'numero_offerte_ammesse', 'numero_offerte_escluse', 'num_imprese_offerenti',
    'pct_offerte_escluse', 'ribasso_aggiudicazione', 'ribasso_spread',
    'flag_vince_minimo', 'flag_progettazione_esterna', 'lag_aggiudicazione_giorni',
    'lag_comunicazione_esito_giorni', 'asta_elettronica', 'flag_scomputo',
    'flag_proc_accelerata', 'importo_aggiudicazione', 'tipo_soggetto_agg', 'flag_subappalto',
]

# Feature di esecuzione contratto (avvio, varianti, sospensioni, sal, collaudo)
# Presenti in entrambe le versioni M3 (nativi e preprocessed)
M3_ESECUZIONE = [
    'lag_stipula_aggiudicazione_giorni', 'durata_pianificata_giorni',
    'consegna_frazionata', 'consegna_sotto_riserva',
    'n_varianti', 'flag_variante_sostanziale', 'flag_variante_oltre_termine',
    'n_sospensioni', 'durata_totale_sospensioni_gg',
    'flag_sospensione', 'flag_sosp_giudiziaria',
    'pct_durata_sospesa', 'n_sal', 'flag_in_ritardo', 'flag_proroga', 'esito_collaudo',
]

# Solo nei nativi (XGBoost/LightGBM): variabili continue con 93-98% NA.
# Hanno analogo binario già in M3_ESECUZIONE per i modelli non-nativi.
M3_NATIVI_CONTINUI = ['pct_overrun_variante', 'pct_vita_prima_variante',
                       'pct_overrun_core', 'pct_riserva_consumata']

# Solo nel preprocessed: flag derivato da pct_overrun_core (1 se CONSUNTIVO presente in QE)
M3_PREPROCESSED_FLAGS = ['flag_consuntivo_presente']

M2_FEAT      = M1_FEAT + M2_AGGIUDICAZIONI
M3_NAT_FEAT  = M2_FEAT + M3_ESECUZIONE + M3_NATIVI_CONTINUI
M3_PREP_FEAT = M2_FEAT + M3_ESECUZIONE + M3_PREPROCESSED_FLAGS

print("Caricamento sorgente...")
pq = pd.read_parquet(SRC)
print(f"  {pq.shape[0]:,} righe x {pq.shape[1]} colonne")

# Drop data_pubblicazione (colonna ausiliaria, non e' una feature)
if 'data_pubblicazione' in pq.columns:
    pq.drop(columns=['data_pubblicazione'], inplace=True)
    pq.to_parquet(SRC, index=False)
    print(f"  data_pubblicazione droppata da bando_cig_all.parquet")
    print(f"  Colonne dopo drop: {pq.shape[1]}")

# Verifica colonne
all_needed = set(ID_COLS + M3_NAT_FEAT + ['pct_overrun_core', 'esito'])  # esito: usato per row masks, non nei parquet output
missing = all_needed - set(pq.columns)
if missing:
    print(f"⚠️  COLONNE MANCANTI: {sorted(missing)}")
    raise SystemExit("Build interrotto: colonne mancanti nel sorgente.")
print("  OK Tutte le colonne richieste presenti")

pq['flag_consuntivo_presente'] = pq['pct_overrun_core'].notna().astype('int8')
n_cons = pq['flag_consuntivo_presente'].sum()
print(f"  flag_consuntivo_presente: {n_cons:,} CIG con CONSUNTIVO ({n_cons/len(pq)*100:.1f}%)")

mask_m2 = pq['esito'] == 'AGGIUDICATA'
mask_m3 = mask_m2 & (
    pq['lag_stipula_aggiudicazione_giorni'].notna() |
    (pq['n_sal'] > 0) |
    (pq['n_varianti'] > 0) |
    (pq['n_sospensioni'] > 0)
)

def labeled_stats(df):
    cond = (df['label'] == 1).sum()
    scag = (df['label'] == 0).sum()
    return int(cond + scag), int(cond), int(scag)

print(f"\n  Row selection:")
for nm, msk in [('M1', None), ('M2', mask_m2), ('M3', mask_m3)]:
    n = msk.sum() if msk is not None else len(pq)
    print(f"    {nm}: {n:,} righe")

print("\nComputo imputazioni su M1 (mediana non-zero + moda)...")
def med_nonzero(s):
    return s[s.notna() & (s > 0)].median()

imp = {
    'importo_complessivo_gara':       med_nonzero(pq['importo_complessivo_gara']),
    'finestra_offerta_giorni':        round(med_nonzero(pq['finestra_offerta_giorni'])),
    'lag_perfezionamento_giorni':     round(med_nonzero(pq['lag_perfezionamento_giorni'])),
    # Moda su M1 per colonne categoriali con NA piccolo (< 1% ma non-zero)
    'tipo_scelta_4cls':               pq['tipo_scelta_4cls'].mode()[0],
    'oggetto_principale_contratto':   pq['oggetto_principale_contratto'].mode()[0],
    'natura_giuridica_SA':            pq['natura_giuridica_SA'].mode()[0],
}
for k, v in imp.items():
    print(f"  {k}: {v!r}" if isinstance(v, str) else f"  {k}: {v:,.0f}")

print("\nComputo breaks quantili...")

# Su M1 (tutti i non-null)
def q4_breaks(series):
    v = series.dropna()
    return v.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

qb = {}
qb['tasso_disoccupazione']   = q4_breaks(pq['tasso_disoccupazione'])

# importo_sicurezza_pct: distribuzione degenere (molti zeri) -> bins fissi
# [0] / (0, 0.01] / (0.01, 0.03] / (0.03+]
# (nessun calcolo quantile necessario)

# Su M2 (log scale per importo_aggiudicazione) - solo valori positivi
pq_m2_agg = pq.loc[mask_m2, 'importo_aggiudicazione']
pq_m2_pos = pq_m2_agg[pq_m2_agg.notna() & (pq_m2_agg > 0)]
log_q = np.log1p(pq_m2_pos).quantile([0.25, 0.5, 0.75]).tolist()
# bins con -inf / +inf per catturare tutti i valori
qb['importo_aggiudicazione_log']  = [-np.inf] + log_q + [np.inf]
qb['importo_aggiudicazione_orig'] = [0] + np.expm1(log_q).tolist() + [np.inf]

print(f"  tasso_disoccupazione   Q-breaks: {[f'{x:.1f}'  for x in qb['tasso_disoccupazione']]}")
print(f"  importo_aggiudicazione Q-breaks (orig): {[f'{x:,.0f}' for x in qb['importo_aggiudicazione_orig'][1:-1]]}")

print("\nBUILD NATIVI")

dof_nativi = {}
for name, mask, features in [
    ('M1', None,    M1_FEAT),
    ('M2', mask_m2, M2_FEAT),
    ('M3', mask_m3, M3_NAT_FEAT),
]:
    avail_feats = [c for c in features if c in pq.columns]
    cols = ID_COLS + avail_feats
    sub = pq.loc[mask, cols].copy() if mask is not None else pq[cols].copy()
    path = f'{OUT_NAT}/{name}.parquet'
    sub.to_parquet(path, index=False)
    n_lab, nc, ns = labeled_stats(sub)
    dof_nativi[name] = len(avail_feats)
    print(f"  {name}: {len(sub):,} righe x {len(avail_feats)} feature "
          f"| labeled={n_lab} (C={nc}/S={ns}) -> {path}")

def _cat(s): return s.astype('category')

def apply_preprocessing(df, qb, imp):
    df = df.copy()

    if 'importo_complessivo_gara' in df.columns:
        df['importo_complessivo_gara'] = df['importo_complessivo_gara'].fillna(imp['importo_complessivo_gara'])
    if 'flag_urgenza' in df.columns:
        df['flag_urgenza'] = df['flag_urgenza'].fillna(0).astype('int8')
    if 'finestra_offerta_giorni' in df.columns:
        df['finestra_offerta_giorni'] = df['finestra_offerta_giorni'].fillna(imp['finestra_offerta_giorni'])
    if 'lag_perfezionamento_giorni' in df.columns:
        df['lag_perfezionamento_giorni'] = df['lag_perfezionamento_giorni'].fillna(imp['lag_perfezionamento_giorni'])
    if 'pct_durata_sospesa' in df.columns:
        df['pct_durata_sospesa'] = df['pct_durata_sospesa'].fillna(0.0)
    if 'n_lotti_componenti' in df.columns:
        df['n_lotti_componenti'] = df['n_lotti_componenti'].fillna(1)

    flag_3cat = [
        'flag_delega', 'flag_vince_minimo', 'flag_progettazione_esterna',
        'consegna_frazionata', 'consegna_sotto_riserva', 'flag_in_ritardo',
    ]
    for col in flag_3cat:
        if col in df.columns:
            s = df[col]
            out = pd.Series('MISSING', index=s.index, dtype=object)
            out.loc[s == 0] = '0'
            out.loc[s == 1] = '1'
            # boolean/Int8 True/False
            if s.dtype == 'boolean' or str(s.dtype) == 'bool':
                out.loc[s == True]  = '1'
                out.loc[s == False] = '0'
            df[col] = _cat(out)

    def fixed_bins(series, conditions_labels):
        """conditions_labels: list of (mask_fn, label)"""
        out = pd.Series('MISSING', index=series.index, dtype=object)
        nn = series.notna()
        for fn, lbl in conditions_labels:
            out.loc[nn & fn(series)] = lbl
        return _cat(out)

    def quantile_bins(series, breaks, labels=('Q1','Q2','Q3','Q4')):
        out = pd.Series('MISSING', index=series.index, dtype=object)
        nn = series.notna()
        binned = pd.cut(series[nn], bins=breaks, labels=labels,
                        right=True, include_lowest=True, duplicates='drop')
        out.loc[nn] = binned.astype(str)
        return _cat(out)

    # reddito_irpef_procapite  [≤18K / (18K-22K] / (22K-27K] / (27K+]
    if 'reddito_irpef_procapite' in df.columns:
        c = df['reddito_irpef_procapite']
        df['reddito_irpef_procapite'] = fixed_bins(c, [
            (lambda x: x <= 18000,                         '≤18K'),
            (lambda x: (x > 18000) & (x <= 22000),         '(18K-22K]'),
            (lambda x: (x > 22000) & (x <= 27000),         '(22K-27K]'),
            (lambda x: x > 27000,                          '(27K+]'),
        ])

    # tasso_omicidi_100k  [0 / (0-0.5] / (0.5-1] / (1+]
    if 'tasso_omicidi_100k' in df.columns:
        c = df['tasso_omicidi_100k']
        df['tasso_omicidi_100k'] = fixed_bins(c, [
            (lambda x: x == 0,                             '0'),
            (lambda x: (x > 0) & (x <= 0.5),               '(0-0.5]'),
            (lambda x: (x > 0.5) & (x <= 1.0),             '(0.5-1]'),
            (lambda x: x > 1.0,                            '(1+]'),
        ])

    # pct_riserva_base  [≤0.05 / (0.05-0.15] / (0.15-0.30] / (0.30+]
    if 'pct_riserva_base' in df.columns:
        c = df['pct_riserva_base']
        df['pct_riserva_base'] = fixed_bins(c, [
            (lambda x: x <= 0.05,                          '≤0.05'),
            (lambda x: (x > 0.05) & (x <= 0.15),           '(0.05-0.15]'),
            (lambda x: (x > 0.15) & (x <= 0.30),           '(0.15-0.30]'),
            (lambda x: x > 0.30,                           '(0.30+]'),
        ])

    # importo_sicurezza_pct  bins fissi: [0 / (0-0.01] / (0.01-0.03] / (0.03+]
    # (distribuzione degenere: >75% dei non-null sono 0)
    if 'importo_sicurezza_pct' in df.columns:
        c = df['importo_sicurezza_pct']
        df['importo_sicurezza_pct'] = fixed_bins(c, [
            (lambda x: x == 0,                             '0'),
            (lambda x: (x > 0)    & (x <= 0.01),           '(0-0.01]'),
            (lambda x: (x > 0.01) & (x <= 0.03),           '(0.01-0.03]'),
            (lambda x: x > 0.03,                           '(0.03+]'),
        ])

    # tasso_disoccupazione  4 quantili M1
    if 'tasso_disoccupazione' in df.columns:
        df['tasso_disoccupazione'] = quantile_bins(
            df['tasso_disoccupazione'], qb['tasso_disoccupazione'])

    # lag_comunicazione_esito_giorni  [<0 / [0-365] / (365-730] / (730+]
    if 'lag_comunicazione_esito_giorni' in df.columns:
        c = df['lag_comunicazione_esito_giorni']
        df['lag_comunicazione_esito_giorni'] = fixed_bins(c, [
            (lambda x: x < 0,                              '<0'),
            (lambda x: (x >= 0) & (x <= 365),              '[0-365]'),
            (lambda x: (x > 365) & (x <= 730),             '(365-730]'),
            (lambda x: x > 730,                            '(730+]'),
        ])

    # pct_offerte_escluse  [0 / (0-0.25] / (0.25-0.5] / (0.5+]
    if 'pct_offerte_escluse' in df.columns:
        c = df['pct_offerte_escluse']
        df['pct_offerte_escluse'] = fixed_bins(c, [
            (lambda x: x == 0,                             '0'),
            (lambda x: (x > 0) & (x <= 0.25),              '(0-0.25]'),
            (lambda x: (x > 0.25) & (x <= 0.5),            '(0.25-0.5]'),
            (lambda x: x > 0.5,                            '(0.5+]'),
        ])

    # numero_offerte_ammesse  [1 / 2 / 3-5 / 6+]
    if 'numero_offerte_ammesse' in df.columns:
        c = df['numero_offerte_ammesse']
        df['numero_offerte_ammesse'] = fixed_bins(c, [
            (lambda x: x == 1,                             '1'),
            (lambda x: x == 2,                             '2'),
            (lambda x: (x >= 3) & (x <= 5),                '3-5'),
            (lambda x: x >= 6,                             '6+'),
        ])

    # ribasso_aggiudicazione  [<0 / [0-15] / (15-30] / (30+]
    if 'ribasso_aggiudicazione' in df.columns:
        c = df['ribasso_aggiudicazione']
        df['ribasso_aggiudicazione'] = fixed_bins(c, [
            (lambda x: x < 0,                              '<0'),
            (lambda x: (x >= 0) & (x <= 15),               '[0-15]'),
            (lambda x: (x > 15) & (x <= 30),               '(15-30]'),
            (lambda x: x > 30,                             '(30+]'),
        ])

    # ribasso_spread  [0-5] / (5-15] / (15+]
    if 'ribasso_spread' in df.columns:
        c = df['ribasso_spread']
        df['ribasso_spread'] = fixed_bins(c, [
            (lambda x: x <= 5,                             '[0-5]'),
            (lambda x: (x > 5) & (x <= 15),                '(5-15]'),
            (lambda x: x > 15,                             '(15+]'),
        ])

    # lag_aggiudicazione_giorni  [≤30 / (30-90] / (90-210] / (210+]
    if 'lag_aggiudicazione_giorni' in df.columns:
        c = df['lag_aggiudicazione_giorni']
        df['lag_aggiudicazione_giorni'] = fixed_bins(c, [
            (lambda x: x <= 30,                            '≤30'),
            (lambda x: (x > 30)  & (x <= 90),              '(30-90]'),
            (lambda x: (x > 90)  & (x <= 210),             '(90-210]'),
            (lambda x: x > 210,                            '(210+]'),
        ])

    # importo_aggiudicazione  4 quantili log (breaks M2)
    if 'importo_aggiudicazione' in df.columns:
        c = df['importo_aggiudicazione']
        out = pd.Series('MISSING', index=c.index, dtype=object)
        pos = c.notna() & (c > 0)
        binned = pd.cut(
            np.log1p(c[pos]),
            bins=qb['importo_aggiudicazione_log'],
            labels=['Q1','Q2','Q3','Q4'],
            right=True, include_lowest=True, duplicates='drop'
        )
        out.loc[pos] = binned.astype(str)
        out.loc[c.notna() & (c == 0)] = 'Q1'   # importo=0 -> primo bin
        df['importo_aggiudicazione'] = _cat(out)

    # num_imprese_offerenti  [1 / 2 / 3-5 / 6+]
    if 'num_imprese_offerenti' in df.columns:
        c = df['num_imprese_offerenti']
        df['num_imprese_offerenti'] = fixed_bins(c, [
            (lambda x: x == 1,                             '1'),
            (lambda x: x == 2,                             '2'),
            (lambda x: (x >= 3) & (x <= 5),                '3-5'),
            (lambda x: x >= 6,                             '6+'),
        ])

    # numero_offerte_escluse  [0 / 1 / 2-3 / 4+]
    if 'numero_offerte_escluse' in df.columns:
        c = df['numero_offerte_escluse']
        df['numero_offerte_escluse'] = fixed_bins(c, [
            (lambda x: x == 0,                             '0'),
            (lambda x: x == 1,                             '1'),
            (lambda x: (x >= 2) & (x <= 3),                '2-3'),
            (lambda x: x >= 4,                             '4+'),
        ])

    # lag_stipula_aggiudicazione_giorni  [≤30 / (30-90] / (90+]
    if 'lag_stipula_aggiudicazione_giorni' in df.columns:
        c = df['lag_stipula_aggiudicazione_giorni']
        df['lag_stipula_aggiudicazione_giorni'] = fixed_bins(c, [
            (lambda x: x <= 30,                            '≤30'),
            (lambda x: (x > 30) & (x <= 90),               '(30-90]'),
            (lambda x: x > 90,                             '(90+]'),
        ])

    # durata_pianificata_giorni  [≤60 / (60-180] / (180-365] / (365-730] / (730+]
    if 'durata_pianificata_giorni' in df.columns:
        c = df['durata_pianificata_giorni']
        df['durata_pianificata_giorni'] = fixed_bins(c, [
            (lambda x: x <= 60,                            '≤60'),
            (lambda x: (x > 60)  & (x <= 180),             '(60-180]'),
            (lambda x: (x > 180) & (x <= 365),             '(180-365]'),
            (lambda x: (x > 365) & (x <= 730),             '(365-730]'),
            (lambda x: x > 730,                            '(730+]'),
        ])

    # Categoriali con NA piccolo (<1%): imputazione con moda
    for col in ['tipo_scelta_4cls', 'oggetto_principale_contratto', 'natura_giuridica_SA']:
        if col in df.columns and col in imp:
            df[col] = _cat(df[col].fillna(imp[col]))

    # esito_collaudo: POSITIVO / NEGATIVO / MISSING
    if 'esito_collaudo' in df.columns:
        df['esito_collaudo'] = _cat(df['esito_collaudo'].fillna('MISSING'))

    # tipo_soggetto_agg: NaN -> MISSING
    if 'tipo_soggetto_agg' in df.columns:
        df['tipo_soggetto_agg'] = _cat(df['tipo_soggetto_agg'].fillna('MISSING'))

    # Final sweep: NA -> MISSING per tutte le colonne feature.
    # Garantisce che nei parquet preprocessed non ci siano NA (tranne 'label').
    # Copre variabili non trattate esplicitamente sopra:
    #   - oggetto_principale_contratto, sezione_regionale, tipo_scelta_4cls,
    #     tipo_lavorazione_macro, natura_giuridica_SA  (object)
    #   - cod_strumento_svolgimento, cod_motivo_urgenza  (Int64 nominale -> string cat)
    _id_skip = {'cig', 'label'}
    for col in df.columns:
        if col in _id_skip or not df[col].isnull().any():
            continue
        s = df[col]
        if hasattr(s, 'cat'):
            if 'MISSING' not in s.cat.categories:
                s = s.cat.add_categories(['MISSING'])
            df[col] = s.fillna('MISSING')
        elif s.dtype == object:
            df[col] = _cat(s.fillna('MISSING'))
        elif str(s.dtype).startswith('Int') or str(s.dtype) in ('int32', 'int64'):
            # Codici nominali interi (es. cod_strumento_svolgimento) -> string cat
            out = pd.Series('MISSING', index=s.index, dtype=object)
            nn = s.notna()
            out.loc[nn] = s[nn].astype(str)
            df[col] = _cat(out)
        else:
            print(f"  WARNING: NA residui in colonna numerica {col}: {s.isnull().sum():,}")

    return df


def count_dof(df, id_cols):
    """Effective DoF: for each categorical col = n_categories; for numeric = 1."""
    total = 0
    details = {}
    for c in df.columns:
        if c in id_cols:
            continue
        if hasattr(df[c], 'cat'):
            k = df[c].nunique()
            details[c] = k
            total += k
        else:
            details[c] = 1
            total += 1
    return total, details


print("\nBUILD PREPROCESSED")

dof_prep = {}
for name, mask, features in [
    ('M1', None,    M1_FEAT),
    ('M2', mask_m2, M2_FEAT),
    ('M3', mask_m3, M3_PREP_FEAT),
]:
    avail_feats = [c for c in features if c in pq.columns]
    cols = ID_COLS + avail_feats
    sub = pq.loc[mask, cols].copy() if mask is not None else pq[cols].copy()
    sub = apply_preprocessing(sub, qb, imp)
    path = f'{OUT_PREP}/{name}.parquet'
    sub.to_parquet(path, index=False)

    n_lab, nc, ns = labeled_stats(sub)
    total_dof, _ = count_dof(sub, ID_COLS)
    dof_prep[name] = (len(avail_feats), total_dof)
    print(f"  {name}: {len(sub):,} righe x {len(avail_feats)} feature "
          f"| DoF one-hot={total_dof} "
          f"| labeled={n_lab} (C={nc}/S={ns}) -> {path}")

print("\nRIEPILOGO")

models = [
    ('M1', None,    M1_FEAT,      M3_PREP_FEAT),
    ('M2', mask_m2, M2_FEAT,      M3_PREP_FEAT),
    ('M3', mask_m3, M3_NAT_FEAT,  M3_PREP_FEAT),
]
row_counts = {
    'M1': len(pq),
    'M2': int(mask_m2.sum()),
    'M3': int(mask_m3.sum()),
}
lab_counts = {
    'M1': labeled_stats(pq),
    'M2': labeled_stats(pq.loc[mask_m2]),
    'M3': labeled_stats(pq.loc[mask_m3]),
}

print(f"\n{'Modello':<8} {'Righe':>12} {'Labeled':>9} {'C':>5} {'S':>6} "
      f"{'FeatNat':>8} {'FeatPrep':>9} {'DoF(prep)':>10}")
print("-" * 75)
for nm in ['M1','M2','M3']:
    n_lab, nc, ns = lab_counts[nm]
    fn = dof_nativi[nm]
    fp, dof = dof_prep[nm]
    print(f"  {nm:<6} {row_counts[nm]:>12,} {n_lab:>9,} {nc:>5,} {ns:>6,} "
          f"{fn:>8} {fp:>9} {dof:>10}")

print("\nFiles scritti:")
for nm in ['M1','M2','M3']:
    print(f"  {OUT_NAT}/{nm}.parquet")
    print(f"  {OUT_PREP}/{nm}.parquet")
print("\nDone.")
