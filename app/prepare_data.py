"""
app/prepare_data.py — Script one-time: pre-encoding dei parquet nativi.
Esegui UNA VOLTA prima di avviare l'app:
    python app/prepare_data.py

Produce:
    app/data/M{1,2,3}_ready.parquet  — feature matrix (float32) + label
    app/data/encodings.json          — {col: [sorted_unique_values]} per le
                                       colonne categoriali convertite

Note:
  - Carica TUTTE le righe (non solo fold-assigned): serve per la distribuzione
    di riferimento del rank_pct (tutti gli appalti 2008-oggi).
  - Compatibilità backward: se il parquet ha cod_modalita_realizzazione ma non
    modalita_realizzazione_macro, applica il mapping al volo.
  - La formula anno ISTAT è min(anno_pub − 1, 2024): già applicata nel pipeline.
    I parquet esistenti usano la vecchia formula (min(anno_pub, 2024) − 1),
    con differenza max di 1 anno. L'impatto sul rank_pct è trascurabile.
"""

import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NATIVI_PATH  = str(ROOT / "anac" / "output" / "parquet" / "model" / "nativi")
LABEL_COL    = "label"
FOLD_COL     = "fold"
COLS_TO_DROP = ["cig", "esito", "anno_pubblicazione", "regione"]


def _convert_categoricals_nativi(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Converte colonne object/category in float32 integer codes (LightGBM native).
    NA in categoriali → NaN. Ritorna (df_convertito, cat_feature_names)."""
    drop_set = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    cat_cols = [c for c in df.columns
                if c not in drop_set and df[c].dtype.kind not in "iufb"]
    for col in cat_cols:
        df[col] = (df[col].astype("category")
                          .cat.codes
                          .replace(-1, np.nan)
                          .astype("float32"))
    return df, cat_cols


M1_FEAT = [
    "importo_complessivo_gara", "importo_lotto", "n_lotti_componenti",
    "oggetto_principale_contratto", "modalita_realizzazione_macro", "sezione_regionale",
    "cod_strumento_svolgimento", "flag_urgenza", "cod_motivo_urgenza",
    "flag_delega", "finestra_offerta_giorni", "lag_perfezionamento_giorni",
    "flag_accordo_quadro", "flag_ripetizioni", "settore_speciale", "flag_appalto_riservato",
    "tipo_scelta_4cls", "cpv_macro_categoria", "importo_sicurezza_pct", "pct_riserva_base",
    "natura_giuridica_SA",
    "tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k",
]

M2_FEAT = M1_FEAT + [
    "numero_offerte_ammesse", "numero_offerte_escluse", "num_imprese_offerenti",
    "pct_offerte_escluse", "ribasso_aggiudicazione", "ribasso_spread",
    "flag_vince_minimo", "flag_progettazione_esterna", "lag_aggiudicazione_giorni",
    "lag_comunicazione_esito_giorni", "asta_elettronica", "flag_scomputo",
    "flag_proc_accelerata", "importo_aggiudicazione", "tipo_soggetto_agg", "flag_subappalto",
]

M3_FEAT = M2_FEAT + [
    "lag_stipula_aggiudicazione_giorni", "durata_pianificata_giorni",
    "consegna_frazionata", "consegna_sotto_riserva",
    "n_varianti", "flag_variante_sostanziale", "flag_variante_oltre_termine",
    "n_sospensioni", "durata_totale_sospensioni_gg",
    "flag_sospensione", "flag_sosp_giudiziaria",
    "pct_durata_sospesa", "n_sal", "flag_in_ritardo", "flag_proroga", "esito_collaudo",
    "pct_overrun_variante", "pct_vita_prima_variante",
    "pct_overrun_core", "pct_riserva_consumata",
]

FEAT_BY_STAGE = {1: M1_FEAT, 2: M2_FEAT, 3: M3_FEAT}


_MODALITA_MAP = {
    1: "APPALTO",
    7: "ECONOMIA",
    2: "ACCORDO_QUADRO",  9: "ACCORDO_QUADRO", 11: "ACCORDO_QUADRO",
    17: "ACCORDO_QUADRO", 18: "ACCORDO_QUADRO",
    3: "CONCESSIONE",     4: "CONCESSIONE",    20: "CONCESSIONE",    21: "CONCESSIONE",
    5: "PPP",             6: "PPP",             8: "PPP",
    12: "PPP",           13: "PPP",
    10: "ALTRO",         14: "ALTRO",          15: "ALTRO",
    16: "ALTRO",         19: "ALTRO",          90: "ALTRO",         999: "ALTRO",
}

OUT_DIR = ROOT / "app" / "data"


def _apply_backward_compat(df: pd.DataFrame) -> pd.DataFrame:
    """Se il parquet ha cod_modalita_realizzazione ma non modalita_realizzazione_macro,
    costruisce la colonna aggregata al volo."""
    if "modalita_realizzazione_macro" not in df.columns:
        if "cod_modalita_realizzazione" in df.columns:
            df["modalita_realizzazione_macro"] = (
                df["cod_modalita_realizzazione"]
                .map(_MODALITA_MAP)
                .astype("object")       # evita problemi dtype con category su parquet vecchi
            )
            print("  [compat] modalita_realizzazione_macro costruita da cod_modalita_realizzazione")
        else:
            print("  [WARN] Né modalita_realizzazione_macro né cod_modalita_realizzazione presenti")
    return df


def _extract_encodings(df: pd.DataFrame, encodings: dict) -> None:
    """Salva in encodings il mapping {colonna: [sorted_unique_values]} per ogni
    colonna categoriale (object/category, non Int64/float/bool) che sarà convertita
    da _convert_categoricals_nativi. Chiamare PRIMA della conversione."""
    drop_set = set(COLS_TO_DROP + [LABEL_COL, FOLD_COL])
    for col in df.columns:
        if col in drop_set or df[col].dtype.kind in "iufb":
            continue
        if col not in encodings:
            unique_vals = sorted(
                v for v in df[col].dropna().unique().tolist()
                if v is not None
            )
            encodings[col] = unique_vals
            print(f"    encoding [{col}]: {len(unique_vals)} valori -> {unique_vals[:6]}{'...' if len(unique_vals)>6 else ''}")


def process_stage(mn: int, encodings: dict) -> None:
    print(f"\n{'='*58}")
    print(f"  M{mn}  —  {Path(NATIVI_PATH) / f'M{mn}.parquet'}")
    print(f"{'='*58}")

    feats   = FEAT_BY_STAGE[mn]
    path    = Path(NATIVI_PATH) / f"M{mn}.parquet"

    # Carica colonne necessarie (label + feature; NO fold — vogliamo tutte le righe)
    available_in_parquet = pq.read_schema(str(path)).names
    base_cols = list({LABEL_COL} | (set(feats) & set(available_in_parquet)))
    # Aggiungi cod_modalita_realizzazione se serve per la compat
    if ("modalita_realizzazione_macro" not in available_in_parquet
            and "cod_modalita_realizzazione" in available_in_parquet):
        base_cols = list(set(base_cols) | {"cod_modalita_realizzazione"})

    df = pd.read_parquet(path, columns=base_cols)
    print(f"  Caricate {len(df):,} righe × {len(base_cols)} colonne")

    # Backward compat
    df = _apply_backward_compat(df)

    # Verifica feature disponibili
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"  [WARN] Feature mancanti nel parquet M{mn}: {missing}")
    available_feats = [f for f in feats if f in df.columns]

    # Estrai encoding PRIMA della conversione
    _extract_encodings(df, encodings)

    # Applica _convert_categoricals_nativi (converte object/category → float32 codes)
    df, cat_names = _convert_categoricals_nativi(df)
    print(f"  Categoriali convertite: {cat_names}")

    # Mantieni solo label + feature disponibili
    keep_cols = [LABEL_COL] + available_feats
    df_out = df[[c for c in keep_cols if c in df.columns]].copy()

    # Converti tutto in float32 (tranne label che resta float64)
    for col in available_feats:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype("float32")

    label_vals = df_out[LABEL_COL].to_numpy(dtype=float, na_value=np.nan)
    n_P = int((label_vals == 1).sum())
    n_N = int((label_vals == 0).sum())
    n_U = int(np.isnan(label_vals).sum())
    print(f"  Righe output: {len(df_out):,}  |  P={n_P:,}  N={n_N:,}  U={n_U:,}")
    print(f"  Colonne output: {len(df_out.columns)}  ({df_out.memory_usage(deep=True).sum() / 1e6:.0f} MB)")

    out_path = OUT_DIR / f"M{mn}_ready.parquet"
    df_out.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"  Salvato -> {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encodings: dict = {}

    for mn in [1, 2, 3]:
        process_stage(mn, encodings)

    enc_path = OUT_DIR / "encodings.json"
    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(encodings, f, ensure_ascii=False, indent=2)
    print(f"\nEncodings salvati -> {enc_path}  ({len(encodings)} colonne)")

    print("\nSetup completato. Avvia l'app con:  streamlit run app/app.py")


if __name__ == "__main__":
    main()
