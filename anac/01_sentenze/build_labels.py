"""
build_labels.py — Combina TAR/CdS e fine-contratto ANAC per produrre i label finali.

Fonti dei condannati (label=1):
  - Sentenze TAR con esito favorevole al ricorrente (da run_pipeline.py)
  - Fine-contratto ANAC con causa "REATI ACCERTATI" (cod_motivo_risoluzione = 4)
  - Fine-contratto ANAC con causa "CODICE ANTIMAFIA" (cod_motivo_interruzione_anticipata = 6)

Fonte degli scagionati (label=0):
  - Sentenze TAR con esito sfavorevole al ricorrente (da build_zero_list.py)
  - Un CIG che compare tra i condannati viene rimosso dagli scagionati (non-contraddizione).

Output (in labels/):
  labels/cig_condannati.csv — CIG, fonte, motivo
  labels/cig_scagionati.csv — CIG, fonte, motivo
"""

import logging
import sys
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Codici fine-contratto che indicano illecito accertato (nel CSV sono float: "4.0", "6.0")
MOTIVO_RISOLUZIONE_REATI        = "4.0"   # REATI ACCERTATI
MOTIVO_INTERRUZIONE_ANTIMAFIA   = "6.0"   # RECESSO DAL CONTRATTO (CODICE ANTIMAFIA)

FINE_CONTRATTO_CSV = config.BASE_DIR.parent / "data" / "raw" / "fine-contratto.csv"


def main() -> None:
    logger.info("Build labels — avvio")

    tar_condannati = _load_tar_condannati()
    logger.info("  TAR condannati: %d CIG", len(tar_condannati))

    fc_condannati = _load_fine_contratto_condannati()
    logger.info("  Fine-contratto condannati: %d CIG nuovi", fc_condannati["CIG"].nunique())

    # fine-contratto non sovrascrive entry TAR già presenti
    all_condannati = pd.concat([tar_condannati, fc_condannati], ignore_index=True)
    all_condannati = all_condannati.drop_duplicates(subset="CIG", keep="first")
    all_condannati = all_condannati.sort_values("CIG").reset_index(drop=True)

    condannati_cig_set = set(all_condannati["CIG"])
    logger.info("  Totale condannati (TAR + fine-contratto): %d CIG", len(condannati_cig_set))

    out_condannati = config.LABELS_DIR / "cig_condannati.csv"
    all_condannati.to_csv(out_condannati, index=False, encoding="utf-8")
    logger.info("  Salvati in %s", out_condannati)

    scagionati = _load_scagionati(condannati_cig_set)
    logger.info("  Scagionati (dopo filtro): %d CIG", len(scagionati))

    out_scagionati = config.LABELS_DIR / "cig_scagionati.csv"
    scagionati.to_csv(out_scagionati, index=False, encoding="utf-8")
    logger.info("  Salvati in %s", out_scagionati)

    logger.info("Build labels completato.")


def _load_tar_condannati() -> pd.DataFrame:
    """Carica la lista CIG condannati prodotta dalla pipeline TAR."""
    path = config.OUTPUT_CIG_CONDANNATI
    if not path.exists():
        logger.error("cig_condannati.csv non trovato in %s — eseguire prima run_pipeline.py.", path)
        sys.exit(1)
    df = pd.read_csv(path, dtype=str)
    df["CIG"] = df["CIG"].str.strip()
    # Garantisce le colonne attese anche se il file è solo "CIG"
    if "fonte" not in df.columns:
        df["fonte"] = "sentenza_TAR"
    if "motivo" not in df.columns:
        df["motivo"] = ""
    df["fonte"] = df["fonte"].fillna("sentenza_TAR")
    return df[["CIG", "fonte", "motivo"]]


def _load_fine_contratto_condannati() -> pd.DataFrame:
    """
    Estrae da fine-contratto.csv i CIG chiusi per reato o codice antimafia.
    Restituisce solo CIG non già presenti in tar_condannati (la dedup avviene nel chiamante).
    """
    if not FINE_CONTRATTO_CSV.exists():
        logger.warning("fine-contratto.csv non trovato in %s — fonte saltata.", FINE_CONTRATTO_CSV)
        return pd.DataFrame(columns=["CIG", "fonte", "motivo"])

    fc = pd.read_csv(FINE_CONTRATTO_CSV, sep=";", dtype=str, usecols=[
        "cig", "cod_motivo_risoluzione", "motivo_risoluzione",
        "cod_motivo_interruzione_anticipata", "motivo_interruzione_anticipata",
    ])
    fc["cig"] = fc["cig"].str.strip()

    mask_reati = (
        fc["cod_motivo_risoluzione"].str.strip() == MOTIVO_RISOLUZIONE_REATI
    )
    mask_antimafia = (
        fc["cod_motivo_interruzione_anticipata"].str.strip() == MOTIVO_INTERRUZIONE_ANTIMAFIA
    )
    fc_labeled = fc[mask_reati | mask_antimafia].copy()

    def _motivo(row: pd.Series) -> str:
        parti = []
        if str(row.get("cod_motivo_risoluzione", "")).strip() == MOTIVO_RISOLUZIONE_REATI:
            parti.append(row.get("motivo_risoluzione", "REATI ACCERTATI"))
        if str(row.get("cod_motivo_interruzione_anticipata", "")).strip() == MOTIVO_INTERRUZIONE_ANTIMAFIA:
            parti.append(row.get("motivo_interruzione_anticipata", "CODICE ANTIMAFIA"))
        return "; ".join(str(p) for p in parti if pd.notna(p))

    fc_labeled["motivo"] = fc_labeled.apply(_motivo, axis=1)
    fc_labeled["fonte"]  = "fine_contratto"
    fc_labeled = fc_labeled.rename(columns={"cig": "CIG"})
    fc_labeled = fc_labeled.drop_duplicates(subset="CIG", keep="first")

    return fc_labeled[["CIG", "fonte", "motivo"]]


def _load_scagionati(condannati_cig_set: set[str]) -> pd.DataFrame:
    """
    Carica la lista scagionati e rimuove eventuali CIG presenti tra i condannati.
    """
    path = config.OUTPUT_CIG_SCAGIONATI
    if not path.exists():
        logger.error("cig_scagionati.csv non trovato in %s — eseguire prima build_zero_list.py.", path)
        sys.exit(1)
    df = pd.read_csv(path, dtype=str)
    df["CIG"] = df["CIG"].str.strip()

    before = len(df)
    df = df[~df["CIG"].isin(condannati_cig_set)].copy()
    removed = before - len(df)
    if removed:
        logger.info("  Non-contraddizione: rimossi %d CIG da scagionati (presenti anche tra condannati)",
                    removed)

    return df.sort_values("CIG").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        logger.exception("build_labels fallito: %s", exc)
        sys.exit(1)
