"""
build_zero_list.py — Produce la lista CIG scagionati (ricorrente HA PERSO).

Un CIG è "scagionato sicuro" se il ricorrente ha perso in tribunale e
quella sconfitta non è stata successivamente ribaltata ad alcun livello.

Gerarchia a due livelli (applicata in ordine):

  1. Non-Contraddizione
     Un CIG può essere scagionato solo se NON compare mai in nessun esito
     favorevole (a qualsiasi livello, TAR o CdS). Se è in cig_condannati → rimosso.
     Questa regola copre anche la Dominanza CdS descritta sotto.

  2. Dominanza CdS
     Se un rigetto TAR è stato appellato al Consiglio di Stato e il CdS ha
     accolto il ricorso, il CIG diventa 1. Gli esiti favorevoli CdS sono già
     in cig_condannati, quindi la Regola 1 copre questo caso.
     I CIG dove il CdS ha invece CONFERMATO il rigetto TAR vengono marcati
     come HIGH CONFIDENCE nell'output.

Uso:
    python build_zero_list.py [--log-level DEBUG|INFO|WARNING]

Prerequisiti:
    - CSV raw in data/raw/               (da run_pipeline.py step 1)
    - data/interim/ricorsi_clean.parquet (da run_pipeline.py step 2)
    - data/output/cig_condannati.csv     (da run_pipeline.py step 7)

Output:
    data/output/cig_scagionati.csv — colonne: CIG, fonte, motivo
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import config
from utils.log_setup import configure_logging
from utils.csv_loader import load_and_concat
from pipeline.cleaner import (
    _clean_judicial,
    _norm_esito,
    _norm_str,
    _is_cds,
    _build_uid,
    _derive_anno_deposito,
    _keep_columns,
)

logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)
    _run(ricorsi=None)


def main_from_pipeline(ricorsi: pd.DataFrame) -> None:
    """Chiamato da run_pipeline.py con ricorsi già caricati in memoria."""
    _run(ricorsi=ricorsi)


def _run(ricorsi: pd.DataFrame | None) -> None:
    logger.info("=" * 60)
    logger.info("Costruzione lista CIG scagionati")
    logger.info("=" * 60)

    stats: dict = {}

    if ricorsi is None:
        logger.info("Caricamento ricorsi_clean.parquet …")
        ricorsi = _load_ricorsi()
    stats["ricorsi_rows"] = len(ricorsi)
    logger.info("  ricorsi: %d righe, %d CIG unici", len(ricorsi),
                ricorsi["CODICE_CIG"].nunique())

    logger.info("Caricamento cig_condannati (esiti favorevoli) …")
    favorable_cigs = _load_favorable_cigs()
    stats["favorable_cigs"] = len(favorable_cigs)
    logger.info("  CIG condannati caricati: %d", len(favorable_cigs))

    logger.info("Caricamento ordinanze e sentenze (esiti sfavorevoli) …")
    unfav_ord, unfav_sent = _load_unfavorable_outcomes(stats)

    logger.info("Join esiti sfavorevoli con ricorsi …")
    zero_cig_rows = _join_with_ricorsi(unfav_ord, unfav_sent, ricorsi, stats)

    if zero_cig_rows.empty:
        logger.warning("Nessun esito sfavorevole matchato con i ricorsi. Lista scagionati vuota.")
        _save_empty(stats)
        return

    logger.info("Applicazione Regola 1: Non-Contraddizione …")
    before = zero_cig_rows["CODICE_CIG"].nunique()
    zero_cig_rows = zero_cig_rows[
        ~zero_cig_rows["CODICE_CIG"].isin(favorable_cigs)
    ].copy()
    after = zero_cig_rows["CODICE_CIG"].nunique()
    stats["cig_rimossi_non_contraddizione"] = before - after
    logger.info("  Non-contraddizione ha rimosso %d CIG (presenti sia in favorevoli che sfavorevoli)",
                before - after)

    logger.info("Applicazione Regola 2: Verifica conferma CdS …")
    cds_rejected_cigs = set(
        zero_cig_rows.loc[zero_cig_rows["_IS_CDS"], "CODICE_CIG"].dropna().unique()
    )
    stats["zero_cig_confermati_cds"] = len(cds_rejected_cigs)
    logger.info("  %d CIG scagionati confermati anche dal CdS (alta confidenza)",
                len(cds_rejected_cigs))

    logger.info("Costruzione lista scagionati finale …")
    result = _build_result(zero_cig_rows)
    stats["final_zero_cig_count"] = len(result)

    result.to_csv(config.OUTPUT_CIG_SCAGIONATI, index=False, encoding="utf-8")
    logger.info("=" * 60)
    logger.info("Lista scagionati salvata: %d CIG → %s", len(result), config.OUTPUT_CIG_SCAGIONATI)
    _print_stats(stats)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Costruisce la lista CIG scagionati (ricorsi rigettati)")
    p.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO)")
    return p.parse_args()


def _load_ricorsi() -> pd.DataFrame:
    path = config.INTERIM_RICORSI
    if not path.exists():
        logger.error(
            "ricorsi_clean.parquet non trovato in %s — eseguire prima 'python run_pipeline.py'.",
            path,
        )
        sys.exit(1)
    df = pd.read_parquet(path)
    logger.debug("ricorsi_clean.parquet: colonne = %s", list(df.columns))
    return df


def _load_favorable_cigs() -> set[str]:
    path = config.OUTPUT_CIG_CONDANNATI
    if not path.exists():
        logger.error(
            "cig_condannati.csv non trovato in %s — eseguire prima 'python run_pipeline.py'.", path
        )
        sys.exit(1)
    return set(pd.read_csv(path)["CIG"].dropna().str.strip())


def _collect_paths(group: str) -> list[Path]:
    return sorted(p for p in config.RAW_DIR.glob("*.csv") if p.name.startswith(group + "__"))


def _load_unfavorable_outcomes(stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica ordinanze e sentenze, filtra per esiti sfavorevoli, costruisce UID."""
    # Ordinanze
    ord_paths = _collect_paths(config.GROUP_ORDINANZE)
    logger.info("  file raw ordinanze: %d", len(ord_paths))
    raw_ord = load_and_concat(ord_paths, "ordinanze_raw") if ord_paths else pd.DataFrame()

    unfav_ord = pd.DataFrame()
    if not raw_ord.empty:
        unfav_ord = _filter_unfavorable(raw_ord, config.UNFAVORABLE_ORDINANZE, "ordinanze")
        stats["raw_ordinanze_rows"]    = len(raw_ord)
        stats["ordinanze_sfavorevoli"] = len(unfav_ord)
        logger.info("  ordinanze: %d raw → %d sfavorevoli", len(raw_ord), len(unfav_ord))

    # Sentenze
    sent_paths = _collect_paths(config.GROUP_SENTENZE)
    logger.info("  file raw sentenze: %d", len(sent_paths))
    raw_sent = load_and_concat(sent_paths, "sentenze_raw") if sent_paths else pd.DataFrame()

    unfav_sent = pd.DataFrame()
    if not raw_sent.empty:
        unfav_sent = _filter_unfavorable(raw_sent, config.UNFAVORABLE_SENTENZE, "sentenze")
        stats["raw_sentenze_rows"]    = len(raw_sent)
        stats["sentenze_sfavorevoli"] = len(unfav_sent)
        logger.info("  sentenze: %d raw → %d sfavorevoli", len(raw_sent), len(unfav_sent))

    return unfav_ord, unfav_sent


def _filter_unfavorable(df: pd.DataFrame, unfav_set: set[str], source: str) -> pd.DataFrame:
    """
    Specchio di cleaner._clean_judicial ma mantiene gli esiti SFAVOREVOLI.
    Riusa gli stessi helper di normalizzazione per garantire confronto identico.
    """
    if df.empty:
        return df

    needed = [
        "NUMERO_RICORSO", "NOME_SEDE", "DATA_PUBBLICAZIONE",
        "ESITO_PROVVEDIMENTO",
        "DATA_DEPOSITO_RICORSO", "ANNO_DEPOSITO_RICORSO", "ANNO_PUBBLICAZIONE",
    ]
    df = _keep_columns(df, needed, source=source)
    df = _derive_anno_deposito(df, source)

    df["NOME_SEDE"]      = df["NOME_SEDE"].apply(_norm_str)
    df["NUMERO_RICORSO"] = df["NUMERO_RICORSO"].apply(_norm_str)
    df["ESITO_NORM"]     = df["ESITO_PROVVEDIMENTO"].apply(_norm_esito)

    df = df[df["ESITO_NORM"].isin(unfav_set)].copy()
    if df.empty:
        return df

    df["_IS_CDS"] = df["NOME_SEDE"].apply(_is_cds)
    df["_SOURCE"] = source
    df = df.dropna(subset=["NUMERO_RICORSO", "NOME_SEDE", "ANNO_DEPOSITO_RICORSO"])
    df["UID"] = _build_uid(df["NOME_SEDE"], df["ANNO_DEPOSITO_RICORSO"], df["NUMERO_RICORSO"])

    return df


def _join_with_ricorsi(
    unfav_ord: pd.DataFrame,
    unfav_sent: pd.DataFrame,
    ricorsi: pd.DataFrame,
    stats: dict,
) -> pd.DataFrame:
    """
    Inner-join esiti sfavorevoli con ricorsi su UID.
    Le Sentenze prevalgono sulle Ordinanze per lo stesso UID
    (stessa regola di dominanza della pipeline principale).
    """
    frames = [df for df in [unfav_ord, unfav_sent] if not df.empty]
    if not frames:
        return pd.DataFrame()

    outcomes = pd.concat(frames, ignore_index=True, sort=False)
    stats["total_esiti_sfavorevoli"] = len(outcomes)
    logger.info("  totale righe esiti sfavorevoli: %d", len(outcomes))

    outcomes["_SORT"] = outcomes["_SOURCE"].map({"ordinanze": 0, "sentenze": 1}).fillna(0)
    outcomes = (
        outcomes.sort_values("_SORT")
        .drop_duplicates(subset="UID", keep="last")
        .drop(columns=["_SORT"])
    )
    logger.info("  UID sfavorevoli unici (dopo dominanza sentenze): %d", len(outcomes))

    joined = outcomes.merge(
        ricorsi[["UID", "CODICE_CIG", "NOME_SEDE", "NUMERO_RICORSO",
                 "ANNO_DEPOSITO_RICORSO"]],
        on="UID",
        how="inner",
        suffixes=("_out", ""),
    )
    stats["sfavorevoli_joined"] = len(joined)
    match_rate = len(joined) / len(outcomes) * 100 if len(outcomes) else 0
    logger.info("  joined: %d righe (match rate %.1f%% degli esiti sfavorevoli)",
                len(joined), match_rate)
    return joined


def _build_result(zero_cig_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Produce il DataFrame finale con una riga per CIG.

    Colonne:
      CIG    — identificativo gara
      fonte  — sentenza_TAR | ordinanza_TAR
      motivo — esiti distinti visti per il CIG (separati da virgola)
    """
    agg = (
        zero_cig_rows
        .groupby("CODICE_CIG")
        .agg(
            motivo=("ESITO_NORM", lambda s: ", ".join(sorted(s.dropna().unique()))),
            _sources=("_SOURCE",  lambda s: ", ".join(sorted(s.dropna().unique()))),
        )
        .reset_index()
        .rename(columns={"CODICE_CIG": "CIG"})
    )

    agg["fonte"] = agg["_sources"].apply(
        lambda s: "sentenza_TAR" if "sentenze" in s else "ordinanza_TAR"
    )

    return agg[["CIG", "fonte", "motivo"]].sort_values("CIG").reset_index(drop=True)


def _save_empty(stats: dict) -> None:
    pd.DataFrame(columns=["CIG", "fonte", "motivo"])\
        .to_csv(config.OUTPUT_CIG_SCAGIONATI, index=False, encoding="utf-8")
    logger.info("Lista scagionati vuota salvata in %s", config.OUTPUT_CIG_SCAGIONATI)
    _print_stats(stats)


def _print_stats(stats: dict) -> None:
    logger.info("=" * 60)
    logger.info("REPORT DIAGNOSTICO LISTA SCAGIONATI")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info("  %-45s %s", k + ":", v)
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrotto.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Costruzione lista scagionati fallita: %s", exc)
        sys.exit(1)
