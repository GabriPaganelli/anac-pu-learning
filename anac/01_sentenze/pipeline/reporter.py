"""
pipeline/reporter.py — Step 7: salvataggio output e report diagnostico.

Output:
  data/output/cig_condannati.csv   — CIG con sentenza favorevole al ricorrente
                                     (join esatto TAR + filtro CdS)
  data/output/diagnostic_report.json — statistiche chiave della pipeline
  logs/decision_log.csv            — decisioni non banali del cleaner
"""

import json
import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def save_cig_condannati(cig_set: set[str]) -> Path:
    """Salva la lista CIG condannati (join esatto + dominanza CdS) in CSV."""
    df = pd.DataFrame(sorted(cig_set), columns=["CIG"])
    df.to_csv(config.OUTPUT_CIG_CONDANNATI, index=False)
    logger.info("Condannati: %d CIG → %s", len(df), config.OUTPUT_CIG_CONDANNATI)
    return config.OUTPUT_CIG_CONDANNATI


def save_diagnostic_report(stats: dict) -> Path:
    """Salva il report diagnostico in JSON."""
    config.OUTPUT_DIAGNOSTIC.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Report diagnostico salvato in %s", config.OUTPUT_DIAGNOSTIC)
    _print_report(stats)
    return config.OUTPUT_DIAGNOSTIC


def save_decision_log(df: pd.DataFrame) -> Path:
    if df.empty:
        logger.info("Decision log vuoto — niente da salvare.")
        return config.LOG_DECISIONS
    df.to_csv(config.LOG_DECISIONS, index=False, encoding="utf-8")
    logger.info("Decision log: %d righe → %s", len(df), config.LOG_DECISIONS)
    return config.LOG_DECISIONS


def save_intermediate(df: pd.DataFrame, path: Path, label: str = "") -> None:
    """Salva un DataFrame intermedio come Parquet (per restart della pipeline)."""
    if df.empty:
        logger.warning("Intermedio [%s] vuoto — salvataggio saltato.", label or path.name)
        return
    df.to_parquet(path, index=False)
    logger.debug("Intermedio salvato: %s (%d righe)", path.name, len(df))


def load_intermediate(path: Path) -> pd.DataFrame:
    """Carica un file Parquet intermedio precedentemente salvato."""
    if path.exists():
        df = pd.read_parquet(path)
        logger.debug("Intermedio caricato: %s (%d righe)", path.name, len(df))
        return df
    return pd.DataFrame()


def _print_report(stats: dict) -> None:
    logger.info("=" * 60)
    logger.info("REPORT DIAGNOSTICO PIPELINE")
    logger.info("=" * 60)
    for key, value in stats.items():
        logger.info("  %-45s %s", key + ":", value)
    logger.info("=" * 60)
