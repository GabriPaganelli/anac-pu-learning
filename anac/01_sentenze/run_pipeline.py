"""
run_pipeline.py — Entry point per la pipeline OpenGA CIG.

Uso:
    python run_pipeline.py [--no-cache] [--dry-run] [--log-level DEBUG|INFO|WARNING]
                           [--skip-download]   # usa file già scaricati

Passi:
    1. Estrazione  — scarica tutti i CSV dal portale OpenGA
    2. Pulizia     — seleziona colonne, valida CIG, costruisce UID
    3. Dedup       — risolve UID duplicati per dataset
    4. Consolida   — unisce Ordinanze + Sentenze (Sentenza vince)
    5. Join        — inner join esiti con i Ricorsi
    6. CdS         — applica la dominanza del Consiglio di Stato
    7. Output      — salva cig_condannati, log, report diagnostico
    8. Scagionati  — costruisce cig_scagionati tramite build_zero_list
    9. Labels      — copia i file finali in labels/ tramite build_labels
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import config
from utils.log_setup import configure_logging
from utils.csv_loader import load_and_concat

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline OpenGA CIG")
    p.add_argument("--no-cache",      action="store_true", help="Forza il re-download di tutti i CSV")
    p.add_argument("--dry-run",       action="store_true", help="Scopre gli URL senza scaricare")
    p.add_argument("--skip-download", action="store_true", help="Usa i file raw già scaricati")
    p.add_argument("--log-level",     default="INFO",      help="Livello di log (default: INFO)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Pipeline OpenGA CIG — avvio")
    logger.info("=" * 60)

    stats: dict = {}

    logger.info("STEP 1 — Estrazione")

    if args.skip_download:
        paths_by_group = _collect_existing_raw_files()
    else:
        if args.no_cache:
            config.CACHE_ENABLED = False
        from pipeline import extractor
        paths_by_group = extractor.run(dry_run=args.dry_run)

    stats["csvs_ricorsi"]        = len(paths_by_group.get(config.GROUP_RICORSI,   []))
    stats["csvs_ordinanze"]      = len(paths_by_group.get(config.GROUP_ORDINANZE, []))
    stats["csvs_sentenze"]       = len(paths_by_group.get(config.GROUP_SENTENZE,  []))
    stats["total_csvs_scaricati"] = sum(stats[k] for k in
        ["csvs_ricorsi", "csvs_ordinanze", "csvs_sentenze"])
    logger.info("File: %d ricorsi, %d ordinanze, %d sentenze",
                stats["csvs_ricorsi"], stats["csvs_ordinanze"], stats["csvs_sentenze"])

    if args.dry_run:
        logger.info("[DRY-RUN] Fermato dopo la scoperta degli URL.")
        return

    logger.info("STEP 2 — Caricamento e pulizia")
    from pipeline.cleaner import clean_ricorsi, clean_ordinanze, clean_sentenze

    raw_ricorsi   = load_and_concat(paths_by_group.get(config.GROUP_RICORSI,   []), "ricorsi")
    raw_ordinanze = load_and_concat(paths_by_group.get(config.GROUP_ORDINANZE, []), "ordinanze")
    raw_sentenze  = load_and_concat(paths_by_group.get(config.GROUP_SENTENZE,  []), "sentenze")

    _log_sample(raw_ricorsi,   "raw_ricorsi")
    _log_sample(raw_ordinanze, "raw_ordinanze")
    _log_sample(raw_sentenze,  "raw_sentenze")

    stats["raw_rows_ricorsi"]   = len(raw_ricorsi)
    stats["raw_rows_ordinanze"] = len(raw_ordinanze)
    stats["raw_rows_sentenze"]  = len(raw_sentenze)

    ricorsi   = clean_ricorsi(raw_ricorsi)
    ordinanze = clean_ordinanze(raw_ordinanze)
    sentenze  = clean_sentenze(raw_sentenze)

    del raw_ricorsi, raw_ordinanze, raw_sentenze

    stats["ricorsi_con_cig_valido"] = len(ricorsi)
    stats["ricorsi_scartati_senza_cig"] = stats["raw_rows_ricorsi"] - len(ricorsi)
    stats["ordinanze_favorevoli"] = len(ordinanze)
    stats["sentenze_favorevoli"]  = len(sentenze)

    from pipeline.reporter import save_intermediate
    save_intermediate(ricorsi,   config.INTERIM_RICORSI,   "ricorsi")
    save_intermediate(ordinanze, config.INTERIM_ORDINANZE, "ordinanze")
    save_intermediate(sentenze,  config.INTERIM_SENTENZE,  "sentenze")

    logger.info("STEP 3 — Deduplicazione")
    from pipeline.deduplicator import resolve_duplicates

    ordinanze = resolve_duplicates(ordinanze, source="ordinanze")
    sentenze  = resolve_duplicates(sentenze,  source="sentenze")
    ricorsi   = resolve_duplicates(ricorsi,   source="ricorsi")

    stats["dedup_ordinanze"] = len(ordinanze)
    stats["dedup_sentenze"]  = len(sentenze)
    stats["dedup_ricorsi"]   = len(ricorsi)

    logger.info("STEP 4 — Consolidamento esiti")
    from pipeline.consolidator import consolidate

    tar_ordinanze = ordinanze[~ordinanze["_IS_CDS"]].copy()
    tar_sentenze  = sentenze[~sentenze["_IS_CDS"]].copy()
    cds_ordinanze = ordinanze[ordinanze["_IS_CDS"]].copy()
    cds_sentenze  = sentenze[sentenze["_IS_CDS"]].copy()

    tar_outcomes = consolidate(tar_ordinanze, tar_sentenze)
    cds_outcomes = consolidate(cds_ordinanze, cds_sentenze)

    stats["tar_esiti_totali"] = len(tar_outcomes)
    stats["cds_esiti_totali"] = len(cds_outcomes)

    save_intermediate(tar_outcomes, config.INTERIM_OUTCOMES, "tar_outcomes")
    save_intermediate(cds_outcomes, config.INTERIM_CDS,      "cds_outcomes")

    logger.info("STEP 5 — Join esiti con Ricorsi")
    from pipeline.joiner import join_outcomes_to_ricorsi

    tar_joined, cds_joined, unmatched = join_outcomes_to_ricorsi(tar_outcomes, ricorsi)

    from pipeline.cleaner import _is_cds
    cds_ricorsi = ricorsi[ricorsi["NOME_SEDE"].apply(_is_cds)].copy()
    _, cds_joined_full, _ = join_outcomes_to_ricorsi(cds_outcomes, cds_ricorsi)

    stats["tar_righe_joined"]    = len(tar_joined)
    stats["cds_righe_joined"]    = len(cds_joined_full)
    stats["esiti_non_matchati"]  = len(unmatched)

    save_intermediate(tar_joined, config.INTERIM_JOINED, "tar_joined")

    logger.info("STEP 6 — Dominanza Consiglio di Stato")
    from pipeline.cds_dominator import apply_cds_dominance

    tar_cig_set = set(tar_joined["CODICE_CIG"].dropna().unique())
    logger.info("CIG condannati prima del filtro CdS: %d", len(tar_cig_set))

    condannati_cig_set, _ = apply_cds_dominance(tar_cig_set, cds_joined_full)
    stats["cig_prima_cds"]     = len(tar_cig_set)
    stats["cig_rimossi_da_cds"] = len(tar_cig_set) - len(condannati_cig_set)
    stats["cig_condannati_count"] = len(condannati_cig_set)

    logger.info("STEP 7 — Salvataggio output")
    from pipeline.cleaner  import get_decision_log
    from pipeline.reporter import save_cig_condannati, save_diagnostic_report, save_decision_log

    save_cig_condannati(condannati_cig_set)
    save_diagnostic_report(stats)
    save_decision_log(get_decision_log())

    logger.info("CIG condannati (TAR): %d", len(condannati_cig_set))

    logger.info("STEP 8 — Costruzione lista scagionati")
    import build_zero_list
    build_zero_list.main_from_pipeline(ricorsi=ricorsi)

    logger.info("STEP 9 — Build labels → labels/")
    import build_labels
    build_labels.main()

    logger.info("Pipeline completata.")


def _log_sample(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        logger.warning("Dataset '%s' vuoto dopo il caricamento.", label)
        return
    logger.info("Dataset '%s': %d righe, %d colonne: %s",
                label, len(df), len(df.columns), list(df.columns))
    for col in ["ESITO_PROVVEDIMENTO", "NOME_SEDE"]:
        if col in df.columns:
            top = df[col].value_counts().head(5)
            logger.debug("  %s valori top:\n%s", col, top.to_string())


def _collect_existing_raw_files() -> dict[str, list[Path]]:
    """Raccoglie i file già scaricati in data/raw/ per prefisso di gruppo."""
    result: dict[str, list[Path]] = {
        config.GROUP_RICORSI:   [],
        config.GROUP_ORDINANZE: [],
        config.GROUP_SENTENZE:  [],
    }
    for p in config.RAW_DIR.glob("*.csv"):
        for group in result:
            if p.name.startswith(group + "__"):
                result[group].append(p)
                break
    for g, ps in result.items():
        logger.info("File raw esistenti [%s]: %d", g, len(ps))
    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrotta dall'utente.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline fallita con errore: %s", exc)
        sys.exit(1)
