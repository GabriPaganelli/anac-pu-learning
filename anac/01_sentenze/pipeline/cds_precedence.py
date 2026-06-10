"""
pipeline/cds_precedence.py — Step 6: Consiglio di Stato dominance.

Logic:
  - CDS and TAR have different case numbers; linkage is via CODICE_CIG.
  - For each CIG in the TAR blacklist, check if there is a CDS outcome:
      * CDS outcome in CDS_REMOVE_OUTCOMES  → remove CIG from blacklist.
      * CDS outcome in CDS_AMBIGUOUS_OUTCOMES → keep CIG, log as UNCERTAIN.
      * CDS outcome favourable (accoglie etc.) → keep CIG (CDS confirmed TAR win).
      * No CDS outcome → keep CIG (TAR result stands).
  - If a CIG has multiple CDS outcomes (e.g., different appeals), the most
    recent DATA_PUBBLICAZIONE takes precedence.
"""

import logging

import pandas as pd

import config
from pipeline.cleaner import _DECISION_LOG, _norm_esito

logger = logging.getLogger(__name__)


def apply_cds_dominance(
    tar_cig_set: set[str],
    cds_joined: pd.DataFrame,
) -> tuple[set[str], pd.DataFrame]:
    """
    Apply CDS dominance rules to the TAR CIG blacklist.

    Args:
        tar_cig_set:  Set of CIG codes produced by the TAR pipeline.
        cds_joined:   CDS rows that were joined to ricorsi (have CODICE_CIG).

    Returns:
        final_cig_set:  Updated CIG set after CDS overrides.
        cds_summary:    DataFrame with one row per CIG showing the CDS outcome applied.
    """
    if cds_joined.empty:
        logger.info("cds_precedence: no CDS outcomes — TAR results unchanged.")
        return tar_cig_set.copy(), pd.DataFrame()

    summary_rows: list[dict] = []
    final_cigs = tar_cig_set.copy()

    # Take the most recent CDS outcome per CIG
    cds = cds_joined.copy()
    cds["_PUB_DATE_PARSED"] = pd.to_datetime(
        cds["DATA_PUBBLICAZIONE"], errors="coerce", dayfirst=True
    )
    cds_by_cig = (
        cds.sort_values("_PUB_DATE_PARSED", ascending=False, na_position="last")
        .drop_duplicates(subset="CODICE_CIG", keep="first")
    )

    removed_count    = 0
    uncertain_count  = 0
    confirmed_count  = 0

    for _, row in cds_by_cig.iterrows():
        cig         = row["CODICE_CIG"]
        esito_norm  = _norm_esito(row.get("ESITO_PROVVEDIMENTO", ""))
        esito_raw   = str(row.get("ESITO_PROVVEDIMENTO", ""))
        pub_date    = str(row.get("DATA_PUBBLICAZIONE", ""))

        if cig not in tar_cig_set:
            # CIG was already not in TAR blacklist — CDS outcome irrelevant here
            continue

        if esito_norm in config.CDS_REMOVE_OUTCOMES:
            final_cigs.discard(cig)
            action = "REMOVED"
            removed_count += 1
            _DECISION_LOG.append({
                "rule": "cds_removes_tar_win",
                "reason": f"CDS outcome '{esito_raw}' overrides TAR win",
                "confidence": "HIGH",
                "uid": "",
                "source_file": "cds_precedence",
                "extra": cig,
            })
        elif esito_norm in config.CDS_AMBIGUOUS_OUTCOMES:
            action = "UNCERTAIN_KEPT"
            uncertain_count += 1
            _DECISION_LOG.append({
                "rule": "cds_ambiguous",
                "reason": f"CDS outcome '{esito_raw}' is ambiguous — CIG kept but flagged",
                "confidence": "LOW",
                "uid": "",
                "source_file": "cds_precedence",
                "extra": cig,
            })
        else:
            action = "CONFIRMED"
            confirmed_count += 1

        summary_rows.append({
            "CODICE_CIG":          cig,
            "CDS_ESITO":           esito_raw,
            "CDS_ESITO_NORM":      esito_norm,
            "CDS_DATA_PUBBL":      pub_date,
            "ACTION":              action,
        })

    logger.info(
        "cds_precedence: %d CIGs examined → %d removed, %d uncertain (kept), %d confirmed",
        len(cds_by_cig), removed_count, uncertain_count, confirmed_count,
    )
    logger.info(
        "cds_precedence: TAR blacklist %d → %d after CDS overrides",
        len(tar_cig_set), len(final_cigs),
    )

    cds_summary = pd.DataFrame(summary_rows)
    return final_cigs, cds_summary
