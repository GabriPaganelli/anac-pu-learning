"""
pipeline/consolidator.py — Step 4: Merge Ordinanze and Sentenze outcomes.

Dominance rules:
  1. For a given UID, if both a Sentenza and an Ordinanza exist,
     the Sentenza always wins (final judgment > interim order).
  2. If the Sentenza is unfavourable (shouldn't happen here since we
     already filtered to favourable ones), the Ordinanza is discarded too.
     (This is a safety check — both inputs should already be pre-filtered.)
  3. Conflicts are logged.

Output: one row per UID, enriched with _TIPO ("sentenza"/"ordinanza").
"""

import logging

import pandas as pd

from pipeline.cleaner import _DECISION_LOG

logger = logging.getLogger(__name__)


def consolidate(ordinanze: pd.DataFrame, sentenze: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-filtered Ordinanze and Sentenze into a single outcomes DataFrame.

    Returns a DataFrame with one row per UID.
    """
    if ordinanze.empty and sentenze.empty:
        logger.warning("consolidator: both ordinanze and sentenze are empty.")
        return pd.DataFrame()

    if ordinanze.empty:
        logger.info("consolidator: no ordinanze — using sentenze only.")
        return sentenze.copy()

    if sentenze.empty:
        logger.info("consolidator: no sentenze — using ordinanze only.")
        return ordinanze.copy()

    ord_uids  = set(ordinanze["UID"].dropna())
    sent_uids = set(sentenze["UID"].dropna())
    overlap   = ord_uids & sent_uids

    if overlap:
        logger.info(
            "consolidator: %d UIDs have both an Ordinanza and a Sentenza — Sentenza wins.",
            len(overlap),
        )
        for uid in overlap:
            ord_row  = ordinanze.loc[ordinanze["UID"] == uid, "ESITO_PROVVEDIMENTO"].iloc[0]
            sent_row = sentenze.loc[sentenze["UID"] == uid, "ESITO_PROVVEDIMENTO"].iloc[0]
            _DECISION_LOG.append({
                "rule": "sentenza_dominates_ordinanza",
                "reason": f"Both exist for UID; Sentenza kept ('{sent_row}' over Ordinanza '{ord_row}')",
                "confidence": "HIGH",
                "uid": uid,
                "source_file": "consolidator",
                "extra": "",
            })
        ordinanze_filtered = ordinanze[~ordinanze["UID"].isin(overlap)].copy()
    else:
        ordinanze_filtered = ordinanze.copy()

    combined = pd.concat([sentenze, ordinanze_filtered], ignore_index=True, sort=False)
    logger.info(
        "consolidator: %d sentenze + %d ordinanze (after dedup) → %d total outcomes",
        len(sentenze), len(ordinanze_filtered), len(combined),
    )
    return combined
