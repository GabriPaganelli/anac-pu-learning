"""
pipeline/deduplicator.py — Step 3: Resolve duplicate UIDs.

Rule: for the same UID, keep the row with the most recent DATA_PUBBLICAZIONE.
If DATA_PUBBLICAZIONE is null, keep the row with the highest NUMERO_PROVVEDIMENTO
(as a proxy for recency) and log the situation.
"""

import logging

import pandas as pd

import config
from pipeline.cleaner import _DECISION_LOG

logger = logging.getLogger(__name__)


def resolve_duplicates(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    Deduplicate by UID, keeping the most recent publication date.

    Args:
        df:     DataFrame with at least columns UID and DATA_PUBBLICAZIONE.
        source: Label for log messages (e.g. "ordinanze", "sentenze").

    Returns:
        DataFrame with one row per UID.
    """
    if df.empty:
        return df

    initial = len(df)
    dups = df.duplicated(subset="UID", keep=False)
    n_dup_uids = df.loc[dups, "UID"].nunique()

    if n_dup_uids == 0:
        logger.info("dedup [%s]: no duplicates found.", source)
        return df

    logger.info("dedup [%s]: %d UIDs have duplicates — resolving…", source, n_dup_uids)

    df = df.copy()

    # Parse publication date for sorting (optional column — ricorsi doesn't have it)
    has_pub_date = "DATA_PUBBLICAZIONE" in df.columns
    if has_pub_date:
        df["_PUB_DATE_PARSED"] = pd.to_datetime(
            df["DATA_PUBBLICAZIONE"], errors="coerce", dayfirst=True
        )
    else:
        df["_PUB_DATE_PARSED"] = pd.NaT

    unique_mask = ~dups
    dup_df = df[dups].copy()

    # Build sort keys: date desc, then NUMERO_PROVVEDIMENTO desc (if present)
    sort_cols = ["UID", "_PUB_DATE_PARSED"]
    sort_asc  = [True, False]
    if "NUMERO_PROVVEDIMENTO" in dup_df.columns:
        sort_cols.append("NUMERO_PROVVEDIMENTO")
        sort_asc.append(False)

    resolved = (
        dup_df
        .sort_values(by=sort_cols, ascending=sort_asc, na_position="last")
        .drop_duplicates(subset="UID", keep="first")
    )

    # Log UIDs where ALL dates were null (only meaningful if column exists)
    null_uids: list[str] = []
    if has_pub_date:
        null_date_uids = (
            dup_df.groupby("UID")["_PUB_DATE_PARSED"]
            .apply(lambda s: s.isna().all())
        )
        null_uids = null_date_uids[null_date_uids].index.tolist()
    for uid in null_uids:
        _DECISION_LOG.append({
            "rule": "dedup_null_date",
            "reason": f"All DATA_PUBBLICAZIONE null for UID; kept first row by NUMERO_PROVVEDIMENTO",
            "confidence": "MEDIUM",
            "uid": uid,
            "source_file": source,
            "extra": "",
        })
    if null_uids:
        logger.warning(
            "dedup [%s]: %d UIDs had no publication date; used NUMERO_PROVVEDIMENTO as tiebreaker",
            source, len(null_uids),
        )

    result = pd.concat([df[unique_mask], resolved], ignore_index=True)
    result = result.drop(columns=["_PUB_DATE_PARSED"], errors="ignore")

    logger.info(
        "dedup [%s]: %d → %d rows (removed %d duplicates)",
        source, initial, len(result), initial - len(result),
    )
    return result
