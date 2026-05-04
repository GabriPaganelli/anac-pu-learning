"""
pipeline/joiner.py — Step 5: Join consolidated judicial outcomes with Ricorsi.

Strategy:
  - Inner join on UID (strict matching).
  - Il join è solo su match esatto; gli esiti non matchati vengono scartati.
  - Logs join rate.
  - Separates CDS rows (to be processed by cds_dominator.py).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def join_outcomes_to_ricorsi(
    outcomes: pd.DataFrame,
    ricorsi: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Inner-join outcomes (ordinanze+sentenze) with ricorsi on UID.

    Returns:
        tar_joined:   TAR rows successfully joined (CIG attached).
        cds_joined:   CDS rows successfully joined (CIG attached).
        unmatched:    Outcome rows that did not find a matching ricorso UID.
    """
    if outcomes.empty or ricorsi.empty:
        logger.warning("joiner: one or both inputs are empty — nothing to join.")
        return pd.DataFrame(), pd.DataFrame(), outcomes.copy()

    n_outcomes = len(outcomes)
    n_ricorsi  = len(ricorsi)

    # Deduplicate ricorsi on UID before join (take first CIG per UID for the join key;
    # multiple CIG per UID are retained because ricorsi is already exploded)
    joined = outcomes.merge(
        ricorsi[["UID", "CODICE_CIG", "NUMERO_RICORSO", "NOME_SEDE",
                  "ANNO_DEPOSITO_RICORSO"]].copy(),
        on="UID",
        how="left",
        suffixes=("_OUT", "_RIC"),
    )

    matched_mask = joined["CODICE_CIG"].notna()
    matched   = joined[matched_mask].copy()
    unmatched = outcomes[~outcomes["UID"].isin(matched["UID"])].copy()

    match_rate = len(matched["UID"].unique()) / max(n_outcomes, 1) * 100
    logger.info(
        "joiner: %d outcomes × %d ricorsi → %d matched rows "
        "(%.1f%% of outcome UIDs matched)",
        n_outcomes, n_ricorsi, len(matched), match_rate,
    )

    if match_rate < 50:
        logger.warning(
            "joiner: match rate %.1f%% is below 50%% — check UID construction consistency!",
            match_rate,
        )

    is_cds = matched.get("_IS_CDS", pd.Series(False, index=matched.index))
    tar_joined = matched[~is_cds].copy()
    cds_joined = matched[is_cds].copy()

    logger.info(
        "joiner: %d TAR rows, %d CDS rows, %d unmatched",
        len(tar_joined), len(cds_joined), len(unmatched),
    )
    return tar_joined, cds_joined, unmatched
