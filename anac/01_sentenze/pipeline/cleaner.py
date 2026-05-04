"""
pipeline/cleaner.py — Step 2: Column selection, CIG cleaning, UID construction.

Three public functions:
  clean_ricorsi(df)   → DataFrame with one row per valid CIG
  clean_ordinanze(df) → DataFrame filtered to favourable outcomes
  clean_sentenze(df)  → DataFrame filtered to favourable outcomes

Each function builds a UID used for joining across datasets:
  UID = norm(NOME_SEDE) + "|" + ANNO_DEPOSITO_RICORSO + "|" + norm(NUMERO_RICORSO)

Decision log rows are appended to the module-level _DECISION_LOG list;
the caller (run_pipeline.py) persists it after the pipeline completes.
"""

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

# Module-level accumulator — flushed to disk by reporter.py
_DECISION_LOG: list[dict] = []


def clean_ricorsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean procurement-appeal rows.

    Returns a DataFrame with columns:
      UID, CODICE_CIG, NUMERO_RICORSO, NOME_SEDE, ANNO_DEPOSITO_RICORSO, _SOURCE_FILE
    One row per valid CIG (multi-CIG rows are exploded).
    """
    if df.empty:
        return df

    initial = len(df)

    needed = ["CODICE_CIG", "NUMERO_RICORSO", "NOME_SEDE", "ANNO_DEPOSITO_RICORSO"]
    df = _keep_columns(df, needed, source="ricorsi")

    df["NOME_SEDE"]            = df["NOME_SEDE"].apply(_norm_str)
    df["NUMERO_RICORSO"]       = df["NUMERO_RICORSO"].apply(_norm_str)
    df["ANNO_DEPOSITO_RICORSO"] = df["ANNO_DEPOSITO_RICORSO"].apply(_norm_str)

    df["CODICE_CIG"] = df["CODICE_CIG"].apply(_split_cigs)
    df = df.explode("CODICE_CIG", ignore_index=True)

    df["CODICE_CIG"] = df["CODICE_CIG"].apply(_clean_cig)

    invalid_mask = df["CODICE_CIG"].isna()
    if invalid_mask.any():
        _log_dropped(
            df[invalid_mask],
            rule="invalid_cig",
            reason="CIG is missing, placeholder, or wrong length",
        )
    df = df[~invalid_mask].copy()

    before_drop = len(df)
    df = df.dropna(subset=["NUMERO_RICORSO", "NOME_SEDE", "ANNO_DEPOSITO_RICORSO"])
    dropped_ids = before_drop - len(df)
    if dropped_ids:
        logger.warning("Ricorsi: dropped %d rows with missing NUMERO_RICORSO/NOME_SEDE/ANNO", dropped_ids)

    df["UID"] = _build_uid(df["NOME_SEDE"], df["ANNO_DEPOSITO_RICORSO"], df["NUMERO_RICORSO"])

    logger.info(
        "clean_ricorsi: %d rows in → %d rows out (%d dropped)",
        initial, len(df), initial - len(df),
    )
    return df[["UID", "CODICE_CIG", "NUMERO_RICORSO", "NOME_SEDE", "ANNO_DEPOSITO_RICORSO",
               "_SOURCE_FILE"] if "_SOURCE_FILE" in df.columns else
              ["UID", "CODICE_CIG", "NUMERO_RICORSO", "NOME_SEDE", "ANNO_DEPOSITO_RICORSO"]]


def clean_ordinanze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to favourable outcomes and build UIDs for joining.

    Returns a DataFrame with columns:
      UID, NUMERO_RICORSO, NOME_SEDE, ANNO_DEPOSITO_RICORSO, DATA_PUBBLICAZIONE,
      ESITO_PROVVEDIMENTO, _IS_CDS
    """
    return _clean_judicial(df, config.FAVORABLE_ORDINANZE, "ordinanze")


def clean_sentenze(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to favourable outcomes and build UIDs for joining.
    Same column contract as clean_ordinanze, plus _TIPO="sentenza".
    """
    return _clean_judicial(df, config.FAVORABLE_SENTENZE, "sentenze")


def get_decision_log() -> pd.DataFrame:
    """Return accumulated decision-log rows as a DataFrame."""
    return pd.DataFrame(_DECISION_LOG) if _DECISION_LOG else pd.DataFrame(
        columns=["rule", "reason", "confidence", "uid", "source_file", "extra"]
    )


def _clean_judicial(df: pd.DataFrame, favorable_set: set[str], source: str) -> pd.DataFrame:
    """Common logic for Ordinanze and Sentenze."""
    if df.empty:
        return df

    initial = len(df)

    needed = [
        "NUMERO_RICORSO", "NOME_SEDE", "DATA_PUBBLICAZIONE",
        "ESITO_PROVVEDIMENTO",
        # Year can come from deposit date or publication year
        "DATA_DEPOSITO_RICORSO", "ANNO_DEPOSITO_RICORSO", "ANNO_PUBBLICAZIONE",
    ]
    df = _keep_columns(df, needed, source=source)
    df = _derive_anno_deposito(df, source)

    df["NOME_SEDE"]      = df["NOME_SEDE"].apply(_norm_str)
    df["NUMERO_RICORSO"] = df["NUMERO_RICORSO"].apply(_norm_str)
    df["ESITO_NORM"] = df["ESITO_PROVVEDIMENTO"].apply(_norm_esito)

    mask_favorable = df["ESITO_NORM"].isin(favorable_set)
    dropped = (~mask_favorable).sum()
    df = df[mask_favorable].copy()
    logger.info(
        "clean_%s: %d rows in → %d favorable → %d dropped",
        source, initial, len(df), dropped,
    )

    df["_IS_CDS"] = df["NOME_SEDE"].apply(_is_cds)
    df["_TIPO"] = source.rstrip("e")   # "ordinanz" / "sentenz"

    df = df.dropna(subset=["NUMERO_RICORSO", "NOME_SEDE", "ANNO_DEPOSITO_RICORSO"])
    df["UID"] = _build_uid(df["NOME_SEDE"], df["ANNO_DEPOSITO_RICORSO"], df["NUMERO_RICORSO"])

    out_cols = [
        "UID", "NUMERO_RICORSO", "NOME_SEDE", "ANNO_DEPOSITO_RICORSO",
        "DATA_PUBBLICAZIONE", "ESITO_PROVVEDIMENTO", "ESITO_NORM",
        "_IS_CDS", "_TIPO",
    ]
    if "_SOURCE_FILE" in df.columns:
        out_cols.append("_SOURCE_FILE")
    return df[[c for c in out_cols if c in df.columns]]


def _derive_anno_deposito(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Ensure ANNO_DEPOSITO_RICORSO is populated.

    Priority:
      1. Column already exists and is non-null.
      2. Extract year from DATA_DEPOSITO_RICORSO.
      3. Fall back to ANNO_PUBBLICAZIONE (with a warning logged).
    """
    if "ANNO_DEPOSITO_RICORSO" not in df.columns:
        df["ANNO_DEPOSITO_RICORSO"] = pd.NA

    if "DATA_DEPOSITO_RICORSO" in df.columns:
        extracted = df["DATA_DEPOSITO_RICORSO"].apply(_year_from_date)
        missing = df["ANNO_DEPOSITO_RICORSO"].isna()
        df.loc[missing, "ANNO_DEPOSITO_RICORSO"] = extracted[missing]

    if "ANNO_PUBBLICAZIONE" in df.columns:
        still_missing = df["ANNO_DEPOSITO_RICORSO"].isna()
        n = still_missing.sum()
        if n:
            logger.warning(
                "%s: falling back to ANNO_PUBBLICAZIONE for %d rows "
                "(DATA_DEPOSITO_RICORSO missing) — UID may mismatch",
                source, n,
            )
            df.loc[still_missing, "ANNO_DEPOSITO_RICORSO"] = (
                df.loc[still_missing, "ANNO_PUBBLICAZIONE"]
            )

    df["ANNO_DEPOSITO_RICORSO"] = df["ANNO_DEPOSITO_RICORSO"].apply(_norm_str)
    return df


def _keep_columns(df: pd.DataFrame, wanted: list[str], source: str) -> pd.DataFrame:
    """
    Keep only the requested columns (ignoring those that are absent).
    Logs missing columns as a warning.
    """
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        logger.debug("%s: columns not found (will be NaN): %s", source, missing)
    extra = [c for c in df.columns if c not in wanted and c.startswith("_")]
    return df[present + extra].copy()


def _norm_str(val) -> Optional[str]:
    """Strip, upper-case, collapse whitespace. Return None for empty/null values."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def _norm_esito(val) -> str:
    """
    Normalise ESITO_PROVVEDIMENTO for set-membership tests:
    strip, lower, collapse spaces, remove accents, remove apostrophes/hyphens.
    """
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    # Remove accents (è → e, à → a, …)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Replace apostrophes, hyphens with space
    s = re.sub(r"['\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_cig(val) -> Optional[str]:
    """
    Clean a single CIG value.
    Returns None if the value is a placeholder or fails length validation.
    """
    if pd.isna(val):
        return None
    cig = str(val).strip().upper()
    if not cig:
        return None
    if cig in config.CIG_PLACEHOLDER_VALUES:
        return None
    if not cig.isalnum():
        return None
    if len(cig) != config.CIG_EXPECTED_LENGTH:
        # partial codes do appear; discard silently
        return None
    return cig


def _split_cigs(val) -> list[str]:
    """
    Split a cell that may contain multiple CIG codes into a list.
    Preserves a single-element list for the common case.
    """
    if pd.isna(val):
        return [None]
    raw = str(val).strip()
    for sep in config.CIG_SEPARATORS:
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep)]
            return [p for p in parts if p] or [None]
    return [raw]


def _build_uid(nome_sede: pd.Series, anno: pd.Series, numero: pd.Series) -> pd.Series:
    """Build composite UID from three series."""
    return nome_sede.fillna("") + "|" + anno.fillna("") + "|" + numero.fillna("")


def _year_from_date(val) -> Optional[str]:
    """Extract 4-digit year string from a date-like string."""
    if pd.isna(val):
        return None
    m = re.search(r"\b(20[0-9]{2})\b", str(val))
    return m.group(1) if m else None


def _is_cds(nome_sede: Optional[str]) -> bool:
    """Return True if this row belongs to the Consiglio di Stato."""
    if not nome_sede:
        return False
    ns = nome_sede.upper()
    # Match "CDS GIURISDIZIONALE - ROMA", "CDS", "CONSIGLIO DI STATO ..."
    return "CONSIGLIO DI STATO" in ns or ns == "CDS" or ns.startswith("CDS ")


def _log_dropped(df: pd.DataFrame, rule: str, reason: str) -> None:
    """Append rows to the decision log."""
    for _, row in df.iterrows():
        _DECISION_LOG.append({
            "rule": rule,
            "reason": reason,
            "confidence": "HIGH",
            "uid": row.get("UID", ""),
            "source_file": row.get("_SOURCE_FILE", ""),
            "extra": str(row.get("CODICE_CIG", "")),
        })
