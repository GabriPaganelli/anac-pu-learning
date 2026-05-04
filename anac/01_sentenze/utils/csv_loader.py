"""
utils/csv_loader.py — Robust CSV reader for Italian open-data files.

Handles:
  - Encoding: tries UTF-8, then latin-1 (ISO-8859-1), then cp1252
  - Delimiter: sniffs between ";" and "," (Italian CSVs overwhelmingly use ";")
  - Column normalisation: strip, upper, replace spaces/hyphens with "_"
  - Column alias resolution from config.COLUMN_ALIASES
  - FULL LINE RECOVERY: rows with extra fields (unescaped delimiter inside a
    trailing text column) are repaired by re-joining excess fields into the
    last column — no row is silently dropped.
"""

import csv
import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
_DELIMITERS = [";", ",", "\t"]


def load_csv_robust(path: Path) -> pd.DataFrame:
    """
    Load a CSV file defensively and return a normalised DataFrame.

    Tries multiple encodings and delimiters. Within each combination, uses
    _read_csv_recovering() which never silently drops rows: malformed lines
    (more fields than the header) are repaired by merging excess fields back
    into the last column (common when a text field contains an unescaped
    delimiter).

    Returns an empty DataFrame (with a warning) if the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return pd.DataFrame()

    for enc in _ENCODINGS:
        for delim in _DELIMITERS:
            try:
                df = _read_csv_recovering(path, enc, delim)
                if df.empty:
                    continue
                if df.shape[1] <= 1 and delim != _DELIMITERS[-1]:
                    # Only one column → wrong delimiter, try next
                    continue
                df = _normalise_columns(df)
                logger.debug(
                    "Loaded %s  [enc=%s sep=%r rows=%d cols=%d]",
                    path.name, enc, delim, len(df), len(df.columns),
                )
                return df
            except Exception:
                continue

    logger.error("Could not parse CSV with any encoding/delimiter: %s", path)
    return pd.DataFrame()


def _read_csv_recovering(path: Path, enc: str, delim: str) -> pd.DataFrame:
    """
    Read a CSV using the csv module with full line recovery.

    For rows that have MORE fields than the header:
      - The excess fields are rejoined into the last column using the delimiter.
      - This correctly reconstructs trailing text fields that contain an
        unescaped delimiter (e.g. OGGETTO_RICORSO with a stray semicolon).

    For rows with FEWER fields than the header:
      - Missing trailing fields are padded with empty strings.

    No rows are silently dropped; all data is preserved.
    """
    with open(path, encoding=enc, errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()

        n_cols = len(header)
        rows: list[list[str]] = []
        n_recovered = 0
        n_padded = 0

        for row in reader:
            n = len(row)
            if n == n_cols:
                rows.append(row)
            elif n > n_cols:
                # Extra fields: rejoin excess into the last column.
                # This handles unescaped delimiters in trailing text fields.
                fixed = row[: n_cols - 1] + [delim.join(row[n_cols - 1 :])]
                rows.append(fixed)
                n_recovered += 1
            elif n > 0:
                # Too few fields: pad with empty strings
                rows.append(row + [""] * (n_cols - n))
                n_padded += 1
            # Completely empty lines are silently ignored

    if n_recovered:
        logger.info(
            "  CSV line recovery [%s]: %d malformed rows repaired "
            "(extra fields rejoined into last column)",
            path.name, n_recovered,
        )
    if n_padded:
        logger.debug(
            "  CSV line padding [%s]: %d short rows padded with empty fields",
            path.name, n_padded,
        )

    return pd.DataFrame(rows, columns=header)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names:
      1. strip whitespace
      2. upper-case
      3. replace runs of spaces, hyphens, dots with a single underscore
      4. apply alias mapping from config
    """
    import re

    def clean(name: str) -> str:
        name = str(name).strip().upper()
        name = re.sub(r"[\s\-\.]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    df.columns = [clean(c) for c in df.columns]
    df.rename(columns=config.COLUMN_ALIASES, inplace=True)
    return df


def load_and_concat(paths: list[Path], source_label: str = "") -> pd.DataFrame:
    """
    Load multiple CSV files and concatenate them.
    Adds a '_SOURCE_FILE' column for traceability.
    """
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = load_csv_robust(p)
        if df.empty:
            logger.warning("Skipping empty/unparseable file: %s", p)
            continue
        df["_SOURCE_FILE"] = p.name
        frames.append(df)

    if not frames:
        logger.warning("No data loaded for %s", source_label or "unknown source")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(
        "Concatenated %d files → %d rows  [%s]",
        len(frames), len(combined), source_label,
    )
    return combined
