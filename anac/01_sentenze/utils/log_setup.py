"""
utils/log_setup.py — Centralised logging configuration.
Call configure_logging() once at pipeline startup.
"""

import logging
import sys
from pathlib import Path

import config


def configure_logging(level: str = "INFO") -> None:
    """Set up root logger with file + console handlers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Remove any handlers already attached (idempotent on re-import)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    # Console — force UTF-8 on Windows to avoid cp1252 encoding errors
    import io
    stdout_utf8 = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    ch = logging.StreamHandler(stdout_utf8)
    ch.setLevel(numeric_level)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(config.LOG_PIPELINE, encoding="utf-8", mode="a")
    fh.setLevel(logging.DEBUG)   # always verbose in file
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
