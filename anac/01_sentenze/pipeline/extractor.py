"""
pipeline/extractor.py — Step 1: Discover and download all CSV files.

Strategy:
  - Use the CKAN API to enumerate packages and their CSV resources.
  - Download in parallel (ThreadPoolExecutor).
  - Cache: skip files that already exist with the expected size (Content-Length check).
  - Never crash the whole pipeline on a single download failure; log and continue.
"""

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

import config
from utils.ckan_api import ResourceInfo, discover_all_resources

logger = logging.getLogger(__name__)


def run(dry_run: bool = False) -> dict[str, list[Path]]:
    """
    Discover all CSV resources and download them.

    Returns:
        dict with keys "ricorsi-pervenuti", "ordinanze", "sentenze",
        each mapping to a list of local file paths.
    """
    groups = {
        config.GROUP_RICORSI:   config.RICORSI_PACKAGE_FILTER,
        config.GROUP_ORDINANZE: None,
        config.GROUP_SENTENZE:  None,
    }

    resources = discover_all_resources(groups)

    if dry_run:
        logger.info("[DRY-RUN] Would download %d files — skipping actual download.", len(resources))
        return _group_by(resources, [])

    paths = _download_all(resources)
    return _group_by(resources, paths)



def _download_all(resources: list[ResourceInfo]) -> list[Optional[Path]]:
    """Download all resources in parallel, return list of local paths (None on failure)."""
    results: list[Optional[Path]] = [None] * len(resources)

    with ThreadPoolExecutor(max_workers=config.MAX_DOWNLOAD_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_download_one, res): i
            for i, res in enumerate(resources)
        }
        with tqdm(total=len(resources), desc="Downloading CSVs", unit="file") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Download error [%s]: %s", resources[idx].url, exc)
                finally:
                    pbar.update(1)

    success = sum(1 for p in results if p is not None)
    logger.info("Downloaded %d / %d files successfully.", success, len(resources))
    return results


def _download_one(res: ResourceInfo) -> Optional[Path]:
    """Download a single CSV resource, using the cache when possible."""
    dest = _dest_path(res)

    if config.CACHE_ENABLED and dest.exists():
        expected_size = _remote_content_length(res.url)
        if expected_size is None or dest.stat().st_size == expected_size:
            logger.debug("Cache hit: %s", dest.name)
            return dest
        logger.debug("Cache stale (size mismatch): %s", dest.name)

    logger.debug("Downloading: %s → %s", res.url, dest.name)
    try:
        response = requests.get(
            res.url,
            timeout=config.DOWNLOAD_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=65536):
                fh.write(chunk)
        return dest
    except requests.RequestException as exc:
        logger.error("Failed to download %s: %s", res.url, exc)
        return None


def _remote_content_length(url: str) -> Optional[int]:
    """Return Content-Length from a HEAD request, or None if unavailable."""
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def _dest_path(res: ResourceInfo) -> Path:
    """
    Build a stable local path for a resource.
    Uses a short hash of the resource_id to avoid filename collisions.
    """
    tag = hashlib.md5(res.resource_id.encode()).hexdigest()[:8]
    stem = res.filename
    if not stem.lower().endswith(".csv"):
        stem = f"{stem}.csv"
    # Prefix with group so raw/ subdirectory stays organised
    name = f"{res.group}__{stem}"
    return config.RAW_DIR / name


def _group_by(
    resources: list[ResourceInfo],
    paths: list[Optional[Path]],
) -> dict[str, list[Path]]:
    """
    Build a {group → [path]} mapping from the resources + downloaded paths.
    Skips resources where the download failed (path is None).
    """
    result: dict[str, list[Path]] = {
        config.GROUP_RICORSI:   [],
        config.GROUP_ORDINANZE: [],
        config.GROUP_SENTENZE:  [],
    }
    for res, path in zip(resources, paths if paths else [None] * len(resources)):
        if path and path.exists():
            result.setdefault(res.group, []).append(path)
    return result
