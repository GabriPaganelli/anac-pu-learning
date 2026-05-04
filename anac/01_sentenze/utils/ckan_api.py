"""
utils/ckan_api.py — CKAN API client for the OpenGA portal.

Discovers all dataset packages in a given group and extracts CSV resource URLs.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

import config

logger = logging.getLogger(__name__)


@dataclass
class ResourceInfo:
    package_id:   str
    package_name: str
    resource_id:  str
    url:          str
    year:         Optional[int]
    group:        str              # "ricorsi-pervenuti" | "ordinanze" | "sentenze"
    filename:     str = field(init=False)

    def __post_init__(self) -> None:
        self.filename = self.url.rstrip("/").split("/")[-1]


def _api_get(action: str, params: dict) -> dict:
    """Call one CKAN action and return the parsed JSON result."""
    url = f"{config.CKAN_API_URL}/{action}"
    try:
        resp = requests.get(url, params=params, timeout=config.DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise ValueError(f"CKAN API error for {action}: {data.get('error')}")
        return data["result"]
    except requests.RequestException as exc:
        logger.error("CKAN API request failed [%s %s]: %s", action, params, exc)
        raise


def discover_group_packages(group_id: str, name_filter: Optional[str] = None) -> list[dict]:
    """
    Return all packages in a CKAN group.

    Args:
        group_id:    CKAN group slug (e.g. "ricorsi-pervenuti").
        name_filter: If given, keep only packages whose name (lowercased)
                     contains this substring.  Used to skip aggregate-stats
                     datasets that lack CIG/case-level data.
    """
    logger.info("Discovering packages in group '%s' (filter=%r)…", group_id, name_filter)

    rows = 1000
    start = 0
    packages: list[dict] = []

    while True:
        result = _api_get(
            "package_search",
            {"fq": f"groups:{group_id}", "rows": rows, "start": start},
        )
        batch = result.get("results", [])
        packages.extend(batch)
        if len(packages) >= result.get("count", 0) or not batch:
            break
        start += rows

    logger.info("  → %d packages found in '%s'", len(packages), group_id)

    if name_filter:
        before = len(packages)
        packages = [p for p in packages if name_filter.lower() in p["name"].lower()]
        logger.info(
            "  → %d packages kept after name filter %r (%d dropped)",
            len(packages), name_filter, before - len(packages),
        )

    return packages


def get_csv_resources(package: dict, group: str) -> list[ResourceInfo]:
    """
    Extract all CSV resources from a package dict.
    Tries to parse the year from the resource URL or name.
    """
    resources: list[ResourceInfo] = []
    for res in package.get("resources", []):
        fmt = (res.get("format") or "").upper()
        if fmt != "CSV":
            continue

        url = res.get("url", "")
        if not url:
            continue

        # Skip CKAN datastore dump endpoints — they return 404 on this portal.
        # Only the /dataset/.../resource/.../download/ URLs work reliably.
        if "/datastore/dump/" in url:
            logger.debug("Skipping datastore dump URL: %s", url)
            continue

        year = _extract_year(res.get("name", "") + " " + url)
        resources.append(
            ResourceInfo(
                package_id=package["id"],
                package_name=package["name"],
                resource_id=res["id"],
                url=url,
                year=year,
                group=group,
            )
        )

    return resources


def discover_all_resources(
    groups: dict[str, Optional[str]],
) -> list[ResourceInfo]:
    """
    Discover all CSV resources across the given groups.

    Args:
        groups: mapping of group_id → name_filter (or None for no filter).

    Returns:
        Flat list of ResourceInfo objects.
    """
    all_resources: list[ResourceInfo] = []

    for group_id, name_filter in groups.items():
        packages = discover_group_packages(group_id, name_filter)
        for pkg in packages:
            resources = get_csv_resources(pkg, group_id)
            if not resources:
                logger.debug("No CSV resources in package '%s'", pkg["name"])
            all_resources.extend(resources)

    logger.info("Total CSV resources discovered: %d", len(all_resources))
    return all_resources


def _extract_year(text: str) -> Optional[int]:
    """Parse a 4-digit year (2000–2099) from a string."""
    import re
    m = re.search(r"\b(20[0-9]{2})\b", text)
    return int(m.group(1)) if m else None
