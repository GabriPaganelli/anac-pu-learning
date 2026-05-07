"""
download_bdncp.py — Scarica i CSV BDNCP da dati.anticorruzione.it (CKAN).

Struttura dei dati ANAC:
  - Dataset singoli (aggiudicazioni, varianti, ecc.): un ZIP per dataset,
    contenente un CSV. Viene estratto e salvato con il nome canonico atteso
    dalla pipeline.
  - Bandi CIG annuali (cig-2008 … cig-2025): 12 ZIP mensili per anno.
    Vengono estratti e concatenati in un unico cig_YYYY.csv per anno.

Il WAF di dati.anticorruzione.it richiede header Referer e User-Agent
browser-like; senza di essi risponde con "Request Rejected".

Uso:
  python utils/download_bdncp.py [--no-cache] [--dry-run] [--since ANNO]

  --no-cache    Forza il re-download anche se il file di output esiste già.
  --dry-run     Stampa gli URL senza scaricare nulla.
  --since ANNO  Scarica i bandi-cig solo dall'anno ANNO in poi (default: 2008).

Output: data/raw/
"""

import argparse
import io
import re
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE    = Path(__file__).parent.parent
RAW_DIR = BASE / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CKAN_API        = "https://dati.anticorruzione.it/opendata/api/3/action"
DOWNLOAD_TIMEOUT = 600   # secondi — i file sono grandi (fino a 500 MB)
CHUNK_SIZE       = 65536
MAX_RETRIES      = 3
RETRY_SLEEP      = 20

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json,*/*",
    "Accept-Language": "it-IT,it;q=0.9",
    "Referer":         "https://dati.anticorruzione.it/",
})

# categorie-dpcm-aggregazione_csv.zip sul server ANAC è corrotto (stream DEFLATE
# troncato, EOCD assente). Il download fallisce gracefully.
SINGLE_DATASETS: list[tuple[str, str]] = [
    ("aggiudicazioni",                   "aggiudicazioni.csv"),
    ("aggiudicatari",                    "aggiudicatari.csv"),
    ("avvio-contratto",                  "avvio-contratto.csv"),
    ("varianti",                         "varianti.csv"),
    ("sospensioni",                      "sospensioni.csv"),
    ("stati-avanzamento",                "stati-avanzamento.csv"),
    ("collaudo",                         "collaudo.csv"),
    ("fine-contratto",                   "fine-contratto.csv"),
    ("lavorazioni",                      "lavorazioni.csv"),
    ("subappalti",                       "subappalti.csv"),
    ("partecipanti",                     "partecipanti.csv"),
    ("fonti-finanziamento",              "fonti-finanziamento.csv"),
    ("quadro-economico",                 "quadro-economico.csv"),
    ("stazioni-appaltanti",              "stazioni-appaltanti.csv"),
    ("categorie-opera",                  "categorie-opera.csv"),
    ("categorie-dpcm-aggregazione",      "categorie-dpcm-aggregazione.csv"),
    ("bandi-cig-tipo-scelta-contraente", "bandi-cig-tipo-scelta-contraente.csv"),
    ("bandi-cig-modalita-realizzazione", "bandi-cig-modalita-realizzazione.csv"),
]

_DATED_PREFIX = re.compile(r"^\d{8}-")


def ckan_get(action: str, **params) -> dict:
    url = f"{CKAN_API}/{action}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if not data.get("success"):
                raise ValueError(f"CKAN errore: {data.get('error')}")
            return data["result"]
        except Exception as exc:
            print(f"    [CKAN] tentativo {attempt}/{MAX_RETRIES} fallito: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
    raise RuntimeError(f"CKAN {action} irraggiungibile dopo {MAX_RETRIES} tentativi")


def get_csv_resources(package_id: str) -> list[dict]:
    """Risorse CSV di un package, escluse quelle con prefisso data e logCsv."""
    pkg = ckan_get("package_show", id=package_id)
    result = []
    for res in pkg.get("resources", []):
        fmt  = (res.get("format") or "").upper()
        name = res.get("name", "")
        url  = res.get("url", "")
        if fmt != "CSV" or not url:
            continue
        if _DATED_PREFIX.match(name) or name.endswith("_logCsv"):
            continue
        result.append(res)
    return result


def download_bytes(url: str, label: str) -> bytes | None:
    """Scarica un URL e restituisce i byte, con retry e verifica Content-Length."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"    [DOWN] {label} (tentativo {attempt})...")
            r = SESSION.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            r.raise_for_status()
            expected = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else None
            buf = io.BytesIO()
            for chunk in r.iter_content(CHUNK_SIZE):
                buf.write(chunk)
            data = buf.getvalue()
            if expected is not None and len(data) != expected:
                raise IOError(f"download troncato: {len(data)} / {expected} bytes")
            return data
        except Exception as exc:
            print(f"    [ERROR] tentativo {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
    print(f"    [FAIL] {label} — saltato dopo {MAX_RETRIES} tentativi")
    return None


def extract_csv_from_zip(data: bytes, label: str) -> bytes | None:
    """
    Estrae il primo CSV da un archivio ZIP in memoria.
    Fallback per streaming ZIP senza central directory (EOCD assente):
    legge direttamente il local file header e decomprime il raw DEFLATE stream.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            print(f"    [WARN] nessun CSV nel ZIP di {label}")
            return None
        if len(csv_names) > 1:
            print(f"    [WARN] {len(csv_names)} CSV nel ZIP di {label} — uso il primo")
        return zf.read(csv_names[0])
    except zipfile.BadZipFile:
        pass

    # Fallback per streaming ZIP senza EOCD: legge il local file header manualmente
    if not data[:4] == b"PK\x03\x04":
        print(f"    [ERROR] formato sconosciuto per {label}")
        return None
    try:
        fname_len  = struct.unpack_from("<H", data, 26)[0]
        extra_len  = struct.unpack_from("<H", data, 28)[0]
        method     = struct.unpack_from("<H", data,  8)[0]
        offset = 30 + fname_len + extra_len
        compressed = data[offset:]
        if method == 8:
            decompressed = zlib.decompress(compressed, -15)
        elif method == 0:
            decompressed = compressed
        else:
            print(f"    [ERROR] metodo di compressione {method} non supportato per {label}")
            return None
        print(f"    [INFO] streaming ZIP: decompresso via raw DEFLATE ({len(decompressed):,} bytes)")
        return decompressed
    except Exception as exc:
        print(f"    [ERROR] fallback DEFLATE fallito per {label}: {exc}")
        return None


def download_single(package_id: str, filename: str, cache: bool, dry_run: bool) -> bool:
    dest = RAW_DIR / filename
    if cache and dest.exists() and dest.stat().st_size > 0:
        print(f"  [CACHE] {filename}")
        return True

    try:
        resources = get_csv_resources(package_id)
    except RuntimeError as exc:
        print(f"  [SKIP] {package_id}: {exc}")
        return False

    if not resources:
        print(f"  [SKIP] {package_id}: nessuna risorsa CSV")
        return False

    res = resources[0]
    url = res["url"]

    if dry_run:
        print(f"  [DRY-RUN] {filename} <- {url}")
        return True

    print(f"  {filename}")
    raw = download_bytes(url, filename)
    if raw is None:
        return False

    csv_data = extract_csv_from_zip(raw, filename)
    if csv_data is None:
        return False

    dest.write_bytes(csv_data)
    size_mb = dest.stat().st_size / 1024**2
    print(f"    -> {dest.name} ({size_mb:.1f} MB)")
    return True


def _month_resources(package_id: str, year: int) -> list[dict]:
    """Risorse mensili CSV di un package cig-YYYY, ordinate per mese."""
    try:
        resources = get_csv_resources(package_id)
    except RuntimeError:
        return []
    month_pattern = re.compile(rf"cig_csv_{year}_(\d{{2}})$")
    monthly = []
    for res in resources:
        m = month_pattern.match(res.get("name", ""))
        if m:
            monthly.append((int(m.group(1)), res))
    monthly.sort(key=lambda x: x[0])
    return [res for _, res in monthly]


def download_cig_year(year: int, cache: bool, dry_run: bool) -> bool:
    dest = RAW_DIR / f"cig_{year}.csv"
    if cache and dest.exists() and dest.stat().st_size > 0:
        print(f"  [CACHE] cig_{year}.csv")
        return True

    package_id = f"cig-{year}"
    monthly = _month_resources(package_id, year)
    if not monthly:
        print(f"  [SKIP] {package_id}: nessuna risorsa mensile trovata")
        return False

    if dry_run:
        for res in monthly:
            print(f"  [DRY-RUN] cig_{year}.csv <- {res['url']}")
        return True

    print(f"  cig_{year}.csv ({len(monthly)} mesi)")
    chunks: list[bytes] = []
    header_written = False

    for res in monthly:
        name = res.get("name", "?")
        raw = download_bytes(res["url"], name)
        if raw is None:
            print(f"    [WARN] mese {name} saltato — file potrebbe essere incompleto")
            continue

        csv_data = extract_csv_from_zip(raw, name)
        if csv_data is None:
            continue

        lines = csv_data.splitlines(keepends=True)
        if not lines:
            continue

        if not header_written:
            chunks.extend(lines)
            header_written = True
        else:
            chunks.extend(lines[1:])  # salta header duplicato

    if not chunks:
        print(f"    [FAIL] nessun dato estratto per cig_{year}.csv")
        return False

    dest.write_bytes(b"".join(chunks))
    size_mb = dest.stat().st_size / 1024**2
    print(f"    -> cig_{year}.csv ({size_mb:.1f} MB)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scarica CSV BDNCP da dati.anticorruzione.it"
    )
    parser.add_argument("--no-cache", action="store_true", help="Forza re-download")
    parser.add_argument("--dry-run",  action="store_true", help="Stampa URL senza scaricare")
    parser.add_argument("--since",    type=int, default=2008, metavar="ANNO",
                        help="Scarica bandi-cig solo da ANNO in poi (default: 2008)")
    args = parser.parse_args()

    cache = not args.no_cache

    print("=" * 60)
    print("download_bdncp.py — ANAC BDNCP CSV downloader")
    print(f"  output : {RAW_DIR}")
    print(f"  cache  : {'on' if cache else 'off'}")
    print(f"  since  : {args.since}")
    print("=" * 60)

    failed: list[str] = []

    print("\n--- Bandi CIG annuali ---")
    all_packages = ckan_get("package_list")
    cig_years = sorted(
        int(m.group(1))
        for p in all_packages
        if (m := re.match(r"^cig-(\d{4})$", p))
        and int(m.group(1)) >= args.since
    )
    for year in cig_years:
        ok = download_cig_year(year, cache, args.dry_run)
        if not ok:
            failed.append(f"cig_{year}.csv")

    print("\n--- Dataset singoli ---")
    for package_id, filename in SINGLE_DATASETS:
        ok = download_single(package_id, filename, cache, args.dry_run)
        if not ok:
            failed.append(filename)

    print("\n" + "=" * 60)
    if failed:
        print(f"  ATTENZIONE: {len(failed)} file non scaricati:")
        for f in failed:
            print(f"    - {f}")
    else:
        print("  OK — tutti i file scaricati correttamente.")
    print(f"  Output: {RAW_DIR}")


if __name__ == "__main__":
    main()
