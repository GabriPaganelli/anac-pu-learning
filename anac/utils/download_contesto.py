"""
download_contesto.py — Scarica e costruisce il pannello provinciale annuale (2007-2024).

Fonti:
  - Reddito IRPEF pro capite: ZIP MEF per comune, aggregato a livello provinciale
  - Tasso di disoccupazione:  sdmx.istat.it (flow 151_1193 per 2007-2020, 151_914 per 2021-2024)
  - Tasso omicidi per 100k:  ZIP ISTAT BES 2025 (indicatore 07SIC001P)

Patch applicate dopo il download:
  1. Rimozione ex-province sarde (104-107): abolite nel 2016, non nella geografia attuale
  2. BZ (21) + TN (22) disoccupazione: download da codici NUTS2 ITD1/ITD2
  3. Sud Sardegna (111) pre-2016 + omicidi: proxy da Cagliari (92)
  4. Backward fill omicidi per anni iniziali mancanti (max 10 anni)

Output: data/territorial/contesto_province.csv
  Colonne: codice_provincia, nome_provincia, anno,
           tasso_disoccupazione, reddito_irpef_procapite, tasso_omicidi_100k
"""

import io
import re
import sys
import time
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE    = Path(__file__).parent.parent
OUTFILE = BASE / "data" / "territorial" / "contesto_province.csv"
OUTFILE.parent.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2007, 2025))

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 (research)"


def fetch(url, label, timeout=300, retries=3, sleep=10):
    for attempt in range(1, retries + 1):
        try:
            print(f"  Fetching [{label}] attempt {attempt}: {url[:90]}...")
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            print(f"  ERROR attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(sleep)
    print(f"  FAILED [{label}] after {retries} attempts")
    return None


NUTS3_TO_ISTAT = {
    # Piemonte
    "ITC11": (1, "Torino"), "ITC12": (2, "Vercelli"), "ITC13": (96, "Biella"),
    "ITC14": (103, "Verbano-Cusio-Ossola"), "ITC15": (3, "Novara"),
    "ITC16": (4, "Cuneo"), "ITC17": (5, "Asti"), "ITC18": (6, "Alessandria"),
    # Valle d'Aosta
    "ITC20": (7, "Aosta"),
    # Liguria
    "ITC31": (8, "Imperia"), "ITC32": (9, "Savona"),
    "ITC33": (10, "Genova"), "ITC34": (11, "La Spezia"),
    # Lombardia
    "ITC41": (12, "Varese"), "ITC42": (13, "Como"), "ITC43": (97, "Lecco"),
    "ITC44": (14, "Sondrio"), "ITC45": (16, "Bergamo"), "ITC46": (15, "Milano"),
    "ITC47": (17, "Brescia"), "ITC48": (18, "Pavia"), "ITC49": (19, "Cremona"),
    "ITC4A": (20, "Mantova"), "ITC4B": (98, "Lodi"),
    "IT108": (108, "Monza e della Brianza"),
    # Trentino-Alto Adige non presente nei dati NUTS3 disoccupazione — gestito nel patch
    # Veneto
    "ITD31": (23, "Verona"), "ITD32": (24, "Vicenza"), "ITD33": (25, "Belluno"),
    "ITD34": (26, "Treviso"), "ITD35": (27, "Venezia"), "ITD36": (28, "Padova"),
    "ITD37": (29, "Rovigo"),
    # Friuli-Venezia Giulia
    "ITD41": (93, "Pordenone"), "ITD42": (30, "Udine"),
    "ITD43": (31, "Gorizia"), "ITD44": (32, "Trieste"),
    # Emilia-Romagna
    "ITD51": (33, "Piacenza"), "ITD52": (34, "Parma"),
    "ITD53": (35, "Reggio nell'Emilia"), "ITD54": (36, "Modena"),
    "ITD55": (37, "Bologna"), "ITD56": (38, "Ferrara"),
    "ITD57": (39, "Ravenna"), "ITD58": (40, "Forli-Cesena"), "ITD59": (99, "Rimini"),
    # Toscana
    "ITE11": (45, "Massa-Carrara"), "ITE12": (46, "Lucca"), "ITE13": (47, "Pistoia"),
    "ITE14": (48, "Firenze"), "ITE15": (49, "Livorno"), "ITE16": (50, "Pisa"),
    "ITE17": (51, "Arezzo"), "ITE18": (52, "Siena"), "ITE19": (53, "Grosseto"),
    "ITE1A": (100, "Prato"),
    # Umbria
    "ITE21": (54, "Perugia"), "ITE22": (55, "Terni"),
    # Marche
    "ITE31": (41, "Pesaro e Urbino"), "ITE32": (42, "Ancona"),
    "ITE33": (43, "Macerata"), "ITE34": (44, "Ascoli Piceno"),
    "IT109": (109, "Fermo"),
    # Lazio
    "ITE41": (56, "Viterbo"), "ITE42": (57, "Rieti"), "ITE43": (58, "Roma"),
    "ITE44": (59, "Latina"), "ITE45": (60, "Frosinone"),
    # Abruzzo
    "ITF11": (66, "L'Aquila"), "ITF12": (67, "Teramo"),
    "ITF13": (68, "Pescara"), "ITF14": (69, "Chieti"),
    # Molise
    "ITF21": (70, "Campobasso"), "ITF22": (94, "Isernia"),
    # Campania
    "ITF31": (61, "Caserta"), "ITF32": (62, "Benevento"),
    "ITF33": (63, "Napoli"), "ITF34": (64, "Avellino"), "ITF35": (65, "Salerno"),
    # Puglia
    "ITF41": (71, "Foggia"), "ITF42": (72, "Bari"),
    "ITF43": (73, "Taranto"), "ITF44": (74, "Brindisi"), "ITF45": (75, "Lecce"),
    "IT110": (110, "Barletta-Andria-Trani"),
    # Basilicata
    "ITF51": (76, "Potenza"), "ITF52": (77, "Matera"),
    # Calabria
    "ITF61": (78, "Cosenza"), "ITF62": (101, "Crotone"),
    "ITF63": (79, "Catanzaro"), "ITF64": (102, "Vibo Valentia"),
    "ITF65": (80, "Reggio di Calabria"),
    # Sicilia
    "ITG11": (81, "Trapani"), "ITG12": (82, "Palermo"), "ITG13": (83, "Messina"),
    "ITG14": (84, "Agrigento"), "ITG15": (85, "Caltanissetta"),
    "ITG16": (86, "Enna"), "ITG17": (87, "Catania"),
    "ITG18": (88, "Ragusa"), "ITG19": (89, "Siracusa"),
    # Sardegna
    "ITG25": (90, "Sassari"), "ITG26": (91, "Nuoro"),
    "ITG27": (92, "Cagliari"), "ITG28": (95, "Oristano"),
    "IT111": (111, "Sud Sardegna"),
}

# Provincia madre per le nuove province create nel 2009 (assente nei dati 2007-2008)
PROVINCIA_MADRE = {108: 15, 109: 44, 110: 72}  # MB->MI, FM->AP, BT->BA

ISTAT_TO_NOME = {v[0]: v[1] for v in NUTS3_TO_ISTAT.values()}
ISTAT_TO_NOME[21] = "Bolzano/Bozen"
ISTAT_TO_NOME[22] = "Trento"

BES_NAME_TO_ISTAT = {v[1].lower(): v[0] for v in NUTS3_TO_ISTAT.values()}
BES_NAME_TO_ISTAT.update({
    "valle d'aosta/vallee d'aoste": 7, "valle d'aosta": 7, "aosta": 7,
    "bolzano/bozen": 21, "bolzano-bozen": 21, "bolzano": 21,
    "trento": 22,
    "reggio di calabria": 80, "reggio calabria": 80,
    "reggio nell'emilia": 35, "reggio emilia": 35,
    "pesaro e urbino": 41, "pesaro-urbino": 41,
    "forli'-cesena": 40, "forli-cesena": 40, "forli cesena": 40,
    "forlì-cesena": 40,
    "massa-carrara": 45, "massa carrara": 45,
    "la spezia": 11,
    "verbano-cusio-ossola": 103,
    "monza e della brianza": 108, "monza e brianza": 108, "monza": 108,
    "barletta-andria-trani": 110,
    "sud sardegna": 111,
    "carbonia-iglesias": 107,
    "medio campidano": 106,
    "ogliastra": 105,
    "olbia-tempio": 104,
})


print("PARTE 1: Reddito IRPEF pro capite (MEF)")

MEF_BASE = (
    "https://www1.finanze.gov.it/finanze/analisi_stat/public/"
    "v_4_0_0/contenuti/"
    "Redditi_e_principali_variabili_IRPEF_su_base_comunale_CSV_{year}.zip"
)

REDDITO_COLS_PATTERN = re.compile(r"reddito complessivo.*?ammontare", re.I)

irpef_rows = []
for year in YEARS:
    mef_year = min(year, 2023)  # MEF disponibile fino al 2023; il 2024 userà il 2023 via ffill
    label = f"IRPEF {year} (mef={mef_year})"
    url = MEF_BASE.format(year=mef_year)
    raw = fetch(url, label, timeout=120, sleep=5)
    if raw is None:
        print(f"  Saltato IRPEF {year}")
        continue

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        raw_csv = zf.read(csv_name)
    for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw_csv), sep=";", dtype=str,
                             low_memory=False, encoding=enc, index_col=False)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        print(f"  WARNING: impossibile decodificare il CSV per {year}, saltato")
        continue

    istat_col = next(
        (c for c in df.columns if "codice istat" in c.lower()),
        None
    )
    if istat_col is None:
        print(f"  WARNING: colonna codice ISTAT assente per {year}, saltata")
        continue
    df["codice_provincia"] = pd.to_numeric(
        df[istat_col].str.strip().str.zfill(6).str[:3],
        errors="coerce"
    )

    df["n_contrib"] = pd.to_numeric(
        df["Numero contribuenti"]
        .str.replace(r"\.", "", regex=True)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Totale diretto se disponibile (2023+), altrimenti somma fasce di reddito
    direct_col = next(
        (c for c in df.columns
         if re.match(r"^reddito complessivo\s*-\s*ammontare", c.strip(), re.I)
         and "euro" in c.lower()
         and "da " not in c.lower() and "oltre" not in c.lower()
         and "minore" not in c.lower()),
        None,
    )
    if direct_col:
        df["reddito_tot"] = pd.to_numeric(
            df[direct_col]
            .str.replace(r"\.", "", regex=True)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
    else:
        reddito_cols = [c for c in df.columns if REDDITO_COLS_PATTERN.match(c.strip())]
        for col in reddito_cols:
            df[col] = pd.to_numeric(
                df[col]
                .str.replace(r"\.", "", regex=True)
                .str.replace(",", ".", regex=False),
                errors="coerce",
            )
        df["reddito_tot"] = df[reddito_cols].sum(axis=1, min_count=1)

    prov = (
        df.groupby("codice_provincia")
        .agg(n_contrib=("n_contrib", "sum"), reddito_tot=("reddito_tot", "sum"))
        .reset_index()
    )
    prov = prov[(prov["codice_provincia"] > 0) & (prov["n_contrib"] > 0)].copy()
    prov["reddito_irpef_procapite"] = prov["reddito_tot"] / prov["n_contrib"]
    prov["anno"] = year

    irpef_rows.append(prov[["codice_provincia", "anno", "reddito_irpef_procapite"]])
    print(
        f"  IRPEF {year}: {len(prov)} province, procapite "
        f"[{prov['reddito_irpef_procapite'].min():.0f} - "
        f"{prov['reddito_irpef_procapite'].max():.0f}]"
    )

df_irpef = pd.concat(irpef_rows, ignore_index=True) if irpef_rows else pd.DataFrame()
print(f"Totale righe IRPEF: {len(df_irpef)}")


print("\nPARTE 2: Tasso di disoccupazione (sdmx.istat.it)")

SDMX_BASE = (
    "https://sdmx.istat.it/SDMXWS/rest/data/{flow}"
    "?format=csv&startPeriod={y1}&endPeriod={y2}"
)


def download_disoccupazione(flow, start_year, end_year, eta_class):
    url = SDMX_BASE.format(flow=flow, y1=start_year, y2=end_year)
    raw = fetch(url, f"disoccupazione {flow} {start_year}-{end_year}", timeout=180, sleep=15)
    if raw is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
    except Exception as e:
        print(f"  Errore parsing CSV: {e}")
        return pd.DataFrame()
    if df.empty or "SESSO" not in df.columns:
        print(f"  Risposta vuota o inattesa per {flow}")
        return pd.DataFrame()

    mask = (
        (df["SESSO"] == "9")
        & (df["TITOLO_STUDIO"] == "99")
        & (df["DURATA_DISOCCUPAZ"] == "TOTAL")
        & (df["CLASSE_ETA"] == eta_class)
    )
    df = df[mask].copy()
    df = df[df["ITTER107"].str.startswith("IT") & (df["ITTER107"].str.len() == 5)].copy()
    df["codice_provincia"] = df["ITTER107"].map(
        lambda x: NUTS3_TO_ISTAT.get(x, (None,))[0]
    )
    df = df[df["codice_provincia"].notna()].copy()
    df["codice_provincia"] = df["codice_provincia"].astype(int)
    df["anno"] = pd.to_numeric(df["TIME_PERIOD"])
    df["tasso_disoccupazione"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    return df[["codice_provincia", "anno", "tasso_disoccupazione"]]


print("  Download 2007-2020 (flow 151_1193, classe età Y_GE15)...")
df_disco_old = download_disoccupazione("151_1193", 2007, 2020, "Y_GE15")
print(
    f"  Righe: {len(df_disco_old)}, "
    f"anni: {sorted(df_disco_old['anno'].unique()) if len(df_disco_old) else []}"
)

print("  Download 2020-2024 (flow 151_914, classe età Y15-74)...")
df_disco_new = download_disoccupazione("151_914", 2020, 2024, "Y15-74")
print(
    f"  Righe: {len(df_disco_new)}, "
    f"anni: {sorted(df_disco_new['anno'].unique()) if len(df_disco_new) else []}"
)

# Per il 2020 (anno di sovrapposizione) si usa il nuovo flow
df_disco = pd.concat(
    [df_disco_old[df_disco_old["anno"] < 2020], df_disco_new],
    ignore_index=True,
)
df_disco = df_disco.drop_duplicates(["codice_provincia", "anno"])
print(
    f"Disoccupazione totale: {len(df_disco)} righe, "
    f"{df_disco['codice_provincia'].nunique()} province"
)


print("\nPARTE 3: Tasso omicidi per 100k (ISTAT BES 2025)")

BES_URL = (
    "https://www.istat.it/wp-content/uploads/2025/06/"
    "Bes_dei_territori_edizione_2025.zip"
)
bes_raw = fetch(BES_URL, "BES 2025", timeout=120, sleep=10)

omicidi_rows = []
if bes_raw:
    with zipfile.ZipFile(io.BytesIO(bes_raw)) as zf:
        xlsx_name = [
            n for n in zf.namelist()
            if n.endswith(".xlsx") and "Indicatori_per_provincia" in n
        ][0]
        wb = openpyxl.load_workbook(zf.open(xlsx_name), read_only=True, data_only=True)
        ws = wb.active
        all_rows = list(ws.iter_rows(values_only=True))

    header = list(all_rows[0])
    year_cols = {col: idx for idx, col in enumerate(header) if col and str(col).startswith("V")}
    terr_idx  = header.index("TERRITORIO")
    codice_idx = header.index("CODICE")
    sesso_idx = header.index("SESSO")
    w_geo_idx = header.index("W_GEO") if "W_GEO" in header else None

    unmapped = set()
    for row in all_rows[1:]:
        if not row or row[codice_idx] != "07SIC001P":
            continue
        if row[sesso_idx] != "Totale":
            continue
        if w_geo_idx is not None:
            geo = str(row[w_geo_idx] or "")
            if geo.endswith("-000") or geo == "":
                continue

        nome = str(row[terr_idx] or "").strip()
        nome_key = (
            nome.lower()
            .replace("’", "'").replace("‘", "'")
            .replace("à", "a").replace("è", "e")
            .replace("é", "e").replace("ì", "i")
            .replace("ò", "o").replace("ù", "u")
        )
        istat_code = BES_NAME_TO_ISTAT.get(nome_key) or BES_NAME_TO_ISTAT.get(nome.lower())
        if istat_code is None:
            unmapped.add(nome)
            continue

        for vcol, vidx in year_cols.items():
            year = int(vcol[1:])
            if year < 2007 or year > 2024:
                continue
            val = row[vidx]
            if val is None:
                continue
            try:
                tasso = float(str(val).replace(",", ".").strip())
            except ValueError:
                continue
            omicidi_rows.append(
                {"codice_provincia": istat_code, "anno": year, "tasso_omicidi_100k": tasso}
            )

    if unmapped:
        print(f"  WARNING: province BES non mappate ({len(unmapped)}): {sorted(unmapped)}")

df_omicidi = pd.DataFrame(omicidi_rows)
print(
    f"Omicidi righe: {len(df_omicidi)}, "
    f"{df_omicidi['codice_provincia'].nunique() if len(df_omicidi) else 0} province"
)


print("\nPARTE 4: Costruzione pannello province x anni")

all_prov_codes = sorted(
    set(
        list(df_irpef["codice_provincia"].dropna().astype(int)) if len(df_irpef) else []
        + list(df_disco["codice_provincia"].dropna().astype(int)) if len(df_disco) else []
        + list(df_omicidi["codice_provincia"].dropna().astype(int)) if len(df_omicidi) else []
    )
)
print(f"Province trovate tra le fonti: {len(all_prov_codes)}")

panel = pd.DataFrame(
    [(p, y) for p in all_prov_codes for y in YEARS],
    columns=["codice_provincia", "anno"],
)
panel["nome_provincia"] = panel["codice_provincia"].map(
    lambda c: ISTAT_TO_NOME.get(c, f"Provincia_{c}")
)

if len(df_irpef):
    panel = panel.merge(
        df_irpef[["codice_provincia", "anno", "reddito_irpef_procapite"]],
        on=["codice_provincia", "anno"], how="left",
    )
else:
    panel["reddito_irpef_procapite"] = float("nan")

if len(df_disco):
    panel = panel.merge(
        df_disco[["codice_provincia", "anno", "tasso_disoccupazione"]],
        on=["codice_provincia", "anno"], how="left",
    )
else:
    panel["tasso_disoccupazione"] = float("nan")

if len(df_omicidi):
    panel = panel.merge(
        df_omicidi[["codice_provincia", "anno", "tasso_omicidi_100k"]],
        on=["codice_provincia", "anno"], how="left",
    )
else:
    panel["tasso_omicidi_100k"] = float("nan")

# Proxy per le province nate nel 2009 (MB, FM, BT): usa la provincia madre per 2007-2008
for new_code, old_code in PROVINCIA_MADRE.items():
    for col in ["tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k"]:
        mask_new = (panel["codice_provincia"] == new_code) & (panel["anno"] <= 2008)
        if not mask_new.any():
            continue
        for y in [2007, 2008]:
            old_val_rows = panel[
                (panel["codice_provincia"] == old_code) & (panel["anno"] == y)
            ][col]
            if len(old_val_rows) and not old_val_rows.isna().all():
                panel.loc[mask_new & (panel["anno"] == y), col] = old_val_rows.iloc[0]

# Forward-fill per provincia (no backward fill in questa fase)
panel = panel.sort_values(["codice_provincia", "anno"])
for col in ["tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k"]:
    panel[col] = panel.groupby("codice_provincia")[col].transform(lambda x: x.ffill())

panel = panel[[
    "codice_provincia", "nome_provincia", "anno",
    "tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k",
]]

print(f"\nPannello grezzo: {panel.shape}")
for col in ["tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k"]:
    n_null = panel[col].isna().sum()
    print(f"  {col}: {n_null} NaN ({n_null / len(panel) * 100:.1f}%)")


print("\nPARTE 5: Patch valori mancanti")


def fetch_sdmx(flow, start, end, timeout=120):
    url = (f"https://sdmx.istat.it/SDMXWS/rest/data/{flow}"
           f"?format=csv&startPeriod={start}&endPeriod={end}")
    for attempt in range(1, 4):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), dtype=str)
        except Exception as e:
            print(f"  Tentativo {attempt} fallito: {e}")
            if attempt < 3:
                time.sleep(8)
    return pd.DataFrame()


# Patch 1: Rimozione ex-province sarde (104-107) — abolite nel 2016
# I comuni di queste province compaiono nei dati MEF ma non nella geografia attuale.
old_sarde = [104, 105, 106, 107]
n_before = len(panel)
panel = panel[~panel["codice_provincia"].isin(old_sarde)].reset_index(drop=True)
print(f"\nPatch 1: Rimosse ex-province sarde (104-107): {n_before - len(panel)} righe eliminate")

# Patch 2: BZ (21) + TN (22) disoccupazione da codici NUTS2
# I dati NUTS3 non includono ITD1/ITD2, scaricati separatamente.
NUTS2_TO_ISTAT = {"ITD1": 21, "ITD2": 22}

print("\nPatch 2: Disoccupazione BZ/TN da NUTS2...")
bz_tn_rows = []

for flow, eta, y1, y2 in [
    ("151_1193", "Y_GE15", 2007, 2019),
    ("151_914",  "Y15-74", 2020, 2024),
]:
    raw = fetch_sdmx(flow, y1, y2)
    if raw.empty:
        print(f"  WARNING: nessun dato da {flow}")
        continue
    mask = (
        raw["ITTER107"].isin(["ITD1", "ITD2"])
        & (raw["SESSO"] == "9")
        & (raw["TITOLO_STUDIO"] == "99")
        & (raw["DURATA_DISOCCUPAZ"] == "TOTAL")
        & (raw["CLASSE_ETA"] == eta)
        & (~raw["TIME_PERIOD"].str.contains("Q", na=False))
    )
    sub = raw[mask][["ITTER107", "TIME_PERIOD", "OBS_VALUE"]].copy()
    sub["codice_provincia"] = sub["ITTER107"].map(NUTS2_TO_ISTAT)
    sub["anno"] = pd.to_numeric(sub["TIME_PERIOD"])
    sub["tasso_disoccupazione"] = pd.to_numeric(sub["OBS_VALUE"], errors="coerce")
    bz_tn_rows.append(sub[["codice_provincia", "anno", "tasso_disoccupazione"]])
    print(f"  {flow}: {len(sub)} righe annuali per BZ/TN")

if bz_tn_rows:
    bz_tn = pd.concat(bz_tn_rows).drop_duplicates(["codice_provincia", "anno"])
    patched = 0
    for _, row in bz_tn.iterrows():
        idx = panel.index[
            (panel["codice_provincia"] == row["codice_provincia"]) &
            (panel["anno"] == row["anno"])
        ]
        if len(idx) and pd.isna(panel.loc[idx[0], "tasso_disoccupazione"]):
            panel.loc[idx[0], "tasso_disoccupazione"] = row["tasso_disoccupazione"]
            patched += 1
    print(f"  Patchate {patched} celle per BZ/TN disoccupazione")

# Patch 3: Sud Sardegna (111) pre-2016 + omicidi → proxy da Cagliari (92)
# Sud Sardegna istituita nel 2016; per gli anni precedenti si usa Cagliari come proxy.
print("\nPatch 3: Sud Sardegna (111) → proxy Cagliari (92)...")
ca_rows = panel[panel["codice_provincia"] == 92].set_index("anno")

patched_su = 0
for col in ["tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k"]:
    su_idx = panel.index[(panel["codice_provincia"] == 111) & panel[col].isna()]
    for idx in su_idx:
        anno = panel.loc[idx, "anno"]
        if anno in ca_rows.index and not pd.isna(ca_rows.loc[anno, col]):
            panel.loc[idx, col] = ca_rows.loc[anno, col]
            patched_su += 1
print(f"  Patchate {patched_su} celle per SU (111) con proxy CA")

# Patch 4: tasso_omicidi_100k — backward fill limitato per anni iniziali mancanti
# Province con dati omicidi che iniziano dopo il 2007 ricevono il primo valore noto
# come proxy per gli anni precedenti (max 10 anni indietro).
print("\nPatch 4: Backward fill limitato per tasso_omicidi_100k...")
patched_bfill = 0
for prov_code, grp in panel.groupby("codice_provincia"):
    omicidi = grp.set_index("anno")["tasso_omicidi_100k"]
    first_valid_year = omicidi.first_valid_index()
    if first_valid_year is None:
        continue
    first_valid_val = omicidi[first_valid_year]
    nan_before = omicidi[omicidi.index < first_valid_year].index[
        omicidi[omicidi.index < first_valid_year].isna()
    ]
    for yr in nan_before:
        if first_valid_year - yr <= 10:
            idx = panel.index[(panel["codice_provincia"] == prov_code) & (panel["anno"] == yr)]
            if len(idx):
                panel.loc[idx[0], "tasso_omicidi_100k"] = first_valid_val
                patched_bfill += 1
print(f"  Patchate {patched_bfill} celle via backward fill")


print("\nSalvataggio finale")

print(f"Shape finale: {panel.shape}")
print(f"NaN finali:")
for col in ["tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k"]:
    n_null = panel[col].isna().sum()
    print(f"  {col}: {n_null} NaN ({n_null / len(panel) * 100:.1f}%)")

# Dettaglio NaN residui
remaining = []
for col in ["tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k"]:
    for prov, grp in panel.groupby(["codice_provincia", "nome_provincia"]):
        n = grp[col].isna().sum()
        if n > 0:
            remaining.append({"codice": prov[0], "nome": prov[1], "colonna": col, "n_nan": n})
if remaining:
    print("\nNaN residui:")
    for r in remaining:
        print(f"  [{r['codice']:3d}] {r['nome']:<30s} | {r['colonna']:<30s} | {r['n_nan']:2d}")
else:
    print("\nNessun NaN residuo — dataset completo!")

panel.to_csv(OUTFILE, index=False)
print(f"\nSalvato -> {OUTFILE}")
print(panel.head(12).to_string())
