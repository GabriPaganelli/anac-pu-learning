"""
app/app.py — Streamlit UI per il risk scoring di appalti pubblici italiani.
"""

import sys
import json
from datetime import date
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from scorer import (
    load_stage_data,
    encode_contract,
    run_scoring,
    run_scoring_batch,
    run_bootstrap,
    run_shap,
)


DATA_DIR       = APP_DIR / "data"
REGISTRY_PATH  = DATA_DIR / "feature_registry.json"
PROVINCE_PATH  = DATA_DIR / "contesto_province.csv"
ENCODINGS_PATH = DATA_DIR / "encodings.json"


RISK_BANDS = [
    (80, 101, "🔴", "Molto elevato",  "#c0392b"),
    (50,  80, "🟠", "Elevato",        "#e67e22"),
    (20,  50, "🟡", "Moderato",       "#d4ac0d"),
    ( 0,  20, "🟢", "Basso",          "#27ae60"),
]


LABEL_CONSUMER: dict[str, str] = {
    "importo_complessivo_gara":           "Importo complessivo della gara (€)",
    "importo_lotto":                      "Importo del lotto (€)",
    "n_lotti_componenti":                 "Numero di lotti nella gara",
    "oggetto_principale_contratto":       "Tipo di contratto",
    "modalita_realizzazione_macro":       "Modalità di realizzazione",
    "sezione_regionale":                  "Sezione ANAC competente",
    "cod_strumento_svolgimento":          "Strumento di svolgimento della gara",
    "flag_urgenza":                       "Procedura d'urgenza",
    "cod_motivo_urgenza":                 "Motivo dell'urgenza",
    "flag_delega":                        "Contratto delegato ad altro ente",
    "finestra_offerta_giorni":            "Giorni per presentare l'offerta",
    "lag_perfezionamento_giorni":         "Giorni al perfezionamento del bando",
    "flag_accordo_quadro":                "Discendente da accordo quadro",
    "flag_ripetizioni":                   "Clausola di ripetizione lavori",
    "settore_speciale":                   "Settore speciale (acqua/gas/energia/trasporti)",
    "flag_appalto_riservato":             "Appalto riservato",
    "tipo_scelta_4cls":                   "Tipo di procedura di gara",
    "cpv_macro_categoria":                "Categoria merceologica (CPV)",
    "importo_sicurezza_pct":              "Quota costi sicurezza sul lotto (%)",
    "pct_riserva_base":                   "Somme a disposizione / totale (base d'asta)",
    "natura_giuridica_SA":                "Tipo di stazione appaltante",
    "numero_offerte_ammesse":             "Offerte ammesse",
    "numero_offerte_escluse":             "Offerte escluse",
    "num_imprese_offerenti":              "Imprese che hanno offerto",
    "pct_offerte_escluse":                "Quota offerte escluse (%)",
    "ribasso_aggiudicazione":             "Ribasso di aggiudicazione (%)",
    "ribasso_spread":                     "Differenza ribasso max − vincitore (pp)",
    "flag_vince_minimo":                  "Il vincitore ha offerto il ribasso più basso",
    "flag_progettazione_esterna":         "Progettazione affidata a professionista esterno",
    "lag_aggiudicazione_giorni":          "Giorni dalla pubblicazione all'aggiudicazione",
    "lag_comunicazione_esito_giorni":     "Giorni dalla pubblicazione alla comunicazione esito",
    "asta_elettronica":                   "Asta elettronica",
    "flag_scomputo":                      "Scomputo oneri sicurezza dal ribasso",
    "flag_proc_accelerata":               "Procedura accelerata",
    "importo_aggiudicazione":             "Importo di aggiudicazione (€)",
    "tipo_soggetto_agg":                  "Tipo di aggiudicatario",
    "flag_subappalto":                    "Subappalto previsto o autorizzato",
    "lag_stipula_aggiudicazione_giorni":  "Giorni tra aggiudicazione e firma contratto",
    "durata_pianificata_giorni":          "Durata pianificata del contratto (giorni)",
    "consegna_frazionata":                "Consegna frazionata del cantiere",
    "consegna_sotto_riserva":             "Consegna cantiere sotto riserva",
    "n_varianti":                         "Numero di varianti al contratto",
    "flag_variante_sostanziale":          "Almeno una variante sostanziale",
    "flag_variante_oltre_termine":        "Variante approvata oltre il termine",
    "n_sospensioni":                      "Numero di sospensioni dei lavori",
    "durata_totale_sospensioni_gg":       "Durata totale sospensioni (giorni)",
    "flag_sospensione":                   "Almeno una sospensione dei lavori",
    "flag_sosp_giudiziaria":              "Sospensione disposta dall'Autorità Giudiziaria",
    "pct_durata_sospesa":                 "Quota durata contrattuale in sospensione",
    "n_sal":                              "Numero di SAL (stati di avanzamento)",
    "flag_in_ritardo":                    "Lavori in ritardo (da SAL)",
    "flag_proroga":                       "Proroga dei termini concessa",
    "esito_collaudo":                     "Esito del collaudo",
    "pct_overrun_variante":               "Overrun da varianti / importo aggiudicazione",
    "pct_vita_prima_variante":            "Quota vita contrattuale prima della prima variante",
    "pct_overrun_core":                   "Costo finale / costo iniziale pianificato",
    "pct_riserva_consumata":              "Quota riserve finanziarie consumate",
}

CHOICE_DISPLAY: dict[str, dict[str, str]] = {
    "cod_strumento_svolgimento": {
        "1": "MePA",
        "2": "CONSIP",
        "3": "Accordo Quadro",
        "4": "Gara telematica",
        "5": "Piattaforma regionale (SATER/SINTEL/ecc.)",
        "6": "Sistema dinamico di acquisizione",
        "7": "Altro strumento digitale",
    },
    "cod_motivo_urgenza": {
        "1":  "Calamità naturale o eventi eccezionali",
        "2":  "Protezione civile",
        "3":  "Rischio per la salute pubblica",
        "4":  "Esigenze operative urgenti",
        "5":  "Motivi tecnici imprevisti",
        "6":  "Ragioni di sicurezza pubblica",
        "7":  "Altro motivo documentato",
        "98": "Altro",
        "99": "Non specificato",
    },
}

CHOICES_DISPLAY: dict[str, dict[str, str]] = {
    "oggetto_principale_contratto": {
        "FORNITURE": "Forniture", "LAVORI": "Lavori", "SERVIZI": "Servizi",
    },
    "modalita_realizzazione_macro": {
        "APPALTO":        "Appalto",
        "ECONOMIA":       "Acquisizione in economia",
        "ACCORDO_QUADRO": "Accordo quadro / convenzione",
        "CONCESSIONE":    "Concessione",
        "PPP":            "Partenariato pubblico-privato",
        "ALTRO":          "Altro",
    },
    "sezione_regionale": {
        "CENTRALE":  "Centrale (ANAC Roma)", "REGIONALE": "Sezione regionale",
    },
    "tipo_scelta_4cls": {
        "AFFIDAMENTO_DIRETTO": "Affidamento diretto",
        "APERTA":              "Procedura aperta",
        "NEGOZIATA":           "Procedura negoziata",
        "RISTRETTA":           "Procedura ristretta",
    },
    "cpv_macro_categoria": {
        "FORNITURE": "Forniture",
        "ING_PROF":  "Ingegneria / consulenza professionale",
        "IT":        "Informatica / software",
        "LAVORI":    "Lavori",
        "SANITA":    "Sanitario / farmaceutico",
        "SERVIZI":   "Servizi",
    },
    "natura_giuridica_SA": {
        "PA_ISTITUZIONALE": "PA centrale (ministeri/agenzie)",
        "PA_ENTE":          "Ente locale / regione / provincia",
        "SANITARIO":        "ASL / ospedale / IRCCS",
        "SOCIETA":          "Società mista o partecipata",
        "EPE":              "Ente pubblico economico",
        "NON_PROFIT":       "Fondazione / associazione",
        "CONSORZIO":        "Consorzio",
        "ALTRO":            "Altro",
    },
    "tipo_soggetto_agg": {
        "SINGOLA":   "Impresa singola",
        "ATI":       "Raggruppamento temporaneo (ATI)",
        "CONSORZIO": "Consorzio",
        "SA":        "Soggetto aggregatore",
        "GEIE":      "Gruppo europeo (GEIE)",
        "ALTRO":     "Altro",
    },
    "esito_collaudo": {
        "POSITIVO": "Positivo", "NEGATIVO": "Negativo / con riserve",
    },
}

N_COLS = 3
MAX_BOOTSTRAP_WORKERS = 4   # cap thread paralleli per bootstrap su Streamlit Cloud


_CHILDREN_M1 = {"cod_motivo_urgenza"}
_CHILDREN_M2 = {
    "numero_offerte_ammesse", "numero_offerte_escluse",
    "num_imprese_offerenti",  "pct_offerte_escluse", "ribasso_spread",
}
_CHILDREN_M3 = {
    "lag_stipula_aggiudicazione_giorni", "durata_pianificata_giorni",
    "consegna_frazionata",               "consegna_sotto_riserva",
    "flag_variante_sostanziale",         "flag_variante_oltre_termine",
    "pct_overrun_variante",              "pct_vita_prima_variante",
    "flag_in_ritardo",
    "pct_overrun_core",                  "pct_riserva_consumata",
}
_CHILDREN_ALL = _CHILDREN_M1 | _CHILDREN_M2 | _CHILDREN_M3

NA_OPTIONS: dict[str, str] = {
    "n_lotti_componenti":    "Gara a lotto singolo (non suddivisa in lotti)",
    "importo_sicurezza_pct": "Costi sicurezza non dichiarati nel bando",
    "pct_riserva_base":      "Riserve finanziarie non compilate nel quadro economico",
}



@st.cache_resource
def _load_registry() -> list[dict]:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def _load_encodings() -> dict:
    if not ENCODINGS_PATH.exists():
        return {}
    with open(ENCODINGS_PATH, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def _load_province_df() -> pd.DataFrame:
    return pd.read_csv(PROVINCE_PATH)

@st.cache_data(show_spinner=False)
def _load_parquet(stage: int, feat_key: frozenset) -> pd.DataFrame:
    return load_stage_data(stage, sorted(feat_key), DATA_DIR)



def _province_list(df_prov: pd.DataFrame) -> list[str]:
    return sorted(df_prov["nome_provincia"].unique().tolist())

def _istat_lookup(df_prov: pd.DataFrame, nome: str, anno_pub: int) -> dict:
    anno_join = min(int(anno_pub) - 1, 2024)
    sub = df_prov[(df_prov["nome_provincia"] == nome) & (df_prov["anno"] == anno_join)]
    if sub.empty:
        sub = df_prov[df_prov["nome_provincia"] == nome].sort_values("anno", ascending=False).head(1)
    if sub.empty:
        return {}
    r = sub.iloc[0]
    return {
        "tasso_disoccupazione":    float(r["tasso_disoccupazione"]),
        "reddito_irpef_procapite": float(r["reddito_irpef_procapite"]),
        "tasso_omicidi_100k":      float(r["tasso_omicidi_100k"]),
    }



def _detect_stage(values: dict, registry: list[dict]) -> int:
    m2 = {r["name"] for r in registry if r["stage"] == "M2"}
    m3 = {r["name"] for r in registry if r["stage"] == "M3"}
    provided = {k for k, v in values.items() if v is not None}
    if provided & m3: return 3
    if provided & m2: return 2
    return 1

def _build_active_features(values, registry, stage):
    stage_tags = {f"M{s}" for s in range(1, stage + 1)}
    dtype_map  = {r["name"]: r["dtype"] for r in registry}
    ordered    = [r["name"] for r in registry if r["stage"] in stage_tags]
    active     = [f for f in ordered if values.get(f) is not None]
    cat_names  = [f for f in active if dtype_map.get(f, "") in {"categorical", "binary_flag"}]
    return active, cat_names


def _feat_by_name(registry: list[dict], name: str) -> dict:
    return next(r for r in registry if r["name"] == name)


def _apply_dependencies(values: dict, overwrite: bool = False) -> None:
    """Imposta i valori di default per le feature dipendenti non mostrate nel form.

    None        = feature esclusa dal modello (campo non compilato)
    float('nan') = segnale informativo esplicito (scenario 'non applicabile')

    overwrite=True (batch path): le regole parent→child sono autoritative e
    sovrascrivono valori già presenti in values. Necessario per dati ANAC raw
    dove parent e child possono essere incoerenti.
    overwrite=False (UI path): non sovrascrive valori già settati dal form.
    """
    fu = values.get("flag_urgenza")
    if "cod_motivo_urgenza" not in values or (overwrite and fu == 0.0):
        values["cod_motivo_urgenza"] = float("nan") if fu == 0.0 else None

    for f in _CHILDREN_M2:
        if f not in values:
            values[f] = None
    for f in ["lag_stipula_aggiudicazione_giorni", "durata_pianificata_giorni",
              "consegna_frazionata", "consegna_sotto_riserva"]:
        if f not in values:
            values[f] = None

    nv = values.get("n_varianti")
    try:
        nv_zero = nv is not None and float(nv) == 0.0
    except (TypeError, ValueError):
        nv_zero = False

    for f in ["flag_variante_sostanziale", "flag_variante_oltre_termine",
              "pct_overrun_variante", "pct_vita_prima_variante"]:
        if f not in values or (overwrite and nv_zero):
            if nv is not None:
                try:
                    if float(nv) == 0.0:
                        values[f] = float("nan") if f.startswith("pct_") else 0.0
                    else:
                        values[f] = None
                except (TypeError, ValueError):
                    values[f] = None
            else:
                values[f] = None

    ns = values.get("n_sal")
    try:
        ns_zero = ns is not None and float(ns) == 0.0
    except (TypeError, ValueError):
        ns_zero = False

    if "flag_in_ritardo" not in values or (overwrite and ns_zero):
        if ns is not None:
            try:
                values["flag_in_ritardo"] = float("nan") if float(ns) == 0.0 else None
            except (TypeError, ValueError):
                values["flag_in_ritardo"] = None
        else:
            values["flag_in_ritardo"] = None

    for f in ["pct_overrun_core", "pct_riserva_consumata"]:
        if f not in values:
            values[f] = None



def _render_widget(feat: dict, key_pfx: str = "") -> object:
    name  = feat["name"]
    label = LABEL_CONSUMER.get(name, feat["label_ui"])
    itype = feat["ui_input_type"]
    help_ = feat.get("description", "")
    key   = f"{key_pfx}{name}"

    if itype == "derived_from_province":
        return None

    if itype == "number":
        if name in NA_OPTIONS:
            is_na = st.checkbox(NA_OPTIONS[name], key=f"{key}_na")
            if is_na:
                return float("nan")
        txt = st.text_input(label, value="", key=key, help=help_,
                            placeholder="Non disponibile")
        if txt.strip() == "":
            return None
        try:
            return float(txt.replace(",", "."))
        except ValueError:
            st.warning(f"Valore non valido per '{label}': inserisci un numero.")
            return None

    elif itype == "select":
        choices = feat.get("choices") or []
        if name in CHOICES_DISPLAY:
            dm      = CHOICES_DISPLAY[name]
            options = ["— non specificato —"] + [dm.get(c, c) for c in choices]
            sel     = st.selectbox(label, options, index=0, key=key, help=help_)
            if sel == "— non specificato —": return None
            return {v: k for k, v in dm.items()}.get(sel, sel)
        elif name in CHOICE_DISPLAY:
            dm      = CHOICE_DISPLAY[name]
            options = ["— non specificato —"] + list(dm.values())
            sel     = st.selectbox(label, options, index=0, key=key, help=help_)
            if sel == "— non specificato —": return None
            return {v: k for k, v in dm.items()}.get(sel, sel)
        else:
            options = ["— non specificato —"] + choices
            sel     = st.selectbox(label, options, index=0, key=key, help=help_)
            return None if sel == "— non specificato —" else sel

    elif itype == "toggle":
        sel = st.selectbox(label, ["— non specificato —", "Sì", "No"],
                           index=0, key=key, help=help_)
        if sel == "Sì": return 1.0
        if sel == "No": return 0.0
        return None

    return None



def _risk_band(rank: float) -> tuple[str, str, str]:
    for lo, hi, icon, label, color in RISK_BANDS:
        if lo <= rank < hi:
            return icon, label, color
    return "🔴", "Molto elevato", "#c0392b"

def _risk_bar_html(rank: float, color: str,
                   lo: float | None = None, hi: float | None = None) -> str:
    filled = max(2, int(rank))
    base = (
        f"<div style='background:#e0e0e0;border-radius:8px;height:22px;"
        f"position:relative;overflow:hidden'>"
    )
    if lo is not None and hi is not None:
        lo_pct = max(0, int(lo))
        hi_pct = min(100, int(hi))
        base += (
            f"<div style='position:absolute;left:{lo_pct}%;width:{hi_pct-lo_pct}%;"
            f"height:100%;background:{color};opacity:0.25'></div>"
            f"<div style='position:absolute;left:{lo_pct}%;width:2px;height:100%;"
            f"background:{color};opacity:0.7'></div>"
            f"<div style='position:absolute;left:{hi_pct}%;width:2px;height:100%;"
            f"background:{color};opacity:0.7'></div>"
        )
    base += (
        f"<div style='position:absolute;left:0;width:{filled}%;height:100%;"
        f"background:{color};opacity:0.85'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.72rem;color:#888;margin-top:2px'>"
        f"<span>Basso</span><span>Molto elevato</span></div>"
    )
    return base

def _risk_sentence(rank: float) -> str:
    pct = round(rank, 1)
    if pct >= 95:
        return f"Risulta tra gli appalti **più a rischio in assoluto**: supera il {pct}% degli appalti storici analizzati."
    elif pct >= 50:
        return f"Risulta più rischioso del **{pct}%** degli appalti storici analizzati."
    elif pct >= 20:
        return f"Presenta un rischio nella media: risulta più rischioso del **{pct}%** degli appalti storici analizzati."
    else:
        return f"Presenta un rischio contenuto: risulta più rischioso solo del **{pct}%** degli appalti storici analizzati."



def _render_result(res: dict, lo: float | None = None, hi: float | None = None) -> None:
    rank              = res["rank_pct"]
    icon, band, color = _risk_band(rank)

    st.markdown("---")
    c_main, c_bar = st.columns([3, 2])
    with c_main:
        st.markdown(f"## {icon} Rischio {band}")
        st.markdown(_risk_sentence(rank))
        st.caption(f"Confrontato con {res['n_contracts']:,} contratti pubblici italiani (2008–oggi).")

    with c_bar:
        st.markdown(
            _risk_bar_html(rank, color, lo, hi),
            unsafe_allow_html=True,
        )
        if lo is not None and hi is not None:
            st.caption(f"Intervallo di variabilità: {lo:.0f}% – {hi:.0f}% degli appalti storici")



def _shap_chart(shap_items: list[tuple[str, float]]) -> go.Figure:
    names   = [LABEL_CONSUMER.get(n, n) for n, _ in shap_items]
    values  = [v for _, v in shap_items]
    colors  = ["#c0392b" if v > 0 else "#27ae60" for v in values]
    labels  = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

    order   = sorted(range(len(values)), key=lambda i: abs(values[i]))
    names   = [names[i]  for i in order]
    values  = [values[i] for i in order]
    colors  = [colors[i] for i in order]
    labels  = [labels[i] for i in order]

    abs_max = max(abs(v) for v in values) if values else 1
    x_range = [-abs_max * 1.55, abs_max * 1.55]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Impatto: %{x:.4f}<extra></extra>",
    ))

    for i, (val, lbl) in enumerate(zip(values, labels)):
        xanchor = "left" if val >= 0 else "right"
        xshift  = 6 if val >= 0 else -6
        fig.add_annotation(
            x=val, y=i,
            text=f"<b>{lbl}</b>",
            showarrow=False,
            xanchor=xanchor,
            xshift=xshift,
            font=dict(size=10, color="#111111"),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
        )

    fig.add_vline(x=0, line_color="#888", line_width=1.5)

    fig.update_layout(
        height=max(300, len(shap_items) * 42),
        margin=dict(l=10, r=20, t=10, b=40),
        xaxis=dict(
            title="Impatto sul punteggio di rischio",
            range=x_range,
            zeroline=False,
            tickfont=dict(size=10),
        ),
        yaxis=dict(automargin=True, tickfont=dict(size=11)),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig



def _render_sidebar_legend() -> None:
    with st.sidebar:
        st.title("📚 Legenda — APPalti")
        st.caption("Descrizione dei gruppi e delle semplificazioni usate nel modulo.")

        with st.expander("Modalità di realizzazione", expanded=False):
            st.dataframe(pd.DataFrame([
                ("Appalto",                   "Contratto d'appalto standard",                    "Cod. 1"),
                ("Acquisizione in economia",  "Acquisto diretto sotto soglia",                   "Cod. 7"),
                ("Accordo quadro",            "Contratti da AQ / convenzione CONSIP/MePA",       "Cod. 2,9,11,17,18"),
                ("Concessione",               "Concessione di lavori o servizi pubblici",         "Cod. 3,4,20,21"),
                ("Partenariato (PPP)",         "PPP, finanza progetto, leasing operativo",        "Cod. 5,6,8,12,13"),
                ("Altro",                     "Concorsi, procedure speciali, ND",                "Cod. 10,14-16,19,90,999"),
            ], columns=["Classe", "Descrizione", "Codici ANAC"]), width="stretch", hide_index=True)

        with st.expander("Procedura di gara", expanded=False):
            st.dataframe(pd.DataFrame([
                ("Affidamento diretto", "Nessuna gara: assegnazione diretta sotto soglia (<40k€)"),
                ("Procedura negoziata", "La SA sceglie quali operatori invitare"),
                ("Procedura ristretta", "Prequalifica prima della gara"),
                ("Procedura aperta",    "Chiunque può partecipare"),
            ], columns=["Classe", "Descrizione"]), width="stretch", hide_index=True)

        with st.expander("Categoria merceologica (CPV)", expanded=False):
            st.dataframe(pd.DataFrame([
                ("Lavori",                  "Costruzioni, infrastrutture, opere civili",      "CPV 45"),
                ("Forniture",               "Beni, materiali, attrezzature",                  "CPV 02-44, 46-48"),
                ("Servizi",                 "Servizi generici, gestionali, logistici",         "CPV 50-79, 85-98"),
                ("IT",                      "Informatica, software, telco",                   "CPV 30-32, 48, 72"),
                ("Ingegneria/consulenza",   "Progettazione e consulenza tecnica",              "CPV 71"),
                ("Sanitario/farmaceutico",  "Dispositivi medici, farmaci, servizi sanitari",  "CPV 33, 85"),
            ], columns=["Categoria", "Descrizione", "Codici CPV"]), width="stretch", hide_index=True)

        with st.expander("Tipo di stazione appaltante", expanded=False):
            st.dataframe(pd.DataFrame([
                ("PA centrale",       "Ministeri, agenzie nazionali, presidenza del Consiglio"),
                ("Ente locale",       "Comuni, province, regioni, città metropolitane"),
                ("Sanitario",         "ASL, aziende ospedaliere, IRCCS, policlinici"),
                ("Società mista",     "Società partecipate da enti pubblici"),
                ("Ente pubbl. ec.",   "ANAS, RFI e simili"),
                ("Non-profit",        "Fondazioni, associazioni a controllo pubblico"),
                ("Consorzio",         "Consorzi di enti pubblici o di committenza"),
                ("Altro",             "Soggetti non classificabili"),
            ], columns=["Tipo", "Descrizione"]), width="stretch", hide_index=True)

        with st.expander("Strumento di svolgimento", expanded=False):
            st.dataframe(pd.DataFrame([
                ("1 — MePA",           "Mercato Elettronico della PA"),
                ("2 — CONSIP",         "Gara o convenzione CONSIP"),
                ("3 — Accordo Quadro", "Strumento a ordinativi"),
                ("4 — Gara telematica","Piattaforma e-procurement propria"),
                ("5 — Piattaf. reg.",  "SATER, SINTEL, ecc."),
                ("6 — Sist. dinamico", "Sistema dinamico di acquisizione"),
                ("7 — Altro digitale", "Altro strumento elettronico"),
            ], columns=["Codice", "Descrizione"]), width="stretch", hide_index=True)

        with st.expander("Motivo di urgenza", expanded=False):
            st.dataframe(pd.DataFrame([
                ("1",  "Calamità naturale o eventi eccezionali"),
                ("2",  "Protezione civile"),
                ("3",  "Rischio per la salute pubblica"),
                ("4",  "Esigenze operative urgenti"),
                ("5",  "Motivi tecnici imprevisti"),
                ("6",  "Ragioni di sicurezza pubblica"),
                ("7",  "Altro motivo documentato"),
                ("98", "Altro"), ("99", "Non specificato"),
            ], columns=["Codice ANAC", "Descrizione"]), width="stretch", hide_index=True)



def main() -> None:
    st.set_page_config(
        page_title="APPalti",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "**APPalti** — Strumento di analisi del rischio corruttivo negli appalti pubblici italiani.",
        },
    )
    st.markdown(
        "<style>footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )

    if not ENCODINGS_PATH.exists() or not (DATA_DIR / "M1_ready.parquet").exists():
        st.error("**Dati non trovati.** Esegui prima: `python app/setup.py`")
        st.stop()

    registry  = _load_registry()
    encodings = _load_encodings()
    df_prov   = _load_province_df()
    sna_map   = {r["name"]: r["structural_na"] for r in registry}

    by_stage: dict[str, list[dict]] = {"M1": [], "M2": [], "M3": []}
    for feat in registry:
        s = feat.get("stage", "")
        if s in by_stage and feat.get("ui_input_type") != "derived_from_province":
            by_stage[s].append(feat)

    _render_sidebar_legend()

    st.title("🏛️ APPalti")
    st.markdown(
        "Analisi del rischio corruttivo negli appalti pubblici italiani (2008–oggi).  "
        "← Apri il pannello laterale per la guida alle categorie."
    )
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # FORM
    # ════════════════════════════════════════════════════════════════════════

    values: dict = {}

    col_anno, col_prov = st.columns([1, 2])
    with col_anno:
        anno_pub = int(st.number_input(
            "Anno di pubblicazione", min_value=2008, max_value=2030,
            value=date.today().year, step=1,
        ))
    with col_prov:
        prov_sel = st.selectbox(
            "Provincia della stazione appaltante",
            ["— nessuna selezione —"] + _province_list(df_prov), index=0,
        )

    if prov_sel != "— nessuna selezione —":
        values.update(_istat_lookup(df_prov, prov_sel, anno_pub))

    st.markdown("---")

    st.subheader("📋 Dati di gara")
    m1_main = [f for f in by_stage["M1"] if f["name"] not in _CHILDREN_ALL]
    cols_m1 = st.columns(N_COLS)
    for idx, feat in enumerate(m1_main):
        with cols_m1[idx % N_COLS]:
            values[feat["name"]] = _render_widget(feat)

    if values.get("flag_urgenza") == 1.0:
        st.markdown("*Specificare il motivo dell'urgenza:*")
        cols_urg = st.columns(N_COLS)
        with cols_urg[0]:
            values["cod_motivo_urgenza"] = _render_widget(
                _feat_by_name(registry, "cod_motivo_urgenza")
            )

    with st.expander("🏆 Dati di aggiudicazione  *(opzionale)*", expanded=False):
        st.caption("Se inseriti, l'analisi sarà più accurata.")

        gara_comp = st.selectbox(
            "La gara ha avuto più partecipanti in competizione?",
            ["— non specificato —",
             "Sì — più offerte in competizione",
             "No — affidamento diretto o offerta unica"],
            index=0, key="toggle_gara_comp",
            help="Per affidamenti diretti o gare con un solo offerente, seleziona No.",
        )
        if gara_comp == "No — affidamento diretto o offerta unica":
            st.caption("ℹ️ I dati di partecipazione non verranno inclusi nell'analisi.")
            for f in _CHILDREN_M2:
                values[f] = float("nan")
        elif gara_comp == "Sì — più offerte in competizione":
            st.markdown("*Dettaglio partecipazione:*")
            m2_comp_feats = [_feat_by_name(registry, n) for n in [
                "numero_offerte_ammesse", "numero_offerte_escluse",
                "num_imprese_offerenti",  "pct_offerte_escluse", "ribasso_spread",
            ]]
            cols_comp = st.columns(N_COLS)
            for idx, feat in enumerate(m2_comp_feats):
                with cols_comp[idx % N_COLS]:
                    values[feat["name"]] = _render_widget(feat, key_pfx="m2_")

        m2_other = [f for f in by_stage["M2"] if f["name"] not in _CHILDREN_ALL]
        cols_m2 = st.columns(N_COLS)
        for idx, feat in enumerate(m2_other):
            with cols_m2[idx % N_COLS]:
                values[feat["name"]] = _render_widget(feat, key_pfx="m2_")

    with st.expander("🔧 Dati di esecuzione  *(opzionale)*", expanded=False):
        st.caption("Se inseriti, l'analisi sarà la più completa possibile.")

        avvio = st.selectbox(
            "I dati di avvio cantiere sono disponibili?",
            ["— non specificato —",
             "Sì — firma contratto e consegna cantiere registrati",
             "No — avvio cantiere non registrato in ANAC"],
            index=0, key="toggle_avvio",
            help="Include: giorni tra aggiudicazione e firma, durata pianificata, modalità consegna.",
        )
        if avvio == "No — avvio cantiere non registrato in ANAC":
            st.caption("ℹ️ I dati di avvio non verranno inclusi nell'analisi.")
            for f in ["lag_stipula_aggiudicazione_giorni", "durata_pianificata_giorni",
                      "consegna_frazionata", "consegna_sotto_riserva"]:
                values[f] = float("nan")
        elif avvio == "Sì — firma contratto e consegna cantiere registrati":
            m3_avvio_feats = [_feat_by_name(registry, n) for n in [
                "lag_stipula_aggiudicazione_giorni", "durata_pianificata_giorni",
                "consegna_frazionata", "consegna_sotto_riserva",
            ]]
            cols_avvio = st.columns(N_COLS)
            for idx, feat in enumerate(m3_avvio_feats):
                with cols_avvio[idx % N_COLS]:
                    values[feat["name"]] = _render_widget(feat, key_pfx="m3_")

        st.markdown("---")
        col_nv, _, _ = st.columns(N_COLS)
        with col_nv:
            values["n_varianti"] = _render_widget(
                _feat_by_name(registry, "n_varianti"), key_pfx="m3_"
            )
        nv = values.get("n_varianti")
        if nv is not None and float(nv) > 0:
            st.markdown("*Dettaglio varianti:*")
            m3_var_feats = [_feat_by_name(registry, n) for n in [
                "flag_variante_sostanziale", "flag_variante_oltre_termine",
                "pct_overrun_variante",      "pct_vita_prima_variante",
            ]]
            cols_var = st.columns(N_COLS)
            for idx, feat in enumerate(m3_var_feats):
                with cols_var[idx % N_COLS]:
                    values[feat["name"]] = _render_widget(feat, key_pfx="m3_")

        st.markdown("---")
        m3_sosp_feats = [f for f in by_stage["M3"] if f["name"] in {
            "n_sospensioni", "durata_totale_sospensioni_gg",
            "flag_sospensione", "flag_sosp_giudiziaria", "pct_durata_sospesa",
        }]
        cols_sosp = st.columns(N_COLS)
        for idx, feat in enumerate(m3_sosp_feats):
            with cols_sosp[idx % N_COLS]:
                values[feat["name"]] = _render_widget(feat, key_pfx="m3_")

        st.markdown("---")
        col_sal, _, _ = st.columns(N_COLS)
        with col_sal:
            values["n_sal"] = _render_widget(
                _feat_by_name(registry, "n_sal"), key_pfx="m3_"
            )
        ns = values.get("n_sal")
        if ns is not None and float(ns) > 0:
            st.markdown("*Avanzamento lavori:*")
            cols_sal2 = st.columns(N_COLS)
            with cols_sal2[0]:
                values["flag_in_ritardo"] = _render_widget(
                    _feat_by_name(registry, "flag_in_ritardo"), key_pfx="m3_"
                )

        m3_fine_feats = [f for f in by_stage["M3"]
                         if f["name"] in {"flag_proroga", "esito_collaudo"}]
        cols_fine = st.columns(N_COLS)
        for idx, feat in enumerate(m3_fine_feats):
            with cols_fine[idx % N_COLS]:
                values[feat["name"]] = _render_widget(feat, key_pfx="m3_")

        st.markdown("---")
        consuntivo = st.selectbox(
            "I dati di costo finale (consuntivo) sono disponibili?",
            ["— non specificato —",
             "Sì — contratto completato con consuntivo registrato",
             "No — contratto in corso o consuntivo non disponibile"],
            index=0, key="toggle_consuntivo",
            help="Include: costo finale rispetto al pianificato, riserve finanziarie consumate.",
        )
        if consuntivo == "No — contratto in corso o consuntivo non disponibile":
            st.caption("ℹ️ I dati di consuntivo non verranno inclusi nell'analisi.")
            for f in ["pct_overrun_core", "pct_riserva_consumata"]:
                values[f] = float("nan")
        elif consuntivo == "Sì — contratto completato con consuntivo registrato":
            m3_cons_feats = [_feat_by_name(registry, n)
                             for n in ["pct_overrun_core", "pct_riserva_consumata"]]
            cols_cons = st.columns(N_COLS)
            for idx, feat in enumerate(m3_cons_feats):
                with cols_cons[idx % N_COLS]:
                    values[feat["name"]] = _render_widget(feat, key_pfx="m3_")

    _apply_dependencies(values)

    st.markdown("---")
    do_score = st.button("🔍 Analizza rischio", type="primary")

    # ════════════════════════════════════════════════════════════════════════
    # CALCOLO
    # ════════════════════════════════════════════════════════════════════════

    if do_score:
        stage_det               = _detect_stage(values, registry)
        active_features, cat_nm = _build_active_features(values, registry, stage_det)
        feat_key                = frozenset(active_features)

        df_score     = _load_parquet(stage_det, feat_key)
        contract_vec = encode_contract(values, active_features, encodings)

        prog = st.progress(15, text="Analisi in corso...")
        result = run_scoring(
            df_score, active_features, contract_vec, cat_nm,
            stage=stage_det, progress_cb=None, return_model=True,
        )
        prog.progress(100, text="Completato.")
        prog.empty()

        result["stage_int"] = stage_det
        st.session_state.update({
            "last_result":   result,
            "last_df":       df_score,
            "last_features": active_features,
            "last_cats":     cat_nm,
            "last_vec":      contract_vec,
            "last_stage":    stage_det,
            "boot_lo":       None,
            "boot_hi":       None,
            "shap_items":    None,
        })

    # ════════════════════════════════════════════════════════════════════════
    # RISULTATO
    # ════════════════════════════════════════════════════════════════════════

    if "last_result" not in st.session_state:
        return

    res     = st.session_state["last_result"]
    boot_lo = st.session_state.get("boot_lo")
    boot_hi = st.session_state.get("boot_hi")

    _render_result(res, lo=boot_lo, hi=boot_hi)

    st.markdown("---")
    st.subheader("🔁 Quanto è affidabile il risultato?")
    st.caption(
        "Ripete l'analisi su sottoinsiemi diversi dei dati storici per stimare "
        "quanto il punteggio potrebbe variare. Più iterazioni = stima più stabile."
    )

    col_sl, col_btn = st.columns([3, 1])
    with col_sl:
        n_iter = st.slider(
            "Precisione della verifica",
            min_value=30, max_value=200, value=50, step=10,
            key="boot_n", format="%d iterazioni",
        )
    with col_btn:
        st.write("")
        do_boot = st.button("Avvia verifica", key="boot_btn")

    if do_boot:
        boot_bar = st.progress(0.0, text="Verifica in corso... 0%")

        def _boot_cb(b: int, total: int) -> None:
            boot_bar.progress(b / total, text=f"Verifica in corso... {int(b/total*100)}%")

        lo, hi = run_bootstrap(
            st.session_state["last_df"],
            st.session_state["last_features"],
            st.session_state["last_vec"],
            st.session_state["last_cats"],
            stage=st.session_state["last_stage"],
            n_boot=n_iter,
            n_workers=MAX_BOOTSTRAP_WORKERS,
            progress_cb=_boot_cb,
        )
        boot_bar.progress(1.0, text="Completato.")
        boot_bar.empty()

        st.session_state["boot_lo"] = lo
        st.session_state["boot_hi"] = hi

    # Mostra risultati bootstrap se disponibili (persistono tra rerun)
    _boot_lo = st.session_state.get("boot_lo")
    _boot_hi = st.session_state.get("boot_hi")
    if _boot_lo is not None and _boot_hi is not None:
        rank_pt       = res["rank_pct"]
        _, lo_band, _ = _risk_band(_boot_lo)
        _, hi_band, color = _risk_band(rank_pt)

        st.markdown(_risk_bar_html(rank_pt, color, _boot_lo, _boot_hi), unsafe_allow_html=True)

        if lo_band == hi_band:
            st.success(
                f"**Risultato stabile**: in tutti gli scenari analizzati il rischio rimane "
                f"**{lo_band.lower()}**.  \n"
                f"Il punteggio varia tra il **{_boot_lo:.0f}%** e il **{_boot_hi:.0f}%** degli appalti storici."
            )
        else:
            st.info(
                f"Il punteggio varia tra il **{_boot_lo:.0f}%** e il **{_boot_hi:.0f}%** degli appalti storici, "
                f"corrispondendo a un rischio tra **{lo_band.lower()}** e **{hi_band.lower()}**."
            )
            if _boot_hi - _boot_lo > 30:
                st.warning(
                    "La variabilità è elevata. "
                    "Aggiungere più dati sull'appalto può aumentare la precisione."
                )

    st.markdown("---")
    st.subheader("📊 Contributo delle variabili al punteggio")
    st.caption(
        "Mostra quali dati hanno aumentato o ridotto il punteggio di rischio "
        "e di quanto, rispetto al valore atteso medio."
    )

    col_k, col_shap_btn = st.columns([3, 1])
    with col_k:
        top_k = st.slider(
            "Numero di variabili da mostrare",
            min_value=5, max_value=20, value=10, step=1, key="shap_k",
        )
    with col_shap_btn:
        st.write("")
        do_shap = st.button("Calcola contributi", key="shap_btn")

    if do_shap:
        booster = res.get("booster")
        if booster is None:
            st.warning("Riesegui l'analisi per poter calcolare i contributi.")
        else:
            with st.spinner("Calcolo in corso..."):
                shap_items = run_shap(
                    booster,
                    st.session_state["last_vec"],
                    st.session_state["last_features"],
                    top_k=top_k,
                )
            st.session_state["shap_items"] = shap_items

    # Mostra grafico SHAP se disponibile (persiste tra rerun)
    _shap_items = st.session_state.get("shap_items")
    if _shap_items:
        st.plotly_chart(_shap_chart(_shap_items), width="stretch")
        st.caption(
            "🔴 Rosso = aumenta il rischio  ·  🟢 Verde = riduce il rischio  ·  "
            "Le barre più lunghe indicano le variabili più influenti."
        )

    # ════════════════════════════════════════════════════════════════════════
    # BATCH CSV
    # ════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    with st.expander("📂 Analisi in batch (CSV)", expanded=False):
        st.markdown(
            "Carica un CSV con una riga per appalto. "
            "Per ogni riga verrà calcolato il punteggio di rischio."
        )
        st.info(
            "Righe con lo stesso set di dati disponibili vengono elaborate insieme "
            "(un solo training per gruppo) — il batch è significativamente più veloce "
            "di N analisi singole."
        )

        template_cols = (
            ["anno_pubblicazione", "provincia"]
            + [r["name"] for r in registry if r.get("ui_input_type") != "derived_from_province"]
        )
        st.download_button(
            "Scarica template CSV",
            data=pd.DataFrame(columns=template_cols).to_csv(index=False),
            file_name="template_appalti.csv", mime="text/csv",
        )

        uploaded = st.file_uploader("Carica CSV", type=["csv"], key="batch_upload")
        if uploaded is not None:
            df_batch = pd.read_csv(uploaded)
            st.write(f"**{len(df_batch):,} appalti caricati.** Anteprima:")
            st.dataframe(df_batch.head(5), width="stretch")

            if st.button("Avvia analisi", key="batch_run"):
                n_rows    = len(df_batch)
                prog_b    = st.progress(0.0, text="Preparazione...")
                completed = 0

                rows_data: list[dict] = []
                for _, row in df_batch.iterrows():
                    row_vals: dict = {}
                    for k, v in row.items():
                        if isinstance(v, float) and np.isnan(v):
                            # NaN strutturale → segnale informativo; NaN casuale → escluso
                            row_vals[k] = float("nan") if sna_map.get(k, False) else None
                        else:
                            row_vals[k] = v

                    anno_r = int(row_vals.get("anno_pubblicazione") or date.today().year)
                    prov_r = str(row_vals.get("provincia", "") or "").strip()
                    if prov_r:
                        row_vals.update(_istat_lookup(df_prov, prov_r, anno_r))

                    _apply_dependencies(row_vals, overwrite=True)

                    stage_r          = _detect_stage(row_vals, registry)
                    active_r, cat_r  = _build_active_features(row_vals, registry, stage_r)
                    vec_r            = encode_contract(row_vals, active_r, encodings)
                    rows_data.append({
                        "stage":    stage_r,
                        "active":   active_r,
                        "cats":     cat_r,
                        "vec":      vec_r,
                        "feat_key": frozenset(active_r),
                    })

                groups: dict[tuple, list[int]] = defaultdict(list)
                for i, rd in enumerate(rows_data):
                    groups[(rd["stage"], rd["feat_key"])].append(i)

                results_batch: list[dict | None] = [None] * n_rows

                for (stage_g, feat_key_g), row_indices in groups.items():
                    df_g    = _load_parquet(stage_g, feat_key_g)
                    active_g = rows_data[row_indices[0]]["active"]
                    cats_g   = rows_data[row_indices[0]]["cats"]
                    vecs_g   = [rows_data[i]["vec"] for i in row_indices]

                    group_results = run_scoring_batch(
                        df_g, active_g, vecs_g, cats_g, stage=stage_g
                    )

                    for local_idx, global_idx in enumerate(row_indices):
                        r            = group_results[local_idx]
                        _, band_r, _ = _risk_band(r["rank_pct"])
                        results_batch[global_idx] = {
                            "appalto":  global_idx + 1,
                            "rischio":  band_r,
                            "punteggio (% appalti superati)": round(r["rank_pct"], 1),
                        }

                    completed += len(row_indices)
                    prog_b.progress(
                        completed / n_rows,
                        text=f"Analisi in corso... {int(completed/n_rows*100)}% ({completed}/{n_rows})",
                    )

                prog_b.progress(1.0, text="Completato.")
                df_out = pd.DataFrame(results_batch)
                st.dataframe(df_out, width="stretch")
                st.download_button(
                    "Scarica risultati",
                    data=df_out.to_csv(index=False),
                    file_name="risultati_rischio.csv", mime="text/csv",
                )


if __name__ == "__main__":
    main()
