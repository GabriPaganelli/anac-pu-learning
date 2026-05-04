"""
preprocessing.py
Funzioni di preprocessing condivise tra tutti gli script di modellazione.
Importabile da qualsiasi script che aggiunge Appalti/ al sys.path.

Uso:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))
    from metrics.preprocessing import encode_categoricals
"""

import pandas as pd

# Colonne amministrative sempre escluse dal modello
_ADMIN_COLS = {"cig", "esito", "anno_pubblicazione", "regione", "label", "fold"}


def encode_categoricals(df: pd.DataFrame,
                        extra_drop: list | None = None,
                        drop_first: bool = False,
                        verbose: bool = True) -> pd.DataFrame:
    """
    One-hot encoding delle colonne non numeriche (object / category / string).

    Le colonne amministrative standard (cig, esito, anno_pubblicazione, regione,
    label, fold) sono sempre escluse dall'encoding. Colonne aggiuntive da escludere
    possono essere passate via extra_drop.

    Le dummy sono costruite sui livelli del dataset completo prima del CV split:
    i livelli categorici sono fissi per costruzione (tipo procedura, categoria
    merceologica, ecc.), quindi non c'e' data leakage.

    Parameters
    ----------
    df         : DataFrame in ingresso (non modificato in-place)
    extra_drop : colonne aggiuntive da non encodare (oltre a quelle amministrative)
    drop_first : se True, elimina il primo livello per evitare multicollinearita'
                 (default False — con Elastic Net / tree la regularizzazione gestisce
                 la collinearita' e conservare tutti i livelli aiuta SHAP)
    verbose    : stampa le colonne encodate

    Returns
    -------
    DataFrame con le colonne categoriche sostituite dalle rispettive dummy (dtype float).
    """
    exclude = _ADMIN_COLS | set(extra_drop or [])
    cat_cols = [c for c in df.columns
                if c not in exclude and df[c].dtype.kind not in "iufb"]
    if not cat_cols:
        return df
    if verbose:
        print(f"  encode_categoricals: {len(cat_cols)} colonne -> {cat_cols}")
    return pd.get_dummies(df, columns=cat_cols, drop_first=drop_first, dtype=float)
