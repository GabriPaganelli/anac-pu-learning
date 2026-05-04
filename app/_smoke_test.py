"""Smoke test per app/ — dati reali in sola lettura, nessuna scrittura su data/."""
import sys
import json
import shutil
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd

print("1. Import moduli...")
import re_lgbm
import scorer
import setup as app_setup
print("   OK")

# Monkeypatch costanti pesanti (ripristinate nel finally)
_orig_rounds = re_lgbm.N_ROUNDS_MAX
_orig_stop   = re_lgbm.EARLY_STOP
re_lgbm.N_ROUNDS_MAX = 10
re_lgbm.EARLY_STOP   = 5

try:
    print("2. scorer.compute_u_size...")
    u = scorer.compute_u_size(n_P=100, pi=0.02)
    assert scorer.U_FLOOR <= u <= scorer.U_CAP, f"u={u} fuori range"
    print("   OK")

    print("3. scorer.encode_contract...")
    with open(APP_DIR / "data" / "encodings.json", encoding="utf-8") as f:
        encodings = json.load(f)
    with open(APP_DIR / "data" / "feature_registry.json", encoding="utf-8") as f:
        registry = json.load(f)

    m1_feats = [r["name"] for r in registry
                if r["stage"] == "M1" and r["ui_input_type"] != "derived_from_province"]
    values_dict = {f: None for f in m1_feats}
    vec = scorer.encode_contract(values_dict, m1_feats, encodings)
    assert vec.shape == (len(m1_feats),), f"shape errata: {vec.shape}"
    assert vec.dtype == np.float32
    print("   OK")

    print("4. scorer.run_scoring su M1_ready.parquet (N_ROUNDS=10)...")
    df = pd.read_parquet(APP_DIR / "data" / "M1_ready.parquet")
    all_m1 = [r["name"] for r in registry if r["stage"] == "M1"]
    active  = [f for f in all_m1 if f in df.columns]
    cat_names = [f for f in active if f in encodings]

    first_row = df[active].iloc[0].to_dict()
    vec_full  = scorer.encode_contract(first_row, active, encodings)

    result = scorer.run_scoring(df, active, vec_full, cat_names, stage=1)
    assert "rank_pct" in result
    assert 0.0 <= result["rank_pct"] <= 100.0
    print(f"   OK — rank_pct={result['rank_pct']:.1f}%  best_iter={result['best_iter']}")

    print("5. scorer.run_bootstrap (n_boot=3, n_workers=1)...")
    lo, hi = scorer.run_bootstrap(
        df, active, vec_full, cat_names,
        stage=1, n_boot=3, n_workers=1,
    )
    assert 0.0 <= lo <= hi <= 100.0, f"CI non valido: [{lo}, {hi}]"
    print(f"   OK — CI=[{lo:.1f}%, {hi:.1f}%]")

    print("6. setup._extract_encodings + _convert_categoricals_nativi (sintetico)...")
    df_syn = pd.DataFrame({
        "label":   [1.0, 0.0, None],
        "fold":    [0.0, 1.0, None],
        "cat_col": ["A", "B", None],
        "num_col": [1.0, 2.0, 3.0],
        "cig":     ["c1", "c2", "c3"],
        "esito":   [0, 0, 0],
        "anno_pubblicazione": [2020, 2021, 2022],
        "regione": ["Lombardia", "Lazio", "Sicilia"],
    })
    enc_test = {}
    app_setup._extract_encodings(df_syn, enc_test)
    assert "cat_col" in enc_test

    df_conv, cat_out = app_setup._convert_categoricals_nativi(df_syn.copy())
    assert "cat_col" in cat_out
    assert df_conv["cat_col"].dtype == np.float32
    print("   OK")

    print("7. setup.process_stage(1) su dir temporanea (non tocca data/)...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="appalti_smoke_"))
    try:
        _orig_out = app_setup.OUT_DIR
        app_setup.OUT_DIR = tmp_dir
        enc_tmp = {}
        app_setup.process_stage(1, enc_tmp)
        out_file = tmp_dir / "M1_ready.parquet"
        assert out_file.exists(), "M1_ready.parquet non trovato in tmp"
        mb = out_file.stat().st_size / 1e6
        print(f"   OK — {mb:.1f} MB scritti in {tmp_dir} (cancellati ora)")
    finally:
        app_setup.OUT_DIR = _orig_out
        shutil.rmtree(tmp_dir, ignore_errors=True)

finally:
    re_lgbm.N_ROUNDS_MAX = _orig_rounds
    re_lgbm.EARLY_STOP   = _orig_stop

print("\nOK - Tutti i smoke test superati.")
