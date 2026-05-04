"""
Genera grafici lift@1% vs gamma per i 4 modelli finalisti.

Linee tratteggiate colorate = gamma_best automatico (argmax media per dataset).
Linee verticali rosse      = gamma* scelto manualmente (gamma_star.json).

Uso:
  python plot_mixing.py           # salva results/mixing_lift_vs_gamma.{png,pdf}
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

_HERE      = Path(__file__).resolve().parent
RESULTS    = _HERE / "results"
GAMMA_JSON = _HERE.parents[0] / "evaluation" / "gamma_star.json"

MODELS  = ["lgb_supervised", "bagging_lgbm", "re_lgbm", "puet"]
TITLES  = {
    "lgb_supervised": "LGB P vs N certi (EM-Hard)",
    "bagging_lgbm":   "Bagging LightGBM",
    "re_lgbm":        "RE LightGBM (nnPU)",
    "puet":           "PUET (Extra Trees)",
}
COLORS  = {"M1": "#2196F3", "M2": "#FF9800", "M3": "#4CAF50"}
MARKERS = {"M1": "o", "M2": "s", "M3": "^"}

# Carica gamma* da gamma_star.json (se esiste)
gamma_star = {}
if GAMMA_JSON.exists():
    with open(GAMMA_JSON) as f:
        gamma_star = json.load(f)   # {model: {"M1": float, "M2": float, "M3": float}}

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
axes = axes.flatten()

for ax, model in zip(axes, MODELS):
    model_gs = gamma_star.get(model, {})

    for mn in [1, 2, 3]:
        path = RESULTS / f"mixing_{model}_M{mn}_summary.csv"
        if not path.exists():
            continue
        df    = pd.read_csv(path)
        label = f"M{mn}"
        ax.errorbar(
            df["gamma"], df["lift@1%_mean"],
            yerr=df["lift@1%_sd"],
            label=label,
            color=COLORS[label], marker=MARKERS[label],
            linewidth=1.8, markersize=5,
            capsize=3, elinewidth=0.8, alpha=0.9,
        )
        # Tratteggio colorato = argmax automatico (riferimento veloce)
        best_idx = df["lift@1%_mean"].idxmax()
        ax.axvline(df.loc[best_idx, "gamma"], color=COLORS[label],
                   linestyle="--", linewidth=0.8, alpha=0.4)

    # Linea rossa = gamma* scelto manualmente da gamma_star.json
    # (unica per modello: se M1=M2=M3 una sola linea; altrimenti tre)
    plotted_gs = set()
    for mn in [1, 2, 3]:
        gs_key = f"M{mn}"
        gs_val = model_gs.get(gs_key)
        if gs_val is not None and gs_val not in plotted_gs:
            ax.axvline(gs_val, color="red", linewidth=1.4, linestyle="-",
                       alpha=0.7,
                       label=f"g* M{mn}={gs_val}" if len(set(model_gs.values())) > 1
                             else f"g*={gs_val}")
            plotted_gs.add(gs_val)

    ax.set_title(TITLES[model], fontsize=11, fontweight="bold")
    ax.set_xlabel("gamma", fontsize=10)
    ax.set_ylabel("Lift@1% (media +/- SD)", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    ax.set_xlim(-0.03, 1.03)

fig.suptitle("PNU Mixing — Lift@1% vs gamma (4-fold OOF, seed=2025)\n"
             "Linee tratteggiate colorate = gamma_best auto | Linee rosse = gamma* scelto",
             fontsize=12)
fig.tight_layout()

for fmt in ("png", "pdf"):
    out = RESULTS / f"mixing_lift_vs_gamma.{fmt}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Salvato: {out}")
plt.show()
