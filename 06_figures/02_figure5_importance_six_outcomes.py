# -*- coding: utf-8 -*-
"""
Figure: RF feature importance for all six cardiometabolic outcomes, plus the
three-ensemble spatial-block CV comparison (extended version of
rf_feature_importance.py).

Panels
  (a)-(f) RF Gini importance (sum-to-1) for SBP, DBP, MAP, FPG, hypertension,
          hyperglycaemia.
  (g)     Spatial-block CV R2 for the four continuous outcomes.
  (h)     Spatial-block CV ROC-AUC for the two binary outcomes.

Inputs: RF_METRICS_CSV, RF_IMPORTANCE_CSV, OUT_DIR (see the sibling script).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from rf_feature_importance import (COL, PARAM_TXT, color_of,
                                           panel_importance, panel_perf,
                                           legend_handles)

METRICS_CSV = os.environ.get("RF_METRICS_CSV", "RF_控制协议_指标.csv")
IMPORT_CSV = os.environ.get("RF_IMPORTANCE_CSV", "RF_控制协议_重要度.csv")
OUT_DIR = os.environ.get("OUT_DIR", ".")

SIX = ["sbp", "dbp", "map", "fpg", "是否高血压", "是否高血糖"]
TAGS = list("abcdef")


def main():
    met = pd.read_csv(METRICS_CSV)
    imp = pd.read_csv(IMPORT_CSV)
    met = met[met[COL["qc"]] == "on"].copy()

    fig = plt.figure(figsize=(18.0, 15.6))
    gs = GridSpec(3, 6, figure=fig, hspace=0.42, wspace=0.30)
    for k, (oc, tg) in enumerate(zip(SIX, TAGS)):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(gs[r, c * 2:(c + 1) * 2])
        panel_importance(ax, imp, met, oc, tg)

    axg = fig.add_subplot(gs[2, 0:3])
    axh = fig.add_subplot(gs[2, 3:6])
    panel_perf(axg, met, ["sbp", "dbp", "map", "fpg"], "g",
               "Spatial-block CV R2")
    panel_perf(axh, met, ["是否高血压", "是否高血糖"], "h",
               "Spatial-block CV ROC-AUC", is_auc=True)

    fig.text(0.5, 0.014, PARAM_TXT, ha="center", va="bottom", fontsize=8.0,
             family="DejaVu Sans Mono", color="#222222", linespacing=1.55,
             bbox=dict(boxstyle="round,pad=0.55", fc="#F7F7F7", ec="#9E9E9E", lw=0.8))
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.958),
               ncol=3, fontsize=8.6, frameon=True, framealpha=0.95)
    fig.tight_layout(rect=[0.005, 0.125, 0.995, 0.952])

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, "Figure_importance_six_outcomes")
    for ext, kw in ((".png", {"dpi": 600, "bbox_inches": "tight"}),
                    (".tiff", {"dpi": 600, "bbox_inches": "tight",
                               "pil_kwargs": {"compression": "tiff_lzw"}}),
                    (".pdf", {"dpi": 600, "bbox_inches": "tight"}),
                    (".svg", {"dpi": 600, "bbox_inches": "tight"})):
        fig.savefig(base + ext, **kw)
    plt.close(fig)
    print("saved", base)


if __name__ == "__main__":
    main()
