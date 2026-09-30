# -*- coding: utf-8 -*-
"""
Appendix figure: three-model validation (RF, ET, XGB), one row of two panels.

Bars show spatial block CV performance with 95% CI, triangles the in-sample
(fitted) performance, and open circles the random CV performance.
Panel titles are limited to (a) and (b); model settings are given in the caption.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

METRICS_CSV = os.environ.get("RF_METRICS", "output/ml/rf_metrics.csv")
OUT_DIR = os.environ.get("FIG_OUT", "output/figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "font.size": 9})

M = pd.read_csv(METRICS_CSV)
MI = M[M.qc == "on"].copy()

MODEL_COLORS = {"RF": "#2E7D32", "ET": "#FBC02D", "XGB": "#1E88E5"}
MODELS = ["RF", "ET", "XGB"]


def panel_perf(ax, outcomes, tag, ylab, fmt):
    """Grouped bars of spatial block CV performance with CI and reference markers."""
    x = np.arange(len(outcomes))
    w = 0.24
    all_in, all_top = [], []

    for i, mo in enumerate(MODELS):
        vals, los, his, ins, rnd = [], [], [], [], []
        for oc in outcomes:
            r = MI[(MI.outcome == oc) & (MI.model == mo)].iloc[0]
            vals.append(r["spatial_cv_mean"])
            los.append(r["spatial_cv_mean"] - r["spatial_cv_lo"])
            his.append(r["spatial_cv_hi"] - r["spatial_cv_mean"])
            ins.append(r["in_sample"])
            rnd.append(r["random_cv_mean"])
        all_in += ins

        off = (i - 1) * w
        ax.bar(x + off, vals, w, yerr=np.array([los, his]), capsize=2.5,
               color=MODEL_COLORS[mo], alpha=0.92, label=mo, zorder=3,
               edgecolor="white", linewidth=0.5)
        ax.plot(x + off, ins, marker="^", ls="none", ms=4.2, color="#B71C1C", zorder=4)
        ax.plot(x + off, rnd, marker="o", ls="none", ms=3.5, mfc="none",
                mec="#444444", zorder=4)

        pad = 0.012 if fmt == "R2" else 0.006
        for xi, v, hi in zip(x + off, vals, his):
            ax.text(xi, v + hi + pad, f"{v:.3f}", ha="center", fontsize=6.3,
                    color="#333333", zorder=5, rotation=90)
            all_top.append(v + hi + (0.030 if fmt == "R2" else 0.014))

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=9.5)
    ax.set_ylabel(ylab, fontsize=9.8)
    ax.set_title(f"({tag})", fontsize=11.5, fontweight="bold", pad=8)
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [Patch(fc=MODEL_COLORS[m], label=m) for m in MODELS]
    handles += [Line2D([], [], marker="o", ls="none", ms=4, mfc="none", mec="#444444",
                       label="random CV"),
                Line2D([], [], marker="^", ls="none", ms=4.5, color="#B71C1C",
                       label="in-sample (fitted)")]
    ax.legend(handles=handles, fontsize=7.6, loc="upper center",
              bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True,
              framealpha=0.95, columnspacing=1.0, handletextpad=0.5)
    ax.set_ylim(0.0, max(max(all_top), max(all_in)) * 1.07)


PARAM_TXT = (
    "RandomForest (RF): n_estimators = 300, max_depth = 30, max_features = 1.0, "
    "min_samples_leaf = 10, bootstrap = True, oob_score = True, random_state = 42\n"
    "ExtraTrees (ET)  : n_estimators = 300, max_depth = 30, max_features = 1.0, "
    "min_samples_leaf = 10, random_state = 42\n"
    "XGBoost (XGB)    : n_estimators = 300, max_depth = 6, learning_rate = 0.03, "
    "subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 1.0, random_state = 42\n"
    "Control protocol : confounders (cubic age, sex, BMI, smoking, alcohol, physical "
    "activity, diet regularity) enter as OLS offsets, not as forest features; the "
    "forest sees only the ten greenness and environment features.\n"
    "Validation       : random 5-fold CV (repeated) and spatial block CV "
    "(KMeans 10 blocks x 3 seeds = 30 folds), 95% CI."
)


def make_fig(path):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    panel_perf(axes[0], ["SBP", "DBP", "MAP", "FPG"], "a",
               "Spatial-block CV  R$^2$", "R2")
    panel_perf(axes[1], ["Hypertension", "Hyperglycaemia"], "b",
               "Spatial-block CV  ROC-AUC", "AUC")

    fig.tight_layout(rect=[0.005, 0.155, 0.995, 0.99])
    fig.text(0.5, 0.012, PARAM_TXT, ha="center", va="bottom", fontsize=7.8,
             family="DejaVu Sans Mono", color="#222222", linespacing=1.55,
             bbox=dict(boxstyle="round,pad=0.55", fc="#F7F7F7", ec="#9E9E9E", lw=0.8))

    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".tiff"), dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.replace(".png", ".pdf"), dpi=300)
    fig.savefig(path.replace(".png", ".svg"), dpi=300)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    make_fig(os.path.join(OUT_DIR, "rf_three_model_validation.png"))
