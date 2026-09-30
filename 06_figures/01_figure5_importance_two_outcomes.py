# -*- coding: utf-8 -*-
"""
Figure: Random-forest feature attribution and spatially validated performance.

Panels
  (a)(b) RF Gini importance (sum-to-1) for two representative outcomes.
  (c)(d) Spatial-block cross-validated performance of RF / ET / XGB for the
         continuous and the binary outcomes, with 95% CI across folds.

Inputs (set through environment variables)
  RF_METRICS_CSV      per-model performance table (one row per outcome x model)
  RF_IMPORTANCE_CSV   per-feature importance table (long format)
  OUT_DIR             output directory for png / tiff / pdf / svg

The individual-level confounders are not forest features; they enter as an OLS
offset (see machine_learning/rf_control_offset.py). The forest therefore sees
only the ten greenness and environment features.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

METRICS_CSV = os.environ.get("RF_METRICS_CSV", "RF_控制协议_指标.csv")
IMPORT_CSV = os.environ.get("RF_IMPORTANCE_CSV", "RF_控制协议_重要度.csv")
OUT_DIR = os.environ.get("OUT_DIR", ".")

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "font.size": 8.5})

# Source tables use Chinese column names; map them once for readability.
COL = {
    "qc": "质控", "outcome": "结局", "model": "模型", "feature": "特征",
    "importance": "重要性", "otype": "结局类型", "in_sample": "样本内",
    "cv": "空间CV均值", "cv_lo": "空间CV_CI低", "cv_hi": "空间CV_CI高",
    "oob": "OOB_原始尺度", "mae": "样本内_MAE", "rmse": "样本内_RMSE",
    "leak": "空间泄漏", "random_cv": "随机CV均值", "n": "样本量",
}

NAME = {
    "绿度强度_r500": "Greenness intensity",
    "绿度聚集度_GAI": "Aggregation index",
    "季节振幅_谐波_r500": "Seasonal amplitude",
    "绿度年龄_r500": "Greenness age",
    "绿度暴露_r500": "Greenness exposure",
    "道路距离_主干道_r500": "Distance to main roads",
    "夜光_r500": "Night light",
    "DEM_r500": "Elevation",
    "PM2.5_r500": "PM2.5",
    "人口密度_r500": "Population density",
}
OUTNAME = {"sbp": "SBP", "dbp": "DBP", "map": "MAP", "fpg": "FPG",
           "是否高血压": "Hypertension", "是否高血糖": "Hyperglycaemia"}

# First four entries are the spatiotemporal greenness factors, the fifth is the
# conventional greenness exposure, the remainder are the environment features.
ORDER = ["绿度强度_r500", "绿度聚集度_GAI", "季节振幅_谐波_r500", "绿度年龄_r500",
         "绿度暴露_r500", "道路距离_主干道_r500", "夜光_r500", "DEM_r500",
         "PM2.5_r500", "人口密度_r500"]
C_SPAT, C_EXP, C_ENV = "#2E7D32", "#8BC34A", "#1E88E5"

PARAM_TXT = (
    "Random forest (RF): n_estimators=300, max_depth=30, max_features=1.0, "
    "min_samples_leaf=10, bootstrap=True, oob_score=True, random_state=42\n"
    "Extra trees (ET): n_estimators=300, max_depth=30, max_features=1.0, "
    "min_samples_leaf=10, bootstrap=False, random_state=42\n"
    "XGBoost (XGB): n_estimators=300, max_depth=6, learning_rate=0.03, "
    "subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42\n"
    "Control protocol: the seven individual confounders (age cubic polynomial, "
    "sex, BMI, smoking, alcohol, physical activity, diet regularity) are not\n"
    "                    forest features; they enter as an OLS offset, identical "
    "across models and outcomes. The forest sees only the ten greenness and\n"
    "                    environment features. Validation: 5-fold random CV "
    "(x3 repeats) and spatial-block CV (KMeans 10 blocks x 3 seeds = 30 folds)."
)


def color_of(feat):
    if feat == "绿度暴露_r500":
        return C_EXP
    if feat in ORDER[:4]:
        return C_SPAT
    return C_ENV


def panel_importance(ax, imp, met, outcome, tag):
    sub = imp[(imp[COL["qc"]] == "on") & (imp[COL["model"]] == "RF") &
              (imp[COL["outcome"]] == outcome)]
    sub = sub.set_index(COL["feature"])[COL["importance"]]
    sub = sub.reindex(ORDER).dropna().sort_values(ascending=True)

    y = np.arange(len(sub))
    ax.barh(y, sub.values, color=[color_of(f) for f in sub.index], height=0.68,
            edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[f] for f in sub.index], fontsize=8.6)
    for yi, v in zip(y, sub.values):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.7,
                color="#333333", zorder=4)
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_ylabel("Feature", fontsize=9)
    ax.set_xlim(0, max(0.30, float(sub.max()) * 1.45))
    ax.axvline(0.10, ls=(0, (3, 3)), lw=0.9, color="#9E9E9E", zorder=1)
    ax.set_title(f"({tag})", fontsize=10.5, fontweight="bold", pad=8, loc="left")
    ax.grid(axis="x", ls=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    r = met[(met[COL["outcome"]] == outcome) & (met[COL["model"]] == "RF")].iloc[0]
    is_bin = r[COL["otype"]] == "二分类"
    lines = []
    if is_bin:
        lines.append(f"AUC (in-sample) = {r[COL['in_sample']]:.3f}")
        lines.append(f"AUC (spatial CV) = {r[COL['cv']]:.3f}")
        lines.append(f"95% CI [{r[COL['cv_lo']]:.3f}, {r[COL['cv_hi']]:.3f}]")
        if COL["oob"] in met.columns and pd.notna(r.get(COL["oob"], np.nan)):
            lines.append(f"OOB AUC = {float(r[COL['oob']]):.3f}")
    else:
        lines.append(f"R2 (in-sample) = {r[COL['in_sample']]:.3f}")
        lines.append(f"R2 (spatial CV) = {r[COL['cv']]:.3f}")
        lines.append(f"95% CI [{r[COL['cv_lo']]:.3f}, {r[COL['cv_hi']]:.3f}]")
        lines.append(f"MAE = {r[COL['mae']]:.3f}")
        lines.append(f"RMSE = {r[COL['rmse']]:.3f}")
        if COL["oob"] in met.columns and pd.notna(r.get(COL["oob"], np.nan)):
            lines.append(f"OOB R2 = {float(r[COL['oob']]):.3f}")
    lines.append(f"Leakage = {r[COL['leak']]:+.3f}")

    bx0, bw, by0 = 0.60, 0.385, 0.30
    bh = 0.060 * len(lines) + 0.045
    ax.add_patch(FancyBboxPatch((bx0, by0), bw, bh,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                transform=ax.transAxes, fc="white",
                                ec="#9E9E9E", lw=0.9, zorder=5))
    ax.text(bx0 + bw / 2, by0 + bh / 2, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=7.4, style="italic",
            color="#222222", zorder=6, linespacing=1.42)


def legend_handles():
    """Shared colour legend for the three feature blocks."""
    return [Patch(fc=C_SPAT, label="Spatiotemporal greenness"),
            Patch(fc=C_EXP, label="Greenness exposure"),
            Patch(fc=C_ENV, label="Environment")]


def panel_perf(ax, met, outcomes, tag, ylab, is_auc=False):
    models = ["RF", "ET", "XGB"]
    cols = {"RF": "#2E7D32", "ET": "#FBC02D", "XGB": "#1E88E5"}
    x = np.arange(len(outcomes))
    w = 0.24
    allv, alli, alltop = [], [], []
    for i, mo in enumerate(models):
        vals, los, his, ins, rnd = [], [], [], [], []
        for oc in outcomes:
            r = met[(met[COL["outcome"]] == oc) & (met[COL["model"]] == mo)].iloc[0]
            vals.append(r[COL["cv"]])
            los.append(r[COL["cv"]] - r[COL["cv_lo"]])
            his.append(r[COL["cv_hi"]] - r[COL["cv"]])
            ins.append(r[COL["in_sample"]])
            rnd.append(r[COL["random_cv"]])
        allv += vals
        alli += ins
        off = (i - 1) * w
        ax.bar(x + off, vals, w, yerr=np.array([los, his]), capsize=2.5,
               color=cols[mo], alpha=0.92, label=mo, zorder=3,
               edgecolor="white", linewidth=0.5)
        ax.plot(x + off, ins, marker="^", ls="none", ms=3.8, color="#B71C1C", zorder=4)
        ax.plot(x + off, rnd, marker="o", ls="none", ms=3.2, mfc="none",
                mec="#444444", zorder=4)
        pad = 0.006 if is_auc else 0.012
        for xi, v, hi in zip(x + off, vals, his):
            ax.text(xi, v + hi + pad, f"{v:.3f}", ha="center", fontsize=6.0,
                    color="#333333", zorder=5, rotation=90)
            alltop.append(v + hi + (0.014 if is_auc else 0.030))

    ax.set_xticks(x)
    ax.set_xticklabels([OUTNAME.get(o, o) for o in outcomes], fontsize=8.8)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(f"({tag})", fontsize=10.5, fontweight="bold", pad=8, loc="left")
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(fc=cols[m], label=m) for m in models] +
              [Line2D([], [], marker="o", ls="none", ms=4, mfc="none",
                      mec="#444444", label="random CV"),
               Line2D([], [], marker="^", ls="none", ms=4.5, color="#B71C1C",
                      label="in-sample")],
              fontsize=6.9, loc="upper center", bbox_to_anchor=(0.5, 1.005),
              ncol=3, frameon=True, framealpha=0.95, columnspacing=1.0,
              handletextpad=0.5)
    lo = 0.0 if min(allv) < 0.30 else max(0.0, min(allv) - 0.09)
    ax.set_ylim(lo, max(max(alltop), max(alli)) * 1.06)


def main():
    met = pd.read_csv(METRICS_CSV)
    imp = pd.read_csv(IMPORT_CSV)
    met = met[met[COL["qc"]] == "on"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 11.0))
    panel_importance(axes[0, 0], imp, met, "map", "a")
    panel_importance(axes[0, 1], imp, met, "fpg", "b")
    panel_perf(axes[1, 0], met, ["sbp", "dbp", "map", "fpg"], "c",
               "Spatial-block CV R2")
    panel_perf(axes[1, 1], met, ["是否高血压", "是否高血糖"], "d",
               "Spatial-block CV ROC-AUC", is_auc=True)

    fig.tight_layout(rect=[0.005, 0.135, 0.995, 0.945])
    fig.legend(handles=legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=8.6, frameon=True,
               framealpha=0.95)
    fig.text(0.5, 0.018, PARAM_TXT, ha="center", va="bottom", fontsize=7.6,
             family="DejaVu Sans Mono", color="#222222", linespacing=1.55,
             bbox=dict(boxstyle="round,pad=0.55", fc="#F7F7F7", ec="#9E9E9E", lw=0.8))

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, "Figure_importance_performance")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".tiff", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", base)


if __name__ == "__main__":
    main()
