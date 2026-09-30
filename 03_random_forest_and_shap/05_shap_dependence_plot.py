# -*- coding: utf-8 -*-
"""
Figure: SHAP dependence panels for the random forest fitted under the control
(offset) protocol. Rows are outcomes, columns are the four highest-importance
features of that outcome.

Each panel shows the SHAP value of every participant against the feature value,
with a binned mean trend line and its 95% CI band. The x range is trimmed to the
dense 2nd-98th percentile interval so that sparse tails do not distort the fit.

Inputs
  SHAP_LONG_CSV   long-format SHAP table with columns: outcome, feature, x, shap
  OUT_DIR         output directory for png / tiff / pdf / svg
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

SHAP_CSV = os.environ.get("SHAP_LONG_CSV", "SHAP_RF_6因变量_long.csv")
OUT_DIR = os.environ.get("OUT_DIR", ".")

# Source table uses Chinese column names.
C_OUT, C_FEAT = "结局", "特征"

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
OUTCOMES = ["sbp", "dbp", "map", "fpg", "是否高血压", "是否高血糖"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"
C_DOT, C_LINE, C_BAND = "#6FA8DC", "#B71C1C", "#F4A6A6"

# Display range (2nd-98th percentile) per feature; outliers are excluded from
# both the scatter and the trend fit.
XLIM = {
    "PM2.5_r500": (38.1, 50.7),
    "道路距离_主干道_r500": (64.0, 322.0),
    "绿度年龄_r500": (7.4, 16.5),
    "季节振幅_谐波_r500": (0.044, 0.119),
    "DEM_r500": (8.6, 35.6),
    "绿度聚集度_GAI": (-3.8, 1.85),
}
XFORMAT = {
    "PM2.5_r500": "{:.0f}", "道路距离_主干道_r500": "{:.0f}",
    "绿度年龄_r500": "{:.0f}", "季节振幅_谐波_r500": "{:.2f}",
    "DEM_r500": "{:.0f}", "绿度聚集度_GAI": "{:.1f}",
}
XSTEP = {
    "PM2.5_r500": 3.0, "道路距离_主干道_r500": 60.0, "绿度年龄_r500": 2.0,
    "季节振幅_谐波_r500": 0.02, "DEM_r500": 6.0, "绿度聚集度_GAI": 1.5,
}


def trend(x, s, nb=20):
    """Binned mean and 95% CI, smoothed with a 3-point moving average."""
    qs = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    cx, cy, clo, chi = [], [], [], []
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        m = (x >= lo) & (x <= hi) if i == 0 else (x > lo) & (x <= hi)
        if m.sum() < 6:
            continue
        mu = s[m].mean()
        se = s[m].std(ddof=1) / np.sqrt(m.sum())
        cx.append(float(np.median(x[m])))
        cy.append(mu)
        clo.append(mu - 1.96 * se)
        chi.append(mu + 1.96 * se)
    cx, cy, clo, chi = map(np.asarray, (cx, cy, clo, chi))
    if len(cx) >= 5:
        for arr in (cy, clo, chi):
            arr[:] = pd.Series(arr).rolling(3, center=True, min_periods=1).mean().values
    return cx, cy, clo, chi


def panel(ax, x, s, feat, letter):
    xlo, xhi = XLIM[feat]
    m = (x >= xlo) & (x <= xhi)
    x, s = x[m], s[m]
    ax.scatter(x, s, s=3.4, color=C_DOT, alpha=0.34, linewidths=0,
               rasterized=True, zorder=2)
    cx, cy, clo, chi = trend(x, s)
    if len(cx) >= 2:
        ax.fill_between(cx, clo, chi, color=C_BAND, alpha=0.42, lw=0, zorder=1)
        ax.plot(cx, cy, color=C_LINE, lw=1.5, zorder=3, solid_capstyle="round")
        ax.set_xlim(cx.min(), cx.max())
    else:
        ax.set_xlim(xlo, xhi)

    ylo, yhi = np.percentile(s, [0.5, 99.5])
    pad = 0.06 * (yhi - ylo + 1e-12)
    ax.set_ylim(ylo - pad, yhi + pad)

    yloc = MaxNLocator(nbins=4, steps=[1, 2, 5, 10])
    ax.yaxis.set_major_locator(yloc)
    tk = np.unique(yloc.tick_values(*ax.get_ylim()))
    step = float(np.min(np.diff(tk))) if len(tk) > 1 else 0.1
    dec = int(max(0, np.ceil(-np.log10(step))))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.{dec}f}"))
    ax.xaxis.set_major_locator(MultipleLocator(XSTEP[feat]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: XFORMAT[feat].format(v)))

    ax.tick_params(labelsize=9.0, length=3.2, width=0.8, pad=2.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.85)
    ax.text(0.026, 0.958, f"({letter})", transform=ax.transAxes, ha="left",
            va="top", fontsize=15.0, fontweight="bold")
    ax.set_xlabel(NAME[feat], fontsize=11.2, fontweight="bold", labelpad=2.5)
    ax.grid(False)


def main():
    d = pd.read_csv(SHAP_CSV)
    ncol = 4
    fig, axes = plt.subplots(len(OUTCOMES), ncol, figsize=(13.6, 13.3),
                             squeeze=False)
    fig.subplots_adjust(left=0.158, right=0.995, top=0.985, bottom=0.045,
                        wspace=0.34, hspace=0.46)
    k = 0
    for r, oc in enumerate(OUTCOMES):
        sub = d[d[C_OUT] == oc]
        feats = list(dict.fromkeys(sub[C_FEAT]))[:ncol]
        for c in range(ncol):
            ax = axes[r, c]
            if c < len(feats):
                ss = sub[sub[C_FEAT] == feats[c]]
                panel(ax, ss["x"].values, ss["shap"].values, feats[c], LETTERS[k])
                k += 1
            else:
                ax.axis("off")
        p = axes[r, 0].get_position()
        fig.text(0.146, (p.y0 + p.y1) / 2.0, OUTNAME[oc], ha="right",
                 va="center", fontsize=13.0, fontweight="bold")

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, "Figure_SHAP_dependence")
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
