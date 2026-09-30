# -*- coding: utf-8 -*-
"""
Appendix diagnostic panel for the GWR analysis (Figure S6).

Four panels:
  (a) AICc bandwidth-selection curves, one panel per greenness factor;
  (b) per-model diagnostics heatmap (bandwidth %, AICc, residual Moran's I,
      median local SE, share of locally significant locations);
  (c) Pearson correlation matrix of the greenness factors;
  (d) variance inflation factors of the greenness factors.

No figure title is drawn; the caption is added in the manuscript.
"""
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

RESULT_PKL = os.environ.get("GWR_PKL", "output/gwr/gwr_results.pkl")
OUT_DIR = os.environ.get("GWR_OUT", "output/gwr")
os.makedirs(OUT_DIR, exist_ok=True)

R = pickle.load(open(RESULT_PKL, "rb"))
models = R["models"]
C = R["green_corr"]
FACTOR_ORDER = R["factors"]
n = R["n"]
FAC_EN = R["factor_en"]
OUT_EN = R["outcome_en"]
ORDER = ["SBP", "DBP", "MAP", "FPG", "HTN", "HBG"]

OCOL = dict(zip(ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]))
FCOL = dict(zip(FACTOR_ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))

# candidate bandwidths are identical across models
kk, _ = zip(*models[(ORDER[0], FACTOR_ORDER[0])]["aicc_curve"])
K = np.array(sorted(kk))

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})

fig = plt.figure(figsize=(15.5, 11.0))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.9], hspace=0.28, left=0.105)

# ------------------------------------------------------------------ (a)
gsA = gridspec.GridSpecFromSubplotSpec(1, len(FACTOR_ORDER), subplot_spec=gs[0], wspace=0.28)
ax_top0 = None
for fi, f in enumerate(FACTOR_ORDER):
    ax = fig.add_subplot(gsA[fi])
    for o in ORDER:
        curve = dict(models[(o, f)]["aicc_curve"])
        ks = np.array(sorted(curve.keys()))
        aic = np.array([curve[k] for k in ks])
        aic = aic - aic.min()          # each outcome shifted to its own minimum
        sel = models[(o, f)]["k"]
        ax.plot(ks / n * 100.0, aic, color=OCOL[o], lw=1.1, alpha=0.85)
        ax.scatter([sel / n * 100.0], [0.0], color=OCOL[o], s=16, zorder=6,
                   edgecolor="k", lw=0.4)
    ax.set_title(FAC_EN[f], fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.4)
    if fi == 0:
        ax.set_ylabel("AICc \u2212 min (per outcome)", fontsize=8)
        ax_top0 = ax
    if fi == len(FACTOR_ORDER) // 2:
        ax.set_xlabel("Bandwidth (% of n)", fontsize=8)
    if fi == len(FACTOR_ORDER) - 1:
        ax.legend([OUT_EN[o] for o in ORDER], loc="upper right", fontsize=5.5,
                  framealpha=0.8, ncol=1)

# ------------------------------------------------------------------ bottom row
gsB = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1],
                                       wspace=0.34, width_ratios=[1.25, 1.0, 0.85])

# (b) per-model diagnostics
axC = fig.add_subplot(gsB[0])
col_names = ["BW %", "AICc", "Moran I", "Med SE", "% sig"]
cvals, clabs = [], []
for f in FACTOR_ORDER:
    for o in ORDER:
        m = models[(o, f)]
        cvals.append([m["k"] / n * 100.0, m["aicc"], m["moran"], m["med_se"], m["pct_sig"]])
        clabs.append(FAC_EN[f].split()[0] + "|" + OUT_EN[o].split()[0])
cvals = np.array(cvals, dtype=float)
norm = (cvals - cvals.min(0)) / (cvals.max(0) - cvals.min(0) + 1e-12)
imC = axC.imshow(norm, aspect="auto", cmap="Blues")
fmt = {0: "{:.1f}", 1: "{:.0f}", 2: "{:.3f}", 3: "{:.3f}", 4: "{:.1f}"}
for i in range(cvals.shape[0]):
    for j in range(cvals.shape[1]):
        axC.text(j, i, fmt[j].format(cvals[i, j]), ha="center", va="center",
                 fontsize=4.8, color="#111")
axC.set_xticks(range(len(col_names)))
axC.set_xticklabels(col_names, fontsize=6.3, rotation=38, ha="right")
axC.set_yticks(range(len(clabs)))
axC.set_yticklabels(clabs, fontsize=5.6)
axC.set_title("Per-model diagnostics", fontsize=9.5, fontweight="bold")
cbC = fig.colorbar(imC, ax=axC, fraction=0.025, pad=0.02)
cbC.ax.tick_params(labelsize=5.5)

# (c) correlation matrix
axD = fig.add_subplot(gsB[1])
imD = axD.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
short = []
for f in FACTOR_ORDER:
    parts = FAC_EN[f].split()
    short.append(FAC_EN[f] if len(parts) == 1 else parts[0] + " " + parts[1])
axD.set_xticks(range(5))
axD.set_xticklabels(short, fontsize=6.0, rotation=40, ha="right")
axD.set_yticks(range(5))
axD.set_yticklabels(short, fontsize=6.0)
for i in range(5):
    for j in range(5):
        axD.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="#111")
axD.set_title("Greenness factor correlation", fontsize=9.5, fontweight="bold")

# (d) variance inflation factors
axE = fig.add_subplot(gsB[2])
vif = np.diag(np.linalg.inv(C))
axE.bar(range(5), vif, color=[FCOL[f] for f in FACTOR_ORDER], width=0.62)
axE.axhline(5.0, color="#c0392b", ls="--", lw=0.9)
axE.set_xticks(range(5))
axE.set_xticklabels(short, fontsize=6.0, rotation=40, ha="right")
for i, v in enumerate(vif):
    axE.text(i, v + 0.08, f"{v:.1f}", ha="center", fontsize=6.6)
axE.set_ylabel("VIF", fontsize=8)
axE.set_title("Multicollinearity (VIF)", fontsize=9.5, fontweight="bold")
axE.text(0.5, -0.16, "dashed line = concern threshold 5",
         transform=axE.transAxes, ha="center", fontsize=6.5, style="italic", color="#555")


def place_letter(ref_ax, letter):
    ref_ax.text(0.0, 1.02, letter, transform=ref_ax.transAxes,
                ha="left", va="bottom", fontsize=11, fontweight="bold")


place_letter(ax_top0, "(a)")
place_letter(axC, "(b)")
place_letter(axD, "(c)")
place_letter(axE, "(d)")

out_png = os.path.join(OUT_DIR, "gwr_diagnostics.png")
plt.savefig(out_png, dpi=300, facecolor="white")
plt.close(fig)
print("saved", out_png)
