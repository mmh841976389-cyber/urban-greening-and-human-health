# -*- coding: utf-8 -*-
"""
Appendix figure: nested ablation of the greenness dimensions (spatial block CV).

Four settings:
  A  all ten factors (environment + spatiotemporal greenness + exposure)
  C  spatiotemporal greenness (4) + environment
  B  greenness exposure (1) + environment
  D  environment only (no greenness)

Continuous outcomes are reported as R2 and binary outcomes as ROC-AUC.
Values are the results produced by machine_learning/rf_nested_ablation.py.
Panel titles are limited to (a) and (b); the caption gives the full definition.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = os.environ.get("FIG_OUT", "output/figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "font.size": 9})

SETS = ["A", "C", "B", "D"]
SET_LABEL = {
    "A": "All 10 factors",
    "C": "Spatiotemporal greenness (4) + env",
    "B": "Greenness exposure (1) + env",
    "D": "Environment baseline (no greenness)",
}
COLORS = {"A": "#1B5E20", "C": "#43A047", "B": "#A5D6A7", "D": "#90A4AE"}

CONTINUOUS = {
    "SBP": {"A": 0.2395, "C": 0.1981, "B": 0.1581, "D": 0.1214},
    "DBP": {"A": 0.2070, "C": 0.1848, "B": 0.1586, "D": 0.0989},
    "MAP": {"A": 0.2299, "C": 0.1876, "B": 0.1488, "D": 0.1108},
    "FPG": {"A": 0.1061, "C": 0.0921, "B": 0.0926, "D": 0.0812},
}
BINARY = {
    "Hypertension": {"A": 0.7253, "C": 0.6727, "B": 0.6487, "D": 0.6217},
    "Hyperglycaemia": {"A": 0.7212, "C": 0.6738, "B": 0.6254, "D": 0.5795},
}


def panel(ax, data, title, ylab):
    """Grouped bars of the four ablation settings."""
    outcomes = list(data.keys())
    x = np.arange(len(outcomes))
    w = 0.20
    for i, s in enumerate(SETS):
        vals = [data[o][s] for o in outcomes]
        off = (i - 1.5) * w
        ax.bar(x + off, vals, w, color=COLORS[s], label=SET_LABEL[s],
               edgecolor="white", linewidth=0.5, zorder=3, alpha=0.95)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=5.6,
                    color="#333333", zorder=4, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=9.5)
    ax.set_ylabel(ylab, fontsize=9.8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, loc="left")
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(fc=COLORS[s], label=f"{s}: {SET_LABEL[s]}") for s in SETS],
              fontsize=7.2, loc="upper right", frameon=True, framealpha=0.95,
              borderpad=0.4, handlelength=1.4)
    ax.set_ylim(0.0, max(v for o in data.values() for v in o.values()) * 1.16)


def make_fig(path):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
    panel(axes[0], CONTINUOUS, "(a)", "R\u00b2")
    panel(axes[1], BINARY, "(b)", "ROC-AUC")
    fig.tight_layout(rect=[0.005, 0.035, 0.995, 0.99])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".tiff"), dpi=300,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(path.replace(".png", ".pdf"), dpi=300)
    fig.savefig(path.replace(".png", ".svg"), dpi=300)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    make_fig(os.path.join(OUT_DIR, "rf_nested_ablation.png"))
