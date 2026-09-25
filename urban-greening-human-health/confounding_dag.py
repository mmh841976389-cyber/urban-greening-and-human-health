# -*- coding: utf-8 -*-
"""
Conceptual confounding framework drawn as a directed acyclic graph (Figure S4).

Nodes:
  A individual-level confounders (age, sex, BMI, smoking, alcohol, PA, diet)
  B natural / physical environment (PM2.5, elevation)
  C socio-economic environment (road proximity, nighttime light, population density)
  D residential selection
  E spatial greenness (intensity, aggregation index)
  F temporal greenness (seasonal amplitude, greenness age)
  G greenness exposure (1 km residential vegetation percentage)
  H cardiometabolic outcome

Arrow colours:
  grey    confounding (shared cause, open backdoor path)
  blue    environment constrains greenness, or direct environmental effect
  purple  residential selection (individual and socio-economic -> selection -> greenness)
  green   effect of interest (greenness -> outcome)

Adjustment set: A + B + C. Residential selection (D) is a collider and is not
conditioned on. No title or caption text is drawn inside the figure.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.environ.get("FIG_OUT", "output/figures")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1320, 900
fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=300)
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 9.0)
ax.axis("off")

COLORS = {
    "confound": "#6b6b6b",
    "env": "#1565c0",
    "select": "#6a1b9a",
    "effect": "#2e7d32",
}

BW, BH = 2.0, 1.0


def box(cx, cy, fc, ec, title, lines, tcol, tsize=12.5, lsize=9.6):
    """Draw one node as a rounded box with a bold title and detail lines."""
    x, y = cx - BW / 2, cy - BH / 2
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0,rounding_size=8",
                                linewidth=2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(cx, cy - 0.14, title, ha="center", va="center", fontsize=tsize,
            fontweight="bold", color=tcol, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(cx, cy + 0.16 + i * 0.22, ln, ha="center", va="center",
                fontsize=lsize, color="#333333", zorder=3)


def arrow(x1, y1, x2, y2, col, width=1.2, rad=0.0, dash=None, shrink=22):
    kw = dict(arrowstyle="-|>", connectionstyle=f"arc3,rad={rad}",
              mutation_scale=13, linewidth=width, color=col, zorder=1,
              shrinkA=shrink, shrinkB=shrink)
    if dash:
        kw["linestyle"] = (0, (4, 3))
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))


def edge_label(x1, y1, x2, y2, text, col, fs=9.0, dx=0.0, dy=0.0):
    mx, my = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
    ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=col,
            fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


# node centres
A = (1.2, 6.3)
B = (1.2, 4.0)
C = (1.2, 1.7)
D = (4.6, 1.7)
E = (7.8, 6.3)
F = (7.8, 4.0)
G = (7.8, 1.7)
Hh = (11.2, 4.0)

box(*A, "#fff3e0", "#b26a00", "Individual-level\nconfounders",
    ["age, sex, BMI, smoking,", "alcohol, PA, diet"], "#7a4500")
box(*B, "#e3f2fd", "#1565c0", "Natural / physical\nenvironment",
    ["PM2.5, elevation"], "#0d47a1")
box(*C, "#e3f2fd", "#1565c0", "Socio-economic\nenvironment",
    ["road proximity, nighttime", "light, population density"], "#0d47a1")
box(*D, "#f3e5f5", "#6a1b9a", "Residential\nselection", [], "#4a148c", tsize=12.0)
box(*E, "#e8f5e9", "#2e7d32", "Spatial greenness",
    ["intensity, aggregation index"], "#1b5e20")
box(*F, "#e8f5e9", "#2e7d32", "Temporal greenness",
    ["seasonal amplitude,", "greenness age"], "#1b5e20")
box(*G, "#e8f5e9", "#2e7d32", "Greenness exposure",
    ["1-km residential veg. %"], "#1b5e20")
box(*Hh, "#ffebee", "#c62828", "Cardiometabolic\noutcome",
    ["SBP, DBP, MAP, FPG,", "hypertension, hyperglycaemia"], "#b71c1c")

# confounding (grey)
for tgt in (E, F, G):
    arrow(A[0], A[1], tgt[0], tgt[1], COLORS["confound"], rad=0.12)
arrow(A[0], A[1], Hh[0], Hh[1], COLORS["confound"], rad=0.12)
edge_label(A[0], A[1], E[0], E[1], "confounding", COLORS["confound"], dy=-0.32)

# environment (blue)
for src in (B, C):
    for tgt in (E, F, G):
        arrow(src[0], src[1], tgt[0], tgt[1], COLORS["env"], rad=0.10)
    arrow(src[0], src[1], Hh[0], Hh[1], COLORS["env"], rad=0.06)
edge_label(B[0], B[1], F[0], F[1], "environment", COLORS["env"], dy=-0.30)
edge_label(B[0], B[1], Hh[0], Hh[1], "direct env. effect", COLORS["env"], dy=-0.28)

# residential selection (purple, dashed into the collider)
arrow(A[0], A[1], D[0], D[1], COLORS["select"], width=1.5, rad=0.10, dash=True)
arrow(C[0], C[1], D[0], D[1], COLORS["select"], width=1.5, rad=0.10, dash=True)
edge_label(C[0], C[1], D[0], D[1], "residential selection", COLORS["select"], dy=-0.32)
for tgt in (E, F, G):
    arrow(D[0], D[1], tgt[0], tgt[1], COLORS["select"], width=1.5, rad=0.10)
edge_label(D[0], D[1], F[0], F[1], "determines exposure", COLORS["select"], dy=-0.30)

# effect of interest (green)
for tgt in (E, F, G):
    arrow(tgt[0], tgt[1], Hh[0], Hh[1], COLORS["effect"], width=2.1, rad=0.10)
edge_label(E[0], E[1], Hh[0], Hh[1], "effect of interest", COLORS["effect"], dy=0.26)
edge_label(F[0], F[1], Hh[0], Hh[1], "effect of interest", COLORS["effect"], dy=-0.30)
edge_label(G[0], G[1], Hh[0], Hh[1], "effect of interest", COLORS["effect"], dy=0.26)

handles = [
    Line2D([0], [0], color=COLORS["confound"], lw=1.6,
           label="Confounding (shared cause; open backdoor)"),
    Line2D([0], [0], color=COLORS["env"], lw=1.6,
           label="Environment constrains greenness / direct env. effect"),
    Line2D([0], [0], color=COLORS["select"], lw=1.6, ls="--",
           label="Residential selection (individual & socio-economic \u2192 selection \u2192 greenness)"),
    Line2D([0], [0], color=COLORS["effect"], lw=2.4,
           label="Effect of interest (greenness \u2192 outcome)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06),
          ncol=2, frameon=False, fontsize=9.5, handlelength=2.4,
          columnspacing=1.6, borderpad=0.4)

out = os.path.join(OUT_DIR, "confounding_dag.png")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.12)
fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.12)
fig.savefig(out.replace(".png", ".tiff"), dpi=300,
            pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight", pad_inches=0.12)
fig.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.12)
print("saved", out)
