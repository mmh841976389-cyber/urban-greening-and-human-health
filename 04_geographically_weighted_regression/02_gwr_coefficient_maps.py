# -*- coding: utf-8 -*-
"""
GWR coefficient maps (Figure 8) and the appendix local-significance map (Figure S5).

Reads gwr_results.pkl produced by gwr_pipeline.py.

Cartography:
  - the local coefficient field is interpolated by inverse-distance weighting
    (k = 12 nearest observations, weight 1/d^2) onto a regular grid;
  - the field, the significance contours and the sample dots are vector-clipped
    to the study-area outline, so nothing spills outside the city boundary;
  - the view is the study-area bounding box plus a small margin, so the whole
    city sits inside the panel;
  - colour scale is per outcome: +/- the 95th percentile of |beta| across factors.

Outputs: gwr_coefficients.png, gwr_local_significance.png
"""
import json
import math
import os
import pickle
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import unary_union
import shapely

warnings.filterwarnings("ignore")

RESULT_PKL = os.environ.get("GWR_PKL", "output/gwr/gwr_results.pkl")
DISTRICT_GEOJSON = os.environ.get("DISTRICT_GEOJSON", "data/study_area.geojson")
OUT_DIR = os.environ.get("GWR_OUT", "output/gwr")
os.makedirs(OUT_DIR, exist_ok=True)

SHOW_DISTRICTS = True   # thin internal district boundaries
MARGIN = 0.03           # view margin around the study-area outline
NX = 600                # grid resolution along the long axis
IDW_K = 12

# local equirectangular projection (must match gwr_pipeline.py)
LON0, LAT0 = 120.15, 30.27
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110540.0


def ll2xy(lon, lat):
    return (lon - LON0) * KX, (lat - LAT0) * KY


RES = pickle.load(open(RESULT_PKL, "rb"))
points = np.asarray(RES["points"], float)
study_area = RES["study_area"]
models = RES["models"]
factors = RES["factors"]
FACTOR_EN = RES["factor_en"]

OUTCOME_ORDER = ["SBP", "DBP", "MAP", "FPG", "HTN", "HBG"]
OUTCOME_LABEL = {"SBP": "SBP", "DBP": "DBP", "MAP": "MAP", "FPG": "FPG",
                 "HTN": "Hypertension", "HBG": "Hyperglycaemia"}
FACTOR_ORDER = factors
FACTOR_LABEL = [FACTOR_EN[f] for f in FACTOR_ORDER]

# study area as a single valid polygon
union = study_area.buffer(0)
if union.geom_type == "MultiPolygon":
    union = unary_union(list(union.geoms))
if union.geom_type == "MultiPolygon":
    union = max(union.geoms, key=lambda g: g.area)


def mpl_path(poly):
    """Matplotlib clip path (exterior plus interior holes) for a shapely polygon."""
    v = list(poly.exterior.coords)
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(v) - 2) + [MplPath.CLOSEPOLY]
    verts = list(v)
    for h in poly.interiors:
        hv = list(h.coords)
        verts += hv
        codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(hv) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(np.asarray(verts, float), codes)


CLIP = mpl_path(union)
outer_ring = np.asarray(union.exterior.coords, float)

# internal district boundaries (optional, subordinate to the outer outline)
district_rings = []
try:
    gj = json.load(open(DISTRICT_GEOJSON, encoding="utf-8"))
    for ft in (gj.get("features") or [gj]):
        g = shape(ft.get("geometry") or ft).buffer(0)
        parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for p in parts:
            district_rings.append(np.asarray([ll2xy(x, y) for x, y in p.exterior.coords], float))
except Exception as e:
    print("district outlines unavailable:", e)

# view extent
bx0, by0, bx1, by1 = union.bounds
mw, mh = (bx1 - bx0) * MARGIN, (by1 - by0) * MARGIN
vx0, vx1 = bx0 - mw, bx1 + mw
vy0, vy1 = by0 - mh, by1 + mh
ar = (vx1 - vx0) / (vy1 - vy0)

# interpolation grid over the study area
NY = max(2, int(round(NX * (by1 - by0) / (bx1 - bx0))))
GX, GY = np.meshgrid(np.linspace(bx0, bx1, NX), np.linspace(by0, by1, NY))
grid = np.c_[GX.ravel(), GY.ravel()]
inside = np.asarray(shapely.contains(union, shapely.points(grid))).reshape(GX.shape)

tree = cKDTree(points)
d12, i12 = tree.query(grid, k=IDW_K)
w12 = 1.0 / (d12 ** 2 + 1e-6)
w12 /= w12.sum(axis=1, keepdims=True)
_, i1 = tree.query(grid, k=1)

# per-outcome symmetric colour scale
out_scale = {}
for o in OUTCOME_ORDER:
    allb = np.concatenate([np.abs(np.nan_to_num(models[(o, f)]["beta1"])) for f in FACTOR_ORDER])
    out_scale[o] = float(np.nanpercentile(allb, 95)) or 1e-6

cmap = plt.cm.coolwarm


def coloured_field(o, f, mx):
    """IDW-interpolated coefficient field rendered with a diverging colour map."""
    vals = np.nan_to_num(models[(o, f)]["beta1"])
    cf = (vals[i12] * w12).sum(axis=1).reshape(GX.shape)
    rgba = cmap(TwoSlopeNorm(vmin=-mx, vcenter=0.0, vmax=mx)(np.clip(cf, -mx, mx)))
    rgba[~inside, 3] = 0.0
    return rgba


def sig_grid(o, f):
    return (models[(o, f)]["p"][i1] <= 0.05).reshape(GX.shape)


def draw_boundaries(ax):
    if SHOW_DISTRICTS:
        for ring in district_rings:
            ax.plot(ring[:, 0], ring[:, 1], color="#5a5a5a", lw=0.45, alpha=0.65,
                    zorder=5, solid_joinstyle="round")
    ax.plot(outer_ring[:, 0], outer_ring[:, 1], color="#0d0d0d", lw=1.5,
            zorder=6, solid_joinstyle="round")


def clip_artist(art, ax):
    """Clip an artist (or its collections) to the study-area outline."""
    try:
        art.set_clip_path(CLIP, ax.transData)
    except Exception:
        for c in getattr(art, "collections", []):
            c.set_clip_path(CLIP, ax.transData)


def fit_check(fig, texts, tag):
    """Report any text that extends beyond the figure canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    bad = 0
    for t in texts:
        bb = t.get_window_extent(renderer=r)
        if bb.x0 < 0 or bb.x1 > W or bb.y0 < 0 or bb.y1 > H:
            bad += 1
            print(f"  [WARN] text outside canvas ({tag}): {t.get_text()[:60]!r}")
    print(f"  [{tag}] text-fit: {len(texts)} texts, {bad} outside")
    return bad


# ------------------------------------------------------------------ Figure 8
L, R, T, B = 0.12, 0.93, 0.945, 0.05
figW = 15.0
figH = figW * (R - L) * 6.0 / ((T - B) * 5.0 * ar)
fig, axes = plt.subplots(6, 5, figsize=(figW, figH), squeeze=False)
label_texts = []

for i, o in enumerate(OUTCOME_ORDER):
    for j, f in enumerate(FACTOR_ORDER):
        ax = axes[i][j]
        mx = out_scale[o]
        im = ax.imshow(coloured_field(o, f, mx), extent=(bx0, bx1, by0, by1),
                       origin="lower", interpolation="bilinear", zorder=1)
        clip_artist(im, ax)

        sga = np.ma.masked_where(~inside, sig_grid(o, f).astype(float))
        if np.ma.count(sga) and np.ma.max(sga) > 0:
            cs = ax.contour(GX, GY, sga, levels=[0.5], colors=["#101010"],
                            linewidths=0.6, alpha=0.6, zorder=7)
            clip_artist(cs, ax)

        sc = ax.scatter(points[:, 0], points[:, 1], s=0.35, c="0.25", alpha=0.12,
                        linewidths=0, zorder=4)
        clip_artist(sc, ax)
        draw_boundaries(ax)

        ax.set_xlim(vx0, vx1)
        ax.set_ylim(vy0, vy1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

        if j == 0:
            label_texts.append(ax.set_ylabel(OUTCOME_LABEL[o], fontsize=10.5, fontweight="bold",
                                             rotation=0, labelpad=30, va="center", ha="right"))
        if i == 0:
            ax.set_title(FACTOR_LABEL[j], fontsize=11, fontweight="bold", pad=7)

plt.subplots_adjust(left=L, right=R, top=T, bottom=B, wspace=0.03, hspace=0.04)

# one vertical colour bar per row (per outcome)
cbar_texts = []
for i, o in enumerate(OUTCOME_ORDER):
    pos0 = axes[i][0].get_position()
    pos4 = axes[i][4].get_position()
    cax = fig.add_axes([pos4.x1 + 0.005, pos0.y0, 0.010, pos4.y1 - pos0.y0])
    mx = out_scale[o]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(vmin=-mx, vcenter=0, vmax=mx))
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=8.0, length=2, pad=1.5)
    cb.set_ticks([-mx * 0.82, 0, mx * 0.82])
    cb.set_ticklabels([f"{-mx:.2f}", "0", f"{mx:.2f}"])
    cb.outline.set_linewidth(0.5)
    cbar_texts += list(cb.ax.get_yticklabels())

fit_check(fig, label_texts + cbar_texts, "main")
p1 = os.path.join(OUT_DIR, "gwr_coefficients.png")
plt.savefig(p1, dpi=300, facecolor="white")
plt.close(fig)
print("saved", p1)

# ------------------------------------------------------------------ Figure S5
fig2, axes2 = plt.subplots(1, 5, figsize=(16, 4.0), squeeze=False)
plt.subplots_adjust(left=0.025, right=0.885, top=0.92, bottom=0.06, wspace=0.06)
cmap2 = plt.cm.YlOrRd
titles2 = []

for j, f in enumerate(FACTOR_ORDER):
    ax = axes2[0][j]
    frac = np.zeros(grid.shape[0])
    for o in OUTCOME_ORDER:
        frac += (models[(o, f)]["p"][i1] <= 0.05).astype(float)
    frac /= len(OUTCOME_ORDER)

    rgba2 = cmap2(Normalize(0, 1)(frac.reshape(GX.shape)))
    rgba2[~inside, 3] = 0.0
    im2 = ax.imshow(rgba2, extent=(bx0, bx1, by0, by1), origin="lower",
                    interpolation="bilinear", zorder=1)
    clip_artist(im2, ax)
    sc2 = ax.scatter(points[:, 0], points[:, 1], s=0.35, c="0.25", alpha=0.12,
                     linewidths=0, zorder=4)
    clip_artist(sc2, ax)
    draw_boundaries(ax)

    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vy0, vy1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    titles2.append(ax.set_title(f"({'abcde'[j]}) {FACTOR_LABEL[j]}",
                                fontsize=10.0, fontweight="bold", pad=6))

sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=Normalize(0, 1))
cax2 = fig2.add_axes([0.905, 0.16, 0.011, 0.60])
cb2 = fig2.colorbar(sm2, cax=cax2, orientation="vertical")
cb2.set_label("Fraction of the six outcomes with a\nlocally significant (p<0.05) coefficient",
              fontsize=8.0)
cb2.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cb2.ax.tick_params(labelsize=8, length=2)
cb2.outline.set_linewidth(0.5)
fit_check(fig2, titles2 + list(cb2.ax.get_yticklabels()) + [cb2.ax.yaxis.label], "appendix")

p2 = os.path.join(OUT_DIR, "gwr_local_significance.png")
plt.savefig(p2, dpi=300, facecolor="white")
plt.close(fig2)
print("saved", p2)
