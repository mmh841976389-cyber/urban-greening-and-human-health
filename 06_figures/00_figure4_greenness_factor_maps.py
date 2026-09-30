# -*- coding: utf-8 -*-
"""
Figure: spatial distribution of the five greenness factors and land cover.

Layout: 2 rows x 3 columns.
  (a) greenness intensity (mean growing-season NDVI)
  (b) aggregation index (GAI)
  (c) seasonal amplitude
  (d) greenness age (residential-block construction age, masked to built-up areas)
  (e) greenness exposure (percentage of vegetated cells in a 1 km window)
  (f) land cover classification

Each continuous panel carries a nine-class discrete colour ramp with a vertical
colour bar, a north arrow and a scale bar. Panels (a)-(e) are winsorised at the
2.5th and 97.5th percentiles to suppress extreme values.

Inputs (environment variables)
  FACTOR_FACTOR_TIF_DIR  directory holding the per-factor GeoTIFF inputs
  GAI_TIF         GAI raster (aggregation index)
  LANDCOVER_TIF   land cover raster (any projection; reprojected to the reference grid)
  OUT_DIR         output directory
"""
import os
import math
import numpy as np
import rasterio
import rasterio.warp
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                               BoundaryNorm)
from matplotlib.patches import Rectangle, Circle, Polygon

FACTOR_TIF_DIR = os.environ.get("FACTOR_FACTOR_TIF_DIR", r"C:\Users\mmh\Desktop\数据\各因子tif")
GAI_TIF = os.environ.get("GAI_TIF", "")
LANDCOVER_TIF = os.environ.get("LANDCOVER_TIF", r"C:\Users\mmh\Desktop\数据\CLCD_v01_2021_albert_zhejiang.tif")
OUT_DIR = os.environ.get("OUT_DIR", ".")

REF_TIF = os.path.join(FACTOR_TIF_DIR, "平均绿度", "平均绿度_裁剪后_杭州中心城区.tif")
INTENSITY_TIF = REF_TIF
AMPLITUDE_TIF = os.path.join(FACTOR_TIF_DIR, "绿度季节变化",
                             "Landsat30m_Seasonality_2000_2020_裁剪灰度_杭州中心城区.tif")
EXPOSURE_TIF = os.path.join(FACTOR_TIF_DIR, "平均绿度", "平均绿度_33x33计数_裁剪_杭州中心城区.tif")
AGE_TIF = os.path.join(FACTOR_TIF_DIR, "小区建成时间", "小区建成时间_处理后_杭州中心城区.tif")
AGE_FILTER_TIF = os.path.join(FACTOR_TIF_DIR, "小区建成时间", "小区建成时间_裁剪过滤.tif")

# Nine-class diverging ramp, low to high.
RB = ["#0571B0", "#4798C5", "#92C5DE", "#C4DEEA", "#F7F7F7",
      "#F5CBB8", "#F4A582", "#DE4D4E", "#CA0020"]
RAMP = LinearSegmentedColormap.from_list("RdBu9", RB)
RAMP.set_bad("white")

LC = {1: ("Cropland", "#FAE39C"), 2: ("Forest", "#446F33"), 3: ("Shrub", "#33A02C"),
      4: ("Grassland", "#ABD37B"), 5: ("Water", "#1E69B4"), 6: ("Snow", "#A6CEE3"),
      7: ("Barren", "#CFBDA3"), 8: ("Impervious", "#A92F6C")}


def read(path, masked=True):
    """Read band 1 as float, replacing nodata sentinels with NaN."""
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float64")
        nd = ds.nodata
    if masked:
        for v in ([nd] if nd is not None else []) + [-9999.0, 36.0]:
            a = np.where(np.isclose(a, v), np.nan, a)
        a = np.where(np.isfinite(a), a, np.nan)
    return a


def reproject_to_ref(data, src_path, ref_path, resampling):
    """Resample a source raster onto the reference grid."""
    with rasterio.open(src_path) as s:
        st, sc = s.transform, s.crs
    with rasterio.open(ref_path) as r:
        prof = r.profile.copy()
    prof.update(dtype="float64", count=1, nodata=np.nan)
    dst = np.full((prof["height"], prof["width"]), np.nan)
    rasterio.warp.reproject(data.astype("float64"), dst,
                            src_transform=st, src_crs=sc,
                            dst_transform=prof["transform"], dst_crs=prof["crs"],
                            resampling=resampling, src_nodata=np.nan,
                            dst_nodata=np.nan)
    return dst


def winsorize(a, lo=0.025, hi=0.975):
    """Clip to the given percentiles, keeping NaN as NaN."""
    m = np.isfinite(a)
    if m.sum() == 0:
        return a
    pl, ph = np.nanpercentile(a, [lo * 100, hi * 100])
    return np.where(m, np.clip(a, pl, ph), a)


def quantile_boundaries(a, n=9):
    """Monotone n+1 quantile boundaries used as discrete colour breaks."""
    qs = np.nanpercentile(a[np.isfinite(a)], np.linspace(0.0, 100.0, n + 1))
    for i in range(1, n + 1):
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + 1e-6
    return [float(v) for v in qs]


def draw_north(ax, cx=0.930, cy=0.890, r=0.026):
    """North arrow with a compass rose."""
    ax.add_patch(Circle((cx, cy), r * 1.30, transform=ax.transAxes, facecolor="none",
                        edgecolor="black", linewidth=1.4, zorder=6))
    ax.add_patch(Circle((cx, cy), r * 1.08, transform=ax.transAxes, facecolor="none",
                        edgecolor="black", linewidth=0.7, zorder=6))
    half, h = r * 0.30, r * 2.55
    tip = (cx, cy + h)
    ax.add_patch(Polygon([tip, (cx - half, cy - r * 0.35), (cx, cy - r * 0.15)],
                         closed=True, transform=ax.transAxes, facecolor="white",
                         edgecolor="black", linewidth=0.9, zorder=7))
    ax.add_patch(Polygon([tip, (cx + half, cy - r * 0.35), (cx, cy - r * 0.15)],
                         closed=True, transform=ax.transAxes, facecolor="#dcdcdc",
                         edgecolor="black", linewidth=0.9, hatch="....", zorder=7))
    for sx in (-1, 1):
        ax.add_patch(Polygon([(cx + sx * r * 1.15, cy - r * 1.45),
                              (cx + sx * r * 0.42, cy - r * 0.05),
                              (cx + sx * r * 0.10, cy - r * 0.30)], closed=True,
                             transform=ax.transAxes, facecolor="#dcdcdc",
                             edgecolor="black", linewidth=0.9, hatch="....", zorder=7))


def draw_scalebar(ax, extent, km=10.0, x0=0.695, y0=0.048, h=0.026):
    """Two-segment scale bar sized from the panel longitude span."""
    mlon = 111.32 * math.cos(math.radians((extent[2] + extent[3]) / 2.0))
    seg = (km / ((extent[1] - extent[0]) * mlon)) / 2.0
    for i, fc in enumerate(("black", "white")):
        ax.add_patch(Rectangle((x0 + i * seg, y0), seg, h, transform=ax.transAxes,
                               facecolor=fc, edgecolor="black", linewidth=0.9, zorder=6))
    for i, lab in enumerate(["0", f"{int(km / 2)}", f"{int(km)} km"]):
        ha = "left" if i == 0 else ("right" if i == 2 else "center")
        ax.text(x0 + i * seg, y0 + h * 1.6, lab, transform=ax.transAxes,
                fontsize=6.9, fontweight="bold", ha=ha, va="bottom", zorder=7)


def draw_vlegend(ax, lo, hi, unit, fmt, vlabels=None):
    """Vertical nine-class colour bar with five tick labels."""
    s = (hi - lo) / 9.0
    x0, w, y0, hh = 0.020, 0.046, 0.075, 0.345
    bh = hh / 9.0
    for j in range(9):
        ax.add_patch(Rectangle((x0, y0 + (8 - j) * bh), w, bh, transform=ax.transAxes,
                               facecolor=RB[8 - j], edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((x0, y0), w, hh, transform=ax.transAxes, facecolor="none",
                           edgecolor="black", linewidth=0.8, zorder=7))
    ax.text(x0 - 0.003, y0 + hh + 0.030, unit, transform=ax.transAxes,
            fontsize=8.8, ha="left", va="bottom", zorder=7)
    for i, k in enumerate((1, 3, 5, 7, 9)):
        yy = y0 + hh * (9 - k) / 9.0
        ax.plot([x0 + w, x0 + w + 0.009], [yy, yy], transform=ax.transAxes,
                color="black", linewidth=0.7, zorder=7)
        lab = "%d" % int(vlabels[i]) if vlabels is not None else fmt % (hi - k * s)
        ax.text(x0 + w + 0.013, yy, lab, transform=ax.transAxes, fontsize=7.6,
                ha="left", va="center", zorder=7)


def draw_catlegend(ax):
    """Two-column categorical legend for land cover."""
    xs = [0.010, 0.175]
    ys = [0.360, 0.272, 0.184, 0.096]
    sw, sh = 0.029, 0.042
    ax.add_patch(Rectangle((0.0, 0.045), 0.365, 0.445, transform=ax.transAxes,
                           facecolor="white", edgecolor="none", zorder=4))
    order = [("Cropland", 1), ("Shrub", 3), ("Grassland", 4), ("Barren", 7),
             ("Forest", 2), ("Water", 5), ("Snow", 6), ("Impervious", 8)]
    ax.text(xs[0], 0.428, "Type", transform=ax.transAxes, fontsize=10.4,
            ha="left", va="bottom", zorder=8)
    for j, (name, cls) in enumerate(order):
        col, row = j // 4, j % 4
        ax.add_patch(Rectangle((xs[col], ys[row]), sw, sh, transform=ax.transAxes,
                               facecolor=LC[cls][1], edgecolor="#9c9c9c",
                               linewidth=0.35, zorder=8))
        ax.text(xs[col] + sw + 0.010, ys[row] + sh / 2, name,
                transform=ax.transAxes, fontsize=7.4, ha="left", va="center", zorder=8)


def main():
    intensity = winsorize(read(INTENSITY_TIF))
    amplitude = winsorize(read(AMPLITUDE_TIF))
    exposure = winsorize(np.clip(read(EXPOSURE_TIF) / (33 * 33) * 100.0, 0, 100))
    gai = winsorize(read(GAI_TIF)) if GAI_TIF else None
    age = winsorize(read(AGE_TIF))

    # Mask greenness age to areas with valid construction-year records.
    flt = read(AGE_FILTER_TIF)
    flt = np.where(np.isclose(flt, -9999.0) | np.isclose(flt, -2147483648.0), np.nan, flt)
    valid = np.where(np.isfinite(flt), 1.0, np.nan)
    valid = reproject_to_ref(valid, AGE_FILTER_TIF, REF_TIF, rasterio.warp.Resampling.nearest)
    age = np.where(np.isfinite(valid) & np.isfinite(age), age, np.nan)

    # Land cover: read the window covering the reference extent, then reproject.
    with rasterio.open(REF_TIF) as r:
        rb = r.bounds
        prof_ref = r.profile.copy()
    xs = [rb.left, rb.right, rb.right, rb.left]
    ys = [rb.bottom, rb.bottom, rb.top, rb.top]
    with rasterio.open(LANDCOVER_TIF) as s:
        tx, ty = rasterio.warp.transform(4326, s.crs, xs, ys)
        win = from_bounds(min(tx), min(ty), max(tx), max(ty), s.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, s.width, s.height))
        lc = s.read(1, window=win)
        lt = s.window_transform(win)
    lc = np.where(lc == 0, np.nan, lc.astype("float64"))
    lc_re = np.full((prof_ref["height"], prof_ref["width"]), np.nan)
    rasterio.warp.reproject(lc, lc_re, src_transform=lt,
                            src_crs=rasterio.open(LANDCOVER_TIF).crs,
                            dst_transform=prof_ref["transform"],
                            dst_crs=prof_ref["crs"],
                            resampling=rasterio.warp.Resampling.nearest,
                            src_nodata=np.nan, dst_nodata=np.nan)
    lc_re = np.where(np.isfinite(intensity), lc_re, np.nan)

    aq = quantile_boundaries(amplitude)
    alabels = [int(round(aq[8])), int(round(aq[6])), int(round(aq[4])),
               int(round(aq[2])), int(round(aq[0]))]

    panels = [
        dict(tag="a", title="Greenness intensity", data=intensity, unit="NDVI",
             fmt="%.2f", drange=(-0.10, 0.41)),
        dict(tag="b", title="Aggregation index", data=gai, unit="GAI", fmt="%.2f",
             drange=(-2.50, 2.225)),
        dict(tag="c", title="Seasonal amplitude", data=amplitude, unit="Amplitude",
             fmt="%.0f", boundaries=aq, vlabels=alabels),
        dict(tag="d", title="Greenness age", data=age, unit="Year", fmt="%.0f",
             drange=(0.0, 27.0)),
        dict(tag="e", title="Greenness exposure", data=exposure, unit="Percentage",
             fmt="%.0f", drange=(0.0, 112.5)),
        dict(tag="f", title="Land cover", data=None, unit=None, fmt=None),
    ]

    extent = [rb.left, rb.right, rb.bottom, rb.top]
    plt.rcParams.update({"font.family": "sans-serif",
                        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    fig = plt.figure(figsize=(12.4, 5.60), dpi=600)
    left, right, top, bottom = 0.010, 0.994, 0.985, 0.032
    gx, gy = 0.004, 0.008
    pw = (right - left - 2 * gx) / 3.0
    ph = (top - bottom - gy) / 2.0
    ml, mb, mw, mh = 0.081, 0.045, 0.912, 0.933

    for i, p in enumerate(panels):
        r_, c_ = divmod(i, 3)
        x0 = left + c_ * (pw + gx)
        y0 = top - (r_ + 1) * ph - r_ * gy
        ax = fig.add_axes([x0 + ml * pw, y0 + mb * ph, mw * pw, mh * ph])
        cell = fig.add_axes([x0, y0, pw, ph])
        cell.set_zorder(ax.get_zorder() + 10)
        cell.set_axis_off()
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_axis_off()
        ax.set_facecolor("white")

        if p["data"] is not None:
            if "boundaries" in p:
                cmap = ListedColormap(RB)
                cmap.set_bad("white")
                norm = BoundaryNorm(p["boundaries"], len(RB), clip=True)
                ax.imshow(p["data"], extent=extent, cmap=cmap, norm=norm,
                          interpolation="nearest", aspect="auto", zorder=2)
                draw_vlegend(cell, p["boundaries"][0], p["boundaries"][-1],
                             p["unit"], p["fmt"], vlabels=p.get("vlabels"))
            else:
                vmin, vmax = p["drange"]
                ax.imshow(p["data"], extent=extent, cmap=RAMP, vmin=vmin, vmax=vmax,
                          interpolation="nearest", aspect="auto", zorder=2)
                draw_vlegend(cell, vmin, vmax, p["unit"], p["fmt"])
        else:
            img = np.full(lc_re.shape, np.nan)
            for cls in LC:
                img = np.where(lc_re == cls, cls, img)
            keys = sorted(LC)
            ax.imshow(np.ma.masked_invalid(img), extent=extent,
                      cmap=ListedColormap([LC[c][1] for c in keys]),
                      norm=BoundaryNorm([c - 0.5 for c in keys] + [max(keys) + 0.5],
                                        len(keys)),
                      interpolation="nearest", aspect="auto", zorder=2)
            draw_catlegend(cell)

        cell.text(0.010, 0.985, f"({p['tag']}) {p['title']}",
                  transform=cell.transAxes, fontsize=10.4, fontweight="bold",
                  ha="left", va="top", zorder=8)
        draw_north(ax)
        draw_scalebar(ax, extent)

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, "Figure_greenness_factor_maps")
    for ext, kw in ((".png", {"dpi": 600}),
                    (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
                    (".pdf", {}), (".svg", {})):
        fig.savefig(base + ext, facecolor="white", **kw)
    plt.close(fig)
    print("saved", base)


if __name__ == "__main__":
    main()
