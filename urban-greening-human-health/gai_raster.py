# -*- coding: utf-8 -*-
"""
GAI raster: gridded Greenness Aggregation Index for mapping.

Green pixels are defined by thresholding the growing-season maximum-value
composite NDVI (0.30) inside a 500 m circular moving window on the 30 m grid.
For each window the four landscape metrics (COHESION, PD, AI, AREA_MN) are
computed, each is regressed on PLAND (the green percentage of the same window)
across all sampled windows, and the first principal component of the four
standardised residuals is taken as GAI. The sign is anchored to COHESION so
that a higher GAI always means a more aggregated configuration.

The resulting coarse grid is interpolated back to the full 30 m grid and
reprojected onto the mapping grid.

Inputs (environment variables)
  MVC_TIF       growing-season MVC NDVI raster (e.g. ndviGS_MVC_4yr_2016-2019.tif)
  REF_TIF       mapping-grid reference raster
  OUT_DIR       output directory
"""
import os
import math
import time
import numpy as np
import rasterio
import rasterio.warp
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

MVC_TIF = os.environ.get("MVC_TIF", "")
REF_TIF = os.environ.get("REF_TIF", "")
OUT_DIR = os.environ.get("OUT_DIR", ".")

CELL = 30.0
THRESHOLD = 0.30
RADIUS = 500.0
STRIDE = 2                      # sampling step (cells); metrics use full-resolution windows
SQ2 = math.sqrt(2.0)
NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))

_MAX_CACHE = {}


def adj_like(g):
    """Total like adjacencies (orthogonal 1, diagonal 1/sqrt(2))."""
    gg = g.astype(np.float64)
    h, w = gg.shape
    tot = 0.0
    for dx, dy, wt in NEIGHBOURS:
        a = gg[max(0, dx):h + min(0, dx), max(0, dy):w + min(0, dy)]
        b = gg[max(0, -dx):h + min(0, -dx), max(0, -dy):w + min(0, -dy)]
        tot += wt * float(np.sum(a * b))
    return tot


def max_adj_like(n):
    """Like adjacencies of the most compact packing of n cells."""
    if n <= 1:
        return 0.0
    if n in _MAX_CACHE:
        return _MAX_CACHE[n]
    w_ = int(math.ceil(math.sqrt(n)))
    h_ = int(math.ceil(n / w_))
    blk = np.zeros((h_, w_), dtype=bool)
    blk.flat[:n] = True
    _MAX_CACHE[n] = adj_like(blk)
    return _MAX_CACHE[n]


def window_metrics(green, circ):
    """Return (n_patches, PD, COHESION, AI, AREA_MN, PLAND) for one window."""
    lab, n = ndimage.label(green, structure=np.ones((3, 3)))
    if n == 0:
        return None
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    gg = green.astype(np.float64)
    h, w = gg.shape
    perm = np.zeros_like(gg)
    for dx, dy, wt in NEIGHBOURS:
        a = gg[max(0, dx):h + min(0, dx), max(0, dy):w + min(0, dy)]
        b = gg[max(0, -dx):h + min(0, -dx), max(0, -dy):w + min(0, -dy)]
        sub = np.zeros_like(gg)
        sub[max(0, -dx):h + min(0, -dx), max(0, -dy):w + min(0, -dy)] = a * (1.0 - b)
        perm += wt * sub
    perm_lab = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    sp = float(perm.sum())
    spa = float(np.sum(perm_lab * np.sqrt(areas)))

    a_ha = math.pi * RADIUS * RADIUS / 10000.0
    a_cells = math.pi * RADIUS * RADIUS / (CELL * CELL)
    den = 1.0 - 1.0 / math.sqrt(a_cells)
    coh = ((1.0 - sp / spa) / den * 100.0) if (spa > 0 and den > 0) else 0.0
    coh = min(max(coh, 0.0), 100.0)
    gii = adj_like(green)
    mg = max_adj_like(int(areas.sum()))
    ai = (gii / mg * 100.0) if mg > 0 else 0.0
    ai = min(max(ai, 0.0), 100.0)
    pland = float(green.sum()) / float(circ.sum()) * 100.0
    return (float(n), n / a_ha, coh, ai, float(np.mean(areas * CELL * CELL)), pland)


def main():
    t0 = time.time()
    with rasterio.open(MVC_TIF) as ds:
        mvc = ds.read(1).astype(np.float32)
        prof = ds.profile.copy()
        h, w = mvc.shape
        src_transform, src_crs = ds.transform, ds.crs

    green = np.isfinite(mvc) & (mvc >= THRESHOLD)
    print("green share %.2f%%" % (100 * green.mean()), flush=True)

    rp = RADIUS / CELL
    pad = int(math.ceil(rp)) + 2
    ys = np.arange(pad, h - pad, STRIDE)
    xs = np.arange(pad, w - pad, STRIDE)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    n_pts = gx.size
    print("sample points", n_pts, flush=True)

    yy, xx = np.ogrid[:2 * pad + 1, :2 * pad + 1]
    circ = (yy - pad) ** 2 + (xx - pad) ** 2 <= rp * rp

    met = np.full((n_pts, 4), np.nan)          # PD, COH, AI, AREA_MN
    pland = np.full(n_pts, np.nan)
    for i in range(n_pts):
        r, c = gy[i], gx[i]
        sub = green[r - pad:r + pad + 1, c - pad:c + pad + 1]
        m = window_metrics(sub & circ, circ)
        if m is not None:
            met[i] = (m[1], m[2], m[3], m[4])
            pland[i] = m[5]
        if i and i % 20000 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1fs" % (i, n_pts, el), flush=True)

    ok = np.isfinite(met).all(1) & np.isfinite(pland)
    m_ok, p_ok = met[ok], pland[ok]
    print("valid windows", int(ok.sum()), flush=True)

    # Residualise each metric on PLAND, standardise, then take PC1.
    design = np.column_stack([np.ones_like(p_ok), p_ok])
    resid = np.empty_like(m_ok)
    for j, nm in enumerate(("PD", "COH", "AI", "AREA_MN")):
        beta, *_ = np.linalg.lstsq(design, m_ok[:, j], rcond=None)
        resid[:, j] = m_ok[:, j] - design @ beta
        print("  %-8s r with PLAND = %.3f" % (nm, np.corrcoef(m_ok[:, j], p_ok)[0, 1]))
    resid[:, 0] = -resid[:, 0]                 # reverse PD
    z = (resid - resid.mean(0)) / resid.std(0)

    pca = PCA(n_components=2).fit(z)
    pc1 = pca.transform(z)[:, 0]
    load = pca.components_[0]
    if load[1] < 0:                            # anchor the sign to COHESION
        pc1 = -pc1
        load = -load
    print("PC1 explains %.1f%%; loadings (PD, COH, AI, AREA_MN) ="
          % (100 * pca.explained_variance_ratio_[0]), np.round(load, 3))
    pc1 = (pc1 - pc1.mean()) / pc1.std()

    # Fill the sampling grid, then interpolate to the full 30 m grid.
    coarse = np.full(n_pts, np.nan, dtype=np.float32)
    coarse[ok] = pc1.astype(np.float32)
    coarse = coarse.reshape(ys.size, xs.size)
    idx = ndimage.distance_transform_edt(~np.isfinite(coarse),
                                         return_distances=False, return_indices=True)
    coarse = coarse[tuple(idx)]
    rgi = RegularGridInterpolator((ys.astype(float), xs.astype(float)), coarse,
                                  method="linear", bounds_error=False, fill_value=None)
    gyy, gxx = np.mgrid[0:h, 0:w]
    full = rgi((gyy.ravel(), gxx.ravel())).reshape(h, w).astype(np.float32)
    full = ndimage.gaussian_filter(full, sigma=0.8)

    os.makedirs(OUT_DIR, exist_ok=True)
    prof.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    out1 = os.path.join(OUT_DIR, "GAI_UTM30m.tif")
    with rasterio.open(out1, "w", **prof) as dst:
        dst.write(full, 1)
    print("wrote", out1)

    if REF_TIF:
        with rasterio.open(REF_TIF) as ds:
            rprof = ds.profile.copy()
            rprof.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
            rmask = ds.read(1, masked=True).mask
        warped = np.full((rprof["height"], rprof["width"]), -9999.0, dtype=np.float32)
        rasterio.warp.reproject(source=full, destination=warped,
                                src_transform=src_transform, src_crs=src_crs,
                                dst_transform=rprof["transform"], dst_crs=rprof["crs"],
                                resampling=rasterio.warp.Resampling.bilinear)
        warped[rmask] = -9999.0
        out2 = os.path.join(OUT_DIR, "GAI_mapgrid.tif")
        with rasterio.open(out2, "w", **rprof) as dst:
            dst.write(warped, 1)
        print("wrote", out2)

    np.savez_compressed(os.path.join(OUT_DIR, "gai_raster_detail.npz"),
                        metrics=met, pland=pland, valid=ok, pc1=pc1, loadings=load,
                        explained=float(pca.explained_variance_ratio_[0]))
    print("total %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
