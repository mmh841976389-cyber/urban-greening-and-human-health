# -*- coding: utf-8 -*-
"""
Build the participant-level factor table (ten factors: five greenness, five
environment) by extracting raster values at each residential location.

Greenness factors
  1. intensity      growing-season maximum-value-composite NDVI, mean of the
                    examination year and the three preceding years (Y-3..Y)
  2. aggregation    GAI, from data_prep/gai_raster.py or the participant-level
                    GAI script
  3. seasonal       harmonic amplitude of the NDVI seasonal cycle
  4. greenness age  years since neighbourhood construction, aligned to the
                    examination year
  5. exposure       percentage of vegetated pixels (NDVI > 0.20) in a 1 km
                    moving window

Environment factors: elevation, population density, distance to main roads,
PM2.5, night-time light. Year-specific layers are matched to the examination
year; intensity and aggregation use the four-year window.

Inputs (environment variables)
  HEALTH_CSV    participant table (id, longitude, latitude, examination year)
  NDVI_DIR      directory of annual growing-season MVC NDVI rasters
  ENV_DIR       directory of environmental rasters
  GAI_CSV       participant-level GAI table
  SEASON_CSV    participant-level seasonal amplitude table
  ROAD_CSV      participant-level road distance table
  OUT_DIR       output directory
"""
import os
import warnings

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform as rt
from rasterio.warp import transform
from scipy import ndimage

warnings.filterwarnings("ignore")

HEALTH_CSV = os.environ.get("HEALTH_CSV", "中心城区_最终分析表.csv")
NDVI_DIR = os.environ.get("NDVI_DIR", r"F:\Hangzhou_Exposure_2013-2019\NDVI_GS_MVC")
ENV_DIR = os.environ.get("ENV_DIR", r"C:\Users\mmh\Desktop\数据\各因子tif")
GAI_CSV = os.environ.get("GAI_CSV", "GAI_正交化_明细.csv")
SEASON_CSV = os.environ.get("SEASON_CSV", "绿度季节变化_谐波.csv")
ROAD_CSV = os.environ.get("ROAD_CSV", "道路距离_OSM_逐人.csv")
OUT_DIR = os.environ.get("OUT_DIR", ".")

RADII = [300, 500, 1000]
EXPOSURE_THRESHOLD = 0.20
EXPOSURE_WINDOW_M = 1000.0
MIN_VALID_FRAC = 0.5            # windows with less valid data are set to NaN

_CACHE = {}


def pixel_size_m(tf, crs, lat0=30.3):
    """Ground cell size in metres (transform.a is degrees for geographic CRS)."""
    try:
        geographic = bool(crs.is_geographic)
    except Exception:
        geographic = True
    a, e = abs(float(tf.a)), abs(float(tf.e))
    if geographic:
        dy = e * 111320.0
        dx = a * 111320.0 * np.cos(np.deg2rad(lat0))
        return float(np.sqrt(dx * dy)) if dx > 0 and dy > 0 else 30.0
    return a


def load(path):
    if path in _CACHE:
        return _CACHE[path]
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            a = np.where(a == ds.nodata, np.nan, a)
        a[a < -3e38] = np.nan
        res_m = pixel_size_m(ds.transform, ds.crs)
        _CACHE[path] = (a, ds.transform, ds.crs, ds.width, ds.height, res_m)
    return _CACHE[path]


def to_xy(crs, lons, lats):
    """Reproject WGS84 coordinates to the raster CRS when needed."""
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg is not None and epsg != 4326:
        xs, ys = transform("EPSG:4326", crs, lons.tolist(), lats.tolist())
        return np.array(xs), np.array(ys)
    return np.asarray(lons, float), np.asarray(lats, float)


def rows_cols(tf, crs, lons, lats, height, width):
    xs, ys = to_xy(crs, lons, lats)
    rows, cols = rt.rowcol(tf, xs, ys)
    return (np.clip(np.asarray(rows), 0, height - 1),
            np.clip(np.asarray(cols), 0, width - 1))


def buffer_mean(arr, rows, cols, idx, k):
    """Mean of the valid cells inside a circular buffer of radius k cells."""
    h, w = arr.shape
    vals = np.full(len(rows), np.nan)
    for i in idx:
        r0, c0 = rows[i], cols[i]
        rs = slice(max(0, r0 - k), min(h, r0 + k + 1))
        cs = slice(max(0, c0 - k), min(w, c0 + k + 1))
        rr, cc = np.mgrid[rs, cs]
        m = (rr - r0) ** 2 + (cc - c0) ** 2 <= k * k
        v = arr[rs, cs][m]
        v = v[np.isfinite(v)]
        if v.size:
            vals[i] = v.mean()
    return vals


def green_exposure(ndvi, res, thr=EXPOSURE_THRESHOLD, win_m=EXPOSURE_WINDOW_M):
    """Percentage of vegetated cells inside a square moving window."""
    valid = np.isfinite(ndvi)
    binary = np.where(valid, (ndvi >= thr).astype(np.float32), 0.0)
    k = max(1, int(round(win_m / res)))
    k = k + 1 if k % 2 == 0 else k
    veg = ndimage.uniform_filter(binary, size=k, mode="reflect")
    cnt = ndimage.uniform_filter(valid.astype(np.float32), size=k, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(cnt > 0, veg / cnt * 100.0, np.nan)
    frac = np.where(cnt < MIN_VALID_FRAC, np.nan, frac)
    return frac, k


def extract(path, lons, lats, radii=RADII):
    """Point value plus circular-buffer means for each requested radius."""
    arr, tf, crs, w, h, res = load(path)
    rows, cols = rows_cols(tf, crs, lons, lats, h, w)
    out = {"point": arr[rows, cols]}
    idx = np.arange(len(rows))
    for r in radii:
        k = max(1, min(int(round(r / res)), 400))
        out[f"r{r}"] = buffer_mean(arr, rows, cols, idx, k)
    return out


def main():
    health = pd.read_csv(HEALTH_CSV, encoding="gb18030", sep=None, engine="python")
    if "ID" not in health.columns:
        health = pd.read_csv(HEALTH_CSV, encoding="gb18030")
    health.columns = [str(c).lstrip("\ufeff").strip() for c in health.columns]
    health["ID"] = health["ID"].astype(str)
    lons = health["经度"].astype(float).values
    lats = health["纬度"].astype(float).values
    year = health["体检年"].astype(int).values

    res = pd.DataFrame({"ID": health["ID"], "体检年": year})
    if "map" not in health.columns and {"sbp", "dbp"}.issubset(health.columns):
        sbp = pd.to_numeric(health["sbp"], errors="coerce")
        dbp = pd.to_numeric(health["dbp"], errors="coerce")
        health["map"] = dbp + (sbp - dbp) / 3.0
    keep_cols = ["经度", "纬度", "sbp", "dbp", "fpg", "map", "age", "gender", "bmi",
                 "smoking", "drinking", "pa", "meal_reg", "体检月", "体检日",
                 "是否高血压", "是否高血糖", "t2d_base", "hbp", "hbp_med", "tdm",
                 "tdm_med", "地理名称"]
    for c in keep_cols:
        if c in health.columns:
            res[c] = health[c].values
    print("sample", len(res), "| examination years:",
          pd.Series(year).value_counts().sort_index().to_dict())

    years = sorted(set(year))
    ndvi_year = {}
    for y in sorted(set(years + [min(years) - 3])):
        p = os.path.join(NDVI_DIR, f"ndviGS_MVC_{y}.tif")
        if os.path.exists(p):
            arr, tf, crs, w, h, r = load(p)
            ndvi_year[y] = (arr, tf, crs, h, r)

    # Greenness exposure, year-specific.
    for y in years:
        if y not in ndvi_year:
            continue
        arr, tf, crs, h, r = ndvi_year[y]
        frac, k = green_exposure(arr, r)
        rows, cols = rows_cols(tf, crs, lons, lats, h, arr.shape[1])
        sel = np.where(year == y)[0]
        res.loc[sel, "绿度暴露_point"] = frac[rows, cols][sel]
        kb = max(1, min(int(round(500 / r)), 400))
        res.loc[sel, "绿度暴露_r500"] = buffer_mean(frac, rows, cols, sel, kb)[sel]

    # Four-year mean NDVI: exposure (sensitivity) and intensity.
    for y in years:
        ys4 = [yy for yy in range(y - 3, y + 1) if yy in ndvi_year]
        if len(ys4) < 2:
            continue
        with np.errstate(invalid="ignore"):
            stack = np.nanmean(np.stack([ndvi_year[yy][0] for yy in ys4], axis=0), axis=0)
        _, tf, crs, h, r = ndvi_year[ys4[-1]]
        rows, cols = rows_cols(tf, crs, lons, lats, h, stack.shape[1])
        sel = np.where(year == y)[0]
        frac, _ = green_exposure(stack, r)
        kb = max(1, min(int(round(500 / r)), 400))
        res.loc[sel, "绿度暴露_4yr_r500"] = buffer_mean(frac, rows, cols, sel, kb)[sel]
        res.loc[sel, "绿度强度_point"] = stack[rows, cols][sel]
        for rad in RADII:
            kb = max(1, min(int(round(rad / r)), 400))
            res.loc[sel, f"绿度强度_r{rad}"] = buffer_mean(stack, rows, cols, sel, kb)[sel]
        del stack

    # Aggregation index (GAI) from the participant-level table.
    if os.path.exists(GAI_CSV):
        g = pd.read_csv(GAI_CSV)
        g["ID"] = g["ID"].astype(str)
        cols = [c for c in ["ID", "GAI_orth", "GAI_orig", "PLAND"] if c in g.columns]
        res = res.merge(g[cols], on="ID", how="left")
        res = res.rename(columns={"GAI_orth": "绿度聚集度_GAI",
                                  "GAI_orig": "绿度聚集度_GAI_未调整",
                                  "PLAND": "绿度_PLAND"})

    # Seasonal amplitude.
    if os.path.exists(SEASON_CSV):
        s = pd.read_csv(SEASON_CSV)
        s["ID"] = s["ID"].astype(str)
        cols = [c for c in ["ID", "季节振幅_谐波_point", "季节振幅_谐波_r300",
                            "季节振幅_谐波_r500", "季节振幅_谐波_r1000",
                            "季节振幅_剖面maxmin_r500"] if c in s.columns]
        res = res.merge(s[cols], on="ID", how="left")

    # Greenness age: years since construction, aligned to the examination year.
    p = os.path.join(ENV_DIR, "小区建成时间", "小区建成时间_裁剪过滤_2020反转.tif")
    if os.path.exists(p):
        e = extract(p, lons, lats, [500])
        res["至2020年限_r500"] = e["r500"]
        res["绿度年龄_r500"] = res["至2020年限_r500"] - (2020 - res["体检年"])

    # Elevation (static).
    p = os.path.join(ENV_DIR, "地形高程", "地形高程_裁剪后.tif")
    if os.path.exists(p):
        for k, v in extract(p, lons, lats, [500]).items():
            res[f"DEM_{k}"] = v

    # Population density, night-time light, PM2.5 (year-specific).
    for y in sorted(set(year)):
        sel = year == y
        for sub, col in (("POP", "人口密度"), ("NTL", "夜光"), ("PM25", "PM2.5")):
            fp = os.path.join(os.path.dirname(NDVI_DIR), sub, f"{sub}_{y}.tif")
            if not os.path.exists(fp):
                print(f"  missing {col} {y}")
                continue
            e = extract(fp, lons[sel], lats[sel], [500, 1000])
            for k, v in e.items():
                res.loc[sel, f"{col}_{k}"] = v

    # Road distance.
    if os.path.exists(ROAD_CSV):
        rd = pd.read_csv(ROAD_CSV)
        rd["ID"] = rd["ID"].astype(str)
        cols = ["ID"] + [c for c in rd.columns
                         if c.startswith(("道路距离_主干道", "道路距离_含三级"))]
        res = res.merge(rd[cols], on="ID", how="left")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "factor_table.csv")
    res.round(6).to_csv(out_path, index=False, encoding="utf-8-sig")
    print("wrote", out_path, "rows", len(res), "cols", len(res.columns))


if __name__ == "__main__":
    main()
