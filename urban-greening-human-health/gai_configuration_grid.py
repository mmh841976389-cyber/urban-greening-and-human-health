# -*- coding: utf-8 -*-
"""
GAI sensitivity grid: orthogonalised Greenness Aggregation Index across
NDVI thresholds and buffer radii, with health associations per configuration.

For every (threshold x radius) configuration:
  1. compute the four landscape metrics (COHESION, PD, AI, AREA_MN) and PLAND
     (percentage of the buffer occupied by green pixels);
  2. regress each metric on PLAND and keep the residuals, so that the metrics
     are orthogonal to greenness amount;
  3. take the first principal component of the four standardised residuals as
     GAI_orth, with the sign anchored to COHESION;
  4. fit the fully adjusted health models with cluster-robust standard errors.

The primary configuration reported in the manuscript is NDVI > 0.30 with a
500 m buffer.

Inputs (environment variables)
  MVC_DIR       growing-season maximum-value-composite NDVI rasters
  PARTICIPANT_CSV  participant table (id, longitude, latitude, examination year)
  HEALTH_CSV    health and covariate table
  OUT_DIR       output directory
"""
import os
import csv
import math
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from scipy import ndimage
from sklearn.decomposition import PCA
import statsmodels.api as sm

warnings.filterwarnings("ignore")

MVC_DIR = os.environ.get("MVC_DIR", r"F:\Hangzhou_Exposure_2013-2019\NDVI_GS_MVC")
PARTICIPANT_CSV = os.environ.get("PARTICIPANT_CSV", "中心城区_最终分析表.csv")
HEALTH_CSV = os.environ.get("HEALTH_CSV", PARTICIPANT_CSV)
OUT_DIR = os.environ.get("OUT_DIR", ".")

CELL = 30.0
EPSG_WGS84, EPSG_UTM = 4326, 32650
THRESHOLDS = [0.20, 0.25, 0.30, 0.40]      # primary: 0.30
RADII = [300, 500, 1000]                   # primary: 500
WINDOWS = [(2013, 2016), (2014, 2017), (2015, 2018), (2016, 2019)]
SQ2 = math.sqrt(2.0)
NEIGHBOURS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))

INDIV = ["age", "gender", "bmi", "smoking", "drinking", "pa", "meal_reg"]
ENV = ["PM2.5_r500", "夜光_r500", "人口密度_r500", "DEM_r500", "道路距离_主干道_r500"]
YEAR = ["体检年"]
CLUSTER = "地理名称"
BINARY_OUT = ["是否高血压", "是否高血糖"]
CONT_OUT = ["sbp", "dbp", "map", "fpg"]

CONFIGS = [(t, r) for r in RADII for t in THRESHOLDS]


def tname(t):
    return f"T{int(round(t * 100))}"


def load_participants(path):
    with open(path, encoding="gb18030", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        head = next(rd)
        rows = list(rd)
    i_id, i_lon, i_lat = head.index("ID"), head.index("经度"), head.index("纬度")
    i_ey = head.index("体检年") if "体检年" in head else None
    pts = []
    for x in rows:
        try:
            ey = int(float(x[i_ey])) if (i_ey is not None and x[i_ey].strip()) else 2018
        except Exception:
            ey = 2018
        try:
            pts.append((x[i_id], float(x[i_lon]), float(x[i_lat]), ey))
        except Exception:
            continue
    lon = [p[1] for p in pts]
    lat = [p[2] for p in pts]
    xs, ys = transform(EPSG_WGS84, EPSG_UTM, lon, lat)
    return list(zip([p[0] for p in pts], xs, ys, [p[3] for p in pts]))


def load_windows():
    arr, left, top = {}, None, None
    for s, e in WINDOWS:
        with rasterio.open(os.path.join(MVC_DIR, f"ndviGS_MVC_4yr_{s}-{e}.tif")) as ds:
            a = ds.read(1).astype(np.float32)
            if left is None:
                left, top = ds.bounds.left, ds.bounds.top
        a[~np.isfinite(a)] = np.nan
        a[(a < -1) | (a > 1)] = np.nan
        arr[e] = a
    return arr, left, top


def adj_like(g):
    """Total like adjacencies (orthogonal weight 1, diagonal weight 1/sqrt(2))."""
    gg = g.astype(np.float64)
    h, w = gg.shape
    tot = 0.0
    for dx, dy, wt in NEIGHBOURS:
        a = gg[max(0, dx):h + min(0, dx), max(0, dy):w + min(0, dy)]
        b = gg[max(0, -dx):h + min(0, -dx), max(0, -dy):w + min(0, -dy)]
        tot += wt * np.sum(a * b)
    return tot


def max_adj_like(n):
    """Like adjacencies of the most compact packing of n cells."""
    if n <= 1:
        return 0.0
    w_ = int(math.ceil(math.sqrt(n)))
    h_ = int(math.ceil(n / w_))
    blk = np.zeros((h_, w_), dtype=bool)
    blk.flat[:n] = True
    return adj_like(blk)


def perimeter_len(g):
    """Perimeter length counted against non-green or outside cells."""
    gg = g.astype(np.float64)
    h, w = gg.shape
    tot = 0.0
    for dx, dy, wt in NEIGHBOURS:
        a = gg[max(0, dx):h + min(0, dx), max(0, dy):w + min(0, dy)]
        b = gg[max(0, -dx):h + min(0, -dx), max(0, -dy):w + min(0, -dy)]
        tot += wt * np.sum(a * (1 - b))
    return tot


def extract_buffer(arr, x, y, radius, left, top):
    cx = (x - left) / CELL
    cy = (top - y) / CELL
    rp = radius / CELL
    pad = int(math.ceil(rp)) + 2
    c0, r0 = int(round(cx)), int(round(cy))
    c1, r1 = max(c0 - pad, 0), max(r0 - pad, 0)
    c2, r2 = min(c0 + pad + 1, arr.shape[1]), min(r0 + pad + 1, arr.shape[0])
    if c2 <= c1 or r2 <= r1:
        return None, None
    sub = arr[r1:r2, c1:c2]
    h_, w_ = sub.shape
    gy, gx = np.ogrid[:h_, :w_]
    return sub, ((gx - (c0 - c1)) ** 2 + (gy - (r0 - r1)) ** 2) <= rp * rp


def window_metrics(sub, mask, thr, radius):
    """Return (n_patches, PD, ED, COHESION, AI, AREA_MN, PLAND)."""
    with np.errstate(invalid="ignore"):
        green = (sub >= thr) & mask
    if not green.any():
        return None
    lab, n = ndimage.label(green, structure=np.ones((3, 3)))
    if n == 0:
        return None
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    sp = spa = 0.0
    for k in range(1, n + 1):
        pk = perimeter_len(lab == k)
        sp += pk
        spa += pk * math.sqrt(float(areas[k - 1]))
    a_ha = math.pi * radius * radius / 10000.0
    a_cells = math.pi * radius * radius / (CELL * CELL)
    den = 1.0 - 1.0 / math.sqrt(a_cells)
    coh = ((1.0 - sp / spa) / den * 100.0) if (spa > 0 and den > 0) else 0.0
    coh = min(max(coh, 0.0), 100.0)
    gii = adj_like(green)
    mg = max_adj_like(int(areas.sum()))
    ai = (gii / mg * 100.0) if mg > 0 else 0.0
    ai = min(max(ai, 0.0), 100.0)
    pland = float(green.sum()) / float(mask.sum()) * 100.0
    return (n, n / a_ha, (sp * CELL) / a_ha, coh, ai,
            float(np.mean(areas * CELL * CELL)), pland)


def orthogonalised_gai(df, cfg):
    """PCA of the four PLAND-residualised metrics, sign anchored to COHESION."""
    need = [f"COH_{cfg}", f"PD_{cfg}", f"AI_{cfg}", f"AM_{cfg}", f"PLAND_{cfg}"]
    d = df.dropna(subset=need)
    if len(d) < 100:
        return None
    res = {}
    for src, nm in ((f"COH_{cfg}", "COH"), (f"PD_{cfg}", "PD"),
                    (f"AI_{cfg}", "AI"), (f"AM_{cfg}", "AM")):
        x = sm.add_constant(d[f"PLAND_{cfg}"])
        res[nm] = sm.OLS(d[src], x).fit().resid
    res["PD"] = -res["PD"]                      # reversed: higher PD = more fragmented
    rz = (pd.DataFrame(res) - pd.DataFrame(res).mean()) / pd.DataFrame(res).std()

    pca = PCA(n_components=1)
    sc = pca.fit_transform(rz.values)
    if pca.components_[0][0] < 0:               # anchor the sign to COHESION
        sc = -sc

    # Non-orthogonalised index, kept as a contrast.
    raw = pd.DataFrame({"COH": d[f"COH_{cfg}"], "PD": -d[f"PD_{cfg}"],
                        "AI": d[f"AI_{cfg}"], "AM": d[f"AM_{cfg}"]})
    oz = (raw - raw.mean()) / raw.std()
    p2 = PCA(n_components=1)
    s2 = p2.fit_transform(oz.values)
    if p2.components_[0][0] < 0:
        s2 = -s2
    return pd.DataFrame({"ID": d["ID"], f"GAI_orth_{cfg}": sc[:, 0],
                         f"GAI_raw_{cfg}": s2[:, 0]})


def fit(df, exposure, outcome, health_cols):
    """Fully adjusted model per 1 SD of exposure, cluster-robust SE."""
    binary = outcome in BINARY_OUT
    d = df.copy()
    d["y"] = pd.to_numeric(d[outcome], errors="coerce")
    d["x"] = pd.to_numeric(d[exposure], errors="coerce")
    adj = health_cols + [CLUSTER]
    d = d.dropna(subset=["y", "x"] + adj)
    if binary:
        d = d[d["y"].isin([0, 1])]
    if len(d) < 100 or (binary and d["y"].nunique() < 2):
        return None
    x = pd.DataFrame({"exposure_z": (d["x"] - d["x"].mean()) / d["x"].std()})
    for c in health_cols:
        x[c] = pd.to_numeric(d[c], errors="coerce")
    x = sm.add_constant(x)
    mod = sm.Logit(d["y"].astype(int), x) if binary else sm.OLS(d["y"], x)
    m = mod.fit(disp=0, cov_type="cluster", cov_kwds={"groups": d[CLUSTER]})
    b, se = m.params["exposure_z"], m.bse["exposure_z"]
    if binary:
        est, lo, hi = math.exp(b), math.exp(b - 1.96 * se), math.exp(b + 1.96 * se)
    else:
        est, lo, hi = b, b - 1.96 * se, b + 1.96 * se
    return {"config": cfg_of(exposure), "exposure": exposure, "outcome": outcome,
            "n": len(d), "estimate": round(est, 3), "ci_low": round(lo, 3),
            "ci_high": round(hi, 3), "p": float(f"{m.pvalues['exposure_z']:.3g}")}


def cfg_of(col):
    return col.split("_")[-1]


def main():
    pts = load_participants(PARTICIPANT_CSV)
    win_arr, left, top = load_windows()
    print("participants:", len(pts), "| windows:", sorted(win_arr))

    recs = [dict() for _ in pts]
    for i, (pid, x, y, ey) in enumerate(pts):
        arr = win_arr[ey if ey in win_arr else 2018]
        for rad in RADII:
            sub, mask = extract_buffer(arr, x, y, rad, left, top)
            if sub is None:
                continue
            for t in THRESHOLDS:
                m = window_metrics(sub, mask, t, rad)
                if m:
                    recs[i][f"{tname(t)}_R{rad}"] = m
        if (i + 1) % 3000 == 0:
            print(f"  ...{i + 1}/{len(pts)}")

    cols = {"ID": [p[0] for p in pts]}
    for t, r in CONFIGS:
        c = f"{tname(t)}_R{r}"
        nan6 = (np.nan,) * 7
        for j, key in enumerate(("n", "PD", "ED", "COH", "AI", "AM", "PLAND")):
            cols[f"{key}_{c}"] = [rec.get(c, nan6)[j] for rec in recs]
    metrics = pd.DataFrame(cols)

    health = pd.read_csv(HEALTH_CSV, encoding="gb18030", sep="\t")
    health["map"] = (health["sbp"] + 2 * health["dbp"]) / 3.0
    health_cols = [c for c in INDIV + ENV + YEAR if c in health.columns]

    rows = []
    for t, r in CONFIGS:
        c = f"{tname(t)}_R{r}"
        g = orthogonalised_gai(metrics, c)
        if g is None:
            continue
        gg = (metrics[["ID", f"PLAND_{c}"]]
              .merge(g, on="ID", how="left")
              .merge(health, on="ID", how="inner"))
        corr = {"orth": gg[f"GAI_orth_{c}"].corr(gg[f"PLAND_{c}"]),
                "raw": gg[f"GAI_raw_{c}"].corr(gg[f"PLAND_{c}"])}
        for outcome in BINARY_OUT + CONT_OUT:
            for exp in (f"GAI_orth_{c}", f"GAI_raw_{c}"):
                res = fit(gg, exp, outcome, health_cols)
                if res:
                    res["version"] = "orthogonalised" if "orth" in exp else "raw"
                    res["corr_with_PLAND"] = round(corr["orth" if "orth" in exp else "raw"], 3)
                    rows.append(res)
        print(f"  {c} done")

    os.makedirs(OUT_DIR, exist_ok=True)
    metrics.to_csv(os.path.join(OUT_DIR, "gai_configuration_metrics.csv"),
                   index=False, encoding="utf-8-sig")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "gai_configuration_results.csv"),
               index=False, encoding="utf-8-sig")
    print("saved gai_configuration_metrics.csv / gai_configuration_results.csv")


if __name__ == "__main__":
    main()
