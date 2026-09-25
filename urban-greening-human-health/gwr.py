# -*- coding: utf-8 -*-
"""
Geographically weighted regression (GWR) for the greenness - cardiometabolic analysis.

Each greenness factor is modelled separately (univariate local regression) at the
individual level. Outcomes and greenness factors are both residualised on the full
confounder set and z-scored, so the local coefficient is a partial association.

Confounder set matches the main manuscript:
  - individual confounders: age (cubic polynomial) + gender + BMI + smoking +
    drinking + physical activity + meal regularity
  - environmental covariates: PM2.5, population density, elevation,
    nighttime light, distance to major roads

Bandwidth: adaptive Gaussian kernel, bandwidth = distance to the k-th nearest
neighbour. k is selected by AICc over candidate fractions of n.
Neighbour searches use cKDTree (no O(n^2) distance matrix).

Inputs (paths are configurable; cohort data are not distributed):
  FACTOR_CSV  per-individual factor table with outcomes, confounders, coordinates
  STUDY_AREA  GeoJSON polygon used to clip the Voronoi map

Output: gwr_results.pkl, gwr_diagnostics.csv, gwr_factor_correlation.csv
"""
import csv
import json
import math
import os
import pickle
import warnings

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ------------------------------------------------------------------ configuration
FACTOR_CSV = os.environ.get("FACTOR_CSV", "data/factor_table.csv")
STUDY_AREA = os.environ.get("STUDY_AREA", "data/study_area.geojson")
OUT_DIR = os.environ.get("GWR_OUT", "output/gwr")
os.makedirs(OUT_DIR, exist_ok=True)

# outcomes: (key, column, type); binary outcomes use clinical thresholds
OUTCOMES = [
    ("SBP", "sbp", "continuous"),
    ("DBP", "dbp", "continuous"),
    ("MAP", None, "continuous"),
    ("FPG", "fpg", "continuous"),
    ("HTN", None, "binary"),
    ("HBG", None, "binary"),
]

# five greenness factors
FACTORS = [
    "绿度暴露_r500",
    "绿度强度_r500",
    "绿度聚集度_GAI",
    "季节振幅_谐波_r500",
    "绿度年龄_r500",
]
FACTOR_EN = {
    "绿度暴露_r500": "Greenness exposure",
    "绿度强度_r500": "Greenness intensity",
    "绿度聚集度_GAI": "Aggregation index",
    "季节振幅_谐波_r500": "Seasonal amplitude",
    "绿度年龄_r500": "Greenness age",
}
OUTCOME_EN = {
    "SBP": "Systolic BP", "DBP": "Diastolic BP", "MAP": "Mean arterial P",
    "FPG": "Fasting glucose", "HTN": "Hypertension", "HBG": "Hyperglycaemia",
}

# individual confounders (age handled separately as a cubic polynomial)
INDIV_COV = ["gender", "bmi", "smoking", "drinking", "pa", "meal_reg"]
AGE_COL = "age"
AGE_DEG = 3

# environmental covariates
ENV_COV = ["PM2.5_r500", "人口密度_r500", "DEM_r500", "夜光_r500", "道路距离_主干道_r500"]

BW_FRACTIONS = (0.05, 0.08, 0.12, 0.17, 0.23, 0.30, 0.40, 0.50, 0.62)

# local equirectangular projection around Hangzhou (metres)
LON0, LAT0 = 120.15, 30.27
KX = 111320.0 * math.cos(math.radians(LAT0))
KY = 110540.0


def ll_to_xy(lon, lat):
    """Project lon/lat to local metres."""
    return (lon - LON0) * KX, (lat - LAT0) * KY


# ------------------------------------------------------------------ data loading
def to_float(v):
    """Robust float conversion; returns nan for missing or invalid entries."""
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", "")
    if s == "" or s.lower() in ("nan", "none", "na", "null"):
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load():
    """Load the factor table, apply the analysis QC, and build the arrays."""
    with open(FACTOR_CSV, encoding="utf-8-sig", errors="ignore") as f:
        rows = list(csv.DictReader(f))

    needed = ["经度", "纬度", AGE_COL] + INDIV_COV + ENV_COV + FACTORS + ["sbp", "dbp", "fpg"]
    good, dropped = [], 0
    for r in rows:
        ok = all((c in r) and np.isfinite(to_float(r[c])) for c in needed)
        if ok:
            good.append(r)
        else:
            dropped += 1
    rows = good
    n = len(rows)
    print(f"loaded {len(rows) + dropped} rows; dropped {dropped}; using n = {n}")

    def col(c):
        return np.array([to_float(r[c]) for r in rows], dtype=float)

    # continuous outcomes: winsorise at the 1st/99th percentile (main-analysis QC)
    sbp, dbp, fpg = col("sbp"), col("dbp"), col("fpg")
    for arr in (sbp, dbp, fpg):
        lo, hi = np.nanpercentile(arr, 1), np.nanpercentile(arr, 99)
        np.clip(arr, lo, hi, out=arr)
    mapv = (sbp + 2.0 * dbp) / 3.0

    # binary outcomes from clinical thresholds on the winsorised values
    htn = ((sbp >= 140) | (dbp >= 90)).astype(float)
    hbg = (fpg >= 7.0).astype(float)

    outcomes = {"SBP": sbp, "DBP": dbp, "MAP": mapv, "FPG": fpg, "HTN": htn, "HBG": hbg}

    lon, lat = col("经度"), col("纬度")
    x, y = ll_to_xy(lon, lat)
    coords = np.column_stack([x, y]).astype(np.float64)

    age = col(AGE_COL)
    indiv = {c: col(c) for c in INDIV_COV}
    env = {c: col(c) for c in ENV_COV}
    green = {f: col(f) for f in FACTORS}
    return rows, age, indiv, env, green, outcomes, coords, n


# ------------------------------------------------------------------ adjustment
def design_matrix(age, indiv, env):
    """Full confounder design matrix: cubic age + individual + environmental."""
    blocks = [np.vander(age, AGE_DEG + 1)]
    blocks += [np.asarray(indiv[c], dtype=float).reshape(-1, 1) for c in INDIV_COV]
    blocks += [np.asarray(env[c], dtype=float).reshape(-1, 1) for c in ENV_COV]
    return np.column_stack(blocks)


def residualize_ols(y, A):
    """Residual of y on the confounder design matrix A."""
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def residualize_logit(y, A, max_iter=25):
    """IRLS logistic residual for a binary outcome (deviance residual)."""
    ncol = A.shape[1]
    beta = np.zeros(ncol)
    eta = A @ beta
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        W = p * (1 - p)
        z = eta + (y - p) / W
        XtWX = A.T @ (A * W[:, None])
        XtWz = A.T @ (W * z)
        beta = np.linalg.solve(XtWX + np.eye(ncol) * 1e-8, XtWz)
        eta = A @ beta
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    return y - p_hat


def zscore(v):
    s, m = float(np.nanstd(v)), float(np.nanmean(v))
    return (v - m) / s if s > 1e-9 else np.zeros_like(v)


# ------------------------------------------------------------------ GWR core
def solve_k(idx_k, dist_k, ydev, X1, k, chunk=600):
    """Local weighted least squares at bandwidth k (Gaussian kernel)."""
    n = len(ydev)
    b0 = np.zeros(n); b1 = np.zeros(n)
    lev = np.zeros(n); resid = np.zeros(n); se = np.zeros(n)

    bw = dist_k[:, k].copy()
    if np.any(bw <= 0):
        med = np.median(bw[bw > 0]) if np.any(bw > 0) else 1.0
        bw[bw <= 0] = med

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ii = idx_k[s:e, :k]
        dd = dist_k[s:e, :k]
        bwi = bw[s:e][:, None]
        w = np.exp(-0.5 * (dd / bwi) ** 2)

        x = X1[ii]
        yv = ydev[ii]
        sw = w.sum(1); swx = (w * x).sum(1); swxx = (w * x * x).sum(1)
        swy = (w * yv).sum(1); swxy = (w * x * yv).sum(1)

        det = sw * swxx - swx * swx
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        b0c = (swxx * swy - swx * swxy) / det
        b1c = (sw * swxy - swx * swy) / det

        xs = X1[s:e]
        hat = (swxx - 2.0 * swx * xs + sw * xs * xs) / det
        lev[s:e] = np.clip(hat, 0, 1)

        resN = yv - (b0c[:, None] + b1c[:, None] * x)
        sig2 = (w * resN ** 2).sum(1) / max(k - 2, 1)
        var_b1 = sig2 * sw / det

        b0[s:e] = b0c
        b1[s:e] = b1c
        se[s:e] = np.sqrt(np.clip(var_b1, 0, None))
        resid[s:e] = ydev[s:e] - (b0c + b1c * xs)
    return b0, b1, lev, resid, se


def gwr_univariate(tree, ydev, X1, k_candidates):
    """Fit univariate GWR over candidate bandwidths; return AICc-optimal fit."""
    n = len(ydev)
    Kq = min(max(k_candidates) + 1, n)
    dist_k, idx_k = tree.query(tree.data, k=Kq)

    best, curve = None, []
    for k in k_candidates:
        if k >= Kq:
            k = Kq - 1
        b0, b1, lev, resid, se = solve_k(idx_k, dist_k, ydev, X1, k)
        rss = float(np.sum(resid ** 2))
        trH = float(np.sum(lev))
        sigma2 = rss / n
        aicc = n * math.log(sigma2) + n * math.log(2 * math.pi) + n * (n + trH) / (n - 2 - trH)
        curve.append((k, aicc))
        if best is None or aicc < best[0]:
            best = (aicc, k, b0, b1, lev, resid, se)
    return best, curve


def moran_i(resid, coords, knn=25):
    """Residual Moran's I under a row-normalised k-nearest-neighbour weights matrix."""
    n = len(resid)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=knn + 1)
    w = np.zeros((n, knn + 1))
    w[:, 1:] = 1.0 / knn
    z = resid - resid.mean()
    Wz = (w * z[idx]).sum(1)
    return float((z * Wz).sum() / (z * z).sum())


def ols_fit(ydev, X1):
    """Global OLS reference fit; returns coefficients and residual variance."""
    A = np.column_stack([np.ones_like(X1), X1])
    beta, *_ = np.linalg.lstsq(A, ydev, rcond=None)
    resid = ydev - A @ beta
    return beta, resid


# ------------------------------------------------------------------ mapping
def build_voronoi(coords, study_area):
    """Voronoi polygons around the sample points, clipped to the study area."""
    from scipy.spatial import Voronoi
    from shapely.geometry import Point

    vor = Voronoi(coords)
    polys, pid = [], []
    for i in range(len(coords)):
        rverts = vor.regions[vor.point_region[i]]
        if not rverts or -1 in rverts:
            continue
        verts = vor.vertices[rverts]
        if len(verts) < 3:
            continue
        p = Polygon(verts)
        if not p.is_valid or p.area == 0:
            continue
        inter = p.intersection(study_area)
        if inter.is_empty:
            continue
        if inter.geom_type == "MultiPolygon":
            for g in inter.geoms:
                polys.append(g)
                pid.append(i)
        else:
            polys.append(inter)
            pid.append(i)
    pid = np.array(pid, dtype=int)

    # assign uncovered gaps to the nearest sample point
    blanks = study_area.difference(unary_union(polys))
    if not blanks.is_empty:
        bpoly = blanks.geoms if blanks.geom_type == "MultiPolygon" else [blanks]
        for b in bpoly:
            c = b.centroid
            d = np.sum((coords - np.array([c.x, c.y])) ** 2, axis=1)
            polys.append(b)
            pid = np.append(pid, int(np.argmin(d)))
    return polys, pid


def load_study_area():
    """Load and project the study-area polygon."""
    with open(STUDY_AREA, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features") or [gj]
    polys_ll = []
    for ft in feats:
        geom = shape(ft.get("geometry") or ft)
        if geom.geom_type == "Polygon":
            polys_ll.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polys_ll.extend(geom.geoms)

    def proj_poly(p):
        ext = [ll_to_xy(x, y) for x, y in p.exterior.coords]
        holes = [[ll_to_xy(x, y) for x, y in h.coords] for h in p.interiors]
        return Polygon(ext, holes)

    proj = [proj_poly(p) for p in polys_ll]
    return MultiPolygon(proj) if len(proj) > 1 else proj[0]


def t_cdf(t, df):
    """Student t CDF (SciPy if available, otherwise a normal approximation)."""
    try:
        from scipy import stats
        return stats.t.cdf(np.clip(t, -1e6, 1e6), df)
    except Exception:
        from math import erf, sqrt
        x = t / sqrt(1 + t * t / df)
        return 0.5 * (1 + erf(x / sqrt(2)))


# ------------------------------------------------------------------ main
def main():
    rows, age, indiv, env, green, outcomes, coords, n = load()

    A = design_matrix(age, indiv, env)
    print(f"confounder design matrix: {A.shape[1]} columns "
          f"(cubic age + {len(INDIV_COV)} individual + {len(ENV_COV)} environmental)")

    # residualise outcomes on the full confounder set
    ydev = {}
    for key, _col, typ in OUTCOMES:
        y = outcomes[key]
        ydev[key] = residualize_ols(y, A) if typ == "continuous" else residualize_logit(y, A)

    # residualise greenness factors symmetrically, then z-score
    fdev = {f: zscore(residualize_ols(green[f], A)) for f in FACTORS}

    study_area = load_study_area()
    polys, pid = build_voronoi(coords, study_area)
    print("voronoi polygons:", len(polys))

    tree = cKDTree(coords)
    k_candidates = sorted({max(int(round(fr * n)), 10) for fr in BW_FRACTIONS})

    models, diag_rows = {}, []
    for okey, _col, _typ in OUTCOMES:
        yv = ydev[okey]
        for f in FACTORS:
            (aicc, kbest, b0, b1, lev, resid, se), curve = gwr_univariate(tree, yv, fdev[f], k_candidates)
            tval = b1 / se
            pval = 2 * (1 - t_cdf(np.abs(tval), n - 2))
            moran = moran_i(resid, coords)
            _beta_ols, resid_ols = ols_fit(yv, fdev[f])
            med_se = float(np.nanmedian(se))
            pct_sig = float(np.mean(pval < 0.05) * 100)

            models[(okey, f)] = dict(
                beta1=b1, se=se, p=pval, resid=resid, k=kbest, aicc=aicc,
                moran=moran, med_se=med_se, pct_sig=pct_sig,
                mean_b1=float(np.mean(b1)), med_b1=float(np.median(b1)),
                aicc_curve=[(int(k), float(a)) for k, a in curve],
            )
            diag_rows.append(dict(
                outcome=okey, outcome_en=OUTCOME_EN[okey], factor=f, factor_en=FACTOR_EN[f],
                bw_k=kbest, bw_pct=round(100 * kbest / n, 1), aicc_gwr=round(aicc, 1),
                moran_I=round(moran, 4), median_se=round(med_se, 5),
                pct_sig=round(pct_sig, 1), mean_coef=round(float(np.mean(b1)), 5),
                median_coef=round(float(np.median(b1)), 5),
            ))
            print(f"{okey:4s} {f:16s} k={kbest:4d} ({100 * kbest / n:4.1f}%) "
                  f"AICc={aicc:10.1f} Moran={moran:+.4f} medSE={med_se:.5f} "
                  f"%sig={pct_sig:5.1f} meanBeta={np.mean(b1):+.4f}")

    # multicollinearity among the greenness factors
    green_corr = np.corrcoef([green[f] for f in FACTORS])
    print("\ngreenness pairwise correlation:\n", np.round(green_corr, 3))

    out = dict(points=coords, polys=polys, pid=pid, study_area=study_area, models=models,
               green_corr=green_corr, factors=FACTORS, factor_en=FACTOR_EN,
               outcome_en=OUTCOME_EN, n=n)
    with open(os.path.join(OUT_DIR, "gwr_results.pkl"), "wb") as f:
        pickle.dump(out, f)

    with open(os.path.join(OUT_DIR, "gwr_diagnostics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        w.writeheader()
        w.writerows(diag_rows)

    with open(os.path.join(OUT_DIR, "gwr_factor_correlation.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([""] + FACTORS)
        for i, fr in enumerate(FACTORS):
            w.writerow([fr] + [round(float(green_corr[i, j]), 3) for j in range(len(FACTORS))])

    print("\nsaved ->", os.path.join(OUT_DIR, "gwr_results.pkl"))


if __name__ == "__main__":
    main()
