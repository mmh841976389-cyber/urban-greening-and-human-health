# -*- coding: utf-8 -*-
"""
Temporal-alignment sensitivity of the greenness intensity association.

Greenness intensity (500 m buffer mean growing-season NDVI) is assigned under
five alignment schemes, all derived from the same per-year growing-season NDVI
rasters (2013-2019):
  Main (Y-3..Y)      four-year window including the examination year (primary)
  Examination year Y the examination year only
  Lagged Y-1         one year before the examination year
  Lagged Y-3..Y-1    the three years before the examination year
  Lagged Y-5..Y-1    the five years before the examination year (available years only)

Models are fully adjusted (seven individual confounders, examination month and
year, five environmental covariates) with residence-based cluster-robust
standard errors (KMeans, 100 clusters). Effects are reported per one SD of
intensity: beta for continuous outcomes and odds ratios for binary outcomes.

The lagged windows exclude the examination year, so they address reverse
causation: associations that persist cannot be driven by contemporaneous
health-related changes in the environment.

Output: temporal_alignment_sensitivity.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

FACTOR_CSV = os.environ.get("FACTOR_CSV", "data/factor_table.csv")
EXPOSURE_CSV = os.environ.get("EXPOSURE_CSV", "data/exposure_by_year.csv")
OUT_DIR = os.environ.get("SENS_OUT", "output/sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

INDIV = ["age", "gender", "bmi", "smoking", "drinking", "pa", "meal_reg", "体检月"]
ENV = ["DEM_r500", "人口密度_r500", "道路距离_主干道_r500", "PM2.5_r500", "夜光_r500"]
YEAR = ["体检年"]

CONTINUOUS_OUT = ["sbp", "dbp", "map", "fpg"]
BINARY_OUT = ["是否高血压", "是否高血糖"]
OUTCOME_EN = {"sbp": "SBP", "dbp": "DBP", "map": "MAP", "fpg": "FPG",
              "是否高血压": "Hypertension", "是否高血糖": "Hyperglycaemia"}

YEARS = list(range(2013, 2020))
GS_COLS = {y: f"GS_y{y}_r500" for y in YEARS}

WINDOWS = ["Main (Y-3..Y)", "Examination year Y", "Lagged Y-1",
           "Lagged Y-3..Y-1", "Lagged Y-5..Y-1"]


def build_windows(e):
    """Build the five alignment windows from the per-year NDVI columns."""
    out = {
        "Main (Y-3..Y)": pd.to_numeric(e["NDVI_GS_4yrW_r500"], errors="coerce"),
        "Examination year Y": pd.to_numeric(e["NDVI_GS_同年_r500"], errors="coerce"),
    }
    gs = {y: pd.to_numeric(e[GS_COLS[y]], errors="coerce").values.astype(float) for y in YEARS}
    M = np.column_stack([gs[y] for y in YEARS])
    yr_idx = {y: i for i, y in enumerate(YEARS)}
    exam_year = pd.to_numeric(e["体检年"], errors="coerce").values.astype(float)

    def window_mean(lo_off, hi_off):
        res = np.full(len(e), np.nan)
        for i in range(len(e)):
            yY = exam_year[i]
            if not np.isfinite(yY):
                continue
            yrs = [int(yY) + off for off in range(lo_off, hi_off + 1)]
            idxs = [yr_idx[y] for y in yrs if y in yr_idx]
            if idxs:
                vals = M[i, idxs]
                if np.any(~np.isnan(vals)):
                    res[i] = np.nanmean(vals)
        return res

    out["Lagged Y-1"] = window_mean(-1, -1)
    out["Lagged Y-3..Y-1"] = window_mean(-3, -1)
    out["Lagged Y-5..Y-1"] = window_mean(-5, -1)

    w = pd.DataFrame(out)
    w["ID"] = e["ID"].astype(str)
    return w


def fit(d, exposure, outcome, covs):
    """Fit one fully adjusted model and return the exposure effect."""
    use = [exposure] + [c for c in covs if c in d.columns and c != exposure]
    sub = d[use + [outcome, "_clus"]].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 500:
        return None

    is_binary = outcome in BINARY_OUT
    if is_binary and sub[outcome].nunique() < 2:
        return None

    X = sub[use].astype(float)
    sd = X[exposure].std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return None
    X[exposure] = (X[exposure] - X[exposure].mean()) / sd     # effect per 1 SD
    X = sm.add_constant(X)
    y = sub[outcome].astype(float)
    groups = sub["_clus"].values

    try:
        if is_binary:
            r = sm.GLM(y, X, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": groups})
            eff = float(np.exp(r.params[exposure]))
            lo, hi = np.exp(r.conf_int().loc[exposure].values)
            return dict(n=len(sub), estimate="OR", eff=eff, lo=lo, hi=hi,
                        p=float(r.pvalues[exposure]))
        r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        eff = float(r.params[exposure])
        lo, hi = r.conf_int().loc[exposure].values
        return dict(n=len(sub), estimate="beta", eff=eff, lo=lo, hi=hi,
                    p=float(r.pvalues[exposure]))
    except Exception:
        return None


def fmt(r):
    if r is None:
        return "-"
    sig = "*" if r["p"] < 0.05 else ""
    return f"{r['eff']:.3f} ({r['lo']:.3f}, {r['hi']:.3f}){sig}"


def main():
    f = pd.read_csv(FACTOR_CSV, low_memory=False)
    e = pd.read_csv(EXPOSURE_CSV, low_memory=False)
    f["ID"] = f["ID"].astype(str)
    e["ID"] = e["ID"].astype(str)

    w = build_windows(e)
    d = f.merge(w, on="ID", how="inner").set_index("ID")

    # fixed analytic sample: complete cases on confounders, outcomes and environment
    required = INDIV + ENV + YEAR + CONTINUOUS_OUT + BINARY_OUT
    ok = d[required].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    dbase = d[ok].copy()
    print("analytic sample n =", len(dbase))

    clus = pd.Series(KMeans(n_clusters=100, random_state=42, n_init=10)
                     .fit_predict(dbase[["经度", "纬度"]].values), index=dbase.index)
    dbase["_clus"] = clus

    rows = []
    for wn in WINDOWS:
        rec = {"alignment_window": wn}
        for oc in CONTINUOUS_OUT + BINARY_OUT:
            r = fit(dbase, wn, oc, INDIV + ENV + YEAR)
            rec[OUTCOME_EN[oc]] = fmt(r)
            rec[OUTCOME_EN[oc] + "_p"] = None if r is None else r["p"]
        rows.append(rec)

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    out_csv = os.path.join(OUT_DIR, "temporal_alignment_sensitivity.csv")
    tab.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("saved", out_csv)


if __name__ == "__main__":
    main()
