# -*- coding: utf-8 -*-
"""
Main effect estimates: greenness factors versus cardiometabolic outcomes.

  Continuous outcomes (SBP, DBP, MAP, FPG): linear regression, beta per 1 SD.
  Binary outcomes (hypertension, hyperglycaemia): logistic regression,
  odds ratio per 1 SD.

Models are adjusted for the seven individual confounders, the five environment
factors and the examination year and month. Standard errors are clustered by
residential area (KMeans, k = 100, on residential coordinates). One exposure
enters each model, because greenness intensity and greenness exposure are
strongly correlated (r = 0.914).

Inputs
  FACTOR_CSV  participant-level factor table
  OUT_DIR     output directory
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from sklearn.cluster import KMeans

FACTOR_CSV = os.environ.get("FACTOR_CSV", "因子表_最终版_v2.csv")
OUT_DIR = os.environ.get("OUT_DIR", ".")

CONT_OUT = {"sbp": "SBP (mmHg)", "dbp": "DBP (mmHg)",
            "map": "MAP (mmHg)", "fpg": "FPG (mmol/L)"}
BIN_OUT = {"是否高血压": "Hypertension", "是否高血糖": "Hyperglycaemia"}
OUTCOMES_LABEL = {**CONT_OUT, **BIN_OUT}
OUTCOMES = list(CONT_OUT) + list(BIN_OUT)

GREEN = {
    "绿度强度_r500": "Greenness intensity",
    "绿度聚集度_GAI": "Aggregation index",
    "季节振幅_谐波_r500": "Seasonal amplitude",
    "绿度年龄_r500": "Greenness age",
    "绿度暴露_r500": "Greenness exposure",
}
INDIV = ["age", "gender", "bmi", "smoking", "drinking", "pa", "meal_reg"]
EXAM = ["体检年", "体检月"]
ENV5 = ["DEM_r500", "人口密度_r500", "道路距离_主干道_r500", "PM2.5_r500", "夜光_r500"]
ADJ = INDIV + EXAM + ENV5
N_CLUSTERS = 100


def build_sample(df):
    """Complete-case sample on outcomes, adjusters and the four non-GAI factors."""
    required = OUTCOMES + ADJ + ["绿度强度_r500", "季节振幅_谐波_r500",
                                 "绿度年龄_r500", "绿度暴露_r500"] + ENV5
    sub = df[required].apply(pd.to_numeric, errors="coerce")
    return df[sub.notna().all(axis=1)].copy().reset_index(drop=True)


def fit(df, gcol, outcome, groups):
    """Return (estimate, ci_low, ci_high, p) for a 1 SD increase in gcol."""
    d = df[df[gcol].notna()].copy()
    d["exposure_z"] = (d[gcol] - d[gcol].mean()) / d[gcol].std()
    d["y"] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.dropna(subset=["y", "exposure_z"] + ADJ)
    if outcome in BIN_OUT:
        d = d[d["y"].isin([0, 1])]
    if len(d) < 100 or (outcome in BIN_OUT and d["y"].nunique() < 2):
        return None

    adj_terms = " + ".join(f"Q('{a}')" for a in ADJ)
    formula = f"Q('y') ~ exposure_z + {adj_terms}"
    g = groups.loc[d.index].values
    if outcome in BIN_OUT:
        res = logit(formula, data=d).fit(disp=0, cov_type="cluster",
                                         cov_kwds={"groups": g})
    else:
        res = ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": g})

    idx = list(res.params.index).index("exposure_z")
    coef = float(res.params.iloc[idx])
    ci = np.asarray(res.conf_int())
    p = float(res.pvalues.iloc[idx])
    if outcome in BIN_OUT:
        return (np.exp(coef), np.exp(ci[idx, 0]), np.exp(ci[idx, 1]), p, len(d))
    return (coef, float(ci[idx, 0]), float(ci[idx, 1]), p, len(d))


def main():
    df_all = pd.read_csv(FACTOR_CSV, low_memory=False)
    df = build_sample(df_all)
    print("analytic sample n =", len(df))

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    groups = pd.Series(km.fit_predict(df[["经度", "纬度"]].values), index=df.index)

    rows = []
    for gcol, name in GREEN.items():
        for outcome in OUTCOMES:
            r = fit(df, gcol, outcome, groups)
            if r is None:
                continue
            est, lo, hi, p, n = r
            rows.append({"factor": name, "factor_column": gcol, "outcome": OUTCOMES_LABEL[outcome],
                         "metric": "OR" if outcome in BIN_OUT else "beta",
                         "estimate": round(est, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4), "p": round(p, 4), "n": n})

    os.makedirs(OUT_DIR, exist_ok=True)
    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "main_effect_estimates.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print("saved", path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
