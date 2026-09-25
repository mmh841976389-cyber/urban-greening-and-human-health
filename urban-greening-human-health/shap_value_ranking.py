# -*- coding: utf-8 -*-
"""
TreeSHAP attribution for the random forest fitted under the control-factor protocol.

The forest is trained on the residual of the outcome after regressing out the
confounders, so the SHAP values describe the model's use of the greenness and
environment features with confounding held fixed. For each outcome the four
features with the largest Gini importance are exported for plotting.

Outputs: shap_long.csv (outcome, feature, value, shap), shap_importance_rank.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

CSV = "data/factor_table.csv"
OUT_LONG = "output/ml/shap_long.csv"
OUT_RANK = "output/ml/shap_importance_rank.csv"

RNG, N_EST, MAX_DEPTH, MIN_LEAF = 42, 300, 30, 10
N_SHAP = 3000            # number of individuals sampled for the SHAP computation

FEATURES = [
    "绿度暴露_r500", "绿度强度_r500", "绿度聚集度_GAI", "季节振幅_谐波_r500", "绿度年龄_r500",
    "PM2.5_r500", "人口密度_r500", "DEM_r500", "夜光_r500", "道路距离_主干道_r500",
]

# individual confounders used as control factors (cubic age added below)
CONF = ["age", "gender", "bmi", "smoking", "drinking", "pa", "meal_reg"]

CONTINUOUS = ["sbp", "dbp", "map", "fpg"]
BINARY = ["是否高血压", "是否高血糖"]
OUTCOMES = CONTINUOUS + BINARY

OUTCOME_EN = {"sbp": "SBP", "dbp": "DBP", "map": "MAP", "fpg": "FPG",
              "是否高血压": "Hypertension", "是否高血糖": "Hyperglycaemia"}


def apply_qc(df):
    """Drop physiologically implausible values, then winsorise at 1/99."""
    d = df.copy()
    mask = np.ones(len(d), bool)
    for c, (lo, hi) in {"sbp": (80, 250), "dbp": (40, 150),
                        "map": (50, 180), "fpg": (3, 15)}.items():
        v = pd.to_numeric(d[c], errors="coerce")
        mask &= (v >= lo) & (v <= hi)
    d = d[mask].copy()
    for c in CONTINUOUS:
        v = pd.to_numeric(d[c], errors="coerce")
        lo, hi = np.nanpercentile(v, [1, 99])
        d[c] = v.clip(lo, hi)
    return d.reset_index(drop=True)


def conf_matrix(d):
    Xc = d[CONF].astype(float).copy()
    Xc["age2"] = Xc["age"] ** 2
    Xc["age3"] = Xc["age"] ** 3
    return Xc.fillna(Xc.median()).values


def feature_matrix(d):
    return d[FEATURES].astype(float).fillna(d[FEATURES].median()).values


def main():
    df = pd.read_csv(CSV)
    d = apply_qc(df)
    print(f"analysis n = {len(d)}", flush=True)

    Xf_all = feature_matrix(d)
    Xc_all = conf_matrix(d)
    rows, rank = [], []
    t0 = time.time()

    for oc in OUTCOMES:
        y = pd.to_numeric(d[oc], errors="coerce").values.astype(float)
        keep = ~np.isnan(y)
        Xf, Xc, yy = Xf_all[keep], Xc_all[keep], y[keep]

        lr = LinearRegression().fit(Xc, yy)
        resid = yy - lr.predict(Xc)
        rf = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH,
                                   min_samples_leaf=MIN_LEAF, max_features=1.0,
                                   random_state=RNG, n_jobs=-1).fit(Xf, resid)

        imp = rf.feature_importances_
        order = np.argsort(imp)[::-1]
        top4 = [int(j) for j in order[:4]]
        for r_, j in enumerate(order):
            rank.append(dict(outcome=OUTCOME_EN[oc], feature=FEATURES[int(j)],
                             importance=round(float(imp[int(j)]), 5), rank=r_ + 1))

        idx = np.random.RandomState(RNG).choice(len(Xf), size=min(N_SHAP, len(Xf)), replace=False)
        sv = np.array(shap.TreeExplainer(rf).shap_values(Xf[idx]))
        for j in top4:
            for k in range(len(idx)):
                rows.append((OUTCOME_EN[oc], FEATURES[j], float(Xf[idx][k, j]), float(sv[k, j])))

        print(f"[{OUTCOME_EN[oc]}] top4 = {[FEATURES[j] for j in top4]}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows, columns=["outcome", "feature", "value", "shap"]).to_csv(
        OUT_LONG, index=False, encoding="utf-8-sig")
    pd.DataFrame(rank).to_csv(OUT_RANK, index=False, encoding="utf-8-sig")
    print("done", flush=True)


if __name__ == "__main__":
    main()
