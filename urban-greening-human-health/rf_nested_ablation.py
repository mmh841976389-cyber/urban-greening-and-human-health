# -*- coding: utf-8 -*-
"""
Nested ablation of the greenness dimensions (reviewer comment R2#6).

Factor grouping used in the manuscript:
  spatiotemporal greenness (4) = intensity, aggregation index, seasonal amplitude, greenness age
  greenness exposure (1)       = 1 km residential vegetation exposure
  non-greenness environment (5)= PM2.5, population density, elevation, nighttime light, road distance

Four nested settings, all of which keep the seven individual confounders as OLS offsets:
  (A) all ten factors                       = environment + spatiotemporal + exposure
  (B) spatiotemporal greenness removed      = environment + exposure
  (C) greenness exposure removed            = environment + spatiotemporal
  (D) all greenness removed (baseline)      = environment only

Validation: spatial block CV (KMeans, 10 blocks x 3 seeds = 30 folds);
random 5-fold CV (x3 repeats) is also reported as a reference.

Output: rf_nested_ablation.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

CSV = "data/factor_table.csv"
OUT = "output/ml/rf_nested_ablation.csv"

RNG, N_EST, MAX_DEPTH, MIN_LEAF = 42, 300, 30, 10

FEATURES = [
    "绿度暴露_r500", "绿度强度_r500", "绿度聚集度_GAI", "季节振幅_谐波_r500", "绿度年龄_r500",
    "PM2.5_r500", "人口密度_r500", "DEM_r500", "夜光_r500", "道路距离_主干道_r500",
]
ENV5 = ["PM2.5_r500", "人口密度_r500", "DEM_r500", "夜光_r500", "道路距离_主干道_r500"]
SPATIOTEMPORAL4 = ["绿度强度_r500", "绿度聚集度_GAI", "季节振幅_谐波_r500", "绿度年龄_r500"]
EXPOSURE1 = ["绿度暴露_r500"]

# individual confounders used as control factors (cubic age added below)
CONF = ["age", "gender", "bmi", "smoking", "drinking", "pa", "meal_reg"]

CONTINUOUS = ["sbp", "dbp", "map", "fpg"]
BINARY = ["是否高血压", "是否高血糖"]
OUTCOMES = CONTINUOUS + BINARY

OUTCOME_EN = {"sbp": "SBP", "dbp": "DBP", "map": "MAP", "fpg": "FPG",
              "是否高血压": "Hypertension", "是否高血糖": "Hyperglycaemia"}

SPECS = {
    "A_all_ten": ENV5 + SPATIOTEMPORAL4 + EXPOSURE1,
    "B_no_spatiotemporal": ENV5 + EXPOSURE1,
    "C_no_exposure": ENV5 + SPATIOTEMPORAL4,
    "D_environment_only": ENV5,
}


def apply_qc(df, qc="on"):
    """QC-on: drop implausible values, then winsorise the outcomes at 1/99."""
    d = df.copy()
    if qc == "on":
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


def spatial_folds(coords, k=10, seeds=(0, 1, 2)):
    folds = []
    for s in seeds:
        lab = KMeans(n_clusters=k, random_state=s, n_init=10).fit(coords).labels_
        for c in range(k):
            tr = np.where(lab != c)[0]
            te = np.where(lab == c)[0]
            if len(te) < 50 or len(tr) < 200:
                continue
            folds.append((tr, te))
    return folds


def run_one(Xf, Xc, y, is_binary, train_idx, test_idx):
    """Fit with the confounder offset and score the held-out fold."""
    lr = LinearRegression().fit(Xc[train_idx], y[train_idx])
    c_tr = lr.predict(Xc[train_idx])
    c_te = lr.predict(Xc[test_idx])
    mdl = RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH,
                                min_samples_leaf=MIN_LEAF, max_features=1.0,
                                random_state=RNG, n_jobs=-1).fit(Xf[train_idx], y[train_idx] - c_tr)
    pred = c_te + mdl.predict(Xf[test_idx])
    mae = mean_absolute_error(y[test_idx], pred)
    rmse = float(np.sqrt(mean_squared_error(y[test_idx], pred)))
    score = roc_auc_score(y[test_idx], pred) if is_binary else r2_score(y[test_idx], pred)
    return score, mae, rmse


def mean_ci(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(max(len(a) - 1, 1)) if len(a) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def main():
    df = pd.read_csv(CSV)
    d = apply_qc(df, "on")
    coords = d[["经度", "纬度"]].astype(float).values
    Xc_all = conf_matrix(d)
    Xf_all = d[FEATURES].astype(float).fillna(d[FEATURES].median()).values
    col_idx = {f: i for i, f in enumerate(FEATURES)}
    sp_folds = spatial_folds(coords, 10, (0, 1, 2))

    rows = []
    t_all = time.time()
    print(f"n={len(d)}  spatial folds={len(sp_folds)}  confounder offset in every model", flush=True)

    for oc in OUTCOMES:
        is_binary = oc in BINARY
        y = pd.to_numeric(d[oc], errors="coerce").values.astype(float)
        keep = ~np.isnan(y)
        Xc = Xc_all[keep]
        yy = y[keep]
        xy = coords[keep]
        if len(yy) < 500:
            continue
        print(f"\n### {OUTCOME_EN[oc]} (n={len(yy)})", flush=True)

        for spec_name, feats in SPECS.items():
            cols = [col_idx[f] for f in feats]
            Xf = Xf_all[keep][:, cols]

            sp_score, sp_mae, sp_rmse = [], [], []
            for tr, te in sp_folds:
                s, mae, rmse = run_one(Xf, Xc, yy, is_binary, tr, te)
                sp_score.append(s)
                sp_mae.append(mae)
                sp_rmse.append(rmse)

            rnd_score, rnd_mae, rnd_rmse = [], [], []
            for rep in range(3):
                kf = KFold(5, shuffle=True, random_state=RNG + rep)
                for tr, te in kf.split(Xf):
                    s, mae, rmse = run_one(Xf, Xc, yy, is_binary, tr, te)
                    rnd_score.append(s)
                    rnd_mae.append(mae)
                    rnd_rmse.append(rmse)

            sm, sl, sh = mean_ci(sp_score)
            rm, rl, rh = mean_ci(rnd_score)
            rows.append(dict(
                outcome=OUTCOME_EN[oc], outcome_type="binary" if is_binary else "continuous",
                specification=spec_name,
                spatial_cv_primary=round(sm, 4), spatial_cv_lo=round(sl, 4), spatial_cv_hi=round(sh, 4),
                spatial_cv_MAE=round(float(np.mean(sp_mae)), 4),
                spatial_cv_RMSE=round(float(np.mean(sp_rmse)), 4),
                random_cv_primary=round(rm, 4), random_cv_lo=round(rl, 4), random_cv_hi=round(rh, 4),
                random_cv_MAE=round(float(np.mean(rnd_mae)), 4),
                random_cv_RMSE=round(float(np.mean(rnd_rmse)), 4),
                offset="seven individual confounders (OLS) in every model",
            ))
            metric = "AUC" if is_binary else "R2"
            print(f"  {spec_name:22s} spatial {metric}={sm:.4f} [{sl:.4f}, {sh:.4f}] "
                  f"MAE={np.mean(sp_mae):.4f} RMSE={np.mean(sp_rmse):.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"\ndone in {(time.time() - t_all) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
