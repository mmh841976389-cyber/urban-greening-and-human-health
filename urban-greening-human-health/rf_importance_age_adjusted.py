# -*- coding: utf-8 -*-
"""
Random forest / extra trees / XGBoost under the control-factor (offset) protocol.

Design
  - Confounders are control factors, not predictors: an OLS model (cubic age plus
    gender, BMI, smoking, drinking, physical activity and meal regularity) is fitted
    first, and the forest then predicts only the residual using the ten greenness and
    environment features.
  - Performance is evaluated on the original outcome scale:
    prediction = confounder offset + forest prediction of the residual.
  - Three validation schemes are reported: in-sample, random 5-fold CV (repeated),
    and spatial block CV (KMeans, 10 blocks x 3 seeds). The difference between the
    in-sample and spatial block estimates is reported as spatial leakage.
  - Importance: Gini (RF/ET), gain (XGB) and spatially blocked permutation importance.

Quality control
  - physiological range filter on the outcomes, then winsorising at the 1st/99th
    percentile. Both the QC-on and QC-off variants are run; QC-on is the analysis set.

Outputs: rf_metrics.csv, rf_importance.csv
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

CSV = "data/factor_table.csv"
OUT_METRICS = "output/ml/rf_metrics.csv"
OUT_IMPORTANCE = "output/ml/rf_importance.csv"

RNG, N_EST, MAX_DEPTH, MIN_LEAF = 42, 300, 30, 10

# ten model features: five greenness + five environment
FEATURES = [
    "绿度暴露_r500", "绿度强度_r500", "绿度聚集度_GAI", "季节振幅_谐波_r500", "绿度年龄_r500",
    "PM2.5_r500", "人口密度_r500", "DEM_r500", "夜光_r500", "道路距离_主干道_r500",
]

# individual confounders used as control factors (cubic age added below)
CONF = ["age", "gender", "bmi", "smoking", "drinking", "pa", "meal_reg"]

GROUP = {}
for f in ["绿度强度_r500", "绿度聚集度_GAI", "季节振幅_谐波_r500", "绿度年龄_r500"]:
    GROUP[f] = "Spatiotemporal greenness"
GROUP["绿度暴露_r500"] = "Greenness exposure"
for f in FEATURES[5:]:
    GROUP[f] = "Environment"
assert set(GROUP) == set(FEATURES)

CONTINUOUS = ["sbp", "dbp", "map", "fpg"]
BINARY = ["是否高血压", "是否高血糖"]
OUTCOMES = CONTINUOUS + BINARY

OUTCOME_EN = {"sbp": "SBP", "dbp": "DBP", "map": "MAP", "fpg": "FPG",
              "是否高血压": "Hypertension", "是否高血糖": "Hyperglycaemia"}


def apply_qc(df, qc):
    """QC-on: drop physiologically implausible values, then winsorise at 1/99."""
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
    """Confounder design matrix: cubic age plus the remaining individual confounders."""
    Xc = d[CONF].astype(float).copy()
    Xc["age2"] = Xc["age"] ** 2
    Xc["age3"] = Xc["age"] ** 3
    return Xc.fillna(Xc.median()).values


def feature_matrix(d):
    return d[FEATURES].astype(float).fillna(d[FEATURES].median()).values


def make_model(model, oob=False):
    """Instantiate one of the three ensemble learners."""
    if model == "RF":
        return RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH,
                                     min_samples_leaf=MIN_LEAF, max_features=1.0,
                                     random_state=RNG, n_jobs=-1,
                                     oob_score=oob, bootstrap=True)
    if model == "ET":
        return ExtraTreesRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH,
                                   min_samples_leaf=MIN_LEAF, max_features=1.0,
                                   random_state=RNG, n_jobs=-1,
                                   oob_score=oob, bootstrap=True)
    if model == "XGB":
        return xgb.XGBRegressor(n_estimators=N_EST, max_depth=6, learning_rate=0.03,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_weight=5, reg_lambda=1.0,
                                random_state=RNG, n_jobs=-1)
    raise ValueError(model)


def metrics(y_true, y_pred, is_binary):
    """AUC for binary outcomes, R2 for continuous ones, plus scale-dependent errors."""
    if is_binary:
        return dict(primary=roc_auc_score(y_true, y_pred), R2=np.nan,
                    MAE=mean_absolute_error(y_true, y_pred),
                    MSE=mean_squared_error(y_true, y_pred),
                    RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))
    return dict(primary=r2_score(y_true, y_pred), R2=r2_score(y_true, y_pred),
                MAE=mean_absolute_error(y_true, y_pred),
                MSE=mean_squared_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))))


def mean_ci(a):
    """Mean and normal-approximation 95% CI."""
    a = np.asarray(a, float)
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(max(len(a) - 1, 1)) if len(a) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def spatial_folds(coords, k=10, seeds=(0, 1, 2)):
    """Spatially blocked folds: KMeans clusters, one fold left out at a time."""
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


def run_one(model, Xf, Xc, y, is_binary, train_idx, test_idx=None, oob=False):
    """Fit with the confounder offset and evaluate on the original outcome scale."""
    if test_idx is None:                       # in-sample
        lr = LinearRegression().fit(Xc, y)
        c_pred = lr.predict(Xc)
        mdl = make_model(model, oob=oob).fit(Xf, y - c_pred)
        pred = c_pred + mdl.predict(Xf)
        out = metrics(y, pred, is_binary)
        out["_model"] = mdl
        return out

    lr = LinearRegression().fit(Xc[train_idx], y[train_idx])
    c_tr = lr.predict(Xc[train_idx])
    c_te = lr.predict(Xc[test_idx])
    mdl = make_model(model).fit(Xf[train_idx], y[train_idx] - c_tr)
    pred = c_te + mdl.predict(Xf[test_idx])
    return metrics(y[test_idx], pred, is_binary)


def main():
    df = pd.read_csv(CSV)
    metric_rows, importance_rows = [], []
    t_all = time.time()

    for qc in ["off", "on"]:
        d = apply_qc(df, qc)
        coords = d[["经度", "纬度"]].astype(float).values
        Xc_all = conf_matrix(d)
        Xf_all = feature_matrix(d)
        n_random_reps = 3 if qc == "on" else 1
        sp_folds = spatial_folds(coords, 10, (0, 1, 2) if qc == "on" else (0, 1))
        print(f"\n### QC={qc}  n={len(d)}  spatial folds={len(sp_folds)}")

        for oc in OUTCOMES:
            is_binary = oc in BINARY
            y = pd.to_numeric(d[oc], errors="coerce").values.astype(float)
            keep = ~np.isnan(y)
            Xf, Xc, yy, xy = Xf_all[keep], Xc_all[keep], y[keep], coords[keep]
            if len(yy) < 500:
                continue

            for model in ["RF", "ET", "XGB"]:
                t0 = time.time()
                r_ins = run_one(model, Xf, Xc, yy, is_binary, None, oob=(model in ("RF", "ET")))
                ins = r_ins["primary"]
                oob_score = getattr(r_ins["_model"], "oob_score_", np.nan)

                rnd = []
                for rep in range(n_random_reps):
                    kf = KFold(5, shuffle=True, random_state=RNG + rep)
                    for tr, te in kf.split(Xf):
                        rnd.append(run_one(model, Xf, Xc, yy, is_binary, tr, te)["primary"])

                sp = [run_one(model, Xf, Xc, yy, is_binary, tr, te)["primary"]
                      for tr, te in sp_folds]

                rm, rl, rh = mean_ci(rnd)
                sm, sl, sh = mean_ci(sp)
                metric_rows.append(dict(
                    outcome=OUTCOME_EN[oc], outcome_type="binary" if is_binary else "continuous",
                    model=model, qc=qc, primary_metric="AUC" if is_binary else "R2",
                    n=int(len(yy)), in_sample=round(ins, 4),
                    oob=round(oob_score, 4) if oob_score == oob_score else "",
                    random_cv_mean=round(rm, 4), random_cv_lo=round(rl, 4), random_cv_hi=round(rh, 4),
                    spatial_cv_mean=round(sm, 4), spatial_cv_lo=round(sl, 4), spatial_cv_hi=round(sh, 4),
                    spatial_leakage=round(ins - sm, 4),
                    in_sample_MAE=round(r_ins["MAE"], 4),
                    in_sample_MSE=round(r_ins["MSE"], 4),
                    in_sample_RMSE=round(r_ins["RMSE"], 4),
                ))
                pd.DataFrame(metric_rows).to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
                print(f"[{qc}] {OUTCOME_EN[oc]:14s} {model:3s}: in-sample={ins:.3f} "
                      f"randomCV={rm:.3f} spatialCV={sm:.3f} leakage={ins - sm:+.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

            # importance (analysis set only)
            if qc == "on":
                for model in ["RF", "ET", "XGB"]:
                    mdl = make_model(model).fit(Xf, yy - LinearRegression().fit(Xc, yy).predict(Xc))
                    if model in ("RF", "ET"):
                        imp = mdl.feature_importances_
                    else:
                        gain = mdl.get_booster().get_score(importance_type="gain")
                        imp = np.array([gain.get(f"f{j}", 0.0) for j in range(len(FEATURES))])
                    imp = imp / imp.sum() if imp.sum() > 0 else imp
                    for j, f in enumerate(FEATURES):
                        importance_rows.append(dict(outcome=OUTCOME_EN[oc], qc=qc, model=model,
                                                    feature=f, group=GROUP[f],
                                                    importance=round(float(imp[j]), 4)))

                # spatially blocked permutation importance (RF)
                perm = np.zeros(len(FEATURES))
                cnt = 0
                scorer = roc_auc_score if is_binary else r2_score
                for tr, te in spatial_folds(coords, 10, (0,)):
                    lr = LinearRegression().fit(Xc[tr], yy[tr])
                    c_tr = lr.predict(Xc[tr])
                    c_te = lr.predict(Xc[te])
                    mm = make_model("RF").fit(Xf[tr], yy[tr] - c_tr)
                    base = scorer(yy[te], c_te + mm.predict(Xf[te]))
                    for j in range(len(FEATURES)):
                        drops = []
                        for _ in range(3):
                            Xp = Xf[te].copy()
                            Xp[:, j] = np.random.permutation(Xp[:, j])
                            drops.append(base - scorer(yy[te], c_te + mm.predict(Xp)))
                        perm[j] += np.mean(drops)
                    cnt += 1
                perm /= max(cnt, 1)
                for j, f in enumerate(FEATURES):
                    importance_rows.append(dict(outcome=OUTCOME_EN[oc], qc=qc,
                                                model="RF_spatial_permutation", feature=f,
                                                group=GROUP[f], importance=round(float(perm[j]), 4)))
                pd.DataFrame(importance_rows).to_csv(OUT_IMPORTANCE, index=False, encoding="utf-8-sig")
                print(f"      importance updated ({OUTCOME_EN[oc]})", flush=True)

    print(f"\ndone in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
