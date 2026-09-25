# Replication code: urban greenness patterns and cardiometabolic health

Code accompanying the revised manuscript submitted to *Earth's Future*
("Satellite Datasets Reveal That Urban Greenness Impacts Human Health Through
their Spatial and Temporal Patterns"). All scripts are flat, standalone and
run from the repository root.

## File index

### Factor preparation
- `merge_points_by_coordinate.py` — build the participant factor table: extract
  the ten factors (five greenness, five environment) at each residential
  location, with year-specific matching and circular buffer means
  (300 / 500 / 1000 m).
- `gai_raster.py` — gridded Greenness Aggregation Index: landscape metrics in a
  500 m window, each residualised on PLAND, first principal component.
- `greenness_factor_maps.py` — spatial distribution of the five greenness
  factors and land cover (Figure 4). Reads the rasters in `tif/`.

### Main regression
- `main_effect_estimates.py` — fully adjusted models per 1 SD of each greenness
  factor: beta for continuous outcomes, odds ratio for binary outcomes;
  cluster-robust standard errors (KMeans, k = 100, residential areas).

### Random forest and SHAP
- `rf_importance_age_adjusted.py` — RF / ET / XGBoost under the control-factor
  (offset) protocol with random and spatial block cross-validation; confounders
  (cubic age, sex, BMI, smoking, alcohol, physical activity, meal regularity)
  enter as an OLS offset, not as forest features.
- `shap_value_ranking.py` — TreeSHAP values for the fitted random forest.
- `shap_dependence_plot_20251013.py` — SHAP dependence panels
  (outcome x feature).
- `rf_feature_importance.py` — importance and spatially validated performance
  figure for two representative outcomes.
- `rf_six_outcome_panel.py` — six-outcome extension of the figure above.
- `rf_three_model_validation.py` — three-ensemble (RF / ET / XGB) comparison
  figure.
- `rf_nested_ablation.py` — nested ablation of the greenness dimensions
  (settings A to D).
- `rf_nested_ablation_figure.py` — nested ablation figure.

### Geographically weighted regression
- `gwr.py` — univariate geographically weighted regressions with adaptive
  Gaussian kernels, bandwidth chosen by AICc; outcomes and factors are
  residualised on the full confounder set before the local regression.
- `gwr_coefficient_maps.py` — maps of the local coefficients.
- `gwr_diagnostics_figure.py` — bandwidth, diagnostic, correlation and VIF
  panels.

### Sensitivity analyses
- `gai_configuration_grid.py` — GAI across NDVI thresholds (0.20 / 0.25 / 0.30 /
  0.40) and buffer radii (300 / 500 / 1000 m), orthogonalised on PLAND, with
  health associations per configuration.
- `temporal_alignment_sensitivity.py` — exposure windows Y, Y-1, Y-3..Y-1,
  Y-5..Y-1 against the primary Y-3..Y window.

### Conceptual framework
- `confounding_dag.py` — conceptual confounding framework (directed acyclic
  graph).

## Analysis sample and variables

- Cohort: WELL-China, Hangzhou, China. The analysis sample is n = 9,212 after
  complete-case selection on outcomes and covariates (Table S1).
- Five greenness factors: greenness intensity, aggregation index (GAI), seasonal
  amplitude, greenness age, and 1 km greenness exposure.
- Five environmental covariates: PM2.5, population density, elevation, night-time
  light, distance to main roads.
- Individual confounders: age (cubic polynomial), sex, BMI, smoking, alcohol use,
  physical activity, meal regularity.

## Key design decisions

1. **Control-factor (offset) protocol.** Confounders are not entered as forest
   features. An OLS model of the confounders is fitted first and the forest then
   predicts the residual; performance is evaluated on the original outcome scale
   as `confounder offset + forest prediction of the residual`.
2. **Spatial block cross-validation.** KMeans clustering of residential
   coordinates, 10 blocks x 3 seeds = 30 folds. The difference between the
   in-sample and spatial block estimates is reported as spatial leakage.
3. **GWR with full confounder control.** Outcomes and greenness factors are both
   residualised on the full confounder set (seven individual confounders plus
   five environmental covariates) before the local regression, so the local
   coefficient is a partial association. Each greenness factor is modelled
   separately.
4. **Orthogonalised configuration.** Landscape metrics are regressed on PLAND
   before the principal component, so GAI measures configuration at a fixed
   greenness amount (correlation with PLAND is 0 by construction).

## Running the code

```
python merge_points_by_coordinate.py
python gai_raster.py
python main_effect_estimates.py
python rf_importance_age_adjusted.py
python shap_value_ranking.py
python rf_nested_ablation.py
python gwr.py
python gwr_coefficient_maps.py
python gwr_diagnostics_figure.py
python gai_configuration_grid.py
python temporal_alignment_sensitivity.py
python greenness_factor_maps.py
python rf_feature_importance.py
python rf_six_outcome_panel.py
python shap_dependence_plot_20251013.py
python rf_three_model_validation.py
python rf_nested_ablation_figure.py
python confounding_dag.py
```

## Input data

The individual-level cohort data are **not** distributed with this repository
(participant data cannot be shared). Each script reads a per-participant factor
table whose columns include the outcomes, the confounders, the five greenness
and five environmental variables, and residential coordinates. Set the input
path with the environment variables documented at the top of each script (for
example `FACTOR_CSV`, `MVC_DIR`, `OUT_DIR`), or edit the constant directly.

Column names in the source tables remain in Chinese because the exported cohort
tables use them; all identifiers, comments and output column names in the code
are in English.

## Requirements

Python 3.9+; install the dependencies with:

```
pip install -r requirements.txt
```
