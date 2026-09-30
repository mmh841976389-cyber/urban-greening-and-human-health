# Spatial and Temporal Characteristics of Urban Greenness Are Associated with Cardiometabolic Indicators in Hangzhou, China

This repository contains the analysis workflow and code used for a study investigating whether the
spatial configuration and temporal dynamics of urban greenness, rather than the amount of green
cover alone, are associated with cardiometabolic health among urban residents of Hangzhou, China.

By combining annual Landsat-based greenness time series, landscape-configuration metrics, and
statistical and machine-learning analyses, the project derives four spatiotemporal greenness
dimensions alongside a conventional greenness exposure measure, and evaluates how each of them
relates to blood pressure and glycaemic outcomes in a large population cohort.

## Research Summary

Urban greening is widely promoted as a public-health intervention, yet most epidemiological
evidence treats greenness as a single quantity, the amount of vegetation within a buffer around
each residence. Green cover nevertheless varies not only in how much of it there is, but also in
how it is arranged in space and how it fluctuates through time, and these two aspects are rarely
separated from amount in cohort studies.

This study draws on the WELL-China cohort in Hangzhou and characterises each participant's
residential greenness with five measures: a conventional greenness exposure indicator, plus four
spatiotemporal dimensions describing greenness intensity, spatial aggregation, seasonal amplitude
and greenness age. Five non-greenness environmental factors are carried through the same models,
so that the greenness contributions are assessed against a common environmental baseline. The
analysis sample comprises 7,670 participants after complete-case selection and after excluding
treated participants whose measured value no longer reflects the untreated outcome (antihypertensive
medication with blood pressure below 140/90 mmHg, or glucose-lowering medication with fasting
glucose below 7.0 mmol/L).

Across six cardiometabolic outcomes, the spatiotemporal dimensions carried a larger share of the
greenness signal than exposure alone, and the full ten-factor specification outperformed both the
exposure-only and the environment-only specifications under random five-fold and spatially blocked
cross-validation. The leading factor differed by outcome rather than being the same dimension
everywhere, and SHAP dependence analysis showed threshold-like relationships rather than simple
monotonic trends. Absolute predictive performance remained modest throughout, reflecting
leakage-free cross-validation at the individual level.

These findings suggest that the spatial and temporal structure of urban greenness carries
information that greenness amount alone does not capture, and that greening policy aimed at
cardiometabolic health may need to consider configuration and temporal stability alongside
coverage.

## Project Structure

```
urban_greening/
├── 01_factor_preparation/
│   ├── 01_merge_points_by_coordinate.py
│   └── 02_gai_raster.py
├── 02_main_regression/
│   └── 01_main_effect_estimates.py
├── 03_random_forest_and_shap/
│   ├── 01_rf_importance_three_model.py
│   ├── 02_rf_nested_ablation.py
│   ├── 03_rf_nested_ablation_figure.py
│   ├── 04_shap_value_ranking.py
│   └── 05_shap_dependence_plot.py
├── 04_geographically_weighted_regression/
│   ├── 01_gwr_pipeline.py
│   ├── 02_gwr_coefficient_maps.py
│   └── 03_gwr_diagnostics_figure.py
├── 05_sensitivity_analyses/
│   ├── 01_gai_configuration_grid.py
│   └── 02_temporal_alignment_sensitivity.py
├── 06_figures/
│   ├── 00_figure4_greenness_factor_maps.py
│   ├── 01_figure5_importance_two_outcomes.py
│   ├── 02_figure5_importance_six_outcomes.py
│   ├── 03_figureS1_confounding_dag.py
│   └── 04_figureS2_three_model_validation.py
├── data/
│   └── rasters/
│       ├── a_NDVI.tif
│       ├── b_GAI.tif
│       ├── c_Amplitude.tif
│       ├── d_Age.tif
│       ├── e_Exposure.tif
│       └── f_CLCD_EPSG4326.tif
├── requirements.txt
└── README.md
```

## Workflow

```
Greenness time series and environmental rasters
  → Residential factor extraction          (01_factor_preparation)
  → Greenness aggregation index grid       (01_factor_preparation)
  → Main effect estimates                  (02_main_regression)
  → Random forest, importance and SHAP     (03_random_forest_and_shap)
  → Geographically weighted regression     (04_geographically_weighted_regression)
  → Sensitivity analyses                   (05_sensitivity_analyses)
  → Manuscript figures                     (06_figures)
```

## Module Descriptions

### 01_factor_preparation

This module builds the participant-level factor table and the gridded aggregation-index product,
and it forms the foundation of the entire analysis workflow.

- **01_merge_points_by_coordinate.py**
  - Builds the participant factor table by extracting the ten environmental factors at each
    residential location, with year-specific raster matching and circular buffer means at 300 m,
    500 m and 1000 m.
  - Produces the factor table consumed by the main regression, the machine-learning and the GWR
    modules.

- **02_gai_raster.py**
  - Computes the greenness aggregation index as a 30 m grid: landscape metrics are calculated in a
    500 m moving window on the growing-season maximum-value-composite NDVI, each metric is
    residualised on the percentage of green cells (PLAND), and the first principal component of the
    residual metrics is retained.
  - Outputs both a UTM grid and a grid resampled to the mapping reference raster.

### 02_main_regression

- **01_main_effect_estimates.py**
  - Fits fully adjusted regression models per 1 SD of each greenness factor: beta coefficients for
    the continuous outcomes and odds ratios for the binary outcomes.
  - Standard errors are cluster-robust at the residential-area level, with clusters formed by
    K-means on residential coordinates (k = 100).

### 03_random_forest_and_shap

This module contains the machine-learning analyses and their validation.

- **01_rf_importance_three_model.py**
  - Fits random forest, extremely randomised trees and gradient-boosted trees under the
    control-factor (offset) protocol: confounders enter as an OLS offset rather than as forest
    features, so the forest sees only the ten environmental factors.
  - Reports in-sample, random five-fold and spatially blocked cross-validated performance; the gap
    between in-sample and spatially blocked performance is reported as spatial leakage.
  - Outputs `rf_metrics.csv` and `rf_importance.csv`.

- **02_rf_nested_ablation.py**
  - Runs the nested ablation of the greenness dimensions. Four settings keep the same confounders as
    offsets and the same five non-greenness environment factors, differing only in which greenness
    factors are retained: all ten factors; the four spatiotemporal dimensions only; the exposure
    indicator only; and the environment baseline with all greenness removed.
  - Outputs `rf_nested_ablation.csv`.

- **03_rf_nested_ablation_figure.py**
  - Draws the nested ablation figure from the table produced above.

- **04_shap_value_ranking.py**
  - Computes TreeSHAP values for the fitted random forest and ranks the features by mean absolute
    SHAP value for each outcome.
  - Outputs `shap_long.csv` and `shap_importance_rank.csv`.

- **05_shap_dependence_plot.py**
  - Draws the SHAP dependence panels: rows are outcomes and columns are the four highest-ranked
    features of that outcome, with a binned mean trend line and its 95% confidence band.

### 04_geographically_weighted_regression

This module maps the spatial heterogeneity of each greenness-outcome association.

- **01_gwr_pipeline.py**
  - Fits univariate geographically weighted regressions with adaptive Gaussian kernels, with the
    bandwidth selected by AICc. Outcomes and greenness factors are both residualised on the full
    confounder set and standardised, so each local coefficient is a partial association per SD.
  - Each greenness factor is fitted separately, so the five factors are never entered in the same
    local model.
  - Outputs `gwr_results.pkl`, `gwr_diagnostics.csv` and `gwr_factor_correlation.csv`.

- **02_gwr_coefficient_maps.py**
  - Interpolates the local coefficients onto a regular grid by inverse-distance weighting (k = 12
    nearest observations), vector-clips the field to the study-area outline, and maps the
    coefficients and the locally significant locations.

- **03_gwr_diagnostics_figure.py**
  - Draws the appendix diagnostic panel: AICc bandwidth-selection curves, per-model diagnostics, the
    correlation matrix of the greenness factors and their variance inflation factors.

### 05_sensitivity_analyses

- **01_gai_configuration_grid.py**
  - Recomputes the aggregation index across NDVI thresholds and buffer radii, orthogonalises each
    configuration on PLAND, and repeats the health associations, so that the sensitivity of the
    results to the aggregation-index definition can be assessed.

- **02_temporal_alignment_sensitivity.py**
  - Repeats the main associations under alternative exposure windows, contrasting the primary
    alignment with the examination year alone and with lagged windows that exclude it, which
    addresses reverse temporality.

### 06_figures

- **00_figure4_greenness_factor_maps.py** - spatial distribution of the five greenness factors and
  the land-cover classification (Figure 4).
- **01_figure5_importance_two_outcomes.py** - feature importance and spatially validated performance
  for two representative outcomes.
- **02_figure5_importance_six_outcomes.py** - six-outcome extension of the figure above.
- **03_figureS1_confounding_dag.py** - conceptual confounding framework (Figure S1).
- **04_figureS2_three_model_validation.py** - three-ensemble validation figure (Figure S2).

## Environment

The workflow relies mainly on Python and on geospatial and machine-learning libraries. All analyses
run locally; no cloud platform is required.

### Tested software versions

The analyses were performed on Python 3.9+ in a 64-bit environment. The main package versions used
are listed in `requirements.txt`.

### Python dependencies

- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- shap
- statsmodels
- rasterio
- matplotlib
- geopandas
- shapely

## Installation

Clone the repository:

```
git clone https://github.com/Remote-Sensing-of-Land-Resource-Lab/urban_greening.git
cd urban_greening
```

Install the required Python dependencies:

```
pip install -r requirements.txt
```

## Input Data

The workflow combines remote-sensing, topographic, land-cover and population datasets.

The original data sources and access information are described in the manuscript Data Availability
statement. Individual scripts require processed raster, vector or tabular inputs derived from those
source datasets.

Before running a script, check:

1. the input file paths defined in the script or in the environment variables listed at its top;
2. the required attribute or column names;
3. the expected coordinate reference system;
4. the expected spatial resolution for raster inputs; and
5. the output directory and file naming conventions.

The six factor rasters in `data/rasters/` are a worked example on a common 30 m grid (EPSG:4326),
with the continuous factors winsorised and NoData set to -9999.

Individual-level cohort data are **not** distributed with this repository, because participant data
cannot be shared. Every script that needs them reads a participant-level factor table whose columns
include the six outcomes, the individual confounders, the five greenness factors, the five
environmental factors and residential coordinates. The analysis sample has n = 7,670. Set the input
path through the environment variables documented at the top of each script, or edit the constant
directly.

Column names in the source tables remain in Chinese, because the exported cohort tables use them;
all identifiers, comments and output column names in the code are in English.

## Running the Workflow

The repository is organised to follow the main analysis sequence. Run each script from the
repository root, after setting the input paths:

```
python 01_factor_preparation/01_merge_points_by_coordinate.py
python 01_factor_preparation/02_gai_raster.py
python 02_main_regression/01_main_effect_estimates.py
python 03_random_forest_and_shap/01_rf_importance_three_model.py
python 03_random_forest_and_shap/02_rf_nested_ablation.py
python 03_random_forest_and_shap/03_rf_nested_ablation_figure.py
python 03_random_forest_and_shap/04_shap_value_ranking.py
python 03_random_forest_and_shap/05_shap_dependence_plot.py
python 04_geographically_weighted_regression/01_gwr_pipeline.py
python 04_geographically_weighted_regression/02_gwr_coefficient_maps.py
python 04_geographically_weighted_regression/03_gwr_diagnostics_figure.py
python 05_sensitivity_analyses/01_gai_configuration_grid.py
python 05_sensitivity_analyses/02_temporal_alignment_sensitivity.py
python 06_figures/00_figure4_greenness_factor_maps.py
python 06_figures/01_figure5_importance_two_outcomes.py
python 06_figures/02_figure5_importance_six_outcomes.py
python 06_figures/03_figureS1_confounding_dag.py
python 06_figures/04_figureS2_three_model_validation.py
```

## Reproducing Key Analyses

The table below links the major analyses in the manuscript to the corresponding code modules.

| Analysis | Main code |
| --- | --- |
| Residential factor extraction | `01_factor_preparation/01_merge_points_by_coordinate.py` |
| Greenness aggregation index | `01_factor_preparation/02_gai_raster.py` |
| Main effect estimates | `02_main_regression/01_main_effect_estimates.py` |
| Random-forest importance and validation | `03_random_forest_and_shap/01_rf_importance_three_model.py` |
| Nested ablation of greenness dimensions | `03_random_forest_and_shap/02_rf_nested_ablation.py` |
| SHAP value ranking | `03_random_forest_and_shap/04_shap_value_ranking.py` |
| SHAP dependence analysis | `03_random_forest_and_shap/05_shap_dependence_plot.py` |
| Geographically weighted regression | `04_geographically_weighted_regression/01_gwr_pipeline.py` |
| GWR coefficient maps | `04_geographically_weighted_regression/02_gwr_coefficient_maps.py` |
| Aggregation-index configuration sensitivity | `05_sensitivity_analyses/01_gai_configuration_grid.py` |
| Exposure-window sensitivity | `05_sensitivity_analyses/02_temporal_alignment_sensitivity.py` |
| Greenness factor maps | `06_figures/00_figure4_greenness_factor_maps.py` |
| Feature importance figure | `06_figures/02_figure5_importance_six_outcomes.py` |
| Three-model validation figure | `06_figures/04_figureS2_three_model_validation.py` |

## Key Design Decisions

1. **Control-factor (offset) protocol.** Confounders are not entered as forest features. An OLS
   model of the confounders is fitted first and the forest then predicts the residual; performance
   is evaluated on the original outcome scale as confounder offset plus the forest prediction of the
   residual.
2. **Two cross-validation schemes.** Random five-fold cross-validation is repeated over several
   partitions, and spatially blocked cross-validation holds out K-means clusters of residential
   coordinates in turn. The gap between the in-sample and the spatially blocked estimates is
   reported as spatial leakage, and the two leakage figures are reported separately rather than
   combined.
3. **GWR with full confounder control.** Outcomes and greenness factors are both residualised on the
   full confounder set before the local regression, so the local coefficient is a partial
   association, and each greenness factor is modelled separately.
4. **Orthogonalised configuration metric.** Landscape metrics are regressed on PLAND before the
   principal component, so the aggregation index measures configuration at a fixed greenness amount,
   and its correlation with PLAND is zero by construction.

## Requirements

Python 3.9 or later. Install the dependencies with:

```
pip install -r requirements.txt
```

## License

Released under the MIT License.

## Contact

For questions about the code or the analysis, please open an issue in this repository.
