# Urban Greening and Human Health

This repository contains a portion of the core code used by our research group.
It accompanies the study:

> **Satellite Datasets Reveal That Urban Greenness Impacts Human Health Through
> their Spatial and Temporal Patterns**

The scripts implement the key analytical steps behind the paper — from
remote-sensing factor preparation to spatial regression and machine-learning
interpretation — for the central urban area of Hangzhou, China.

A collection of Python scripts for analysing the relationship between urban
greenery and human health indicators (blood pressure, blood glucose, etc.) in
the central urban area of Hangzhou, China.

The workflow covers:

- Preparing remote-sensing / geographic factors (greenness, road distance, etc.)
- Age-adjusting health outcome variables
- Geographically Weighted Regression (GWR) of a target factor
- Spatial interpolation / IDW mapping of a target variable
- Random-Forest modelling and SHAP-based feature-importance ranking
- Raster clipping and grayscale normalisation utilities

## Repository structure

| File | Purpose |
| --- | --- |
| `gwr.py` | Geographically Weighted Regression (GWR) for one target factor; outputs coefficient map + report. |
| `dmap_spatial_distribution.py` | IDW interpolation of a target variable over the whole study area (blue–red gradient). |
| `qgis_road_distance_tif.py` | QGIS processing script: compute distance-to-nearest-road raster (TIFF). |
| `shap_value_ranking.py` | Random-Forest + SHAP mean(|SHAP|) ranking of original factors. |
| `shap_dependence_plot_20251013.py` | SHAP dependence plot (LOWESS trend + 95% CI) for one factor of interest. |
| `rf_feature_importance.py` | Random-Forest built-in feature-importance ranking. |
| `rf_importance_age_adjusted.py` | Polynomial age-correction, then Random-Forest importance ranking. |
| `merge_points_by_coordinate.py` | Merge duplicate sample points by coordinate and average their values. |
| `crop_raster_to_hangzhou.py` | Clip a TIFF to the Hangzhou boundary (SHP mask) and filter by threshold. |
| `greenness_range_image.py` | Binarise greenness raster and count green pixels in a 33×33 window. |
| `greenness_seasonal_grayscale.py` | Normalise greenness-seasonality raster to 1–1000 grayscale. |

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Notes:

- `mgwr` provides the GWR model (`gwr.py`).
- `qgis_road_distance_tif.py` must be run **inside QGIS** (it uses the
  `qgis` / `qgis.PyQt` APIs and is not a standalone script).

## Data & configuration

The scripts read real survey / remote-sensing data through **absolute Windows
paths** and **Chinese CSV column names** that are specific to the original
dataset. Before running any script:

1. Update the input/output paths (e.g. `csv_path`, `geojson_path`,
   `output_root`, `file_path`, `input_path`) to match your local data.
2. Keep the Chinese factor/column names (e.g. `平均绿度`, `血糖偏离度`,
   `平均动脉压偏离度`) unchanged — they are dictionary keys that must match the
   headers of your source CSV files. Only the human-readable comments, logs and
   output labels in this repository have been translated to English.

## License

No license file is included yet. Add a `LICENSE` (e.g. MIT) before publishing if
you intend to open-source the code.
