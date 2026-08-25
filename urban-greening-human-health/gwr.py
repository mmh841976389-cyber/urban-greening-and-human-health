import pandas as pd
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
import mapclassify as mc
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
import warnings
warnings.filterwarnings("ignore")

# ====================== Core parameter configuration (everything you need to edit is here) ======================
# 1. Basic configuration (target variable first for easy editing)
TARGET_VARIABLE = "平均动脉压偏离度"  # Target variable (placed first, can be edited directly)
TARGET_FACTOR = "绿度季节变化"  # The target factor to process in this run (switch manually)

# 2. Data path configuration
csv_path = "C:/Users/DELL/Desktop/杭州中心城区/杭州中心城区数据提取.csv"
geojson_path = "C:/Users/DELL/Desktop/杭州中心城区/中心城区带县.geojson"
target_crs = "EPSG:32650"  # UTM 50N projection (consistent with the GeoJSON)
output_root = "C:/Users/DELL/Desktop/杭州中心城区/GWR单因子输出"  # Output root directory

# 3. Data-processing parameters (age removal + standardisation)
POLY_DEGREE = 3  # Polynomial regression degree (fixed to 3)
GREENNESS_FACTOR = "绿度季节变化"  # Greenness factor to be z-score standardised (only applied when the target factor is this one)

# 4. Image output parameters (reversed colormap + no white margin + adjustable border)
DPI = 600  # Image resolution (high DPI)
FIG_SIZE = (15, 12)  # Image size (width x height)
CMAP = plt.cm.coolwarm  # Reversed colormap (blue = low, red = high; originally coolwarm_r, now coolwarm)
POLYGON_EDGE_WIDTH = 0.1  # Voronoi polygon edge width (default 0.1, adjust thinner/thicker freely)
STUDY_AREA_BORDER_WIDTH = 0.8  # Study-area border width (default 0.8, adjust thinner/thicker freely)

# 5. Fixed variable configuration (no need to edit)
all_factors = ["平均绿度", "绿度聚集度", "绿度季节变化", "植被年龄", "绿度范围"]  # List of all factors
# ==================================================================================

# Validate parameters
if TARGET_FACTOR not in all_factors:
    raise ValueError(f"Target factor '{TARGET_FACTOR}' is not in the factor list! Available factors: {all_factors}")
if TARGET_VARIABLE not in pd.read_csv(csv_path).columns:
    warnings.warn(f"Target variable '{TARGET_VARIABLE}' was not found in the CSV; modelling may fail, please check the column names!")

# Create output directory (named after target variable + target factor)
output_img_dir = os.path.join(output_root, f"{TARGET_VARIABLE}_{TARGET_FACTOR}")
os.makedirs(output_img_dir, exist_ok=True)
print(f"Current run configuration:")
print(f"- Target variable: {TARGET_VARIABLE} (placed first, can be edited directly)")
print(f"- Factor processed this run: {TARGET_FACTOR}")
print(f"- Polynomial regression degree: {POLY_DEGREE} (only the target factor is processed)")
print(f"- Standardised greenness factor: {GREENNESS_FACTOR} (applied only when the target factor is this one)")
print(f"- Image config: reversed colormap (blue = low, red = high), no white margin, no colorbar, no text")
print(f"- Voronoi polygon edge width: {POLYGON_EDGE_WIDTH}")
print(f"- Study-area border width: {STUDY_AREA_BORDER_WIDTH}")
print(f"- Image DPI: {DPI}")
print(f"- Output directory: {output_img_dir}")

# ---------------------- Data reading and preprocessing ----------------------
# 1. Read data + study-area boundary
df_points = pd.read_csv(csv_path, encoding="utf-8")
gdf_study_area = gpd.read_file(geojson_path).to_crs(target_crs)
study_area = unary_union(gdf_study_area.geometry)
print(f"\nNumber of GeoJSON partitions in study area: {len(gdf_study_area)}")
print(f"Number of CSV point records: {len(df_points)}")

# 2. Point-data preprocessing (keep valid points inside the study area)
gdf_points = gpd.GeoDataFrame(
    df_points,
    geometry=gpd.points_from_xy(df_points["经度"], df_points["纬度"]),
    crs="EPSG:4326"
).to_crs(target_crs)

# Keep only points inside the study area
gdf_points = gdf_points[gdf_points.within(study_area)].copy()
print(f"Number of valid points inside study area: {len(gdf_points)}")

# 3. Data completeness check (must include age column, target variable, all factors)
required_cols = [TARGET_VARIABLE] + all_factors + ["年龄"]
missing_cols = [col for col in required_cols if col not in gdf_points.columns]
if missing_cols:
    raise ValueError(f"Data is missing variables: {missing_cols}\nPlease check that the CSV column names match!")

# 4. Handle missing values and outliers
y_std = gdf_points[TARGET_VARIABLE].std()
y_mean = gdf_points[TARGET_VARIABLE].mean()
gdf_points = gdf_points[
    (gdf_points[TARGET_VARIABLE] >= y_mean - 3*y_std) &
    (gdf_points[TARGET_VARIABLE] <= y_mean + 3*y_std)
]
n = len(gdf_points)
print(f"Number of valid points after outlier removal: {n}")

# 5. Core step 1: 3rd-order polynomial regression to remove age effect for the target factor only
print(f"\nRemoving age effect from target factor '{TARGET_FACTOR}' via {POLY_DEGREE}-order polynomial regression...")
age_vals = gdf_points["年龄"].values

# Store all processed factors (target factor processed, others kept as raw)
processed_factor_cols = []
target_raw_range = None  # Original value range of the target factor
target_processed_range = None  # Range of the target factor after age removal
target_normalized_range = None  # Range of the target factor after standardisation (if any)

for factor in all_factors:
    if factor == TARGET_FACTOR:
        # Record original value range
        raw_vals = gdf_points[factor].values
        target_raw_range = (np.min(raw_vals), np.max(raw_vals))
        # Remove age effect with 3rd-order polynomial
        coeffs = np.polyfit(age_vals, raw_vals, deg=POLY_DEGREE)
        predicted_vals = np.polyval(coeffs, age_vals)
        residual_vals = raw_vals - predicted_vals  # Factor values after age removal
        # Record range after age removal
        target_processed_range = (np.min(residual_vals), np.max(residual_vals))
        processed_col = f"{factor}_processed"
        gdf_points[processed_col] = residual_vals
        processed_factor_cols.append(processed_col)
        print(f"- {factor} original value range: [{target_raw_range[0]:.4f}, {target_raw_range[1]:.4f}]")
        print(f"- {factor} range after age removal: [{target_processed_range[0]:.4f}, {target_processed_range[1]:.4f}]")
        print(f"- {factor}: age removal done (residual mean: {residual_vals.mean():.4f}, std: {residual_vals.std():.4f})")
    else:
        # Non-target factors: keep original values
        processed_col = f"{factor}_processed"
        gdf_points[processed_col] = gdf_points[factor].values
        processed_factor_cols.append(processed_col)
        print(f"- {factor}: kept as original (age effect not removed)")

# 6. Core step 2: z-score standardisation only when the target factor is a greenness factor
target_processed_col = f"{TARGET_FACTOR}_processed"
if TARGET_FACTOR == GREENNESS_FACTOR:
    print(f"\nTarget factor is a greenness factor; performing z-score standardisation...")
    greenness_vals = gdf_points[target_processed_col].values
    mean_g = np.mean(greenness_vals)
    std_g = np.std(greenness_vals)

    if std_g < 1e-6:
        greenness_z = np.zeros_like(greenness_vals)
        print(f"Warning: standard deviation of {GREENNESS_FACTOR} after age removal is near 0; all values set to 0 after standardisation")
    else:
        greenness_z = (greenness_vals - mean_g) / std_g  # z-score formula

    # Record range after standardisation
    target_normalized_range = (np.min(greenness_z), np.max(greenness_z))
    print(f"- {GREENNESS_FACTOR} range after standardisation: [{target_normalized_range[0]:.4f}, {target_normalized_range[1]:.4f}]")
    # Update standardised values
    gdf_points[target_processed_col] = greenness_z
    print(f"- {GREENNESS_FACTOR}: standardisation done (mean: {greenness_z.mean():.4f}, std: {greenness_z.std():.4f})")
else:
    print(f"\nTarget factor is not a greenness factor; no standardisation performed")

# 7. Prepare GWR modelling data (using all processed factors)
X = np.array(gdf_points[processed_factor_cols]).astype(np.float64)
y = np.array(gdf_points[TARGET_VARIABLE]).reshape((-1, 1)).astype(np.float64)
coords = list(zip(gdf_points.geometry.x, gdf_points.geometry.y))
print(f"\nNumber of coordinates used for GWR modelling: {len(coords)}")

# ---------------------- GWR model construction ----------------------
# 1. Automatically select the optimal bandwidth
sel_bw = Sel_BW(coords, y, X)
optimal_bw = sel_bw.search()
print(f"\nGWR optimal bandwidth: {optimal_bw:.2f} m")

# 2. Fit the GWR model
gwr_model = GWR(coords, y, X, optimal_bw)
gwr_results = gwr_model.fit()

# 3. Save the local coefficients of each factor (by index)
target_coeffs = None  # GWR coefficients of the target factor
for idx, factor in enumerate(all_factors):
    coeffs = gwr_results.params[:, idx]
    gdf_points[f"{factor}_coefficient"] = coeffs
    if factor == TARGET_FACTOR:
        target_coeffs = coeffs  # Record the GWR coefficients of the target factor
print(f"GWR model fitted; all factor coefficients saved")

# Compute the final range of the target factor's GWR coefficients (after age regression + standardisation if any)
final_gwr_coeff_range = (np.min(target_coeffs), np.max(target_coeffs)) if target_coeffs is not None else (0, 0)

# ---------------------- Voronoi polygon generation and gap filling ----------------------
# 1. Generate Voronoi polygons
vor = Voronoi(coords)

# 2. Convert Voronoi polygons to a GeoDataFrame
def voronoi_to_gdf(vor, crs):
    polygons = []
    for region in vor.regions:
        if not region or -1 in region:
            continue
        try:
            vertices = vor.vertices[region]
        except Exception:
            continue
        if len(vertices) < 3:
            continue
        poly = Polygon(vertices)
        if not poly.is_valid or poly.area == 0:
            continue
        polygons.append(poly)
    return gpd.GeoDataFrame(
        {"voronoi_id": range(len(polygons)), "geometry": polygons},
        crs=crs
    )

gdf_voronoi = voronoi_to_gdf(vor, target_crs)
print(f"Number of initial Voronoi polygons: {len(gdf_voronoi)}")

# 3. Filter and clip to the study area
gdf_voronoi_in_study = gdf_voronoi[gdf_voronoi.intersects(study_area)].copy()
gdf_voronoi_in_study.geometry = gdf_voronoi_in_study.geometry.apply(lambda g: g.intersection(study_area))
print(f"Number of Voronoi polygons inside study area (after clipping): {len(gdf_voronoi_in_study)}")

# 4. Join the target factor's GWR coefficients
coeff_col = f"{TARGET_FACTOR}_coefficient"  # Only the target factor's coefficient is of interest
gdf_voronoi_join = gpd.sjoin(
    gdf_voronoi_in_study,
    gdf_points[["geometry", coeff_col]],
    how="left",
    predicate="contains"
)

# Filter valid polygons
gdf_voronoi_valid = gdf_voronoi_join.dropna(subset=[coeff_col]).copy()
gdf_voronoi_valid = gdf_voronoi_valid[["voronoi_id", "geometry", coeff_col]].reset_index(drop=True)
print(f"Number of valid Voronoi polygons inside study area (with {TARGET_FACTOR} coefficient): {len(gdf_voronoi_valid)}")

# 5. Compute and fill the blank areas
voronoi_union = unary_union(gdf_voronoi_valid.geometry) if len(gdf_voronoi_valid) > 0 else Polygon()
blank_areas = study_area.difference(voronoi_union)

# Handle geometry type of blank areas
def get_blank_list(blank_geom):
    if blank_geom is None or getattr(blank_geom, "is_empty", False):
        return []
    return [blank_geom] if isinstance(blank_geom, Polygon) else list(blank_geom.geoms)

blank_list = get_blank_list(blank_areas)
print(f"Number of blank areas inside study area: {len(blank_list)}")

# Fill blank areas with the nearest coefficient
gdf_blank = gpd.GeoDataFrame(columns=["geometry", coeff_col], crs=target_crs)
if len(blank_list) > 0:
    gdf_blank = gpd.GeoDataFrame({"geometry": blank_list}, crs=target_crs)
    gdf_blank["centroid"] = gdf_blank.geometry.centroid
    gdf_blank_centroid = gpd.GeoDataFrame(
        gdf_blank[["centroid"]].rename(columns={"centroid": "geometry"}),
        crs=target_crs
    )
    # Nearest-neighbour matching
    gdf_blank_join = gpd.sjoin_nearest(
        gdf_blank_centroid,
        gdf_voronoi_valid[["geometry", coeff_col]],
        how="left",
        distance_col="dist_to_voronoi"
    )
    gdf_blank[coeff_col] = gdf_blank_join[coeff_col].values
    gdf_blank = gdf_blank[["geometry", coeff_col]].copy()

# 6. Merge complete polygons (Voronoi + blank fill)
gdf_voronoi_complete = pd.concat([
    gdf_voronoi_valid[["geometry", coeff_col]],
    gdf_blank
], ignore_index=True)
# Ensure valid geometry
gdf_voronoi_complete["geometry"] = gdf_voronoi_complete.geometry.apply(
    lambda g: g.intersection(study_area) if g is not None else g
)
gdf_voronoi_complete = gdf_voronoi_complete[~gdf_voronoi_complete.geometry.is_empty].reset_index(drop=True)
print(f"Total number of polygons finally covering the study area: {len(gdf_voronoi_complete)}")

# ---------------------- Visualisation (reversed colormap + no white margin + no colorbar + VSCode display) ----------------------
print(f"\nGenerating high-DPI image for {TARGET_VARIABLE}_{TARGET_FACTOR} (blue = low, red = high, no white margin)...")
# Create a standalone figure
fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)

# Compute coefficient range (for plotting)
coeffs_valid = gdf_voronoi_complete[coeff_col].dropna().values
local_vmin = np.min(coeffs_valid) if len(coeffs_valid) > 0 else 0
local_vmax = np.max(coeffs_valid) if len(coeffs_valid) > 0 else 1
print(f"{TARGET_FACTOR} coefficient range (for plotting): [{local_vmin:.4f}, {local_vmax:.4f}]")

# Plot polygons (reversed colormap + adjustable border)
plot_gdf = gdf_voronoi_complete.copy()
plot = plot_gdf.plot(
    column=coeff_col,
    ax=ax,
    cmap=CMAP,  # Changed to coolwarm for blue-low / red-high
    vmin=local_vmin,
    vmax=local_vmax,
    edgecolor="#E0E0E0",  # Border color (editable)
    linewidth=POLYGON_EDGE_WIDTH,  # Adjustable border width
    legend=False,  # No colorbar
    alpha=0.95,
    missing_kwds={
        "color": "#F5F5F5",
        "hatch": "///",
        "label": ""  # No text label
    }
)

# Draw study-area border (adjustable border width)
gpd.GeoDataFrame(geometry=[study_area], crs=target_crs).plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",  # Border color (editable)
    linewidth=STUDY_AREA_BORDER_WIDTH,  # Adjustable border width
    alpha=0.8
)

# Remove all text and axes completely
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(False)

# Save image (no white margin: pad_inches=0, bbox_inches="tight")
img_filename = f"{TARGET_VARIABLE}_{TARGET_FACTOR}_gwr_coefficient_map.png"
img_path = os.path.join(output_img_dir, img_filename)
plt.savefig(
    img_path,
    dpi=DPI,
    bbox_inches="tight",  # Compact layout
    pad_inches=0,  # Remove all white margins
    facecolor="white",
    edgecolor="none"
)
print(f"Image saved: {img_path}")

# Display image in VSCode
plt.show()
plt.close(fig)  # Close figure after display to free memory

# ---------------------- Result export (filenames include target variable) ----------------------
# Export GeoJSON
geojson_filename = f"{TARGET_VARIABLE}_{TARGET_FACTOR}_gwr_result.geojson"
out_geojson = os.path.join(output_img_dir, geojson_filename)
gdf_voronoi_complete.to_file(out_geojson, driver="GeoJSON", encoding="utf-8")

# Export report
report_filename = f"{TARGET_VARIABLE}_{TARGET_FACTOR}_gwr_report.txt"
out_report = os.path.join(output_img_dir, report_filename)
with open(out_report, "w", encoding="utf-8") as f:
    f.write(f"=== {TARGET_VARIABLE}_{TARGET_FACTOR} GWR Model Report ===\n")
    f.write(f"Core configuration:\n")
    f.write(f"  - Target variable: {TARGET_VARIABLE}\n")
    f.write(f"  - Target factor: {TARGET_FACTOR}\n")
    f.write(f"  - Polynomial regression degree: {POLY_DEGREE} (target factor only)\n")
    f.write(f"  - Standardised greenness factor: {GREENNESS_FACTOR} (applied only when the target factor is this one)\n")
    f.write(f"  - Image configuration:\n")
    f.write(f"    - Colormap: {CMAP.name} (blue = low, red = high)\n")
    f.write(f"    - Image features: no white margin, no colorbar, no text, high DPI\n")
    f.write(f"    - Voronoi polygon edge width: {POLYGON_EDGE_WIDTH}\n")
    f.write(f"    - Study-area border width: {STUDY_AREA_BORDER_WIDTH}\n")
    f.write(f"    - Image DPI: {DPI}\n")
    f.write(f"    - Image size: {FIG_SIZE}\n")
    f.write(f"\nData-processing notes:\n")
    f.write(f"  - Only '{TARGET_FACTOR}' had its age effect removed\n")
    f.write(f"  - Other factors use raw values (age effect not removed)\n")
    f.write(f"  - {'Target factor was z-score standardised' if TARGET_FACTOR == GREENNESS_FACTOR else 'No standardisation performed'}\n")
    f.write(f"\nData range statistics:\n")
    f.write(f"  - {TARGET_FACTOR} original value range: [{target_raw_range[0]:.4f}, {target_raw_range[1]:.4f}]\n")
    f.write(f"  - {TARGET_FACTOR} range after age removal: [{target_processed_range[0]:.4f}, {target_processed_range[1]:.4f}]\n")
    if target_normalized_range is not None:
        f.write(f"  - {TARGET_FACTOR} range after standardisation: [{target_normalized_range[0]:.4f}, {target_normalized_range[1]:.4f}]\n")
    f.write(f"  - {TARGET_FACTOR} final GWR coefficient range: [{final_gwr_coeff_range[0]:.4f}, {final_gwr_coeff_range[1]:.4f}]\n")
    f.write(f"\nData information:\n")
    f.write(f"  - Study-area GeoJSON path: {geojson_path}\n")
    f.write(f"  - Number of valid points: {n}\n")
    f.write(f"  - Number of Voronoi polygons inside study area: {len(gdf_voronoi_in_study)}\n")
    f.write(f"  - Number of blank areas: {len(blank_list)}\n")
    f.write(f"  - Total polygons finally covering the area: {len(gdf_voronoi_complete)}\n")
    f.write(f"  - GWR optimal bandwidth: {optimal_bw:.2f} m\n")
    f.write(f"\nCoefficient statistics:\n")
    f.write(f"  - Coefficient range: [{local_vmin:.4f}, {local_vmax:.4f}]\n")
    f.write(f"  - Median: {np.percentile(coeffs_valid, 50):.4f}\n")
    f.write(f"  - Interquartile range: [{np.percentile(coeffs_valid, 25):.4f}, {np.percentile(coeffs_valid, 75):.4f}]\n")
    f.write(f"  - Number of no-data areas: {np.sum(np.isnan(gdf_voronoi_complete[coeff_col].values))}\n")

#print(f"\nExport of this run finished!")
print(f"1. Image file: {img_path} (displayed in VSCode)")
#print(f"2. Data file: {out_geojson}")
#print(f"3. Report file: {out_report}")
print(f"\n=====================================")
print(f"[Key range statistics] (shown last in terminal)")
print(f"Target factor: {TARGET_FACTOR}")
print(f"Pipeline: raw values -> 3rd-order polynomial age correction -> {'z-score standardisation' if TARGET_FACTOR == GREENNESS_FACTOR else 'no standardisation'} -> GWR modelling")
print(f"1. Original value range: [{target_raw_range[0]:.4f}, {target_raw_range[1]:.4f}]")
print(f"2. Range after age removal: [{target_processed_range[0]:.4f}, {target_processed_range[1]:.4f}]")
if target_normalized_range is not None:
    print(f"3. Range after standardisation: [{target_normalized_range[0]:.4f}, {target_normalized_range[1]:.4f}]")
print(f"4. Final GWR coefficient range: [{final_gwr_coeff_range[0]:.4f}, {final_gwr_coeff_range[1]:.4f}]")
print(f"=====================================")
#print(f"\nTo change the target variable: edit 'TARGET_VARIABLE' in the core parameter section")
#print(f"To switch the target factor: edit 'TARGET_FACTOR' in the core parameter section (options: {all_factors})")
#print(f"To adjust border/boundary: edit 'POLYGON_EDGE_WIDTH' and 'STUDY_AREA_BORDER_WIDTH'")
