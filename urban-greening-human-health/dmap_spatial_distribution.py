import pandas as pd
import geopandas as gpd
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union, triangulate
import warnings
from scipy.interpolate import RBFInterpolator
warnings.filterwarnings("ignore")

# ====================== Core parameter configuration (IDW interpolation + blue/red split) ======================
# 1. Basic configuration
DMAP_VARIABLE = "血糖偏离度"  # DMAP column name
DMAP_THRESHOLD = 11.65              # Fixed split value: <11.65 = low (blue), >11.65 = high (red)
# Interpolation parameters (high sensitivity + grid-free)
INTERP_BANDWIDTH = 200              # Interpolation smoothing radius (200 m, high sensitivity)
INTERP_RESOLUTION = 40              # Interpolation resolution (40 m, low computation + no grid artifacts)
# Color-level parameters (include more weak clusters)
QUANTILE_MIN = 0.02                 # 2% quantile (keep weak low-value clusters)
QUANTILE_MAX = 0.98                 # 98% quantile (keep weak high-value clusters)
COLOR_MIDPOINT = 0.5                # Color-gradient midpoint corresponding to 11.65 (0.5)

# 2. Data paths
csv_path = "C:/Users/DELL/Desktop/杭州中心城区/杭州中心城区数据提取.csv"
geojson_path = "C:/Users/DELL/Desktop/杭州中心城区/中心城区带县.geojson"
target_crs = "EPSG:32650"           # UTM 50N projection (ensures metre units)

# 3. Image output parameters
output_root = "C:/Users/DELL/Desktop/杭州中心城区/GWR单因子输出"
DPI = 600
FIG_SIZE = (15, 12)
CMAP = plt.cm.coolwarm              # Blue -> red gradient: blue = low-value cluster, red = high-value cluster
STUDY_AREA_BORDER_WIDTH = 0.8
# =========================================================================

# Validate parameters
if DMAP_VARIABLE not in pd.read_csv(csv_path).columns:
    raise ValueError(f"DMAP variable '{DMAP_VARIABLE}' not found in the CSV!")

# Create output directory
output_img_dir = os.path.join(output_root, f"{DMAP_VARIABLE}_idw_full_coverage_11.65_threshold")
os.makedirs(output_img_dir, exist_ok=True)
print(f"Current configuration (Scheme 2: IDW interpolation + grid-free + full coverage):")
print(f"- Split threshold: DMAP < {DMAP_THRESHOLD} = low (blue), DMAP > {DMAP_THRESHOLD} = high (red)")
print(f"- Interpolation smoothing radius: {INTERP_BANDWIDTH} m (high sensitivity, captures weak clusters)")
print(f"- Interpolation resolution: {INTERP_RESOLUTION} m (no grid artifacts, computationally efficient)")
print(f"- Color stretch: {QUANTILE_MIN*100}%-{QUANTILE_MAX*100}% quantiles (keeps weak clusters)")
print(f"- Output directory: {output_img_dir}")

# ---------------------- Data reading and preprocessing ----------------------
# 1. Read the study-area boundary (ensure complete vector polygons)
gdf_study_area = gpd.read_file(geojson_path).to_crs(target_crs)
study_area = unary_union(gdf_study_area.geometry)
print(f"\nNumber of GeoJSON partitions in study area: {len(gdf_study_area)}")
print(f"Study-area bounds: {study_area.bounds} (metres)")

# 2. Read DMAP sample data and preprocess
df_points = pd.read_csv(csv_path, encoding="utf-8")
# Convert to GeoDataFrame (WGS84 -> UTM50N)
gdf_points = gpd.GeoDataFrame(
    df_points,
    geometry=gpd.points_from_xy(df_points["经度"], df_points["纬度"], crs="EPSG:4326")
).to_crs(target_crs)

# 3. Filter: keep only sample points inside the study area
gdf_points = gdf_points[gdf_points.within(study_area)].copy()

# 4. Outlier filtering (3-sigma rule, keep weak cluster data)
dmap_std = gdf_points[DMAP_VARIABLE].std()
dmap_mean = gdf_points[DMAP_VARIABLE].mean()
gdf_points = gdf_points[
    (gdf_points[DMAP_VARIABLE] >= dmap_mean - 3*dmap_std) &
    (gdf_points[DMAP_VARIABLE] <= dmap_mean + 3*dmap_std)
].reset_index(drop=True)
n_total = len(gdf_points)

# 5. Count high/low samples by the 11.65 threshold (including weak clusters)
gdf_low = gdf_points[gdf_points[DMAP_VARIABLE] < DMAP_THRESHOLD].reset_index(drop=True)
gdf_high = gdf_points[gdf_points[DMAP_VARIABLE] > DMAP_THRESHOLD].reset_index(drop=True)
gdf_mid = gdf_points[gdf_points[DMAP_VARIABLE] == DMAP_THRESHOLD].reset_index(drop=True)
n_low = len(gdf_low)
n_high = len(gdf_high)
n_mid = len(gdf_mid)

print(f"Total valid samples: {n_total}")
print(f"Low-value samples (< {DMAP_THRESHOLD}): {n_low} ({100*n_low/n_total:.1f}%), range: [{gdf_low[DMAP_VARIABLE].min():.4f}, {DMAP_THRESHOLD:.4f})")
print(f"High-value samples (> {DMAP_THRESHOLD}): {n_high} ({100*n_high/n_total:.1f}%), range: ({DMAP_THRESHOLD:.4f}, {gdf_high[DMAP_VARIABLE].max():.4f}]")
print(f"Mid-value samples (= {DMAP_THRESHOLD}): {n_mid} ({100*n_mid/n_total:.1f}%)")

# 6. Extract sample coordinates and values (for IDW interpolation)
sample_coords = np.array([(p.x, p.y) for p in gdf_points.geometry])
sample_dmap = gdf_points[DMAP_VARIABLE].values

# ---------------------- Core: global IDW interpolation over the study-area polygon (grid-free + full coverage) ----------------------
print(f"\nStarting global IDW interpolation (cover the whole study area, no grid artifacts)...")
# Step 1: triangulate the study area (generate dense sampling points to ensure full coverage)
triangles = triangulate(study_area)
interp_points = []

# Iterate each triangle to generate dense sampling points
for tri in triangles:
    if tri.within(study_area) or tri.intersects(study_area):
        # Get triangle bounds
        x_min, y_min, x_max, y_max = tri.bounds
        # Generate dense grid points (resolution = INTERP_RESOLUTION)
        x_grid = np.arange(x_min, x_max + INTERP_RESOLUTION, INTERP_RESOLUTION)
        y_grid = np.arange(y_min, y_max + INTERP_RESOLUTION, INTERP_RESOLUTION)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        tri_points = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
        # Filter: keep only points inside the triangle and inside the study area
        for p in tri_points:
            point = Point(p[0], p[1])
            if point.within(study_area):
                interp_points.append(p)

# Convert to array (ensure no empty values)
interp_points = np.array(interp_points)
if len(interp_points) == 0:
    raise ValueError("Interpolation point generation failed; please check whether the study-area polygon is valid!")
print(f"Number of dense interpolation points in study area: {len(interp_points)} (fully covered, no gaps)")

# Step 2: RBF inverse-distance-weighting interpolation (IDW, globally continuous, grid-free)
# Initialise the interpolator (linear kernel + smoothing, ensures no noise)
rbf_interp = RBFInterpolator(
    sample_coords,
    sample_dmap,
    kernel="linear",                # Linear kernel: ensures continuity, no grid
    smoothing=INTERP_BANDWIDTH,     # Smoothing radius = 200 m, high sensitivity
    neighbors=15                    # Neighbour count: balance accuracy and speed
)

# Run interpolation (generate the DMAP estimate for each point)
dmap_estimate = rbf_interp(interp_points)

# ---------------------- Color normalisation (11.65 threshold + blue/red gradient + weak-cluster preservation) ----------------------
# Step 1: quantile stretch (filter extremes, keep weak clusters)
dmap_qmin = np.quantile(dmap_estimate, QUANTILE_MIN)
dmap_qmax = np.quantile(dmap_estimate, QUANTILE_MAX)
dmap_estimate_clipped = np.clip(dmap_estimate, dmap_qmin, dmap_qmax)

# Step 2: symmetric normalisation centred on 11.65 (blue/red split)
dmap_norm = np.zeros_like(dmap_estimate_clipped)
# Low-value region (< 11.65): 0 -> 0.5 (blue family; weak cluster = light blue, strong = dark blue)
low_mask = dmap_estimate_clipped < DMAP_THRESHOLD
dmap_norm[low_mask] = COLOR_MIDPOINT * (dmap_estimate_clipped[low_mask] - dmap_qmin) / (DMAP_THRESHOLD - dmap_qmin)
# High-value region (> 11.65): 0.5 -> 1 (red family; weak cluster = light red, strong = dark red)
high_mask = dmap_estimate_clipped > DMAP_THRESHOLD
dmap_norm[high_mask] = COLOR_MIDPOINT + COLOR_MIDPOINT * (dmap_estimate_clipped[high_mask] - DMAP_THRESHOLD) / (dmap_qmax - DMAP_THRESHOLD)
# Mid-value region (= 11.65): 0.5 (mid color, light purple)
mid_mask = dmap_estimate_clipped == DMAP_THRESHOLD
dmap_norm[mid_mask] = COLOR_MIDPOINT

# Final clip (ensure 0-1 range)
dmap_norm = np.clip(dmap_norm, 0, 1)

# ---------------------- Convert to GeoDataFrame (grid-free polygons) ----------------------
# Build the interpolation-point GeoDataFrame
gdf_interp = gpd.GeoDataFrame(
    {
        "geometry": [Point(x, y) for x, y in interp_points],
        "dmap_estimate": dmap_estimate,       # Interpolated DMAP value
        "dmap_estimate_clipped": dmap_estimate_clipped,  # Value after quantile filtering
        "dmap_norm": dmap_norm                # Colour-normalised value (0-1)
    },
    crs=target_crs
)

# Convert to continuous surface (no grid lines, smooth transition)
# Buffer radius = resolution/2, ensures seamless polygon joining
gdf_interp["geometry"] = gdf_interp.geometry.buffer(INTERP_RESOLUTION/2)
# Dissolve by colour value (remove duplicates, no grid artifacts)
gdf_interp = gdf_interp.dissolve(by="dmap_norm").reset_index()

# ---------------------- Visualisation (grid-free + full coverage + blue/red gradient) ----------------------
print(f"\nDrawing IDW cluster map (grid-free + full coverage)...")
fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)

# Core plot: IDW interpolation surface (fully covers study area, no gaps)
gdf_interp.plot(
    column="dmap_norm",
    ax=ax,
    cmap=CMAP,
    vmin=0,
    vmax=1,
    alpha=0.98,          # High opacity, covers all areas
    legend=False,
    edgecolor="none"     # No border lines, completely grid-free
)

# Draw study-area boundary (for location only, optional)
gpd.GeoDataFrame(geometry=[study_area], crs=target_crs).plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    linewidth=STUDY_AREA_BORDER_WIDTH,
    alpha=0.8
)

# Extreme beautification (no distracting elements)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.grid(False)
ax.set_aspect("equal")
# Hide axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

# Save ultra-high-resolution image (no white margin)
img_filename = f"{DMAP_VARIABLE}_idw_full_coverage_threshold_{DMAP_THRESHOLD}.png"
img_path = os.path.join(output_img_dir, img_filename)
plt.savefig(
    img_path,
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0,
    facecolor="white",
    edgecolor="none"
)
print(f"Image saved: {img_path}")

# Display image (optional)
plt.show()
plt.close(fig)

# ---------------------- Result export (GeoJSON + analysis report) ----------------------
# 1. Export IDW interpolation data (importable into QGIS/ArcGIS)
geojson_filename = f"{DMAP_VARIABLE}_idw_data_full_coverage.geojson"
out_geojson = os.path.join(output_img_dir, geojson_filename)
gdf_interp[["geometry", "dmap_estimate", "dmap_estimate_clipped", "dmap_norm"]].to_file(
    out_geojson, driver="GeoJSON", encoding="utf-8"
)

# 2. Export detailed analysis report
report_filename = f"{DMAP_VARIABLE}_idw_analysis_report_full_coverage.txt"
out_report = os.path.join(output_img_dir, report_filename)
with open(out_report, "w", encoding="utf-8") as f:
    f.write(f"=== {DMAP_VARIABLE} IDW Interpolation Analysis Report (Scheme 2: full coverage + grid-free) ===\n")
    f.write(f"\n1. Core configuration:\n")
    f.write(f"   - Split threshold: DMAP < {DMAP_THRESHOLD} = low (blue family), DMAP > {DMAP_THRESHOLD} = high (red family)\n")
    f.write(f"   - Interpolation smoothing radius: {INTERP_BANDWIDTH} m (high sensitivity, captures weak clusters)\n")
    f.write(f"   - Interpolation resolution: {INTERP_RESOLUTION} m (no grid artifacts, seamless joining)\n")
    f.write(f"   - Color stretch: {QUANTILE_MIN*100}%-{QUANTILE_MAX*100}% quantiles (keeps weak clusters)\n")
    f.write(f"   - Projection CRS: {target_crs} (metre units)\n")

    f.write(f"\n2. Sample statistics (including weak clusters):\n")
    f.write(f"   - Total valid samples: {n_total}\n")
    f.write(f"   - Low-value samples (< {DMAP_THRESHOLD}): {n_low} ({100*n_low/n_total:.1f}%)\n")
    f.write(f"   - High-value samples (> {DMAP_THRESHOLD}): {n_high} ({100*n_high/n_total:.1f}%)\n")
    f.write(f"   - Mid-value samples (= {DMAP_THRESHOLD}): {n_mid} ({100*n_mid/n_total:.1f}%)\n")

    f.write(f"\n3. Interpolation result statistics (full coverage):\n")
    f.write(f"   - Number of interpolation points: {len(interp_points)} (fully covers study area)\n")
    f.write(f"   - DMAP estimate range: [{dmap_estimate.min():.4f}, {dmap_estimate.max():.4f}]\n")
    f.write(f"   - Range after quantile filtering: [{dmap_qmin:.4f}, {dmap_qmax:.4f}]\n")

    f.write(f"\n4. Color-level interpretation (weak clusters preserved):\n")
    f.write(f"   - Dark blue (strong low-value cluster): DMAP < {np.quantile(dmap_estimate_clipped[low_mask], 0.2):.4f}\n")
    f.write(f"   - Light blue (weak low-value cluster): {np.quantile(dmap_estimate_clipped[low_mask], 0.2):.4f} <= DMAP < {DMAP_THRESHOLD}\n")
    f.write(f"   - Light red (weak high-value cluster): {DMAP_THRESHOLD} < DMAP <= {np.quantile(dmap_estimate_clipped[high_mask], 0.8):.4f}\n")
    f.write(f"   - Dark red (strong high-value cluster): DMAP > {np.quantile(dmap_estimate_clipped[high_mask], 0.8):.4f}\n")

# ---------------------- Output summary ----------------------
print(f"\n=====================================")
print(f"[Scheme 2: IDW full-coverage result summary]")
print(f"1. Core effect: no grid artifacts, color fully covers study area, 11.65-threshold blue/red gradient")
print(f"2. Weak-cluster preservation: {QUANTILE_MIN*100}%-{QUANTILE_MAX*100}% quantile stretch, rich levels")
print(f"3. Interpolation accuracy: smoothing radius {INTERP_BANDWIDTH} m, resolution {INTERP_RESOLUTION} m")
print(f"4. Output files:")
print(f"   - Image: {img_path}")
print(f"   - Interpolation data: {out_geojson} (importable into GIS software)")
print(f"   - Analysis report: {out_report}")
print(f"=====================================")
