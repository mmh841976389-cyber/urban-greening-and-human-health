import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import mapping


def clip_and_filter_raster(
    input_raster_path,
    input_shp_path,
    output_path,
    filter_threshold=1999,
    nodata_value=-9999,
):
    """
    Clip a TIFF file to the shape of an SHP file and filter out values below a given threshold.

    Parameters:
    input_raster_path (str): path of the input TIFF file
    input_shp_path (str): path of the input SHP file
    output_path (str): path of the output TIFF file
    filter_threshold (int/float): filtering threshold; pixels below this value are set to NoData (default 1999)
    nodata_value (int): integer value used to represent NoData (default -9999)
    """
    # Check that the input files exist
    if not os.path.exists(input_raster_path):
        raise FileNotFoundError(f"Raster file not found: {input_raster_path}")

    if not os.path.exists(input_shp_path):
        raise FileNotFoundError(f"Vector file not found: {input_shp_path}")

    print(f"Processing file: {input_raster_path}")
    print(f"Using SHP file for clipping: {input_shp_path}")

    try:
        # Read the SHP file
        gdf = gpd.read_file(input_shp_path)

        # Ensure the SHP CRS matches the raster CRS
        with rasterio.open(input_raster_path) as src:
            raster_crs = src.crs

            # Reproject the SHP if its CRS differs from the raster
            if gdf.crs != raster_crs:
                print(
                    f"Warning: SHP CRS ({gdf.crs}) differs from raster CRS ({raster_crs}); reprojecting..."
                )
                gdf = gdf.to_crs(raster_crs)

            # Convert geometries in the GeoDataFrame to a list of GeoJSON features
            geometries = [mapping(geom) for geom in gdf.geometry]

            # Clip the raster
            print("Clipping raster data...")
            out_image, out_transform = mask(src, geometries, crop=True)

            # Get metadata and update it
            out_meta = src.meta
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "nodata": nodata_value,  # Set NoData to the specified integer
                }
            )

            # Filter out values below the threshold
            print(f"Filtering values below {filter_threshold}...")
            # Keep values >= threshold; everything else becomes NoData
            out_image = np.where(out_image >= filter_threshold, out_image, nodata_value)

            # Check that the clipped and filtered data contains valid pixels
            valid_pixels = np.sum(out_image != nodata_value)
            if valid_pixels == 0:
                raise ValueError(
                    "Cropped and filtered data contains no valid pixels; please check the threshold and clipping extent"
                )

            # Save the processed raster
            print(f"Saving result to: {output_path}")
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"Processing complete; result saved to: {output_path}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


# Set the file paths to process
tif_path = r"C:/Users/mmh27/Desktop/数据/各因子tif/小区建成时间/小区建成时间.tif"
shp_path = r"C:/Users/mmh27/Desktop/srtp/srtp中期/杭州/hangzhousheng.shp"

# Set the output path (avoid overwriting the original data)
output_dir = os.path.dirname(tif_path)
output_name = os.path.basename(tif_path).replace(".tif", "_cropped_filtered.tif")
output_path = os.path.join(output_dir, output_name)

# Run the clipping and filtering
try:
    clip_and_filter_raster(
        tif_path, shp_path, output_path, filter_threshold=1999, nodata_value=-9999
    )
except Exception as e:
    print(f"Program execution failed: {str(e)}")
    import traceback

    traceback.print_exc()
