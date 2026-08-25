import rasterio
import os
import numpy as np

# Input file path
input_path = "C:/Users/mmh27/Desktop/数据/各因子tif/绿度季节变化/绿度季节变化.tif"

# Extract folder path and file name
folder_path = os.path.dirname(input_path)
file_name = os.path.basename(input_path)
name, ext = os.path.splitext(file_name)

# Output file path: append the "_grayscale_normalized" tag to the original file name
output_path = os.path.join(folder_path, f"{name}_grayscale_normalized{ext}")

try:
    # Open the input file
    with rasterio.open(input_path) as src:
        # Read the first band (band index in rasterio starts at 1)
        band1 = src.read(1)

        # Compute the min and max of the current band
        min_val = np.nanmin(band1)
        max_val = np.nanmax(band1)

        print(f"Original data range: {min_val} - {max_val}")

        # Handle the extreme case (if all values are identical)
        if max_val == min_val:
            normalized = np.ones_like(band1, dtype=np.uint16) * 500  # mid value
        else:
            # Normalise to the 1-1000 range
            # Formula: new_value = 1 + (value - min_val) * 999 / (max_val - min_val)
            normalized = 1 + (band1 - min_val) * 999 / (max_val - min_val)
            # Convert to integer
            normalized = np.round(normalized).astype(np.uint16)
            # Ensure within the 1-1000 range
            normalized = np.clip(normalized, 1, 1000)

        # Get metadata and update to single band + new data type
        meta = src.meta
        meta.update(
            count=1, dtype=np.uint16  # Change band count to 1  # Update data type to unsigned 16-bit integer
        )

        # Create the output file and write the normalised band data
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(normalized, 1)  # Write the first band

    print(f"Normalised grayscale image saved to: {output_path}")
    print(f"Normalised data range: 1 - 1000")
except Exception as e:
    print(f"Error during processing: {str(e)}")
    print("Please ensure the input path is correct and the file is a valid TIFF")
