import rasterio
import numpy as np
from rasterio.transform import from_origin
import os

def process_tif_image():
    # Input file path
    input_path = "C:/Users/DELL/Desktop/数据/各因子tif/平均绿度/平均绿度.tif"

    try:
        # Create output directory
        output_dir = os.path.dirname(input_path)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: read the TIFF and binarise it (unchanged)
        print("Reading TIFF and binarising...")
        with rasterio.open(input_path) as src:
            data = src.read(1)  # Read the first band
            meta = src.meta.copy()  # Copy metadata

            # Set cells with grayscale >= 0.2 to 1, others to 0
            binary_data = np.where(data >= 0.2, 1, 0).astype(np.uint8)

            # Save the binarised result
            binary_output_path = os.path.join(output_dir, "average_greenness_binary.tif")
            with rasterio.open(binary_output_path, 'w', **meta) as dst:
                dst.write(binary_data, 1)
            print(f"Binarisation done; result saved to: {binary_output_path}")

        # Step 2: count the number of 1s around each cell using a 33x33 window
        print("Counting with a 33x33 window...")
        with rasterio.open(binary_output_path) as src:
            binary_data = src.read(1)
            meta = src.meta.copy()

            # Get image dimensions
            rows, cols = binary_data.shape

            # Radius of the 33x33 window (33 = 2*16 + 1)
            radius = 16  # Half window width; total width = 2*radius + 1 = 33

            # Result array; max count in a 33x33 window is 33*33 = 1089, requires uint16
            count_data = np.zeros_like(binary_data, dtype=np.uint16)

            # Iterate the image with a 33x33 window (skip border pixels that cannot form a full window)
            # Loop range: i from radius to rows-1-radius, same for j
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    # Extract the 33x33 window (left-closed right-open slice, so add 1 to the end)
                    window = binary_data[i-radius:i+radius+1, j-radius:j+radius+1]
                    # Count the number of 1s in the window
                    count = np.sum(window)
                    count_data[i, j] = count

            # Border pixels (cannot form a full 33x33 window) stay 0

            # Update metadata data type (to uint16)
            meta.update(dtype=np.uint16)

            # Save the count result (updated output file name)
            count_output_path = os.path.join(output_dir, "average_greenness_33x33_count.tif")
            with rasterio.open(count_output_path, 'w', **meta) as dst:
                dst.write(count_data, 1)
            print(f"33x33 window counting done; result saved to: {count_output_path}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    process_tif_image()
