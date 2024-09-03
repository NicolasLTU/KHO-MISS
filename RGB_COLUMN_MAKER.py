"""
This script constantly monitors the specified directory for new averaged spectrograms (PNG 16-bit) by MISS1 or MISS2 and produces 
(300, 1, 3) RGB PNG files (8-bit unsigned integer) based on the spectral calibration data auroral emission lines at specific wavelengths.
All RGB columns are then saved timely for live or ulterior keogram creation. 60 * 24 = 1440 RGB columns per daily keogram.

Nicolas Martinez (UNIS/LTU) 2024
"""

import os
import numpy as np
from scipy import signal
from PIL import Image
from datetime import datetime, timezone
import time
from parameters import get_spectro_path, get_rgb_columns_path, get_wavelength_coeffs, get_sensitivity_coeffs, get_horizon_limits, get_emission_rows, get_binning_factor

# Paths
spectro_path = get_spectro_path()
output_folder_base = get_rgb_columns_path()

# Main processing function
def create_rgb_columns():
    global processed_images

    current_time_UT = datetime.now(timezone.utc)
    current_day = current_time_UT.day

    spectro_path_dir = os.path.join(spectro_path, current_time_UT.strftime("%Y/%m/%d"))

    if current_time_UT.day != current_day:
        processed_images.clear()
        current_day = current_time_UT.day

    ensure_directory_exists(spectro_path_dir)
    output_folder = os.path.join(output_folder_base, current_time_UT.strftime("%Y/%m/%d"))
    ensure_directory_exists(output_folder)

    matching_files = [f for f in os.listdir(spectro_path_dir) if f.endswith(".png")]

    for filename in matching_files:
        if filename in processed_images:
            continue

        png_file_path = os.path.join(spectro_path_dir, filename)

        if not verify_image_integrity(png_file_path):
            print(f"Skipping corrupted image: {filename}")
            continue

        # Read image and extract metadata
        spectro_data, metadata = read_png_with_metadata(png_file_path)

        # Extract binning factor from metadata (using binY)
        binning_factor = get_binning_factor("MISS1" if "MISS1" in filename else "MISS2")

        # Determine the spectrograph type (MISS1 or MISS2) from the filename
        if "MISS1" in filename:
            pixel_range = get_horizon_limits("MISS1")
        elif "MISS2" in filename:
            pixel_range = get_horizon_limits("MISS2")
        else:
            print(f"Unknown spectrograph type for {filename}")
            continue

        # Use calibrated and adjusted rows to create RGB columns
        RGB_image = create_rgb_column(spectro_data, *get_emission_rows("MISS1" if "MISS1" in filename else "MISS2"), binning_factor, pixel_range)
        RGB_pil_image = Image.fromarray(RGB_image.astype('uint8'))
        resized_RGB_image = RGB_pil_image.resize((1, 300), Image.Resampling.LANCZOS)

        base_filename = filename[:-4]
        output_filename = f"{base_filename[:-2]}00.png"
        output_filename_path = os.path.join(output_folder, output_filename)

        resized_RGB_image.save(output_filename_path)
        print(f"Saved RGB column image: {output_filename}")

        processed_images.add(filename)

while True:
    start_time = time.time()
    
    create_rgb_columns()
    
    elapsed_time = time.time() - start_time
    sleep_time = 60 - elapsed_time  
    if sleep_time > 0:
        time.sleep(sleep_time)
