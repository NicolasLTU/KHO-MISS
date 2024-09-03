"""
This script constantly monitors the specified directory for new averaged spectrograms (PNG 16-bit) by MISS1 or MISS2 and produces 
(300, 1, 3) RGB PNG files (8-bit unsigned integer) based on the spectral calibration data auroral emission lines at specific wavelengths.
All RGB columns are then saved timely for live or ulterior keogram creation. 60 * 24 = 1440 RGB columns per daily keogram.

Nicolas Martinez (UNIS/LTU) 2024
"""

import os
import numpy as np
from scipy import signal
from PIL import Image, PngImagePlugin
from datetime import datetime, timezone, timedelta
import time

# Wavelength calibration coefficients
miss1_wavelength_coeffs = [4217.273360, 2.565182, 0.000170]
miss2_wavelength_coeffs = [4088.509, 2.673936, 1.376154e-4]

# Polynomial coefficients for MISS1 and MISS2 K(lambda)
coeffs_sensitivity_miss1 = [6.649e+01, -2.922e-02, 1.889e-07, 1.752e-09, -3.134e-13, 1.681e-17]
coeffs_sensitivity_miss2 = [-1.378573e-16, 4.088257e-12, -4.806258e-08, 2.802435e-04, -8.109943e-01, 9.329611e+02]

# Horizon limits for MISS1 and MISS2
miss1_horizon_limits = (280, 1140)
miss2_horizon_limits = (271, 1116)

processed_images = set()

# Function to calculate wavelengths
def calculate_wavelength(pixel_columns, coeffs):
    return coeffs[0] + coeffs[1] * pixel_columns + coeffs[2] * (pixel_columns ** 2)

# Function to calculate K(lambda)
def calculate_k_lambda(wavelengths, coeffs):
    return np.polyval(coeffs, wavelengths)

# Ensure directory exists before trying to open or save
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Verify image integrity
def verify_image_integrity(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify() 
        with Image.open(file_path) as img:
            img.load()  
        return True 
    except Exception as e:
        print(f"Corrupted PNG detected: {file_path} - {e}")
        return False

# Read the PNG image and extract metadata
def read_png_with_metadata(filename):
    with Image.open(filename) as img:
        raw_data = np.array(img)
        metadata = img.info  # Extract metadata
    return raw_data, metadata

# Extract binning factor from metadata (use binY)
def extract_binning_from_metadata(metadata):
    binning_info = metadata.get("Binning", "1x1")
    binX, binY = map(int, binning_info.split('x'))
    return binY

# Process and average emission line rows
def process_emission_line(spectro_array, emission_row, binning_factor, pixel_range):
    num_rows_to_average = max(1, int(12 / binning_factor))
    start_row = max(emission_row - num_rows_to_average // 2, 0)
    end_row = min(emission_row + num_rows_to_average // 2, spectro_array.shape[0])

    # Crop the array to the desired pixel range
    spectro_array_cropped = spectro_array[pixel_range[0]:pixel_range[1], :]

    extracted_rows = spectro_array_cropped[start_row:end_row, :]
    processed_rows = signal.medfilt2d(extracted_rows.astype('float32'))
    averaged_row = np.mean(processed_rows, axis=0)
    return averaged_row.flatten()

# Function to create the RGB image from the extracted rows
def create_rgb_column(spectro_array, row_630, row_558, row_428, binning_factor, pixel_range):
    # Process each emission line and extract the corresponding rows
    column_RED = process_emission_line(spectro_array, row_630, binning_factor, pixel_range)
    column_GREEN = process_emission_line(spectro_array, row_558, binning_factor, pixel_range)
    column_BLUE = process_emission_line(spectro_array, row_428, binning_factor, pixel_range)

    # Combine the channels into an RGB image
    RGB_image_raw = np.stack((column_RED, column_GREEN, column_BLUE), axis=-1)

    # Scale each channel individually based on its own dynamic range
    scaled_red_channel = np.clip((column_RED - np.min(column_RED)) / (np.max(column_RED) - np.min(column_RED)) * 255, 0, 255).astype(np.uint8)
    scaled_green_channel = np.clip((column_GREEN - np.min(column_GREEN)) / (np.max(column_GREEN) - np.min(column_GREEN)) * 255, 0, 255).astype(np.uint8)
    scaled_blue_channel = np.clip((column_BLUE - np.min(column_BLUE)) / (np.max(column_BLUE) - np.min(column_BLUE)) * 255, 0, 255).astype(np.uint8)

    # Combine the individually scaled channels into the final RGB image
    true_rgb_image = np.stack((scaled_red_channel, scaled_green_channel, scaled_blue_channel), axis=-1)

    return true_rgb_image

# Process images to create RGB columns
def create_rgb_columns(config):  # Config passed in from the main script
    global processed_images

    current_time_UT = datetime.now(timezone.utc)
    current_day = current_time_UT.day 

    spectro_path_dir = os.path.join(config['spectro_path'], current_time_UT.strftime("%Y/%m/%d"))

    if current_time_UT.day != current_day:
        processed_images.clear()
        current_day = current_time_UT.day

    ensure_directory_exists(spectro_path_dir)
    output_folder = os.path.join(config['output_folder_base'], current_time_UT.strftime("%Y/%m/%d"))
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
        binning_factor = extract_binning_from_metadata(metadata)

        # Determine the spectrograph type (MISS1 or MISS2) from the filename
        if "MISS1" in filename:
            pixel_range = miss1_horizon_limits
        elif "MISS2" in filename:
            pixel_range = miss2_horizon_limits
        else:
            print(f"Unknown spectrograph type for {filename}")
            continue

        # Use calibrated and adjusted rows to create RGB columns
        RGB_image = create_rgb_column(spectro_data, 724, 723, 140, binning_factor, pixel_range)  # Adjusted rows used here
        RGB_pil_image = Image.fromarray(RGB_image.astype('uint8'))
        resized_RGB_image = RGB_pil_image.resize((1, 300), Image.Resampling.LANCZOS)

        base_filename = filename[:-4]
        output_filename = f"{base_filename[:-2]}00.png"
        output_filename_path = os.path.join(output_folder, output_filename)

        resized_RGB_image.save(output_filename_path)
        print(f"Saved RGB column image: {output_filename}")

        processed_images.add(filename)

# Main loop to constantly monitor for new images and process them
while True:
    start_time = time.time()  # Record the start time of the loop
    
    create_rgb_columns(config)  # Ensure the config is passed here
    
    # Calculate the time to wait until the start of the next minute
    elapsed_time = time.time() - start_time
    sleep_time = 60 - elapsed_time  
    if sleep_time > 0:
        time.sleep(sleep_time)

# PATHS TO INPUT AND OUTPUT - for standalone use/test only
#home_dir = os.path.expanduser("~")
#spectro_path = os.path.join(home_dir, ".venvMISS2", "MISS2", "Captured_PNG", "averaged_PNG")
#output_folder_base = os.path.join(home_dir, ".venv

