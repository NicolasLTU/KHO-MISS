"""
This script constantly monitors the specified directory for new averaged spectrograms (PNG 16-bit) by MISS1 or MISS2 and produces
(300, 1, 3) RGB PNG files (8-bit unsigned integer) based on the spectral calibration data auroral emission lines at specific wavelengths.
All RGB columns are then saved timely for live or ulterior keogram creation. 60 * 24 = 1440 RGB columns per daily keogram.

- **Red channel**: 6300 Å (Oxygen emission line)
- **Green channel**: 5577 Å (Oxygen emission line)
- **Blue channel**: 4278 Å (Nitrogen emission line)

Nicolas Martinez (UNIS/LTU) 2024
"""

import os
import numpy as np
from scipy import signal
from PIL import Image
from datetime import datetime, timezone
import time
from parameters import parameters  # Import parameters from parameters.py

# Extract paths and constants from parameters.py
spectro_path = parameters['spectro_path']
output_folder_base = parameters['RGB_folder']
miss1_wavelength_coeffs = parameters['miss1_wavelength_coeffs']  # Coefficients as is
miss2_wavelength_coeffs = parameters['miss2_wavelength_coeffs']  # Coefficients as is
coeffs_sensitivity_miss1 = parameters['coeffs_sensitivity']['MISS1']
coeffs_sensitivity_miss2 = parameters['coeffs_sensitivity']['MISS2']
miss1_horizon_limits = parameters['miss1_horizon_limits']
miss2_horizon_limits = parameters['miss2_horizon_limits']

processed_images = set()
current_day = datetime.now(timezone.utc).day

# Function to calculate pixel position from wavelength using the spectral fit coefficients and binning factor
def calculate_pixel_position(wavelength, coeffs, max_pixel_value, binning_factor):
    """
    Calculates the pixel position corresponding to a given wavelength, 
    taking into account the spectral fit coefficients and binning factor.
    """
    # The spectral fit equation is: lambda = coeffs[0] * pixel^2 + coeffs[1] * pixel + coeffs[2]
    # Rearranged: coeffs[0] * pixel^2 + coeffs[1] * pixel + (coeffs[2] - wavelength) = 0

    # Calculate the discriminant for the quadratic equation
    discriminant = coeffs[1]**2 - 4 * coeffs[0] * (coeffs[2] - wavelength)
    
    if discriminant < 0:
        print(f"No real solution for wavelength {wavelength}, discriminant < 0.")
        return None
    
    sqrt_discriminant = np.sqrt(discriminant)

    # Calculate the positive root (pixel position > 0)
    pixel_position = (-coeffs[1] + sqrt_discriminant) / (2 * coeffs[0])

    # Adjust for binning factor (divide by binning factor to get correct row in binned image)
    binned_pixel_position = pixel_position / binning_factor

    # Check if the calculated binned pixel position is within the valid range
    if 0 <= binned_pixel_position <= max_pixel_value:
        return binned_pixel_position
    else:
        print(f"Calculated pixel position {binned_pixel_position} is out of valid range for wavelength {wavelength}")
        return None

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

    # Crop the array to the desired pixel range (columns)
    spectro_array_cropped = spectro_array[:, pixel_range[0]:pixel_range[1]]

    extracted_rows = spectro_array_cropped[start_row:end_row, :]
    processed_rows = signal.medfilt2d(extracted_rows.astype('float32'))
    averaged_row = np.mean(processed_rows, axis=0)
    return averaged_row.flatten()

# Function to create the RGB image from the extracted rows
def create_rgb_column(spectro_array, row_6300, row_5577, row_4278, binning_factor, pixel_range):
    # Process each emission line and extract the corresponding rows
    column_RED = process_emission_line(spectro_array, row_6300, binning_factor, pixel_range)
    column_GREEN = process_emission_line(spectro_array, row_5577, binning_factor, pixel_range)
    column_BLUE = process_emission_line(spectro_array, row_4278, binning_factor, pixel_range)

    # Scale each channel individually based on its own dynamic range
    def scale_channel(channel_data):
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        range_val = max_val - min_val
        if range_val == 0:
            # If the channel has constant values, set to zero
            return np.zeros_like(channel_data, dtype=np.uint8)
        else:
            scaled = ((channel_data - min_val) / range_val) * 255
            return np.clip(scaled, 0, 255).astype(np.uint8)

    scaled_red_channel = scale_channel(column_RED)
    scaled_green_channel = scale_channel(column_GREEN)
    scaled_blue_channel = scale_channel(column_BLUE)

    # Combine the individually scaled channels into the final RGB image
    true_rgb_image = np.stack((scaled_red_channel, scaled_green_channel, scaled_blue_channel), axis=-1)

    return true_rgb_image

# Process images to create RGB columns
def create_rgb_columns():
    global processed_images, current_day

    current_time_UT = datetime.now(timezone.utc)

    if current_time_UT.day != current_day:
        processed_images.clear()
        current_day = current_time_UT.day

    spectro_path_dir = os.path.join(spectro_path, current_time_UT.strftime("%Y/%m/%d"))

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
        binning_factor = extract_binning_from_metadata(metadata)

        # Determine the spectrograph type (MISS1 or MISS2) from the filename
        if "MISS1" in filename:
            pixel_range = miss1_horizon_limits
            coeffs = miss1_wavelength_coeffs
        elif "MISS2" in filename:
            pixel_range = miss2_horizon_limits
            coeffs = miss2_wavelength_coeffs
        else:
            print(f"Unknown spectrograph type for {filename}")
            continue

        # Define emission line wavelengths in Angstroms
        emission_wavelengths = [6300, 5577, 4278]  # Adjust wavelengths as needed [Å]

        # Calculate the pixel positions (rows) for each emission line
        max_pixel_value = spectro_data.shape[0] - 1  # Maximum valid pixel index (rows)
        row_6300 = calculate_pixel_position(emission_wavelengths[0], coeffs, max_pixel_value, binning_factor)
        row_5577 = calculate_pixel_position(emission_wavelengths[1], coeffs, max_pixel_value, binning_factor)
        row_4278 = calculate_pixel_position(emission_wavelengths[2], coeffs, max_pixel_value, binning_factor)

        if None in (row_6300, row_5577, row_4278):
            print(f"Could not calculate pixel positions for emission lines in {filename}")
            continue

        # Round pixel positions to nearest integer. Fit lambda(pixel_row) is starting from last pixel!!!
        row_6300 = max_pixel_value - int(round(row_6300))
        row_5577 = max_pixel_value - int(round(row_5577))
        row_4278 = max_pixel_value - int(round(row_4278))

        # Use calculated rows to create RGB columns
        RGB_image = create_rgb_column(
            spectro_data, row_6300, row_5577, row_4278, binning_factor, pixel_range
        )
        RGB_pil_image = Image.fromarray(RGB_image.astype('uint8'), mode='RGB')
        resized_RGB_image = RGB_pil_image.resize((1, 300), Image.LANCZOS)

        filename_without_ext, ext = os.path.splitext(filename)
        output_filename = f"{filename_without_ext}_RGB.png"
        output_filename_path = os.path.join(output_folder, output_filename)

        resized_RGB_image.save(output_filename_path)
        print(f"Saved RGB column image: {output_filename}")

        processed_images.add(filename)

# Main loop to constantly monitor for new images and process them
while True:
    try:
        create_rgb_columns()
    except Exception as e:
        print(f"An error occurred: {e}")
    # Synchronize to the start of the next minute
    time_now = time.time()
    sleep_time = 60 - (time_now % 60)
    time.sleep(sleep_time)
