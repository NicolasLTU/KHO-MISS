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
from parameters import SPECTRO_PATH, OUTPUT_FOLDER_BASE, MISS1_WAVELENGTH_COEFFS, MISS2_WAVELENGTH_COEFFS, COEFFS_SENSITIVITY_MISS1, COEFFS_SENSITIVITY_MISS2, MISS1_HORIZON_LIMITS, MISS2_HORIZON_LIMITS

processed_images = set()

def calculate_wavelength(pixel_columns, coeffs):
    return coeffs[0] + coeffs[1] * pixel_columns + coeffs[2] * (pixel_columns ** 2)

def calculate_k_lambda(wavelengths, coeffs):
    return np.polyval(coeffs, wavelengths)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def read_png_with_metadata(filename):
    with Image.open(filename) as img:
        raw_data = np.array(img)
        metadata = img.info  
    return raw_data, metadata

def extract_binning_from_metadata(metadata):
    binning_info = metadata.get("Binning", "1x1")
    binX, binY = map(int, binning_info.split('x'))
    return binY

def process_emission_line(spectro_array, emission_row, binning_factor, pixel_range):
    num_rows_to_average = max(1, int(12 / binning_factor))
    start_row = max(emission_row - num_rows_to_average // 2, 0)
    end_row = min(emission_row + num_rows_to_average // 2, spectro_array.shape[0])

    spectro_array_cropped = spectro_array[pixel_range[0]:pixel_range[1], :]
    extracted_rows = spectro_array_cropped[start_row:end_row, :]
    processed_rows = signal.medfilt2d(extracted_rows.astype('float32'))
    averaged_row = np.mean(processed_rows, axis=0)
    return averaged_row.flatten()

def create_rgb_column(spectro_array, row_630, row_558, row_428, binning_factor, pixel_range):
    column_RED = process_emission_line(spectro_array, row_630, binning_factor, pixel_range)
    column_GREEN = process_emission_line(spectro_array, row_558, binning_factor, pixel_range)
    column_BLUE = process_emission_line(spectro_array, row_428, binning_factor, pixel_range)

    RGB_image_raw = np.stack((column_RED, column_GREEN, column_BLUE), axis=-1)
    scaled_red_channel = np.clip((column_RED - np.min(column_RED)) / (np.max(column_RED) - np.min(column_RED)) * 255, 0, 255).astype(np.uint8)
    scaled_green_channel = np.clip((column_GREEN - np.min(column_GREEN)) / (np.max(column_GREEN) - np.min(column_GREEN)) * 255, 0, 255).astype(np.uint8)
    scaled_blue_channel = np.clip((column_BLUE - np.min(column_BLUE)) / (np.max(column_BLUE) - np.min(column_BLUE)) * 255, 0, 255).astype(np.uint8)
    true_rgb_image = np.stack((scaled_red_channel, scaled_green_channel, scaled_blue_channel), axis=-1)

    return true_rgb_image

def create_rgb_columns():
    global processed_images

    current_time_UT = datetime.now(timezone.utc)
    current_day = current_time_UT.day 

    spectro_path_dir = os.path.join(SPECTRO_PATH, current_time_UT.strftime("%Y/%m/%d"))

    if current_time_UT.day != current_day:
        processed_images.clear()
        current_day = current_time_UT.day

    ensure_directory_exists(spectro_path_dir)
    output_folder = os.path.join(OUTPUT_FOLDER_BASE, current_time_UT.strftime("%Y/%m/%d"))
    ensure_directory_exists(output_folder)

    for filename in os.listdir(spectro_path_dir):
        if filename.endswith('.png') and filename not in processed_images:
            file_path = os.path.join(spectro_path_dir, filename)
            if verify_image_integrity(file_path):
                spectro_array, metadata = read_png_with_metadata(file_path)
                binning_factor = extract_binning_from_metadata(metadata)

                if "MISS1" in filename:
                    row_630, row_558, row_428 = EMISSION_ROWS['MISS1']
                    pixel_range = MISS1_HORIZON_LIMITS
                elif "MISS2" in filename:
                    row_630, row_558, row_428 = EMISSION_ROWS['MISS2']
                    pixel_range = MISS2_HORIZON_LIMITS

                rgb_image = create_rgb_column(spectro_array, row_630, row_558, row_428, binning_factor, pixel_range)
                output_filename = f"{os.path.splitext(filename)[0]}_RGB.png"
                output_path = os.path.join(output_folder, output_filename)
                Image.fromarray(rgb_image).save(output_path)

                processed_images.add(filename)
                print(f"Created RGB column image for {filename}, saved as {output_filename}")

if __name__ == "__main__":
    create_rgb_columns()
