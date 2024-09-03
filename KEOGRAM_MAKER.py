"""
This program uses RGB image-columns generated every minute using the spectrograms captured by MISS* to update a daily keogram.
At 00:00 UTC, a new keogram for the day is created (empty). At 00:05 UTC, the previous day's keogram receives its last update,
and the new day's keogram receives its first update.

The script ensures only available past data is used for subplot analysis. The RGB channels are named according to the three main
emission lines of the aurora: 4278 Å (Blue), 5577 Å (Green), and 6300 Å (Red).

Author: Nicolas Martinez (UNIS/LTU)
Last update: August 2024
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import time
import parameters as params  # Import parameters from the parameters.py file

# Set spectrograph name from parameters.py
spectrograph = params.SPECTROGRAPH

# Base directory and sensitivity coefficients from parameters.py
rgb_dir_base = params.RGB_DIR_BASE
output_dir = params.OUTPUT_DIR
coeffs_sensitivity = params.SENSITIVITY_COEFFS

# Define auroral emission line wavelengths in Ångström (fixed values)
emission_wavelengths = {
    'blue': 4278,  # Blue auroral emission line
    'green': 5577, # Green auroral emission line
    'red': 6300    # Red auroral emission line
}

num_pixels_y = 300  # Number of pixels along the y-axis (for RGB with 300 rows)
num_minutes = 24 * 60  # Total number of minutes in a day

def verify_image_integrity(file_path):
    """
    Verify the integrity of an image file.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception as e:
        print(f"Corrupted PNG detected: {file_path} - {e}")
        return False

def initialise_keogram():
    """
    Initialise an empty keogram with white pixels.
    """
    return np.full((num_pixels_y, num_minutes, 3), 255, dtype=np.uint8)

def load_existing_keogram(output_dir, spectrograph):
    """
    Load an existing keogram or create a new one if none exists.
    """
    current_utc_time = datetime.now(timezone.utc)
    current_date = current_utc_time.strftime('%Y/%m/%d')
    keogram_path = os.path.join(output_dir, current_date, f'{spectrograph}-keogram-{current_utc_time.strftime("%Y%m%d")}.png')

    if os.path.exists(keogram_path):
        with Image.open(keogram_path) as img:
            keogram = np.array(img)
        if keogram.shape != (num_pixels_y, num_minutes, 3):
            keogram = initialise_keogram()
            last_processed_minute = 0
        else:
            last_processed_minute = np.max(np.where(np.any(keogram != 255, axis=0))[0])
        return keogram, last_processed_minute
    else:
        return initialise_keogram(), 0

def save_keogram(keogram, output_dir, spectrograph):
    """
    Save the updated keogram.
    """
    current_utc_time = datetime.now(timezone.utc)
    current_date_str = current_utc_time.strftime('%Y%m%d')
    current_date_dir = os.path.join(output_dir, current_utc_time.strftime('%Y/%m/%d'))
    os.makedirs(current_date_dir, exist_ok=True)
    keogram_to_save = keogram[:, :num_minutes, :]
    keogram_filename = os.path.join(current_date_dir, f'{spectrograph}-keogram-{current_date_str}.png')
    Image.fromarray(keogram_to_save).save(keogram_filename)

def apply_sensitivity_correction(radiance, wavelength, coeffs):
    """
    Apply sensitivity correction to radiance using the respective coefficients.
    """
    correction_factor = np.polyval(coeffs, wavelength)
    return radiance * correction_factor

def convert_8bit_to_16bit(value):
    """
    Convert 8-bit image data to 16-bit.
    """
    return value * (65535 / 255)

def rgb_to_radiance_kR(rgb_data, wavelengths, sensitivity_coeffs):
    """
    Convert RGB data to radiance in kR.
    """
    radiance = np.zeros_like(rgb_data, dtype=np.float64)
    for i, color in enumerate(['blue', 'green', 'red']):
        wavelength = wavelengths[color]
        radiance_16bit = convert_8bit_to_16bit(rgb_data[:, :, i])
        corrected_radiance = apply_sensitivity_correction(radiance_16bit, wavelength, sensitivity_coeffs)
        radiance[:, :, i] = corrected_radiance / 1000
    return radiance

def add_rgb_columns(keogram, base_dir, last_processed_minute, spectrograph):
    """
    Add RGB columns to the keogram and convert to radiance in kR.
    """
    now_UT = datetime.now(timezone.utc)
    current_minute_of_the_day = now_UT.hour * 60 + now_UT.minute
    found_minutes = set()

    today_RGB_dir = os.path.join(base_dir, now_UT.strftime("%Y/%m/%d"))

    if not os.path.exists(today_RGB_dir):
        print(f"No directory found for today's date ({today_RGB_dir}). Proceeding with black RGB data.")
        today_RGB_dir = None  # Indicate that the directory doesn't exist

    for minute in range(last_processed_minute + 1, current_minute_of_the_day + 1):
        timestamp = now_UT.replace(hour=minute // 60, minute=minute % 60, second=0, microsecond=0)
        filename = f"{spectrograph}-{timestamp.strftime('%Y%m%d-%H%M00')}.png"
        file_path = os.path.join(today_RGB_dir, filename) if today_RGB_dir else None

        if file_path and os.path.exists(file_path) and verify_image_integrity(file_path):
            try:
                rgb_data = np.array(Image.open(file_path))
                if rgb_data.shape != (num_pixels_y, 1, 3):
                    print(f"Unexpected image shape {rgb_data.shape} for {filename}. Skipping this image.")
                    continue

                radiance_data = rgb_to_radiance_kR(rgb_data, emission_wavelengths, coeffs_sensitivity[spectrograph])
                keogram[:, minute:minute+1, :] = radiance_data.astype(np.uint8)
                found_minutes.add(minute)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        else:
            if current_minute_of_the_day - minute > 4:
                keogram[:, minute:minute+1, :] = np.zeros((num_pixels_y, 1, 3), dtype=np.uint8)
    return keogram

def save_keogram_with_subplots(keogram, output_dir, spectrograph):
    """
    Save the keogram with subplot analysis.
    """
    current_utc_time = datetime.now(timezone.utc)
    current_date_str = current_utc_time.strftime('%Y%m%d')
    current_date_dir = os.path.join(output_dir, current_utc_time.strftime('%Y/%m/%d'))
    os.makedirs(current_date_dir, exist_ok=True)

    non_empty_columns = np.where(np.any(keogram != 255, axis=0))[0]

    if non_empty_columns.size == 0:
        print("Error: Keogram contains no data. Creating an empty image.")
        Image.fromarray(keogram[:, :num_minutes, :]).save(os.path.join(current_date_dir, f'{spectrograph}-keogram-{current_date_str}.png'))
        return

    for start in range(0, non_empty_columns[-1] + 1, 600):
        end = min(start + 600, non_empty_columns[-1] + 1)
        plt.figure(figsize=(15, 5))
        plt.imshow(keogram[:, start:end, :], aspect='auto')
        plt.colorbar(label='Radiance (kR)')
        plt.title(f'Keogram {spectrograph} from {start} to {end} minutes')
        plt.xlabel('Minutes from midnight UTC')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(os.path.join(current_date_dir, f'{spectrograph}-keogram-{current_date_str}_{start}_{end}.png'))
        plt.close()

def main():
    """
    Main function to run the keogram update process.
    """
    try:
        keogram, last_processed_minute = load_existing_keogram(output_dir, spectrograph)
        keogram = add_rgb_columns(keogram, rgb_dir_base, last_processed_minute, spectrograph)
        save_keogram(keogram, output_dir, spectrograph)
        save_keogram_with_subplots(keogram, output_dir, spectrograph)
        print("Keogram update completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
