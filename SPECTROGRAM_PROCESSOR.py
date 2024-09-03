"""
This script processes spectrogram images from the MISS1 and MISS2 spectrographs. It handles image orientation 
(flipping and rotating), performs background subtraction, and applies wavelength and sensitivity calibration 
to prepare the data for spectral and spatial analysis.

Steps:
1. Monitors a directory for new images every 60 seconds minus processing time.
2. Processes the images by flipping, rotating, subtracting background, and calibrating.
3. Generates and saves plots for spectral and spatial analysis.

Author: Nicolas Martinez (UNIS/LTU)
Last update: August 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from PIL import Image
import os
import time
from datetime import datetime, timezone
import re

# Polynomial coefficients for MISS1 and MISS2 K(lambda) (2024)
coeffs_sensitivity_miss1 = [1.678e-17, -3.126e-13, 1.743e-09, 2.366e-07, -2.935e-02, 6.663e+01]  
coeffs_sensitivity_miss2 = [-1.378573e-16, 4.088257e-12, -4.806258e-08, 2.802435e-04, -8.109943e-01, 9.329611e+02]

# Wavelength calibration coefficients (2024)
miss1_wavelength_coeffs = [4217.273360, 2.565182, 0.000170]
miss2_wavelength_coeffs = [4088.509, 2.673936, 1.376154e-4]

def calculate_wavelength(pixel_columns, coeffs):
    return coeffs[0] + coeffs[1] * pixel_columns + coeffs[2] * (pixel_columns ** 2)

def calculate_k_lambda(wavelengths, coeffs):
    return np.polyval(coeffs, wavelengths)

def process_and_plot_with_flip_and_rotate(image_array, spectrograph_type):
    """
    Flips, rotates 90° counterclockwise, subtracts background, and calibrates the image.
    """
    flipped_image = np.flipud(image_array)
    background = np.median(flipped_image, axis=0)
    background_subtracted_image = np.clip(flipped_image - background[np.newaxis, :], 0, None)
    rotated_image = rotate(background_subtracted_image, angle=90, reshape=True)

    if spectrograph_type == "MISS1":
        wavelengths = calculate_wavelength(np.arange(rotated_image.shape[1]), miss1_wavelength_coeffs)
        k_lambda = calculate_k_lambda(wavelengths, coeffs_sensitivity_miss1)
        fov_start, fov_end = 280, 1140
    elif spectrograph_type == "MISS2":
        wavelengths = calculate_wavelength(np.arange(rotated_image.shape[1]), miss2_wavelength_coeffs)
        k_lambda = calculate_k_lambda(wavelengths, coeffs_sensitivity_miss2)
        fov_start, fov_end = 271, 1116
    else:
        raise ValueError("Unknown spectrograph type. Please choose 'MISS1' or 'MISS2'.")

    calibrated_image = rotated_image * k_lambda[np.newaxis, :] / 1000  # Convert to kR/Å
    elevation_scale = np.linspace(-90, 90, fov_end - fov_start)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"{spectrograph_type}-{datetime.now().strftime('%Y%m%d-%H%M00')}.png", fontsize=18)

    gs = plt.GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[1, 4, 1])

    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.imshow(calibrated_image, cmap='gray', aspect='auto', extent=[wavelengths.min(), wavelengths.max(), 0, calibrated_image.shape[0]])
    tick_positions = np.linspace(fov_start, fov_end, num=7)
    tick_labels = ["South", "-60", "-30", "Zenith", "30", "60", "North"]
    ax_main.set_yticks(tick_positions)
    ax_main.set_yticklabels(tick_labels)
    ax_main.set_xlabel("Wavelength [Å]", fontsize=12)
    ax_main.set_ylabel("Elevation [Degrees]", fontsize=12)
    ax_main.grid(False)

    ax_spectral = fig.add_subplot(gs[0, 0])
    spectral_avg = np.mean(calibrated_image[fov_start:fov_end, :], axis=0)
    ax_spectral.plot(wavelengths, spectral_avg)
    ax_spectral.set_ylabel("Spectral Radiance [kR/Å]", fontsize=12)
    ax_spectral.set_title("Spectral Analysis", fontsize=12)
    ax_spectral.grid()

    ax_spatial = fig.add_subplot(gs[1, 1])
    spatial_avg = np.mean(calibrated_image[fov_start:fov_end, :], axis=1)
    ax_spatial.plot(spatial_avg, elevation_scale)
    ax_spatial.set_xlabel("Spatial Radiance [kR/θ]", fontsize=12)
    ax_spatial.set_title("Spatial Analysis", fontsize=12)
    ax_spatial.set_yticks(np.linspace(-90, 90, num=9))
    ax_spatial.set_yticklabels(["South", "-60", "-45", "-30", "Zenith", "30", "45", "60", "North"])
    ax_spatial.grid()

    plt.tight_layout()
    plt.show()
    
def check_and_process_latest_image(config):
    last_processed_image = None

    while True:
        start_time = time.time()
        latest_image_file = get_latest_image_path(config['spectro_path'])
        if latest_image_file and latest_image_file != last_processed_image:
            image_array = np.array(Image.open(latest_image_file))

            if "MISS1" in latest_image_file:
                process_and_plot_with_flip_and_rotate(image_array, "MISS1")
            elif "MISS2" in latest_image_file:
                process_and_plot_with_flip_and_rotate(image_array, "MISS2")

            last_processed_image = latest_image_file

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M00")
            processed_image_name = f"{spectrograph_type}-spectrogram-{timestamp}.png"
            processed_image_path = os.path.join(config['processed_spectrogram_dir'], processed_image_name)
            plt.savefig(processed_image_path, format='png', bbox_inches='tight')
            plt.close(fig)

            print(f"Processed and saved spectrogram: {processed_image_path}")

        elapsed_time = time.time() - start_time
        sleep_time = max(60 - elapsed_time, 0)
        time.sleep(sleep_time)

def get_latest_image_path(image_folder):
    today_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    full_path = os.path.join(image_folder, today_path)
    
    if os.path.exists(full_path):
        pattern = r'MISS[12]-\d{8}-\d{6}\.png'
        all_files = [f for f in os.listdir(full_path) if re.match(pattern, f)]
        
        if all_files:
            all_files.sort(key=lambda x: re.findall(r'\d{8}-\d{6}', x)[0])
            latest_file = all_files[-1]
            return os.path.join(full_path, latest_file)
    
    return None

check_and_process_latest_image(config)


# Paths - For standalone use only
#image_folder = os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Captured_PNG", "averaged_PNG")
#output_folder = os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "Captured_PNG", "Processed_spectrograms")

