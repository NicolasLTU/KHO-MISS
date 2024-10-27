"""
This script processes averaged spectrogram images from the MISS1 and MISS2 spectrographs. 
It handles image orientation (flipping and rotating), performs background subtraction, and 
applies wavelength and sensitivity calibration to prepare the data for spectral and spatial analysis.

Steps:
1. Monitors a directory for new images every 60 seconds minus processing time.
2. Processes the images by flipping, rotating, subtracting background, and calibrating.
3. Generates and saves plots for spectral and spatial analysis, accounting for any binning of the spectrogram.

Author: Nicolas Martinez (UNIS/LTU)
Last update: October 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from PIL import Image
import os
import time
from datetime import datetime, timezone
import re
from parameters import parameters  # Import parameters from parameters.py

# Extract parameters from the parameters dictionary
coeffs_sensitivity_miss1 = parameters['coeffs_sensitivity']['MISS1']
coeffs_sensitivity_miss2 = parameters['coeffs_sensitivity']['MISS2']
miss1_wavelength_coeffs = parameters['miss1_wavelength_coeffs']
miss2_wavelength_coeffs = parameters['miss2_wavelength_coeffs']
averaged_PNG_folder = parameters['averaged_PNG_folder']
processed_spectrogram_dir = parameters['processed_spectrogram_dir']
binX = parameters['binX']
binY = parameters['binY']


def calculate_wavelength(pixel_columns, coeffs):
    return coeffs[0] + coeffs[1] * pixel_columns + coeffs[2] * (pixel_columns ** 2)

def calculate_k_lambda(wavelengths, coeffs):
    # Apply polynomial fit directly to wavelength values
    return np.polyval(coeffs, wavelengths)

def process_and_plot_with_flip_and_rotate(image_array, spectrograph_type, filename):
    # Flip the image and subtract background without altering FOV visuals
    flipped_image = np.flipud(image_array)
    background = np.median(flipped_image, axis=0)
    background_subtracted_image = np.clip(flipped_image - background[np.newaxis, :], 0, None)
    rotated_image = rotate(background_subtracted_image, angle=90, reshape=True)
    
    # Assign wavelength and sensitivity based on spectrograph type and limit FOV
    if spectrograph_type == "MISS1":
        wavelengths = calculate_wavelength(np.arange(rotated_image.shape[1]) * binY, miss1_wavelength_coeffs)
        k_lambda = calculate_k_lambda(wavelengths, coeffs_sensitivity_miss1)
        fov_start, fov_end = parameters['miss1_horizon_limits']
    elif spectrograph_type == "MISS2":
        wavelengths = calculate_wavelength(np.arange(rotated_image.shape[1]) * binY, miss2_wavelength_coeffs)
        k_lambda = calculate_k_lambda(wavelengths, coeffs_sensitivity_miss2)
        fov_start, fov_end = parameters['miss2_horizon_limits']
    else:
        raise ValueError("Unknown spectrograph type. Please choose 'MISS1' or 'MISS2'.")

    # Adjust FOV limits for binX and wavelength range for binY
    fov_start_binned, fov_end_binned = fov_start // binX, fov_end // binX
    wavelength_range = wavelengths[::binY]

    # Extract the binned FOV portion from the rotated image for spatial analysis
    binned_image = rotated_image[::binX, ::binY]
    binned_image_fov = binned_image[fov_start_binned:fov_end_binned, :]  # Limit FOV within binned image

    elevation_scale = np.linspace(fov_start, fov_end, binned_image_fov.shape[0])


    # Create figure and main spectrogram plot, restricting FOV and keeping visuals consistent
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(filename, fontsize=18)
    gs = plt.GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[1, 4, 1])

    # Main spectrogram plot
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.imshow(rotated_image, cmap='gray', aspect='auto', extent=[wavelengths.min(), wavelengths.max(), 0, rotated_image.shape[0]])
    tick_positions = np.linspace(fov_start_binned, fov_end_binned, num=7)
    tick_labels = ["South", "-60", "-30", "Zenith", "30", "60", "North"]
    ax_main.set_yticks(tick_positions)
    ax_main.set_yticklabels(tick_labels)
    ax_main.set_xlabel("Wavelength [Å]", fontsize=12)
    ax_main.set_ylabel("Elevation [Degrees]", fontsize=12)
    ax_main.grid(False)

    # Spectral analysis subplot using the full wavelength range
    ax_spectral = fig.add_subplot(gs[0, 0])
    spectral_avg = np.mean(binned_image * k_lambda[np.newaxis, :len(wavelength_range)], axis=0) / 1000 # Convert to kR/Å
    ax_spectral.plot(wavelength_range[:len(spectral_avg)], spectral_avg)
    ax_spectral.set_ylabel("Spectral Radiance [kR/Å]", fontsize=12)
    ax_spectral.set_title("Spectral Analysis", fontsize=12)
    ax_spectral.grid()

    # Spatial analysis subplot with calibration
    ax_spatial = fig.add_subplot(gs[1, 1])
    spatial_avg = np.mean(rotated_image[fov_start_binned:fov_end_binned,:] * k_lambda[np.newaxis,:], axis=1) / 1000  # Convert to kR/Å
    elevation_scale = np.linspace(-90, 90, spatial_avg.shape[0])  # This represents the elevation range
    ax_spatial.plot(spatial_avg, elevation_scale)
    ax_spatial.set_xlabel("Spatial Radiance [kR/θ]", fontsize=12)
    ax_spatial.set_title("Spatial Analysis", fontsize=12)
    ax_spatial.set_yticks(np.linspace(-90, 90, num=9))
    ax_spatial.set_yticklabels(["South", "-60", "-45", "-30", "Zenith", "30", "45", "60", "North"])
    ax_spatial.grid()

    plt.tight_layout()
    return fig 


def check_and_process_latest_image(averaged_PNG_folder, processed_spectrogram_dir):
    last_processed_image = None

    while True:
        start_time = time.time()
        latest_image_file = get_latest_image_path(averaged_PNG_folder)
        if latest_image_file and latest_image_file != last_processed_image:
            filename = os.path.basename(latest_image_file)

            # Determine spectrograph type
            if "MISS1" in filename.upper():
                spectrograph_type = "MISS1"
            elif "MISS2" in filename.upper():
                spectrograph_type = "MISS2"
            else:
                print("Unknown spectrograph type in filename.")
                last_processed_image = latest_image_file
                continue  # Skip processing

            # Extract date and time from the filename
            filename_no_ext, ext = os.path.splitext(filename)
            parts = filename_no_ext.split('-')

            if len(parts) == 3:
                spectrograph_type_in_file = parts[0]
                date_str = parts[1]
                time_str = parts[2]
                processed_image_name = f"{spectrograph_type_in_file}-spectrogram-{date_str}-{time_str}{ext}"
            else:
                print("Filename format not recognized.")
                last_processed_image = latest_image_file
                continue  # Skip processing

            image_array = np.array(Image.open(latest_image_file))

            # Process the image and get the figure
            fig = process_and_plot_with_flip_and_rotate(image_array, spectrograph_type, filename)

            processed_image_path = os.path.join(processed_spectrogram_dir, processed_image_name)
            fig.savefig(processed_image_path, format='png', bbox_inches='tight')
            plt.close(fig)

            print(f"Processed and saved spectrogram: {processed_image_path}")
            last_processed_image = latest_image_file

        elapsed_time = time.time() - start_time
        sleep_time = max(60 - elapsed_time, 0)
        time.sleep(sleep_time)

def get_latest_image_path(averaged_PNG_folder):
    today_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    full_path = os.path.join(averaged_PNG_folder, today_path)
    
    if os.path.exists(full_path):
        pattern = r'MISS[12]-\d{8}-\d{6}\.png'
        all_files = [f for f in os.listdir(full_path) if re.match(pattern, f)]
        
        if all_files:
            all_files.sort(key=lambda x: re.findall(r'\d{8}-\d{6}', x)[0])
            latest_file = all_files[-1]
            return os.path.join(full_path, latest_file)
    
    return None

# Start the image checking and processing loop
check_and_process_latest_image(averaged_PNG_folder, processed_spectrogram_dir)
