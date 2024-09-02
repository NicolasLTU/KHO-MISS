'''
This script checks for the latest averaged PNG file every minute, processes it by applying filtering, background subtraction, and rotation, and then saves the resulting spectrogram image with the format: MISS?-spectrogram-YYYYMMDD-HHMMSS.png.


Author: Nicolas Martinez (UNIS/LTU)
Last Update: 2024

Description:
The script checks for the latest averaged PNG file every minute, processes it by applying filtering, background subtraction, and rotation, and then saves the resulting spectrogram image with the format: MISS?-spectrogram-YYYYMMDD-HHMMSS.png.

'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  # Used to have multiple plots in the same image
from scipy import signal
import os
import re
from datetime import datetime, timezone
from scipy.ndimage import rotate
import time

# Define the base path where the averaged PNG date directory is located
image_folder = os.path.join(os.path.expanduser('~'), '.venvMISS2', 'MISS2', 'Captured_PNG', 'averaged_MISS2_PNG')

# Define the output path for processed spectrograms
output_folder = os.path.join(os.path.expanduser('~'), '.venvMISS2', 'MISS2', 'Processed_Spectrograms')

# Function to read PNG file
def read_png(filename):
    with Image.open(filename) as img:
        image_data = np.array(img)
    return image_data

# Function to process the raw image
def process_image(raw_image):
    processed_image = signal.medfilt2d(raw_image.astype('float32'))
    bg = np.average(processed_image[0:30, 0:30])
    processed_image = np.maximum(0, processed_image - bg)
    processed_image = rotate(processed_image, 90, reshape=True)  # Rotate 90 degrees clockwise
    return processed_image

# Normalize the light intensity
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Retrieve the path to the latest averaged image
def get_latest_image_path(image_folder):
    today_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    full_path = os.path.join(image_folder, today_path)
    if os.path.exists(full_path):
        pattern = r'MISS2-\d{8}-\d{6}\.png'
        all_files = [f for f in os.listdir(full_path) if re.match(pattern, f)]
        if all_files:
            all_files.sort()
            latest_file = all_files[-1]
            return os.path.join(full_path, latest_file)
    return None

# Process the latest image and save it
def process_and_save_latest_image(latest_image_file):
    image_data = read_png(latest_image_file)
    processed_image = process_image(image_data)

    start_wavelength = 3950  # Start wavelength in Ångström
    end_wavelength = 7300    # End wavelength in Ångström
    num_wavelengths = processed_image.shape[1]
    wavelengths = np.linspace(start_wavelength, end_wavelength, num_wavelengths)

    image_title = os.path.basename(latest_image_file)
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(image_title, fontsize=14)
    gs = plt.GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[1, 4, 1])

    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.imshow(processed_image, cmap='gray', aspect='auto')
    ax_main.set_xlabel("Wavelength (Å)")
    ax_main.set_ylabel("Elevation Angle (Degrees)")
    num_columns = processed_image.shape[1]
    ax_main.set_xticks(np.linspace(0, num_columns - 1, 5))
    ax_main.set_xticklabels(np.linspace(3900, 7300, 5).astype(int))

    ax_spectral = fig.add_subplot(gs[0, 0])
    wavelengths = np.linspace(start_wavelength, end_wavelength, len(processed_image[0]))
    spectral_data = np.mean(processed_image, axis=0)
    normalized_spectral_data = normalize(spectral_data)
    ax_spectral.plot(wavelengths, normalized_spectral_data)
    ax_spectral.set_yticks(np.linspace(0, 1, 3))
    ax_spectral.grid()
    ax_spectral.set_title("Spectral Analysis")       

    ax_spatial = fig.add_subplot(gs[1, 1])
    spatial_data = np.mean(processed_image, axis=1)
    normalized_spatial_data = normalize(spatial_data)
    ax_spatial.plot(normalized_spatial_data, range(processed_image.shape[0]))
    ax_spatial.set_xticks(np.linspace(0, 1, 3))
    ax_spatial.set_title("Spatial Analysis")
    ax_spatial.invert_yaxis()
    ax_spatial.grid()
    plt.setp(ax_spectral.get_xticklabels(), visible=True)
    plt.setp(ax_spectral.get_yticklabels(), visible=True)

    plt.tight_layout()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    processed_image_name = f"MISS2-spectrogram-{timestamp}.png"
    processed_image_path = os.path.join(output_folder, processed_image_name)
    plt.savefig(processed_image_path, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Processed and saved spectrogram: {processed_image_path}")

if __name__ == "__main__":
    last_processed_image = None
    while True:
        start_time = time.time()
        latest_image_file = get_latest_image_path(image_folder)
        if latest_image_file and latest_image_file != last_processed_image:
            process_and_save_latest_image(latest_image_file)
            last_processed_image = latest_image_file
        
        # Calculate how long the processing took and sleep for the remainder of the 60 seconds
        elapsed_time = time.time() - start_time
        sleep_time = max(60 - elapsed_time, 0)
        time.sleep(sleep_time)
