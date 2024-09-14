"""
This program uses RGB image-columns generated every 5 (FIVE) minutes using the spectrograms captured by MISS* to update a daily keogram.
At 00:00 UTC, a new keogram for the day is created (empty). At 00:05 UTC, the previous day's keogram receives its last update,
and the new day's keogram receives its first update.

The script ensures only available past data is used for analysis. The RGB channels are named according to the three main
emission lines of the aurora: 

- **Red channel**: 6300 Å (Oxygen emission line)
- **Green channel**: 5577 Å (Oxygen emission line)
- **Blue channel**: 4278 Å (Nitrogen emission line)

Author: Nicolas Martinez (UNIS/LTU)
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import time
from parameters import parameters  # Import parameters from parameters.py

# Extract parameters from parameters.py
spectrograph = parameters['device_name']
RGB_folder = parameters['RGB_folder']
keogram_dir = parameters['keogram_dir']
num_pixels_y = parameters['num_pixels_y']
num_minutes = parameters['num_minutes']

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

# Initialise an empty keogram with white pixels
def initialise_keogram():
    return np.full((num_pixels_y, num_minutes, 3), 255, dtype=np.uint8)

# Load an existing keogram or create a new one if none exists
def load_existing_keogram(keogram_dir, spectrograph):
    current_utc_time = datetime.now(timezone.utc)
    current_date = current_utc_time.strftime('%Y/%m/%d')

    keogram_path = os.path.join(keogram_dir, current_date, f'{spectrograph}-keogram-{current_utc_time.strftime("%Y%m%d")}.png')

    if os.path.exists(keogram_path):
        with Image.open(keogram_path) as img:
            keogram = np.array(img)

        # Validate the loaded keogram dimensions
        if keogram.shape != (num_pixels_y, num_minutes, 3):
            keogram = initialise_keogram()  # Reinitialise to correct dimensions
            last_processed_minute = 0
        else:
            last_processed_minute = np.max(np.where(np.any(keogram != 255, axis=0))[0])

        return keogram, last_processed_minute
    else:
        return initialise_keogram(), 0

# Save the updated keogram with axes and units
def save_keogram_with_axes(keogram, keogram_dir, spectrograph):
    current_utc_time = datetime.now(timezone.utc)
    current_date_str = current_utc_time.strftime('%Y%m%d')
    current_date_dir = os.path.join(keogram_dir, current_utc_time.strftime('%Y/%m/%d'))
    os.makedirs(current_date_dir, exist_ok=True)

    # Create figure for keogram with axes and units
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(keogram, aspect='auto', extent=[0, num_minutes, 90, -90])

    # Set correct title format for the keogram
    spectrograph_title = "I" if spectrograph == "MISS1" else "II"
    ax.set_title(f"Meridian Imaging Svalbard Spectrograph {spectrograph_title} {current_utc_time.strftime('%Y-%m-%d')}", fontsize=20)

    # Set axis labels and ticks
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Elevation angle [degrees]")

    # Set x-axis ticks for time (every 2 hours)
    x_ticks = np.arange(0, num_minutes + 1, 120)
    x_labels = [(datetime(2024, 1, 1) + timedelta(minutes=int(t))).strftime('%H:%M') for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # Set y-axis ticks for elevation angle
    y_ticks = np.linspace(-90, 90, 7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['90° S', '60° S', '30° S', 'Zenith', '30° N', '60° N', '90° N'])

    # Save the keogram with axes (without subplots)
    keogram_filename = os.path.join(current_date_dir, f'{spectrograph}-keogram-{current_date_str}.png')
    plt.savefig(keogram_filename)
    plt.close(fig)

# Add RGB columns to the keogram
def add_rgb_columns(keogram, base_dir, last_processed_minute, spectrograph):
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

                # Validate the shape of the image data
                if rgb_data.shape != (num_pixels_y, 1, 3):
                    print(f"Unexpected image shape {rgb_data.shape} for {filename}. Expected ({num_pixels_y}, 1, 3). Skipping this image.")
                    continue

                keogram[:, minute:minute+1, :] = rgb_data.astype(np.uint8)
                found_minutes.add(minute)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        else:
            # Handle missing data with conditions
            if current_minute_of_the_day - minute > 4:  # Only fill with black if the slot is older than 4 minutes
                keogram[:, minute:minute+1, :] = np.zeros((num_pixels_y, 1, 3), dtype=np.uint8)  # Black RGB
    return keogram

# Main function to update keogram every minute (without subplots)
def main():
    while True:
        try:
            current_utc_time = datetime.now(timezone.utc)
            if current_utc_time.minute % 1 == 0:  # Check every minute
                keogram, last_processed_minute = load_existing_keogram(keogram_dir, spectrograph)
                keogram = add_rgb_columns(keogram, RGB_folder, last_processed_minute, spectrograph)
                save_keogram_with_axes(keogram, keogram_dir, spectrograph)
                print("Keogram update completed.")
            else:
                print("Waiting for the next update...")

            time.sleep(60)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()