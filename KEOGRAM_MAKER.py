"""

This program uses RGB image-columns generated every minute using the spectrograms captured by MISS* to update a daily keogram.
At 00:00 UTC, a new keogram for the day is created (empty). At 00:05 UTC, the previous day's keogram receives its last update,
and the new day's keogram receives its first update.

The script ensures only available past data is used for subplot analysis. The RGB channels are named according to the three main
emission lines of the aurora: 4278 Å (Blue), 5577 Å (Green), and 6300 Å (Red).

ENTER SPECTROGRAPH NAME: MISS1 OR MISS2


Author: Nicolas Martinez (UNIS/LTU)
Last update: August 2024
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import time

# SET SPECTROGRAPH NAME MANUALLY ('MISS1' OR 'MISS2')
spectrograph = "MISS1"

# Base directory where the RGB columns are saved (yyyy/mm/dd)
home_dir = os.path.expanduser("~")
rgb_dir_base = os.path.join(home_dir, ".venvMISS2", "MISS2", "RGB_columns")
output_dir = os.path.join(home_dir, ".venvMISS2", "MISS2", "Keograms")

# Define auroral emission line wavelengths in Ångström (thousands of Å)
emission_wavelengths = {
    'blue': 4278,  # Blue auroral emission line
    'green': 5577, # Green auroral emission line
    'red': 6300    # Red auroral emission line
}

# Sensitivity correction coefficients for MISS1 and MISS2 (2024)
coeffs_sensitivity = {
    'MISS1': [-1.378573e-16, 4.088257e-12, -4.806258e-08, 2.802435e-04, -8.109943e-01, 9.329611e+02],
    'MISS2': [-1.287537e-16, 3.929045e-12, -4.725879e-08, 2.645489e-04, -7.809561e-01, 9.221457e+02]
}

num_pixels_y = 300  # Number of pixels along the y-axis (for RGB with 300 rows)
num_minutes = 24 * 60  # Total number of minutes in a day

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

# initialise an empty keogram with white pixels
def initialise_keogram():
    return np.full((num_pixels_y, num_minutes, 3), 255, dtype=np.uint8)

# Load an existing keogram or create a new one if none exists
def load_existing_keogram(output_dir, spectrograph):
    current_utc_time = datetime.now(timezone.utc)
    current_date = current_utc_time.strftime('%Y/%m/%d')

    keogram_path = os.path.join(output_dir, current_date, f'{spectrograph}-keogram-{current_utc_time.strftime("%Y%m%d")}.png')

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

# Save the updated keogram
def save_keogram(keogram, output_dir, spectrograph):
    current_utc_time = datetime.now(timezone.utc)
    current_date_str = current_utc_time.strftime('%Y%m%d')
    current_date_dir = os.path.join(output_dir, current_utc_time.strftime('%Y/%m/%d'))
    os.makedirs(current_date_dir, exist_ok=True)

    # Ensure that only the core keogram data (300x1440x3) is saved
    keogram_to_save = keogram[:, :num_minutes, :]
    keogram_filename = os.path.join(current_date_dir, f'{spectrograph}-keogram-{current_date_str}.png')
    Image.fromarray(keogram_to_save).save(keogram_filename)

# Apply sensitivity correction to radiance using the respective coefficients
def apply_sensitivity_correction(radiance, wavelength, coeffs):
    correction_factor = np.polyval(coeffs, wavelength)
    return radiance * correction_factor

# Convert 8-bit image data to 16-bit
def convert_8bit_to_16bit(value):
    return value * (65535 / 255)

# Convert RGB data to radiance in kR
def rgb_to_radiance_kR(rgb_data, wavelengths, sensitivity_coeffs):
    radiance = np.zeros_like(rgb_data, dtype=np.float64)
    for i, color in enumerate(['blue', 'green', 'red']):
        wavelength = wavelengths[color]
        radiance_16bit = convert_8bit_to_16bit(rgb_data[:, :, i])
        corrected_radiance = apply_sensitivity_correction(radiance_16bit, wavelength, sensitivity_coeffs)
        radiance[:, :, i] = corrected_radiance / 1000  # Convert to kR by dividing by 1000
    return radiance

# Add RGB columns to the keogram and convert to radiance in kR
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

                radiance_data = rgb_to_radiance_kR(rgb_data, emission_wavelengths, coeffs_sensitivity[spectrograph])
                keogram[:, minute:minute+1, :] = radiance_data.astype(np.uint8)
                found_minutes.add(minute)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        else:
            # Handle missing data with conditions
            if current_minute_of_the_day - minute > 4:  # Only fill with black if the slot is older than 4 minutes
                keogram[:, minute:minute+1, :] = np.zeros((num_pixels_y, 1, 3), dtype=np.uint8)  # Black RGB
    return keogram

# Save the keogram with subplot analysis
def save_keogram_with_subplots(keogram, output_dir, spectrograph):
    current_utc_time = datetime.now(timezone.utc)
    current_date_str = current_utc_time.strftime('%Y%m%d')
    current_date_dir = os.path.join(output_dir, current_utc_time.strftime('%Y/%m/%d'))
    os.makedirs(current_date_dir, exist_ok=True)

    # Find the last minute with data
    non_empty_columns = np.where(np.any(keogram != 255, axis=0))[0]

    if non_empty_columns.size == 0:
        print("Error: Keogram contains no data. Creating an empty keogram.")
        non_empty_columns = np.array([0])  # Force a single column to prevent empty array issues
        last_minute = 0
    else:
        last_minute = np.max(non_empty_columns)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 4])

    # Determine title based on spectrograph
    spectrograph_title = "I" if spectrograph == "MISS1" else "II"

    # Main keogram plot (show the entire day, with future hours remaining white)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.imshow(keogram, aspect='auto', extent=[0, num_minutes, 90, -90])
    ax_main.set_title(f"Meridian Imaging Svalbard Spectrograph {spectrograph_title} {current_utc_time.strftime('%Y-%m-%d')}", fontsize=20)
    ax_main.set_xlabel("Time (UT)")
    ax_main.set_ylabel("Elevation angle [degrees]")

    # Set x-axis for hours (entire day)
    x_ticks = np.arange(0, num_minutes + 1, 120)  # Positions for time labels every second hour (120min)
    x_labels = [(datetime(2024, 1, 1) + timedelta(minutes=int(t))).strftime('%H:%M') for t in x_ticks]
    ax_main.set_xticks(x_ticks)
    ax_main.set_xticklabels(x_labels)

    # Set y-axis for south, zenith, and north (Elevation angle)
    y_ticks = np.linspace(-90, 90, 7)
    ax_main.set_yticks(y_ticks)
    ax_main.set_yticklabels(['90° S', '60° S', '30° S', 'Zenith', '30° N', '60° N', '90° N'])
    ax_main.set_ylim(-90, 90)

    # Temporal analysis subplot: from 00:00 to most recent time
    ax_temporal = fig.add_subplot(gs[0, 0])
    temporal_data = keogram[:, :last_minute+1, :]

    # Exclude columns that are still initialised to white
    valid_columns = np.any(temporal_data != 255, axis=(0, 2))
    temporal_avg = np.mean(temporal_data[:, valid_columns, :], axis=0)

    # Ensure temporal_avg does not contain NaNs or infinities
    temporal_avg = np.nan_to_num(temporal_avg, nan=0.0, posinf=None, neginf=None)

    # Plotting
    ax_temporal.plot(np.arange(temporal_avg.shape[0]), temporal_avg[:, 0], label="4278 Å", color='b')
    ax_temporal.plot(np.arange(temporal_avg.shape[0]), temporal_avg[:, 1], label="5577 Å", color='g')
    ax_temporal.plot(np.arange(temporal_avg.shape[0]), temporal_avg[:, 2], label="6300 Å", color='r')
    #ax_temporal.set_xlabel("Time (UT)")
    ax_temporal.set_ylabel("Radiance [kR]")
    ax_temporal.legend()

    # Adjust y-axis limits based on actual data
    ax_temporal.set_ylim([0, np.max(temporal_avg) + 1])

    # Set x-axis for temporal analysis
    ax_temporal.set_xlim(0, num_minutes)
    x_ticks_temporal = np.arange(0, num_minutes + 1, 60)
    x_labels_temporal = [(datetime(2024, 1, 1) + timedelta(minutes=int(t))).strftime('%H:%M') for t in x_ticks_temporal]
    ax_temporal.set_xticks(x_ticks_temporal)
    ax_temporal.set_xticklabels(x_labels_temporal)

    # Spatial analysis subplot: across available elevation angles from 00:00 to the most recent time
    ax_spatial = fig.add_subplot(gs[1, 1])

    # Filter out any placeholder or invalid data (e.g., 255 or similar artifacts)
    filtered_keogram = np.where(keogram == 255, np.nan, keogram)

    # Extract the relevant portion of the keogram (from 00:00 to last_minute)
    relevant_keogram = filtered_keogram[:, :last_minute+1, :]

    # Calculate the average radiance for each elevation angle across the time range
    spatial_avg = np.nanmean(relevant_keogram, axis=1)  # Average over time, ignoring NaNs

    # Ensure all NaNs or infs are converted to zero (if no valid data exists after filtering)
    spatial_avg = np.nan_to_num(spatial_avg, nan=0.0, posinf=None, neginf=None)

    # Plot the spatial averages for each emission line
    ax_spatial.plot(spatial_avg[:, 0], np.linspace(90, -90, num_pixels_y), label="4278 Å", color='b')
    ax_spatial.plot(spatial_avg[:, 1], np.linspace(90, -90, num_pixels_y), label="5577 Å", color='g')
    ax_spatial.plot(spatial_avg[:, 2], np.linspace(90, -90, num_pixels_y), label="6300 Å", color='r')
    ax_spatial.set_xlabel("Radiance [kR]")
    #ax_spatial.set_ylabel("Elevation angle [degrees]")
    ax_spatial.set_ylim(-90, 90)
    # Set Y-Ticks Exactly Every 15 Degrees from -90 to 90
    ax_spatial.set_yticks(np.arange(-90, 91, 15))  # -90, -75, -60, ..., 75, 90

    # Set Y-Tick Labels with Degree Symbol
    ax_spatial.set_yticklabels([f"{int(angle)}°" for angle in np.arange(-90, 91, 15)])
    ax_spatial.legend()

    # Automatically adjust x-axis limits based on data for spatial subplot
    min_spatial_radiance = np.min(spatial_avg)
    max_spatial_radiance = np.max(spatial_avg)
    if min_spatial_radiance == max_spatial_radiance:
        ax_spatial.set_xlim(0 , max_spatial_radiance + 1)  # Avoid singular transformation
    else:
        ax_spatial.set_xlim(min_spatial_radiance, max_spatial_radiance)

    plt.tight_layout()
    keogram_filename = os.path.join(current_date_dir, f'{spectrograph}-keogram-{current_date_str}.png')
    plt.savefig(keogram_filename)
    plt.close(fig)  # Explicitly close the figure to avoid memory warnings



# Main function to update keogram every 5 minutes
def main():
    while True:
        try:
            current_utc_time = datetime.now(timezone.utc)

            if current_utc_time.minute % 1 == 0:  # Every 5 minutes
                keogram, last_processed_minute = load_existing_keogram(output_dir, spectrograph)
                keogram = add_rgb_columns(keogram, rgb_dir_base, last_processed_minute, spectrograph)
                save_keogram(keogram, output_dir, spectrograph)
                save_keogram_with_subplots(keogram, output_dir, spectrograph)
                print("Update completed.")
            else:
                print("Waiting for the next update...")

            time.sleep(60)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()