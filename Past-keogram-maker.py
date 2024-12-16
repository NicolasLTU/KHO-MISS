'''
This script allows the user to perform the generation of a keogram for any date with available captured spectrogram.

It asks the user to:
- Pick a date to generate the keogram for
- Select spectrograph (MISS 1 or MISS 2)
- Choose a keogram with or without spatial and temporal plots.

It accounts for the fact the calibration was made on automatically flipped spectrograms (Artemis Software)

Author: Nicolas Martinez (LTU/UNIS)

Last update: September 2024

'''

import os
import numpy as np
from PIL import Image, PngImagePlugin
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from collections import defaultdict
import importlib

# Load parameters from parameters.py
parameters = importlib.import_module('parameters').parameters

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

# Average images minute-wise
def average_images(processed_minutes, device_name):
    print("Starting the averaging of images...")
    images_by_minute = defaultdict(list)
    filename_regex = re.compile(r'^.+-(\d{8})-(\d{6})\.png$')

    for root, dirs, files in os.walk(parameters['raw_PNG_folder']):
        for filename in files:
            filepath = os.path.join(root, filename)
            match = filename_regex.match(filename)
            if match:
                date_part, time_part = match.groups()
                images_by_minute[date_part + '-' + time_part[:4]].append(filepath)

    for minute_key, filepaths in tqdm(images_by_minute.items(), desc="Averaging images", unit="minute"):
        if minute_key not in processed_minutes:
            year, month, day, hour, minute = map(int, [minute_key[:4], minute_key[4:6], minute_key[6:8], minute_key[9:11], minute_key[11:]])
            sum_img_array = None
            count = 0
            metadata = None

            for filepath in filepaths:
                try:
                    img = Image.open(filepath)
                    img_array = np.array(img)

                    if sum_img_array is None:
                        sum_img_array = np.zeros_like(img_array, dtype='float64')

                    sum_img_array += img_array
                    count += 1

                    if metadata is None:
                        metadata = img.info

                except Exception as e:
                    print(f"Error processing image {os.path.basename(filepath)}: {e}")

            if count > 0:
                averaged_image = (sum_img_array / count).astype(np.uint16)

                date_specific_folder = os.path.join(parameters['averaged_PNG_folder'], f"{year:04d}", f"{month:02d}", f"{day:02d}")
                os.makedirs(date_specific_folder, exist_ok=True)

                averaged_image_path = os.path.join(date_specific_folder, f"{device_name}-{year:04d}{month:02d}{day:02d}-{hour:02d}{minute:02d}00.png")

                averaged_img = Image.fromarray(averaged_image, mode='I;16')

                if metadata:
                    pnginfo = PngImagePlugin.PngInfo()
                    for key, value in metadata.items():
                        if key in ["Exposure Time", "Date/Time", "Temperature", "Binning", "Note"]:
                            pnginfo.add_text(key, str(value))
                    pnginfo.add_text("Note", f"1-minute average image. {metadata.get('Note', '')}")
                    averaged_img.save(averaged_image_path, pnginfo=pnginfo)
                else:
                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("Note", "1-minute average image.")
                    averaged_img.save(averaged_image_path, pnginfo=pnginfo)

                processed_minutes.append(minute_key)
                print(f"Averaged image saved: {averaged_image_path}")
    print("Averaging of images completed.")

# Function to calculate the pixel row for a given wavelength
def wavelength_to_pixel_row(wavelength, coeffs):
    """Converts a given wavelength to the corresponding pixel row using the polynomial coefficients."""
    return np.polyval(coeffs[::-1], wavelength)  # Apply coeffs in the order (A0, A1, A2)

# Function to calculate radiance based on sensitivity coefficients
def calculate_radiance(pixel_value, coeffs, num_pixels_y, row):
    flipped_row = num_pixels_y - row  # Flip the row index - calibration coeffs are for flipped spectrogram!!!!
    return np.polyval(coeffs, flipped_row)  # Apply the sensitivity coefficients


# Function to scale RGB channels based on their dynamic range
def scale_channel(channel_data):
    min_val = np.min(channel_data)
    max_val = np.max(channel_data)
    range_val = max_val - min_val
    if range_val == 0:
        return np.zeros_like(channel_data, dtype=np.uint8)
    else:
        scaled = ((channel_data - min_val) / range_val) * 255
        return np.clip(scaled, 0, 255).astype(np.uint8)

# Process and average emission line rows
def process_emission_line(spectro_array, emission_row, binning_factor, pixel_range):
    """
    Extracts rows around the emission line and calculates the intensity by subtracting BASE from PEAK.
    """
    num_rows_to_average = max(1, int(12 / binning_factor))  # 2×FWHM rows

    # PEAK region
    start_peak = max(emission_row - num_rows_to_average // 2, 0)
    end_peak = min(emission_row + num_rows_to_average // 2, spectro_array.shape[0])

    # BASE region (buffer to avoid contamination)
    buffer_rows = 2 * num_rows_to_average
    if end_peak + buffer_rows + num_rows_to_average <= spectro_array.shape[0]:
        start_base = end_peak + buffer_rows
        end_base = start_base + num_rows_to_average
    else:
        end_base = start_peak - buffer_rows
        start_base = end_base - num_rows_to_average

    # Crop the spectrogram to pixel range
    spectro_array_cropped = spectro_array[:, pixel_range[0]:pixel_range[1]]

    # Extract PEAK and BASE rows
    PEAK_rows = spectro_array_cropped[start_peak:end_peak, :]
    BASE_rows = spectro_array_cropped[start_base:end_base, :]

    # Calculate intensities
    PEAK_intensity = np.mean(PEAK_rows, axis=0)
    BASE_intensity = np.mean(BASE_rows, axis=0)

    # Subtract BASE from PEAK
    line_intensity = PEAK_intensity - BASE_intensity
    return line_intensity.flatten()


# Create the RGB image from the extracted rows
def create_rgb_column(spectro_array, wavelengths, coeffs, sensitivity_coeffs, binning_factor, pixel_range):
    """
    Processes the spectrogram to create an RGB column with calibrated intensities.
    """
    # Calculate row positions for emission lines
    row_6300 = int(np.polyval(coeffs[::-1], wavelengths['6300']) / binning_factor)
    row_5577 = int(np.polyval(coeffs[::-1], wavelengths['5577']) / binning_factor)
    row_4278 = int(np.polyval(coeffs[::-1], wavelengths['4278']) / binning_factor)

    # Calculate k_lambda values for sensitivity calibration
    k_lambda_6300 = np.polyval(sensitivity_coeffs, wavelengths['6300'])
    k_lambda_5577 = np.polyval(sensitivity_coeffs, wavelengths['5577'])
    k_lambda_4278 = np.polyval(sensitivity_coeffs, wavelengths['4278'])

    # Extract and process emission line intensities
    column_RED = process_emission_line(spectro_array, row_6300, binning_factor, pixel_range)
    column_GREEN = process_emission_line(spectro_array, row_5577, binning_factor, pixel_range)
    column_BLUE = process_emission_line(spectro_array, row_4278, binning_factor, pixel_range)

    # Apply sensitivity scaling and square root scaling
    red_scaled = np.sqrt(np.clip(column_RED / k_lambda_6300, 0, None))
    green_scaled = np.sqrt(np.clip(column_GREEN / k_lambda_5577, 0, None))
    blue_scaled = np.sqrt(np.clip(column_BLUE / k_lambda_4278, 0, None))

    # Combine into an RGB column
    rgb_image = np.stack((red_scaled, green_scaled, blue_scaled), axis=-1)

    # Normalise to 8-bit range
    max_val = np.max(rgb_image)
    if max_val > 0:
        rgb_image = (rgb_image / max_val) * 255

    return rgb_image.astype(np.uint8)

# Create RGB columns for the day
def create_rgb_columns_for_day(date_str, spectrograph):
    print(f"Starting the creation of RGB columns for {date_str} using {spectrograph}...")

    spectro_path_dir = os.path.join(parameters['averaged_PNG_folder'], date_str)
    ensure_directory_exists(spectro_path_dir)
    output_folder = os.path.join(parameters['RGB_folder'], date_str)
    ensure_directory_exists(output_folder)

    matching_files = [f for f in os.listdir(spectro_path_dir) if f.endswith(".png")]

    # Define the central wavelengths for each emission line
    wavelengths = {
        '6300': 6300,  # Oxygen 6300 Å
        '5577': 5577,  # Oxygen 5577 Å
        '4278': 4278,  # Nitrogen 4278 Å
    }

    # Choose the appropriate wavelength coefficients based on the spectrograph
    if spectrograph == 'MISS1':
        coeffs = parameters['miss1_wavelength_coeffs']
        sensitivity_coeffs = parameters['coeffs_sensitivity']['MISS1']
    elif spectrograph == 'MISS2':
        coeffs = parameters['miss2_wavelength_coeffs']
        sensitivity_coeffs = parameters['coeffs_sensitivity']['MISS2']
    else:
        raise ValueError(f"Unknown spectrograph: {spectrograph}")

    # Extract binning factor from parameters
    binning_factor = parameters['binY']

    for filename in tqdm(matching_files, desc="Creating RGB columns", unit="image"):
        png_file_path = os.path.join(spectro_path_dir, filename)

        if not verify_image_integrity(png_file_path):
            print(f"Skipping corrupted image: {filename}")
            continue

        spectro_data = np.array(Image.open(png_file_path))

        pixel_range = parameters['miss2_horizon_limits'] if spectrograph == 'MISS2' else parameters['miss1_horizon_limits']

        RGB_image = create_rgb_column(spectro_data, wavelengths, coeffs, sensitivity_coeffs, binning_factor, pixel_range)

        if RGB_image.shape != (parameters['num_pixels_y'], 1, 3):
            print(f"Error: Unexpected shape {RGB_image.shape} for {filename}. Skipping this image.")
            continue

        RGB_pil_image = Image.fromarray(RGB_image.astype('uint8'), mode='RGB')
        resized_RGB_image = RGB_pil_image.resize((1, parameters['num_pixels_y']), Image.Resampling.LANCZOS)

        base_filename = filename[:-4]
        output_filename = f"{base_filename[:-2]}00.png"
        output_filename_path = os.path.join(output_folder, output_filename)

        resized_RGB_image.save(output_filename_path)
        print(f"Saved RGB column image: {output_filename}")

    print(f"RGB column creation completed for {date_str}.")

# Save keogram with or without subplots
def save_keogram_with_subplots(keogram, output_dir, date_str, spectrograph, add_subplots=True):
    current_date_dir = os.path.join(output_dir, date_str)
    ensure_directory_exists(current_date_dir)

    fig, ax = plt.subplots(figsize=(20, 6)) if not add_subplots else plt.figure(figsize=(24, 12))

    if add_subplots:
        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=3)
        ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
        ax3 = plt.subplot2grid((4, 4), (1, 3), rowspan=3)

        # Plot the temporal data in hours
        hours = np.linspace(0, 24, parameters['num_minutes'])
        ax1.plot(hours, keogram.mean(axis=0)[:, 0], color='red', label='6300 Å')
        ax1.plot(hours, keogram.mean(axis=0)[:, 1], color='green', label='5577 Å')
        ax1.plot(hours, keogram.mean(axis=0)[:, 2], color='blue', label='4278 Å')
        ax1.set_ylabel("Radiance [kR]", fontsize=21)
        ax1.set_xlabel("Time [Hours]", fontsize=21)
        ax1.legend()
        ax1.grid(True)

        # Keogram plot
        ax2.imshow(keogram, aspect='auto', extent=[0, 24*60, 90, -90])
        ax2.set_title(f"{spectrograph} Keogram for {date_str.replace('/', '-')}", fontsize=28)
        ax2.set_ylabel("Elevation angle [degrees]", fontsize=21)
        ax2.set_xlabel("Time (UT)", fontsize=21)
        ax2.set_xticks(np.append(np.arange(0, 24*60, 120), 24*60))
        ax.set_xticklabels([f"{hour}:00" for hour in range(0, 24, 2)] + ["24:00"], fontsize=18)
        ax2.set_yticks(np.linspace(-90, 90, num=7))
        ax2.set_yticklabels(['90° S', '60° S', '30° S', 'Zenith', '30° N', '60° N', '90° N'])

        # Spatial data subplot
        ax3.plot(keogram[:, :, 0].mean(axis=1), np.linspace(-90, 90, parameters['num_pixels_y']), color='red', label='6300 Å')
        ax3.plot(keogram[:, :, 1].mean(axis=1), np.linspace(-90, 90, parameters['num_pixels_y']), color='green', label='5577 Å')
        ax3.plot(keogram[:, :, 2].mean(axis=1), np.linspace(-90, 90, parameters['num_pixels_y']), color='blue', label='4278 Å')
        ax3.set_xlabel("Radiance [kR]", fontsize=21)
        ax3.set_ylabel("Elevation angle [degrees]", fontsize=21)
        ax3.set_yticks(np.linspace(-90, 90, num=7))
        ax3.set_yticklabels(['90° S', '60° S', '30° S', 'Zenith', '30° N', '60° N', '90° N'], labelsize=18)
        ax3.legend()

    else:
        ax = fig.add_subplot(111)
        ax.imshow(keogram, aspect='auto', extent=[0, 24*60, 90, -90])
        ax.set_title(f"{spectrograph} Keogram for {date_str.replace("/", "-")}", fontsize=28)
        ax.set_xticks(np.append(np.arange(0, 24*60, 120), 24*60))
        ax.set_xticklabels([f"{hour}:00" for hour in range(0, 24, 2)] + ["24:00"], fontsize=18)
        ax.set_xlabel("Time (UT)", fontsize=21)
        ax.set_yticks(np.linspace(-90, 90, num=7))
        ax.set_yticklabels(['90° S', '60° S', '30° S', 'Zenith', '30° N', '60° N', '90° N'])
        ax.set_ylabel("Elevation angle [degrees]", fontsize=21)

    keogram_filename = os.path.join(current_date_dir, f'{spectrograph}-keogram-{date_str.replace("/", "")}.png')
    plt.savefig(keogram_filename)
    plt.close(fig)
    print(f"Keogram saved as: {keogram_filename}")

# Load an existing keogram or create a new one if none exists
def load_existing_keogram(output_dir, date_str, spectrograph):
    keogram_path = os.path.join(output_dir, date_str, f'{spectrograph}-keogram-{date_str.replace("/", "")}.png')

    if os.path.exists(keogram_path):
        with Image.open(keogram_path) as img:
            keogram = np.array(img)

        if keogram.shape != (parameters['num_pixels_y'], parameters['num_minutes'], 3):
            keogram = np.full((parameters['num_pixels_y'], parameters['num_minutes'], 3), 255, dtype=np.uint8)
            last_processed_minute = 0
        else:
            last_processed_minute = np.max(np.where(np.any(keogram != 255, axis=0))[0])

        print(f"Loaded existing keogram for {date_str}.")
        return keogram, last_processed_minute
    else:
        print(f"No existing keogram found for {date_str}. Creating a new one.")
        return np.full((parameters['num_pixels_y'], parameters['num_minutes'], 3), 255, dtype=np.uint8), 0

# Add RGB columns to the keogram
def add_rgb_columns(keogram, base_dir, last_processed_minute, date_str, spectrograph):
    today_RGB_dir = os.path.join(base_dir, date_str)

    if not os.path.exists(today_RGB_dir):
        print(f"No directory found for the date ({today_RGB_dir}). Proceeding with blank RGB data.")
        today_RGB_dir = None

    # Select the appropriate sensitivity coefficients for the spectrograph
    if spectrograph == 'MISS1':
        sensitivity_coeffs = parameters['coeffs_sensitivity']['MISS1']
    elif spectrograph == 'MISS2':
        sensitivity_coeffs = parameters['coeffs_sensitivity']['MISS2']
    else:
        raise ValueError(f"Unknown spectrograph: {spectrograph}")

    for minute in tqdm(range(last_processed_minute + 1, parameters['num_minutes']), desc="Adding RGB columns to keogram", unit="minute"):
        filename = f"{spectrograph}-{date_str.replace('/', '')}-{minute // 60:02d}{minute % 60:02d}00.png"
        file_path = os.path.join(today_RGB_dir, filename) if today_RGB_dir else None

        if file_path and os.path.exists(file_path) and verify_image_integrity(file_path):
            try:
                # Open the RGB column and convert it to a NumPy array
                rgb_data = np.array(Image.open(file_path))

                # Ensure k_lambda calibration for each channel
                rgb_data[:, 0] = calculate_radiance(rgb_data[:, 0], sensitivity_coeffs, parameters['num_pixels_y'], 6300)
                rgb_data[:, 1] = calculate_radiance(rgb_data[:, 1], sensitivity_coeffs, parameters['num_pixels_y'], 5577)
                rgb_data[:, 2] = calculate_radiance(rgb_data[:, 2], sensitivity_coeffs, parameters['num_pixels_y'], 4278)

                if rgb_data.shape != (parameters['num_pixels_y'], 1, 3):
                    print(f"Unexpected image shape {rgb_data.shape} for {filename}. Expected (300, 1, 3). Skipping this image.")
                    continue

                # Add the calibrated RGB column to the keogram
                keogram[:, minute:minute+1, :] = rgb_data

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        else:
            # Fill with zeros if no data is available for the minute, but only for gaps larger than 4 minutes
            if parameters['num_minutes'] - minute > 4:
                keogram[:, minute:minute+1, :] = np.zeros((parameters['num_pixels_y'], 1, 3), dtype=np.uint8)

    print(f"RGB columns added to keogram for {date_str}.")
    return keogram

# Main function to prompt for subplot option
umn creation:

python
Kopiera kod
def main():
    date_input = input("Enter the date to process (yyyy/mm/dd): ")
    spectrograph = parameters['device_name']
    add_subplots = input("Do you want to include subplots with calibration data? (yes/no): ").strip().lower() == 'yes'

    processed_minutes = []
    
    # Step 1: Average spectrograms
    average_images(processed_minutes, spectrograph)

    # Step 2: Create RGB columns from averaged images
    averaged_folder = os.path.join(parameters['averaged_PNG_folder'], date_input)
    ensure_directory_exists(averaged_folder)
    matching_files = [f for f in os.listdir(averaged_folder) if f.endswith(".png")]

    for filename in tqdm(matching_files, desc="Creating RGB columns", unit="image"):
        file_path = os.path.join(averaged_folder, filename)
        if not verify_image_integrity(file_path):
            continue

        spectro_data = np.array(Image.open(file_path))
        pixel_range = parameters['miss2_horizon_limits'] if spectrograph == 'MISS2' else parameters['miss1_horizon_limits']

        # Generate RGB column
        wavelengths = {'6300': 6300, '5577': 5577, '4278': 4278}
        RGB_image = create_rgb_column(
            spectro_data, wavelengths, parameters[f'{spectrograph.lower()}_wavelength_coeffs'],
            parameters['coeffs_sensitivity'][spectrograph], parameters['binY'], pixel_range
        )

        # Save the RGB column
        rgb_output_folder = os.path.join(parameters['RGB_folder'], date_input)
        ensure_directory_exists(rgb_output_folder)
        output_filename = f"{os.path.splitext(filename)[0]}_RGB.png"
        Image.fromarray(RGB_image).save(os.path.join(rgb_output_folder, output_filename))

    # Step 3: Create or load the keogram
    keogram, last_processed_minute = load_existing_keogram(parameters['keogram_dir'], date_input, spectrograph)

    # Step 4: Add RGB columns to the keogram
    keogram = add_rgb_columns(keogram, parameters['RGB_folder'], last_processed_minute, date_input, spectrograph)

    # Step 5: Save the final keogram
    if add_subplots:
        save_keogram_with_subplots(keogram, parameters['keogram_dir'], date_input, spectrograph)
    else:
        save_keogram_with_axes(keogram, parameters['keogram_dir'], spectrograph)

    print(f"Keogram for {date_input} has been created and saved.")
    
if __name__ == "__main__":
    main()

