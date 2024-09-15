"""
This script processes PNG images captured by MISS*, averages them minute-wise, and saves the averaged images in a 
shared `averaged_PNG` directory. The script checks the device name in the image metadata to determine which device captured the image. 
The processed images are then saved using the device name in the filename, and the relevant metadata is preserved with a note indicating that the image is a 1-minute spectrogram average.

Parameters:
1. PNG_base_folder (str): The base directory where the averaged images will be saved.
2. raw_PNG_folder (str): The directory containing the raw PNG images.
3. current_time (datetime): The current UTC time.
4. processed_minutes (list): A list to keep track of already processed minutes

Author: Nicolas Martinez (UNIS/LTU)
Last update: September 2024
"""

import os
import datetime
import time
import numpy as np
from PIL import Image, PngImagePlugin
import re
from collections import defaultdict
from parameters import parameters  # Import the parameters dictionary

def get_device_name_from_metadata(filepath):
    '''
    Extracts the device name (MISS1, MISS2...) from the PNG metadata.

    Parameters:
        filepath (str): The file path of the image.

    Returns:
        str: The device name if found, otherwise None.
    '''
    try:
        img = Image.open(filepath)
        metadata = img.info
        note = metadata.get("Note", "")
        if note:
            match = re.search(r'(MISS\d)', note)
            if match: 
                return match.group(1)
        return None
    except Exception as e:
        print(f"Error reading metadata from {os.path.basename(filepath)}: {e}")
        return None

def average_images(PNG_base_folder, raw_PNG_folder, current_time, processed_minutes):

    images_by_minute = defaultdict(list)
    filename_regex = re.compile(r'^.+-(\d{8})-(\d{6})\.png$')  # Regex to match filenames

    # Convert current time to UTC
    current_time_utc = current_time.astimezone(datetime.timezone.utc)

    # Group images based on the minute they belong to
    for root, dirs, files in os.walk(raw_PNG_folder):
        for filename in files:
            filepath = os.path.join(root, filename)
            match = filename_regex.match(filename)
            if match:
                date_part, time_part = match.groups()
                image_date = datetime.datetime.strptime(date_part, "%Y%m%d").replace(tzinfo=datetime.timezone.utc).date()
                current_date = current_time_utc.date()
                if image_date == current_date:
                    image_utc = datetime.datetime.strptime(date_part + time_part[:4], "%Y%m%d%H%M").replace(tzinfo=datetime.timezone.utc)
                    prev_minute = current_time_utc - datetime.timedelta(minutes=1)
                    if image_utc < prev_minute:
                        images_by_minute[date_part + '-' + time_part[:4]].append(filepath)

    # Process each minute-group of images IF not already processed
    for minute_key, filepaths in images_by_minute.items():
        if minute_key not in processed_minutes:
            year, month, day, hour, minute = map(int, [minute_key[:4], minute_key[4:6], minute_key[6:8], minute_key[9:11], minute_key[11:]])
            target_utc = datetime.datetime(year, month, day, hour, minute, tzinfo=datetime.timezone.utc)

            if target_utc < current_time_utc - datetime.timedelta(seconds=15):

                sum_img_array = None
                count = 0
                device_name = None
                metadata = None  # Variable to hold metadata

                for filepath in filepaths:
                    try:
                        if not device_name:
                            device_name = get_device_name_from_metadata(filepath)

                        img = Image.open(filepath)
                        img_array = np.array(img)

                        if sum_img_array is None:
                            sum_img_array = np.zeros_like(img_array, dtype='float64')

                        sum_img_array += img_array
                        count += 1

                        # Capture metadata from the first image
                        if metadata is None:
                            metadata = img.info

                    except Exception as e:
                        print(f"Error processing image {os.path.basename(filepath)}: {e}")

                # If images were found for this minute, average them and save
                if count > 0 and device_name:
                    averaged_image = (sum_img_array / count).astype(np.uint16)
                    
                    # Create a shared directory for averaged PNGs
                    averaged_PNG_folder = os.path.join(PNG_base_folder, "averaged_PNG")
                    os.makedirs(averaged_PNG_folder, exist_ok=True)

                    # Create a date-specific folder within the shared directory
                    save_folder = os.path.join(averaged_PNG_folder, f"{year:04d}", f"{month:02d}", f"{day:02d}")
                    os.makedirs(save_folder, exist_ok=True)

                    # Save the averaged image
                    averaged_image_path = os.path.join(save_folder, f"{device_name}-{year:04d}{month:02d}{day:02d}-{hour:02d}{minute:02d}00.png")

                    # Convert numpy array back to an Image object and specify the mode for 16-bit
                    averaged_img = Image.fromarray(averaged_image, mode='I;16')

                    # Preserve the original metadata and add a note about the averaging
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
                    
                    print(f"Saved averaged image with metadata: {averaged_image_path}")

                    # Update the list of already processed minutes
                    processed_minutes.append(minute_key)

# Use the parameters dictionary to get paths
raw_PNG_folder = parameters['raw_PNG_folder']
PNG_base_folder = parameters['averaged_PNG_folder']

# List to keep track of processed minutes 
processed_minutes = []

while True:
    try:
        current_time = datetime.datetime.now(datetime.timezone.utc)
        average_images(PNG_base_folder, raw_PNG_folder, current_time, processed_minutes)
        time.sleep(30 - (current_time.second % 30))  # Sleep until 30 seconds past the minute
    except Exception as e:
        print(f"An error occurred: {e}")
