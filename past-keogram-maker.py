import os
import numpy as np
from PIL import Image, PngImagePlugin
from scipy import signal
import re
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directories and parameters
home_dir = os.path.expanduser("~")
raw_PNG_folder = os.path.join(home_dir, ".venvMISS2", "MISS2", "Captured_PNG", "raw_PNG")
averaged_PNG_folder = os.path.join(home_dir, ".venvMISS2", "MISS2", "Captured_PNG", "averaged_PNG")
rgb_dir_base = os.path.join(home_dir, ".venvMISS2", "MISS2", "RGB_columns")
output_dir = os.path.join(home_dir, ".venvMISS2", "MISS2", "Keograms")

# Spectrogram's binning binX = binY
binX = 1  # ADJUST ACCORDINGLY!

# Horizon limits for MISS1 and MISS2, adjusted for binning
miss1_horizon_limits = (280 // binX, 1140 // binX)
miss2_horizon_limits = (271 // binX, 1116 // binX)

processed_images = set()

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
def average_images(PNG_base_folder, raw_PNG_folder, processed_minutes, device_name):
    print("Starting the averaging of images...")
    images_by_minute = defaultdict(list)
    filename_regex = re.compile(r'^.+-(\d{8})-(\d{6})\.png$')

    for root, dirs, files in os.walk(raw_PNG_folder):
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

                date_specific_folder = os.path.join(PNG_base_folder, f"{year:04d}", f"{month:02d}", f"{day:02d}")
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

# Process and average emission line rows
def process_emission_line(spectro_array, emission_row, binX, pixel_range):
    num_rows_to_average = max(1, int(12 / binX))
    start_row = max(emission_row - num_rows_to_average // 2, 0)
    end_row = min(emission_row + num_rows_to_average // 2, spectro_array.shape[0])

    spectro_array_cropped = spectro_array[pixel_range[0]:pixel_range[1], :]
    extracted_rows = spectro_array_cropped[start_row:end_row, :]
    processed_rows = signal.medfilt2d(extracted_rows.astype('float32'))
    averaged_row = np.mean(processed_rows, axis=0)
    return averaged_row.flatten()

# Create the RGB image from the extracted rows
def create_rgb_column(spectro_array, row_630, row_558, row_428, binX, pixel_range):
    # Process each emission line and extract the corresponding rows
    column_RED = process_emission_line(spectro_array, row_630, binX, pixel_range)
    column_GREEN = process_emission_line(spectro_array, row_558, binX, pixel_range)
    column_BLUE = process_emission_line(spectro_array, row_428, binX, pixel_range)

    # Stack the columns together to form an RGB image
    true_rgb_image = np.stack((column_RED, column_GREEN, column_BLUE), axis=-1)

    # Resize to ensure the output is in (300, 1, 3) format if needed
    if true_rgb_image.shape[0] != 300:
        true_rgb_image = np.resize(true_rgb_image, (300, 1, 3))

    # Ensure the output image is saved in 24-bit RGB format
    return true_rgb_image

# Create RGB columns for the day
def create_rgb_columns_for_day(date_str, spectrograph):
    global processed_images

    print(f"Starting the creation of RGB columns for {date_str} using {spectrograph}...")
    spectro_path_dir = os.path.join(averaged_PNG_folder, date_str)
    ensure_directory_exists(spectro_path_dir)
    output_folder = os.path.join(rgb_dir_base, date_str)
    ensure_directory_exists(output_folder)

    matching_files = [f for f in os.listdir(spectro_path_dir) if f.endswith(".png")]

    for filename in tqdm(matching_files, desc="Creating RGB columns", unit="image"):
        if filename in processed_images:
            continue

        png_file_path = os.path.join(spectro_path_dir, filename)

        if not verify_image_integrity(png_file_path):
            print(f"Skipping corrupted image: {filename}")
            continue

        spectro_data = np.array(Image.open(png_file_path))

        if spectrograph == "MISS1":
            pixel_range = miss1_horizon_limits
        elif spectrograph == "MISS2":
            pixel_range = miss2_horizon_limits
        else:
            print(f"Unknown spectrograph type for {filename}")
            continue

        RGB_image = create_rgb_column(spectro_data, 724, 723, 140, binX, pixel_range)
        
        # Check the shape before creating the image
        print(f"Processing {filename} - RGB_image shape: {RGB_image.shape}")
        
        if RGB_image.shape != (300, 1, 3):
            print(f"Error: Unexpected shape {RGB_image.shape} for {filename}. Skipping this image.")
            continue
        
        RGB_pil_image = Image.fromarray(RGB_image.astype('uint8'), mode='RGB')
        resized_RGB_image = RGB_pil_image.resize((1, 300), Image.Resampling.LANCZOS)

        base_filename = filename[:-4]
        output_filename = f"{base_filename[:-2]}00.png"
        output_filename_path = os.path.join(output_folder, output_filename)

        resized_RGB_image.save(output_filename_path)
        print(f"Saved RGB column image: {output_filename}")

        processed_images.add(filename)

    print(f"RGB column creation completed for {date_str}.")

# Initialize an empty keogram with white pixels
def initialise_keogram():
    return np.full((300, 24 * 60, 3), 255, dtype=np.uint8)

# Load an existing keogram or create a new one if none exists
def load_existing_keogram(output_dir, date_str, spectrograph):
    keogram_path = os.path.join(output_dir, date_str, f'{spectrograph}-keogram-{date_str.replace("/", "")}.png')

    if os.path.exists(keogram_path):
        with Image.open(keogram_path) as img:
            keogram = np.array(img)

        if keogram.shape != (300, 24 * 60, 3):
            keogram = initialise_keogram()  # Reinitialize to correct dimensions
            last_processed_minute = 0
        else:
            last_processed_minute = np.max(np.where(np.any(keogram != 255, axis=0))[0])

        print(f"Loaded existing keogram for {date_str}.")
        return keogram, last_processed_minute
    else:
        print(f"No existing keogram found for {date_str}. Creating a new one.")
        return initialise_keogram(), 0

# Add RGB columns to the keogram
def add_rgb_columns(keogram, base_dir, last_processed_minute, date_str, spectrograph):
    today_RGB_dir = os.path.join(base_dir, date_str)

    if not os.path.exists(today_RGB_dir):
        print(f"No directory found for the date ({today_RGB_dir}). Proceeding with blank RGB data.")
        today_RGB_dir = None  # Indicate that the directory doesn't exist

    for minute in tqdm(range(last_processed_minute + 1, 24 * 60), desc="Adding RGB columns to keogram", unit="minute"):
        filename = f"{spectrograph}-{date_str.replace('/', '')}-{minute // 60:02d}{minute % 60:02d}00.png"
        file_path = os.path.join(today_RGB_dir, filename) if today_RGB_dir else None

        if file_path and os.path.exists(file_path) and verify_image_integrity(file_path):
            try:
                rgb_data = np.array(Image.open(file_path))

                if rgb_data.shape != (300, 1, 3):
                    print(f"Unexpected image shape {rgb_data.shape} for {filename}. Expected (300, 1, 3). Skipping this image.")
                    continue

                keogram[:, minute:minute+1, :] = rgb_data

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        else:
            if 24 * 60 - minute > 4:
                keogram[:, minute:minute+1, :] = np.zeros((300, 1, 3), dtype=np.uint8)

    print(f"RGB columns added to keogram for {date_str}.")
    return keogram

# Save the keogram with proper labeling
def save_keogram(keogram, output_dir, date_str, spectrograph):
    current_date_dir = os.path.join(output_dir, date_str)
    ensure_directory_exists(current_date_dir)

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.imshow(keogram, aspect='auto', extent=[0, 24*60, 90, -90])
    ax.set_title(f"{spectrograph} Keogram for {date_str.replace('/', '-')}", fontsize=20)
    
    # Set x-axis for every 2 hours
    x_ticks = np.append(np.arange(0, 24*60, 120), 24*60)
    x_labels = [f"{hour}:00" for hour in range(0, 24, 2)] + ["24:00"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (UT)")

    # Set y-axis for elevation angles
    y_ticks = np.linspace(-90, 90, num=7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['90° S', '60° S', '30° S', 'Zenith', '30° N', '60° N', '90° N'])
    ax.set_ylim(-90, 90)
    ax.set_ylabel("Elevation angle [degrees]")

    keogram_filename = os.path.join(current_date_dir, f'{spectrograph}-keogram-{date_str.replace("/", "")}.png')
    plt.savefig(keogram_filename)
    plt.close(fig)
    print(f"Keogram saved as: {keogram_filename}")

# Main function for the entire process
def main():
    date_input = input("Enter the date to process (yyyy/mm/dd): ")
    spectrograph = input("Enter the spectrograph name (MISS1 or MISS2): ")

    processed_minutes = []
    average_images(averaged_PNG_folder, raw_PNG_folder, processed_minutes, spectrograph)  # Generate averaged PNGs
    create_rgb_columns_for_day(date_input, spectrograph)  # Generate RGB columns from the averaged PNGs
    keogram, last_processed_minute = load_existing_keogram(output_dir, date_input, spectrograph)
    keogram = add_rgb_columns(keogram, rgb_dir_base, last_processed_minute, date_input, spectrograph)
    save_keogram(keogram, output_dir, date_input, spectrograph)

if __name__ == "__main__":
    main()
