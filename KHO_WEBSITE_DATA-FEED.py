# Import the configuration dictionary from the main script
from main_script import config  # Replace 'main_script' with the actual name of your main script if different
import os
import shutil
import time
from PIL import Image

# Define paths from the configuration dictionary
processed_spectrogram_dir = config['processed_spectrogram_dir']
keogram_dir = config['keogram_dir']
feed_dir = config['feed_dir']

# Ensure Feed directory exists
os.makedirs(feed_dir, exist_ok=True)

# Track last copied files to avoid redundant copies
last_copied_spectrogram = None
last_copied_keogram = None

def verify_image_integrity(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
        with Image.open(file_path) as img:
            img.load()  # Ensure the file can be loaded fully
        return True
    except Exception as e:
        print(f"Corrupted or incomplete file detected: {file_path} - {e}")
        return False

def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return latest_file

def copy_latest_to_feed():
    global last_copied_spectrogram, last_copied_keogram

    # Retrieve the latest processed spectrogram
    latest_spectrogram = get_latest_file(processed_spectrogram_dir)
    if latest_spectrogram and latest_spectrogram != last_copied_spectrogram:
        spectrogram_path = os.path.join(processed_spectrogram_dir, latest_spectrogram)
        if verify_image_integrity(spectrogram_path):
            shutil.copy(spectrogram_path, feed_dir)
            last_copied_spectrogram = latest_spectrogram
            print(f"Copied spectrogram: {latest_spectrogram} to Feed directory.")
    
    # Retrieve the latest keogram
    latest_keogram = get_latest_file(keogram_dir)
    if latest_keogram and latest_keogram != last_copied_keogram:
        keogram_path = os.path.join(keogram_dir, latest_keogram)
        if verify_image_integrity(keogram_path):
            shutil.copy(keogram_path, feed_dir)
            last_copied_keogram = latest_keogram
            print(f"Copied keogram: {latest_keogram} to Feed directory.")

def main():
    while True:
        start_time = time.time()

        # Check for new files and update the feed directory
        copy_latest_to_feed()

        # Calculate processing time and determine sleep time
        elapsed_time = time.time() - start_time
        sleep_time = max(30 - elapsed_time, 0)  # Sleep for 30 seconds minus processing time
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()

