import os
import shutil
import time
from PIL import Image
import logging
from parameters import PROCESSED_SPECTROGRAM_DIR, KEOGRAM_DIR, FEED_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Track last copied files to avoid redundant copies
last_copied_spectrogram = None
last_copied_keogram = None

def verify_image_integrity(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
        return True
    except Exception as e:
        logging.error(f"Corrupted or incomplete file detected: {file_path} - {e}")
        return False

def get_latest_file(directory):
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            return None
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
        return latest_file
    except Exception as e:
        logging.error(f"Error retrieving latest file from {directory}: {e}")
        return None

def copy_latest_to_feed():
    global last_copied_spectrogram, last_copied_keogram

    # Retrieve the latest processed spectrogram
    latest_spectrogram = get_latest_file(PROCESSED_SPECTROGRAM_DIR)
    if latest_spectrogram and latest_spectrogram != last_copied_spectrogram:
        spectrogram_path = os.path.join(PROCESSED_SPECTROGRAM_DIR, latest_spectrogram)
        if verify_image_integrity(spectrogram_path):
            shutil.copy(spectrogram_path, FEED_DIR)
            last_copied_spectrogram = latest_spectrogram
            logging.info(f"Copied spectrogram: {latest_spectrogram} to Feed directory.")
    
    # Retrieve the latest keogram
    latest_keogram = get_latest_file(KEOGRAM_DIR)
    if latest_keogram and latest_keogram != last_copied_keogram:
        keogram_path = os.path.join(KEOGRAM_DIR, latest_keogram)
        if verify_image_integrity(keogram_path):
            shutil.copy(keogram_path, FEED_DIR)
            last_copied_keogram = latest_keogram
            logging.info(f"Copied keogram: {latest_keogram} to Feed directory.")

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
