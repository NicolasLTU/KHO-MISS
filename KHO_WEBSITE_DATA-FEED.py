'''
Feeds https://kho.unis.no/Data.html with the latest processed spectrogram and keogram it finds.

Author: Nicolas Martinez
Latest update: August 2024
'''


import os
import shutil
import time
from PIL import Image

def update_website_feed(config):
    feed_dir = config['feed_dir']
    processed_spectrogram_dir = config['processed_spectrogram_dir']
    keogram_dir = config['keogram_dir']

    # Ensure the feed directory exists
    os.makedirs(feed_dir, exist_ok=True)

    last_copied_spectrogram = None
    last_copied_keogram = None

    # Main loop to keep checking for new files and update the website feed
    while True:
        # Copy the latest processed spectrogram
        latest_spectrogram = get_latest_file(processed_spectrogram_dir)
        if latest_spectrogram and latest_spectrogram != last_copied_spectrogram:
            spectrogram_path = os.path.join(processed_spectrogram_dir, latest_spectrogram)
            if verify_image_integrity(spectrogram_path):
                shutil.copy(spectrogram_path, feed_dir)
                last_copied_spectrogram = latest_spectrogram
                print(f"Copied spectrogram: {latest_spectrogram} to Feed directory.")

        # Copy the latest keogram
        latest_keogram = get_latest_file(keogram_dir)
        if latest_keogram and latest_keogram != last_copied_keogram:
            keogram_path = os.path.join(keogram_dir, latest_keogram)
            if verify_image_integrity(keogram_path):
                shutil.copy(keogram_path, feed_dir)
                last_copied_keogram = latest_keogram
                print(f"Copied keogram: {latest_keogram} to Feed directory.")

        # Wait before checking again
        time.sleep(60)  # Check for updates every minute

def get_latest_file(directory):
    """Get the latest file from a directory."""
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return latest_file

def verify_image_integrity(file_path):
    """Verify the integrity of an image file."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
        with Image.open(file_path) as img:
            img.load()  # Ensure the file can be loaded fully
        return True
    except Exception as e:
        print(f"Corrupted or incomplete file detected: {file_path} - {e}")
        return False

# Main entry point of the script
if __name__ == "__main__":
    update_website_feed(config)
