'''
This script makes sure that all non-needed processing data (averaged spectrograms, RGB columns, processed spectrograms) contained in YYYY/MM/DD date directories from past days is erased.

Author: Nicolas Martinez (LTU/UNIS)

Created: September 2024

'''

import time
from datetime import datetime, timezone
import parameters
import shutil
import os


def delete_old_directories(base_path):
    current_time = datetime.now(timezone.utc)

    for root, dirs, files in os.walk(base_path, topdown=False):  # Checks days, then months, then years
        for d in dirs:
            dir_path = os.path.join(root, d)
            print(f"Processing directory: {dir_path}")  # Debug output

            # Split the path to get directory parts relative to base_path
            parts = os.path.relpath(dir_path, base_path).split(os.sep)

            # Handle YYYY/MM directories
            try:
                if len(parts) == 2:  # Only year and month (YYYY/MM)
                    year = int(parts[-2])
                    month = int(parts[-1])

                    # Construct the directory date using the first day of the month
                    dir_date = datetime(year, month, 1, tzinfo=timezone.utc)

                    # Determine if the directory is older than the criteria (10 minutes older than current time)
                    if dir_date < (current_time - timedelta(minutes=10)):
                        print(f"Deleting directory: {dir_path}")
                        shutil.rmtree(dir_path)
                    else:
                        print(f"Keeping directory: {dir_path}")
                else:
                    print(f"Skipping directory: {dir_path} (Not a YYYY/MM path)")
            except (ValueError, IndexError) as e:
                print(f"Skipping directory {dir_path}: {e}")

# Main function to run the cleanup every ten minutes
def main():
    paths = {
        'raw_PNG_folder': parameters.parameters['raw_PNG_folder'],
        'averaged_PNG_folder': parameters.parameters['averaged_PNG_folder'],
        'RGB_folder': parameters.parameters['RGB_folder']
    }
    
    while True:
        print(f"Running cleanup at {datetime.now(timezone.utc)}")
        
        for path in paths.values():
            print(f"Processing path: {path}")
            delete_old_directories(path)

        # Sleep for 600 seconds (10 minutes)
        time.sleep(600)

# Entry point of the script
if __name__ == "__main__":
    main()
