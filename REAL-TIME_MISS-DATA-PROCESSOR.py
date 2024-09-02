'''
MAIN SPECTROGRAM PROCESSING PROGRAM: Launches and make sure all involved scripts are active (minute-averaged spectrogram 
with spectral and spatial analysis and keogram based on latest available data from MISS 1 or 2 updated live). 

1. Initialises a list to track subprocesses.
2. Defines a function to terminate all subprocesses safely (ctrl + C)
4. Launches multiple subprocesses (keogram maker, RGB column maker, average PNG maker, and SPECTROGRAM_PROCESSOR).
5. Enters a loop to keep processes running, checking every 60 seconds.
6. On interrupt, stops all subprocesses and exits.


Author: Nicolas Martinez (UNIS/LTU)
Last update: August 2024

'''

import signal
import subprocess
import os
import time
import shutil
from PIL import Image

processes = []  # List to keep track of all subprocesses
running = True  # Manage the while loop

# Define paths
home_dir = os.path.expanduser("~")
processed_spectrogram_dir = os.path.join(home_dir, ".venvMISS2", "MISS2", "Captured_PNG", "Processed_spectrograms")
keogram_dir = os.path.join(home_dir, ".venvMISS2", "MISS2", "Keograms")
feed_dir = os.path.join(home_dir, ".venvMISS2", "MISS2", "Feed")

# Ensure Feed directory exists
os.makedirs(feed_dir, exist_ok=True)

# Track last copied files to avoid redundant copies
last_copied_spectrogram = None
last_copied_keogram = None

def stop_processes(processes, timeout=5):
    """Stop all subprocesses gracefully."""
    for process in processes:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

def signal_handler(sig, frame):
    """Handle interrupt signal to safely stop the program."""
    global running
    running = False
    print("Interrupt received, stopping processes...")
    stop_processes(processes)
    exit(0)

def start_subprocess(script_name):
    """Start a subprocess and return it."""
    base_dir = os.path.expanduser("~/.venvMISS2/MISS2/MISS_SOFTWARE-PACKAGE")
    script_path = os.path.join(base_dir, script_name)
    print(f"Starting process: {script_name}")
    process = subprocess.Popen(["python3", script_path])
    return process

def verify_processes(processes):
    """Verify that all subprocesses are running."""
    all_running = True
    for process in processes:
        if process.poll() is not None:
            print(f"Process {process.pid} has stopped.")
            all_running = False
    return all_running

def verify_image_integrity(file_path):
    """Check if an image is complete and not corrupted."""
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
    """Get the latest file from a directory."""
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return latest_file

def copy_latest_to_feed():
    """Copy the latest processed spectrogram and keogram to the Feed directory."""
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

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    last_execution_time = time.time()

    try:
        while running:
            current_time = time.time()
            if current_time - last_execution_time >= 60:
                last_execution_time = current_time

                # Stop and clear existing processes
                stop_processes(processes)
                processes = []

                # Start new processes
                processes.append(start_subprocess("KEOGRAM_MAKER.PY"))
                processes.append(start_subprocess("RGB_COLUMN_MAKER.PY"))
                processes.append(start_subprocess("AVERAGE_PNG_MAKER.PY"))
                processes.append(start_subprocess("SPECTROGRAM_PROCESSOR.PY"))

                # Confirm that all processes were started
                if verify_processes(processes):
                    print("All subprocesses started successfully.")
                else:
                    print("One or more subprocesses failed to start.")

                # Copy the latest spectrogram and keogram to the Feed directory
                copy_latest_to_feed()

            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        stop_processes(processes)
