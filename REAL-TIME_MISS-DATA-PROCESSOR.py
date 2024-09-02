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

processes = []  # List to keep track of all subprocesses
running = True  # Manage the while loop

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
    process = subprocess.Popen(["python", script_path])
    return process

def verify_processes(processes):
    """Verify that all subprocesses are running."""
    all_running = True
    for process in processes:
        if process.poll() is not None:
            print(f"Process {process.pid} has stopped.")
            all_running = False
    return all_running

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
                processes.append(start_subprocess("TEST_KEO_ANALYTICS.py"))
                processes.append(start_subprocess("RGB_column_maker.py"))
                processes.append(start_subprocess("average_PNG_maker.py"))
                processes.append(start_subprocess("SPECTROGRAM_PROCESSOR.py"))

                # Confirm that all processes were started
                if verify_processes(processes):
                    print("All subprocesses started successfully.")
                else:
                    print("One or more subprocesses failed to start.")

            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        stop_processes(processes)
