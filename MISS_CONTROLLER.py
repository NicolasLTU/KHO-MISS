'''
This program operates the MISS's Atik 414EX and Sunshield Lens shutter when the darkness conditions are fullfilled. 

1. Initialise serial communication for SunShield control.
2. Define a function to terminate smoothly (ctrl + C)
3. Check if it is nighttime using a specific function.
4. Manage the Atik camera and SunShield based on nighttime conditions.
5. Run a loop that checks conditions every 60 seconds and updates the system.
6. Handle interrupts by stopping subprocesses and exiting the script.

Author: Nicolas Martinez (UNIS/LTU)
Last update: August 2024

'''

import signal
import subprocess
import time
import serial
import os
from night_condition_calculator import it_is_nighttime  # Program used to check if the Sun is below -10 degrees of elevation at KHO (Kjell Henriksen Observatory), returns a Boolean
from sunshield_controller import SunShield_CLOSE, SunShield_OPEN, init_serial  # Control of the SunShield shutter: Close, Open, Settings for communication to the Serial Port 'COM3'

running = True
image_capture_process = None  # Track the Atik_controller.py process
is_currently_night = None  # To manage the day-night transition

def stop_processes(processes, timeout=5):
    for process in processes:
        process.terminate()  # Ask the process to terminate
        try:
            process.wait(timeout=timeout)  # Wait for the process to terminate
        except subprocess.TimeoutExpired:
            print(f"Process {process.pid} did not terminate in time. Forcing termination.")
            process.kill()  # Forcefully terminate the process

def signal_handler(sig, frame):
    global running
    stop_processes([image_capture_process])
    running = False
    print("Stopped the program.")
    exit(0)

def check_atik_process():
    """Check if the Atik process is running."""
    if image_capture_process:
        if image_capture_process.poll() is not None:
            # Process has exited
            print("Atik camera process has stopped or failed to start.")
            return False
    return True

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)  # Handle interrupt signal to safely stop the program

    try:
        ser = init_serial()  # Initialise serial communication for SunShield control
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        ser = None

    last_execution_time = time.time()  # Track the last execution time for the 60-second interval

    try:
        while running:
            current_time = time.time()
            if current_time - last_execution_time >= 60:  # Check every 60 seconds
                last_execution_time = current_time

                if it_is_nighttime():
                    if is_currently_night is not True:  # Check for transition day-night
                        is_currently_night = True
                        print('Nighttime: Camera and SunShield Operational')
                    if ser:
                        SunShield_OPEN(ser)  # Open the SunShield
                    if not image_capture_process or not check_atik_process():  # Only start the process if it's not running or has failed
                        script_path = os.path.join(os.path.expanduser("~"), ".venvMISS2", "MISS2", "MISS_SOFTWARE-PACKAGE", "CAPTURE_ATIK.py")
                        try:
                            image_capture_process = subprocess.Popen(["python", script_path])
                            print("Atik camera process started.")
                        except Exception as e:
                            print(f"Failed to start Atik camera process: {e}")

                else:  # Daytime
                    if is_currently_night is not False:  # Check for transition night-day
                        is_currently_night = False
                        print("Daytime: Camera and SunShield OFF")
                        if image_capture_process:
                            print("Stopping image capture...")
                            stop_processes([image_capture_process])
                            image_capture_process = None
                        if ser:
                            SunShield_CLOSE(ser)  # Close the SunShield

    except KeyboardInterrupt:
        print("Interrupt received, cleaning up...")
    finally:
        if image_capture_process:
            stop_processes([image_capture_process])
