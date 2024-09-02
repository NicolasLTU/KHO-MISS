'''

This script provides functionality to control the SunShield shutter via serial communication. 
The shutter can be commanded to open or close by sending specific commands to the device connected to the specified serial port. 
The script includes functions for initializing the serial port, opening the shutter, and closing the shutter.

Author: Nicolas Martinez (UNIS/LTU)
Last Update: 2024

'''

import serial
import serial.tools.list_ports

def init_serial():
    '''
    Initializes the serial connection to the SunShield shutter.
    
    Returns:
        Serial object if successful, None otherwise.
    '''
    try:
        ser = serial.Serial(
            port='COM3',  # Specify the correct port here!!
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )  # Parameters according to SunShield User Manual (Keo Scientific)
        print("Serial port initialized successfully.")
        return ser
    except FileNotFoundError as e:
        print(f"Failed to open serial port: {e}")
        return None

def SunShield_CLOSE(ser):
    '''
    Sends the CLOSE command (S1\r) to the SunShield shutter.
    
    Parameters:
        ser (Serial): The serial connection to the SunShield.
    '''
    try:
        print("Sending CLOSE command...")
        ser.write(b'S1\r')
        response = ser.readline()
        print(f"Response to CLOSE command: {response.decode().strip()}")
    except serial.SerialException as e:
        print(f"Serial error: {e}")

def SunShield_OPEN(ser):
    '''
    Sends the OPEN command (S0\r) to the SunShield shutter.
    
    Parameters:
        ser (Serial): The serial connection to the SunShield.
    '''
    try:
        print("Sending OPEN command...")
        ser.write(b'S0\r')
        response = ser.readline()
        print(f"Response to OPEN command: {response.decode().strip()}")
    except serial.SerialException as e:
        print(f"Serial error: {e}")

if __name__ == "__main__":
    ser = init_serial()  
    if ser:
        SunShield_OPEN(ser)     
        SunShield_CLOSE(ser)   
        ser.close()
    else:
        print("Unable to initialize serial connection.")
