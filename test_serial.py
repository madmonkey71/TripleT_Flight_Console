#!/usr/bin/env python3
"""
Test script for serial communication debugging
"""

import sys
from data_comm import DataCommManager

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_serial.py <serial_port> [baud_rate]")
        print("Example: python test_serial.py /dev/ttyUSB0 115200")
        sys.exit(1)
        
    port = sys.argv[1]
    baud_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    
    print(f"Testing serial connection to {port} at {baud_rate} baud...")
    
    comm = DataCommManager()
    
    # Test the connection
    if comm.test_serial_connection(port, baud_rate):
        print("\nConnection test successful!")
    else:
        print("\nConnection test failed. Check the error messages above.")
        sys.exit(1)
        
    # Try to receive some data
    print("\nAttempting to receive data for 5 seconds...")
    comm.connect_serial(port, baud_rate)
    
    try:
        import time
        start_time = time.time()
        while time.time() - start_time < 5:
            if comm.serial_conn.in_waiting > 0:
                data = comm.serial_conn.read(comm.serial_conn.in_waiting).decode('utf-8', errors='replace')
                print(f"Received: {data.strip()}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        comm.close()
        print("\nTest completed")

if __name__ == "__main__":
    main() 