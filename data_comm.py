"""
DataComm Module

This module provides functionality for handling communication with flight controllers
via various interfaces including serial, USB, and network connections.
"""

import serial
import serial.tools.list_ports
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import socket
import io
import pandas as pd
from io import StringIO
import traceback


class DataCommManager:
    """
    A class to manage communication with flight data sources.
    
    This class provides methods for connecting to various data sources,
    receiving data, and managing the data flow to other components.
    """
    
    def __init__(self, data_callback: Callable[[str], None] = None, buffer_size: int = 1024):
        """
        Initialize the DataCommManager object.
        
        Args:
            data_callback: Function to call when new data is received
            buffer_size: Size of receive buffer
        """
        self.data_callback = data_callback
        self.buffer_size = buffer_size
        self.serial_conn = None
        self.tcp_conn = None
        self.udp_socket = None
        self.running = False
        self.receive_thread = None
        self.data_queue = queue.Queue()
        self.line_buffer = ""
    
    def list_serial_ports(self) -> List[Dict[str, str]]:
        """
        List available serial ports on the system.
        
        Returns:
            List of dictionaries with port information
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            # Skip ttyS ports
            if 'ttyS' in port.device:
                continue
            ports.append({
                'device': port.device,
                'description': port.description,
                'manufacturer': port.manufacturer if hasattr(port, 'manufacturer') else None,
                'hwid': port.hwid
            })
        return ports
    
    def connect_serial(self, port: str, baud_rate: int) -> bool:
        """
        Connect to a serial port and start receiving data.
        
        Args:
            port: Serial port device name
            baud_rate: Baud rate for the connection
            
        Returns:
            True if connection successful, False otherwise
        """
        print(f"DataCommManager: Attempting to connect to {port} at {baud_rate} baud...")
        if self.serial_conn:
            print("DataCommManager: Already connected. Closing existing connection first.")
            self.close()
        
        try:
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=baud_rate,
                timeout=0.1,
                write_timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print("DataCommManager: Serial object created.")
            if not self.serial_conn.is_open:
                print("DataCommManager: Failed to open serial port immediately.")
                self.serial_conn = None
                return False
            
            print("DataCommManager: Serial port is open. Starting receive thread...")
            self.running = True
            self.receive_thread = threading.Thread(target=self._read_serial, daemon=True)
            self.receive_thread.start()
            print("DataCommManager: Receive thread started.")
            
            # Test the connection
            self.serial_conn.flushInput()
            self.serial_conn.flushOutput()
            
            # Start receive thread if callback is provided
            if self.data_callback is not None:
                self._start_receive_thread('serial')
            
            return True
        
        except serial.SerialException as e:
            print(f"DataCommManager: Serial connection failed: {e}")
            self.serial_conn = None
            return False
        except Exception as e:
            print(f"DataCommManager: Unexpected error during serial connection: {e}")
            traceback.print_exc()
            self.serial_conn = None
            return False
    
    def connect_tcp(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """
        Connect to a TCP server.
        
        Args:
            host: Server hostname or IP address
            port: TCP port number
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Close existing connection if any
            if self.tcp_conn is not None:
                self.tcp_conn.close()
            
            # Create new connection
            self.tcp_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_conn.settimeout(timeout)
            self.tcp_conn.connect((host, port))
            
            # Start receive thread if callback is provided
            if self.data_callback is not None:
                self._start_receive_thread('tcp')
            
            return True
        
        except Exception as e:
            print(f"TCP connection error: {e}")
            return False
    
    def connect_udp(self, local_port: int, remote_host: Optional[str] = None, remote_port: Optional[int] = None) -> bool:
        """
        Set up a UDP socket for sending/receiving data.
        
        Args:
            local_port: Local UDP port to bind to
            remote_host: Remote host for sending (optional)
            remote_port: Remote port for sending (optional)
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Close existing connection if any
            if self.udp_socket is not None:
                self.udp_socket.close()
            
            # Create new socket
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind(('0.0.0.0', local_port))
            
            # Set remote address if provided
            self.remote_udp_addr = None
            if remote_host is not None and remote_port is not None:
                self.remote_udp_addr = (remote_host, remote_port)
            
            # Start receive thread if callback is provided
            if self.data_callback is not None:
                self._start_receive_thread('udp')
            
            return True
        
        except Exception as e:
            print(f"UDP setup error: {e}")
            return False
    
    def _start_receive_thread(self, conn_type: str) -> None:
        """
        Start a thread for receiving data from the connection.
        
        Args:
            conn_type: Type of connection ('serial', 'tcp', or 'udp')
        """
        if self.receive_thread is not None and self.receive_thread.is_alive():
            self.running = False
            self.receive_thread.join(timeout=1.0)
        
        self.running = True
        self.line_buffer = ""
        
        if conn_type == 'serial':
            self.receive_thread = threading.Thread(target=self._read_serial)
        elif conn_type == 'tcp':
            self.receive_thread = threading.Thread(target=self._receive_tcp_data)
        elif conn_type == 'udp':
            self.receive_thread = threading.Thread(target=self._receive_udp_data)
        
        self.receive_thread.daemon = True
        self.receive_thread.start()
    
    def _read_serial(self) -> None:
        """Continuously read data from the serial port."""
        print("_read_serial thread started.")
        while self.running and self.serial_conn:
            try:
                # Read available bytes
                bytes_available = self.serial_conn.in_waiting
                if bytes_available > 0:
                    raw_data = self.serial_conn.read(bytes_available)
                    
                    # Decode data and handle potential errors
                    try:
                        data = raw_data.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError as decode_error:
                        continue # Skip this chunk if decoding fails
                        
                    # Append to line buffer
                    self.line_buffer += data
                    
                    # Process complete lines
                    while '\n' in self.line_buffer:
                        line, self.line_buffer = self.line_buffer.split('\n', 1)
                        line = line.strip() # Remove leading/trailing whitespace including \r
                        if line:
                            self.data_queue.put(line)
                else:
                    # Sleep briefly if no data is available to avoid busy-waiting
                    time.sleep(0.01)
            except serial.SerialException as e:
                print(f"Serial error in _read_serial: {e}")
                self.running = False # Stop the loop on serial error
                self.close() # Attempt to close the connection
                break
    
    def _receive_tcp_data(self) -> None:
        """Thread function for receiving data from TCP connection."""
        if self.tcp_conn is None:
            return
        
        self.tcp_conn.settimeout(0.1)  # Short timeout for non-blocking reads
        
        while self.running:
            try:
                data = self.tcp_conn.recv(self.buffer_size).decode('utf-8', errors='replace')
                if not data:  # Connection closed by remote
                    break
                self._process_received_data(data)
            except socket.timeout:
                continue  # Just a timeout, try again
            except Exception as e:
                print(f"TCP read error: {e}")
                time.sleep(0.1)
    
    def _receive_udp_data(self) -> None:
        """Thread function for receiving data from UDP connection."""
        if self.udp_socket is None:
            return
        
        self.udp_socket.settimeout(0.1)  # Short timeout for non-blocking reads
        
        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(self.buffer_size)
                self._process_received_data(data.decode('utf-8', errors='replace'))
            except socket.timeout:
                continue  # Just a timeout, try again
            except Exception as e:
                print(f"UDP read error: {e}")
                time.sleep(0.1)
    
    def _process_received_data(self, data: str) -> None:
        """
        Process received data and call the callback function for complete lines.
        
        Args:
            data: Received data string
        """
        self.line_buffer += data
        
        # Process complete lines
        while '\n' in self.line_buffer:
            line, self.line_buffer = self.line_buffer.split('\n', 1)
            line = line.strip()
            
            if line:  # Skip empty lines
                if self.data_callback:
                    self.data_callback(line)
                else:
                    self.data_queue.put(line)
    
    def send_data(self, data: str, conn_type: str = None) -> bool:
        """
        Send data through the selected connection.
        
        Args:
            data: Data string to send
            conn_type: Connection type to use ('serial', 'tcp', 'udp', or None for auto-detect)
            
        Returns:
            True if data was sent successfully, False otherwise
        """
        if not data.endswith('\n'):
            data += '\n'
        
        data_bytes = data.encode('utf-8')
        
        # Auto-detect connection type if not specified
        if conn_type is None:
            if self.serial_conn is not None and self.serial_conn.is_open:
                conn_type = 'serial'
            elif self.tcp_conn is not None:
                conn_type = 'tcp'
            elif self.udp_socket is not None and self.remote_udp_addr is not None:
                conn_type = 'udp'
            else:
                return False
        
        # Send data
        try:
            if conn_type == 'serial' and self.serial_conn is not None:
                self.serial_conn.write(data_bytes)
                return True
            elif conn_type == 'tcp' and self.tcp_conn is not None:
                self.tcp_conn.sendall(data_bytes)
                return True
            elif conn_type == 'udp' and self.udp_socket is not None and self.remote_udp_addr is not None:
                self.udp_socket.sendto(data_bytes, self.remote_udp_addr)
                return True
            
            return False
        
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def get_queued_data(self, timeout: float = 0.1) -> Optional[str]:
        """
        Get the next line of data from the queue.
        
        Args:
            timeout: Time to wait for data (in seconds)
            
        Returns:
            Data string or None if timeout
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def close(self) -> None:
        """Close all connections and stop receive threads."""
        self.running = False
        
        if self.receive_thread is not None and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
        
        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
        
        if self.tcp_conn is not None:
            self.tcp_conn.close()
        
        if self.udp_socket is not None:
            self.udp_socket.close()
        
        self.serial_conn = None
        self.tcp_conn = None
        self.udp_socket = None
        self.receive_thread = None
    
    def test_serial_connection(self, port: str, baud_rate: int = 115200, timeout: float = 1.0) -> bool:
        """
        Test the serial connection and verify data format.
        
        Args:
            port: Serial port device name
            baud_rate: Baud rate for the connection
            timeout: Connection timeout in seconds
            
        Returns:
            True if test successful, False otherwise
        """
        try:
            print(f"\nTesting serial connection to {port}...")
            
            # Try to connect
            if not self.connect_serial(port, baud_rate):
                print("Failed to establish connection")
                return False
            
            # Wait for data
            print("Waiting for data (timeout: {}s)...".format(timeout))
            start_time = time.time()
            data_received = False
            
            while time.time() - start_time < timeout:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='replace')
                    if data:
                        print(f"Received test data: {data.strip()}")
                        data_received = True
                        break
                time.sleep(0.1)
            
            if not data_received:
                print("No data received during test period")
                return False
            
            # Verify data format
            try:
                # Try to parse the data as CSV
                df = pd.read_csv(StringIO(data))
                print(f"Data format verified - Found {len(df.columns)} columns")
                return True
            except Exception as e:
                print(f"Data format verification failed: {str(e)}")
                return False
            
        except Exception as e:
            print(f"Test failed: {str(e)}")
            return False
        finally:
            self.close() 