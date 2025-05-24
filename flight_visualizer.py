"""
DEPRECATION NOTICE: This script (flight_visualizer.py) is deprecated.
Its 3D attitude visualization functionality has been integrated into 
the main application (flight_console.py). 
Please use flight_console.py for all visualizations.
"""
#!/usr/bin/env python3
"""
Flight Data Visualizer

This script creates multiple visualization windows for different aspects of flight data:
- GPS trajectory
- Altitude over time
- Pressure and temperature
- Motion sensor data
- 3D attitude visualization using quaternions
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication
import time
from data_comm import DataCommManager

class FlightVisualizer:
    def __init__(self):
        # Initialize PyQt application
        self.app = QApplication(sys.argv)
        
        # Create data storage
        self.timestamps = []
        self.altitudes = []
        self.pressures = []
        self.temperatures = []
        self.latitudes = []
        self.longitudes = []
        self.accelerations = []
        self.gyros = []
        self.quaternions = []
        
        # Create windows
        self.setup_windows()
        
    def setup_windows(self):
        # Create GPS trajectory window
        plt.figure(1)
        plt.title('GPS Trajectory')
        self.gps_plot = plt.plot([], [], 'b-')[0]
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.gca().set_aspect('equal')
        
        # Create altitude window
        plt.figure(2)
        plt.title('Altitude')
        self.alt_plot = plt.plot([], [], 'r-')[0]
        plt.xlabel('Time (ms)')
        plt.ylabel('Altitude (m)')
        plt.grid(True)
        plt.gca().set_ylim(0, 1000)  # Adjust based on expected altitude range
        
        # Create pressure and temperature window
        plt.figure(3)
        plt.title('Pressure and Temperature')
        self.press_plot = plt.plot([], [], 'b-', label='Pressure (hPa)')[0]
        self.temp_plot = plt.plot([], [], 'r-', label='Temperature (°C)')[0]
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.gca().set_ylim(900, 1100)  # Pressure range
        
        # Create motion sensors window
        plt.figure(4)
        plt.title('Motion Sensors')
        self.accel_plots = [plt.plot([], [], label=f'Accel {axis}')[0] for axis in ['X', 'Y', 'Z']]
        self.gyro_plots = [plt.plot([], [], '--', label=f'Gyro {axis}')[0] for axis in ['X', 'Y', 'Z']]
        plt.xlabel('Time (ms)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.gca().set_ylim(-16, 16)  # Typical range for accelerometer/gyro
        
        # Create 3D attitude visualization
        self.attitude_window = gl.GLViewWidget()
        self.attitude_window.setWindowTitle('3D Attitude')
        self.attitude_window.setGeometry(100, 100, 800, 600)
        
        # Create coordinate system grids
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        
        # Rotate y and z grids
        ygrid.rotate(90, 0, 1, 0)
        zgrid.rotate(90, 1, 0, 0)
        
        # Set grid sizes and add to view
        for grid in (xgrid, ygrid, zgrid):
            grid.setSize(x=10, y=10, z=10)
            grid.setSpacing(x=1, y=1, z=1)
            self.attitude_window.addItem(grid)
        
        # Create reference frame axes (static)
        ref_x = gl.GLLinePlotItem(pos=np.array([[0,0,0], [2,0,0]]), color=(0.8,0.2,0.2,1), width=2)
        ref_y = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,2,0]]), color=(0.2,0.8,0.2,1), width=2)
        ref_z = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,2]]), color=(0.2,0.2,0.8,1), width=2)
        
        self.attitude_window.addItem(ref_x)
        self.attitude_window.addItem(ref_y)
        self.attitude_window.addItem(ref_z)
        
        # Create body frame axes (will be rotated)
        self.body_x = gl.GLLinePlotItem(pos=np.array([[0,0,0], [2,0,0]]), color=(1,0,0,1), width=3)
        self.body_y = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,2,0]]), color=(0,1,0,1), width=3)
        self.body_z = gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,2]]), color=(0,0,1,1), width=3)
        
        self.attitude_window.addItem(self.body_x)
        self.attitude_window.addItem(self.body_y)
        self.attitude_window.addItem(self.body_z)
        
        # Set initial camera position for better view
        self.attitude_window.setCameraPosition(distance=15, elevation=30, azimuth=45)
        
        self.attitude_window.show()
        
        # Set all figures to be non-blocking
        plt.ion()
        for i in range(1, 5):
            plt.figure(i)
            plt.tight_layout()
        
    def quaternion_to_matrix(self, q):
        """Convert quaternion to 4x4 transformation matrix"""
        q0, q1, q2, q3 = q
        # Create 3x3 rotation matrix
        rot = np.array([
            [1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2)]
        ])
        # Convert to 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot
        return transform
        
    def update_plots(self, data):
        # Parse the data
        try:
            values = data.split(',')
            if len(values) < 30:  # Skip incomplete packets
                return
                
            # Try to convert all values to float first
            float_values = []
            for val in values[:30]:  # Only process first 30 values
                try:
                    float_values.append(float(val.strip()))
                except (ValueError, TypeError):
                    return  # Skip this packet if any conversion fails
            
            # Extract values
            timestamp = float_values[0]
            lat = float_values[3]
            lon = float_values[4]
            alt = float_values[5]
            pressure = float_values[11]
            temp = float_values[12]
            
            # Motion data
            accel = float_values[13:16]
            gyro = float_values[16:19]
            
            # Quaternion
            q = float_values[26:30]
            
            # Update data arrays
            self.timestamps.append(timestamp)
            self.altitudes.append(alt)
            self.pressures.append(pressure)
            self.temperatures.append(temp)
            self.latitudes.append(lat)
            self.longitudes.append(lon)
            self.accelerations.append(accel)
            self.gyros.append(gyro)
            self.quaternions.append(q)
            
            # Keep only last 1000 points
            max_points = 1000
            if len(self.timestamps) > max_points:
                self.timestamps = self.timestamps[-max_points:]
                self.altitudes = self.altitudes[-max_points:]
                self.pressures = self.pressures[-max_points:]
                self.temperatures = self.temperatures[-max_points:]
                self.latitudes = self.latitudes[-max_points:]
                self.longitudes = self.longitudes[-max_points:]
                self.accelerations = self.accelerations[-max_points:]
                self.gyros = self.gyros[-max_points:]
                self.quaternions = self.quaternions[-max_points:]
            
            # Update plots only if we have data
            if len(self.timestamps) > 0:
                # Update GPS plot
                self.gps_plot.set_data(self.longitudes, self.latitudes)
                ax = plt.figure(1).gca()
                ax.relim()
                ax.autoscale_view()
                plt.figure(1).canvas.draw_idle()
                
                # Update altitude plot
                self.alt_plot.set_data(self.timestamps, self.altitudes)
                ax = plt.figure(2).gca()
                ax.relim()
                ax.autoscale_view()
                plt.figure(2).canvas.draw_idle()
                
                # Update pressure and temperature plot
                self.press_plot.set_data(self.timestamps, self.pressures)
                self.temp_plot.set_data(self.timestamps, self.temperatures)
                ax = plt.figure(3).gca()
                ax.relim()
                ax.autoscale_view()
                plt.figure(3).canvas.draw_idle()
                
                # Update motion sensors plot
                for i in range(3):
                    self.accel_plots[i].set_data(self.timestamps, [a[i] for a in self.accelerations])
                    self.gyro_plots[i].set_data(self.timestamps, [g[i] for g in self.gyros])
                ax = plt.figure(4).gca()
                ax.relim()
                ax.autoscale_view()
                plt.figure(4).canvas.draw_idle()
                
                # Update 3D attitude
                if len(self.quaternions) > 0:
                    q = self.quaternions[-1]
                    self.update_attitude_display(q)
            
            # Process events
            self.app.processEvents()
            
        except Exception as e:
            print(f"Error processing data packet: {e}")
            return
        
    def update_attitude_display(self, q):
        """Update the 3D attitude display with new quaternion."""
        # Convert quaternion to rotation matrix
        R = self.quaternion_to_matrix(q)[:3, :3]  # Get just the 3x3 rotation part
        
        # Update body frame axes
        x_end = np.dot(R, np.array([2, 0, 0]))
        y_end = np.dot(R, np.array([0, 2, 0]))
        z_end = np.dot(R, np.array([0, 0, 2]))
        
        self.body_x.setData(pos=np.array([[0,0,0], x_end]))
        self.body_y.setData(pos=np.array([[0,0,0], y_end]))
        self.body_z.setData(pos=np.array([[0,0,0], z_end]))
        
    def run(self, port, baud_rate):
        # Connect to serial port
        comm = DataCommManager()
        if not comm.connect_serial(port, baud_rate):
            print("Failed to connect to serial port")
            return
            
        try:
            while True:
                if comm.serial_conn.in_waiting > 0:
                    data = comm.serial_conn.read(comm.serial_conn.in_waiting).decode('utf-8', errors='replace')
                    for line in data.split('\n'):
                        if line.strip():
                            self.update_plots(line.strip())
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
        finally:
            comm.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python flight_visualizer.py <serial_port> [baud_rate]")
        print("Example: python flight_visualizer.py /dev/ttyACM1 115200")
        sys.exit(1)
        
    port = sys.argv[1]
    baud_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
    
    visualizer = FlightVisualizer()
    visualizer.run(port, baud_rate)

if __name__ == "__main__":
    main() 