"""
Flight Console Application

This is the main application module for the TripleT Flight Console.
It provides a GUI for visualizing flight data from model planes and rocketry,
supporting both real-time data streaming and historical data analysis.
"""

import sys
import os
import time
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTabWidget, QPushButton, QLabel, QComboBox, QLineEdit, 
                            QFileDialog, QGroupBox, QGridLayout, QCheckBox, QSplitter,
                            QMessageBox, QSpinBox, QStatusBar, QAction, QMenu, QToolBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QThread, QUrl
from PyQt5.QtGui import QIcon, QFont
# Import WebEngine for Google Maps integration
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
except ImportError:
    print("QtWebEngineWidgets not available. Google Maps integration will be disabled.")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pyqtgraph.opengl as gl

# Import our modules
from flight_data_parser import FlightDataParser
from data_visualizer import DataVisualizer
from data_comm import DataCommManager


class DataReceiverThread(QThread):
    """Thread for receiving data from various sources."""
    
    data_received = pyqtSignal(str)
    
    def __init__(self, comm_manager):
        super().__init__()
        self.comm_manager = comm_manager
        self.running = False
    
    def run(self):
        self.running = True
        print("DataReceiverThread started.")
        while self.running:
            try:
                data = self.comm_manager.get_queued_data(timeout=0.1)
                if data:
                    self.data_received.emit(data)
                # Add a small sleep to prevent busy-waiting if queue is often empty
                # time.sleep(0.01) # Optional: uncomment if CPU usage is high
            except Exception as e:
                print(f"Error in DataReceiverThread: {e}")
                traceback.print_exc()
        print("DataReceiverThread stopped.")
    
    def stop(self):
        self.running = False
        self.wait()


class RealTimePlotWidget(QWidget):
    """Widget for displaying real-time plots."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Ensure we have a QApplication instance
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
            
        self.max_points = 1000
        
        # Initialize data arrays
        self.timestamps = np.array([])
        self.altitudes = np.array([])
        self.pressures = np.array([])
        self.temperatures = np.array([])
        self.latitudes = np.array([])
        self.longitudes = np.array([])
        self.accelerations = np.array([]).reshape(0, 3)
        self.gyros = np.array([]).reshape(0, 3)
        self.quaternions = np.array([]).reshape(0, 4)
        
        # For storing current values for text display
        self.current_sats = 0
        self.current_speed = 0
        self.current_heading = 0
        
        # For Google Maps
        self.maps_html = None
        
        # Setup plot windows
        self.setup_windows()
        
    def setup_windows(self):
        """Set up the visualization windows."""
        # Create plots for altitude, pressure and temperature
        self.altitude_plot = pg.PlotWidget()
        self.altitude_plot.setTitle("Altitude")
        self.altitude_plot.setLabel('left', 'Altitude', units='m')
        self.altitude_plot.setLabel('bottom', 'Time', units='s')
        self.altitude_curve = self.altitude_plot.plot(pen='y')
        
        # Separate plots for pressure and temperature
        self.pressure_plot = pg.PlotWidget()
        self.pressure_plot.setTitle("Pressure")
        self.pressure_plot.setLabel('left', 'Pressure', units='hPa')
        self.pressure_plot.setLabel('bottom', 'Time', units='s')
        self.pressure_curve = self.pressure_plot.plot(pen='c')
        
        self.temperature_plot = pg.PlotWidget()
        self.temperature_plot.setTitle("Temperature")
        self.temperature_plot.setLabel('left', 'Temperature', units='°C')
        self.temperature_plot.setLabel('bottom', 'Time', units='s')
        self.temperature_curve = self.temperature_plot.plot(pen='r')
        
        # Create plot for accelerometer and gyroscope data
        self.motion_plot = pg.PlotWidget()
        self.motion_plot.setTitle("Motion")
        self.motion_plot.addLegend()
        
        # Accelerometer curves
        self.accel_x_curve = self.motion_plot.plot(pen='r', name='Accel X')
        self.accel_y_curve = self.motion_plot.plot(pen='g', name='Accel Y')
        self.accel_z_curve = self.motion_plot.plot(pen='b', name='Accel Z')
        
        # Gyroscope curves
        self.gyro_x_curve = self.motion_plot.plot(pen=(255, 0, 0, 100), name='Gyro X')
        self.gyro_y_curve = self.motion_plot.plot(pen=(0, 255, 0, 100), name='Gyro Y')
        self.gyro_z_curve = self.motion_plot.plot(pen=(0, 0, 255, 100), name='Gyro Z')
        
        # Create 3D plot for attitude visualization
        self.attitude_view = gl.GLViewWidget()
        self.attitude_view.setWindowTitle('Attitude')
        self.attitude_view.setCameraPosition(distance=5, elevation=30, azimuth=45)
        
        # Create a grid for the 3D view
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        self.attitude_view.addItem(grid)
        
        # Create orientation lines (x, y, z axes)
        # X-axis (Red)
        x_axis_pts = np.array([[0, 0, 0], [1, 0, 0]])
        self.orientation_x = gl.GLLinePlotItem(pos=x_axis_pts, color=(1, 0, 0, 1), width=2)
        self.attitude_view.addItem(self.orientation_x)
        
        # Y-axis (Green)
        y_axis_pts = np.array([[0, 0, 0], [0, 1, 0]])
        self.orientation_y = gl.GLLinePlotItem(pos=y_axis_pts, color=(0, 1, 0, 1), width=2)
        self.attitude_view.addItem(self.orientation_y)
        
        # Z-axis (Blue)
        z_axis_pts = np.array([[0, 0, 0], [0, 0, 1]])
        self.orientation_z = gl.GLLinePlotItem(pos=z_axis_pts, color=(0, 0, 1, 1), width=2)
        self.attitude_view.addItem(self.orientation_z)
        
        # Create main layout for the widget
        layout = QVBoxLayout()
        
        # Top row with 3D view and GPS
        top_row = QHBoxLayout()
        
        # Make the 3D view larger
        top_row.addWidget(self.attitude_view, 2)  # Higher stretch factor
        
        # Initialize map view if WebEngine is available
        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        
        try:
            if 'QWebEngineView' in globals():
                self.map_view = QWebEngineView()
                print("DEBUG: QWebEngineView initialized successfully")
                map_layout.addWidget(self.map_view)
            else:
                print("DEBUG: QWebEngineView not available, falling back to GPS plot")
                # Add placeholder for GPS if WebEngine is not available
                gps_plot = pg.PlotWidget()
                gps_plot.setTitle("GPS Position")
                gps_plot.setLabel('left', 'Latitude')
                gps_plot.setLabel('bottom', 'Longitude')
                # Store reference to the GPS plot
                self.gps_plot = gps_plot
                map_layout.addWidget(gps_plot)
                self.map_view = None
        except Exception as e:
            print(f"Error initializing map view: {e}")
            # Add placeholder for GPS
            gps_plot = pg.PlotWidget()
            gps_plot.setTitle("GPS Position")
            gps_plot.setLabel('left', 'Latitude')
            gps_plot.setLabel('bottom', 'Longitude')
            # Store reference to the GPS plot
            self.gps_plot = gps_plot
            map_layout.addWidget(gps_plot)
            self.map_view = None
        
        top_row.addWidget(map_container, 1)
        
        layout.addLayout(top_row)
        
        # Middle row with altitude, pressure and temperature
        middle_row = QHBoxLayout()
        middle_row.addWidget(self.altitude_plot)
        middle_row.addWidget(self.pressure_plot)
        middle_row.addWidget(self.temperature_plot)
        layout.addLayout(middle_row)
        
        # Bottom row with accelerometer and gyroscope data
        layout.addWidget(self.motion_plot)
        
        # Set margins and spacing
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.setLayout(layout)
        
    def create_aircraft_mesh(self):
        """Create a simple aircraft mesh for 3D visualization."""
        try:
            # Create a simple aircraft shape
            verts = np.array([
                [0, 0, 0],     # Nose
                [-1, 0.5, 0],  # Right wing tip
                [-1, -0.5, 0], # Left wing tip
                [-0.5, 0, 0],  # Body mid-point
                [-1, 0, 0.5],  # Vertical stabilizer top
            ])
            
            # Scale the aircraft
            scale = 2.0
            verts *= scale
            
            # Define the faces
            faces = np.array([
                [0, 1, 2],     # Wing triangle
                [0, 3, 1],     # Right wing
                [0, 2, 3],     # Left wing
                [0, 4, 3],     # Vertical stabilizer
            ])
            
            # Create the mesh data
            mesh = gl.MeshData(vertexes=verts, faces=faces)
            return mesh
        except Exception as e:
            print(f"Error creating aircraft mesh: {e}")
            # Return a simple fallback mesh if there's an error
            verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            faces = np.array([[0, 1, 2]])
            return gl.MeshData(vertexes=verts, faces=faces)

    def update_plots(self):
        """Update all plots with the latest data."""
        try:
            
            # Update 2D plots
            if len(self.altitudes) > 0:
                self.altitude_curve.setData(self.timestamps, self.altitudes)
            
            # Update pressure plot
            if len(self.pressures) > 0:
                self.pressure_curve.setData(self.timestamps, self.pressures)
            
            # Update temperature plot
            if len(self.temperatures) > 0:
                self.temperature_curve.setData(self.timestamps, self.temperatures)
            
            # Update motion sensors plot
            if len(self.accelerations) > 0:
                self.accel_x_curve.setData(self.timestamps, self.accelerations[:, 0])
                self.accel_y_curve.setData(self.timestamps, self.accelerations[:, 1])
                self.accel_z_curve.setData(self.timestamps, self.accelerations[:, 2])
            
            if len(self.gyros) > 0:
                self.gyro_x_curve.setData(self.timestamps, self.gyros[:, 0])
                self.gyro_y_curve.setData(self.timestamps, self.gyros[:, 1])
                self.gyro_z_curve.setData(self.timestamps, self.gyros[:, 2])
            
            # Update 3D attitude if quaternions are available
            if len(self.quaternions) > 0:
                # Get the latest quaternion
                q = self.quaternions[-1]
                
                # Convert quaternion to rotation matrix
                R = np.array([
                    [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
                    [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
                    [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
                ])
                
                # Convert 3x3 rotation matrix to 4x4 transformation matrix
                T = np.eye(4)  # Create 4x4 identity matrix
                T[0:3, 0:3] = R  # Set the upper-left 3x3 sub-matrix to R
                
                # Apply rotation to orientation lines
                # Convert numpy array to list for Transform3D
                matrix_list = T.tolist()
                transform = pg.Transform3D(matrix_list)
                self.orientation_x.setTransform(transform)
                self.orientation_y.setTransform(transform)
                self.orientation_z.setTransform(transform)
            
            # Force update of all plot widgets
            self.altitude_plot.update()
            self.pressure_plot.update()
            self.temperature_plot.update()
            self.motion_plot.update()
            self.attitude_view.update()
            
            # Process Qt events to keep UI responsive
            self.app.processEvents()
            
        except Exception as e:
            print(f"Error updating plots: {e}")
            import traceback
            traceback.print_exc()

    def add_data_point(self, data_point):
        """Add a new data point to all plots."""
        try:
            print(f"DEBUG: add_data_point called with data: {data_point}")
            
            # Extract timestamp
            if 'Timestamp' in data_point and data_point['Timestamp'] is not None:
                timestamp = float(data_point['Timestamp'])
                self.timestamps = np.append(self.timestamps, timestamp)
            else:
                # If no timestamp, use incremental counter
                if len(self.timestamps) == 0:
                    next_timestamp = 0
                else:
                    next_timestamp = self.timestamps[-1] + 1
                self.timestamps = np.append(self.timestamps, next_timestamp)
            
            # Extract altitude
            if 'Alt' in data_point and data_point['Alt'] is not None:
                alt_val = float(data_point['Alt'])
                self.altitudes = np.append(self.altitudes, alt_val)
            elif len(self.altitudes) > 0:
                # If no new data, repeat last value
                self.altitudes = np.append(self.altitudes, self.altitudes[-1])
            else:
                self.altitudes = np.append(self.altitudes, 0)
            
            # Extract pressure
            if 'Pressure' in data_point and data_point['Pressure'] is not None:
                self.pressures = np.append(self.pressures, float(data_point['Pressure']))
            elif len(self.pressures) > 0:
                self.pressures = np.append(self.pressures, self.pressures[-1])
            else:
                self.pressures = np.append(self.pressures, 0)
            
            # Extract temperature
            if 'Temperature' in data_point and data_point['Temperature'] is not None:
                self.temperatures = np.append(self.temperatures, float(data_point['Temperature']))
            elif len(self.temperatures) > 0:
                self.temperatures = np.append(self.temperatures, self.temperatures[-1])
            else:
                self.temperatures = np.append(self.temperatures, 0)
            
            # Extract GPS
            if 'Lat' in data_point and 'Long' in data_point and data_point['Lat'] is not None and data_point['Long'] is not None:
                self.latitudes = np.append(self.latitudes, float(data_point['Lat']))
                self.longitudes = np.append(self.longitudes, float(data_point['Long']))
            elif len(self.latitudes) > 0 and len(self.longitudes) > 0:
                self.latitudes = np.append(self.latitudes, self.latitudes[-1])
                self.longitudes = np.append(self.longitudes, self.longitudes[-1])
            else:
                self.latitudes = np.append(self.latitudes, 0)
                self.longitudes = np.append(self.longitudes, 0)
            
            # Extract acceleration
            new_accel = np.zeros(3)
            if 'ICM_AccelX' in data_point and 'ICM_AccelY' in data_point and 'ICM_AccelZ' in data_point and \
               data_point['ICM_AccelX'] is not None and data_point['ICM_AccelY'] is not None and data_point['ICM_AccelZ'] is not None:
                new_accel[0] = float(data_point['ICM_AccelX'])
                new_accel[1] = float(data_point['ICM_AccelY'])
                new_accel[2] = float(data_point['ICM_AccelZ'])
            elif len(self.accelerations) > 0:
                new_accel = self.accelerations[-1]
            
            self.accelerations = np.vstack([self.accelerations, new_accel]) if len(self.accelerations) > 0 else np.array([new_accel])
            
            # Extract gyro
            new_gyro = np.zeros(3)
            if 'ICM_GyroX' in data_point and 'ICM_GyroY' in data_point and 'ICM_GyroZ' in data_point and \
               data_point['ICM_GyroX'] is not None and data_point['ICM_GyroY'] is not None and data_point['ICM_GyroZ'] is not None:
                new_gyro[0] = float(data_point['ICM_GyroX'])
                new_gyro[1] = float(data_point['ICM_GyroY'])
                new_gyro[2] = float(data_point['ICM_GyroZ'])
            elif len(self.gyros) > 0:
                new_gyro = self.gyros[-1]
            
            self.gyros = np.vstack([self.gyros, new_gyro]) if len(self.gyros) > 0 else np.array([new_gyro])
            
            # Extract quaternion
            new_quat = np.zeros(4)
            if 'ICM_QuatW' in data_point and 'ICM_QuatX' in data_point and 'ICM_QuatY' in data_point and 'ICM_QuatZ' in data_point and \
               data_point['ICM_QuatW'] is not None and data_point['ICM_QuatX'] is not None and data_point['ICM_QuatY'] is not None and data_point['ICM_QuatZ'] is not None:
                new_quat[0] = float(data_point['ICM_QuatW'])
                new_quat[1] = float(data_point['ICM_QuatX'])
                new_quat[2] = float(data_point['ICM_QuatY'])
                new_quat[3] = float(data_point['ICM_QuatZ'])
            elif len(self.quaternions) > 0:
                new_quat = self.quaternions[-1]
            else:
                # Default orientation if no quaternion data is available
                # Identity quaternion (no rotation)
                new_quat = np.array([1.0, 0.0, 0.0, 0.0])
                print("DEBUG: Using default identity quaternion")
            
            self.quaternions = np.vstack([self.quaternions, new_quat]) if len(self.quaternions) > 0 else np.array([new_quat])
            
            # Update text display data
            if 'Sats' in data_point and data_point['Sats'] is not None:
                self.current_sats = int(float(data_point['Sats']))
            
            if 'Speed' in data_point and data_point['Speed'] is not None:
                self.current_speed = float(data_point['Speed'])
            
            if 'Heading' in data_point and data_point['Heading'] is not None:
                self.current_heading = float(data_point['Heading'])
            
            # Limit data arrays to max_points
            if len(self.timestamps) > self.max_points:
                self.timestamps = self.timestamps[-self.max_points:]
                self.altitudes = self.altitudes[-self.max_points:]
                self.pressures = self.pressures[-self.max_points:]
                self.temperatures = self.temperatures[-self.max_points:]
                self.latitudes = self.latitudes[-self.max_points:]
                self.longitudes = self.longitudes[-self.max_points:]
                self.accelerations = self.accelerations[-self.max_points:]
                self.gyros = self.gyros[-self.max_points:]
                self.quaternions = self.quaternions[-self.max_points:]
            
            # Update Google Maps if we have GPS data
            if len(self.latitudes) > 0 and len(self.longitudes) > 0:
                self.update_map()
            
            # Explicitly update the plots
            self.update_plots()
            
        except Exception as e:
            print(f"Error adding data point to plots: {e}")
            traceback.print_exc()
    
    def update_map(self):
        """Update the Google Maps view with current GPS data."""
        try:
            DEBUG: print("update_map called, map_view exists:", hasattr(self, 'map_view') and self.map_view is not None, 
                      "has setHtml:", hasattr(self, 'map_view') and self.map_view is not None and hasattr(self.map_view, 'setHtml'))
            
            if not hasattr(self, 'map_view') or self.map_view is None or not hasattr(self.map_view, 'setHtml'):
                return
            
            if not self.latitudes or not self.longitudes:
                # No GPS data yet
                return
            
            DEBUG: print(f"GPS data points: {len(self.latitudes)}, Last lat/long: {self.latitudes[-1]}, {self.longitudes[-1]}")
            
            # Get the current location (last point)
            current_lat = self.latitudes[-1]
            current_lng = self.longitudes[-1]
            
            DEBUG: print(f"Updating map with center at {current_lat}, {current_lng}")
            
            # Create path string with all points
            path_points = []
            for i in range(min(len(self.latitudes), len(self.longitudes))):
                path_points.append(f"{{lat: {self.latitudes[i]}, lng: {self.longitudes[i]}}}")
            
            path_str = ", ".join(path_points)
            
            # Generate HTML content with the map
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://maps.googleapis.com/maps/api/js?key="></script>
                <style>
                    html, body, #map {{
                        height: 100%;
                        width: 100%;
                        margin: 0;
                        padding: 0;
                    }}
                </style>
            </head>
            <body>
                <div id="map"></div>
                <script>
                    function initMap() {{
                        var map = new google.maps.Map(document.getElementById('map'), {{
                            zoom: 18,
                            center: {{lat: {current_lat}, lng: {current_lng}}},
                            mapTypeId: 'satellite'
                        }});
                        
                        // Create the flight path
                        var flightPath = new google.maps.Polyline({{
                            path: [{path_str}],
                            geodesic: true,
                            strokeColor: '#FF0000',
                            strokeOpacity: 1.0,
                            strokeWeight: 3
                        }});
                        
                        flightPath.setMap(map);
                        
                        // Add marker for current position with custom icon
                        var currentPositionMarker = new google.maps.Marker({{
                            position: {{lat: {current_lat}, lng: {current_lng}}},
                            map: map,
                            title: 'Current Position',
                            icon: {{
                                path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                                scale: 6,
                                fillColor: '#00FF00',
                                fillOpacity: 1,
                                strokeColor: '#000000',
                                strokeWeight: 1,
                                rotation: {self.current_heading if hasattr(self, 'current_heading') and self.current_heading is not None else 0}
                            }}
                        }});
                        
                        // Add distance markers every 10 points
                        var distanceMarkers = [];
                        var points = [{path_str}];
                        for (var i = 0; i < points.length; i += 10) {{
                            if (i > 0) {{
                                var marker = new google.maps.Marker({{
                                    position: points[i],
                                    map: map,
                                    icon: {{
                                        path: google.maps.SymbolPath.CIRCLE,
                                        scale: 3,
                                        fillColor: '#FFFF00',
                                        fillOpacity: 0.7,
                                        strokeColor: '#000000',
                                        strokeWeight: 1
                                    }}
                                }});
                                distanceMarkers.push(marker);
                            }}
                        }}
                    }}
                    
                    google.maps.event.addDomListener(window, 'load', initMap);
                </script>
            </body>
            </html>
            """
            
            # Update the HTML content in the QWebEngineView
            self.map_view.setHtml(html)
            DEBUG: print("Map HTML updated")
        except Exception as e:
            DEBUG: print(f"Error updating map: {e}")
    
    def clear_plots(self):
        """Clear all plot data."""
        try:
            # Reset data arrays
            self.timestamps = np.array([])
            self.altitudes = np.array([])
            self.pressures = np.array([])
            self.temperatures = np.array([])
            self.latitudes = np.array([])
            self.longitudes = np.array([])
            self.accelerations = np.array([]).reshape(0, 3)
            self.gyros = np.array([]).reshape(0, 3)
            self.quaternions = np.array([]).reshape(0, 4)
            
            # Clear plot curves
            self.update_plots()
            
            print("All plots cleared")
        except Exception as e:
            print(f"Error clearing plots: {e}")


class StaticPlotWidget(QWidget):
    """Widget for displaying static plots from loaded data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualizer = DataVisualizer()
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add plot controls
        control_layout = QHBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Time Series",
            "GPS Trajectory",
            "3D Trajectory",
            "Sensor Comparison"
        ])
        
        self.column_select_combo = QComboBox()
        self.column_select_combo.addItems([
            "GPS (Alt, Speed, Heading)",
            "KX134 Accelerometer",
            "ICM Accelerometer",
            "ICM Gyroscope",
            "ICM Magnetometer",
            "Environmental"
        ])
        
        self.plot_button = QPushButton("Create Plot")
        
        control_layout.addWidget(QLabel("Plot Type:"))
        control_layout.addWidget(self.plot_type_combo)
        control_layout.addWidget(QLabel("Data:"))
        control_layout.addWidget(self.column_select_combo)
        control_layout.addWidget(self.plot_button)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Add matplotlib canvas
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Connect signals
        self.plot_button.clicked.connect(self.create_plot)
        
        # Store data reference (will be set from main window)
        self.data = None
    
    def set_data(self, data):
        """Set the data to be plotted."""
        self.data = data
    
    def get_selected_columns(self):
        """Get the columns selected for plotting."""
        selection = self.column_select_combo.currentText()
        
        if "GPS" in selection:
            return ["Alt", "Speed", "Heading"]
        elif "KX134 Accelerometer" in selection:
            return ["KX134_AccelX", "KX134_AccelY", "KX134_AccelZ"]
        elif "ICM Accelerometer" in selection:
            return ["ICM_AccelX", "ICM_AccelY", "ICM_AccelZ"]
        elif "ICM Gyroscope" in selection:
            return ["ICM_GyroX", "ICM_GyroY", "ICM_GyroZ"]
        elif "ICM Magnetometer" in selection:
            return ["ICM_MagX", "ICM_MagY", "ICM_MagZ"]
        elif "Environmental" in selection:
            return ["Pressure", "Temperature"]
        
        return []
    
    def create_plot(self):
        """Create the selected plot type."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
        
        plot_type = self.plot_type_combo.currentText()
        
        # Clear the current figure
        self.figure.clear()
        
        if plot_type == "Time Series":
            columns = self.get_selected_columns()
            self.visualizer.create_time_series_plot(self.data, columns, ax=self.figure.add_subplot(111))
            
        elif plot_type == "GPS Trajectory":
            self.visualizer.create_gps_trajectory_plot(self.data, ax=self.figure.add_subplot(111))
            
        elif plot_type == "3D Trajectory":
            if any(col not in self.data.columns for col in ['Lat', 'Long', 'Alt']):
                QMessageBox.warning(self, "Missing Data", "GPS data is incomplete")
                return
            
            ax = self.figure.add_subplot(111, projection='3d')
            ax.plot(self.data['Long'], self.data['Lat'], self.data['Alt'], 'b-')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Altitude (m)')
            ax.set_title('3D Flight Trajectory')
        
        elif plot_type == "Sensor Comparison":
            sensor_groups = {
                "KX134": ["KX134_AccelX", "KX134_AccelY", "KX134_AccelZ"],
                "ICM": ["ICM_AccelX", "ICM_AccelY", "ICM_AccelZ"]
            }
            
            # Create subplots for each sensor group
            n_groups = len(sensor_groups)
            for i, (group_name, columns) in enumerate(sensor_groups.items()):
                ax = self.figure.add_subplot(n_groups, 1, i+1)
                for column in columns:
                    if column in self.data.columns:
                        ax.plot(self.data['Timestamp'], self.data[column], label=column)
                ax.set_title(f"{group_name} Sensors")
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
            
            self.figure.tight_layout()
        
        self.canvas.draw()


class MainWindow(QMainWindow):
    """Main window for the Flight Console application."""
    
    def __init__(self):
        super().__init__()
        
        # Set up the UI
        self.setWindowTitle("TripleT Flight Console")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.parser = FlightDataParser()
        self.visualizer = DataVisualizer()
        self.comm_manager = DataCommManager()
        
        # Initialize data headers
        self.data_headers = [
            "Timestamp", "FixType", "Sats", "Lat", "Long", "Alt", "AltMSL", 
            "Speed", "Heading", "pDOP", "RTK", "Pressure", "Temperature", 
            "KX134_AccelX", "KX134_AccelY", "KX134_AccelZ", "ICM_AccelX", 
            "ICM_AccelY", "ICM_AccelZ", "ICM_GyroX", "ICM_GyroY", "ICM_GyroZ", 
            "ICM_MagX", "ICM_MagY", "ICM_MagZ", "ICM_Temp"
        ]
        
        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tab contents
        self.create_data_tab()
        self.create_realtime_tab()
        self.create_analysis_tab()
        
        # Add status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create data receiver thread
        self.data_thread = DataReceiverThread(self.comm_manager)
        self.data_thread.data_received.connect(self.process_received_data)
        
        # Current connection status
        self.connected = False
    
    def create_data_tab(self):
        """Create the Data Connection tab."""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # File data section
        file_group = QGroupBox("File Data")
        file_layout = QHBoxLayout(file_group)
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file selected")
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)
        
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.load_file_data)
        
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_button)
        file_layout.addWidget(load_button)
        
        layout.addWidget(file_group)
        
        # Serial connection section
        serial_group = QGroupBox("Serial Connection")
        serial_layout = QGridLayout(serial_group)
        
        serial_layout.addWidget(QLabel("Port:"), 0, 0)
        self.port_combo = QComboBox()
        serial_layout.addWidget(self.port_combo, 0, 1)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_ports)
        serial_layout.addWidget(refresh_button, 0, 2)
        
        serial_layout.addWidget(QLabel("Baud Rate:"), 1, 0)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"])
        self.baud_combo.setCurrentText("115200")
        serial_layout.addWidget(self.baud_combo, 1, 1)
        
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        serial_layout.addWidget(self.connect_button, 1, 2)
        
        layout.addWidget(serial_group)
        
        # Network connection section
        network_group = QGroupBox("Network Connection")
        network_layout = QGridLayout(network_group)
        
        network_layout.addWidget(QLabel("Protocol:"), 0, 0)
        self.protocol_combo = QComboBox()
        self.protocol_combo.addItems(["TCP", "UDP"])
        network_layout.addWidget(self.protocol_combo, 0, 1)
        
        network_layout.addWidget(QLabel("Host:"), 1, 0)
        self.host_edit = QLineEdit("localhost")
        network_layout.addWidget(self.host_edit, 1, 1)
        
        network_layout.addWidget(QLabel("Port:"), 2, 0)
        self.network_port_edit = QLineEdit("8080")
        network_layout.addWidget(self.network_port_edit, 2, 1)
        
        self.network_connect_button = QPushButton("Connect")
        self.network_connect_button.clicked.connect(self.connect_network)
        network_layout.addWidget(self.network_connect_button, 2, 2)
        
        layout.addWidget(network_group)
        
        # Data preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.data_preview_label = QLabel("No data loaded")
        preview_layout.addWidget(self.data_preview_label)
        
        layout.addWidget(preview_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Add tab
        self.tabs.addTab(data_tab, "Data Connection")
        
        # Populate ports initially
        self.refresh_ports()
    
    def create_realtime_tab(self):
        """Create the real-time visualization tab."""
        # Create a container widget for the tab
        realtime_tab_container = QWidget()
        tab_layout = QVBoxLayout(realtime_tab_container)

        # Add refresh rate control
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Update Rate (ms):"))
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(10, 1000)
        self.update_rate_spin.setValue(100)
        self.update_rate_spin.setSingleStep(10)
        control_layout.addWidget(self.update_rate_spin)

        # Add clear button
        clear_button = QPushButton("Clear Plots")
        # Connect the clear button to the clear_plots method of the RealTimePlotWidget instance
        # We need to ensure self.real_time_plot exists before connecting
        # Defer connection slightly or ensure instance is created first

        control_layout.addWidget(clear_button)
        control_layout.addStretch()

        # Add the control layout to the main tab layout
        tab_layout.addLayout(control_layout)

        # Create and add the plot widget
        self.real_time_plot = RealTimePlotWidget(self)
        # self.real_time_plot.setup_windows() # This is called in __init__
        tab_layout.addWidget(self.real_time_plot) # Add plot widget below controls

        # Now connect the clear button since self.real_time_plot exists
        clear_button.clicked.connect(self.real_time_plot.clear_plots)

        # Add the container widget as the tab
        self.tabs.addTab(realtime_tab_container, "Real-time Visualization")

        # Remove the erroneous line that added controls to the plot widget's layout
        # self.real_time_plot.layout().addLayout(control_layout) # REMOVED
    
    def create_analysis_tab(self):
        """Create the Data Analysis tab."""
        analysis_tab = QWidget()
        layout = QVBoxLayout(analysis_tab)
        
        # Create static plot widget
        self.static_plot = StaticPlotWidget()
        layout.addWidget(self.static_plot)
        
        # Add tab
        self.tabs.addTab(analysis_tab, "Data Analysis")
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        open_action = QAction("Open Data File", self)
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Current Data", self)
        save_action.triggered.connect(self.save_data)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Connection menu
        conn_menu = menu_bar.addMenu("Connection")
        
        refresh_ports_action = QAction("Refresh Serial Ports", self)
        refresh_ports_action.triggered.connect(self.refresh_ports)
        conn_menu.addAction(refresh_ports_action)
        
        conn_menu.addSeparator()
        
        self.connect_action = QAction("Connect", self)
        self.connect_action.triggered.connect(self.toggle_connection)
        conn_menu.addAction(self.connect_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def browse_file(self):
        """Open a file dialog to select a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
    
    def load_file_data(self):
        """Load data from the selected file."""
        file_path = self.file_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "No File Selected", "Please select a file first")
            return
        
        try:
            data = self.parser.load_from_file(file_path)
            self.update_data_preview(data)
            self.static_plot.set_data(data)
            
            self.status_bar.showMessage(f"Loaded {len(data)} data points from {os.path.basename(file_path)}")
        
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Error: {str(e)}")
    
    def refresh_ports(self):
        """Refresh the list of available serial ports."""
        self.port_combo.clear()
        ports = self.comm_manager.list_serial_ports()
        
        for port in ports:
            self.port_combo.addItem(f"{port['device']} - {port['description']}", port['device'])
        
        if self.port_combo.count() == 0:
            self.port_combo.addItem("No ports available")
            self.connect_button.setEnabled(False)
        else:
            self.connect_button.setEnabled(True)
    
    def toggle_connection(self):
        """Connect to or disconnect from the selected serial port."""
        if not self.connected:
            self.connect_serial()
        else:
            self.disconnect()
    
    def connect_serial(self):
        """Connect to the selected serial port."""
        if self.port_combo.count() == 0:
            return
        
        port = self.port_combo.currentData()
        if port is None:
            return
        
        baud_rate = int(self.baud_combo.currentText())
        
        if self.comm_manager.connect_serial(port, baud_rate):
            self.connected = True
            self.connect_button.setText("Disconnect")
            self.connect_action.setText("Disconnect")
            self.data_thread.start()
            
            self.status_bar.showMessage(f"Connected to {port} at {baud_rate} baud")
        else:
            QMessageBox.critical(self, "Connection Failed", "Failed to connect to the selected port")
    
    def connect_network(self):
        """Connect to the specified network address."""
        protocol = self.protocol_combo.currentText()
        host = self.host_edit.text()
        try:
            port = int(self.network_port_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Port", "Please enter a valid port number")
            return
        
        if protocol == "TCP":
            if self.comm_manager.connect_tcp(host, port):
                self.connected = True
                self.network_connect_button.setText("Disconnect")
                self.data_thread.start()
                
                self.status_bar.showMessage(f"Connected to {host}:{port} via TCP")
            else:
                QMessageBox.critical(self, "Connection Failed", "Failed to connect to the TCP server")
        
        elif protocol == "UDP":
            if self.comm_manager.connect_udp(port, host, port):
                self.connected = True
                self.network_connect_button.setText("Disconnect")
                self.data_thread.start()
                
                self.status_bar.showMessage(f"Connected to {host}:{port} via UDP")
            else:
                QMessageBox.critical(self, "Connection Failed", "Failed to set up UDP connection")
    
    def disconnect(self):
        """Disconnect from the current connection."""
        self.data_thread.stop()
        self.comm_manager.close()
        
        self.connected = False
        self.connect_button.setText("Connect")
        self.connect_action.setText("Connect")
        self.network_connect_button.setText("Connect")
        
        self.status_bar.showMessage("Disconnected")
    
    def process_received_data(self, data):
        """Process received data and update visualizations"""
        try:
            print(f"DEBUG: process_received_data called with data: {data}")
            
            # Parse the data line
            values = data.strip().split(',')

            # Skip header lines or lines with incorrect number of columns
            if len(values) != len(self.data_headers) or values[0] == 'Timestamp':
                print(f"DEBUG: Skipping line - header or wrong column count (got {len(values)}, expected {len(self.data_headers)})")
                return

            # Create data dictionary
            data_dict = {}
            for i in range(len(self.data_headers)):
                try:
                    # Try converting to float, fallback to string if error
                    data_dict[self.data_headers[i]] = float(values[i].strip())
                except (ValueError, IndexError):
                    # Handle potential errors during conversion or if fewer values than headers
                    data_dict[self.data_headers[i]] = values[i].strip() if i < len(values) else None # Assign None if value missing

            print(f"DEBUG: Parsed data dictionary: {data_dict}")
            
            # Update data preview
            self.update_data_preview(data_dict)

            # Update real-time visualization (always add data, plot widget handles visibility)
            if self.real_time_plot:
                print("DEBUG: Updating real_time_plot with new data")
                self.real_time_plot.add_data_point(data_dict)
            else:
                print("DEBUG: real_time_plot is None, not updating")

            # Update status bar with latest timestamp
            if 'Timestamp' in data_dict:
                timestamp_val = data_dict['Timestamp']
                # Ensure timestamp_val is numeric before formatting
                if isinstance(timestamp_val, (int, float)):
                    self.status_bar.showMessage(f"Last update: {timestamp_val:.2f}")
                else:
                    self.status_bar.showMessage(f"Last update: {timestamp_val}")

        except Exception as e:
            print(f"Error processing data: {str(e)}")  # Debug print
            traceback.print_exc()  # Print full traceback
    
    def update_data_preview(self, data):
        """Update the data preview with information about the loaded data."""
        if isinstance(data, dict):
            preview_text = "Latest data point:\n"
            for key, value in data.items():
                preview_text += f"{key}: {value}\n"
            self.data_preview_label.setText(preview_text)
        elif data is not None and not data.empty:
            preview_text = f"Data loaded: {len(data)} rows, {len(data.columns)} columns\n"
            preview_text += f"Time range: {data['Timestamp'].min()} to {data['Timestamp'].max()}\n"
            
            # Add basic statistics
            try:
                stats = self.parser.get_data_statistics()
                for col in ["Alt", "Speed", "Temperature"]:
                    if col in stats:
                        preview_text += f"{col}: Min={stats[col]['min']:.2f}, Max={stats[col]['max']:.2f}, Mean={stats[col]['mean']:.2f}\n"
            except:
                pass
            
            self.data_preview_label.setText(preview_text)
        else:
            self.data_preview_label.setText("No data loaded")
    
    def save_data(self):
        """Save the current data to a CSV file."""
        if self.parser.data is None or self.parser.data.empty:
            QMessageBox.warning(self, "No Data", "No data to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.parser.data.to_csv(file_path, index=False)
                self.status_bar.showMessage(f"Data saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", f"Error: {str(e)}")
    
    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About TripleT Flight Console",
            "TripleT Flight Console\n\n"
            "A data visualization tool for model planes and rocketry flight data.\n\n"
            "This application can visualize both real-time and historical flight data."
        )
    
    def closeEvent(self, event):
        """Handle the window close event."""
        # Clean up resources
        if self.connected:
            self.disconnect()
        
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 