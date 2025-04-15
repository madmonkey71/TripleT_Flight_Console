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
                    print(f"DataReceiverThread received: {data[:100]}...")
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
        
        # For Google Maps
        self.maps_html = None
        
        # Create layout
        layout = QGridLayout(self)
        
        # Setup plot windows
        self.setup_windows()
        
    def setup_windows(self):
        """Set up all visualization windows and layouts."""
        try:
            # Create layout
            layout = QGridLayout()
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(10)
            
            # Create altitude plot
            self.altitude_plot = pg.PlotWidget(title="Altitude")
            self.altitude_plot.setLabel('left', 'Altitude', units='m')
            self.altitude_plot.setLabel('bottom', 'Time', units='s')
            self.altitude_plot.showGrid(x=True, y=True)
            self.altitude_curve = self.altitude_plot.plot(pen=pg.mkPen('b', width=2))
            
            # Create pressure plot
            self.pressure_plot = pg.PlotWidget(title="Pressure")
            self.pressure_plot.setLabel('left', 'Pressure', units='hPa')
            self.pressure_plot.setLabel('bottom', 'Time', units='s')
            self.pressure_plot.showGrid(x=True, y=True)
            self.pressure_curve = self.pressure_plot.plot(pen=pg.mkPen('r', width=2))
            
            # Create temperature plot
            self.temperature_plot = pg.PlotWidget(title="Temperature")
            self.temperature_plot.setLabel('left', 'Temperature', units='°C')
            self.temperature_plot.setLabel('bottom', 'Time', units='s')
            self.temperature_plot.showGrid(x=True, y=True)
            self.temperature_curve = self.temperature_plot.plot(pen=pg.mkPen('g', width=2))
            
            # Create motion sensors plot
            self.motion_plot = pg.PlotWidget(title="Motion Sensors")
            self.motion_plot.setLabel('left', 'Value')
            self.motion_plot.setLabel('bottom', 'Time', units='s')
            self.motion_plot.showGrid(x=True, y=True)
            self.motion_plot.addLegend()
            
            # Add curves for acceleration and gyro data
            self.accel_x_curve = self.motion_plot.plot(pen=pg.mkPen('r', width=2), name="Accel X")
            self.accel_y_curve = self.motion_plot.plot(pen=pg.mkPen('g', width=2), name="Accel Y")
            self.accel_z_curve = self.motion_plot.plot(pen=pg.mkPen('b', width=2), name="Accel Z")
            self.gyro_x_curve = self.motion_plot.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name="Gyro X")
            self.gyro_y_curve = self.motion_plot.plot(pen=pg.mkPen('g', width=2, style=Qt.DashLine), name="Gyro Y")
            self.gyro_z_curve = self.motion_plot.plot(pen=pg.mkPen('b', width=2, style=Qt.DashLine), name="Gyro Z")
            
            # Create 3D attitude view
            self.attitude_view = gl.GLViewWidget()
            self.attitude_view.setCameraPosition(distance=15, elevation=30, azimuth=45)
            self.attitude_view.setWindowTitle('3D Attitude')
            
            # Add coordinate grid
            grid = gl.GLGridItem()
            grid.setSize(10, 10, 1)
            grid.setSpacing(1, 1, 1)
            self.attitude_view.addItem(grid)
            
            # Add coordinate axes
            axis_length = 5
            x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), color=(1, 0, 0, 1), width=2)
            y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), color=(0, 1, 0, 1), width=2)
            z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), color=(0, 0, 1, 1), width=2)
            self.attitude_view.addItem(x_axis)
            self.attitude_view.addItem(y_axis)
            self.attitude_view.addItem(z_axis)
            
            # Create aircraft mesh and add to view
            mesh = self.create_aircraft_mesh()
            self.aircraft = gl.GLMeshItem(meshdata=mesh, smooth=True, color=(0.8, 0.8, 0.8, 1.0))
            self.attitude_view.addItem(self.aircraft)
            
            # Create map view for GPS
            try:
                from PyQt5.QtWebEngineWidgets import QWebEngineView
                self.map_view = QWebEngineView()
                self.map_view.setWindowTitle('GPS Trajectory')
                
                # HTML for a simple Google Maps view
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
                    <meta charset="utf-8">
                    <title>GPS Trajectory</title>
                    <style>
                        html, body, #map-canvas {
                            height: 100%;
                            margin: 0px;
                            padding: 0px
                        }
                    </style>
                    <script src="https://maps.googleapis.com/maps/api/js?v=3.exp"></script>
                    <script>
                        var map;
                        function initialize() {
                            var mapOptions = {
                                zoom: 15,
                                center: new google.maps.LatLng(0, 0),
                                mapTypeId: google.maps.MapTypeId.SATELLITE
                            };
                            map = new google.maps.Map(document.getElementById('map-canvas'), mapOptions);
                        }
                        google.maps.event.addDomListener(window, 'load', initialize);
                    </script>
                </head>
                <body>
                    <div id="map-canvas"></div>
                </body>
                </html>
                """
                self.map_view.setHtml(html)
            except ImportError:
                print("QWebEngineView not available, using placeholder for GPS view")
                self.map_view = QWidget()
                placeholder_layout = QVBoxLayout(self.map_view)
                placeholder_label = QLabel("GPS View (Web Engine not available)")
                placeholder_layout.addWidget(placeholder_label)
            
            # Add all widgets to layout
            layout.addWidget(self.altitude_plot, 0, 0)
            layout.addWidget(self.pressure_plot, 1, 0)
            layout.addWidget(self.temperature_plot, 2, 0)
            layout.addWidget(self.motion_plot, 3, 0)
            layout.addWidget(self.map_view, 0, 1)
            layout.addWidget(self.attitude_view, 1, 1, 3, 1)  # Span 3 rows
            
            # Set column stretch factors (make right column larger)
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 2)
            
            # Set layout to the widget
            self.setLayout(layout)
            
            # Set up timer for updates
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_plots)
            self.timer.start(50)  # Update at 20 Hz
            
        except Exception as e:
            print(f"Error setting up windows: {e}")
            traceback.print_exc()
        
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
        """Update all the plots with the current data."""
        try:
            # Only update if we have data
            if len(self.timestamps) == 0:
                return
            
            # Update 2D plots
            self.altitude_curve.setData(self.timestamps, self.altitudes)
            self.pressure_curve.setData(self.timestamps, self.pressures)
            self.temperature_curve.setData(self.timestamps, self.temperatures)
            
            # Update motion plots
            self.accel_x_curve.setData(self.timestamps, self.accelerations[:, 0])
            self.accel_y_curve.setData(self.timestamps, self.accelerations[:, 1])
            self.accel_z_curve.setData(self.timestamps, self.accelerations[:, 2])
            self.gyro_x_curve.setData(self.timestamps, self.gyros[:, 0])
            self.gyro_y_curve.setData(self.timestamps, self.gyros[:, 1])
            self.gyro_z_curve.setData(self.timestamps, self.gyros[:, 2])
            
            # Update 3D attitude if we have quaternion data
            if len(self.quaternions) > 0 and self.aircraft is not None:
                # Get the latest quaternion
                q = self.quaternions[-1]
                
                # Convert quaternion to rotation matrix
                R = self.quaternion_to_rotation_matrix(q)
                
                # Apply rotation to the aircraft model
                self.aircraft.setTransform(pg.Transform3D(R))
        except Exception as e:
            print(f"Error updating plots: {e}")

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        try:
            # Normalize the quaternion
            norm = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
            if norm < 1e-10:
                return np.identity(4)
            
            q = q / norm
            
            # Calculate rotation matrix
            xx, xy, xz = q[1]*q[1], q[1]*q[2], q[1]*q[3]
            yy, yz, zz = q[2]*q[2], q[2]*q[3], q[3]*q[3]
            wx, wy, wz = q[0]*q[1], q[0]*q[2], q[0]*q[3]
            
            rot_matrix = np.array([
                [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy), 0],
                [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx), 0],
                [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy), 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
            return rot_matrix
        except Exception as e:
            print(f"Error converting quaternion to rotation matrix: {e}")
            return np.identity(4)

    def add_data_point(self, data_point):
        """Add a new data point to all plots."""
        try:
            print(f"Adding data point to plots: {data_point}")
            
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
                self.altitudes = np.append(self.altitudes, float(data_point['Alt']))
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
            
            self.quaternions = np.vstack([self.quaternions, new_quat]) if len(self.quaternions) > 0 else np.array([new_quat])
            
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
            
            print(f"Data arrays updated: timestamps length = {len(self.timestamps)}")
            print(f"All plots updated successfully")
        except Exception as e:
            print(f"Error adding data point to plots: {e}")
            traceback.print_exc()
    
    def update_map(self):
        """Update the Google Maps view with current GPS data."""
        try:
            if hasattr(self, 'map_view') and hasattr(self.map_view, 'setHtml'):
                # Only update if we have at least one GPS point
                if len(self.latitudes) > 0 and len(self.longitudes) > 0:
                    # Get the last position
                    lat = self.latitudes[-1]
                    lng = self.longitudes[-1]
                    
                    # Create path string for all points
                    path_points = ""
                    for i in range(min(len(self.latitudes), len(self.longitudes))):
                        path_points += f"new google.maps.LatLng({self.latitudes[i]}, {self.longitudes[i]}),\n"
                    
                    # Create HTML with updated map center and path
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
                        <meta charset="utf-8">
                        <title>GPS Trajectory</title>
                        <style>
                            html, body, #map-canvas {{
                                height: 100%;
                                margin: 0px;
                                padding: 0px
                            }}
                        </style>
                        <script src="https://maps.googleapis.com/maps/api/js?v=3.exp"></script>
                        <script>
                            var map;
                            function initialize() {{
                                var mapOptions = {{
                                    zoom: 15,
                                    center: new google.maps.LatLng({lat}, {lng}),
                                    mapTypeId: google.maps.MapTypeId.SATELLITE
                                }};
                                map = new google.maps.Map(document.getElementById('map-canvas'), mapOptions);
                                
                                // Add flight path
                                var flightPath = new google.maps.Polyline({{
                                    path: [
                                        {path_points}
                                    ],
                                    geodesic: true,
                                    strokeColor: '#FF0000',
                                    strokeOpacity: 1.0,
                                    strokeWeight: 2
                                }});
                                
                                flightPath.setMap(map);
                                
                                // Add current position marker
                                var marker = new google.maps.Marker({{
                                    position: new google.maps.LatLng({lat}, {lng}),
                                    map: map,
                                    title: 'Current Position'
                                }});
                            }}
                            google.maps.event.addDomListener(window, 'load', initialize);
                        </script>
                    </head>
                    <body>
                        <div id="map-canvas"></div>
                    </body>
                    </html>
                    """
                    
                    # Update the map
                    self.map_view.setHtml(html)
        except Exception as e:
            print(f"Error updating map: {e}")
    
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
            "ICM_MagX", "ICM_MagY", "ICM_MagZ", "ICM_Temp", "ICM_QuatW", 
            "ICM_QuatX", "ICM_QuatY", "ICM_QuatZ"
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
        print(f"MainWindow received signal with data: {data[:100]}...") # Added confirmation print
        try:
            # Parse the data line
            values = data.strip().split(',')
            print(f"Received data values: {values}")  # Debug print

            # Skip header lines or lines with incorrect number of columns
            if len(values) != len(self.data_headers) or values[0] == 'Timestamp':
                print(f"Skipping line: Header or mismatched columns ({len(values)} vs {len(self.data_headers)}). Data: {data}")
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

            print(f"Parsed data_dict: {data_dict}") # Added print

            # Update data preview
            self.update_data_preview(data_dict)

            # Update real-time visualization (always add data, plot widget handles visibility)
            if self.real_time_plot:
                print("Calling real_time_plot.add_data_point...")  # Confirmation print
                self.real_time_plot.add_data_point(data_dict)

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
            print(f"Data that caused error: {data}")  # Debug print
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