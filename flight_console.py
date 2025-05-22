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
from typing import Dict, List, Optional, Union, Iterator, Any, Tuple
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
# Ensure FlightDataParser is imported
from flight_data_parser import FlightDataParser


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

# RealTimePlotWidget class is now removed. Its functionality is merged into MainWindow.

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
            "Sensor Comparison" # Can compare KX134 vs ICM Accel
        ])
        
        self.column_select_combo = QComboBox()
        # Update selections based on LogData fields
        self.column_select_combo.addItems([
            "GPS Position & Speed", # Lat, Lon, Speed, Heading
            "Altitude",           # Altitude, AltitudeMSL
            "Barometer",          # Pressure, Temperature
            "KX134 Accelerometer",# KX134 X, Y, Z
            "ICM Accelerometer",  # ICM X, Y, Z
            "ICM Gyroscope",      # ICM X, Y, Z
            "ICM Magnetometer",   # ICM X, Y, Z
            "ICM Temperature",    # ICM Temp
            "GPS Status"          # FixType, Sats, pDOP, RTK
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
        """Get the columns selected for plotting based on LogData format."""
        selection = self.column_select_combo.currentText()
        
        if "GPS Position & Speed" in selection:
            return ["Latitude", "Longitude", "Speed", "Heading"]
        elif "Altitude" in selection:
            return ["Altitude", "AltitudeMSL", "raw_altitude", "calibrated_altitude"]
        elif "Barometer" in selection:
             return ["Pressure", "Temperature"]
        elif "KX134 Accelerometer" in selection:
            return ["KX134_AccelX", "KX134_AccelY", "KX134_AccelZ"]
        elif "ICM Accelerometer" in selection:
            return ["ICM_AccelX", "ICM_AccelY", "ICM_AccelZ"]
        elif "ICM Gyroscope" in selection:
            return ["ICM_GyroX", "ICM_GyroY", "ICM_GyroZ"]
        elif "ICM Magnetometer" in selection:
            return ["ICM_MagX", "ICM_MagY", "ICM_MagZ"]
        elif "ICM Temperature" in selection:
             return ["ICM_Temp"]
        elif "GPS Status" in selection:
            return ["FixType", "Sats", "pDOP", "RTK"]
        
        return []
    
    def create_plot(self):
        """Create the selected plot type using LogData fields."""
        if self.data is None or self.data.empty:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
            
        # Ensure Timestamp column exists for time series plots
        if 'Timestamp' not in self.data.columns:
            QMessageBox.warning(self, "Missing Data", "Loaded data must contain a 'Timestamp' column")
            return
            
        plot_type = self.plot_type_combo.currentText()
        
        # Clear the current figure
        self.figure.clear()
        
        try:
            if plot_type == "Time Series":
                columns = self.get_selected_columns()
                if not columns:
                    QMessageBox.warning(self, "No Columns", "No columns selected for Time Series plot")
                    return
                # Check if selected columns exist
                missing_cols = [col for col in columns if col not in self.data.columns]
                if missing_cols:
                     QMessageBox.warning(self, "Missing Data", f"Selected columns not found in data: {', '.join(missing_cols)}")
                     return
                     
                ax = self.figure.add_subplot(111)
                for column in columns:
                    # Convert timestamp from ms to seconds for plotting if needed (or assume it's done on load)
                    time_data = self.data['Timestamp'] / 1000.0 if self.data['Timestamp'].max() > 1e6 else self.data['Timestamp']
                    ax.plot(time_data, self.data[column], label=column)
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Value")
                ax.set_title("Time Series Plot")
                ax.grid(True)
                ax.legend()
                
            elif plot_type == "GPS Trajectory":
                # Use new Latitude/Longitude fields
                if all(col in self.data.columns for col in ['Latitude', 'Longitude']):
                    ax = self.figure.add_subplot(111)
                    ax.plot(self.data['Longitude'], self.data['Latitude'], 'b-', linewidth=1)
                    ax.plot(self.data['Longitude'].iloc[0], self.data['Latitude'].iloc[0], 'go', markersize=6, label='Start')
                    ax.plot(self.data['Longitude'].iloc[-1], self.data['Latitude'].iloc[-1], 'ro', markersize=6, label='End')
                    ax.set_xlabel('Longitude (deg)')
                    ax.set_ylabel('Latitude (deg)')
                    ax.set_title('GPS Trajectory')
                    ax.grid(True)
                    ax.legend()
                    ax.set_aspect('equal', adjustable='box') # Make aspect ratio equal
                else:
                    QMessageBox.warning(self, "Missing Data", "GPS data (Latitude, Longitude) is incomplete")
                
            elif plot_type == "3D Trajectory":
                # Use Latitude, Longitude, Altitude fields
                if all(col in self.data.columns for col in ['Latitude', 'Longitude', 'Altitude']):
                    ax = self.figure.add_subplot(111, projection='3d')
                    ax.plot(self.data['Longitude'], self.data['Latitude'], self.data['Altitude'], 'b-')
                    ax.set_xlabel('Longitude (deg)')
                    ax.set_ylabel('Latitude (deg)')
                    ax.set_zlabel('Altitude (m)')
                    ax.set_title('3D Flight Trajectory')
                else:
                    QMessageBox.warning(self, "Missing Data", "GPS data (Latitude, Longitude, Altitude) is incomplete")
            
            elif plot_type == "Sensor Comparison":
                # Compare KX134 Accel vs ICM Accel
                sensor_groups = {
                    "KX134 Accel": ["KX134_AccelX", "KX134_AccelY", "KX134_AccelZ"],
                    "ICM Accel": ["ICM_AccelX", "ICM_AccelY", "ICM_AccelZ"]
                }
                
                # Check if necessary columns exist
                required_cols = [col for group in sensor_groups.values() for col in group] + ['Timestamp']
                missing_cols = [col for col in required_cols if col not in self.data.columns]
                if missing_cols:
                    QMessageBox.warning(self, "Missing Data", f"Required columns for comparison not found: {', '.join(missing_cols)}")
                    return

                n_groups = len(sensor_groups)
                for i, (group_name, columns) in enumerate(sensor_groups.items()):
                    ax = self.figure.add_subplot(n_groups, 1, i+1)
                    time_data = self.data['Timestamp'] / 1000.0 if self.data['Timestamp'].max() > 1e6 else self.data['Timestamp']
                    for column in columns:
                        ax.plot(time_data, self.data[column], label=column.split('_')[-1]) # Use X, Y, Z as label
                    ax.set_title(f"{group_name}")
                    ax.set_ylabel('Acceleration (g)')
                    ax.grid(True)
                    ax.legend()
                
                # Add overall X label to the last subplot
                ax.set_xlabel("Time (s)")
                self.figure.tight_layout()
            
            self.canvas.draw()
            
        except Exception as e:
             QMessageBox.critical(self, "Plotting Error", f"Error creating plot: {str(e)}")
             traceback.print_exc()


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
        # self.parser is already initialized with FlightDataParser() which loads data_mapping.json by default.
        # The self.data_headers list is no longer the primary source of truth for parsing.
        # It's removed to avoid confusion. The mapping file is now the authority.
        
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

        # Dock Widget Settings
        self.setDockNestingEnabled(True)

        # Real-time plot data arrays (moved from RealTimePlotWidget)
        self.max_points = 1000
        self.timestamps = np.array([])
        self.latitudes = np.array([])
        self.longitudes = np.array([])
        self.altitudes = np.array([])
        self.altitudes_msl = np.array([])
        self.raw_altitudes = np.array([])
        self.calibrated_altitudes = np.array([])
        self.speeds = np.array([])
        self.headings = np.array([])
        self.pressures = np.array([])
        self.temperatures = np.array([]) # Baro temp
        self.kx134_accel = np.array([]).reshape(0, 3)
        self.icm_accel = np.array([]).reshape(0, 3)
        self.icm_gyro = np.array([]).reshape(0, 3)
        self.icm_temps = np.array([]) # ICM temp

        # Current status values (moved from RealTimePlotWidget)
        self.current_sats = 0
        self.current_fix_type = 0
        self.current_heading = 0.0 # For map marker
        self.current_rtk = 0
        
        # Create data receiver thread
        self.data_thread = DataReceiverThread(self.comm_manager)
        self.data_thread.data_received.connect(self.process_received_data)
        
        # Current connection status
        self.connected = False

    # --- Implementation of methods that now handle RealTimePlotWidget's logic ---
    def handle_new_data_point(self, data_point: Dict):
        """Handles a new data point, updates arrays, status, and plots."""
        try:
            # Extract timestamp (assuming it's in milliseconds)
            if 'Timestamp' in data_point and data_point['Timestamp'] is not None:
                timestamp = float(data_point['Timestamp']) / 1000.0 # Convert ms to s
                self.timestamps = np.append(self.timestamps, timestamp)
            else: # Fallback if no timestamp
                self.timestamps = np.append(self.timestamps, self.timestamps[-1] + 1 if len(self.timestamps) > 0 else 0)
            
            # Helper to append data or last known value or default
            def append_value(array, key, default_val=0.0):
                val = data_point.get(key)
                if val is not None:
                    return np.append(array, val), val
                elif len(array) > 0:
                    return np.append(array, array[-1]), array[-1]
                else:
                    return np.append(array, default_val), default_val

            self.altitudes, alt_ellip_val = append_value(self.altitudes, 'Altitude')
            self.altitudes_msl, alt_msl_val = append_value(self.altitudes_msl, 'AltitudeMSL')
            self.raw_altitudes, raw_alt_val = append_value(self.raw_altitudes, 'raw_altitude')
            self.calibrated_altitudes, calib_alt_val = append_value(self.calibrated_altitudes, 'calibrated_altitude')
            self.pressures, _ = append_value(self.pressures, 'Pressure')
            self.temperatures, _ = append_value(self.temperatures, 'Temperature') # Baro Temp
            self.speeds, speed_val = append_value(self.speeds, 'Speed')
            self.headings, heading_val = append_value(self.headings, 'Heading')
            if heading_val is not None: self.current_heading = heading_val # For map

            self.icm_temps, _ = append_value(self.icm_temps, 'ICM_Temp') # ICM Temp

            # For 3-axis data (accelerometers, gyros)
            def append_3axis_data(array_3axis, keys_xyz, default_val=np.zeros(3)):
                new_data = np.zeros(3)
                if all(k in data_point and data_point[k] is not None for k in keys_xyz):
                    new_data = np.array([data_point[keys_xyz[0]], data_point[keys_xyz[1]], data_point[keys_xyz[2]]])
                elif len(array_3axis) > 0:
                    new_data = array_3axis[-1]
                else:
                    new_data = default_val
                return np.vstack([array_3axis, new_data]) if len(array_3axis) > 0 else np.array([new_data]), new_data

            self.kx134_accel, _ = append_3axis_data(self.kx134_accel, ['KX134_AccelX', 'KX134_AccelY', 'KX134_AccelZ'])
            self.icm_accel, _ = append_3axis_data(self.icm_accel, ['ICM_AccelX', 'ICM_AccelY', 'ICM_AccelZ'])
            self.icm_gyro, _ = append_3axis_data(self.icm_gyro, ['ICM_GyroX', 'ICM_GyroY', 'ICM_GyroZ'])
            
            # GPS Position
            lat_val, lon_val = data_point.get('Latitude'), data_point.get('Longitude')
            if lat_val is not None and lon_val is not None:
                self.latitudes = np.append(self.latitudes, lat_val)
                self.longitudes = np.append(self.longitudes, lon_val)
                self.position_label.setText(f"Lat: {lat_val:.6f}, Lon: {lon_val:.6f}")
            elif len(self.latitudes) > 0 and len(self.longitudes) > 0: # Hold last position
                self.latitudes = np.append(self.latitudes, self.latitudes[-1])
                self.longitudes = np.append(self.longitudes, self.longitudes[-1])
            else: # Default if no data
                self.latitudes = np.append(self.latitudes, 0)
                self.longitudes = np.append(self.longitudes, 0)
                self.position_label.setText("Lat: N/A, Lon: N/A")

            # Update status labels
            self.altitude_status_label.setText(f"Alt MSL: {alt_msl_val:.2f}m | Ellip: {alt_ellip_val:.2f}m | Raw: {raw_alt_val:.2f}m | Calib: {calib_alt_val:.2f}m")
            self.speed_heading_label.setText(f"Speed: {speed_val:.1f} m/s, Heading: {self.current_heading:.1f} deg")
            
            fix_type = data_point.get('FixType')
            sats = data_point.get('Sats')
            rtk = data_point.get('RTK')
            if fix_type is not None: self.current_fix_type = int(fix_type)
            if sats is not None: self.current_sats = int(sats)
            if rtk is not None: self.current_rtk = int(rtk)
            self.gps_status_label.setText(f"GPS: Fix={self.current_fix_type}, Sats={self.current_sats}, RTK={self.current_rtk}")

            # Limit data arrays to max_points
            if len(self.timestamps) > self.max_points:
                self.timestamps = self.timestamps[-self.max_points:]
                self.latitudes = self.latitudes[-self.max_points:]
                self.longitudes = self.longitudes[-self.max_points:]
                self.altitudes = self.altitudes[-self.max_points:]
                self.altitudes_msl = self.altitudes_msl[-self.max_points:]
                self.raw_altitudes = self.raw_altitudes[-self.max_points:]
                self.calibrated_altitudes = self.calibrated_altitudes[-self.max_points:]
                self.speeds = self.speeds[-self.max_points:]
                self.headings = self.headings[-self.max_points:]
                self.pressures = self.pressures[-self.max_points:]
                self.temperatures = self.temperatures[-self.max_points:]
                self.kx134_accel = self.kx134_accel[-self.max_points:]
                self.icm_accel = self.icm_accel[-self.max_points:]
                self.icm_gyro = self.icm_gyro[-self.max_points:]
                self.icm_temps = self.icm_temps[-self.max_points:]

            self.refresh_realtime_plots()
            if len(self.latitudes) > 0 and len(self.longitudes) > 0:
                 self.update_map_content()

        except Exception as e:
            print(f"Error in MainWindow.handle_new_data_point: {e}")
            traceback.print_exc()

    def refresh_realtime_plots(self):
        """Updates all real-time plot curves with current data."""
        try:
            current_time = self.timestamps[-1] if len(self.timestamps) > 0 else 0
            time_range = (max(0, current_time - 30), current_time) # Show last 30 seconds

            if len(self.altitudes) > 0: self.altitude_curve.setData(self.timestamps, self.altitudes)
            if len(self.altitudes_msl) > 0: self.altitude_msl_curve.setData(self.timestamps, self.altitudes_msl)
            if len(self.raw_altitudes) > 0: self.raw_altitude_curve.setData(self.timestamps, self.raw_altitudes)
            if len(self.calibrated_altitudes) > 0: self.calibrated_altitude_curve.setData(self.timestamps, self.calibrated_altitudes)
            self.altitude_plot_widget.setXRange(*time_range, padding=0)

            if len(self.speeds) > 0: self.speed_curve.setData(self.timestamps, self.speeds)
            self.speed_plot_widget.setXRange(*time_range, padding=0)

            if len(self.pressures) > 0: self.pressure_curve.setData(self.timestamps, self.pressures)
            self.pressure_plot_widget.setXRange(*time_range, padding=0)

            if len(self.temperatures) > 0: self.baro_temp_curve.setData(self.timestamps, self.temperatures) # Baro
            if len(self.icm_temps) > 0: self.icm_temp_curve.setData(self.timestamps, self.icm_temps)       # ICM
            self.temperature_plot_widget.setXRange(*time_range, padding=0)

            if len(self.kx134_accel) > 0:
                self.kx134_accel_x_curve.setData(self.timestamps, self.kx134_accel[:, 0])
                self.kx134_accel_y_curve.setData(self.timestamps, self.kx134_accel[:, 1])
                self.kx134_accel_z_curve.setData(self.timestamps, self.kx134_accel[:, 2])
            self.kx134_plot_widget.setXRange(*time_range, padding=0)

            if len(self.icm_accel) > 0:
                self.icm_accel_x_curve.setData(self.timestamps, self.icm_accel[:, 0])
                self.icm_accel_y_curve.setData(self.timestamps, self.icm_accel[:, 1])
                self.icm_accel_z_curve.setData(self.timestamps, self.icm_accel[:, 2])
            self.icm_accel_plot_widget.setXRange(*time_range, padding=0)

            if len(self.icm_gyro) > 0:
                self.icm_gyro_x_curve.setData(self.timestamps, self.icm_gyro[:, 0])
                self.icm_gyro_y_curve.setData(self.timestamps, self.icm_gyro[:, 1])
                self.icm_gyro_z_curve.setData(self.timestamps, self.icm_gyro[:, 2])
            self.icm_gyro_plot_widget.setXRange(*time_range, padding=0)
            
            # Update GPS fallback plot if it's being used
            if hasattr(self, 'gps_fallback_plot_item') and self.gps_fallback_plot_item is not None:
                if len(self.latitudes) > 0 and len(self.longitudes) > 0:
                    self.gps_fallback_plot_item.setData(self.longitudes, self.latitudes)


        except Exception as e:
            print(f"Error in MainWindow.refresh_realtime_plots: {e}")
            traceback.print_exc()

    def update_map_content(self):
        """Updates the GPS map content using QWebEngineView or fallback."""
        try:
            if not hasattr(self, 'map_view_widget') or self.map_view_widget is None: return

            # If it's the QWebEngineView
            if isinstance(self.map_view_widget, QWebEngineView):
                if not hasattr(self.map_view_widget, 'setHtml'): return # Should exist, but defensive
                
                lat_list = self.latitudes.tolist()
                lon_list = self.longitudes.tolist()
                if not lat_list or not lon_list: return

                current_lat = lat_list[-1]
                current_lng = lon_list[-1]
                current_heading = self.current_heading

                path_points = [f"{{lat: {lat}, lng: {lon}}}" for lat, lon in zip(lat_list, lon_list)]
                path_str = ", ".join(path_points)

                html = f"""
                <!DOCTYPE html>
                <html><head><title>Real-time Flight Path</title>
                    <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&loading=async&libraries=marker"></script>
                    <style>html, body, #map {{ height: 100%; width: 100%; margin: 0; padding: 0; }}</style>
                </head><body><div id="map"></div><script>
                let map, currentPositionMarker, flightPath;
                const pathCoordinates = [{path_str}];
                async function initMap() {{
                    const {{ Map }} = await google.maps.importLibrary("maps");
                    const {{ AdvancedMarkerElement }} = await google.maps.importLibrary("marker");
                    map = new Map(document.getElementById('map'), {{ zoom: 18, center: {{lat: {current_lat}, lng: {current_lng}}}, mapId: 'SATELLITE_MAP_ID' }});
                    flightPath = new google.maps.Polyline({{ path: pathCoordinates, geodesic: true, strokeColor: '#FF0000', strokeOpacity: 1.0, strokeWeight: 3 }});
                    flightPath.setMap(map);
                    const iconElement = document.createElement('div');
                    iconElement.style.width = '20px'; iconElement.style.height = '20px'; iconElement.style.borderRadius = '50%';
                    iconElement.style.backgroundColor = '#00FF00'; iconElement.style.border = '1px solid #000000';
                    iconElement.style.transform = `rotate({current_heading}deg)`;
                    iconElement.innerHTML = '<div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid black;margin:4px auto 0 auto;"></div>';
                    currentPositionMarker = new AdvancedMarkerElement({{ position: {{lat: {current_lat}, lng: {current_lng}}}, map, title: 'Current Position', content: iconElement }});
                }}
                window.addEventListener('load', initMap);
                </script></body></html>"""
                self.map_view_widget.setHtml(html)
            # If it's the pg.PlotWidget fallback (already handled by refresh_realtime_plots)
            # else:
                # self.gps_fallback_plot_item.setData(self.longitudes, self.latitudes) # This is done in refresh_realtime_plots
                # print("DEBUG: map_view_widget is pg.PlotWidget, updated by refresh_realtime_plots")
                
        except Exception as e:
            print(f"Error in MainWindow.update_map_content: {e}")
            traceback.print_exc()

    def clear_realtime_plots(self):
        """Clears all real-time plot data, curves, and resets status labels."""
        try:
            self.timestamps = np.array([])
            self.latitudes = np.array([])
            self.longitudes = np.array([])
            self.altitudes = np.array([])
            self.altitudes_msl = np.array([])
            self.raw_altitudes = np.array([])
            self.calibrated_altitudes = np.array([])
            self.speeds = np.array([])
            self.headings = np.array([])
            self.pressures = np.array([])
            self.temperatures = np.array([]) # Baro
            self.kx134_accel = np.array([]).reshape(0, 3)
            self.icm_accel = np.array([]).reshape(0, 3)
            self.icm_gyro = np.array([]).reshape(0, 3)
            self.icm_temps = np.array([]) # ICM

            # Clear plot curves
            self.altitude_curve.clear()
            self.altitude_msl_curve.clear()
            self.raw_altitude_curve.clear()
            self.calibrated_altitude_curve.clear()
            self.speed_curve.clear()
            self.pressure_curve.clear()
            self.baro_temp_curve.clear()
            self.icm_temp_curve.clear()
            self.kx134_accel_x_curve.clear(); self.kx134_accel_y_curve.clear(); self.kx134_accel_z_curve.clear()
            self.icm_accel_x_curve.clear(); self.icm_accel_y_curve.clear(); self.icm_accel_z_curve.clear()
            self.icm_gyro_x_curve.clear(); self.icm_gyro_y_curve.clear(); self.icm_gyro_z_curve.clear()

            # Reset status labels
            self.current_sats = 0; self.current_fix_type = 0; self.current_heading = 0.0; self.current_rtk = 0
            self.gps_status_label.setText("GPS: Fix=0, Sats=0, RTK=0")
            self.position_label.setText("Lat: 0.000000, Lon: 0.000000")
            self.altitude_status_label.setText("Alt (MSL): 0.0m | Ellip: 0.0m | Raw: 0.0m | Calib: 0.0m")
            self.speed_heading_label.setText("Speed: 0.0 m/s, Heading: 0.0 deg")

            # Clear map (WebEngineView or fallback)
            if hasattr(self, 'map_view_widget'):
                if isinstance(self.map_view_widget, QWebEngineView):
                    self.map_view_widget.setHtml("") 
                elif hasattr(self, 'gps_fallback_plot_item') and self.gps_fallback_plot_item:
                    self.gps_fallback_plot_item.clear()
            
            # Call refresh to ensure plots are visually empty
            self.refresh_realtime_plots() 
            print("MainWindow: Real-time plots and data cleared.")
        except Exception as e:
            print(f"Error in MainWindow.clear_realtime_plots: {e}")
            traceback.print_exc()
    
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
        """Create the real-time visualization tab with dockable plot widgets."""
        # The "Real-time Visualization" tab itself is just a container.
        # QMainWindow handles the areas for dock widgets.
        realtime_tab_page = QWidget() # This widget is added to the QTabWidget
        # It doesn't need a complex layout itself if it's just a placeholder for dock areas.

        # 1. Create individual plot widgets and their curves (attributes of MainWindow)
        # Altitude Plot
        self.altitude_plot_widget = pg.PlotWidget(title="Altitude")
        self.altitude_plot_widget.setLabel('left', 'Altitude', units='m')
        self.altitude_plot_widget.setLabel('bottom', 'Time', units='s')
        self.altitude_plot_widget.addLegend()
        self.altitude_curve = self.altitude_plot_widget.plot(pen='y', name='Ellipsoid')
        self.altitude_msl_curve = self.altitude_plot_widget.plot(pen='g', name='MSL')
        self.raw_altitude_curve = self.altitude_plot_widget.plot(pen='c', name='Raw')
        self.calibrated_altitude_curve = self.altitude_plot_widget.plot(pen='m', name='Calibrated')

        # Speed Plot
        self.speed_plot_widget = pg.PlotWidget(title="Speed")
        self.speed_plot_widget.setLabel('left', 'Speed', units='m/s')
        self.speed_plot_widget.setLabel('bottom', 'Time', units='s')
        self.speed_curve = self.speed_plot_widget.plot(pen='m')

        # Pressure Plot
        self.pressure_plot_widget = pg.PlotWidget(title="Pressure")
        self.pressure_plot_widget.setLabel('left', 'Pressure', units='hPa')
        self.pressure_plot_widget.setLabel('bottom', 'Time', units='s')
        self.pressure_curve = self.pressure_plot_widget.plot(pen='c')

        # Temperature Plot
        self.temperature_plot_widget = pg.PlotWidget(title="Temperature")
        self.temperature_plot_widget.setLabel('left', 'Temperature', units='°C')
        self.temperature_plot_widget.setLabel('bottom', 'Time', units='s')
        self.temperature_plot_widget.addLegend()
        self.baro_temp_curve = self.temperature_plot_widget.plot(pen='r', name='Baro')
        self.icm_temp_curve = self.temperature_plot_widget.plot(pen=(255, 140, 0), name='ICM')
        
        # KX134 Accelerometer Plot
        self.kx134_plot_widget = pg.PlotWidget(title="KX134 Accelerometer")
        self.kx134_plot_widget.setLabel('left', 'Acceleration', units='g') # Added missing label
        self.kx134_plot_widget.setLabel('bottom', 'Time', units='s')    # Added missing label
        self.kx134_plot_widget.addLegend()
        self.kx134_accel_x_curve = self.kx134_plot_widget.plot(pen='r', name='X')
        self.kx134_accel_y_curve = self.kx134_plot_widget.plot(pen='g', name='Y')
        self.kx134_accel_z_curve = self.kx134_plot_widget.plot(pen='b', name='Z')

        # ICM Accelerometer Plot
        self.icm_accel_plot_widget = pg.PlotWidget(title="ICM Accelerometer")
        self.icm_accel_plot_widget.setLabel('left', 'Acceleration', units='g')
        self.icm_accel_plot_widget.setLabel('bottom', 'Time', units='s')
        self.icm_accel_plot_widget.addLegend()
        self.icm_accel_x_curve = self.icm_accel_plot_widget.plot(pen='r', name='X')
        self.icm_accel_y_curve = self.icm_accel_plot_widget.plot(pen='g', name='Y')
        self.icm_accel_z_curve = self.icm_accel_plot_widget.plot(pen='b', name='Z')

        # ICM Gyroscope Plot
        self.icm_gyro_plot_widget = pg.PlotWidget(title="ICM Gyroscope")
        self.icm_gyro_plot_widget.setLabel('left', 'Angular Rate', units='rad/s')
        self.icm_gyro_plot_widget.setLabel('bottom', 'Time', units='s')
        self.icm_gyro_plot_widget.addLegend()
        self.icm_gyro_x_curve = self.icm_gyro_plot_widget.plot(pen=(255, 0, 0, 150), name='X')
        self.icm_gyro_y_curve = self.icm_gyro_plot_widget.plot(pen=(0, 255, 0, 150), name='Y')
        self.icm_gyro_z_curve = self.icm_gyro_plot_widget.plot(pen=(0, 0, 255, 150), name='Z')

        # GPS Map View (attribute of MainWindow)
        try:
            from PyQt5.QtWebEngineWidgets import QWebEngineView
            self.map_view_widget = QWebEngineView()
        except (ImportError, Exception):
            print("DEBUG: QWebEngineView not available, falling back to GPS plot for docking")
            self.map_view_widget = pg.PlotWidget(title="GPS Position (Plot Fallback)")
            self.map_view_widget.setLabel('left', 'Latitude')
            self.map_view_widget.setLabel('bottom', 'Longitude')
            self.gps_fallback_plot_item = pg.PlotDataItem(pen=None, symbol='o', symbolSize=5, symbolBrush=('b'))
            self.map_view_widget.addItem(self.gps_fallback_plot_item)

        # Status Panel (attributes of MainWindow)
        status_panel_container = QGroupBox("Current Status") # This is the widget for the dock
        status_layout = QGridLayout(status_panel_container)
        self.gps_status_label = QLabel("GPS: Fix=0, Sats=0, RTK=0")
        self.position_label = QLabel("Lat: 0.000000, Lon: 0.000000")
        self.altitude_status_label = QLabel("Alt (MSL): 0.0m | Ellip: 0.0m | Raw: 0.0m | Calib: 0.0m")
        self.speed_heading_label = QLabel("Speed: 0.0 m/s, Heading: 0.0 deg")
        status_layout.addWidget(self.gps_status_label, 0, 0)
        status_layout.addWidget(self.position_label, 0, 1)
        status_layout.addWidget(self.altitude_status_label, 1, 0)
        status_layout.addWidget(self.speed_heading_label, 1, 1)

        # Controls Panel (attributes of MainWindow)
        controls_panel_container = QWidget() # This is the widget for the dock
        controls_layout = QHBoxLayout(controls_panel_container)
        controls_layout.addWidget(QLabel("Update Rate (ms):")) # Spinbox for display only, actual update is data-driven
        self.update_rate_spin = QSpinBox()
        self.update_rate_spin.setRange(10, 1000)
        self.update_rate_spin.setValue(100) 
        self.update_rate_spin.setSingleStep(10)
        controls_layout.addWidget(self.update_rate_spin)
        
        self.clear_realtime_button = QPushButton("Clear Plots")
        self.clear_realtime_button.clicked.connect(self.clear_realtime_plots) 
        controls_layout.addWidget(self.clear_realtime_button)
        controls_layout.addStretch()

        # 2. Create QDockWidgets for each plot/item
        dock_definitions = [
            ("Altitude", self.altitude_plot_widget, Qt.TopDockWidgetArea),
            ("Speed", self.speed_plot_widget, Qt.TopDockWidgetArea),
            ("Pressure", self.pressure_plot_widget, Qt.LeftDockWidgetArea),
            ("Temperature", self.temperature_plot_widget, Qt.LeftDockWidgetArea),
            ("KX134 Accel", self.kx134_plot_widget, Qt.RightDockWidgetArea),
            ("ICM Accel", self.icm_accel_plot_widget, Qt.RightDockWidgetArea),
            ("ICM Gyro", self.icm_gyro_plot_widget, Qt.BottomDockWidgetArea),
            ("GPS Map", self.map_view_widget, Qt.TopDockWidgetArea), # Often prominent
            ("Status Panel", status_panel_container, Qt.BottomDockWidgetArea),
            ("Controls Panel", controls_panel_container, Qt.BottomDockWidgetArea)
        ]
        
        self.plot_docks = {} # Store docks for potential later manipulation

        for title, widget_item, initial_area in dock_definitions:
            dock = QDockWidget(title, self)
            dock.setWidget(widget_item)
            dock.setAllowedAreas(Qt.AllDockWidgetAreas) # Allow docking anywhere
            dock.setObjectName(f"{title.replace(' ', '')}Dock") # For saving/restoring layout state
            self.addDockWidget(initial_area, dock)
            self.plot_docks[title] = dock
        
        # Example of tabifying some docks (optional, can be user-driven)
        # self.tabifyDockWidget(self.plot_docks["KX134 Accel"], self.plot_docks["ICM Accel"])
        # self.tabifyDockWidget(self.plot_docks["Pressure"], self.plot_docks["Temperature"])

        # Add the realtime_tab_page to the main QTabWidget
        self.tabs.addTab(realtime_tab_page, "Real-time Visualization")

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
    
    def process_received_data(self, data: str):
        """Process received data string and update visualizations using data mapping."""
        try:
            if not self.parser.mapping_config or not self.parser.mapping_config.get('column_mappings'):
                print("WARNING: No mapping configuration loaded in parser. Cannot process received data.")
                return

            # print(f"DEBUG: process_received_data called with data: {data}")
            
            csv_settings = self.parser.mapping_config['csv_settings']
            delimiter = csv_settings.get('delimiter', ',')
            values = data.strip().split(delimiter)
            
            # Check for header line based on first mapping item's csv_header (if applicable)
            # This is a heuristic for streaming: if the first value matches the first expected header name, skip.
            if csv_settings.get('has_header', True) and self.parser.mapping_config['column_mappings']:
                first_mapping_item = self.parser.mapping_config['column_mappings'][0]
                first_csv_header = first_mapping_item.get('csv_header')
                if first_csv_header and values and values[0].strip() == first_csv_header:
                    print(f"DEBUG: Skipping line - looks like a header line: {data}")
                    return

            data_dict = {}
            num_expected_columns = len(self.parser.mapping_config['column_mappings'])

            # Strategy for streaming: iterate through column_mappings.
            # If has_header is false, use csv_index. Otherwise, use order of column_mappings.
            for i, mapping_item in enumerate(self.parser.mapping_config['column_mappings']):
                internal_name = mapping_item['internal_name']
                target_type = mapping_item['type']
                default_value = mapping_item['default_value']
                raw_value_str = None # Initialize as string or None

                csv_index = mapping_item.get('csv_index')

                if not csv_settings.get('has_header', True) and csv_index is not None:
                    # Index-based mapping
                    if csv_index < len(values):
                        raw_value_str = values[csv_index].strip()
                    else:
                        # print(f"DEBUG: Index {csv_index} out of bounds for values list (len {len(values)}). Using default for {internal_name}.")
                        pass # Will use default_value
                else:
                    # Order-based mapping (for streaming when has_header is true but no header row per line, or if csv_index is missing)
                    if i < len(values):
                        raw_value_str = values[i].strip()
                    else:
                        # print(f"DEBUG: Index {i} out of bounds for values list (len {len(values)}). Using default for {internal_name}.")
                        pass # Will use default_value
                
                # If raw_value_str is empty or still None, it means it wasn't found or was empty.
                # _cast_value will handle None by returning default_value.
                # If raw_value_str is an empty string, _cast_value might also convert it to default_value depending on type.
                if raw_value_str == "": # Explicitly treat empty strings as candidates for default value
                    final_value_to_cast = None 
                else:
                    final_value_to_cast = raw_value_str

                data_dict[internal_name] = self.parser._cast_value(final_value_to_cast, target_type, default_value)

            # print(f"DEBUG: Parsed data dictionary: {data_dict}")
            
            # Update data preview
            self.update_data_preview(data_dict)

            # Call the new method in MainWindow to handle the data for real-time plots
            self.handle_new_data_point(data_dict)

            # Update status bar with latest timestamp
            if 'Timestamp' in data_dict and data_dict['Timestamp'] is not None:
                timestamp_val = data_dict['Timestamp']
                # Assuming timestamp is in milliseconds, convert to seconds for display
                self.status_bar.showMessage(f"Last update: {timestamp_val / 1000.0:.2f} s")
            else:
                 self.status_bar.showMessage(f"Last update: N/A")

        except Exception as e:
            print(f"Error processing data: {str(e)}")  # Debug print
            traceback.print_exc()  # Print full traceback
    
    def update_data_preview(self, data):
        """Update the data preview with information about the loaded data."""
        if isinstance(data, dict):
            # Handle real-time data preview (show latest point)
            preview_text = "Latest data point:\n"
            for key, value in data.items():
                 # Format floats for better readability
                 if key in ["Altitude", "AltitudeMSL"] and isinstance(value, float):
                      preview_text += f"{key}: {value:.5f}\n" # 5 decimal places for Alt
                 elif key in ["raw_altitude", "calibrated_altitude"] and isinstance(value, float):
                      preview_text += f"{key}: {value:.5f}\n" # 5 decimal places for new altitudes
                 elif isinstance(value, float):
                     preview_text += f"{key}: {value:.3f}\n" # Default 3 for others
                 else:
                     preview_text += f"{key}: {value}\n"
            self.data_preview_label.setText(preview_text)
            
        elif isinstance(data, pd.DataFrame) and not data.empty:
            # Handle loaded file data preview
            preview_text = f"Data loaded: {len(data)} rows, {len(data.columns)} columns\n"
            # Ensure 'Timestamp' (or its mapped equivalent) exists before trying to access it.
            # The internal name for timestamp is 'Timestamp' in the default mapping.
            if 'Timestamp' in data.columns:
                 # Assume ms, convert to seconds for range display
                 min_time_s = data['Timestamp'].min() / 1000.0 if data['Timestamp'].min() > 1e6 else data['Timestamp'].min()
                 max_time_s = data['Timestamp'].max() / 1000.0 if data['Timestamp'].max() > 1e6 else data['Timestamp'].max()
                 preview_text += f"Time range: {min_time_s:.2f} s to {max_time_s:.2f} s\n"
            
            # Add basic statistics using internal names from mapping
            # The internal names in data_mapping.json like "AltitudeMSL", "Speed", "Temperature", 
            # "Pressure", "raw_altitude", "calibrated_altitude" should align with these.
            try:
                stats_cols = ["AltitudeMSL", "Speed", "Temperature", "Pressure", "raw_altitude", "calibrated_altitude"]
                for col_internal_name in stats_cols:
                    if col_internal_name in data.columns and pd.api.types.is_numeric_dtype(data[col_internal_name]):
                        col_data = data[col_internal_name].dropna()
                        if not col_data.empty:
                             preview_text += f"{col_internal_name}: Min={col_data.min():.2f}, Max={col_data.max():.2f}, Mean={col_data.mean():.2f}\n"
            except Exception as e:
                print(f"Error calculating preview statistics: {e}")
                pass # Don't crash if stats fail
            
            self.data_preview_label.setText(preview_text)
        else:
            self.data_preview_label.setText("No data loaded or data is empty")
    
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