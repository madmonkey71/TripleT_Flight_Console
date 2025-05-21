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
        
        # Initialize data arrays based on LogData structure
        self.timestamps = np.array([]) # ms
        self.latitudes = np.array([]) # degrees
        self.longitudes = np.array([]) # degrees
        self.altitudes = np.array([]) # meters (Ellipsoid)
        self.altitudes_msl = np.array([]) # meters (MSL)
        self.raw_altitudes = np.array([]) # meters (Raw)
        self.calibrated_altitudes = np.array([]) # meters (Calibrated)
        self.speeds = np.array([]) # m/s
        self.headings = np.array([]) # degrees
        self.pressures = np.array([]) # hPa
        self.temperatures = np.array([]) # deg C (Baro)
        self.kx134_accel = np.array([]).reshape(0, 3) # g
        self.icm_accel = np.array([]).reshape(0, 3) # g
        self.icm_gyro = np.array([]).reshape(0, 3) # rad/s
        self.icm_temps = np.array([]) # deg C (ICM)
                
        # For storing current values for text display (subset needed)
        self.current_sats = 0
        self.current_fix_type = 0
        self.current_heading = 0 # Keep for map marker rotation
        self.current_rtk = 0
        
        # For Google Maps (keep)
        self.maps_html = None
        
        # Setup plot windows
        self.setup_windows()
        
    def setup_windows(self):
        """Set up the visualization windows based on LogData."""
        # Create plots for altitude (Ellipsoid and MSL)
        self.altitude_plot = pg.PlotWidget()
        self.altitude_plot.setTitle("Altitude")
        self.altitude_plot.setLabel('left', 'Altitude', units='m')
        self.altitude_plot.setLabel('bottom', 'Time', units='s')
        self.altitude_curve = self.altitude_plot.plot(pen='y', name='Ellipsoid')
        self.altitude_msl_curve = self.altitude_plot.plot(pen='g', name='MSL')
        self.raw_altitude_curve = self.altitude_plot.plot(pen='c', name='Raw') # Cyan for Raw
        self.calibrated_altitude_curve = self.altitude_plot.plot(pen='m', name='Calibrated') # Magenta for Calibrated
        self.altitude_plot.addLegend()
        
        # Plot for Speed
        self.speed_plot = pg.PlotWidget()
        self.speed_plot.setTitle("Speed")
        self.speed_plot.setLabel('left', 'Speed', units='m/s')
        self.speed_plot.setLabel('bottom', 'Time', units='s')
        self.speed_curve = self.speed_plot.plot(pen='m')

        # Plot for Pressure
        self.pressure_plot = pg.PlotWidget()
        self.pressure_plot.setTitle("Pressure")
        self.pressure_plot.setLabel('left', 'Pressure', units='hPa')
        self.pressure_plot.setLabel('bottom', 'Time', units='s')
        self.pressure_curve = self.pressure_plot.plot(pen='c')
        
        # Plot for Temperature (Baro and ICM)
        self.temperature_plot = pg.PlotWidget()
        self.temperature_plot.setTitle("Temperature")
        self.temperature_plot.setLabel('left', 'Temperature', units='°C')
        self.temperature_plot.setLabel('bottom', 'Time', units='s')
        self.baro_temp_curve = self.temperature_plot.plot(pen='r', name='Baro')
        self.icm_temp_curve = self.temperature_plot.plot(pen=(255, 140, 0), name='ICM')
        self.temperature_plot.addLegend()
        
        # Create plot for KX134 accelerometer data
        self.kx134_plot = pg.PlotWidget()
        self.kx134_plot.setTitle("KX134 Accelerometer")
        self.kx134_plot.addLegend()
        self.kx134_accel_x_curve = self.kx134_plot.plot(pen='r', name='X')
        self.kx134_accel_y_curve = self.kx134_plot.plot(pen='g', name='Y')
        self.kx134_accel_z_curve = self.kx134_plot.plot(pen='b', name='Z')

        # Create plot for ICM Accelerometer data
        self.icm_accel_plot = pg.PlotWidget()
        self.icm_accel_plot.setTitle("ICM Accelerometer")
        self.icm_accel_plot.setLabel('left', 'Acceleration', units='g')
        self.icm_accel_plot.setLabel('bottom', 'Time', units='s')
        self.icm_accel_plot.addLegend()
        self.icm_accel_x_curve = self.icm_accel_plot.plot(pen='r', name='X')
        self.icm_accel_y_curve = self.icm_accel_plot.plot(pen='g', name='Y')
        self.icm_accel_z_curve = self.icm_accel_plot.plot(pen='b', name='Z')
        
        # Create a SEPARATE plot for ICM Gyroscope data
        self.icm_gyro_plot = pg.PlotWidget()
        self.icm_gyro_plot.setTitle("ICM Gyroscope")
        self.icm_gyro_plot.setLabel('left', 'Angular Rate', units='rad/s')
        self.icm_gyro_plot.setLabel('bottom', 'Time', units='s')
        self.icm_gyro_plot.addLegend()
        self.icm_gyro_x_curve = self.icm_gyro_plot.plot(pen=(255, 0, 0, 150), name='X')
        self.icm_gyro_y_curve = self.icm_gyro_plot.plot(pen=(0, 255, 0, 150), name='Y')
        self.icm_gyro_z_curve = self.icm_gyro_plot.plot(pen=(0, 0, 255, 150), name='Z')
        
        # Create main layout for the widget
        layout = QVBoxLayout()
        
        # Top row with GPS Map
        top_row = QHBoxLayout()
        
        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self.gps_plot = None # Initialize gps_plot attribute
        self.map_view = None # Initialize map_view attribute
        
        try:
            # Attempt to import and use QWebEngineView if available
            from PyQt5.QtWebEngineWidgets import QWebEngineView 
            self.map_view = QWebEngineView()
            print("DEBUG: QWebEngineView initialized successfully")
            map_layout.addWidget(self.map_view)
        except (ImportError, Exception) as e: 
            # Fallback to pg.PlotWidget if QWebEngineView unavailable or fails to init
            print(f"DEBUG: QWebEngineView not available or failed ({e}), falling back to GPS plot")
            self.map_view = None # Ensure map_view is None
            gps_plot_widget = pg.PlotWidget()
            gps_plot_widget.setTitle("GPS Position (Plot Fallback)")
            gps_plot_widget.setLabel('left', 'Latitude')
            gps_plot_widget.setLabel('bottom', 'Longitude')
            # Create a PlotDataItem for scatter plot points
            self.gps_plot = pg.PlotDataItem(pen=None, symbol='o', symbolSize=5, symbolBrush=('b')) 
            gps_plot_widget.addItem(self.gps_plot)
            map_layout.addWidget(gps_plot_widget)
        
        top_row.addWidget(map_container, 1)
        layout.addLayout(top_row)
        
        # Grid layout for sensor plots - Adjust grid to accommodate separate ICM plots
        grid = QGridLayout()
        grid.addWidget(self.altitude_plot, 0, 0)   # Row 0, Col 0
        grid.addWidget(self.speed_plot, 0, 1)      # Row 0, Col 1
        grid.addWidget(self.pressure_plot, 1, 0)   # Row 1, Col 0
        grid.addWidget(self.temperature_plot, 1, 1) # Row 1, Col 1
        grid.addWidget(self.kx134_plot, 2, 0)      # Row 2, Col 0
        grid.addWidget(self.icm_accel_plot, 2, 1)  # Row 2, Col 1 (Moved ICM Accel)
        grid.addWidget(self.icm_gyro_plot, 3, 0)   # Row 3, Col 0 (New ICM Gyro)
        # Span gyro plot across two columns? Optional.
        # grid.addWidget(self.icm_gyro_plot, 3, 0, 1, 2) # Span Row 3, Col 0 & 1 
        
        layout.addLayout(grid)
        
        # Status information panel
        status_panel = QGroupBox("Current Status")
        status_layout = QGridLayout(status_panel)
        
        # Create status labels relevant to LogData
        self.gps_status_label = QLabel("GPS: Fix=0, Sats=0, RTK=0")
        self.position_label = QLabel("Lat: 0.000000, Lon: 0.000000")
        self.altitude_status_label = QLabel("Alt (MSL): 0.0m | Ellip: 0.0m | Raw: 0.0m | Calib: 0.0m")
        self.speed_heading_label = QLabel("Speed: 0.0 m/s, Heading: 0.0 deg")

        # Add labels to status panel
        status_layout.addWidget(self.gps_status_label, 0, 0)
        status_layout.addWidget(self.position_label, 0, 1)
        status_layout.addWidget(self.altitude_status_label, 1, 0)
        status_layout.addWidget(self.speed_heading_label, 1, 1)
        
        # Add status panel to layout
        layout.addWidget(status_panel)
        
        # Set margins and spacing
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.setLayout(layout)
        
    def update_plots(self):
        """Update all plots with the latest data based on LogData format."""
        try:
            current_time = self.timestamps[-1] if len(self.timestamps) > 0 else 0
            time_range = (max(0, current_time - 30), current_time) # Show last 30 seconds
            
            # Update Altitude plot (Ellipsoid and MSL)
            if len(self.altitudes) > 0:
                self.altitude_curve.setData(self.timestamps, self.altitudes)
            if len(self.altitudes_msl) > 0:
                self.altitude_msl_curve.setData(self.timestamps, self.altitudes_msl)
            if len(self.raw_altitudes) > 0:
                self.raw_altitude_curve.setData(self.timestamps, self.raw_altitudes)
            if len(self.calibrated_altitudes) > 0:
                self.calibrated_altitude_curve.setData(self.timestamps, self.calibrated_altitudes)
            self.altitude_plot.setXRange(*time_range, padding=0)
                
            # Update Speed plot
            if len(self.speeds) > 0:
                self.speed_curve.setData(self.timestamps, self.speeds)
            self.speed_plot.setXRange(*time_range, padding=0)

            # Update Pressure plot
            if len(self.pressures) > 0:
                self.pressure_curve.setData(self.timestamps, self.pressures)
            self.pressure_plot.setXRange(*time_range, padding=0)
            
            # Update Temperature plot (Baro and ICM)
            if len(self.temperatures) > 0:
                self.baro_temp_curve.setData(self.timestamps, self.temperatures)
            if len(self.icm_temps) > 0:
                self.icm_temp_curve.setData(self.timestamps, self.icm_temps)
            self.temperature_plot.setXRange(*time_range, padding=0)

            # Update KX134 plot
            if len(self.kx134_accel) > 0:
                self.kx134_accel_x_curve.setData(self.timestamps, self.kx134_accel[:, 0])
                self.kx134_accel_y_curve.setData(self.timestamps, self.kx134_accel[:, 1])
                self.kx134_accel_z_curve.setData(self.timestamps, self.kx134_accel[:, 2])
            self.kx134_plot.setXRange(*time_range, padding=0)

            # Update ICM Accelerometer plot
            if len(self.icm_accel) > 0:
                self.icm_accel_x_curve.setData(self.timestamps, self.icm_accel[:, 0])
                self.icm_accel_y_curve.setData(self.timestamps, self.icm_accel[:, 1])
                self.icm_accel_z_curve.setData(self.timestamps, self.icm_accel[:, 2])
            self.icm_accel_plot.setXRange(*time_range, padding=0)
            
            # Update ICM Gyroscope plot
            if len(self.icm_gyro) > 0:
                self.icm_gyro_x_curve.setData(self.timestamps, self.icm_gyro[:, 0])
                self.icm_gyro_y_curve.setData(self.timestamps, self.icm_gyro[:, 1])
                self.icm_gyro_z_curve.setData(self.timestamps, self.icm_gyro[:, 2])
            self.icm_gyro_plot.setXRange(*time_range, padding=0)
            
            # Update GPS plot if not using WebEngine map
            if hasattr(self, 'gps_plot') and self.gps_plot is not None:
                 if len(self.latitudes) > 0 and len(self.longitudes) > 0:
                     self.gps_plot.clear()
                     self.gps_plot.plot(self.longitudes, self.latitudes, pen=None, symbol='o', symbolSize=5, symbolBrush=('b'))

            # Force update of all plot widgets - maybe not needed if setData triggers update
            # self.altitude_plot.update()
            # self.speed_plot.update()
            # self.pressure_plot.update()
            # self.temperature_plot.update()
            # self.kx134_plot.update()
            # self.icm_accel_plot.update()
            # self.icm_gyro_plot.update()
            
            # Process Qt events to keep UI responsive
            # self.app.processEvents() # Be careful with calling this here, might cause issues
            
        except Exception as e:
            print(f"Error updating plots: {e}")
            import traceback
            traceback.print_exc()

    def add_data_point(self, data_point):
        """Add a new data point (from LogData structure) to all plots."""
        try:
            # print(f"DEBUG: add_data_point called with data: {data_point}")
            
            # Extract timestamp (assuming it's in milliseconds)
            if 'Timestamp' in data_point and data_point['Timestamp'] is not None:
                # Convert ms to s for plotting timescale
                timestamp = float(data_point['Timestamp']) / 1000.0 
                self.timestamps = np.append(self.timestamps, timestamp)
            else:
                # Fallback if no timestamp
                if len(self.timestamps) == 0:
                    next_timestamp = 0
                else:
                    next_timestamp = self.timestamps[-1] + 1 # Assume 1s interval
                self.timestamps = np.append(self.timestamps, next_timestamp)
            
            # Extract altitude (Ellipsoid)
            alt_ellip_val = None
            if 'Altitude' in data_point and data_point['Altitude'] is not None:
                alt_ellip_val = data_point['Altitude']
                self.altitudes = np.append(self.altitudes, alt_ellip_val)
            elif len(self.altitudes) > 0:
                self.altitudes = np.append(self.altitudes, self.altitudes[-1])
                alt_ellip_val = self.altitudes[-1] # Keep last value for label
            else:
                self.altitudes = np.append(self.altitudes, 0)
                
            # Extract altitude (MSL)
            alt_msl_val = None
            if 'AltitudeMSL' in data_point and data_point['AltitudeMSL'] is not None:
                alt_msl_val = data_point['AltitudeMSL']
                self.altitudes_msl = np.append(self.altitudes_msl, alt_msl_val)
            elif len(self.altitudes_msl) > 0:
                self.altitudes_msl = np.append(self.altitudes_msl, self.altitudes_msl[-1])
                alt_msl_val = self.altitudes_msl[-1] # Keep last value for label
            else:
                self.altitudes_msl = np.append(self.altitudes_msl, 0)

            # Extract raw altitude
            raw_alt_val = None
            if 'raw_altitude' in data_point and data_point['raw_altitude'] is not None:
                raw_alt_val = data_point['raw_altitude']
                self.raw_altitudes = np.append(self.raw_altitudes, raw_alt_val)
            elif len(self.raw_altitudes) > 0:
                self.raw_altitudes = np.append(self.raw_altitudes, self.raw_altitudes[-1])
                raw_alt_val = self.raw_altitudes[-1]
            else:
                self.raw_altitudes = np.append(self.raw_altitudes, 0)

            # Extract calibrated altitude
            calib_alt_val = None
            if 'calibrated_altitude' in data_point and data_point['calibrated_altitude'] is not None:
                calib_alt_val = data_point['calibrated_altitude']
                self.calibrated_altitudes = np.append(self.calibrated_altitudes, calib_alt_val)
            elif len(self.calibrated_altitudes) > 0:
                self.calibrated_altitudes = np.append(self.calibrated_altitudes, self.calibrated_altitudes[-1])
                calib_alt_val = self.calibrated_altitudes[-1]
            else:
                self.calibrated_altitudes = np.append(self.calibrated_altitudes, 0)
                
            # Update Altitude Status Label (show both Ellipsoid and MSL)
            alt_msl_text = f"{alt_msl_val:.5f}m" if alt_msl_val is not None else "N/A"
            alt_ellip_text = f"{alt_ellip_val:.5f}m" if alt_ellip_val is not None else "N/A"
            raw_alt_text = f"{raw_alt_val:.5f}m" if raw_alt_val is not None else "N/A"
            calib_alt_text = f"{calib_alt_val:.5f}m" if calib_alt_val is not None else "N/A"
            if hasattr(self, 'altitude_status_label'):
                self.altitude_status_label.setText(f"Alt MSL: {alt_msl_text} | Ellip: {alt_ellip_text} | Raw: {raw_alt_text} | Calib: {calib_alt_text}")

            # Extract pressure
            if 'Pressure' in data_point and data_point['Pressure'] is not None:
                self.pressures = np.append(self.pressures, data_point['Pressure'])
            elif len(self.pressures) > 0:
                self.pressures = np.append(self.pressures, self.pressures[-1])
            else:
                self.pressures = np.append(self.pressures, 0)
            
            # Extract temperature (Baro)
            if 'Temperature' in data_point and data_point['Temperature'] is not None:
                self.temperatures = np.append(self.temperatures, data_point['Temperature'])
            elif len(self.temperatures) > 0:
                self.temperatures = np.append(self.temperatures, self.temperatures[-1])
            else:
                self.temperatures = np.append(self.temperatures, 0)
            
            # Extract GPS Position
            lat_val, lon_val = None, None
            if 'Latitude' in data_point and 'Longitude' in data_point and data_point['Latitude'] is not None and data_point['Longitude'] is not None:
                lat_val = data_point['Latitude']
                lon_val = data_point['Longitude']
                self.latitudes = np.append(self.latitudes, lat_val)
                self.longitudes = np.append(self.longitudes, lon_val)
                if hasattr(self, 'position_label'):
                    self.position_label.setText(f"Lat: {lat_val:.6f}, Lon: {lon_val:.6f}")
            elif len(self.latitudes) > 0 and len(self.longitudes) > 0:
                self.latitudes = np.append(self.latitudes, self.latitudes[-1])
                self.longitudes = np.append(self.longitudes, self.longitudes[-1])
            else:
                self.latitudes = np.append(self.latitudes, 0)
                self.longitudes = np.append(self.longitudes, 0)
                if hasattr(self, 'position_label'):
                    self.position_label.setText("Lat: N/A, Lon: N/A")

            # Extract Speed & Heading
            speed_val, heading_val = None, None
            if 'Speed' in data_point and data_point['Speed'] is not None:
                speed_val = data_point['Speed']
                self.speeds = np.append(self.speeds, speed_val)
            elif len(self.speeds) > 0:
                 self.speeds = np.append(self.speeds, self.speeds[-1])
            else:
                 self.speeds = np.append(self.speeds, 0)
                 
            if 'Heading' in data_point and data_point['Heading'] is not None:
                heading_val = data_point['Heading']
                self.headings = np.append(self.headings, heading_val)
                self.current_heading = heading_val # Update for map marker
            elif len(self.headings) > 0:
                 self.headings = np.append(self.headings, self.headings[-1])
            else:
                 self.headings = np.append(self.headings, 0)
                 self.current_heading = 0
                 
            # Update Speed/Heading Label
            speed_text = f"{speed_val:.1f} m/s" if speed_val is not None else "N/A"
            heading_text = f"{heading_val:.1f} deg" if heading_val is not None else "N/A"
            if hasattr(self, 'speed_heading_label'):
                self.speed_heading_label.setText(f"Speed: {speed_text}, Heading: {heading_text}")
                 
            # Extract KX134 acceleration
            new_kx_accel = np.zeros(3)
            if 'KX134_AccelX' in data_point and 'KX134_AccelY' in data_point and 'KX134_AccelZ' in data_point and \
               data_point['KX134_AccelX'] is not None and data_point['KX134_AccelY'] is not None and data_point['KX134_AccelZ'] is not None:
                new_kx_accel[0] = data_point['KX134_AccelX']
                new_kx_accel[1] = data_point['KX134_AccelY']
                new_kx_accel[2] = data_point['KX134_AccelZ']
            elif len(self.kx134_accel) > 0:
                new_kx_accel = self.kx134_accel[-1]
            
            self.kx134_accel = np.vstack([self.kx134_accel, new_kx_accel]) if len(self.kx134_accel) > 0 else np.array([new_kx_accel])
            
            # Extract ICM acceleration
            new_icm_accel = np.zeros(3)
            if 'ICM_AccelX' in data_point and 'ICM_AccelY' in data_point and 'ICM_AccelZ' in data_point and \
               data_point['ICM_AccelX'] is not None and data_point['ICM_AccelY'] is not None and data_point['ICM_AccelZ'] is not None:
                new_icm_accel[0] = data_point['ICM_AccelX']
                new_icm_accel[1] = data_point['ICM_AccelY']
                new_icm_accel[2] = data_point['ICM_AccelZ']
            elif len(self.icm_accel) > 0:
                new_icm_accel = self.icm_accel[-1]
            
            self.icm_accel = np.vstack([self.icm_accel, new_icm_accel]) if len(self.icm_accel) > 0 else np.array([new_icm_accel])

            # Extract ICM gyro
            new_icm_gyro = np.zeros(3)
            if 'ICM_GyroX' in data_point and 'ICM_GyroY' in data_point and 'ICM_GyroZ' in data_point and \
               data_point['ICM_GyroX'] is not None and data_point['ICM_GyroY'] is not None and data_point['ICM_GyroZ'] is not None:
                new_icm_gyro[0] = data_point['ICM_GyroX']
                new_icm_gyro[1] = data_point['ICM_GyroY']
                new_icm_gyro[2] = data_point['ICM_GyroZ']
            elif len(self.icm_gyro) > 0:
                new_icm_gyro = self.icm_gyro[-1]
            
            self.icm_gyro = np.vstack([self.icm_gyro, new_icm_gyro]) if len(self.icm_gyro) > 0 else np.array([new_icm_gyro])

            # Extract ICM temperature
            if 'ICM_Temp' in data_point and data_point['ICM_Temp'] is not None:
                self.icm_temps = np.append(self.icm_temps, data_point['ICM_Temp'])
            elif len(self.icm_temps) > 0:
                self.icm_temps = np.append(self.icm_temps, self.icm_temps[-1])
            else:
                self.icm_temps = np.append(self.icm_temps, 0)

            # Update GPS Status Label
            fix_type = data_point.get('FixType', None)
            sats = data_point.get('Sats', None)
            rtk = data_point.get('RTK', None)
            if fix_type is not None: self.current_fix_type = int(fix_type)
            if sats is not None: self.current_sats = int(sats)
            if rtk is not None: self.current_rtk = int(rtk)
            if hasattr(self, 'gps_status_label'):
                self.gps_status_label.setText(f"GPS: Fix={self.current_fix_type}, Sats={self.current_sats}, RTK={self.current_rtk}")

            # --- Removed processing for obsolete fields --- 
            # (Quaternions, FlightState, AGL, MaxAltitude, Velocities, Battery2, CPU Temp, etc.)

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
            
            # Update Google Maps if we have GPS data
            if len(self.latitudes) > 0 and len(self.longitudes) > 0:
                self.update_map()
            
            # Explicitly update the plots
            self.update_plots()
            
        except Exception as e:
            print(f"Error adding data point to plots: {e}")
            traceback.print_exc()
    
    def update_map(self):
        """Update the Google Maps view with current GPS data using modern API calls."""
        try:
            # print("DEBUG: update_map called, map_view exists:", hasattr(self, 'map_view') and self.map_view is not None, 
            #           "has setHtml:", hasattr(self, 'map_view') and self.map_view is not None and hasattr(self.map_view, 'setHtml'))
            
            if not hasattr(self, 'map_view') or self.map_view is None or not hasattr(self.map_view, 'setHtml'):
                # print("DEBUG: map_view not available or lacks setHtml")
                return
            
            # Convert numpy arrays to lists for JS compatibility if needed
            lat_list = self.latitudes.tolist() if isinstance(self.latitudes, np.ndarray) else list(self.latitudes)
            lon_list = self.longitudes.tolist() if isinstance(self.longitudes, np.ndarray) else list(self.longitudes)
            
            if not lat_list or not lon_list:
                # print("DEBUG: No GPS data yet")
                return
            
            # print(f"GPS data points: {len(lat_list)}, Last lat/long: {lat_list[-1]}, {lon_list[-1]}")
            
            # Get the current location (last point)
            current_lat = lat_list[-1]
            current_lng = lon_list[-1]
            current_heading = self.current_heading if hasattr(self, 'current_heading') and self.current_heading is not None else 0
            
            # print(f"Updating map with center at {current_lat}, {current_lng}")
            
            # Create path string with all points
            path_points = []
            for i in range(min(len(lat_list), len(lon_list))):
                path_points.append(f"{{lat: {lat_list[i]}, lng: {lon_list[i]}}}")
            
            path_str = ", ".join(path_points)
            
            # Generate HTML content with the map using async loading and AdvancedMarkerElement
            # NOTE: Replace YOUR_API_KEY with your actual Google Maps API Key
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Real-time Flight Path</title>
                <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&loading=async&libraries=marker"></script> 
                <style>
                    html, body, #map {{ height: 100%; width: 100%; margin: 0; padding: 0; }}
                </style>
            </head>
            <body>
                <div id="map"></div>
                <script>
                    let map;
                    let currentPositionMarker = null;
                    let flightPath = null;
                    const pathCoordinates = [{path_str}]; // Store path coords

                    async function initMap() {{
                        const {{ Map }} = await google.maps.importLibrary("maps");
                        const {{ AdvancedMarkerElement }} = await google.maps.importLibrary("marker");

                        map = new Map(document.getElementById('map'), {{
                            zoom: 18,
                            center: {{lat: {current_lat}, lng: {current_lng}}},
                            mapId: 'SATELLITE_MAP_ID' // Use mapId for satellite view with Advanced Markers
                        }});

                        // Create the flight path polyline
                        flightPath = new google.maps.Polyline({{
                            path: pathCoordinates,
                            geodesic: true,
                            strokeColor: '#FF0000',
                            strokeOpacity: 1.0,
                            strokeWeight: 3
                        }});
                        flightPath.setMap(map);

                        // Create the current position marker (AdvancedMarkerElement)
                        // Create a DOM element for the icon
                        const iconElement = document.createElement('div');
                        iconElement.style.width = '20px';
                        iconElement.style.height = '20px';
                        iconElement.style.borderRadius = '50%';
                        iconElement.style.backgroundColor = '#00FF00';
                        iconElement.style.border = '1px solid #000000';
                        iconElement.style.transform = `rotate({current_heading}deg)`; // Apply heading rotation
                        // Simple triangle for direction (could be more complex SVG)
                        iconElement.innerHTML = '<div style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid black; margin: 4px auto 0 auto;"></div>'; 

                        currentPositionMarker = new AdvancedMarkerElement({{
                            position: {{lat: {current_lat}, lng: {current_lng}}},
                            map: map,
                            title: 'Current Position',
                            content: iconElement // Use the custom DOM element
                        }});

                        // Add distance markers (using simple circle markers for now)
                        // Advanced Markers don't directly support SymbolPath, need DOM/SVG for custom icons
                        for (let i = 10; i < pathCoordinates.length; i += 10) {{
                            const pointMarkerElement = document.createElement('div');
                            pointMarkerElement.style.width = '8px';
                            pointMarkerElement.style.height = '8px';
                            pointMarkerElement.style.borderRadius = '50%';
                            pointMarkerElement.style.backgroundColor = '#FFFF00';
                            pointMarkerElement.style.border = '1px solid #000000';

                            new AdvancedMarkerElement({{
                                position: pathCoordinates[i],
                                map: map,
                                content: pointMarkerElement
                            }});
                        }}
                    }}

                    // Use standard event listener
                    window.addEventListener('load', initMap);
                </script>
            </body>
            </html>
            """
            
            # Update the HTML content in the QWebEngineView
            self.map_view.setHtml(html)
            # print("DEBUG: Map HTML updated with Advanced Markers")
        except Exception as e:
            print(f"DEBUG: Error updating map: {e}")
            traceback.print_exc()
    
    def clear_plots(self):
        """Clear all plot data relevant to LogData format."""
        try:
            # Reset data arrays
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
            self.temperatures = np.array([])
            self.kx134_accel = np.array([]).reshape(0, 3)
            self.icm_accel = np.array([]).reshape(0, 3)
            self.icm_gyro = np.array([]).reshape(0, 3)
            self.icm_temps = np.array([])
            
            # Reset relevant current values
            self.current_sats = 0
            self.current_fix_type = 0
            self.current_heading = 0
            self.current_rtk = 0
            
            # Reset labels
            self.gps_status_label.setText("GPS: Fix=0, Sats=0, RTK=0")
            self.position_label.setText("Lat: 0.000000, Lon: 0.000000")
            self.altitude_status_label.setText("Alt (MSL): 0.0m | Ellip: 0.0m | Raw: 0.0m | Calib: 0.0m")
            self.speed_heading_label.setText("Speed: 0.0 m/s, Heading: 0.0 deg")
            
            # Clear plot curves by updating with empty data
            self.update_plots()
            
            # Clear GPS plot if it exists
            if hasattr(self, 'gps_plot') and self.gps_plot is not None:
                self.gps_plot.clear()
            
            # Clear map (optional, might reload blank map)
            if hasattr(self, 'map_view') and self.map_view is not None:
                 self.map_view.setHtml("") # Clear map content

            print("Real-time plots cleared")
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
        
        # Initialize data headers based on LogData struct
        self.data_headers = [
            "seqNum", "Timestamp", "FixType", "Sats", "Latitude", "Longitude", 
            "Altitude", "AltitudeMSL", "raw_altitude", "calibrated_altitude", "Speed", "Heading", "pDOP", "RTK", 
            "Pressure", "Temperature", 
            "KX134_AccelX", "KX134_AccelY", "KX134_AccelZ", 
            "ICM_AccelX", "ICM_AccelY", "ICM_AccelZ", 
            "ICM_GyroX", "ICM_GyroY", "ICM_GyroZ", 
            "ICM_MagX", "ICM_MagY", "ICM_MagZ", 
            "ICM_Temp"
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
            # print(f"DEBUG: process_received_data called with data: {data}")
            
            # Parse the data line
            values = data.strip().split(',')
            expected_fields = len(self.data_headers)

            # Basic check for minimum length and header line (using seqNum now)
            if len(values) < expected_fields or values[0].strip().lower() == 'seqnum':
                print(f"DEBUG: Skipping line - header or insufficient data (got {len(values)}, need {expected_fields})")
                return
            
            # If we have more values than headers, log a warning but continue
            if len(values) > expected_fields:
                print(f"WARNING: More data values ({len(values)}) than expected headers ({expected_fields}). Extra values will be ignored.")
            elif len(values) < expected_fields:
                 print(f"WARNING: Fewer data values ({len(values)}) than expected headers ({expected_fields}). Missing values will be None.")

            # Create data dictionary with conversions
            data_dict = {}
            for i, header in enumerate(self.data_headers):
                try:
                    if i >= len(values):
                        data_dict[header] = None # Handle missing values
                        continue
                        
                    value_str = values[i].strip()
                    if value_str == '': # Handle empty strings
                         data_dict[header] = None
                         continue

                    # Apply conversions based on the header name derived from LogData struct
                    if header == "seqNum":
                        data_dict[header] = int(value_str)
                    elif header == "Timestamp":
                        data_dict[header] = int(value_str) # Assuming milliseconds
                    elif header == "FixType":
                        data_dict[header] = int(value_str)
                    elif header == "Sats":
                        data_dict[header] = int(value_str)
                    elif header == "Latitude":
                        data_dict[header] = float(value_str)
                    elif header == "Longitude":
                        data_dict[header] = float(value_str)
                    elif header == "Altitude":
                        data_dict[header] = float(value_str)
                    elif header == "AltitudeMSL":
                        data_dict[header] = float(value_str)
                    elif header == "raw_altitude":
                        data_dict[header] = float(value_str)
                    elif header == "calibrated_altitude":
                        data_dict[header] = float(value_str)
                    elif header == "Speed":
                        data_dict[header] = float(value_str)
                    elif header == "Heading":
                        data_dict[header] = float(value_str)
                    elif header == "pDOP":
                        data_dict[header] = float(value_str) / 100.0 # unitless * 100 to unitless
                    elif header == "RTK":
                        data_dict[header] = int(value_str)
                    elif header == "Pressure":
                        data_dict[header] = float(value_str)
                    elif header == "Temperature":
                        data_dict[header] = float(value_str)
                    elif header in ["KX134_AccelX", "KX134_AccelY", "KX134_AccelZ",
                                     "ICM_AccelX", "ICM_AccelY", "ICM_AccelZ",
                                     "ICM_GyroX", "ICM_GyroY", "ICM_GyroZ",
                                     "ICM_MagX", "ICM_MagY", "ICM_MagZ",
                                     "ICM_Temp"]:
                        data_dict[header] = float(value_str)
                    else:
                        # Default: treat as string if no specific conversion
                        data_dict[header] = value_str
                        
                except ValueError:
                    print(f"WARNING: Could not convert value '{value_str}' for header '{header}'. Setting to None.")
                    data_dict[header] = None # Assign None if conversion fails
                except IndexError:
                    print(f"WARNING: Missing value for header '{header}'. Setting to None.")
                    data_dict[header] = None # Assign None if value is missing

            # print(f"DEBUG: Parsed data dictionary: {data_dict}")
            
            # Update data preview
            self.update_data_preview(data_dict)

            # Update real-time visualization (always add data, plot widget handles visibility)
            if self.real_time_plot:
                # print("DEBUG: Updating real_time_plot with new data")
                self.real_time_plot.add_data_point(data_dict)
            # else:
                # print("DEBUG: real_time_plot is None, not updating")

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
            if 'Timestamp' in data.columns:
                 # Assume ms, convert to seconds for range display
                 min_time_s = data['Timestamp'].min() / 1000.0
                 max_time_s = data['Timestamp'].max() / 1000.0
                 preview_text += f"Time range: {min_time_s:.2f} s to {max_time_s:.2f} s\n"
            
            # Add basic statistics using new field names
            try:
                # Calculate statistics directly if parser isn't used or doesn't have them
                stats_cols = ["AltitudeMSL", "Speed", "Temperature", "Pressure", "raw_altitude", "calibrated_altitude"]
                for col in stats_cols:
                    if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                        col_data = data[col].dropna()
                        if not col_data.empty:
                             preview_text += f"{col}: Min={col_data.min():.2f}, Max={col_data.max():.2f}, Mean={col_data.mean():.2f}\n"
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