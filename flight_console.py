"""
Flight Console Application

This is the main application module for the TripleT Flight Console.
It provides a GUI for visualizing flight data from model planes and rocketry,
supporting both real-time data streaming and historical data analysis.
"""

import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTabWidget, QPushButton, QLabel, QComboBox, QLineEdit, 
                            QFileDialog, QGroupBox, QGridLayout, QCheckBox, QSplitter,
                            QMessageBox, QSpinBox, QStatusBar, QAction, QMenu, QToolBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QIcon, QFont

import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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
        while self.running:
            data = self.comm_manager.get_queued_data(timeout=0.1)
            if data:
                self.data_received.emit(data)
    
    def stop(self):
        self.running = False
        self.wait()


class RealTimePlotWidget(QWidget):
    """Widget for displaying real-time plots."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualizer = DataVisualizer()
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create plot selection controls
        control_layout = QHBoxLayout()
        
        self.plot_selector = QComboBox()
        self.plot_selector.addItems([
            "GPS (Altitude, Speed)",
            "Acceleration (KX134)",
            "Acceleration (ICM)",
            "Gyroscope (ICM)",
            "Magnetometer (ICM)",
            "Environmental"
        ])
        
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(10, 1000)
        self.window_size_spin.setValue(100)
        self.window_size_spin.setSingleStep(10)
        
        control_layout.addWidget(QLabel("Plot:"))
        control_layout.addWidget(self.plot_selector)
        control_layout.addWidget(QLabel("Window Size:"))
        control_layout.addWidget(self.window_size_spin)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Create plot widget
        self.plot_widget, self.plot_data = self.visualizer.create_real_time_plot_widget(self)
        layout.addWidget(self.plot_widget)
        
        # Current column group being plotted
        self.current_columns = ["Alt", "Speed"]
        self.plot_selector.currentIndexChanged.connect(self.update_plot_selection)
        
        # Initialize the plot
        self.update_plot_selection(0)
    
    def update_plot_selection(self, index):
        """Update the plot based on the selected plot type."""
        selection = self.plot_selector.currentText()
        
        if "GPS" in selection:
            self.current_columns = ["Alt", "Speed"]
            self.plot_widget.setTitle("GPS Data")
        elif "Acceleration (KX134)" in selection:
            self.current_columns = ["KX134_AccelX", "KX134_AccelY", "KX134_AccelZ"]
            self.plot_widget.setTitle("KX134 Accelerometer")
        elif "Acceleration (ICM)" in selection:
            self.current_columns = ["ICM_AccelX", "ICM_AccelY", "ICM_AccelZ"]
            self.plot_widget.setTitle("ICM Accelerometer")
        elif "Gyroscope" in selection:
            self.current_columns = ["ICM_GyroX", "ICM_GyroY", "ICM_GyroZ"]
            self.plot_widget.setTitle("ICM Gyroscope")
        elif "Magnetometer" in selection:
            self.current_columns = ["ICM_MagX", "ICM_MagY", "ICM_MagZ"]
            self.plot_widget.setTitle("ICM Magnetometer")
        elif "Environmental" in selection:
            self.current_columns = ["Pressure", "Temperature"]
            self.plot_widget.setTitle("Environmental Data")
        
        # Reset plot data with new columns
        self.plot_data = {}
    
    def add_data_point(self, data_point):
        """Add a new data point to the real-time plot."""
        self.plot_data = self.visualizer.add_data_to_real_time_plot(
            self.plot_widget, 
            self.plot_data, 
            data_point, 
            self.current_columns, 
            self.window_size_spin.value()
        )


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
    
    def create_plot(self):
        """Create and display the selected plot type."""
        if self.data is None or self.data.empty:
            QMessageBox.warning(self, "No Data", "Please load data first")
            return
        
        plot_type = self.plot_type_combo.currentText()
        self.figure.clear()
        
        if plot_type == "Time Series":
            columns = self.get_selected_columns()
            fig = self.visualizer.create_time_series_plot(self.data, columns)
            
            # Transfer from matplotlib figure to our canvas
            for ax in fig.get_axes():
                self.figure.add_axes(ax)
        
        elif plot_type == "GPS Trajectory":
            fig = self.visualizer.create_gps_trajectory_plot(self.data)
            
            # Transfer from matplotlib figure to our canvas
            for ax in fig.get_axes():
                self.figure.add_axes(ax)
        
        elif plot_type == "3D Trajectory":
            # This uses Plotly, which we can't directly embed in matplotlib
            # So we'll create a simple 3D plot in matplotlib instead
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
            
            fig = self.visualizer.create_sensor_comparison_plot(self.data, sensor_groups)
            
            # Transfer from matplotlib figure to our canvas
            for ax in fig.get_axes():
                self.figure.add_axes(ax)
        
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
        """Create the Real-time Visualization tab."""
        realtime_tab = QWidget()
        layout = QVBoxLayout(realtime_tab)
        
        # Create real-time plot widget
        self.realtime_plot = RealTimePlotWidget()
        layout.addWidget(self.realtime_plot)
        
        # Add controls
        control_layout = QHBoxLayout()
        
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.setEnabled(False)  # Disabled until connected
        self.start_stop_button.clicked.connect(self.toggle_realtime)
        
        control_layout.addWidget(self.start_stop_button)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # Add tab
        self.tabs.addTab(realtime_tab, "Real-time Visualization")
    
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
            
            # Enable real-time simulation if desired
            self.start_stop_button.setEnabled(True)
            
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
            self.start_stop_button.setEnabled(True)
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
                self.start_stop_button.setEnabled(True)
                self.data_thread.start()
                
                self.status_bar.showMessage(f"Connected to {host}:{port} via TCP")
            else:
                QMessageBox.critical(self, "Connection Failed", "Failed to connect to the TCP server")
        
        elif protocol == "UDP":
            if self.comm_manager.connect_udp(port, host, port):
                self.connected = True
                self.network_connect_button.setText("Disconnect")
                self.start_stop_button.setEnabled(True)
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
    
    def toggle_realtime(self):
        """Toggle real-time data visualization."""
        if self.start_stop_button.text() == "Start":
            self.start_stop_button.setText("Stop")
            # If we have file data but no connection, simulate real-time
            if not self.connected and self.parser.data is not None:
                self.simulate_realtime()
        else:
            self.start_stop_button.setText("Start")
    
    def simulate_realtime(self):
        """Simulate real-time data from loaded file data."""
        if self.parser.data is None or self.parser.data.empty:
            return
        
        # Create a timer to send data points at regular intervals
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.send_simulated_data)
        self.sim_index = 0
        self.sim_timer.start(100)  # 100ms interval (10Hz)
    
    def send_simulated_data(self):
        """Send a simulated data point from the loaded file data."""
        if self.start_stop_button.text() == "Stop" and self.parser.data is not None:
            if self.sim_index < len(self.parser.data):
                # Get the current row as a dictionary
                row = self.parser.data.iloc[self.sim_index].to_dict()
                
                # Convert to a CSV line
                line = ",".join([str(row[col]) for col in self.parser.data.columns])
                
                # Process the line as if it came from a connection
                self.process_received_data(line)
                
                self.sim_index += 1
            else:
                # Reached the end, stop simulation
                self.sim_timer.stop()
                self.start_stop_button.setText("Start")
    
    def process_received_data(self, data_line):
        """Process a received line of data."""
        try:
            # Parse data line
            if "," in data_line:  # CSV format
                values = data_line.split(",")
                
                # Check if this is a header line
                if values[0].strip().lower() == "timestamp":
                    return  # Skip header line
                
                # Create data dictionary
                # Adjust the column names based on what you expect from the header
                columns = self.parser.column_names or [
                    "Timestamp", "FixType", "Sats", "Lat", "Long", "Alt", "AltMSL", 
                    "Speed", "Heading", "pDOP", "RTK", "Pressure", "Temperature", 
                    "KX134_AccelX", "KX134_AccelY", "KX134_AccelZ", "ICM_AccelX", 
                    "ICM_AccelY", "ICM_AccelZ", "ICM_GyroX", "ICM_GyroY", "ICM_GyroZ", 
                    "ICM_MagX", "ICM_MagY", "ICM_MagZ", "ICM_Temp", "ICM_Q0", "ICM_Q1", "ICM_Q2", "ICM_Q3"
                ]
                
                data_point = {}
                for i, value in enumerate(values):
                    if i < len(columns):
                        try:
                            data_point[columns[i]] = float(value)
                        except ValueError:
                            data_point[columns[i]] = value
                
                # Process batch data
                self.parser.process_batch(data_line)
                
                # Update real-time plot if it's active
                if self.start_stop_button.text() == "Stop":
                    self.realtime_plot.add_data_point(data_point)
                
                # Update status bar occasionally
                if int(time.time()) % 5 == 0:
                    self.status_bar.showMessage(f"Receiving data: {data_point['Timestamp']}")
        
        except Exception as e:
            print(f"Error processing data: {e}")
    
    def update_data_preview(self, data):
        """Update the data preview with information about the loaded data."""
        if data is not None and not data.empty:
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
        
        # Stop simulation if running
        if hasattr(self, 'sim_timer') and self.sim_timer.isActive():
            self.sim_timer.stop()
        
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 