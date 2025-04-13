# TripleT Flight Console

A data visualization and analysis tool for model planes and rocketry flight data.

## Overview

TripleT Flight Console is a Python application designed to visualize and analyze flight data from model planes and rockets. It supports both historical data analysis and real-time data visualization via serial, USB, and network connections.

The application provides a modular structure that can be used with either a graphical user interface (GUI) or a command-line interface (CLI), making it versatile for various usage scenarios and platforms.

## Features

- **Data Parsing**: Load and parse flight data from CSV files or streaming sources
- **Real-time Visualization**: Display live data from connected flight controllers
- **Historical Data Analysis**: Analyze and visualize previously recorded flight data
- **GPS Trajectory Plotting**: Visualize flight paths in 2D and 3D
- **Sensor Data Visualization**: Plot accelerometer, gyroscope, magnetometer and other sensor data
- **Multiple Connection Types**: Connect via Serial, USB, TCP/IP, or UDP
- **Data Export**: Export data to various formats (CSV, JSON, Excel, KML)
- **Cross-Platform**: Works on both Windows and Linux

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/triplet-flight-console.git
   cd triplet-flight-console
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Graphical User Interface (GUI)

To start the GUI application:

```
python flight_console.py
```

The GUI provides tabs for:
- Data Connection: Load files or connect to devices
- Real-time Visualization: View live data from connected devices
- Data Analysis: Analyze historical data with various plot types

### Command-Line Interface (CLI)

The CLI version allows for scripting and quick data analysis from the terminal:

```
python flight_console_cli.py [command] [options]
```

#### Available Commands:

1. **Load and analyze data**:
   ```
   python flight_console_cli.py load flight_data.csv --stats --plot Alt Speed
   ```

2. **Export data to different formats**:
   ```
   python flight_console_cli.py export flight_data.csv trajectory.kml --format kml
   ```

3. **List available serial ports**:
   ```
   python flight_console_cli.py serial list
   ```

4. **Stream data from a serial port**:
   ```
   python flight_console_cli.py serial stream /dev/ttyUSB0 --baud 115200
   ```

## Module Structure

- **flight_data_parser.py**: Parser for flight data from various sources
- **data_visualizer.py**: Data visualization capabilities
- **data_comm.py**: Communication with flight controllers via various interfaces
- **data_exporter.py**: Data export functionality for various formats
- **flight_console.py**: Main GUI application
- **flight_console_cli.py**: Command-line interface version

## Input Data Format

The application expects CSV data with the following columns:

```
Timestamp,FixType,Sats,Lat,Long,Alt,AltMSL,Speed,Heading,pDOP,RTK,Pressure,Temperature,KX134_AccelX,KX134_AccelY,KX134_AccelZ,ICM_AccelX,ICM_AccelY,ICM_AccelZ,ICM_GyroX,ICM_GyroY,ICM_GyroZ,ICM_MagX,ICM_MagY,ICM_MagZ,ICM_Temp,ICM_Q0,ICM_Q1,ICM_Q2,ICM_Q3
```

The application can handle data with missing columns or in a different order, as long as the column names match.

## Examples

### Loading Data from a File

```python
from flight_data_parser import FlightDataParser

parser = FlightDataParser()
data = parser.load_from_file("flight_data.csv")
print(f"Loaded {len(data)} data points")
```

### Creating a Time Series Plot

```python
from flight_data_parser import FlightDataParser
from data_visualizer import DataVisualizer
import matplotlib.pyplot as plt

parser = FlightDataParser()
data = parser.load_from_file("flight_data.csv")

visualizer = DataVisualizer()
fig = visualizer.create_time_series_plot(data, ["Alt", "Speed"], "Altitude and Speed")
plt.show()
```

### Connecting to a Serial Port

```python
from data_comm import DataCommManager

comm = DataCommManager()
ports = comm.list_serial_ports()
print("Available ports:", [p['device'] for p in ports])

# Connect to the first available port
if ports:
    comm.connect_serial(ports[0]['device'], 115200)
    print("Connected to", ports[0]['device'])
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This application was developed for the Triple T team's model aviation and rocketry projects.
- Special thanks to all the contributors and testers who made this project possible.
