#!/usr/bin/env python3
"""
Flight Console CLI

A command-line interface for the TripleT Flight Console.
This allows basic data loading, plotting, and analysis without the GUI.
"""

import argparse
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Import our modules
from flight_data_parser import FlightDataParser
from data_visualizer import DataVisualizer
from data_comm import DataCommManager
from data_exporter import DataExporter


def plot_data(data, columns, title="Time Series Plot", save_path=None):
    """Plot selected columns from the data."""
    visualizer = DataVisualizer()
    fig = visualizer.create_time_series_plot(data, columns, title)
    
    if save_path:
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_gps(data, save_path=None):
    """Plot GPS trajectory from the data."""
    visualizer = DataVisualizer()
    fig = visualizer.create_gps_trajectory_plot(data)
    
    if save_path:
        fig.savefig(save_path)
        print(f"GPS plot saved to {save_path}")
    else:
        plt.show()


def export_data(data, file_path, format_type):
    """Export data to the specified format."""
    exporter = DataExporter()
    
    if format_type == 'csv':
        success = exporter.export_to_csv(data, file_path)
    elif format_type == 'json':
        success = exporter.export_to_json(data, file_path)
    elif format_type == 'excel':
        success = exporter.export_to_excel(data, file_path)
    elif format_type == 'kml':
        success = exporter.export_to_kml(data, file_path)
    elif format_type == 'report':
        success = exporter.export_summary_report(data, file_path)
    else:
        print(f"Unsupported export format: {format_type}")
        return False
    
    if success:
        print(f"Data exported to {file_path}")
    else:
        print(f"Failed to export data to {file_path}")
    
    return success


def list_serial_ports():
    """List available serial ports."""
    comm_manager = DataCommManager()
    ports = comm_manager.list_serial_ports()
    
    if not ports:
        print("No serial ports found")
        return
    
    print("Available serial ports:")
    for port in ports:
        print(f"  - {port['device']}: {port['description']}")


def stream_data(port, baud_rate, plot_columns=None, duration=None):
    """Stream data from a serial port and optionally plot it in real-time."""
    parser = FlightDataParser()
    comm_manager = DataCommManager()
    
    print(f"Connecting to {port} at {baud_rate} baud...")
    if not comm_manager.connect_serial(port, baud_rate):
        print("Failed to connect to serial port")
        return
    
    print("Connection established. Press Ctrl+C to stop.")
    
    try:
        # Set up real-time plotting if requested
        if plot_columns:
            # Create a figure with subplots for each column
            fig, axes = plt.subplots(len(plot_columns), 1, figsize=(10, 2 * len(plot_columns)))
            if len(plot_columns) == 1:
                axes = [axes]
            
            # Initialize data structures for plotting
            timestamps = []
            data_dict = {col: [] for col in plot_columns}
            
            # Maximum number of points to show
            max_points = 100
            
            def update_plot(frame):
                # Get data from the queue
                line = comm_manager.get_queued_data(timeout=0.1)
                if line:
                    try:
                        # Process the data
                        batch_df = parser.process_batch(line)
                        if batch_df is not None and not batch_df.empty:
                            row = batch_df.iloc[0]
                            
                            # Add timestamp
                            timestamps.append(row['Timestamp'])
                            if len(timestamps) > max_points:
                                timestamps.pop(0)
                            
                            # Add data for each column
                            for col in plot_columns:
                                if col in row:
                                    data_dict[col].append(row[col])
                                    if len(data_dict[col]) > max_points:
                                        data_dict[col].pop(0)
                            
                            # Update plots
                            for i, col in enumerate(plot_columns):
                                axes[i].clear()
                                if data_dict[col]:
                                    axes[i].plot(timestamps, data_dict[col])
                                    axes[i].set_ylabel(col)
                                    axes[i].grid(True)
                            
                            axes[-1].set_xlabel('Timestamp')
                            fig.tight_layout()
                    
                    except Exception as e:
                        print(f"Error processing data: {e}")
            
            # Create animation
            ani = FuncAnimation(fig, update_plot, interval=100)
            plt.show()
        
        else:
            # Just display data without plotting
            start_time = time.time()
            while True:
                line = comm_manager.get_queued_data(timeout=0.1)
                if line:
                    print(line)
                
                # Check if duration is reached
                if duration and (time.time() - start_time) > duration:
                    break
    
    except KeyboardInterrupt:
        print("\nStream stopped by user")
    
    finally:
        comm_manager.close()
        print("Connection closed")


def show_data_stats(data):
    """Display basic statistics about the data."""
    if data is None or data.empty:
        print("No data to analyze")
        return
    
    print(f"\nData Statistics:")
    print(f"  - Total data points: {len(data)}")
    
    if 'Timestamp' in data.columns:
        start_time = data['Timestamp'].min()
        end_time = data['Timestamp'].max()
        duration_ms = end_time - start_time
        duration_sec = duration_ms / 1000
        
        print(f"  - Time range: {start_time} to {end_time}")
        print(f"  - Duration: {duration_sec:.2f} seconds")
    
    if all(col in data.columns for col in ['Lat', 'Long']):
        print(f"\nGPS Information:")
        print(f"  - Start position: {data['Lat'].iloc[0]:.6f}, {data['Long'].iloc[0]:.6f}")
        print(f"  - End position: {data['Lat'].iloc[-1]:.6f}, {data['Long'].iloc[-1]:.6f}")
    
    if 'Alt' in data.columns:
        print(f"  - Altitude (m): min={data['Alt'].min():.2f}, max={data['Alt'].max():.2f}, avg={data['Alt'].mean():.2f}")
    
    if 'Speed' in data.columns:
        max_speed = data['Speed'].max()
        print(f"  - Max speed: {max_speed:.2f} m/s ({max_speed * 3.6:.2f} km/h)")
    
    print("\nSensor Statistics:")
    columns = ['KX134_AccelX', 'KX134_AccelY', 'KX134_AccelZ', 
               'ICM_AccelX', 'ICM_AccelY', 'ICM_AccelZ',
               'ICM_GyroX', 'ICM_GyroY', 'ICM_GyroZ',
               'Pressure', 'Temperature']
    
    for col in columns:
        if col in data.columns:
            stats = data[col].describe()
            print(f"  - {col}: min={stats['min']:.4f}, max={stats['max']:.4f}, avg={stats['mean']:.4f}")


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="TripleT Flight Console Command Line Interface")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load and analyze flight data from a file")
    load_parser.add_argument("file", help="Path to the CSV data file")
    load_parser.add_argument("--stats", action="store_true", help="Show data statistics")
    load_parser.add_argument("--plot", nargs="+", help="Plot specified columns (e.g., --plot Alt Speed)")
    load_parser.add_argument("--gps", action="store_true", help="Plot GPS trajectory")
    load_parser.add_argument("--save-plot", help="Save plot to a file instead of displaying it")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export flight data to different formats")
    export_parser.add_argument("input", help="Input CSV data file")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("--format", choices=["csv", "json", "excel", "kml", "report"], 
                             default="csv", help="Export format")
    
    # Serial command
    serial_parser = subparsers.add_parser("serial", help="Work with serial port connections")
    serial_subparsers = serial_parser.add_subparsers(dest="serial_cmd", help="Serial command")
    
    # List serial ports
    list_ports_parser = serial_subparsers.add_parser("list", help="List available serial ports")
    
    # Stream data from serial port
    stream_parser = serial_subparsers.add_parser("stream", help="Stream data from a serial port")
    stream_parser.add_argument("port", help="Serial port to connect to")
    stream_parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    stream_parser.add_argument("--plot", nargs="+", help="Plot specified columns in real-time")
    stream_parser.add_argument("--duration", type=int, help="Duration to stream in seconds")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process commands
    if args.command == "load":
        parser = FlightDataParser()
        try:
            data = parser.load_from_file(args.file)
            print(f"Loaded {len(data)} data points from {args.file}")
            
            if args.stats:
                show_data_stats(data)
            
            if args.plot:
                plot_data(data, args.plot, save_path=args.save_plot)
            
            if args.gps:
                plot_gps(data, save_path=args.save_plot)
        
        except Exception as e:
            print(f"Error loading file: {e}")
    
    elif args.command == "export":
        parser = FlightDataParser()
        try:
            data = parser.load_from_file(args.input)
            print(f"Loaded {len(data)} data points from {args.input}")
            
            export_data(data, args.output, args.format)
        
        except Exception as e:
            print(f"Error during export: {e}")
    
    elif args.command == "serial":
        if args.serial_cmd == "list":
            list_serial_ports()
        
        elif args.serial_cmd == "stream":
            stream_data(args.port, args.baud, args.plot, args.duration)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 