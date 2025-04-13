#!/usr/bin/env python3
"""
Test Script for TripleT Flight Console

This script tests the core functionality of the flight console application.
"""

import os
import sys
import matplotlib.pyplot as plt

# Import our modules
from flight_data_parser import FlightDataParser
from data_visualizer import DataVisualizer
from data_exporter import DataExporter


def test_data_parser():
    """Test the FlightDataParser functionality."""
    print("\n=== Testing FlightDataParser ===")
    
    parser = FlightDataParser()
    
    # Test loading from file
    try:
        data = parser.load_from_file("sample_flight_data.csv")
        print(f"✓ Successfully loaded data from file: {len(data)} rows, {len(data.columns)} columns")
        
        # Test column groups
        for group_name in parser.COLUMN_GROUPS:
            group_data = parser.get_column_group(group_name)
            print(f"✓ Retrieved column group '{group_name}': {len(group_data.columns)} columns")
        
        # Test statistics
        stats = parser.get_data_statistics()
        print(f"✓ Generated statistics for {len(stats)} columns")
        
        # Test time window
        min_time = data['Timestamp'].min()
        max_time = data['Timestamp'].max()
        mid_time = min_time + (max_time - min_time) // 2
        
        window_data = parser.get_time_window(mid_time, max_time)
        print(f"✓ Retrieved time window data: {len(window_data)} rows")
        
        # Test latest data
        latest = parser.get_latest_data(5)
        print(f"✓ Retrieved latest data: {len(latest)} rows")
        
        return data
    
    except Exception as e:
        print(f"✗ Error testing FlightDataParser: {e}")
        return None


def test_data_visualizer(data):
    """Test the DataVisualizer functionality."""
    print("\n=== Testing DataVisualizer ===")
    
    if data is None:
        print("✗ Cannot test visualizer without data")
        return
    
    visualizer = DataVisualizer()
    
    try:
        # Test time series plot
        fig1 = visualizer.create_time_series_plot(data, ["Alt", "Speed"], "Altitude and Speed")
        print("✓ Created time series plot")
        
        # Test GPS trajectory plot
        fig2 = visualizer.create_gps_trajectory_plot(data)
        print("✓ Created GPS trajectory plot")
        
        # Test sensor comparison plot
        sensor_groups = {
            "KX134": ["KX134_AccelX", "KX134_AccelY", "KX134_AccelZ"],
            "ICM": ["ICM_AccelX", "ICM_AccelY", "ICM_AccelZ"]
        }
        fig3 = visualizer.create_sensor_comparison_plot(data, sensor_groups)
        print("✓ Created sensor comparison plot")
        
        # Test quaternion conversion
        q0 = data['ICM_Q0'].iloc[0]
        q1 = data['ICM_Q1'].iloc[0]
        q2 = data['ICM_Q2'].iloc[0]
        q3 = data['ICM_Q3'].iloc[0]
        
        rotation_matrix = visualizer.create_attitude_visualization(q0, q1, q2, q3)
        print(f"✓ Created attitude visualization: {rotation_matrix.shape} rotation matrix")
        
        # Display all figures (optional)
        # plt.show()
        
    except Exception as e:
        print(f"✗ Error testing DataVisualizer: {e}")


def test_data_exporter(data):
    """Test the DataExporter functionality."""
    print("\n=== Testing DataExporter ===")
    
    if data is None:
        print("✗ Cannot test exporter without data")
        return
    
    exporter = DataExporter()
    
    try:
        # Create test directory if it doesn't exist
        test_dir = "test_output"
        os.makedirs(test_dir, exist_ok=True)
        
        # Test CSV export
        csv_path = os.path.join(test_dir, "flight_data_export.csv")
        if exporter.export_to_csv(data, csv_path):
            print(f"✓ Exported data to CSV: {csv_path}")
        
        # Test JSON export
        json_path = os.path.join(test_dir, "flight_data_export.json")
        if exporter.export_to_json(data, json_path):
            print(f"✓ Exported data to JSON: {json_path}")
        
        # Test summary report
        report_path = os.path.join(test_dir, "flight_data_summary.txt")
        if exporter.export_summary_report(data, report_path):
            print(f"✓ Created summary report: {report_path}")
        
        print(f"Test output files saved to directory: {test_dir}")
        
    except Exception as e:
        print(f"✗ Error testing DataExporter: {e}")


def main():
    """Main test function."""
    print("TripleT Flight Console Test Script")
    print("=================================")
    
    # Test data parser
    data = test_data_parser()
    
    # Test visualizer
    test_data_visualizer(data)
    
    # Test exporter
    test_data_exporter(data)
    
    print("\nTests completed!")


if __name__ == "__main__":
    main() 