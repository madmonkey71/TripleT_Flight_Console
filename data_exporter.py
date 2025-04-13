"""
DataExporter Module

This module provides functionality for exporting flight data to various formats,
including CSV, JSON, Excel, and KML for map visualization.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional, Union, Any
import simplekml


class DataExporter:
    """
    A class to export flight data to various formats.
    
    This class provides methods for saving flight data in different formats,
    including CSV, JSON, Excel, and KML for mapping visualizations.
    """
    
    @staticmethod
    def export_to_csv(data: pd.DataFrame, file_path: str) -> bool:
        """
        Export data to a CSV file.
        
        Args:
            data: DataFrame containing the flight data
            file_path: Path where the CSV file will be saved
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            data.to_csv(file_path, index=False)
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def export_to_json(data: pd.DataFrame, file_path: str, orient: str = 'records') -> bool:
        """
        Export data to a JSON file.
        
        Args:
            data: DataFrame containing the flight data
            file_path: Path where the JSON file will be saved
            orient: JSON orientation ('records', 'split', 'index', etc.)
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            data.to_json(file_path, orient=orient, date_format='iso')
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    @staticmethod
    def export_to_excel(data: pd.DataFrame, file_path: str, sheet_name: str = 'Flight Data') -> bool:
        """
        Export data to an Excel file.
        
        Args:
            data: DataFrame containing the flight data
            file_path: Path where the Excel file will be saved
            sheet_name: Name of the sheet within the Excel file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            data.to_excel(file_path, sheet_name=sheet_name, index=False)
            return True
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False
    
    @staticmethod
    def export_to_kml(data: pd.DataFrame, file_path: str, 
                     name_column: Optional[str] = None,
                     description_columns: Optional[List[str]] = None) -> bool:
        """
        Export GPS track to a KML file for visualization in Google Earth or other mapping tools.
        
        Args:
            data: DataFrame containing the flight data (must include Lat, Long, Alt columns)
            file_path: Path where the KML file will be saved
            name_column: Column to use as placemark names
            description_columns: List of columns to include in placemark descriptions
            
        Returns:
            True if export successful, False otherwise
        """
        if 'Lat' not in data.columns or 'Long' not in data.columns:
            print("Error: Data must contain 'Lat' and 'Long' columns")
            return False
        
        try:
            kml = simplekml.Kml()
            
            # Create the main flight track
            track = kml.newlinestring(name="Flight Path")
            track.coords = [(row['Long'], row['Lat'], row.get('Alt', 0)) 
                           for _, row in data.iterrows()]
            
            track.extrude = 1
            track.altitudemode = simplekml.AltitudeMode.absolute
            track.style.linestyle.width = 4
            track.style.linestyle.color = simplekml.Color.blue
            
            # Add points for significant events or all data points if requested
            if name_column or description_columns:
                for idx, row in data.iterrows():
                    point = kml.newpoint()
                    
                    # Set coordinates
                    point.coords = [(row['Long'], row['Lat'], row.get('Alt', 0))]
                    
                    # Set name
                    if name_column and name_column in row:
                        point.name = str(row[name_column])
                    else:
                        point.name = f"Point {idx}"
                    
                    # Set description with selected columns
                    if description_columns:
                        description = ""
                        for col in description_columns:
                            if col in row:
                                description += f"{col}: {row[col]}<br/>"
                        point.description = description
            
            kml.save(file_path)
            return True
            
        except Exception as e:
            print(f"Error exporting to KML: {e}")
            return False
    
    @staticmethod
    def export_summary_report(data: pd.DataFrame, file_path: str, 
                             include_stats: bool = True, 
                             include_columns: Optional[List[str]] = None) -> bool:
        """
        Create a summary report of the flight data in text format.
        
        Args:
            data: DataFrame containing the flight data
            file_path: Path where the report will be saved
            include_stats: Whether to include basic statistics
            include_columns: List of specific columns to include in the report
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                # General information
                f.write("=== FLIGHT DATA SUMMARY ===\n\n")
                f.write(f"Data points: {len(data)}\n")
                
                if 'Timestamp' in data.columns:
                    f.write(f"Time range: {data['Timestamp'].min()} to {data['Timestamp'].max()}\n")
                    duration = data['Timestamp'].max() - data['Timestamp'].min()
                    f.write(f"Flight duration: {duration} ms ({duration/1000:.2f} seconds)\n")
                
                f.write("\n")
                
                # Include statistics for selected columns
                if include_stats:
                    f.write("=== STATISTICS ===\n\n")
                    
                    # Determine which columns to include
                    cols_to_include = include_columns if include_columns else data.select_dtypes(include=['number']).columns
                    
                    for col in cols_to_include:
                        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                            stats = data[col].describe()
                            f.write(f"{col}:\n")
                            f.write(f"  Min: {stats['min']:.4f}\n")
                            f.write(f"  Max: {stats['max']:.4f}\n")
                            f.write(f"  Mean: {stats['mean']:.4f}\n")
                            f.write(f"  Std Dev: {stats['std']:.4f}\n")
                            f.write("\n")
                
                # GPS data summary
                if all(col in data.columns for col in ['Lat', 'Long']):
                    f.write("=== GPS SUMMARY ===\n\n")
                    f.write(f"Start position: {data['Lat'].iloc[0]:.6f}, {data['Long'].iloc[0]:.6f}\n")
                    f.write(f"End position: {data['Lat'].iloc[-1]:.6f}, {data['Long'].iloc[-1]:.6f}\n")
                    
                    if 'Alt' in data.columns:
                        f.write(f"Max altitude: {data['Alt'].max():.2f} m\n")
                    
                    if 'Speed' in data.columns:
                        f.write(f"Max speed: {data['Speed'].max():.2f} m/s ({data['Speed'].max() * 3.6:.2f} km/h)\n")
                
                return True
                
        except Exception as e:
            print(f"Error creating summary report: {e}")
            return False 