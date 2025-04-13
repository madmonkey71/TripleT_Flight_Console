"""
FlightDataParser Module

This module provides functionality for parsing and processing flight data from CSV files
or streaming sources. It handles data formatting, cleaning, and preparation for visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Iterator
import io


class FlightDataParser:
    """
    A class to parse and process flight data from various sources.
    
    This class provides methods for loading data from files or streams,
    processing the data, and preparing it for visualization or analysis.
    """
    
    # Define column groups for easier access
    COLUMN_GROUPS = {
        'gps': ['Timestamp', 'FixType', 'Sats', 'Lat', 'Long', 'Alt', 'AltMSL', 'Speed', 'Heading', 'pDOP', 'RTK'],
        'environmental': ['Pressure', 'Temperature'],
        'kx134_accel': ['KX134_AccelX', 'KX134_AccelY', 'KX134_AccelZ'],
        'icm_accel': ['ICM_AccelX', 'ICM_AccelY', 'ICM_AccelZ'],
        'icm_gyro': ['ICM_GyroX', 'ICM_GyroY', 'ICM_GyroZ'],
        'icm_mag': ['ICM_MagX', 'ICM_MagY', 'ICM_MagZ'],
        'icm_other': ['ICM_Temp', 'ICM_Q0', 'ICM_Q1', 'ICM_Q2', 'ICM_Q3']
    }
    
    def __init__(self):
        """Initialize the FlightDataParser object."""
        self.data = None
        self.column_names = []
    
    def load_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Load flight data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the flight data
        """
        try:
            self.data = pd.read_csv(file_path)
            self.column_names = self.data.columns.tolist()
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data from file: {e}")
    
    def load_from_string(self, data_string: str) -> pd.DataFrame:
        """
        Load flight data from a string (useful for streaming data).
        
        Args:
            data_string: CSV formatted string containing flight data
            
        Returns:
            DataFrame containing the flight data
        """
        try:
            self.data = pd.read_csv(io.StringIO(data_string))
            self.column_names = self.data.columns.tolist()
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data from string: {e}")
    
    def process_batch(self, batch_data: str) -> pd.DataFrame:
        """
        Process a batch of data from a stream.
        
        Args:
            batch_data: New data batch as CSV string (with or without header)
            
        Returns:
            DataFrame containing the processed batch data
        """
        try:
            # Check if this is the first batch with headers
            if not self.column_names and "Timestamp" in batch_data.split('\n')[0]:
                return self.load_from_string(batch_data)
            
            # Process batches without headers
            batch_df = pd.read_csv(io.StringIO(batch_data), header=None)
            
            # If we have column names, apply them
            if self.column_names:
                batch_df.columns = self.column_names
            
            if self.data is None:
                self.data = batch_df
            else:
                self.data = pd.concat([self.data, batch_df], ignore_index=True)
                
            return batch_df
        except Exception as e:
            raise Exception(f"Error processing batch data: {e}")
    
    def get_column_group(self, group_name: str) -> pd.DataFrame:
        """
        Get a specific group of columns from the data.
        
        Args:
            group_name: Name of the column group (gps, environmental, etc.)
            
        Returns:
            DataFrame containing only the specified columns
        """
        if self.data is None:
            raise Exception("No data loaded. Load data first with load_from_file or load_from_string.")
        
        if group_name not in self.COLUMN_GROUPS:
            raise ValueError(f"Unknown column group: {group_name}. Valid groups: {list(self.COLUMN_GROUPS.keys())}")
        
        return self.data[self.COLUMN_GROUPS[group_name]]
    
    def get_data_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate basic statistics for each column in the data.
        
        Returns:
            Dictionary with column names as keys and statistics as values
        """
        if self.data is None:
            raise Exception("No data loaded. Load data first with load_from_file or load_from_string.")
        
        stats = {}
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                stats[column] = {
                    'min': self.data[column].min(),
                    'max': self.data[column].max(),
                    'mean': self.data[column].mean(),
                    'std': self.data[column].std()
                }
        
        return stats
    
    def get_latest_data(self, n_rows: int = 1) -> pd.DataFrame:
        """
        Get the most recent rows from the data.
        
        Args:
            n_rows: Number of rows to return
            
        Returns:
            DataFrame with the most recent n_rows
        """
        if self.data is None:
            raise Exception("No data loaded. Load data first with load_from_file or load_from_string.")
        
        return self.data.tail(n_rows)
    
    def get_time_window(self, start_time: int, end_time: int) -> pd.DataFrame:
        """
        Get data within a specific time window.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            DataFrame with data in the specified time window
        """
        if self.data is None:
            raise Exception("No data loaded. Load data first with load_from_file or load_from_string.")
        
        return self.data[(self.data['Timestamp'] >= start_time) & (self.data['Timestamp'] <= end_time)] 