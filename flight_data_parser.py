"""
FlightDataParser Module

This module provides functionality for parsing and processing flight data from CSV files
or streaming sources. It handles data formatting, cleaning, and preparation for visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Iterator
import io
import json


class FlightDataParser:
    """
    A class to parse and process flight data from various sources.
    
    This class provides methods for loading data from files or streams,
    processing the data, and preparing it for visualization or analysis.
    """
    
    # Define column groups for easier access - uses internal_names
    COLUMN_GROUPS = {
        'gps': ['Timestamp', 'FixType', 'Sats', 'Latitude', 'Longitude', 'Altitude', 'AltitudeMSL', 'Speed', 'Heading', 'pDOP', 'RTK'],
        'environmental': ['Pressure', 'Temperature'],
        'kx134_accel': ['KX134_AccelX', 'KX134_AccelY', 'KX134_AccelZ'],
        'icm_accel': ['ICM_AccelX', 'ICM_AccelY', 'ICM_AccelZ'],
        'icm_gyro': ['ICM_GyroX', 'ICM_GyroY', 'ICM_GyroZ'],
        'icm_mag': ['ICM_MagX', 'ICM_MagY', 'ICM_MagZ'],
        'icm_other': ['ICM_Temp'], # Removed ICM_Q0 to Q3 as they are not in the mapping
        'quaternion': ['Q0', 'Q1', 'Q2', 'Q3'] # Added quaternion group
    }

    def __init__(self, mapping_filepath: str = "data_mapping.json"):
        """Initialize the FlightDataParser object."""
        self.data: Optional[pd.DataFrame] = None
        self.column_names: List[str] = [] # Stores internal column names
        self.mapping_config: Optional[Dict] = None
        self.column_map: Dict[str, Dict] = {}  # internal_name -> mapping_item
        self.internal_to_csv_header: Dict[str, Optional[str]] = {} # internal_name -> csv_header
        self.csv_header_to_internal: Dict[str, str] = {} # csv_header -> internal_name
        self.mapping_filepath: Optional[str] = None # To store the path of the loaded mapping
        self.load_mapping(mapping_filepath)

    def load_mapping(self, mapping_filepath: str):
        """
        Load the data mapping configuration from a JSON file.

        Args:
            mapping_filepath: Path to the JSON mapping file.
        """
        try:
            with open(mapping_filepath, 'r') as f:
                self.mapping_config = json.load(f)
            
            if not self.mapping_config or 'column_mappings' not in self.mapping_config:
                print(f"Error: 'column_mappings' not found in {mapping_filepath}. Current mapping remains: {self.mapping_filepath}")
                # Potentially revert to a known good default or clear, but for now, just don't update filepath
                # self.mapping_config could be set to a default empty mapping here if desired.
                return # Critical error, do not update self.mapping_filepath

            # Clear previous mapping specifics before loading new ones
            self.column_map.clear()
            self.internal_to_csv_header.clear()
            self.csv_header_to_internal.clear()

            for mapping_item in self.mapping_config.get('column_mappings', []):
                internal_name = mapping_item.get('internal_name')
                if internal_name:
                    self.column_map[internal_name] = mapping_item
                    csv_header = mapping_item.get('csv_header')
                    if csv_header:
                        self.internal_to_csv_header[internal_name] = csv_header
                        self.csv_header_to_internal[csv_header] = internal_name
                else:
                    print(f"Warning: Found a mapping item without an 'internal_name': {mapping_item}")
            
            # Set default csv_settings if not present
            if 'csv_settings' not in self.mapping_config:
                self.mapping_config['csv_settings'] = {'delimiter': ',', 'has_header': True}
            if 'delimiter' not in self.mapping_config['csv_settings']:
                self.mapping_config['csv_settings']['delimiter'] = ','
            if 'has_header' not in self.mapping_config['csv_settings']:
                 self.mapping_config['csv_settings']['has_header'] = True
            
            self.mapping_filepath = mapping_filepath # Update path only on successful load and parse
            print(f"Successfully loaded mapping from: {self.mapping_filepath}")

        except FileNotFoundError:
            print(f"Error: Mapping file '{mapping_filepath}' not found. Current mapping remains: {self.mapping_filepath}")
            # self.mapping_config could be reset to a default here if desired.
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in mapping file '{mapping_filepath}'. Current mapping remains: {self.mapping_filepath}")
            # self.mapping_config could be reset to a default here.
        except Exception as e:
            print(f"An unexpected error occurred while loading mapping '{mapping_filepath}': {e}. Current mapping remains: {self.mapping_filepath}")
            # self.mapping_config could be reset to a default here.

    def _cast_value(self, value, target_type: str, default_value):
        """
        Cast a value to a target type.
        
        Args:
            value: The value to cast.
            target_type: String representing the target type ("int", "float", "str").
            default_value: Default value to return on casting error or if value is None/NaN.
            
        Returns:
            Casted value or default_value.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default_value
        
        try:
            if target_type == "int":
                return int(value)
            elif target_type == "float":
                return float(value)
            elif target_type == "str":
                return str(value)
            else:
                return value # Or raise error for unknown type
        except (ValueError, TypeError):
            return default_value

    def load_from_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load flight data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the flight data
        """
        """
        Load flight data from a CSV file using the mapping configuration.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the flight data with internal column names, or None on error.
        """
        if not self.mapping_config:
            print("Error: Mapping configuration not loaded. Cannot load data from file.")
            return None
        
        csv_settings = self.mapping_config['csv_settings']
        delimiter = csv_settings.get('delimiter', ',')
        has_header = csv_settings.get('has_header', True)
        
        try:
            # Read the CSV file
            # If has_header is True, pandas uses the first row as header (header=0)
            # If has_header is False, pandas auto-generates column names (header=None)
            raw_df = pd.read_csv(file_path, delimiter=delimiter, header=0 if has_header else None)
            
            processed_data = pd.DataFrame()
            
            for mapping_item in self.mapping_config['column_mappings']:
                internal_name = mapping_item['internal_name']
                csv_header = mapping_item.get('csv_header')
                csv_index = mapping_item.get('csv_index')
                default_value = mapping_item.get('default_value')
                target_type = mapping_item.get('type', 'str') # Default to string if type not specified
                
                value_series: Optional[pd.Series] = None
                
                if has_header and csv_header and csv_header in raw_df.columns:
                    value_series = raw_df[csv_header]
                elif not has_header and csv_index is not None and csv_index < len(raw_df.columns):
                    value_series = raw_df.iloc[:, csv_index]
                
                if value_series is not None:
                    processed_data[internal_name] = value_series.apply(
                        lambda x: self._cast_value(x, target_type, default_value)
                    )
                else:
                    # If column not found by header or index, fill with default value
                    num_rows = len(raw_df) if not raw_df.empty else 1
                    processed_data[internal_name] = pd.Series(
                        [self._cast_value(None, target_type, default_value)] * num_rows
                    )
            
            self.data = processed_data
            self.column_names = self.data.columns.tolist()
            return self.data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error loading data from file '{file_path}': {e}")
            # Fallback to an empty DataFrame with internal names if loading fails catastrophically after raw_df load
            if self.mapping_config and 'column_mappings' in self.mapping_config:
                 self.data = pd.DataFrame(columns=[m['internal_name'] for m in self.mapping_config['column_mappings']])
                 self.column_names = self.data.columns.tolist()
            else: # Should not happen if load_mapping was successful
                 self.data = pd.DataFrame()
                 self.column_names = []
            return None
    
    def load_from_string(self, data_string: str) -> Optional[pd.DataFrame]:
        """
        Load flight data from a string (useful for streaming data).
        
        Args:
            data_string: CSV formatted string containing flight data
            
        Returns:
            DataFrame containing the flight data
        """
        """
        Load flight data from a string using the mapping configuration.
        
        Args:
            data_string: CSV formatted string containing flight data
            
        Returns:
            DataFrame containing the flight data with internal column names, or None on error.
        """
        if not self.mapping_config:
            print("Error: Mapping configuration not loaded. Cannot load data from string.")
            return None

        csv_settings = self.mapping_config['csv_settings']
        delimiter = csv_settings.get('delimiter', ',')
        has_header = csv_settings.get('has_header', True)

        try:
            raw_df = pd.read_csv(io.StringIO(data_string), delimiter=delimiter, header=0 if has_header else None)
            
            processed_data = pd.DataFrame()

            for mapping_item in self.mapping_config['column_mappings']:
                internal_name = mapping_item['internal_name']
                csv_header = mapping_item.get('csv_header')
                csv_index = mapping_item.get('csv_index')
                default_value = mapping_item.get('default_value')
                target_type = mapping_item.get('type', 'str')

                value_series: Optional[pd.Series] = None

                if has_header and csv_header and csv_header in raw_df.columns:
                    value_series = raw_df[csv_header]
                elif not has_header and csv_index is not None and csv_index < len(raw_df.columns):
                    value_series = raw_df.iloc[:, csv_index]
                
                if value_series is not None:
                    processed_data[internal_name] = value_series.apply(
                        lambda x: self._cast_value(x, target_type, default_value)
                    )
                else:
                    num_rows = len(raw_df) if not raw_df.empty else 1
                    processed_data[internal_name] = pd.Series(
                        [self._cast_value(None, target_type, default_value)] * num_rows
                    )
            
            self.data = processed_data
            self.column_names = self.data.columns.tolist()
            return self.data
        except Exception as e:
            print(f"Error loading data from string: {e}")
            if self.mapping_config and 'column_mappings' in self.mapping_config:
                 self.data = pd.DataFrame(columns=[m['internal_name'] for m in self.mapping_config['column_mappings']])
                 self.column_names = self.data.columns.tolist()
            else: # Should not happen if load_mapping was successful
                 self.data = pd.DataFrame()
                 self.column_names = []
            return None

    def add_data_row(self, data_dict: Dict[str, any]):
        """
        Adds a single row of data (already mapped to internal names) to the parser's DataFrame.

        Args:
            data_dict: A dictionary where keys are internal_names and values are the data points.
        """
        if not self.column_map: # Ensure mapping is loaded
            print("Warning: No mapping loaded. Cannot reliably add data row.")
            # Fallback: use keys from data_dict as is, if no internal names are set yet
            if not self.column_names:
                 self.column_names = list(data_dict.keys())


        # Create a new DataFrame for the row, ensuring columns match self.column_names order
        # and casting values according to mapping
        
        # Initialize row_values with defaults from mapping for all known internal columns
        # This ensures that if data_dict is missing some keys, they get default values.
        processed_row_values = {}
        for internal_name, mapping_item in self.column_map.items():
            target_type = mapping_item.get('type', 'str')
            default_value = mapping_item.get('default_value')
            # Get value from data_dict or use None to trigger default_value in _cast_value
            raw_value = data_dict.get(internal_name) 
            processed_row_values[internal_name] = self._cast_value(raw_value, target_type, default_value)

        # If data_dict contains keys not in self.column_map (e.g. new/unexpected data)
        # include them as is, but log a warning.
        for key, value in data_dict.items():
            if key not in processed_row_values:
                print(f"Warning: Received data for unknown internal_name '{key}'. Adding as is.")
                processed_row_values[key] = value
                if key not in self.column_names: # Add to overall column list if truly new
                    self.column_names.append(key)


        new_row_df = pd.DataFrame([processed_row_values], columns=self.column_names if self.column_names else list(processed_row_values.keys()))

        if self.data is None or self.data.empty:
            self.data = new_row_df
            # Ensure self.column_names is set from the first row if it wasn't from mapping
            if not self.column_names: # Should be set by mapping, but as a fallback
                 self.column_names = self.data.columns.tolist()
        else:
            # Ensure new_row_df has all columns present in self.data, filling with NaN if necessary,
            # then fill NaNs using defaults or appropriate logic
            for col in self.data.columns:
                if col not in new_row_df:
                    # Find default for this col if possible
                    default_val_for_missing_col = self.column_map.get(col, {}).get('default_value')
                    target_type = self.column_map.get(col, {}).get('type', 'str')
                    new_row_df[col] = self._cast_value(None, target_type, default_val_for_missing_col)
            
            # Reorder new_row_df columns to match self.data.columns before concat
            new_row_df = new_row_df[self.data.columns]
            self.data = pd.concat([self.data, new_row_df], ignore_index=True)

    def process_batch(self, batch_data: str) -> Optional[pd.DataFrame]:
        """
        Processes a batch of data from a stream.
        This method is simplified. For CSV string batches, it's recommended to use
        `load_from_string` if the batch represents a full CSV structure.
        If `flight_console.py` parses individual lines into dictionaries,
        `add_data_row` should be used for each dictionary.

        This method can act as a simple wrapper around load_from_string for now
        if a full CSV string is passed.
        """
        print("Warning: process_batch is simplified. For full CSV strings, use load_from_string. For dicts, use add_data_row.")
        # Attempt to load it as a string, assuming it's a CSV formatted block
        return self.load_from_string(batch_data)

    def get_column_group(self, group_name: str) -> pd.DataFrame:
        """
        Get a specific group of columns from the data.
        
        Args:
            group_name: Name of the column group (gps, environmental, etc.)
            
        Returns:
            DataFrame containing only the specified columns
        """
        if self.data is None or self.data.empty:
            # Return an empty DataFrame with expected columns if data is not loaded
            # This helps prevent errors in downstream components expecting certain columns.
            if group_name in self.COLUMN_GROUPS:
                return pd.DataFrame(columns=self.COLUMN_GROUPS[group_name])
            else:
                 raise ValueError(f"Unknown column group: {group_name}. Valid groups: {list(self.COLUMN_GROUPS.keys())}")

        if group_name not in self.COLUMN_GROUPS:
            raise ValueError(f"Unknown column group: {group_name}. Valid groups: {list(self.COLUMN_GROUPS.keys())}")
        
        # Select only available columns from the group that are actually in self.data
        available_columns = [col for col in self.COLUMN_GROUPS[group_name] if col in self.data.columns]
        if not available_columns:
            # Return an empty DataFrame if none of the group's columns are available
            return pd.DataFrame(columns=self.COLUMN_GROUPS[group_name])
            
        return self.data[available_columns]
    
    def get_data_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate basic statistics for each column in the data.
        
        Returns:
            Dictionary with column names as keys and statistics as values
        """
        if self.data is None or self.data.empty:
            # raise Exception("No data loaded. Load data first with load_from_file or load_from_string.")
            return {} # Return empty dict if no data
        
        stats = {}
        for column in self.data.columns:
            # Ensure column exists and has data before trying to calculate stats
            if column in self.data and not self.data[column].empty and pd.api.types.is_numeric_dtype(self.data[column]):
                # Skip columns that might be all NaN after failed casting or missing data
                if self.data[column].notna().any():
                    stats[column] = {
                        'min': self.data[column].min(),
                        'max': self.data[column].max(),
                        'mean': self.data[column].mean(),
                        'std': self.data[column].std()
                    }
                else:
                    stats[column] = {
                        'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan
                    }
            elif column in self.data: # Non-numeric column
                 stats[column] = {'error': 'non-numeric data'}
        
        return stats
    
    def get_latest_data(self, n_rows: int = 1) -> pd.DataFrame:
        """
        Get the most recent rows from the data.
        
        Args:
            n_rows: Number of rows to return
            
        Returns:
            DataFrame with the most recent n_rows
        """
        if self.data is None or self.data.empty:
            # raise Exception("No data loaded. Load data first with load_from_file or load_from_string.")
            # Return empty DataFrame with internal column names if no data
            return pd.DataFrame(columns=self.column_names) 
        
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
        if self.data is None or self.data.empty:
            # raise Exception("No data loaded. Load data first with load_from_file or load_from_string.")
            # Return empty DataFrame with internal column names if no data
             return pd.DataFrame(columns=self.column_names)
        
        # Ensure 'Timestamp' column exists before trying to filter
        if 'Timestamp' not in self.data.columns:
            print("Warning: 'Timestamp' column not found. Cannot filter by time window.")
            return pd.DataFrame(columns=self.column_names)

        return self.data[(self.data['Timestamp'] >= start_time) & (self.data['Timestamp'] <= end_time)]