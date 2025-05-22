import unittest
import pandas as pd
import numpy as np
import json
import os
import io

# Adjust the import path based on your project structure
# Assuming flight_data_parser.py is in the same directory or accessible via PYTHONPATH
from flight_data_parser import FlightDataParser

class TestFlightDataParser(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        # Default mapping content for use in tests
        self.default_mapping_content = {
            "description": "Default mapping for standard flight data CSV.",
            "column_mappings": [
                { "internal_name": "seqNum", "csv_header": "seqNum", "csv_index": 0, "default_value": 0, "type": "int" },
                { "internal_name": "Timestamp", "csv_header": "Timestamp", "csv_index": 1, "default_value": 0, "type": "int" },
                { "internal_name": "FixType", "csv_header": "FixType", "csv_index": 2, "default_value": 0, "type": "int" },
                { "internal_name": "Sats", "csv_header": "Sats", "csv_index": 3, "default_value": 0, "type": "int" },
                { "internal_name": "Latitude", "csv_header": "Latitude", "csv_index": 4, "default_value": 0.0, "type": "float" },
                { "internal_name": "Longitude", "csv_header": "Longitude", "csv_index": 5, "default_value": 0.0, "type": "float" },
                { "internal_name": "Altitude", "csv_header": "Altitude", "csv_index": 6, "default_value": 0.0, "type": "float" },
                { "internal_name": "AltitudeMSL", "csv_header": "AltitudeMSL", "csv_index": 7, "default_value": 0.0, "type": "float" },
                { "internal_name": "raw_altitude", "csv_header": "raw_altitude", "csv_index": 8, "default_value": 0.0, "type": "float" },
                { "internal_name": "calibrated_altitude", "csv_header": "calibrated_altitude", "csv_index": 9, "default_value": 0.0, "type": "float" },
                { "internal_name": "Speed", "csv_header": "Speed", "csv_index": 10, "default_value": 0.0, "type": "float" },
                { "internal_name": "Heading", "csv_header": "Heading", "csv_index": 11, "default_value": 0.0, "type": "float" },
                { "internal_name": "pDOP", "csv_header": "pDOP", "csv_index": 12, "default_value": 9999.0, "type": "float" },
                { "internal_name": "RTK", "csv_header": "RTK", "csv_index": 13, "default_value": 0, "type": "int" },
                { "internal_name": "Pressure", "csv_header": "Pressure", "csv_index": 14, "default_value": 0.0, "type": "float" },
                { "internal_name": "Temperature", "csv_header": "Temperature", "csv_index": 15, "default_value": 0.0, "type": "float" },
                { "internal_name": "KX134_AccelX", "csv_header": "KX134_AccelX", "csv_index": 16, "default_value": 0.0, "type": "float" },
                { "internal_name": "KX134_AccelY", "csv_header": "KX134_AccelY", "csv_index": 17, "default_value": 0.0, "type": "float" },
                { "internal_name": "KX134_AccelZ", "csv_header": "KX134_AccelZ", "csv_index": 18, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_AccelX", "csv_header": "ICM_AccelX", "csv_index": 19, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_AccelY", "csv_header": "ICM_AccelY", "csv_index": 20, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_AccelZ", "csv_header": "ICM_AccelZ", "csv_index": 21, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_GyroX", "csv_header": "ICM_GyroX", "csv_index": 22, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_GyroY", "csv_header": "ICM_GyroY", "csv_index": 23, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_GyroZ", "csv_header": "ICM_GyroZ", "csv_index": 24, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_MagX", "csv_header": "ICM_MagX", "csv_index": 25, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_MagY", "csv_header": "ICM_MagY", "csv_index": 26, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_MagZ", "csv_header": "ICM_MagZ", "csv_index": 27, "default_value": 0.0, "type": "float" },
                { "internal_name": "ICM_Temp", "csv_header": "ICM_Temp", "csv_index": 28, "default_value": 0.0, "type": "float" }
            ],
            "csv_settings": {
                "delimiter": ",",
                "has_header": True
            }
        }
        self.default_mapping_filename = "data_mapping.json" # Default path parser uses

        # Create a parser instance that will attempt to load the default mapping if it exists
        # For some tests, we'll ensure it exists, for others, we'll test its absence.
        self.parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)

        # Sample CSV data strings
        self.csv_data_with_header = (
            "seqNum,Timestamp,FixType,Sats,Latitude,Longitude,Altitude,AltitudeMSL,raw_altitude,calibrated_altitude,Speed,Heading,pDOP,RTK,Pressure,Temperature,KX134_AccelX,KX134_AccelY,KX134_AccelZ,ICM_AccelX,ICM_AccelY,ICM_AccelZ,ICM_GyroX,ICM_GyroY,ICM_GyroZ,ICM_MagX,ICM_MagY,ICM_MagZ,ICM_Temp\n"
            "1,1000,3,10,34.0522,-118.2437,100.0,110.0,95.0,105.0,10.0,90.0,150.0,1,1012.0,25.0,0.1,0.2,9.8,0.11,0.21,9.81,0.01,0.02,0.03,10.0,20.0,30.0,22.0\n"
            "2,2000,3,11,34.0523,-118.2438,102.0,112.0,97.0,107.0,12.0,92.0,120.0,1,1011.0,25.1,0.12,0.22,9.82,0.13,0.23,9.83,0.012,0.022,0.032,12.0,22.0,32.0,22.1"
        )
        self.csv_data_no_header = (
            "1,1000,3,10,34.0522,-118.2437,100.0,110.0,95.0,105.0,10.0,90.0,150.0,1,1012.0,25.0,0.1,0.2,9.8,0.11,0.21,9.81,0.01,0.02,0.03,10.0,20.0,30.0,22.0\n"
            "2,2000,3,11,34.0523,-118.2438,102.0,112.0,97.0,107.0,12.0,92.0,120.0,1,1011.0,25.1,0.12,0.22,9.82,0.13,0.23,9.83,0.012,0.022,0.032,12.0,22.0,32.0,22.1"
        )
        self.csv_data_missing_cols_with_header = ( # Missing last few columns
            "seqNum,Timestamp,FixType,Sats,Latitude,Longitude,Altitude,AltitudeMSL,raw_altitude,calibrated_altitude,Speed,Heading,pDOP,RTK,Pressure,Temperature\n"
            "1,1000,3,10,34.0522,-118.2437,100.0,110.0,95.0,105.0,10.0,90.0,150.0,1,1012.0,25.0\n"
            "2,2000,3,11,34.0523,-118.2438,102.0,112.0,97.0,107.0,12.0,92.0,120.0,1,1011.0,25.1"
        )


    def tearDown(self):
        """Tear down test fixtures, if any."""
        files_to_remove = [
            self.default_mapping_filename, 
            "custom_temp_mapping.json", 
            "invalid_mapping.json",
            "test_data.csv" # For file loading tests
        ]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)

    def test_load_default_mapping_successful(self):
        # Create the default mapping file for this test
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename) # Re-init to load the file
        
        self.assertIsNotNone(parser.mapping_config)
        self.assertEqual(parser.mapping_filepath, self.default_mapping_filename)
        self.assertIn("Timestamp", parser.column_map)
        self.assertIn("Latitude", parser.column_map)
        self.assertEqual(parser.column_map["Latitude"]["csv_header"], "Latitude")
        self.assertEqual(len(parser.mapping_config.get("column_mappings", [])), 29)

    def test_load_custom_mapping_successful(self):
        custom_mapping_content = {
            "description": "Custom test mapping.",
            "column_mappings": [
                { "internal_name": "Time", "csv_header": "timestamp_ms", "csv_index": 0, "default_value": 0, "type": "int" },
                { "internal_name": "TempC", "csv_header": "temperature_c", "csv_index": 1, "default_value": -273.15, "type": "float" }
            ],
            "csv_settings": { "delimiter": ";", "has_header": True }
        }
        custom_filename = "custom_temp_mapping.json"
        with open(custom_filename, 'w') as f:
            json.dump(custom_mapping_content, f, indent=2)
        
        # Use the existing parser instance and call load_mapping
        self.parser.load_mapping(custom_filename)
        
        self.assertIsNotNone(self.parser.mapping_config)
        self.assertEqual(self.parser.mapping_filepath, custom_filename)
        self.assertIn("Time", self.parser.column_map)
        self.assertEqual(self.parser.column_map["TempC"]["default_value"], -273.15)
        self.assertEqual(self.parser.mapping_config["csv_settings"]["delimiter"], ";")
        self.assertEqual(len(self.parser.mapping_config.get("column_mappings", [])), 2)

    def test_load_mapping_file_not_found(self):
        # Ensure the default mapping file (if any) is removed to test fallback
        if os.path.exists(self.default_mapping_filename):
            os.remove(self.default_mapping_filename)
            
        parser = FlightDataParser(mapping_filepath="non_existent_mapping.json")
        # The parser's load_mapping now retains the old config if new one fails,
        # or loads a default empty one if it's the initial load.
        # If "non_existent_mapping.json" isn't found, and no prior mapping was loaded,
        # it should have the default empty/fallback structure.
        self.assertIsNotNone(parser.mapping_config)
        self.assertTrue(len(parser.mapping_config.get("column_mappings", [])) == 0 or parser.mapping_filepath is None)
        # If non_existent_mapping.json fails, mapping_filepath should NOT be non_existent_mapping.json
        # It will be None if it's the first load attempt from __init__
        self.assertNotEqual(parser.mapping_filepath, "non_existent_mapping.json")


    def test_load_mapping_invalid_json(self):
        invalid_filename = "invalid_mapping.json"
        with open(invalid_filename, 'w') as f:
            f.write("{'bad_json': ") # Malformed JSON
        
        # Temporarily remove default to ensure this test focuses on the invalid file
        default_existed = os.path.exists(self.default_mapping_filename)
        if default_existed:
           os.rename(self.default_mapping_filename, self.default_mapping_filename + ".bak")

        parser = FlightDataParser(mapping_filepath=invalid_filename)
        
        # Should fall back to an empty or default mapping config
        self.assertIsNotNone(parser.mapping_config)
        self.assertTrue(len(parser.mapping_config.get("column_mappings", [])) == 0 or parser.mapping_filepath is None)
        self.assertNotEqual(parser.mapping_filepath, invalid_filename)

        if default_existed: # Restore default if it was backed up
            os.rename(self.default_mapping_filename + ".bak", self.default_mapping_filename)


    def test_cast_value(self):
        # Test _cast_value with various inputs
        self.assertEqual(self.parser._cast_value("123", "int", 0), 123)
        self.assertAlmostEqual(self.parser._cast_value("12.3", "float", 0.0), 12.3)
        self.assertEqual(self.parser._cast_value("  -10  ", "int", 0), -10) # Test with whitespace
        self.assertEqual(self.parser._cast_value("text", "str", ""), "text")
        
        # Test defaults on error
        self.assertEqual(self.parser._cast_value("abc", "int", 0), 0)
        self.assertAlmostEqual(self.parser._cast_value("xyz", "float", 0.1), 0.1)
        self.assertEqual(self.parser._cast_value("", "int", 5), 5) # Empty string for int
        self.assertAlmostEqual(self.parser._cast_value("", "float", 0.5), 0.5) # Empty string for float
        
        # Test None input
        self.assertEqual(self.parser._cast_value(None, "int", -1), -1)
        self.assertIsNone(self.parser._cast_value(None, "str", None)) # Default is None
        
        # Test np.nan input
        self.assertAlmostEqual(self.parser._cast_value(np.nan, "float", 0.77), 0.77)
        self.assertEqual(self.parser._cast_value(np.nan, "int", 99), 99)

        # Test unknown type (should return original value or default based on implementation)
        # Current _cast_value returns original value if type is unknown
        self.assertEqual(self.parser._cast_value("val", "unknown_type", "default"), "val")

    def test_load_from_string_default_mapping_with_header(self):
        # Ensure default mapping is loaded for the parser instance for this test
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)

        df = parser.load_from_string(self.csv_data_with_header)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn("Timestamp", df.columns)
        self.assertIn("Latitude", df.columns)
        self.assertIn("ICM_Temp", df.columns) # Check one of the last columns
        
        # Check first row values
        self.assertEqual(df["seqNum"].iloc[0], 1)
        self.assertEqual(df["Timestamp"].iloc[0], 1000)
        self.assertAlmostEqual(df["Latitude"].iloc[0], 34.0522)
        self.assertAlmostEqual(df["AltitudeMSL"].iloc[0], 110.0)
        self.assertAlmostEqual(df["pDOP"].iloc[0], 150.0) # pDOP is float
        self.assertAlmostEqual(df["KX134_AccelZ"].iloc[0], 9.8)
        self.assertAlmostEqual(df["ICM_Temp"].iloc[0], 22.0)

        # Check types (select a few)
        self.assertTrue(pd.api.types.is_integer_dtype(df["Timestamp"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["Latitude"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["Temperature"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["ICM_GyroX"]))


    def test_load_from_string_default_mapping_no_header(self):
        # Ensure default mapping is loaded
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)
        
        # Modify mapping for this test to expect no header
        parser.mapping_config['csv_settings']['has_header'] = False
        # Ensure all column_mappings have csv_index for this to work reliably
        for i, item in enumerate(parser.mapping_config['column_mappings']):
            item['csv_index'] = i 

        df = parser.load_from_string(self.csv_data_no_header)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn("Timestamp", df.columns)
        self.assertIn("Longitude", df.columns)

        self.assertEqual(df["seqNum"].iloc[0], 1)
        self.assertAlmostEqual(df["Longitude"].iloc[1], -118.2438)
        self.assertAlmostEqual(df["Temperature"].iloc[0], 25.0)


    def test_load_from_string_missing_columns_uses_defaults(self):
        # Uses default mapping (which expects more columns than provided in csv_data_missing_cols_with_header)
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)

        df = parser.load_from_string(self.csv_data_missing_cols_with_header)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        
        # Check that existing columns are parsed correctly
        self.assertEqual(df["Timestamp"].iloc[0], 1000)
        self.assertAlmostEqual(df["Temperature"].iloc[1], 25.1)

        # Check that missing columns have default values from the mapping
        # Example: KX134_AccelX's default is 0.0
        self.assertTrue("KX134_AccelX" in df.columns)
        self.assertAlmostEqual(df["KX134_AccelX"].iloc[0], 0.0) 
        self.assertAlmostEqual(df["KX134_AccelX"].iloc[1], 0.0)

        # Example: ICM_Temp's default is 0.0
        self.assertTrue("ICM_Temp" in df.columns)
        self.assertAlmostEqual(df["ICM_Temp"].iloc[0], 0.0)


    def test_add_data_row_valid(self):
        # Load a mapping to define column structure and types
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)
        parser.data = None # Ensure data starts empty

        row_dict = {
            "Timestamp": 5000, "Latitude": 35.1234, "Longitude": -119.5678,
            "AltitudeMSL": 250.7, "Temperature": 28.5, "Speed": 22.3,
            # Add a few more fields to match default mapping
            "seqNum": 10, "FixType": 3, "Sats": 15, "Altitude": 240.0,
            "raw_altitude": 235.0, "calibrated_altitude": 255.0, "Heading": 180.0,
            "pDOP": 1.2, "RTK": 1, "Pressure": 1005.0,
            "KX134_AccelX": 0.01, "KX134_AccelY": 0.02, "KX134_AccelZ": 9.80,
            "ICM_AccelX": 0.03, "ICM_AccelY": 0.04, "ICM_AccelZ": 9.82,
            "ICM_GyroX": 0.1, "ICM_GyroY": 0.2, "ICM_GyroZ": 0.3,
            "ICM_MagX": 1.0, "ICM_MagY": 2.0, "ICM_MagZ": 3.0, "ICM_Temp": 30.0
        }
        parser.add_data_row(row_dict)
        
        self.assertIsNotNone(parser.data)
        self.assertEqual(len(parser.data), 1)
        
        # Verify some values and types
        self.assertEqual(parser.data["Timestamp"].iloc[0], 5000)
        self.assertAlmostEqual(parser.data["Latitude"].iloc[0], 35.1234)
        self.assertAlmostEqual(parser.data["Temperature"].iloc[0], 28.5)
        self.assertTrue(pd.api.types.is_integer_dtype(parser.data["Timestamp"]))
        self.assertTrue(pd.api.types.is_float_dtype(parser.data["Latitude"]))

        # Verify a field that might have been missing in row_dict but has a default in mapping
        # (although in this case, row_dict is quite complete)
        # For example, if pDOP was missing from row_dict, it should be 9999.0
        # self.assertAlmostEqual(parser.data["pDOP"].iloc[0], 1.2) # It's in row_dict

        # Add another row
        row_dict2 = {
            "Timestamp": 6000, "Latitude": 35.5, "Longitude": -119.6,
            "AltitudeMSL": 260.0, "Temperature": 29.0, "Speed": 25.0,
            # Missing some fields, defaults should be used
             "seqNum": 11, "FixType": 3, "Sats": 16
        }
        parser.add_data_row(row_dict2)
        self.assertEqual(len(parser.data), 2)
        self.assertEqual(parser.data["Timestamp"].iloc[1], 6000)
        self.assertAlmostEqual(parser.data["Speed"].iloc[1], 25.0)
        # Check default for a field missing in row_dict2 but present in mapping (e.g., pDOP)
        self.assertAlmostEqual(parser.data["pDOP"].iloc[1], 9999.0) # Default from mapping

    def test_get_column_group(self):
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)
        
        df = parser.load_from_string(self.csv_data_with_header)
        self.assertIsNotNone(df)

        gps_df = parser.get_column_group('gps')
        self.assertIsNotNone(gps_df)
        self.assertIn("Timestamp", gps_df.columns)
        self.assertIn("Latitude", gps_df.columns)
        self.assertIn("Longitude", gps_df.columns)
        self.assertIn("Altitude", gps_df.columns)
        self.assertNotIn("Pressure", gps_df.columns) # Belongs to environmental
        self.assertEqual(len(gps_df.columns), len(FlightDataParser.COLUMN_GROUPS['gps']))

        env_df = parser.get_column_group('environmental')
        self.assertIn("Pressure", env_df.columns)
        self.assertIn("Temperature", env_df.columns)

        with self.assertRaises(ValueError):
            parser.get_column_group('non_existent_group')
            
        # Test with no data loaded
        parser.data = None # Clear data
        empty_gps_df = parser.get_column_group('gps')
        self.assertTrue(empty_gps_df.empty)
        self.assertListEqual(list(empty_gps_df.columns), FlightDataParser.COLUMN_GROUPS['gps'])

    def test_load_from_file_default_mapping_with_header(self):
        # Ensure default mapping is loaded
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)

        test_csv_filename = "test_data.csv"
        with open(test_csv_filename, 'w') as f:
            f.write(self.csv_data_with_header)
        
        df = parser.load_from_file(test_csv_filename)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["Timestamp"].iloc[0], 1000)
        self.assertAlmostEqual(df["Latitude"].iloc[1], 34.0523)

    def test_load_from_file_no_header_uses_indices(self):
        # Setup mapping for no header and indexed columns
        custom_mapping_content = self.default_mapping_content.copy() # Start with default
        custom_mapping_content['csv_settings'] = {'delimiter': ',', 'has_header': False}
        for i, item in enumerate(custom_mapping_content['column_mappings']):
            item['csv_index'] = i # Ensure all items have an index

        custom_mapping_filename = "custom_temp_mapping.json"
        with open(custom_mapping_filename, 'w') as f:
            json.dump(custom_mapping_content, f, indent=2)
        
        parser = FlightDataParser(mapping_filepath=custom_mapping_filename)

        test_csv_filename = "test_data_no_header.csv"
        with open(test_csv_filename, 'w') as f:
            f.write(self.csv_data_no_header)

        df = parser.load_from_file(test_csv_filename)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["seqNum"].iloc[0], 1)
        self.assertAlmostEqual(df["Latitude"].iloc[1], 34.0523)
        self.assertAlmostEqual(df["Temperature"].iloc[0], 25.0)

    def test_load_from_file_custom_delimiter(self):
        # Setup mapping for custom delimiter
        custom_mapping_content = self.default_mapping_content.copy() # Start with default
        custom_mapping_content['csv_settings'] = {'delimiter': ';', 'has_header': True}
        
        custom_mapping_filename = "custom_temp_mapping_delim.json"
        with open(custom_mapping_filename, 'w') as f:
            json.dump(custom_mapping_content, f, indent=2)

        parser = FlightDataParser(mapping_filepath=custom_mapping_filename)

        # Create CSV data with semicolon delimiter
        csv_data_semicolon = self.csv_data_with_header.replace(',', ';')
        test_csv_filename = "test_data_semicolon.csv"
        with open(test_csv_filename, 'w') as f:
            f.write(csv_data_semicolon)

        df = parser.load_from_file(test_csv_filename)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertEqual(df["Timestamp"].iloc[0], 1000)
        self.assertAlmostEqual(df["Latitude"].iloc[1], 34.0523)
        
        # Clean up this specific test file too
        if os.path.exists(test_csv_filename):
            os.remove(test_csv_filename)
        if os.path.exists(custom_mapping_filename):
            os.remove(custom_mapping_filename)


    def test_get_data_statistics(self):
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)
        parser.load_from_string(self.csv_data_with_header)
        
        stats = parser.get_data_statistics()
        self.assertIsNotNone(stats)
        self.assertIn("Temperature", stats)
        self.assertAlmostEqual(stats["Temperature"]["min"], 25.0)
        self.assertAlmostEqual(stats["Temperature"]["max"], 25.1)
        self.assertAlmostEqual(stats["Temperature"]["mean"], 25.05)
        self.assertIn("Speed", stats)
        self.assertAlmostEqual(stats["Speed"]["min"], 10.0)
        self.assertAlmostEqual(stats["Speed"]["max"], 12.0)
        
        # Test with no data
        parser.data = pd.DataFrame() # Empty dataframe
        stats_empty = parser.get_data_statistics()
        self.assertEqual(stats_empty, {})

    def test_get_latest_data(self):
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)
        parser.load_from_string(self.csv_data_with_header)

        latest_one = parser.get_latest_data(1)
        self.assertEqual(len(latest_one), 1)
        self.assertEqual(latest_one["Timestamp"].iloc[0], 2000)

        latest_two = parser.get_latest_data(2)
        self.assertEqual(len(latest_two), 2)
        self.assertEqual(latest_two["Timestamp"].iloc[0], 1000) # First of the two
        self.assertEqual(latest_two["Timestamp"].iloc[1], 2000) # Second of the two

        # Test with no data
        parser.data = None
        latest_empty = parser.get_latest_data(1)
        self.assertTrue(latest_empty.empty)


    def test_get_time_window(self):
        with open(self.default_mapping_filename, 'w') as f:
            json.dump(self.default_mapping_content, f, indent=2)
        parser = FlightDataParser(mapping_filepath=self.default_mapping_filename)
        
        # Create more diverse timestamp data for this test
        csv_data_timestamps = (
            "seqNum,Timestamp,FixType,Sats,Latitude,Longitude,Altitude,AltitudeMSL,raw_altitude,calibrated_altitude,Speed,Heading,pDOP,RTK,Pressure,Temperature,KX134_AccelX,KX134_AccelY,KX134_AccelZ,ICM_AccelX,ICM_AccelY,ICM_AccelZ,ICM_GyroX,ICM_GyroY,ICM_GyroZ,ICM_MagX,ICM_MagY,ICM_MagZ,ICM_Temp\n"
            "1,1000,3,10,34.0,-118.0,100,110,95,105,10,90,1.5,1,1012,25,0,0,9.8,0,0,9.8,0,0,0,0,0,0,22\n"
            "2,1500,3,11,34.1,-118.1,102,112,97,107,12,92,1.2,1,1011,25.1,0,0,9.8,0,0,9.8,0,0,0,0,0,0,22.1\n"
            "3,2000,3,12,34.2,-118.2,104,114,99,109,14,94,1.0,1,1010,25.2,0,0,9.8,0,0,9.8,0,0,0,0,0,0,22.2\n"
            "4,2500,3,13,34.3,-118.3,106,116,101,111,16,96,0.8,1,1009,25.3,0,0,9.8,0,0,9.8,0,0,0,0,0,0,22.3\n"
            "5,3000,3,14,34.4,-118.4,108,118,103,113,18,98,0.7,1,1008,25.4,0,0,9.8,0,0,9.8,0,0,0,0,0,0,22.4"
        )
        parser.load_from_string(csv_data_timestamps)

        # Timestamps are 1000, 1500, 2000, 2500, 3000
        window1 = parser.get_time_window(1500, 2500) # Includes 1500, 2000, 2500
        self.assertEqual(len(window1), 3)
        self.assertEqual(window1["Timestamp"].iloc[0], 1500)
        self.assertEqual(window1["Timestamp"].iloc[-1], 2500)

        window2 = parser.get_time_window(900, 1600) # Includes 1000, 1500
        self.assertEqual(len(window2), 2)
        self.assertEqual(window2["Timestamp"].iloc[0], 1000)

        window_empty = parser.get_time_window(4000, 5000)
        self.assertTrue(window_empty.empty)

        # Test with no data
        parser.data = None
        empty_window = parser.get_time_window(1000, 2000)
        self.assertTrue(empty_window.empty)


if __name__ == '__main__':
    unittest.main()
