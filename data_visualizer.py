"""
DataVisualizer Module

This module provides functionality for visualizing flight data using various plotting libraries.
It supports different chart types and can handle both static and real-time data visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget


class DataVisualizer:
    """
    A class to visualize flight data in various formats.
    
    This class provides methods for creating different types of visualizations
    for flight data, including time-series plots, 3D trajectory plots, and more.
    """
    
    def __init__(self):
        """Initialize the DataVisualizer object."""
        pass
    
    def create_time_series_plot(self, data: pd.DataFrame, columns: List[str],
                                title: str = "Time Series Plot") -> plt.Figure:
        """
        Create a time series plot for selected columns.
        
        Args:
            data: DataFrame containing the flight data
            columns: List of column names to plot
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        if data.empty:
            raise ValueError("Empty data provided for plotting")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for column in columns:
            if column in data.columns:
                ax.plot(data['Timestamp'], data[column], label=column)
        
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def create_gps_trajectory_plot(self, data: pd.DataFrame) -> plt.Figure:
        """
        Create a 2D GPS trajectory plot.
        
        Args:
            data: DataFrame containing GPS data (must have Lat and Long columns)
            
        Returns:
            Matplotlib Figure object
        """
        if 'Lat' not in data.columns or 'Long' not in data.columns:
            raise ValueError("Data must contain 'Lat' and 'Long' columns")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the trajectory
        ax.plot(data['Long'], data['Lat'], 'b-', linewidth=2)
        ax.plot(data['Long'].iloc[0], data['Lat'].iloc[0], 'go', markersize=8, label='Start')
        ax.plot(data['Long'].iloc[-1], data['Lat'].iloc[-1], 'ro', markersize=8, label='End')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('GPS Trajectory')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def create_3d_trajectory_plot(self, data: pd.DataFrame) -> go.Figure:
        """
        Create a 3D trajectory plot using Plotly.
        
        Args:
            data: DataFrame containing GPS data (must have Lat, Long, and Alt columns)
            
        Returns:
            Plotly Figure object
        """
        if any(col not in data.columns for col in ['Lat', 'Long', 'Alt']):
            raise ValueError("Data must contain 'Lat', 'Long', and 'Alt' columns")
        
        # Create 3D trajectory using Plotly
        fig = go.Figure(data=[go.Scatter3d(
            x=data['Long'],
            y=data['Lat'],
            z=data['Alt'],
            mode='lines+markers',
            marker=dict(
                size=4,
                color=data['Timestamp'],
                colorscale='Viridis',
                opacity=0.8
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        )])
        
        # Add start and end points
        fig.add_trace(go.Scatter3d(
            x=[data['Long'].iloc[0]],
            y=[data['Lat'].iloc[0]],
            z=[data['Alt'].iloc[0]],
            mode='markers',
            marker=dict(
                color='green',
                size=8,
                symbol='circle'
            ),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[data['Long'].iloc[-1]],
            y=[data['Lat'].iloc[-1]],
            z=[data['Alt'].iloc[-1]],
            mode='markers',
            marker=dict(
                color='red',
                size=8,
                symbol='circle'
            ),
            name='End'
        ))
        
        # Update layout
        fig.update_layout(
            title='3D Flight Trajectory',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Altitude (m)',
                aspectmode='data'
            ),
            width=800,
            height=700,
            margin=dict(r=20, l=10, b=10, t=50)
        )
        
        return fig
    
    def create_sensor_comparison_plot(self, data: pd.DataFrame, sensor_groups: Dict[str, List[str]]) -> plt.Figure:
        """
        Create a comparison plot for different sensor readings.
        
        Args:
            data: DataFrame containing sensor data
            sensor_groups: Dictionary mapping sensor group names to lists of column names
            
        Returns:
            Matplotlib Figure object
        """
        n_groups = len(sensor_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(12, 3*n_groups), sharex=True)
        
        if n_groups == 1:
            axes = [axes]
        
        for i, (group_name, columns) in enumerate(sensor_groups.items()):
            for column in columns:
                if column in data.columns:
                    axes[i].plot(data['Timestamp'], data[column], label=column)
            
            axes[i].set_title(f"{group_name} Sensors")
            axes[i].set_ylabel('Value')
            axes[i].grid(True)
            axes[i].legend()
        
        axes[-1].set_xlabel('Timestamp')
        fig.tight_layout()
        
        return fig
    
    def create_attitude_visualization(self, q0: float, q1: float, q2: float, q3: float) -> np.ndarray:
        """
        Create a basic visualization of attitude from quaternion values.
        
        Args:
            q0, q1, q2, q3: Quaternion components
            
        Returns:
            Rotation matrix as numpy array
        """
        # Convert quaternion to rotation matrix
        rotation_matrix = np.zeros((3, 3))
        
        rotation_matrix[0, 0] = 1 - 2 * (q2**2 + q3**2)
        rotation_matrix[0, 1] = 2 * (q1*q2 - q0*q3)
        rotation_matrix[0, 2] = 2 * (q1*q3 + q0*q2)
        
        rotation_matrix[1, 0] = 2 * (q1*q2 + q0*q3)
        rotation_matrix[1, 1] = 1 - 2 * (q1**2 + q3**2)
        rotation_matrix[1, 2] = 2 * (q2*q3 - q0*q1)
        
        rotation_matrix[2, 0] = 2 * (q1*q3 - q0*q2)
        rotation_matrix[2, 1] = 2 * (q2*q3 + q0*q1)
        rotation_matrix[2, 2] = 1 - 2 * (q1**2 + q2**2)
        
        return rotation_matrix
    
    def create_real_time_plot_widget(self, parent: QWidget = None) -> Tuple[pg.PlotWidget, Dict[str, Any]]:
        """
        Create a PyQtGraph widget for real-time plotting.
        
        Args:
            parent: Parent widget
            
        Returns:
            Tuple containing (plot_widget, plot_data_dict)
        """
        plot_widget = pg.PlotWidget(parent=parent)
        plot_widget.setBackground('w')
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setLabel('left', 'Value')
        plot_widget.setLabel('bottom', 'Time (ms)')
        plot_widget.addLegend()
        
        # Create a dictionary to store plot data
        plot_data = {}
        
        return plot_widget, plot_data
    
    def add_data_to_real_time_plot(self, plot_widget: pg.PlotWidget, plot_data: Dict[str, Any],
                                   data_point: Dict[str, float], columns: List[str], window_size: int = 100) -> Dict[str, Any]:
        """
        Add new data points to a real-time plot.
        
        Args:
            plot_widget: PyQtGraph PlotWidget
            plot_data: Dictionary containing plot data
            data_point: New data point as a dict mapping column names to values
            columns: List of column names to plot
            window_size: Number of points to show in the window
            
        Returns:
            Updated plot_data dictionary
        """
        # Initialize data structures if not already present
        if 'timestamp' not in plot_data:
            plot_data['timestamp'] = []
            
            # Create pen colors for each column
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] * 10  # Repeat colors if needed
            
            for i, col in enumerate(columns):
                plot_data[col] = {
                    'data': [],
                    'curve': plot_widget.plot([], [], name=col, pen=pg.mkPen(color=colors[i], width=2))
                }
        
        # Add new timestamp and ensure we only keep the window_size most recent points
        if 'timestamp' in data_point:
            plot_data['timestamp'].append(data_point['timestamp'])
            plot_data['timestamp'] = plot_data['timestamp'][-window_size:]
        else:
            # If no timestamp provided, just use sequence number
            if not plot_data['timestamp']:
                plot_data['timestamp'].append(0)
            else:
                plot_data['timestamp'].append(plot_data['timestamp'][-1] + 1)
            plot_data['timestamp'] = plot_data['timestamp'][-window_size:]
        
        # Update data for each column
        for col in columns:
            if col in data_point and col in plot_data:
                plot_data[col]['data'].append(data_point[col])
                plot_data[col]['data'] = plot_data[col]['data'][-window_size:]
                
                # Update plot
                plot_data[col]['curve'].setData(plot_data['timestamp'], plot_data[col]['data'])
        
        return plot_data 