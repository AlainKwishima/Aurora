"""
Rwanda Aurora Utilities and Helper Functions
===========================================

Comprehensive utilities for visualization, data management, performance monitoring,
and other helper functions for the Rwanda weather forecasting system.
"""

import os
import gc
import time
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import xarray as xr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import psutil

# Suppress warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for training and inference
    """
    
    def __init__(self):
        self.metrics = {
            'memory_usage': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'batch_times': [],
            'epoch_times': [],
            'timestamps': []
        }
        
        self.start_time = time.time()
        self.last_check = time.time()
        
        print("Performance monitor initialized")
    
    def log_performance(self, context: str = "general"):
        """Log current performance metrics"""
        current_time = time.time()
        
        # System memory
        memory_info = psutil.virtual_memory()
        self.metrics['memory_usage'].append(memory_info.percent)
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
            self.metrics['gpu_memory'].append(gpu_memory_mb)
        else:
            self.metrics['gpu_memory'].append(0)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics['cpu_usage'].append(cpu_percent)
        
        # Timing
        elapsed = current_time - self.last_check
        self.metrics['batch_times'].append(elapsed)
        self.metrics['timestamps'].append(current_time)
        
        self.last_check = current_time
        
        # Log if requested
        if context != "silent":
            print(f"[{context}] Memory: {memory_info.percent:.1f}%, "
                  f"CPU: {cpu_percent:.1f}%, "
                  f"GPU: {self.metrics['gpu_memory'][-1]:.0f}MB, "
                  f"Time: {elapsed:.2f}s")
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary statistics"""
        if not self.metrics['memory_usage']:
            return {}
        
        summary = {
            'avg_memory_usage': np.mean(self.metrics['memory_usage']),
            'max_memory_usage': np.max(self.metrics['memory_usage']),
            'avg_gpu_memory_mb': np.mean(self.metrics['gpu_memory']),
            'max_gpu_memory_mb': np.max(self.metrics['gpu_memory']),
            'avg_cpu_usage': np.mean(self.metrics['cpu_usage']),
            'max_cpu_usage': np.max(self.metrics['cpu_usage']),
            'avg_batch_time': np.mean(self.metrics['batch_times']),
            'total_runtime': time.time() - self.start_time
        }
        
        return summary
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot performance metrics over time"""
        if not self.metrics['timestamps']:
            print("No performance data to plot")
            return
        
        # Convert timestamps to relative time in minutes
        start_timestamp = self.metrics['timestamps'][0]
        time_minutes = [(t - start_timestamp) / 60 for t in self.metrics['timestamps']]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Rwanda Aurora Performance Monitoring', fontsize=16)
        
        # Memory usage
        axes[0, 0].plot(time_minutes, self.metrics['memory_usage'], 'b-', alpha=0.7)
        axes[0, 0].set_title('System Memory Usage')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Memory Usage (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        axes[0, 0].legend()
        
        # GPU memory
        if max(self.metrics['gpu_memory']) > 0:
            axes[0, 1].plot(time_minutes, self.metrics['gpu_memory'], 'g-', alpha=0.7)
            axes[0, 1].set_title('GPU Memory Usage')
            axes[0, 1].set_xlabel('Time (minutes)')
            axes[0, 1].set_ylabel('GPU Memory (MB)')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No GPU Available', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('GPU Memory Usage')
        
        # CPU usage
        axes[1, 0].plot(time_minutes, self.metrics['cpu_usage'], 'r-', alpha=0.7)
        axes[1, 0].set_title('CPU Usage')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
        axes[1, 0].legend()
        
        # Batch processing times
        if len(self.metrics['batch_times']) > 1:
            axes[1, 1].plot(time_minutes[1:], self.metrics['batch_times'][1:], 'purple', alpha=0.7)
            axes[1, 1].set_title('Batch Processing Times')
            axes[1, 1].set_xlabel('Time (minutes)')
            axes[1, 1].set_ylabel('Batch Time (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Batch Processing Times')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved: {save_path}")
        
        plt.show()

class RwandaVisualizer:
    """
    Advanced visualization tools specifically for Rwanda weather data
    """
    
    def __init__(self):
        # Rwanda geographic bounds
        self.rwanda_bounds = {
            'south': -2.917,
            'north': -1.047,
            'west': 28.862,
            'east': 30.899
        }
        
        # Setup custom colormaps
        self._setup_colormaps()
        
        print("Rwanda visualizer initialized")
    
    def _setup_colormaps(self):
        """Setup custom colormaps for different variables"""
        # Temperature colormap (blue-white-red)
        temp_colors = ['#0066CC', '#4D94FF', '#99CCFF', '#FFFFFF', '#FFB366', '#FF6600', '#CC3300']
        self.temp_cmap = LinearSegmentedColormap.from_list('temperature', temp_colors)
        
        # Precipitation colormap (white-blue-dark blue)
        precip_colors = ['#FFFFFF', '#E6F3FF', '#CCE6FF', '#99CCFF', '#6699FF', '#0066CC', '#003366']
        self.precip_cmap = LinearSegmentedColormap.from_list('precipitation', precip_colors)
        
        # Wind colormap (green-yellow-red)
        wind_colors = ['#006600', '#66CC00', '#CCFF00', '#FFFF00', '#FF9900', '#FF3300', '#990000']
        self.wind_cmap = LinearSegmentedColormap.from_list('wind', wind_colors)
    
    def plot_rwanda_weather_map(self, 
                               data: np.ndarray, 
                               variable: str, 
                               title: str = None,
                               save_path: Optional[str] = None,
                               add_topography: bool = True):
        """
        Plot weather data over Rwanda with geographic context
        
        Args:
            data: 2D array of weather data (lat, lon)
            variable: Variable name ('2t', 'tp', '10u', '10v', 'msl')
            title: Plot title
            save_path: Path to save figure
            add_topography: Whether to add topography contours
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Select appropriate colormap and settings
        if variable == '2t':
            cmap = self.temp_cmap
            vmin, vmax = data.min(), data.max()
            units = '°C' if data.max() < 100 else 'K'
            label = f'Temperature ({units})'
        elif variable == 'tp':
            cmap = self.precip_cmap
            vmin, vmax = 0, np.percentile(data, 95)  # Use 95th percentile to handle outliers
            units = 'mm/day'
            label = f'Precipitation ({units})'
        elif variable in ['10u', '10v']:
            cmap = self.wind_cmap
            vmin, vmax = -np.abs(data).max(), np.abs(data).max()
            units = 'm/s'
            label = f'Wind Speed ({units})'
        elif variable in ['msl', 'sp']:
            cmap = 'viridis'
            vmin, vmax = data.min(), data.max()
            units = 'Pa'
            label = f'Pressure ({units})'
        else:
            cmap = 'viridis'
            vmin, vmax = data.min(), data.max()
            units = ''
            label = variable.upper()
        
        # Create coordinate arrays for Rwanda
        lats = np.linspace(self.rwanda_bounds['south'], self.rwanda_bounds['north'], data.shape[0])
        lons = np.linspace(self.rwanda_bounds['west'], self.rwanda_bounds['east'], data.shape[1])
        
        # Plot data
        im = ax.imshow(data, extent=[lons[0], lons[-1], lats[0], lats[-1]], 
                      cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label=label, shrink=0.8)
        
        # Add topography contours if requested
        if add_topography:
            self._add_topography_contours(ax, lats, lons)
        
        # Add geographic features
        self._add_geographic_features(ax)
        
        # Set labels and title
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Rwanda {label}', fontsize=14, fontweight='bold')
        
        # Set bounds
        ax.set_xlim(self.rwanda_bounds['west'], self.rwanda_bounds['east'])
        ax.set_ylim(self.rwanda_bounds['south'], self.rwanda_bounds['north'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Weather map saved: {save_path}")
        
        plt.show()
    
    def plot_forecast_comparison(self, 
                               predictions: np.ndarray, 
                               targets: np.ndarray,
                               variable: str,
                               lead_times: List[int],
                               save_path: Optional[str] = None):
        """
        Plot comparison between predictions and targets for different lead times
        """
        n_times = min(len(lead_times), predictions.shape[0])
        
        fig, axes = plt.subplots(3, n_times, figsize=(4*n_times, 12))
        if n_times == 1:
            axes = axes.reshape(-1, 1)
        
        for i, lead_time in enumerate(lead_times[:n_times]):
            if i >= predictions.shape[0]:
                break
            
            pred_data = predictions[i]
            target_data = targets[i] if i < targets.shape[0] else predictions[i]
            
            # Determine common scale
            vmin = min(pred_data.min(), target_data.min())
            vmax = max(pred_data.max(), target_data.max())
            
            # Plot target
            im1 = axes[0, i].imshow(target_data, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[0, i].set_title(f'Observed\n{lead_time}h lead time')
            if i == 0:
                axes[0, i].set_ylabel('Latitude')
            
            # Plot prediction
            im2 = axes[1, i].imshow(pred_data, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            axes[1, i].set_title(f'Predicted\n{lead_time}h lead time')
            if i == 0:
                axes[1, i].set_ylabel('Latitude')
            
            # Plot difference
            diff = pred_data - target_data
            diff_max = max(abs(diff.min()), abs(diff.max()))
            im3 = axes[2, i].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, origin='lower')
            axes[2, i].set_title(f'Difference\n(Pred - Obs)')
            axes[2, i].set_xlabel('Longitude')
            if i == 0:
                axes[2, i].set_ylabel('Latitude')
            
            # Add colorbars
            if i == n_times - 1:
                plt.colorbar(im1, ax=axes[0, i])
                plt.colorbar(im2, ax=axes[1, i])
                plt.colorbar(im3, ax=axes[2, i])
        
        plt.suptitle(f'Rwanda {variable.upper()} Forecast Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Forecast comparison saved: {save_path}")
        
        plt.show()
    
    def plot_seasonal_climatology(self, 
                                 data: xr.Dataset, 
                                 variable: str,
                                 save_path: Optional[str] = None):
        """Plot seasonal climatology for Rwanda"""
        # Calculate seasonal means
        seasonal_data = data[variable].groupby('time.season').mean()
        
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        season_names = ['Dec-Jan-Feb', 'Mar-Apr-May', 'Jun-Jul-Aug', 'Sep-Oct-Nov']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (season, season_name) in enumerate(zip(seasons, season_names)):
            if season in seasonal_data.season.values:
                season_data = seasonal_data.sel(season=season).values
                
                # Plot
                im = axes[i].imshow(season_data, cmap='viridis', origin='lower', aspect='auto')
                axes[i].set_title(f'{season_name} ({season})')
                plt.colorbar(im, ax=axes[i])
                
                # Add geographic features
                self._add_geographic_features(axes[i])
        
        plt.suptitle(f'Rwanda {variable.upper()} Seasonal Climatology', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Seasonal climatology saved: {save_path}")
        
        plt.show()
    
    def _add_topography_contours(self, ax, lats: np.ndarray, lons: np.ndarray):
        """Add simplified topography contours for Rwanda"""
        # Create simplified elevation model
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Rwanda topography (simplified): higher in west (volcanic), lower in east
        elevation = 2000 - 800 * (lon_grid - lons[0]) / (lons[-1] - lons[0])
        
        # Add some variation for Lake Kivu region (northwest)
        kivu_mask = (lat_grid > -2.0) & (lon_grid < 29.5)
        elevation[kivu_mask] += 200
        
        # Add contour lines
        contours = ax.contour(lon_grid, lat_grid, elevation, 
                             levels=[1000, 1500, 2000, 2500], 
                             colors='black', alpha=0.3, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%dm')
    
    def _add_geographic_features(self, ax):
        """Add basic geographic features for Rwanda"""
        # Add major cities (approximate locations)
        cities = {
            'Kigali': (-1.944, 30.059),
            'Butare': (-2.596, 29.739),
            'Gitarama': (-2.076, 29.756),
            'Ruhengeri': (-1.499, 29.633)
        }
        
        for city, (lat, lon) in cities.items():
            ax.plot(lon, lat, 'ko', markersize=4)
            ax.annotate(city, (lon, lat), xytext=(3, 3), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        # Add country border (simplified rectangle)
        border = patches.Rectangle(
            (self.rwanda_bounds['west'], self.rwanda_bounds['south']),
            self.rwanda_bounds['east'] - self.rwanda_bounds['west'],
            self.rwanda_bounds['north'] - self.rwanda_bounds['south'],
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(border)

class DataManager:
    """
    Data management utilities for Rwanda weather data
    """
    
    def __init__(self, config):
        self.config = config
        print("Data manager initialized")
    
    def download_era5_rwanda(self, 
                            variables: List[str],
                            years: List[int],
                            output_path: str,
                            api_key: Optional[str] = None):
        """
        Download ERA5 data for Rwanda region
        
        Args:
            variables: List of ERA5 variables
            years: List of years to download
            output_path: Path to save downloaded data
            api_key: CDS API key (if not in ~/.cdsapirc)
        """
        try:
            import cdsapi
        except ImportError:
            print("cdsapi not installed. Install with: pip install cdsapi")
            return
        
        # Initialize CDS client
        if api_key:
            c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/v2",
                            key=api_key)
        else:
            c = cdsapi.Client()
        
        # Rwanda bounding box
        area = [
            self.config.GEOGRAPHY['rwanda_bounds']['north'],  # North
            self.config.GEOGRAPHY['rwanda_bounds']['west'],   # West  
            self.config.GEOGRAPHY['rwanda_bounds']['south'],  # South
            self.config.GEOGRAPHY['rwanda_bounds']['east'],   # East
        ]
        
        print(f"Downloading ERA5 data for Rwanda: {area}")
        print(f"Variables: {variables}")
        print(f"Years: {years}")
        
        # Request data
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variables,
                'year': [str(year) for year in years],
                'month': [f'{month:02d}' for month in range(1, 13)],
                'day': [f'{day:02d}' for day in range(1, 32)],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': area,
                'format': 'netcdf',
            },
            output_path
        )
        
        print(f"Download completed: {output_path}")
    
    def preprocess_rwanda_data(self, 
                              input_path: str, 
                              output_path: str,
                              target_resolution: float = 0.25):
        """
        Preprocess downloaded Rwanda weather data
        
        Args:
            input_path: Path to raw data file
            output_path: Path to save preprocessed data
            target_resolution: Target spatial resolution in degrees
        """
        print(f"Preprocessing Rwanda weather data: {input_path}")
        
        # Load data
        ds = xr.open_dataset(input_path)
        
        # Crop to Rwanda bounds
        rwanda_bounds = self.config.GEOGRAPHY['rwanda_bounds']
        ds_cropped = ds.sel(
            latitude=slice(rwanda_bounds['north'], rwanda_bounds['south']),
            longitude=slice(rwanda_bounds['west'], rwanda_bounds['east'])
        )
        
        # Interpolate to target resolution if needed
        if target_resolution != ds_cropped.attrs.get('resolution', None):
            new_lats = np.arange(rwanda_bounds['south'], rwanda_bounds['north'], target_resolution)
            new_lons = np.arange(rwanda_bounds['west'], rwanda_bounds['east'], target_resolution)
            
            ds_cropped = ds_cropped.interp(latitude=new_lats, longitude=new_lons)
        
        # Quality control
        ds_cropped = self._quality_control(ds_cropped)
        
        # Add metadata
        ds_cropped.attrs.update({
            'title': 'Rwanda Weather Data - Preprocessed',
            'preprocessing_date': datetime.now().isoformat(),
            'target_resolution': target_resolution,
            'spatial_bounds': rwanda_bounds
        })
        
        # Save
        ds_cropped.to_netcdf(output_path, format='NETCDF4', engine='netcdf4')
        
        print(f"Preprocessed data saved: {output_path}")
        print(f"Data shape: {ds_cropped.dims}")
        
        return ds_cropped
    
    def _quality_control(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply quality control to weather data"""
        print("Applying quality control...")
        
        # Remove obviously bad values
        for var in ds.data_vars:
            if var == '2t':  # Temperature
                ds[var] = ds[var].where((ds[var] > 200) & (ds[var] < 350))
            elif var == 'tp':  # Precipitation
                ds[var] = ds[var].where(ds[var] >= 0)
            elif var in ['10u', '10v']:  # Wind
                ds[var] = ds[var].where(abs(ds[var]) < 100)
            elif var in ['msl', 'sp']:  # Pressure
                ds[var] = ds[var].where((ds[var] > 50000) & (ds[var] < 110000))
        
        print("Quality control completed")
        return ds
    
    def create_training_splits(self, 
                              data_path: str,
                              train_years: List[int],
                              val_years: List[int],
                              test_years: List[int],
                              output_dir: str):
        """
        Create temporal splits for training, validation, and testing
        """
        print(f"Creating training splits from {data_path}")
        
        # Load data
        ds = xr.open_dataset(data_path)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create splits
        splits = {
            'train': train_years,
            'val': val_years,
            'test': test_years
        }
        
        for split_name, years in splits.items():
            if years:
                # Select years
                ds_split = ds.sel(time=ds.time.dt.year.isin(years))
                
                # Save split
                split_path = output_path / f"rwanda_weather_{split_name}.nc"
                ds_split.to_netcdf(split_path)
                
                print(f"Saved {split_name} split: {split_path}")
                print(f"  Years: {years}")
                print(f"  Time steps: {len(ds_split.time)}")
        
        return output_path

def cleanup_memory(verbose: bool = False):
    """
    Comprehensive memory cleanup with optional verbose output
    """
    if verbose:
        print("🧹 Cleaning up memory...")
    
    # Python garbage collection
    collected = gc.collect()
    
    # PyTorch GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if verbose:
        print(f"   Collected {collected} objects")
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"   GPU memory: {memory_mb:.1f} MB")

def setup_kaggle_environment():
    """
    Setup optimal environment for Kaggle training
    """
    print("Setting up Kaggle environment...")
    
    # Set environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_BACKENDS_CUDNN_BENCHMARK'] = 'true'
    
    # Configure PyTorch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Memory settings
    cleanup_memory(verbose=True)
    
    print("Kaggle environment setup complete")

def save_training_artifacts(model: nn.Module, 
                          history: Dict, 
                          config: Any,
                          output_dir: str):
    """
    Save all training artifacts for later use
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving training artifacts to {output_dir}")
    
    # Save model state dict
    model_path = output_path / "model_state.pt"
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save configuration
    config_path = output_path / "config.json"
    try:
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")
    
    # Create summary file
    summary_path = output_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("RWANDA AURORA TRAINING SUMMARY\n")
        f.write("="*40 + "\n\n")
        f.write(f"Training completed: {datetime.now().isoformat()}\n")
        f.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Final training loss: {history.get('train_loss', [])[-1] if history.get('train_loss') else 'N/A'}\n")
        f.write(f"Final validation loss: {history.get('val_loss', [])[-1] if history.get('val_loss') else 'N/A'}\n")
        f.write(f"Total epochs: {len(history.get('train_loss', []))}\n")
    
    print("Training artifacts saved successfully")

def load_training_artifacts(artifacts_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load saved training artifacts
    
    Returns:
        Tuple of (model_state_dict, history, config)
    """
    artifacts_path = Path(artifacts_dir)
    
    print(f"Loading training artifacts from {artifacts_dir}")
    
    # Load model state
    model_state_path = artifacts_path / "model_state.pt"
    if model_state_path.exists():
        model_state = torch.load(model_state_path, map_location='cpu')
    else:
        model_state = {}
        print("Warning: Model state not found")
    
    # Load history
    history_path = artifacts_path / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {}
        print("Warning: Training history not found")
    
    # Load config
    config_path = artifacts_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
        print("Warning: Config not found")
    
    print("Training artifacts loaded successfully")
    return model_state, history, config

# Utility functions for common operations
def format_time(seconds: float) -> str:
    """Format time duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_memory(bytes_size: int) -> str:
    """Format memory size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"

def print_model_summary(model: nn.Module):
    """Print detailed model summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nMODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Memory estimate (rough)
    param_size = total_params * 4  # 4 bytes per float32
    print(f"Estimated parameter memory: {format_memory(param_size)}")
    
    print("=" * 50)

if __name__ == "__main__":
    # Test utilities
    print("Testing Rwanda Aurora utilities...")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.log_performance("test")
    
    # Test memory cleanup
    cleanup_memory(verbose=True)
    
    print("Utilities testing complete!")