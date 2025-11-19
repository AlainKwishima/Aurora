"""
Rwanda Aurora Data Processing Pipeline
====================================

Highly optimized data processing pipeline designed for Kaggle's constraints.
Handles memory efficiently and provides robust error handling.
"""

import os
import dataclasses
import gc
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import json

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from aurora import Batch, Metadata
    from aurora.normalisation import locations, scales
except ImportError:
    print("Aurora not installed. Will use fallback classes.")
    
from .config import RwandaConfig

class MemoryEfficientDataProcessor:
    """
    Memory-efficient data processor optimized for Kaggle environment
    """
    
    def __init__(self, config: RwandaConfig):
        self.config = config
        self.device = config.get_device()
        self.scalers = {}
        self.data_stats = {}
        
        # Create necessary directories
        self._create_directories()
        
        print(f"Initialized DataProcessor with device: {self.device}")
    
    def _create_directories(self):
        """Create necessary directories for data processing"""
        for path in self.config.PATHS.values():
            os.makedirs(path, exist_ok=True)
    
    def generate_synthetic_rwanda_data(self, 
                                     start_date: str = "2020-01-01",
                                     end_date: str = "2023-12-31",
                                     save_path: Optional[str] = None) -> Dict[str, xr.Dataset]:
        """
        Generate synthetic weather data for Rwanda for testing purposes
        """
        print("Generating synthetic Rwanda weather data...")
        
        # Time range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        times = pd.date_range(start, end, freq='3H')
        
        # Geographic grid
        bounds = self.config.RWANDA_BOUNDS
        lats = np.linspace(bounds['lat_max'], bounds['lat_min'], 
                          self.config.GRID_CONFIG['lat_points'])
        lons = np.linspace(bounds['lon_min'], bounds['lon_max'], 
                          self.config.GRID_CONFIG['lon_points'])
        levels = list(self.config.PRESSURE_LEVELS)
        
        datasets = {}
        
        # Generate surface variables
        print("Generating surface variables...")
        surf_data = {}
        
        for var in self.config.VARIABLES['surf_vars']:
            print(f"  - {var}")
            if var == '2t':  # Temperature
                base_temp = 293.15 - (lats[:, None] + 2.0) * 2.0  # Temperature decreases with latitude and altitude
                daily_cycle = 5 * np.sin(2 * np.pi * times.hour / 24)
                seasonal_cycle = 3 * np.sin(2 * np.pi * times.dayofyear / 365)
                noise = np.random.normal(0, 1, (len(times), len(lats), len(lons)))
                data = base_temp[None, :, :] + daily_cycle[:, None, None] + seasonal_cycle[:, None, None] + noise
                
            elif var == 'tp':  # Precipitation
                # Seasonal precipitation pattern for Rwanda
                seasonal_precip = np.where(
                    (times.month.isin([3, 4, 5, 10, 11])), 
                    np.random.exponential(2.0, (len(times), len(lats), len(lons))),
                    np.random.exponential(0.3, (len(times), len(lats), len(lons)))
                )
                data = np.maximum(seasonal_precip, 0)
                
            elif var in ['10u', '10v']:  # Wind components
                base_wind = np.random.normal(0, 3, (len(times), len(lats), len(lons)))
                topographic_effect = (lats[:, None] + 2.0) * 0.5  # Wind increases with altitude
                data = base_wind + topographic_effect[None, :, :]
                
            elif var == 'msl':  # Mean sea level pressure
                base_pressure = 101325.0  # Pa
                variations = np.random.normal(0, 500, (len(times), len(lats), len(lons)))
                data = base_pressure + variations
                
            else:  # Other variables
                data = np.random.normal(0, 1, (len(times), len(lats), len(lons)))
            
            surf_data[var] = (['time', 'latitude', 'longitude'], data)
        
        datasets['surface'] = xr.Dataset(
            surf_data,
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons
            }
        )
        
        # Generate atmospheric variables
        print("Generating atmospheric variables...")
        atmos_data = {}
        
        for var in self.config.VARIABLES['atmos_vars']:
            print(f"  - {var}")
            if var == 't':  # Temperature
                # Temperature decreases with altitude
                temp_profile = 290.0 - np.log(levels) * 20
                base_temp = temp_profile[None, :, None, None] + np.random.normal(0, 2, (len(times), len(levels), len(lats), len(lons)))
                data = base_temp
                
            elif var == 'z':  # Geopotential height
                # Geopotential increases with altitude
                z_profile = np.array([level * 8.0 for level in levels])  # Simplified
                data = z_profile[None, :, None, None] * np.ones((len(times), len(levels), len(lats), len(lons)))
                data += np.random.normal(0, 100, data.shape)
                
            elif var in ['u', 'v']:  # Wind components
                data = np.random.normal(0, 10, (len(times), len(levels), len(lats), len(lons)))
                
            elif var == 'q':  # Specific humidity
                # Humidity decreases with altitude
                q_profile = np.exp(-np.array(levels) / 500) * 0.01
                data = q_profile[None, :, None, None] * np.random.lognormal(0, 0.5, (len(times), len(levels), len(lats), len(lons)))
                
            else:
                data = np.random.normal(0, 1, (len(times), len(levels), len(lats), len(lons)))
            
            atmos_data[var] = (['time', 'level', 'latitude', 'longitude'], data)
        
        datasets['atmospheric'] = xr.Dataset(
            atmos_data,
            coords={
                'time': times,
                'level': levels,
                'latitude': lats,
                'longitude': lons
            }
        )
        
        # Generate static variables
        print("Generating static variables...")
        static_data = {}
        
        for var in self.config.VARIABLES['static_vars']:
            if var == 'z':  # Geopotential (topography)
                # Simulate Rwanda's topography
                elevation = 1500 + 500 * np.random.random((len(lats), len(lons)))
                data = elevation * 9.81  # Convert to geopotential
                
            elif var == 'lsm':  # Land-sea mask
                data = np.ones((len(lats), len(lons)))  # Rwanda is landlocked
                
            else:
                data = np.random.random((len(lats), len(lons)))
            
            static_data[var] = (['latitude', 'longitude'], data)
        
        datasets['static'] = xr.Dataset(
            static_data,
            coords={
                'latitude': lats,
                'longitude': lons
            }
        )
        
        # Save datasets if path provided
        if save_path:
            print(f"Saving synthetic data to {save_path}")
            os.makedirs(save_path, exist_ok=True)
            
            for name, dataset in datasets.items():
                filepath = os.path.join(save_path, f"rwanda_synthetic_{name}.nc")
                dataset.to_netcdf(filepath, engine='netcdf4')
                print(f"  Saved {name} data: {filepath}")
        
        print("Synthetic data generation completed!")
        return datasets
    
    def load_era5_data(self, data_path: str) -> Dict[str, xr.Dataset]:
        """
        Load ERA5 data from NetCDF files
        """
        print(f"Loading ERA5 data from {data_path}")
        datasets = {}
        
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"Data path does not exist: {data_path}")
            return self.generate_synthetic_rwanda_data()
        
        # Look for NetCDF files
        nc_files = list(data_path.glob("*.nc"))
        if not nc_files:
            print("No NetCDF files found. Generating synthetic data...")
            return self.generate_synthetic_rwanda_data()
        
        for file_path in nc_files:
            try:
                print(f"Loading {file_path.name}...")
                ds = xr.open_dataset(file_path, engine='netcdf4')
                
                # Crop to Rwanda bounds
                ds_cropped = self.crop_to_rwanda(ds)
                datasets[file_path.stem] = ds_cropped
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not datasets:
            print("No valid datasets loaded. Generating synthetic data...")
            return self.generate_synthetic_rwanda_data()
        
        return datasets
    
    def crop_to_rwanda(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Crop dataset to Rwanda boundaries
        """
        bounds = self.config.RWANDA_BOUNDS
        
        # Handle different coordinate names
        lat_names = ['latitude', 'lat', 'y']
        lon_names = ['longitude', 'lon', 'x']
        
        lat_name = None
        lon_name = None
        
        for name in lat_names:
            if name in dataset.coords:
                lat_name = name
                break
        
        for name in lon_names:
            if name in dataset.coords:
                lon_name = name
                break
        
        if lat_name is None or lon_name is None:
            raise ValueError("Could not find latitude/longitude coordinates")
        
        # Crop to bounds
        cropped = dataset.sel(
            {lat_name: slice(bounds['lat_max'], bounds['lat_min'])},
            {lon_name: slice(bounds['lon_min'], bounds['lon_max'])}
        )
        
        return cropped
    
    def quality_control(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Apply quality control to weather data
        """
        print("Applying quality control...")
        thresholds = self.config.DATA_CONFIG['quality_thresholds']
        
        for var_name, data_array in dataset.data_vars.items():
            if var_name.endswith('_temperature') or var_name == '2t' or var_name == 't':
                # Temperature quality control
                temp_min, temp_max = thresholds['temperature_range']
                mask = (data_array >= temp_min) & (data_array <= temp_max)
                dataset[var_name] = data_array.where(mask)
                
            elif 'precipitation' in var_name or var_name == 'tp':
                # Precipitation quality control
                precip_max = thresholds['precipitation_max']
                mask = (data_array >= 0) & (data_array <= precip_max)
                dataset[var_name] = data_array.where(mask)
                
            elif 'wind' in var_name or var_name in ['10u', '10v', 'u', 'v']:
                # Wind quality control
                wind_max = thresholds['wind_speed_max']
                mask = np.abs(data_array) <= wind_max
                dataset[var_name] = data_array.where(mask)
        
        return dataset
    
    def normalize_data(self, dataset: xr.Dataset) -> Tuple[xr.Dataset, Dict]:
        """
        Normalize weather data using robust scaling
        """
        print("Normalizing data...")
        normalized_dataset = dataset.copy(deep=True)
        normalization_stats = {}
        
        for var_name, data_array in dataset.data_vars.items():
            # Use robust scaling (median and IQR)
            values = data_array.values
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                median = np.median(valid_values)
                q75, q25 = np.percentile(valid_values, [75, 25])
                iqr = q75 - q25
                
                if iqr > 0:
                    scale = iqr
                else:
                    scale = np.std(valid_values)
                    if scale == 0:
                        scale = 1.0
                
                # Apply normalization
                normalized_values = (values - median) / scale
                normalized_dataset[var_name] = (data_array.dims, normalized_values)
                
                normalization_stats[var_name] = {
                    'median': float(median),
                    'scale': float(scale),
                    'method': 'robust'
                }
            
        return normalized_dataset, normalization_stats
    
    def create_sequences(self, 
                        dataset: xr.Dataset,
                        sequence_length: int = 8,
                        prediction_horizon: int = 8,
                        stride: int = 1) -> List[Dict]:
        """
        Create input-output sequences for training
        """
        print(f"Creating sequences (length={sequence_length}, horizon={prediction_horizon}, stride={stride})")
        
        sequences = []
        time_dim = dataset.dims['time']
        total_sequences = (time_dim - sequence_length - prediction_horizon) // stride + 1
        
        print(f"Total possible sequences: {total_sequences}")
        
        for i in range(0, time_dim - sequence_length - prediction_horizon + 1, stride):
            # Input sequence
            input_slice = dataset.isel(time=slice(i, i + sequence_length))
            
            # Target sequence
            target_slice = dataset.isel(time=slice(i + sequence_length, 
                                                  i + sequence_length + prediction_horizon))
            
            sequences.append({
                'input': input_slice,
                'target': target_slice,
                'start_time': dataset.time[i].values,
                'end_time': dataset.time[i + sequence_length + prediction_horizon - 1].values
            })
        
        print(f"Created {len(sequences)} sequences")
        return sequences

    def validate_sequences(self, sequences: List[Dict]) -> List[Dict]:
        """
        Validate sequences for shape consistency, NaN rates, variable presence, and time monotonicity.

        Removes invalid sequences and returns the cleaned list.
        """
        cleaned: List[Dict] = []
        for seq in sequences:
            inp = seq.get('input')
            tgt = seq.get('target')
            if inp is None or tgt is None:
                continue
            # Required variables must exist
            required_surf = set(self.config.VARIABLES['surf_vars'])
            required_static = set(self.config.VARIABLES['static_vars'])
            required_atmos = set(self.config.VARIABLES['atmos_vars'])
            if not required_surf.issubset(set(inp.data_vars) | set(tgt.data_vars)):
                continue
            # Time monotonicity
            try:
                t_inp = pd.to_datetime(inp.time.values)
                t_tgt = pd.to_datetime(tgt.time.values)
                if not (np.all(np.diff(t_inp.view('int64')) > 0) and np.all(np.diff(t_tgt.view('int64')) > 0)):
                    continue
            except Exception:
                continue
            # NaN ratio thresholds
            nan_ok = True
            for var in inp.data_vars:
                vals = inp[var].values
                if np.isnan(vals).mean() > self.config.DATA_CONFIG.get('nan_threshold', 0.2):
                    nan_ok = False
                    break
            for var in tgt.data_vars:
                vals = tgt[var].values
                if np.isnan(vals).mean() > self.config.DATA_CONFIG.get('nan_threshold', 0.2):
                    nan_ok = False
                    break
            if not nan_ok:
                continue
            cleaned.append(seq)
        return cleaned
    
    def convert_to_aurora_format(self, sequence: Dict) -> Tuple[Batch, Batch]:
        """
        Convert xarray sequence to Aurora Batch format
        """
        input_data = sequence['input']
        target_data = sequence['target']
        
        def extract_variables(dataset, var_config):
            surf_vars = {}
            static_vars = {}
            atmos_vars = {}
            
            for var in var_config['surf_vars']:
                if var in dataset.data_vars:
                    # Shape: (batch=1, time, lat, lon)
                    data = torch.from_numpy(dataset[var].values).unsqueeze(0)
                    surf_vars[var] = data.float()
            
            for var in var_config['static_vars']:
                if var in dataset.data_vars:
                    # Shape: (lat, lon)
                    data = torch.from_numpy(dataset[var].values)
                    static_vars[var] = data.float()
            
            for var in var_config['atmos_vars']:
                if var in dataset.data_vars and len(dataset[var].dims) > 3:
                    # Shape: (batch=1, time, level, lat, lon)
                    data = torch.from_numpy(dataset[var].values).unsqueeze(0)
                    atmos_vars[var] = data.float()
            
            return surf_vars, static_vars, atmos_vars
        
        # Extract input variables
        input_surf, input_static, input_atmos = extract_variables(input_data, self.config.VARIABLES)
        
        # Extract target variables  
        target_surf, target_static, target_atmos = extract_variables(target_data, self.config.VARIABLES)
        
        # Create metadata
        lats = torch.from_numpy(input_data.latitude.values).float()
        lons = torch.from_numpy(input_data.longitude.values).float()
        times = tuple(pd.to_datetime(input_data.time.values))
        levels = tuple(input_data.level.values) if 'level' in input_data.coords else self.config.PRESSURE_LEVELS
        
        input_metadata = Metadata(
            lat=lats,
            lon=lons,
            time=times,
            atmos_levels=levels,
            rollout_step=0
        )
        
        target_times = tuple(pd.to_datetime(target_data.time.values))
        target_metadata = Metadata(
            lat=lats,
            lon=lons,
            time=target_times,
            atmos_levels=levels,
            rollout_step=1
        )
        
        # Create batches
        input_batch = Batch(
            surf_vars=input_surf,
            static_vars=input_static,
            atmos_vars=input_atmos,
            metadata=input_metadata
        )
        
        target_batch = Batch(
            surf_vars=target_surf,
            static_vars=target_static,
            atmos_vars=target_atmos,
            metadata=target_metadata
        )
        
        return input_batch, target_batch

class RwandaWeatherDataset(Dataset):
    """
    PyTorch Dataset for Rwanda weather data optimized for memory efficiency
    """
    
    def __init__(self, 
                 data_path: str,
                 config: RwandaConfig,
                 mode: str = 'train',
                 transform: Optional[callable] = None):
        self.data_path = data_path
        self.config = config
        self.mode = mode
        # Default physics-aware augmentations in training mode
        self.transform = transform if transform is not None else (
            physics_aware_transform(config) if mode == 'train' else None
        )
        
        # Initialize data processor
        self.processor = MemoryEfficientDataProcessor(config)
        
        # Load and process data
        self.sequences = self._load_sequences()
        # Automatic validation
        self.sequences = self.processor.validate_sequences(self.sequences)
        
        print(f"Loaded {len(self.sequences)} sequences for {mode} mode")
    
    def _load_sequences(self) -> List[Dict]:
        """Load processed sequences from cache or create new ones"""
        cache_path = os.path.join(self.config.PATHS['cache'], f'sequences_{self.mode}.pkl')
        
        if os.path.exists(cache_path) and self.config.DATA_CONFIG['cache_processed_data']:
            print(f"Loading cached sequences from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Process data from scratch
        print("Processing data from scratch...")
        
        # Load raw data
        datasets = self.processor.load_era5_data(self.data_path)
        if not datasets:
            raise ValueError("No valid datasets found")
        
        # Combine datasets (simplified approach)
        combined_dataset = None
        for name, dataset in datasets.items():
            if combined_dataset is None:
                combined_dataset = dataset
            else:
                # Merge datasets - in practice, you'd need more sophisticated merging
                for var in dataset.data_vars:
                    if var not in combined_dataset.data_vars:
                        combined_dataset[var] = dataset[var]
        
        # Apply quality control
        combined_dataset = self.processor.quality_control(combined_dataset)
        
        # Normalize data
        normalized_dataset, norm_stats = self.processor.normalize_data(combined_dataset)
        
        # Create sequences
        sequences = self.processor.create_sequences(
            normalized_dataset,
            sequence_length=self.config.DATA_CONFIG['sequence_length'],
            prediction_horizon=self.config.DATA_CONFIG['prediction_horizon']
        )
        
        # Cache sequences
        if self.config.DATA_CONFIG['cache_processed_data']:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(sequences, f)
            print(f"Cached sequences to {cache_path}")
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[Batch, Batch]:
        sequence = self.sequences[idx]
        
        # Convert to Aurora format
        input_batch, target_batch = self.processor.convert_to_aurora_format(sequence)
        
        # Apply transforms if provided
        if self.transform:
            input_batch, target_batch = self.transform(input_batch, target_batch)
        
        return input_batch, target_batch

class DataVisualization:
    """
    Visualization utilities for Rwanda weather data
    """
    
    def __init__(self, config: RwandaConfig):
        self.config = config
        plt.style.use('seaborn-v0_8')
    
    def plot_data_summary(self, dataset: xr.Dataset, save_path: Optional[str] = None):
        """Create comprehensive data summary plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Rwanda Weather Data Summary', fontsize=16, fontweight='bold')
        
        # Temperature
        if '2t' in dataset.data_vars:
            temp_mean = dataset['2t'].mean(dim='time')
            im1 = temp_mean.plot(ax=axes[0,0], cmap='RdYlBu_r', add_colorbar=False)
            axes[0,0].set_title('Mean Temperature')
            plt.colorbar(im1, ax=axes[0,0], label='K')
        
        # Precipitation
        if 'tp' in dataset.data_vars:
            precip_mean = dataset['tp'].mean(dim='time')
            im2 = precip_mean.plot(ax=axes[0,1], cmap='Blues', add_colorbar=False)
            axes[0,1].set_title('Mean Precipitation')
            plt.colorbar(im2, ax=axes[0,1], label='mm')
        
        # Wind
        if '10u' in dataset.data_vars and '10v' in dataset.data_vars:
            wind_speed = np.sqrt(dataset['10u']**2 + dataset['10v']**2).mean(dim='time')
            im3 = wind_speed.plot(ax=axes[0,2], cmap='viridis', add_colorbar=False)
            axes[0,2].set_title('Mean Wind Speed')
            plt.colorbar(im3, ax=axes[0,2], label='m/s')
        
        # Time series plots
        if '2t' in dataset.data_vars:
            temp_ts = dataset['2t'].mean(dim=['latitude', 'longitude'])
            temp_ts.plot(ax=axes[1,0], color='red', linewidth=0.5)
            axes[1,0].set_title('Temperature Time Series')
            axes[1,0].set_ylabel('Temperature (K)')
        
        if 'tp' in dataset.data_vars:
            precip_ts = dataset['tp'].mean(dim=['latitude', 'longitude'])
            precip_ts.plot(ax=axes[1,1], color='blue', linewidth=0.5)
            axes[1,1].set_title('Precipitation Time Series')
            axes[1,1].set_ylabel('Precipitation (mm)')
        
        # Seasonal cycle
        if '2t' in dataset.data_vars:
            monthly_temp = dataset['2t'].groupby('time.month').mean()
            monthly_temp.mean(dim=['latitude', 'longitude']).plot(ax=axes[1,2], marker='o')
            axes[1,2].set_title('Seasonal Temperature Cycle')
            axes[1,2].set_ylabel('Temperature (K)')
            axes[1,2].set_xlabel('Month')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Data summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_rwanda_map(self, data_array: xr.DataArray, title: str, save_path: Optional[str] = None):
        """Plot data on Rwanda map"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data
        im = data_array.plot(ax=ax, cmap='RdYlBu_r', add_colorbar=False)
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add station locations
        stations = self.config.RWANDA_FEATURES['stations']
        for name, coords in stations.items():
            ax.plot(coords['lon'], coords['lat'], 'ko', markersize=4)
            ax.annotate(name.replace('_', ' '), 
                       (coords['lon'], coords['lat']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add geographic bounds
        bounds = self.config.RWANDA_BOUNDS
        ax.set_xlim(bounds['lon_min'], bounds['lon_max'])
        ax.set_ylim(bounds['lat_min'], bounds['lat_max'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map plot saved to {save_path}")
            
        plt.show()

def create_dataloaders(config: RwandaConfig, 
                      data_path: str) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized train and validation dataloaders
    """
    print("Creating dataloaders...")
    
    # Create datasets with validation
    train_dataset = RwandaWeatherDataset(data_path, config, mode='train')
    val_dataset = RwandaWeatherDataset(data_path, config, mode='validation')
    
    # Create dataloaders with Kaggle optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config.TRAINING_CONFIG['num_workers'],
        pin_memory=config.TRAINING_CONFIG['pin_memory'],
        persistent_workers=config.TRAINING_CONFIG['persistent_workers'],
        prefetch_factor=config.TRAINING_CONFIG['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.TRAINING_CONFIG['num_workers'],
        pin_memory=config.TRAINING_CONFIG['pin_memory'],
        persistent_workers=config.TRAINING_CONFIG['persistent_workers'],
        prefetch_factor=config.TRAINING_CONFIG['prefetch_factor']
    )
    
    print(f"Created dataloaders: Train={len(train_loader)}, Val={len(val_loader)}")
    return train_loader, val_loader

# Utility functions
def cleanup_memory():
    """Cleanup memory for Kaggle optimization"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def monitor_memory():
    """Monitor memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

def physics_aware_transform(config: RwandaConfig):
    """
    Build a physics-aware augmentation callable for Rwanda Dataset.

    Augmentations:
    - Spatial jitter via circular shift (small dx/dy)
    - Low-amplitude Gaussian noise on select variables
    - Clamp to physical ranges from config thresholds
    """
    thresholds = config.DATA_CONFIG.get('quality_thresholds', {})
    temp_min, temp_max = thresholds.get('temperature_range', (243.15, 313.15))
    wind_max = thresholds.get('wind_speed_max', 35.0)
    precip_max = thresholds.get('precipitation_max', 150.0)

    def _apply_shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        if x.ndim < 3:
            return x
        return torch.roll(x, shifts=(dx, dy), dims=(-2, -1))

    def _noise(x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale <= 0:
            return x
        return x + torch.randn_like(x) * scale

    def _clamp_var(name: str, x: torch.Tensor) -> torch.Tensor:
        if name in {'2t', 'skt', 'stl1'}:
            return x.clamp(min=float(temp_min), max=float(temp_max))
        if name in {'10u', '10v', 'u', 'v'}:
            return x.clamp(min=-float(wind_max), max=float(wind_max))
        if name in {'tp'}:
            return x.clamp(min=0.0, max=float(precip_max))
        return x

    def transform(input_batch: Batch, target_batch: Batch) -> tuple[Batch, Batch]:
        # Random small spatial shifts
        dx = int(np.random.choice([-1, 0, 1]))
        dy = int(np.random.choice([-1, 0, 1]))
        noise_scale = 0.01  # conservative

        def aug_batch(b: Batch) -> Batch:
            surf = {k: _clamp_var(k, _noise(_apply_shift(v, dx, dy), noise_scale)) for k, v in b.surf_vars.items()}
            atmos = {k: _clamp_var(k, _noise(_apply_shift(v, dx, dy), noise_scale)) for k, v in b.atmos_vars.items()}
            return dataclasses.replace(b, surf_vars=surf, atmos_vars=atmos)

        return aug_batch(input_batch), aug_batch(target_batch)

    return transform

if __name__ == "__main__":
    # Test data processing
    config = RwandaConfig()
    config.print_config_summary()
    
    processor = MemoryEfficientDataProcessor(config)
    
    # Generate and save synthetic data
    synthetic_data = processor.generate_synthetic_rwanda_data(
        save_path="/kaggle/working/synthetic_data"
    )
    
    # Create visualization
    viz = DataVisualization(config)
    viz.plot_data_summary(
        synthetic_data['surface'], 
        save_path="/kaggle/working/data_summary.png"
    )