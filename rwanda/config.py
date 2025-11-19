"""
Rwanda-Specific Aurora Configuration
===================================

Optimized configuration for training Aurora specifically for Rwanda weather forecasting.
Designed for efficient training on Kaggle with memory and session constraints.
"""

import torch
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np

class RwandaConfig:
    """Centralized configuration for Rwanda Aurora training"""
    
    # =================== GEOGRAPHIC CONFIGURATION ===================
    RWANDA_BOUNDS = {
        'lat_min': -2.84,    # Cyangugu (South)
        'lat_max': -1.05,    # Gatuna (North)  
        'lon_min': 28.86,    # Rusizi (West)
        'lon_max': 30.90     # Kagitumba (East)
    }
    
    # High-resolution grid optimized for Rwanda's size (~26,338 km²)
    GRID_CONFIG = {
        'native_resolution': 0.25,      # ERA5 native resolution
        'target_resolution': 0.1,       # Target resolution (~11km)
        'lat_points': 18,               # Number of latitude points
        'lon_points': 21,               # Number of longitude points
        'patch_size': 2,                # Optimized for small area
        'overlap': 0.1                  # Grid overlap for stability
    }
    
    # =================== METEOROLOGICAL VARIABLES ===================
    VARIABLES = {
        'surf_vars': (
            '2t',       # 2m temperature - critical for Rwanda's highland climate
            '10u',      # 10m u-wind component
            '10v',      # 10m v-wind component  
            'msl',      # Mean sea level pressure
            'tp',       # Total precipitation - crucial for agriculture
            'sp',       # Surface pressure - important at altitude
            '2d',       # 2m dewpoint - humidity patterns
            'tcc',      # Total cloud cover - solar applications
            'skt',      # Skin temperature - heat island effects
            'stl1',     # Soil temperature layer 1
        ),
        
        'static_vars': (
            'lsm',      # Land-sea mask
            'z',        # Geopotential - crucial for Rwanda's terrain
            'slt',      # Soil type
            'sdfor',    # Std dev of filtered orography
            'cvl',      # Low vegetation cover
            'cvh',      # High vegetation cover
            'al',       # Albedo - important for energy balance
            'anor',     # Angle of sub-gridscale orography
            'isor',     # Anisotropy of sub-gridscale orography
            'slor',     # Slope of sub-gridscale orography
        ),
        
        'atmos_vars': (
            'z',        # Geopotential height
            'u',        # U-component wind
            'v',        # V-component wind
            't',        # Temperature
            'q',        # Specific humidity
            'w',        # Vertical velocity - convection
            'vo',       # Vorticity - atmospheric dynamics
            'pv',       # Potential vorticity
            'd',        # Divergence
            'r',        # Relative humidity
        )
    }
    
    # Pressure levels optimized for Rwanda's elevation (1000-4500m)
    PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    
    # =================== MODEL ARCHITECTURE ===================
    MODEL_CONFIG = {
        # Core architecture - optimized for Rwanda scale
        'patch_size': 2,                    # Small patches for local detail
        'embed_dim': 384,                   # Balanced embedding dimension
        'num_heads': 12,                    # Attention heads
        'mlp_ratio': 3.0,                   # MLP expansion ratio
        
        # Encoder configuration
        'encoder_depths': (3, 6, 3),       # Efficient encoder depth
        'encoder_num_heads': (6, 12, 24),  # Progressive attention
        'enc_depth': 2,                     # Perceiver encoder depth
        
        # Decoder configuration  
        'decoder_depths': (3, 6, 3),       # Matching decoder
        'decoder_num_heads': (24, 12, 6),  # Reverse attention
        'dec_depth': 2,                     # Perceiver decoder depth
        'dec_mlp_ratio': 2.0,               # Decoder MLP ratio
        
        # Regularization
        'drop_rate': 0.1,                   # Dropout rate
        'drop_path': 0.15,                  # Stochastic depth
        
        # Training optimizations
        'use_lora': True,                   # Low-rank adaptation
        'lora_steps': 20,                   # LoRA training steps
        'lora_mode': 'single',              # LoRA mode
        'stabilise_level_agg': True,        # Gradient stability
        'bf16_mode': True,                  # Memory optimization
        'autocast': True,                   # Mixed precision
        
        # Temporal configuration
        'max_history_size': 4,              # 4-step history
        'timestep': timedelta(hours=3),     # 3-hourly predictions
        
        # Rwanda-specific features
        'level_condition': PRESSURE_LEVELS,  # Pressure level conditioning
        'dynamic_vars': True,               # Dynamic variables
        'window_size': (2, 4, 6),          # 3D window size
    }
    
    # =================== TRAINING CONFIGURATION ===================
    TRAINING_CONFIG = {
        # Basic training parameters
        'batch_size': 1,                    # Kaggle GPU memory limit
        'num_epochs': 80,                   # Extended training
        'learning_rate': 2e-4,              # Optimized LR
        'weight_decay': 0.01,               # L2 regularization
        'warmup_epochs': 5,                 # Learning rate warmup
        
        # Advanced optimization
        'optimizer': 'AdamW',               # Optimizer choice
        'scheduler': 'CosineAnnealingWarmRestarts',  # LR scheduling
        'T_0': 15,                          # Cosine restart period
        'T_mult': 2,                        # Period multiplication
        'eta_min': 1e-6,                    # Minimum learning rate
        
        # Gradient management
        'gradient_clip_norm': 1.0,          # Gradient clipping
        'accumulation_steps': 4,            # Gradient accumulation
        'use_amp': True,                    # Automatic mixed precision
        
        # Validation and checkpointing
        'validation_split': 0.15,           # Small validation set
        'val_check_interval': 50,           # Validation frequency
        'save_every': 10,                   # Checkpoint frequency
        'early_stopping_patience': 15,     # Early stopping
        'early_stopping_delta': 1e-4,      # Minimum improvement
        
        # Data loading
        'num_workers': 2,                   # DataLoader workers
        'pin_memory': True,                 # Memory pinning
        'persistent_workers': True,         # Worker persistence
        'prefetch_factor': 2,               # Prefetch batches
        
        # Kaggle-specific optimizations
        'session_save_interval': 100,      # Session state saving
        'memory_cleanup_interval': 50,     # Memory cleanup
        'progress_update_interval': 10,    # Progress updates
    }
    
    # =================== DATA CONFIGURATION ===================
    DATA_CONFIG = {
        # Temporal coverage
        'start_date': '2018-01-01',         # Training start
        'end_date': '2024-01-01',           # Training end
        'test_start': '2024-01-01',         # Test period start
        'test_end': '2024-12-31',           # Test period end
        'time_step_hours': 3,               # Temporal resolution
        'sequence_length': 8,               # Input sequence length
        'prediction_horizon': 24,           # Prediction horizon (8 steps)
        
        # Data quality control
        'quality_thresholds': {
            'temperature_range': (243.15, 313.15),    # -30°C to 40°C
            'precipitation_max': 150.0,               # mm/day
            'wind_speed_max': 35.0,                   # m/s
            'pressure_range': (850.0, 1050.0),       # hPa at altitude
            'humidity_range': (0.0, 1.0),            # kg/kg
            'missing_data_threshold': 0.1,           # 10% max missing
        },
        
        # Preprocessing options
        'interpolation_method': 'linear',           # Gap filling
        'smoothing_window': 3,                      # Temporal smoothing
        'outlier_detection': 'iqr',                # Outlier method
        'outlier_factor': 2.5,                     # IQR multiplier
        
        # Normalization strategy
        'normalization_method': 'robust',          # Robust scaling
        'seasonal_normalization': True,            # Seasonal adjustment
        'altitude_correction': True,               # Altitude-based correction
        
        # Memory optimization
        'chunk_size': 168,                         # Weekly chunks (56 * 3h)
        'cache_processed_data': True,              # Cache processing
        'use_memory_mapping': True,                # Memory-mapped files
        'compression_level': 6,                    # Data compression
    }
    
    # =================== LOSS CONFIGURATION ===================
    LOSS_CONFIG = {
        # Variable-specific weights (Rwanda priorities)
        'variable_weights': {
            'temperature': 2.5,     # Critical for highlands
            'precipitation': 4.0,   # Most important for agriculture
            'wind_u': 1.0,         # Standard weight
            'wind_v': 1.0,         # Standard weight
            'pressure': 1.5,        # Important at altitude
            'humidity': 2.0,        # Health and comfort
            'cloud_cover': 1.2,     # Solar applications
            'skin_temp': 1.3,       # Urban heat island
        },
        
        # Loss function configuration
        'primary_loss': 'mse',              # Mean squared error
        'auxiliary_losses': ['mae', 'huber'], # Additional losses
        'auxiliary_weights': [0.3, 0.2],    # Auxiliary loss weights
        'temporal_weighting': True,         # Weight by prediction time
        'spatial_weighting': True,          # Weight by location
        'extreme_event_bonus': 2.0,         # Extra weight for extremes
        
        # Advanced loss features
        'gradient_penalty': 0.1,            # Gradient penalty weight
        'spectral_loss_weight': 0.05,       # Spectral loss
        'physics_loss_weight': 0.1,         # Physics-informed loss
    }
    
    # =================== EVALUATION METRICS ===================
    METRICS_CONFIG = {
        # Standard meteorological metrics
        'primary_metrics': [
            'rmse', 'mae', 'bias', 'correlation', 'skill_score'
        ],
        
        # Forecast-specific metrics
        'forecast_metrics': [
            'continuous_ranked_probability_score',
            'brier_skill_score',
            'reliability_index',
            'resolution_index'
        ],
        
        # Rwanda-specific thresholds
        'performance_thresholds': {
            'temperature_rmse': 2.0,        # °C
            'precipitation_bias': 0.15,     # Relative bias
            'wind_mae': 2.5,               # m/s
            'pressure_rmse': 1.2,          # hPa
            'correlation_min': 0.82,       # Minimum correlation
            'skill_score_min': 0.75,       # Minimum skill score
        },
        
        # Extreme event thresholds (Rwanda-specific)
        'extreme_thresholds': {
            'heavy_precipitation': 25.0,    # mm/day
            'extreme_precipitation': 50.0,  # mm/day  
            'high_temperature': 30.0,      # °C
            'low_temperature': 10.0,       # °C
            'strong_wind': 15.0,           # m/s
        }
    }
    
    # =================== RWANDA-SPECIFIC FEATURES ===================
    RWANDA_FEATURES = {
        # Administrative regions for analysis
        'provinces': {
            'Northern': {'lat': -1.5, 'lon': 29.6},
            'Southern': {'lat': -2.3, 'lon': 29.7},
            'Eastern': {'lat': -2.0, 'lon': 30.5},
            'Western': {'lat': -2.2, 'lon': 29.3},
            'Kigali': {'lat': -1.95, 'lon': 30.06}
        },
        
        # Key meteorological stations
        'stations': {
            'Kigali_Airport': {'lat': -1.9667, 'lon': 30.1394, 'elevation': 1491},
            'Butare': {'lat': -2.6019, 'lon': 29.7397, 'elevation': 1768},
            'Ruhengeri': {'lat': -1.4986, 'lon': 29.6336, 'elevation': 1850},
            'Gisenyi': {'lat': -1.7025, 'lon': 29.2569, 'elevation': 1542},
            'Kibungo': {'lat': -2.1539, 'lon': 30.5444, 'elevation': 1245},
            'Byumba': {'lat': -1.5761, 'lon': 30.0686, 'elevation': 2240},
            'Gikongoro': {'lat': -2.4178, 'lon': 29.6847, 'elevation': 2135},
            'Kamembe': {'lat': -2.4606, 'lon': 28.9081, 'elevation': 1554}
        },
        
        # Seasonal patterns
        'seasons': {
            'long_dry_season': {
                'months': [6, 7, 8],
                'characteristics': 'minimal_rainfall',
                'avg_temp_range': (20, 26),
                'avg_precipitation': 15
            },
            'short_rainy_season': {
                'months': [9, 10, 11],
                'characteristics': 'moderate_rainfall', 
                'avg_temp_range': (19, 25),
                'avg_precipitation': 85
            },
            'short_dry_season': {
                'months': [12, 1, 2],
                'characteristics': 'dry_cool',
                'avg_temp_range': (18, 24),
                'avg_precipitation': 35
            },
            'long_rainy_season': {
                'months': [3, 4, 5],
                'characteristics': 'heavy_rainfall',
                'avg_temp_range': (19, 24), 
                'avg_precipitation': 130
            }
        },
        
        # Topographic features
        'topography': {
            'min_elevation': 950,           # Rusizi River
            'max_elevation': 4507,          # Mount Karisimbi
            'avg_elevation': 1598,          # Average elevation
            'major_peaks': ['Karisimbi', 'Bisoke', 'Muhabura'],
            'major_lakes': ['Kivu', 'Muhazi', 'Burera', 'Ruhondo'],
            'climate_zones': ['tropical_highland', 'temperate']
        }
    }
    
    # =================== VISUALIZATION CONFIGURATION ===================
    VISUALIZATION_CONFIG = {
        'color_schemes': {
            'temperature': 'RdYlBu_r',
            'precipitation': 'Blues',
            'wind': 'viridis',
            'pressure': 'plasma',
            'humidity': 'BuGn'
        },
        
        'map_settings': {
            'center_lat': -1.95,
            'center_lon': 29.9,
            'zoom_level': 8,
            'tile_layer': 'OpenStreetMap'
        },
        
        'plot_settings': {
            'figsize': (12, 8),
            'dpi': 300,
            'style': 'seaborn-v0_8',
            'font_size': 10
        }
    }
    
    # =================== PATHS AND DIRECTORIES ===================
    PATHS = {
        'data_root': '/kaggle/input',
        'output_root': '/kaggle/working',
        'checkpoints': '/kaggle/working/checkpoints',
        'logs': '/kaggle/working/logs',
        'results': '/kaggle/working/results',
        'cache': '/kaggle/working/cache',
        'temp': '/kaggle/tmp'
    }
    
    @classmethod
    def get_device(cls) -> torch.device:
        """Get the optimal device for training"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        print("=" * 60)
        print("RWANDA AURORA CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Geographic Coverage: {cls.RWANDA_BOUNDS}")
        print(f"Grid Points: {cls.GRID_CONFIG['lat_points']} x {cls.GRID_CONFIG['lon_points']}")
        print(f"Surface Variables: {len(cls.VARIABLES['surf_vars'])}")
        print(f"Atmospheric Variables: {len(cls.VARIABLES['atmos_vars'])}")
        print(f"Static Variables: {len(cls.VARIABLES['static_vars'])}")
        print(f"Pressure Levels: {len(cls.PRESSURE_LEVELS)}")
        print(f"Model Parameters: ~{cls._estimate_parameters():,.0f}")
        print(f"Training Device: {cls.get_device()}")
        print("=" * 60)
    
    @classmethod
    def _estimate_parameters(cls) -> int:
        """Estimate model parameter count"""
        embed_dim = cls.MODEL_CONFIG['embed_dim']
        # Rough estimation based on architecture
        return embed_dim * embed_dim * 50  # Simplified calculation

# Global configuration instance
config = RwandaConfig()

if __name__ == "__main__":
    config.print_config_summary()