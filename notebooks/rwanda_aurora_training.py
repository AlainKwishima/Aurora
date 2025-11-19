"""
Rwanda Aurora Weather Forecasting - Kaggle Training Notebook
===========================================================

Production-ready notebook for training Aurora model specifically for Rwanda
weather forecasting. Optimized for Kaggle's environment with memory management,
session handling, and comprehensive monitoring.

Run this notebook in Kaggle with GPU enabled for optimal performance.
"""

# ================================
# SETUP AND INSTALLATIONS
# ================================

# Install required packages
import subprocess
import sys

def install_packages():
    """Install required packages with error handling"""
    packages = [
        'torch==2.0.1',
        'torchvision==0.15.2', 
        'torchaudio==2.0.2',
        'xarray==2023.1.0',
        'netcdf4==1.6.2',
        'cdsapi==0.6.1',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scipy==1.10.1',
        'scikit-learn==1.3.0',
        'tqdm==4.65.0',
        'rich==13.4.2',
        'wandb==0.15.8'
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✓ {package}")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("Package installation complete!")

# Uncomment the following line if running for the first time
# install_packages()

# ================================
# IMPORTS
# ================================

import os
import gc
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Kaggle environment detection
KAGGLE_ENV = '/kaggle/working' in os.getcwd() or 'KAGGLE_URL_BASE' in os.environ
print(f"Running in Kaggle environment: {KAGGLE_ENV}")

# ================================
# CONFIGURATION
# ================================

class KaggleConfig:
    """Kaggle-specific configuration for Rwanda Aurora training"""
    
    def __init__(self):
        # Paths
        if KAGGLE_ENV:
            self.input_path = Path('/kaggle/input')
            self.working_path = Path('/kaggle/working')
            self.temp_path = Path('/kaggle/tmp')
        else:
            self.input_path = Path('./input')
            self.working_path = Path('./working')
            self.temp_path = Path('./tmp')
        
        # Create directories
        self.working_path.mkdir(exist_ok=True)
        self.temp_path.mkdir(exist_ok=True)
        
        # Model configuration
        self.model_config = {
            'model_type': 'lightning',
            'pretrained': True,
            'use_checkpointing': True,
            'mixed_precision': True,
            'memory_efficient': True
        }
        
        # Training configuration
        self.training_config = {
            'batch_size': 1,  # Small batch for Kaggle GPU memory
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'accumulation_steps': 4,  # Effective batch size = 4
            'save_every': 5,
            'val_check_interval': 100,
            'session_save_interval': 30,  # Save every 30 minutes
            'max_session_hours': 8  # Kaggle limit safety margin
        }
        
        # Data configuration
        self.data_config = {
            'use_synthetic': True,  # Use synthetic data if real data not available
            'spatial_resolution': 0.25,  # Degrees
            'temporal_resolution': 6,  # Hours
            'sequence_length': 4,  # Input sequence length
            'forecast_length': 20,  # Forecast up to 120 hours (20 * 6-hour steps)
            'num_workers': 2,
            'pin_memory': True
        }
        
        print("KaggleConfig initialized successfully")
    
    def get_device(self):
        """Get optimal device for training"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device

# Initialize configuration
config = KaggleConfig()
device = config.get_device()

# ================================
# MEMORY MANAGEMENT UTILITIES
# ================================

def cleanup_memory():
    """Comprehensive memory cleanup for Kaggle environment"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def monitor_memory():
    """Monitor current memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    # System memory (approximate)
    import psutil
    memory_percent = psutil.virtual_memory().percent
    print(f"System Memory Usage: {memory_percent:.1f}%")

def emergency_cleanup():
    """Emergency memory cleanup when running low"""
    print("🚨 Running emergency memory cleanup...")
    cleanup_memory()
    monitor_memory()

# ================================
# DATA SETUP
# ================================

def download_sample_data():
    """Download or generate sample Rwanda weather data"""
    data_path = config.working_path / "rwanda_weather_data.nc"
    
    if data_path.exists():
        print(f"Data already exists: {data_path}")
        return str(data_path)
    
    print("Generating synthetic Rwanda weather data for training...")
    
    # Rwanda geographic bounds
    lat_min, lat_max = -2.9, -1.0
    lon_min, lon_max = 28.8, 30.9
    
    # Create coordinate arrays
    lats = np.arange(lat_min, lat_max, config.data_config['spatial_resolution'])
    lons = np.arange(lon_min, lon_max, config.data_config['spatial_resolution'])
    
    # Time array (1 month of 6-hourly data)
    time_hours = 24 * 30  # 30 days
    times = pd.date_range('2023-01-01', periods=time_hours//6, freq='6H')
    
    # Generate synthetic weather variables
    np.random.seed(42)  # For reproducibility
    
    # Base climate patterns for Rwanda
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    elevation_effect = 2000 - 800 * (lon_grid - lon_min) / (lon_max - lon_min)  # Higher in west
    
    variables = {}
    
    for t, time in enumerate(times):
        # Diurnal and seasonal cycles
        hour_of_day = time.hour
        day_of_year = time.dayofyear
        
        # Temperature (influenced by elevation and time)
        base_temp = 295 + 5 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal
        diurnal_temp = 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Diurnal
        elevation_temp = -0.006 * elevation_effect  # Lapse rate
        noise_temp = np.random.normal(0, 1, elevation_effect.shape)
        
        temp_2m = base_temp + diurnal_temp + elevation_temp + noise_temp
        
        # Precipitation (more realistic patterns)
        precip_base = np.maximum(0, np.random.exponential(0.5e-6, elevation_effect.shape))
        # More rain in western highlands
        precip_topo = precip_base * (1 + 2 * (elevation_effect - elevation_effect.min()) / 
                                   (elevation_effect.max() - elevation_effect.min()))
        
        # Wind components
        wind_u = 3 + 2 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 1, elevation_effect.shape)
        wind_v = 1 + 1 * np.cos(2 * np.pi * day_of_year / 365) + np.random.normal(0, 0.5, elevation_effect.shape)
        
        # Sea level pressure
        pressure = 101325 - 120 * elevation_effect / 10 + np.random.normal(0, 200, elevation_effect.shape)
        
        # Store variables
        if t == 0:
            variables['2t'] = np.zeros((len(times), len(lats), len(lons)))
            variables['tp'] = np.zeros((len(times), len(lats), len(lons)))
            variables['10u'] = np.zeros((len(times), len(lats), len(lons)))
            variables['10v'] = np.zeros((len(times), len(lats), len(lons)))
            variables['msl'] = np.zeros((len(times), len(lats), len(lons)))
        
        variables['2t'][t] = temp_2m
        variables['tp'][t] = precip_topo
        variables['10u'][t] = wind_u
        variables['10v'][t] = wind_v
        variables['msl'][t] = pressure
    
    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={
            var_name: (('time', 'latitude', 'longitude'), var_data)
            for var_name, var_data in variables.items()
        },
        coords={
            'time': times,
            'latitude': lats,
            'longitude': lons
        },
        attrs={
            'title': 'Synthetic Rwanda Weather Data',
            'created': datetime.now().isoformat(),
            'spatial_resolution': config.data_config['spatial_resolution'],
            'temporal_resolution': config.data_config['temporal_resolution']
        }
    )
    
    # Save dataset
    print(f"Saving synthetic data to {data_path}")
    ds.to_netcdf(data_path)
    
    print(f"Generated synthetic data: {ds}")
    print(f"Data shape: {ds['2t'].shape}")
    print(f"Data size: {data_path.stat().st_size / 1e6:.1f} MB")
    
    return str(data_path)

# Download/generate data
data_file = download_sample_data()
cleanup_memory()

# ================================
# RWANDA AURORA MODEL SETUP
# ================================

# Since we can't import from local modules in Kaggle directly,
# we'll include essential components inline

class RwandaNormalization(nn.Module):
    """Rwanda-specific normalization layer"""
    
    def __init__(self):
        super().__init__()
        # Rwanda climate statistics (approximate)
        self.temp_mean = 295.0  # ~22°C
        self.temp_std = 8.0
        self.precip_mean = 1e-6  # Very small for log-normal distribution
        self.precip_std = 5e-6
        self.wind_mean = 2.0
        self.wind_std = 3.0
        self.pressure_mean = 85000  # ~850 hPa at elevation
        self.pressure_std = 2000
        
    def normalize_variable(self, data, var_name):
        """Normalize individual variable"""
        if var_name == '2t':
            return (data - self.temp_mean) / self.temp_std
        elif var_name == 'tp':
            return torch.log(data + 1e-8) / 10  # Log normalization for precipitation
        elif var_name in ['10u', '10v']:
            return (data - self.wind_mean) / self.wind_std
        elif var_name in ['msl', 'sp']:
            return (data - self.pressure_mean) / self.pressure_std
        else:
            return data
    
    def denormalize_variable(self, data, var_name):
        """Denormalize individual variable"""
        if var_name == '2t':
            return data * self.temp_std + self.temp_mean
        elif var_name == 'tp':
            return torch.exp(data * 10) - 1e-8
        elif var_name in ['10u', '10v']:
            return data * self.wind_std + self.wind_mean
        elif var_name in ['msl', 'sp']:
            return data * self.pressure_std + self.pressure_mean
        else:
            return data

class RwandaAuroraLite(nn.Module):
    """Lightweight Rwanda-specific Aurora model for Kaggle"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.input_dim = 5  # 5 surface variables
        self.hidden_dim = 256  # Reduced for memory efficiency
        self.num_layers = 4    # Reduced layers
        self.sequence_length = config.data_config['sequence_length']
        self.forecast_length = config.data_config['forecast_length']
        
        # Rwanda-specific normalization
        self.normalizer = RwandaNormalization()
        
        # Encoder (processes input sequences)
        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Decoder (generates forecasts)
        self.decoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)
        
        # Rwanda-specific components
        self.altitude_embedding = nn.Embedding(100, 32)  # Altitude-based embedding
        self.temporal_embedding = nn.Embedding(366, 32)  # Day of year embedding
        
        print(f"Initialized RwandaAuroraLite with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, x):
        """Forward pass"""
        batch_size, seq_len, height, width, channels = x.shape
        
        # Reshape for processing
        x_reshaped = x.view(batch_size * seq_len * height * width, channels)
        
        # Normalize inputs
        x_norm = torch.zeros_like(x_reshaped)
        var_names = ['2t', 'tp', '10u', '10v', 'msl']
        for i, var_name in enumerate(var_names):
            x_norm[:, i] = self.normalizer.normalize_variable(x_reshaped[:, i], var_name).squeeze()
        
        # Reshape back
        x_norm = x_norm.view(batch_size, seq_len * height * width, channels)
        
        # Encode input sequence
        encoded, (h_n, c_n) = self.encoder(x_norm)
        
        # Initialize decoder
        decoder_input = encoded[:, -1:, :]  # Last encoded state
        predictions = []
        
        # Generate forecasts
        for t in range(self.forecast_length):
            output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            pred = self.output_proj(output)
            predictions.append(pred)
            decoder_input = pred  # Use prediction as next input
        
        # Concatenate predictions
        predictions = torch.cat(predictions, dim=1)
        
        # Reshape to spatial format
        predictions = predictions.view(batch_size, self.forecast_length, height, width, channels)
        
        # Denormalize outputs
        pred_denorm = torch.zeros_like(predictions)
        for i, var_name in enumerate(var_names):
            pred_flat = predictions[..., i].view(-1)
            pred_denorm_flat = self.normalizer.denormalize_variable(pred_flat, var_name)
            pred_denorm[..., i] = pred_denorm_flat.view(predictions[..., i].shape)
        
        return pred_denorm

# ================================
# DATASET CLASS
# ================================

class RwandaWeatherDataset(torch.utils.data.Dataset):
    """PyTorch dataset for Rwanda weather data"""
    
    def __init__(self, data_path, config, mode='train'):
        self.config = config
        self.mode = mode
        
        # Load data
        print(f"Loading data from {data_path}")
        self.ds = xr.open_dataset(data_path)
        
        # Select variables
        self.variables = ['2t', 'tp', '10u', '10v', 'msl']
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        print(f"Created {len(self.sequences)} sequences for {mode}")
    
    def _create_sequences(self):
        """Create input-target sequence pairs"""
        sequences = []
        
        seq_len = self.config.data_config['sequence_length']
        forecast_len = self.config.data_config['forecast_length']
        total_len = seq_len + forecast_len
        
        data_arrays = []
        for var in self.variables:
            data = self.ds[var].values
            data_arrays.append(data)
        
        # Stack variables
        data_combined = np.stack(data_arrays, axis=-1)  # (time, lat, lon, vars)
        
        # Create sequences
        for t in range(len(data_combined) - total_len + 1):
            input_seq = data_combined[t:t+seq_len]
            target_seq = data_combined[t+seq_len:t+total_len]
            sequences.append((input_seq, target_seq))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        
        # Convert to tensors
        input_tensor = torch.FloatTensor(input_seq)
        target_tensor = torch.FloatTensor(target_seq)
        
        return input_tensor, target_tensor

# ================================
# TRAINING SETUP
# ================================

def create_dataloaders(data_path, config):
    """Create training and validation dataloaders"""
    # Create dataset
    dataset = RwandaWeatherDataset(data_path, config)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training_config['batch_size'],
        shuffle=True,
        num_workers=config.data_config['num_workers'],
        pin_memory=config.data_config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training_config['batch_size'],
        shuffle=False,
        num_workers=config.data_config['num_workers'],
        pin_memory=config.data_config['pin_memory']
    )
    
    print(f"Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
    return train_loader, val_loader

# ================================
# LOSS FUNCTION
# ================================

class RwandaLoss(nn.Module):
    """Rwanda-specific loss function"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Variable weights (prioritize temperature and precipitation)
        self.weights = {
            '2t': 2.0,   # Temperature
            'tp': 3.0,   # Precipitation (most important)
            '10u': 1.0,  # U-wind
            '10v': 1.0,  # V-wind
            'msl': 1.5   # Pressure
        }
    
    def forward(self, predictions, targets):
        """Compute weighted loss"""
        total_loss = 0
        var_names = ['2t', 'tp', '10u', '10v', 'msl']
        
        for i, var_name in enumerate(var_names):
            pred_var = predictions[..., i]
            target_var = targets[..., i]
            
            # MSE loss
            mse = self.mse_loss(pred_var, target_var)
            
            # MAE loss
            mae = self.mae_loss(pred_var, target_var)
            
            # Combined loss with variable weight
            var_loss = (mse + 0.1 * mae) * self.weights[var_name]
            total_loss += var_loss
        
        return total_loss

# ================================
# TRAINING FUNCTION
# ================================

def train_model(model, train_loader, val_loader, config, device):
    """Main training function"""
    print("Starting Rwanda Aurora training...")
    print(f"Device: {device}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Setup training components
    criterion = RwandaLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.training_config['learning_rate'],
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    scaler = torch.cuda.amp.GradScaler() if config.model_config['mixed_precision'] else None
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': []}
    
    # Session management
    session_start = datetime.now()
    last_save = datetime.now()
    
    for epoch in range(config.training_config['num_epochs']):
        # Training phase
        model.train()
        train_losses = []
        
        print(f"\nEpoch {epoch+1}/{config.training_config['num_epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    predictions = model(inputs)
                    loss = criterion(predictions, targets)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Memory cleanup
            if batch_idx % 20 == 0:
                cleanup_memory()
        
        avg_train_loss = np.mean(train_losses)
        training_history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if epoch % (config.training_config['val_check_interval'] // len(train_loader) + 1) == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            predictions = model(inputs)
                            loss = criterion(predictions, targets)
                    else:
                        predictions = model(inputs)
                        loss = criterion(predictions, targets)
                    
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            training_history['val_loss'].append(avg_val_loss)
            
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': config
                }, config.working_path / 'best_rwanda_aurora.pt')
                
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping triggered!")
                    break
        
        # Learning rate scheduling
        scheduler.step()
        
        # Session management
        current_time = datetime.now()
        session_hours = (current_time - session_start).total_seconds() / 3600
        
        # Save checkpoint every interval
        if (current_time - last_save).total_seconds() > config.training_config['session_save_interval'] * 60:
            checkpoint_path = config.working_path / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_history': training_history
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            last_save = current_time
        
        # Check session time limit
        if session_hours > config.training_config['max_session_hours']:
            print(f"Approaching session time limit ({session_hours:.1f} hours)")
            break
        
        # Memory monitoring
        if epoch % 5 == 0:
            monitor_memory()
    
    print("Training completed!")
    return training_history

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def plot_training_history(history, save_path=None):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', alpha=0.8)
    if history['val_loss']:
        # Interpolate validation loss to match training epochs
        val_epochs = np.linspace(0, len(history['train_loss'])-1, len(history['val_loss']))
        plt.plot(val_epochs, history['val_loss'], label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'][-20:], label='Recent Training Loss', alpha=0.8)
    if history['val_loss'] and len(history['val_loss']) > 0:
        plt.plot(history['val_loss'][-5:], label='Recent Validation Loss', alpha=0.8)
    plt.xlabel('Recent Epochs')
    plt.ylabel('Loss')
    plt.title('Recent Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def visualize_predictions(model, val_loader, device, save_path=None):
    """Visualize model predictions"""
    model.eval()
    
    # Get one batch for visualization
    inputs, targets = next(iter(val_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        predictions = model(inputs)
    
    # Convert to numpy
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    
    # Select first sample and middle time step
    sample_idx = 0
    time_idx = predictions.shape[1] // 2
    
    # Variable names and units
    variables = ['2t (Temperature)', 'tp (Precipitation)', '10u (Wind U)', '10v (Wind V)', 'msl (Pressure)']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for var_idx in range(5):
        # Target
        im1 = axes[0, var_idx].imshow(
            targets_np[sample_idx, time_idx, :, :, var_idx], 
            cmap='viridis', aspect='auto'
        )
        axes[0, var_idx].set_title(f'{variables[var_idx]} - Target')
        plt.colorbar(im1, ax=axes[0, var_idx])
        
        # Prediction
        im2 = axes[1, var_idx].imshow(
            predictions_np[sample_idx, time_idx, :, :, var_idx], 
            cmap='viridis', aspect='auto'
        )
        axes[1, var_idx].set_title(f'{variables[var_idx]} - Predicted')
        plt.colorbar(im2, ax=axes[1, var_idx])
    
    plt.suptitle(f'Rwanda Weather Forecast - Time Step {time_idx}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# ================================
# MAIN EXECUTION
# ================================

def main():
    """Main execution function"""
    print("🌍 Rwanda Aurora Weather Forecasting")
    print("=====================================")
    
    # Monitor initial memory
    monitor_memory()
    
    # Create dataloaders
    print("\n📊 Creating datasets...")
    train_loader, val_loader = create_dataloaders(data_file, config)
    cleanup_memory()
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = RwandaAuroraLite(config).to(device)
    
    # Enable mixed precision and memory optimizations
    if config.model_config['use_checkpointing']:
        try:
            model = torch.compile(model)  # PyTorch 2.0 optimization
            print("✓ Model compiled with torch.compile")
        except:
            print("✗ torch.compile not available, using regular model")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    monitor_memory()
    
    # Train model
    print("\n🚀 Starting training...")
    history = train_model(model, train_loader, val_loader, config, device)
    
    # Plot results
    print("\n📈 Creating visualizations...")
    plot_training_history(history, config.working_path / 'training_history.png')
    visualize_predictions(model, val_loader, device, config.working_path / 'predictions.png')
    
    # Save final results
    results = {
        'training_history': history,
        'config': {
            'model_config': config.model_config,
            'training_config': config.training_config,
            'data_config': config.data_config
        },
        'model_info': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': str(device)
        }
    }
    
    with open(config.working_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Results saved to: {config.working_path}")
    print(f"🏆 Best model saved as: best_rwanda_aurora.pt")
    
    # Final memory cleanup
    cleanup_memory()
    monitor_memory()

# Run the training
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        emergency_cleanup()
        raise

print("\n🎉 Rwanda Aurora Kaggle Notebook Ready!")
print("To run this notebook:")
print("1. Upload to Kaggle with GPU enabled")
print("2. Ensure data is available in /kaggle/input/ or let it generate synthetic data")
print("3. Run all cells")
print("4. Monitor progress and download results")