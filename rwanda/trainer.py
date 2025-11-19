"""
Rwanda Aurora Training System
============================

Advanced training pipeline optimized for Kaggle environment with comprehensive
memory management, session handling, and monitoring capabilities.
"""

import os
import gc
import time
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from aurora import Batch, rollout
    AURORA_AVAILABLE = True
except ImportError:
    AURORA_AVAILABLE = False
    class Batch:
        pass

from .config import RwandaConfig
from .model import RwandaAurora, RwandaAuroraLightning, create_rwanda_model, save_model_checkpoint
from .data_processing import create_dataloaders, cleanup_memory, monitor_memory

class RwandaLoss(nn.Module):
    """
    Rwanda-specific loss function with weighted variables and physical constraints
    """
    
    def __init__(self, config: RwandaConfig):
        super().__init__()
        self.config = config
        self.loss_config = config.LOSS_CONFIG
        
        # Loss components
        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')
        self.huber_loss = nn.SmoothL1Loss(reduction='none')
        
        # Variable weights
        self.variable_weights = self.loss_config['variable_weights']
        
        print("Initialized Rwanda-specific loss function")
    
    def forward(self, 
                predictions: Batch, 
                targets: Batch,
                batch_metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss for Rwanda weather prediction
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            batch_metadata: Optional batch metadata for temporal weighting
        
        Returns:
            Total loss and loss components dictionary
        """
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))
        loss_components = {}
        
        # Surface variable losses
        surf_loss, surf_components = self._compute_surface_losses(predictions, targets)
        total_loss += surf_loss
        loss_components.update(surf_components)
        
        # Atmospheric variable losses
        atmos_loss, atmos_components = self._compute_atmospheric_losses(predictions, targets)
        total_loss += atmos_loss
        loss_components.update(atmos_components)
        
        # Physics-informed loss
        if self.loss_config.get('physics_loss_weight', 0) > 0:
            physics_loss = self._compute_physics_loss(predictions, targets)
            total_loss += self.loss_config['physics_loss_weight'] * physics_loss
            loss_components['physics'] = physics_loss.item()
        
        # Spectral loss
        if self.loss_config.get('spectral_loss_weight', 0) > 0:
            spectral_loss = self._compute_spectral_loss(predictions, targets)
            total_loss += self.loss_config['spectral_loss_weight'] * spectral_loss
            loss_components['spectral'] = spectral_loss.item()
        
        loss_components['total'] = total_loss.item()
        return total_loss, loss_components
    
    def _compute_surface_losses(self, predictions: Batch, targets: Batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute losses for surface variables"""
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))
        components = {}
        
        for var_name in predictions.surf_vars:
            if var_name in targets.surf_vars:
                pred = predictions.surf_vars[var_name]
                target = targets.surf_vars[var_name]
                
                # Get variable-specific weight
                var_key = self._get_variable_key(var_name)
                weight = self.variable_weights.get(var_key, 1.0)
                
                # Compute primary loss (MSE)
                mse = self.mse_loss(pred, target).mean()
                var_loss = weight * mse
                
                # Add auxiliary losses
                if 'mae' in self.loss_config.get('auxiliary_losses', []):
                    mae = self.mae_loss(pred, target).mean()
                    aux_weight = self.loss_config['auxiliary_weights'][0]
                    var_loss += weight * aux_weight * mae
                
                if 'huber' in self.loss_config.get('auxiliary_losses', []):
                    huber = self.huber_loss(pred, target).mean()
                    aux_weight = self.loss_config['auxiliary_weights'][1] if len(self.loss_config['auxiliary_weights']) > 1 else 0.2
                    var_loss += weight * aux_weight * huber
                
                # Extreme event bonus
                if self.loss_config.get('extreme_event_bonus', 0) > 0:
                    extreme_mask = self._detect_extreme_events(target, var_name)
                    if extreme_mask.sum() > 0:
                        extreme_loss = self.mse_loss(pred[extreme_mask], target[extreme_mask]).mean()
                        var_loss += self.loss_config['extreme_event_bonus'] * extreme_loss
                
                total_loss += var_loss
                components[f'surf_{var_name}'] = var_loss.item()
        
        return total_loss, components
    
    def _compute_atmospheric_losses(self, predictions: Batch, targets: Batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute losses for atmospheric variables"""
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))
        components = {}
        
        for var_name in predictions.atmos_vars:
            if var_name in targets.atmos_vars:
                pred = predictions.atmos_vars[var_name]
                target = targets.atmos_vars[var_name]
                
                # Level-weighted loss (higher weight for lower levels closer to surface)
                if len(pred.shape) == 5:  # Has level dimension
                    level_weights = torch.linspace(2.0, 1.0, pred.shape[2], device=pred.device)
                    level_weights = level_weights.view(1, 1, -1, 1, 1)
                else:
                    level_weights = 1.0
                
                # Compute weighted MSE
                var_key = self._get_variable_key(var_name)
                var_weight = self.variable_weights.get(var_key, 1.0)
                
                mse = self.mse_loss(pred, target)
                weighted_mse = (mse * level_weights).mean()
                var_loss = var_weight * weighted_mse
                
                total_loss += var_loss
                components[f'atmos_{var_name}'] = var_loss.item()
        
        return total_loss, components
    
    def _compute_physics_loss(self, predictions: Batch, targets: Batch) -> torch.Tensor:
        """Compute physics-informed loss (simplified implementation)"""
        physics_loss = torch.tensor(0.0, device=self._get_device(predictions))
        
        # Hydrostatic balance constraint
        if 'z' in predictions.atmos_vars and 't' in predictions.atmos_vars:
            # Simplified hydrostatic check
            z_pred = predictions.atmos_vars['z']
            t_pred = predictions.atmos_vars['t']
            
            if len(z_pred.shape) == 5 and z_pred.shape[2] > 1:
                # Check if geopotential increases with decreasing pressure
                z_diff = z_pred[:, :, :-1] - z_pred[:, :, 1:]  # Should be negative (decreasing)
                physics_loss += torch.relu(z_diff).mean()  # Penalize positive differences
        
        # Temperature lapse rate constraint
        if '2t' in predictions.surf_vars and 't' in predictions.atmos_vars:
            surf_temp = predictions.surf_vars['2t']
            atmos_temp = predictions.atmos_vars['t']
            
            if len(atmos_temp.shape) == 5:
                # Surface should be warmer than upper levels
                upper_temp = atmos_temp[:, :, 0]  # Highest level
                temp_diff = surf_temp - upper_temp
                physics_loss += torch.relu(-temp_diff + 10).mean()  # At least 10K difference
        
        return physics_loss
    
    def _compute_spectral_loss(self, predictions: Batch, targets: Batch) -> torch.Tensor:
        """Compute spectral loss for spatial patterns"""
        spectral_loss = torch.tensor(0.0, device=self._get_device(predictions))
        
        # Focus on temperature and precipitation spectral characteristics
        for var_name in ['2t', 'tp']:
            if (var_name in predictions.surf_vars and 
                var_name in targets.surf_vars):
                
                pred = predictions.surf_vars[var_name]
                target = targets.surf_vars[var_name]
                
                # Compute 2D FFT for spatial frequency analysis
                pred_fft = torch.fft.fft2(pred[..., -2:], dim=(-2, -1))
                target_fft = torch.fft.fft2(target[..., -2:], dim=(-2, -1))
                
                # Compare power spectra
                pred_power = torch.abs(pred_fft)**2
                target_power = torch.abs(target_fft)**2
                
                spectral_diff = self.mse_loss(pred_power, target_power).mean()
                spectral_loss += spectral_diff
        
        return spectral_loss
    
    def _detect_extreme_events(self, values: torch.Tensor, var_name: str) -> torch.Tensor:
        """Detect extreme events for bonus weighting"""
        thresholds = self.config.METRICS_CONFIG['extreme_thresholds']
        
        if var_name == 'tp':  # Precipitation
            return values > thresholds['heavy_precipitation']
        elif var_name == '2t':  # Temperature
            return (values < thresholds['low_temperature'] + 273.15) | \
                   (values > thresholds['high_temperature'] + 273.15)
        elif var_name in ['10u', '10v']:  # Wind
            return torch.abs(values) > thresholds['strong_wind']
        else:
            return torch.zeros_like(values, dtype=torch.bool)
    
    def _get_variable_key(self, var_name: str) -> str:
        """Map variable names to loss weight keys"""
        mapping = {
            '2t': 'temperature',
            'tp': 'precipitation',
            '10u': 'wind_u',
            '10v': 'wind_v',
            'msl': 'pressure',
            'sp': 'pressure',
            'q': 'humidity',
            'tcc': 'cloud_cover',
            'skt': 'skin_temp'
        }
        return mapping.get(var_name, 'default')
    
    def _get_device(self, batch: Batch) -> torch.device:
        """Get device from batch"""
        if hasattr(batch.surf_vars, '__iter__') and batch.surf_vars:
            return next(iter(batch.surf_vars.values())).device
        return torch.device('cpu')

class KaggleTrainer:
    """
    Advanced trainer optimized for Kaggle environment
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: RwandaConfig,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device):
        
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training configuration
        self.training_config = config.TRAINING_CONFIG
        
        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_mixed_precision()
        self._setup_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'memory_usage': []
        }
        
        # Kaggle-specific optimizations
        self.session_start_time = time.time()
        self.last_save_time = time.time()
        self.last_memory_cleanup = time.time()
        
        print("Initialized KaggleTrainer")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Device: {device}")
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for different components"""
        train_config = self.training_config
        
        # Group parameters with different learning rates
        param_groups = []
        
        # Rwanda-specific parameters (higher learning rate)
        rwanda_params = []
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['topographic_encoder', 'seasonal_encoder', 
                                     'regional_encoder', 'precipitation_head', 
                                     'temperature_altitude_head', 'rwanda_norm']):
                rwanda_params.append(param)
        
        if rwanda_params:
            param_groups.append({
                'params': rwanda_params,
                'lr': train_config['learning_rate'] * 2.0,  # Higher LR for new components
                'weight_decay': train_config['weight_decay'] * 0.5
            })
        
        # Encoder parameters (lower learning rate)
        encoder_params = []
        for name, param in self.model.named_parameters():
            if 'encoder' in name and param not in rwanda_params:
                encoder_params.append(param)
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': train_config['learning_rate'] * 0.1,  # Much lower LR for pretrained encoder
                'weight_decay': train_config['weight_decay']
            })
        
        # Decoder parameters (moderate learning rate)
        decoder_params = []
        for name, param in self.model.named_parameters():
            if ('decoder' in name or 'backbone' in name) and param not in rwanda_params:
                decoder_params.append(param)
        
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': train_config['learning_rate'] * 0.5,  # Moderate LR for decoder
                'weight_decay': train_config['weight_decay']
            })
        
        # Remaining parameters (standard learning rate)
        all_special_params = set(rwanda_params + encoder_params + decoder_params)
        remaining_params = [p for p in self.model.parameters() if p not in all_special_params]
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': train_config['learning_rate'],
                'weight_decay': train_config['weight_decay']
            })
        
        # Create optimizer
        if train_config['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(param_groups)
        elif train_config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(param_groups)
        else:
            self.optimizer = optim.SGD(param_groups, momentum=0.9)
        
        print(f"Setup {train_config['optimizer']} optimizer with {len(param_groups)} parameter groups")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        train_config = self.training_config
        
        if train_config['scheduler'] == 'CosineAnnealingWarmRestarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=train_config['T_0'],
                T_mult=train_config['T_mult'],
                eta_min=train_config['eta_min']
            )
        elif train_config['scheduler'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('step_size', 20),
                gamma=train_config.get('gamma', 0.5)
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=train_config.get('scheduler_patience', 5),
                factor=0.5,
                verbose=True
            )
        
        print(f"Setup {train_config['scheduler']} scheduler")
    
    def _setup_loss_function(self):
        """Setup Rwanda-specific loss function"""
        self.loss_fn = RwandaLoss(self.config)
    
    def _setup_mixed_precision(self):
        """Setup automatic mixed precision training"""
        if self.training_config.get('use_amp', False):
            self.scaler = GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        print(f"Mixed precision training: {self.use_amp}")
    
    def _setup_monitoring(self):
        """Setup training monitoring"""
        # Initialize wandb if available
        try:
            wandb.init(
                project="rwanda-aurora",
                config=dict(self.config.TRAINING_CONFIG),
                name=f"rwanda-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                dir=self.config.PATHS['logs']
            )
            self.use_wandb = True
        except Exception as e:
            print(f"Wandb not available: {e}")
            self.use_wandb = False
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop with Kaggle optimizations
        """
        print("Starting training...")
        print(f"Total epochs: {self.training_config['num_epochs']}")
        print(f"Validation every: {self.training_config['val_check_interval']} batches")
        
        try:
            for epoch in range(self.training_config['num_epochs']):
                self.current_epoch = epoch
                
                # Training epoch
                train_metrics = self._train_epoch()
                
                # Validation epoch
                if epoch % (self.training_config['val_check_interval'] // len(self.train_loader) + 1) == 0:
                    val_metrics = self._validate_epoch()
                    
                    # Update learning rate scheduler
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                    
                    # Check for early stopping
                    if self._check_early_stopping(val_metrics['loss']):
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                    
                    # Save checkpoint
                    if epoch % self.training_config['save_every'] == 0:
                        self._save_checkpoint(epoch, val_metrics)
                else:
                    val_metrics = None
                    if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()
                
                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics)
                
                # Kaggle session management
                self._manage_kaggle_session()
                
                # Memory cleanup
                if time.time() - self.last_memory_cleanup > 300:  # Every 5 minutes
                    cleanup_memory()
                    self.last_memory_cleanup = time.time()
            
            print("Training completed successfully!")
            return self.training_history
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            self._save_checkpoint(self.current_epoch, {'loss': float('inf')}, 'interrupted')
        except Exception as e:
            print(f"Training failed with error: {e}")
            self._save_checkpoint(self.current_epoch, {'loss': float('inf')}, 'error')
            raise
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, 
                   desc=f"Epoch {self.current_epoch}", 
                   leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass with gradient accumulation
            loss, loss_components = self._train_step(inputs, targets, batch_idx)
            
            total_loss += loss
            for key, value in loss_components.items():
                if key in total_components:
                    total_components[key] += value
                else:
                    total_components[key] = value
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'temp': f'{loss_components.get("surf_2t", 0):.4f}',
                'precip': f'{loss_components.get("surf_tp", 0):.4f}'
            })
            
            # Memory management
            if batch_idx % self.training_config['memory_cleanup_interval'] == 0:
                cleanup_memory()
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    def _train_step(self, inputs: Batch, targets: Batch, batch_idx: int) -> Tuple[float, Dict[str, float]]:
        """Single training step with gradient accumulation"""
        accumulation_steps = self.training_config.get('accumulation_steps', 1)
        
        # Forward pass
        if self.use_amp:
            with autocast():
                predictions = self.model(inputs)
                loss, loss_components = self.loss_fn(predictions, targets)
        else:
            predictions = self.model(inputs)
            loss, loss_components = self.loss_fn(predictions, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if self.training_config.get('gradient_clip_norm', 0) > 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clip_norm']
                )
            
            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        return loss.item() * accumulation_steps, loss_components
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        predictions = self.model(inputs)
                        loss, loss_components = self.loss_fn(predictions, targets)
                else:
                    predictions = self.model(inputs)
                    loss, loss_components = self.loss_fn(predictions, targets)
                
                total_loss += loss.item()
                for key, value in loss_components.items():
                    if key in total_components:
                        total_components[key] += value
                    else:
                        total_components[key] = value
                
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered"""
        if val_loss < self.best_val_loss - self.training_config['early_stopping_delta']:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.training_config['early_stopping_patience']
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], suffix: str = 'regular'):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.PATHS['checkpoints'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model checkpoint
        checkpoint_path = checkpoint_dir / f"rwanda_aurora_{suffix}_epoch_{epoch}.pt"
        
        save_model_checkpoint(
            model=self.model,
            filepath=str(checkpoint_path),
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            loss=metrics.get('loss', float('inf')),
            metrics=metrics
        )
        
        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Saved checkpoint and history: {checkpoint_path}")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]):
        """Log training metrics"""
        # Update training history
        self.training_history['train_loss'].append(train_metrics['loss'])
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['loss'])
        else:
            self.training_history['val_loss'].append(None)
        
        # Log learning rates
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.training_history['learning_rates'].append(lrs)
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**3
            self.training_history['memory_usage'].append(memory_usage)
        
        # Console output
        if val_metrics:
            print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"LR: {lrs[0]:.6f}")
        else:
            print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
                  f"LR: {lrs[0]:.6f}")
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'learning_rate': lrs[0]
            }
            
            if val_metrics:
                log_dict['val_loss'] = val_metrics['loss']
                
            # Log component losses
            for key, value in train_metrics.items():
                if key != 'loss':
                    log_dict[f'train_{key}'] = value
            
            if val_metrics:
                for key, value in val_metrics.items():
                    if key != 'loss':
                        log_dict[f'val_{key}'] = value
            
            wandb.log(log_dict)
    
    def _manage_kaggle_session(self):
        """Manage Kaggle session limits and save state regularly"""
        current_time = time.time()
        
        # Save session state every configured interval
        if current_time - self.last_save_time > self.training_config['session_save_interval'] * 60:
            self._save_session_state()
            self.last_save_time = current_time
        
        # Check total session time (Kaggle has ~9-12 hour limits)
        session_hours = (current_time - self.session_start_time) / 3600
        
        if session_hours > 8:  # 8 hours as safety margin
            print(f"Approaching session time limit ({session_hours:.1f} hours)")
            print("Saving final checkpoint...")
            self._save_checkpoint(self.current_epoch, {'loss': self.best_val_loss}, 'session_end')
    
    def _save_session_state(self):
        """Save complete session state for resuming"""
        state_path = Path(self.config.PATHS['checkpoints']) / "session_state.pkl"
        
        session_state = {
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'training_history': self.training_history,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict() if self.scaler else None,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(session_state, f)
        
        print(f"Saved session state: {state_path}")
    
    def load_session_state(self, state_path: str):
        """Load session state for resuming training"""
        with open(state_path, 'rb') as f:
            session_state = pickle.load(f)
        
        self.current_epoch = session_state['current_epoch']
        self.best_val_loss = session_state['best_val_loss']
        self.patience_counter = session_state['patience_counter']
        self.training_history = session_state['training_history']
        
        self.optimizer.load_state_dict(session_state['optimizer_state'])
        self.scheduler.load_state_dict(session_state['scheduler_state'])
        
        if self.scaler and session_state['scaler_state']:
            self.scaler.load_state_dict(session_state['scaler_state'])
        
        print(f"Loaded session state from epoch {self.current_epoch}")

def train_rwanda_model(config: RwandaConfig, 
                      data_path: str,
                      model_type: str = 'lightning',
                      resume_from: Optional[str] = None) -> Dict[str, Any]:
    """
    Complete training pipeline for Rwanda Aurora model
    
    Args:
        config: Rwanda configuration
        data_path: Path to training data
        model_type: Type of model to train
        resume_from: Path to resume training from
    
    Returns:
        Training history and metrics
    """
    print("="*60)
    print("RWANDA AURORA TRAINING PIPELINE")
    print("="*60)
    
    # Setup device
    device = config.get_device()
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_rwanda_model(config, model_type=model_type, pretrained=True)
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'configure_activation_checkpointing'):
        model.configure_activation_checkpointing()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(config, data_path)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = KaggleTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        trainer.load_session_state(resume_from)
    
    # Start training
    print("Starting training...")
    training_history = trainer.train()
    
    print("Training completed!")
    return training_history

if __name__ == "__main__":
    # Test training pipeline
    config = RwandaConfig()
    
    # This would be used with actual data
    # history = train_rwanda_model(
    #     config=config,
    #     data_path="/kaggle/input/rwanda-weather-data",
    #     model_type='lightning'
    # )
    
    print("Training pipeline ready for use!")