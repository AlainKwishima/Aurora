"""
Rwanda Aurora Evaluation and Inference System
============================================

Comprehensive evaluation framework for Rwanda weather forecasting with
meteorological skill metrics, visualization tools, and production inference.
"""

import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xarray as xr

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
from .model import RwandaAurora, create_rwanda_model, load_model_checkpoint
from .data_processing import RwandaWeatherDataset, cleanup_memory

class RwandaMetrics:
    """
    Comprehensive meteorological evaluation metrics for Rwanda weather forecasting
    """
    
    def __init__(self, config: RwandaConfig):
        self.config = config
        self.metrics_config = config.METRICS_CONFIG
        self.rwanda_bounds = config.GEOGRAPHY['rwanda_bounds']
        
        # Rwanda elevation data for altitude corrections
        self.altitude_map = self._create_altitude_map()
        
        print("Initialized Rwanda meteorological metrics")
    
    def evaluate_forecasts(self, 
                          predictions: Union[Dict, Batch], 
                          targets: Union[Dict, Batch],
                          lead_times: List[int],
                          metadata: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Comprehensive forecast evaluation with multiple skill metrics
        
        Args:
            predictions: Model predictions
            targets: Ground truth observations
            lead_times: Forecast lead times (hours)
            metadata: Additional metadata (dates, locations, etc.)
        
        Returns:
            Comprehensive metrics dictionary
        """
        results = {
            'overall': {},
            'variables': {},
            'spatial': {},
            'temporal': {},
            'extreme_events': {},
            'skill_scores': {}
        }
        
        # Convert to standard format if needed
        if isinstance(predictions, Batch):
            pred_dict = self._batch_to_dict(predictions)
            target_dict = self._batch_to_dict(targets)
        else:
            pred_dict = predictions
            target_dict = targets
        
        # Overall statistics
        results['overall'] = self._compute_overall_metrics(pred_dict, target_dict)
        
        # Variable-specific metrics
        results['variables'] = self._compute_variable_metrics(pred_dict, target_dict, lead_times)
        
        # Spatial analysis
        results['spatial'] = self._compute_spatial_metrics(pred_dict, target_dict)
        
        # Temporal analysis
        if metadata and 'dates' in metadata:
            results['temporal'] = self._compute_temporal_metrics(
                pred_dict, target_dict, metadata['dates']
            )
        
        # Extreme event analysis
        results['extreme_events'] = self._compute_extreme_event_metrics(
            pred_dict, target_dict
        )
        
        # Skill scores
        results['skill_scores'] = self._compute_skill_scores(pred_dict, target_dict)
        
        return results
    
    def _compute_overall_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute overall forecast statistics"""
        metrics = {}
        
        for var_name in predictions.keys():
            if var_name in targets:
                pred = predictions[var_name]
                target = targets[var_name]
                
                # Ensure arrays are the same shape
                if pred.shape != target.shape:
                    min_shape = [min(p, t) for p, t in zip(pred.shape, target.shape)]
                    pred = pred[tuple(slice(0, s) for s in min_shape)]
                    target = target[tuple(slice(0, s) for s in min_shape)]
                
                # Flatten for overall statistics
                pred_flat = pred.flatten()
                target_flat = target.flatten()
                
                # Remove NaN values
                valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
                if valid_mask.sum() == 0:
                    continue
                
                pred_valid = pred_flat[valid_mask]
                target_valid = target_flat[valid_mask]
                
                var_metrics = {
                    'rmse': np.sqrt(mean_squared_error(target_valid, pred_valid)),
                    'mae': mean_absolute_error(target_valid, pred_valid),
                    'bias': np.mean(pred_valid - target_valid),
                    'correlation': stats.pearsonr(pred_valid, target_valid)[0],
                    'r2': r2_score(target_valid, pred_valid),
                    'std_ratio': np.std(pred_valid) / np.std(target_valid)
                }
                
                # Variable-specific adjustments
                if var_name == '2t':  # Temperature
                    var_metrics['rmse_celsius'] = var_metrics['rmse']
                    var_metrics['mae_celsius'] = var_metrics['mae']
                    var_metrics['bias_celsius'] = var_metrics['bias']
                elif var_name == 'tp':  # Precipitation
                    # Convert to mm/day if needed
                    var_metrics['rmse_mm_day'] = var_metrics['rmse'] * 24 * 1000
                    var_metrics['mae_mm_day'] = var_metrics['mae'] * 24 * 1000
                    var_metrics['bias_mm_day'] = var_metrics['bias'] * 24 * 1000
                
                metrics[var_name] = var_metrics
        
        return metrics
    
    def _compute_variable_metrics(self, predictions: Dict, targets: Dict, lead_times: List[int]) -> Dict:
        """Compute variable-specific metrics for different lead times"""
        var_metrics = {}
        
        for var_name in predictions.keys():
            if var_name not in targets:
                continue
            
            pred = predictions[var_name]
            target = targets[var_name]
            
            var_metrics[var_name] = {}
            
            # Lead time analysis
            if len(pred.shape) >= 2 and pred.shape[1] == len(lead_times):
                for i, lead_time in enumerate(lead_times):
                    pred_lead = pred[:, i] if pred.ndim > 2 else pred[i]
                    target_lead = target[:, i] if target.ndim > 2 else target[i]
                    
                    lead_metrics = self._compute_single_metric(pred_lead, target_lead, var_name)
                    var_metrics[var_name][f'lead_time_{lead_time}h'] = lead_metrics
            else:
                # Single time step
                var_metrics[var_name]['overall'] = self._compute_single_metric(pred, target, var_name)
            
            # Variable-specific skill metrics
            if var_name == 'tp':  # Precipitation
                var_metrics[var_name]['precipitation_skill'] = self._compute_precipitation_skill(pred, target)
            elif var_name == '2t':  # Temperature
                var_metrics[var_name]['temperature_skill'] = self._compute_temperature_skill(pred, target)
            elif var_name in ['10u', '10v']:  # Wind
                var_metrics[var_name]['wind_skill'] = self._compute_wind_skill(pred, target)
        
        return var_metrics
    
    def _compute_spatial_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute spatial analysis metrics"""
        spatial_metrics = {}
        
        for var_name in predictions.keys():
            if var_name not in targets:
                continue
            
            pred = predictions[var_name]
            target = targets[var_name]
            
            if len(pred.shape) < 3:  # Need spatial dimensions
                continue
            
            # Get spatial dimensions (assume last 2 dimensions)
            spatial_pred = pred[..., :, :]
            spatial_target = target[..., :, :]
            
            var_spatial = {}
            
            # Spatial correlation
            var_spatial['spatial_correlation'] = self._compute_spatial_correlation(
                spatial_pred, spatial_target
            )
            
            # Pattern correlation
            var_spatial['pattern_correlation'] = self._compute_pattern_correlation(
                spatial_pred, spatial_target
            )
            
            # Spatial gradients
            var_spatial['gradient_similarity'] = self._compute_gradient_similarity(
                spatial_pred, spatial_target
            )
            
            # Regional analysis for Rwanda
            if hasattr(self, 'rwanda_regions'):
                var_spatial['regional_metrics'] = self._compute_regional_metrics(
                    spatial_pred, spatial_target, var_name
                )
            
            spatial_metrics[var_name] = var_spatial
        
        return spatial_metrics
    
    def _compute_temporal_metrics(self, predictions: Dict, targets: Dict, dates: List) -> Dict:
        """Compute temporal analysis metrics"""
        temporal_metrics = {}
        
        # Convert dates to pandas datetime if needed
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        
        for var_name in predictions.keys():
            if var_name not in targets:
                continue
            
            pred = predictions[var_name]
            target = targets[var_name]
            
            # Ensure we have time dimension
            if pred.shape[0] != len(dates):
                continue
            
            var_temporal = {}
            
            # Seasonal analysis
            seasonal_metrics = {}
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                season_mask = self._get_seasonal_mask(dates, season)
                if season_mask.sum() > 0:
                    seasonal_metrics[season] = self._compute_single_metric(
                        pred[season_mask], target[season_mask], var_name
                    )
            var_temporal['seasonal'] = seasonal_metrics
            
            # Monthly analysis
            monthly_metrics = {}
            for month in range(1, 13):
                month_mask = dates.month == month
                if month_mask.sum() > 0:
                    monthly_metrics[f'month_{month:02d}'] = self._compute_single_metric(
                        pred[month_mask], target[month_mask], var_name
                    )
            var_temporal['monthly'] = monthly_metrics
            
            # Diurnal cycle (if hourly data)
            if len(dates) > 24 and (dates[1] - dates[0]).total_seconds() <= 3600:
                diurnal_metrics = {}
                for hour in range(0, 24, 6):  # Every 6 hours
                    hour_mask = dates.hour == hour
                    if hour_mask.sum() > 0:
                        diurnal_metrics[f'hour_{hour:02d}'] = self._compute_single_metric(
                            pred[hour_mask], target[hour_mask], var_name
                        )
                var_temporal['diurnal'] = diurnal_metrics
            
            temporal_metrics[var_name] = var_temporal
        
        return temporal_metrics
    
    def _compute_extreme_event_metrics(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute extreme event detection and forecast metrics"""
        extreme_metrics = {}
        thresholds = self.metrics_config['extreme_thresholds']
        
        for var_name in predictions.keys():
            if var_name not in targets:
                continue
            
            pred = predictions[var_name].flatten()
            target = targets[var_name].flatten()
            
            # Remove invalid values
            valid_mask = ~(np.isnan(pred) | np.isnan(target))
            pred = pred[valid_mask]
            target = target[valid_mask]
            
            var_extreme = {}
            
            if var_name == 'tp':  # Precipitation
                threshold = thresholds['heavy_precipitation']
                var_extreme = self._compute_event_detection_metrics(
                    pred, target, threshold, 'above'
                )
                
            elif var_name == '2t':  # Temperature
                # High temperature events
                high_threshold = thresholds['high_temperature'] + 273.15
                var_extreme['high_temperature'] = self._compute_event_detection_metrics(
                    pred, target, high_threshold, 'above'
                )
                
                # Low temperature events
                low_threshold = thresholds['low_temperature'] + 273.15
                var_extreme['low_temperature'] = self._compute_event_detection_metrics(
                    pred, target, low_threshold, 'below'
                )
                
            elif var_name in ['10u', '10v']:  # Wind
                threshold = thresholds['strong_wind']
                var_extreme = self._compute_event_detection_metrics(
                    np.abs(pred), np.abs(target), threshold, 'above'
                )
            
            extreme_metrics[var_name] = var_extreme
        
        return extreme_metrics
    
    def _compute_skill_scores(self, predictions: Dict, targets: Dict) -> Dict:
        """Compute forecast skill scores"""
        skill_scores = {}
        
        for var_name in predictions.keys():
            if var_name not in targets:
                continue
            
            pred = predictions[var_name]
            target = targets[var_name]
            
            var_skills = {}
            
            # Mean Squared Error Skill Score (MSESS)
            mse_forecast = np.mean((pred - target) ** 2)
            mse_climatology = np.var(target)
            var_skills['msess'] = 1 - (mse_forecast / mse_climatology)
            
            # Anomaly Correlation
            pred_anomaly = pred - np.mean(pred)
            target_anomaly = target - np.mean(target)
            var_skills['anomaly_correlation'] = stats.pearsonr(
                pred_anomaly.flatten(), target_anomaly.flatten()
            )[0]
            
            # Nash-Sutcliffe Efficiency
            var_skills['nse'] = 1 - (np.sum((target - pred) ** 2) / 
                                    np.sum((target - np.mean(target)) ** 2))
            
            # Index of Agreement
            var_skills['index_of_agreement'] = 1 - (
                np.sum((pred - target) ** 2) /
                np.sum((np.abs(pred - np.mean(target)) + 
                       np.abs(target - np.mean(target))) ** 2)
            )
            
            skill_scores[var_name] = var_skills
        
        return skill_scores
    
    def _compute_single_metric(self, pred: np.ndarray, target: np.ndarray, var_name: str) -> Dict:
        """Compute metrics for a single variable"""
        # Flatten arrays
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Remove invalid values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        if valid_mask.sum() == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'correlation': np.nan}
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(target_valid, pred_valid)),
            'mae': mean_absolute_error(target_valid, pred_valid),
            'bias': np.mean(pred_valid - target_valid),
            'correlation': stats.pearsonr(pred_valid, target_valid)[0],
            'count': len(pred_valid)
        }
        
        return metrics
    
    def _compute_precipitation_skill(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """Compute precipitation-specific skill metrics"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Remove invalid values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Precipitation occurrence (>0.1 mm/day threshold)
        precip_threshold = 0.1 / (24 * 1000)  # Convert to model units
        
        pred_occur = pred_valid > precip_threshold
        target_occur = target_valid > precip_threshold
        
        # Contingency table metrics
        tp = np.sum(pred_occur & target_occur)  # True positives
        fp = np.sum(pred_occur & ~target_occur)  # False positives
        fn = np.sum(~pred_occur & target_occur)  # False negatives
        tn = np.sum(~pred_occur & ~target_occur)  # True negatives
        
        # Avoid division by zero
        eps = 1e-10
        
        metrics = {
            'pod': tp / (tp + fn + eps),  # Probability of detection
            'far': fp / (tp + fp + eps),  # False alarm ratio
            'csi': tp / (tp + fp + fn + eps),  # Critical success index
            'bias_score': (tp + fp) / (tp + fn + eps),  # Frequency bias
            'accuracy': (tp + tn) / (tp + fp + fn + tn + eps)
        }
        
        return metrics
    
    def _compute_temperature_skill(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """Compute temperature-specific skill metrics"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Remove invalid values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Convert to Celsius for interpretation
        pred_celsius = pred_valid - 273.15
        target_celsius = target_valid - 273.15
        
        metrics = {
            'rmse_celsius': np.sqrt(mean_squared_error(target_celsius, pred_celsius)),
            'mae_celsius': mean_absolute_error(target_celsius, pred_celsius),
            'bias_celsius': np.mean(pred_celsius - target_celsius),
            'diurnal_amplitude_error': self._compute_diurnal_amplitude_error(pred_valid, target_valid)
        }
        
        return metrics
    
    def _compute_wind_skill(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """Compute wind-specific skill metrics"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Remove invalid values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Wind speed statistics
        pred_speed = np.abs(pred_valid)
        target_speed = np.abs(target_valid)
        
        metrics = {
            'speed_rmse': np.sqrt(mean_squared_error(target_speed, pred_speed)),
            'speed_mae': mean_absolute_error(target_speed, pred_speed),
            'speed_bias': np.mean(pred_speed - target_speed),
            'vector_correlation': stats.pearsonr(pred_valid, target_valid)[0]
        }
        
        return metrics
    
    def _compute_spatial_correlation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute spatial correlation between prediction and target"""
        if len(pred.shape) < 2:
            return np.nan
        
        # Average over time if present
        if len(pred.shape) > 2:
            pred_spatial = np.mean(pred, axis=0)
            target_spatial = np.mean(target, axis=0)
        else:
            pred_spatial = pred
            target_spatial = target
        
        # Flatten spatial dimensions
        pred_flat = pred_spatial.flatten()
        target_flat = target_spatial.flatten()
        
        # Remove invalid values
        valid_mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
        if valid_mask.sum() < 10:
            return np.nan
        
        return stats.pearsonr(pred_flat[valid_mask], target_flat[valid_mask])[0]
    
    def _compute_pattern_correlation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute pattern correlation (anomaly correlation over space)"""
        if len(pred.shape) < 2:
            return np.nan
        
        # Compute spatial means
        pred_spatial_mean = np.mean(pred, axis=(-2, -1), keepdims=True)
        target_spatial_mean = np.mean(target, axis=(-2, -1), keepdims=True)
        
        # Compute anomalies
        pred_anomaly = pred - pred_spatial_mean
        target_anomaly = target - target_spatial_mean
        
        # Pattern correlation
        numerator = np.sum(pred_anomaly * target_anomaly)
        denominator = np.sqrt(np.sum(pred_anomaly ** 2) * np.sum(target_anomaly ** 2))
        
        if denominator == 0:
            return np.nan
        
        return numerator / denominator
    
    def _compute_gradient_similarity(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """Compute similarity of spatial gradients"""
        if len(pred.shape) < 2:
            return {}
        
        # Compute gradients
        pred_grad_x = np.gradient(pred, axis=-1)
        pred_grad_y = np.gradient(pred, axis=-2)
        target_grad_x = np.gradient(target, axis=-1)
        target_grad_y = np.gradient(target, axis=-2)
        
        # Gradient magnitude
        pred_grad_mag = np.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_grad_mag = np.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # Flatten for correlation
        pred_grad_flat = pred_grad_mag.flatten()
        target_grad_flat = target_grad_mag.flatten()
        
        valid_mask = ~(np.isnan(pred_grad_flat) | np.isnan(target_grad_flat))
        
        if valid_mask.sum() < 10:
            return {'gradient_correlation': np.nan}
        
        gradient_corr = stats.pearsonr(
            pred_grad_flat[valid_mask], 
            target_grad_flat[valid_mask]
        )[0]
        
        return {'gradient_correlation': gradient_corr}
    
    def _compute_event_detection_metrics(self, pred: np.ndarray, target: np.ndarray, 
                                        threshold: float, direction: str) -> Dict:
        """Compute event detection metrics for extreme events"""
        if direction == 'above':
            pred_event = pred > threshold
            target_event = target > threshold
        else:  # below
            pred_event = pred < threshold
            target_event = target < threshold
        
        # Contingency table
        tp = np.sum(pred_event & target_event)
        fp = np.sum(pred_event & ~target_event)
        fn = np.sum(~pred_event & target_event)
        tn = np.sum(~pred_event & ~target_event)
        
        eps = 1e-10
        
        metrics = {
            'hit_rate': tp / (tp + fn + eps),
            'false_alarm_rate': fp / (fp + tn + eps),
            'critical_success_index': tp / (tp + fp + fn + eps),
            'frequency_bias': (tp + fp) / (tp + fn + eps),
            'true_skill_statistic': (tp / (tp + fn + eps)) - (fp / (fp + tn + eps)),
            'event_count_predicted': int(np.sum(pred_event)),
            'event_count_observed': int(np.sum(target_event))
        }
        
        return metrics
    
    def _get_seasonal_mask(self, dates: pd.DatetimeIndex, season: str) -> np.ndarray:
        """Get mask for seasonal filtering"""
        month = dates.month
        
        if season == 'DJF':  # December, January, February
            return (month == 12) | (month == 1) | (month == 2)
        elif season == 'MAM':  # March, April, May
            return (month >= 3) & (month <= 5)
        elif season == 'JJA':  # June, July, August
            return (month >= 6) & (month <= 8)
        elif season == 'SON':  # September, October, November
            return (month >= 9) & (month <= 11)
        else:
            return np.ones(len(dates), dtype=bool)
    
    def _compute_diurnal_amplitude_error(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute error in diurnal temperature amplitude"""
        # This is a simplified version - would need hourly data for full analysis
        pred_range = np.max(pred) - np.min(pred)
        target_range = np.max(target) - np.min(target)
        return abs(pred_range - target_range)
    
    def _batch_to_dict(self, batch: Batch) -> Dict[str, np.ndarray]:
        """Convert Aurora Batch to dictionary format"""
        result = {}
        
        # Surface variables
        for var_name, var_data in batch.surf_vars.items():
            if isinstance(var_data, torch.Tensor):
                result[var_name] = var_data.detach().cpu().numpy()
            else:
                result[var_name] = var_data
        
        # Atmospheric variables
        for var_name, var_data in batch.atmos_vars.items():
            if isinstance(var_data, torch.Tensor):
                result[f'atmos_{var_name}'] = var_data.detach().cpu().numpy()
            else:
                result[f'atmos_{var_name}'] = var_data
        
        return result
    
    def _create_altitude_map(self) -> np.ndarray:
        """Create a simplified altitude map for Rwanda"""
        # This is a placeholder - in practice you'd load actual elevation data
        lat_range = np.linspace(self.rwanda_bounds['south'], self.rwanda_bounds['north'], 32)
        lon_range = np.linspace(self.rwanda_bounds['west'], self.rwanda_bounds['east'], 32)
        
        # Simplified elevation model (higher in the west, lower in the east)
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
        altitude = 2000 - 800 * (lon_grid - self.rwanda_bounds['west']) / \
                  (self.rwanda_bounds['east'] - self.rwanda_bounds['west'])
        
        return altitude

class RwandaVisualizer:
    """
    Advanced visualization tools for Rwanda weather forecasts and evaluation
    """
    
    def __init__(self, config: RwandaConfig):
        self.config = config
        self.rwanda_bounds = config.GEOGRAPHY['rwanda_bounds']
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("Initialized Rwanda weather visualizer")
    
    def plot_forecast_summary(self, metrics: Dict, save_path: Optional[str] = None):
        """Create comprehensive forecast summary plot"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Rwanda Weather Forecast Evaluation Summary', fontsize=16, fontweight='bold')
        
        # Overall RMSE by variable
        if 'overall' in metrics:
            variables = []
            rmse_values = []
            for var, var_metrics in metrics['overall'].items():
                variables.append(var)
                rmse_values.append(var_metrics.get('rmse', 0))
            
            axes[0, 0].bar(variables, rmse_values, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('RMSE by Variable')
            axes[0, 0].set_ylabel('RMSE')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Correlation by variable
        if 'overall' in metrics:
            correlations = []
            for var in variables:
                correlations.append(metrics['overall'][var].get('correlation', 0))
            
            axes[0, 1].bar(variables, correlations, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Correlation by Variable')
            axes[0, 1].set_ylabel('Correlation')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
        
        # Skill scores
        if 'skill_scores' in metrics:
            skill_vars = list(metrics['skill_scores'].keys())[:3]  # Top 3 variables
            skill_types = ['msess', 'anomaly_correlation', 'nse']
            skill_matrix = np.zeros((len(skill_vars), len(skill_types)))
            
            for i, var in enumerate(skill_vars):
                for j, skill in enumerate(skill_types):
                    skill_matrix[i, j] = metrics['skill_scores'][var].get(skill, 0)
            
            im = axes[0, 2].imshow(skill_matrix, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
            axes[0, 2].set_title('Skill Scores')
            axes[0, 2].set_xticks(range(len(skill_types)))
            axes[0, 2].set_xticklabels(skill_types, rotation=45)
            axes[0, 2].set_yticks(range(len(skill_vars)))
            axes[0, 2].set_yticklabels(skill_vars)
            plt.colorbar(im, ax=axes[0, 2])
        
        # Temporal performance (if available)
        if 'temporal' in metrics and '2t' in metrics['temporal']:
            temp_temporal = metrics['temporal']['2t']
            if 'seasonal' in temp_temporal:
                seasons = list(temp_temporal['seasonal'].keys())
                seasonal_rmse = [temp_temporal['seasonal'][s]['rmse'] for s in seasons]
                
                axes[1, 0].plot(seasons, seasonal_rmse, marker='o', linewidth=2, markersize=8)
                axes[1, 0].set_title('Temperature RMSE by Season')
                axes[1, 0].set_ylabel('RMSE (K)')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Extreme event performance
        if 'extreme_events' in metrics:
            if 'tp' in metrics['extreme_events']:
                precip_extreme = metrics['extreme_events']['tp']
                categories = ['Hit Rate', 'False Alarm Rate', 'CSI']
                values = [
                    precip_extreme.get('hit_rate', 0),
                    precip_extreme.get('false_alarm_rate', 0),
                    precip_extreme.get('critical_success_index', 0)
                ]
                
                axes[1, 1].bar(categories, values, color=['green', 'red', 'blue'], alpha=0.7)
                axes[1, 1].set_title('Extreme Precipitation Events')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Lead time performance (if available)
        if 'variables' in metrics and '2t' in metrics['variables']:
            temp_vars = metrics['variables']['2t']
            lead_times = []
            lead_rmse = []
            
            for key, value in temp_vars.items():
                if key.startswith('lead_time_'):
                    lead_time = int(key.split('_')[-1].replace('h', ''))
                    lead_times.append(lead_time)
                    lead_rmse.append(value['rmse'])
            
            if lead_times:
                sorted_pairs = sorted(zip(lead_times, lead_rmse))
                lead_times, lead_rmse = zip(*sorted_pairs)
                
                axes[1, 2].plot(lead_times, lead_rmse, marker='o', linewidth=2, markersize=8)
                axes[1, 2].set_title('Temperature RMSE vs Lead Time')
                axes[1, 2].set_xlabel('Lead Time (hours)')
                axes[1, 2].set_ylabel('RMSE (K)')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved forecast summary to {save_path}")
        
        plt.show()
    
    def plot_spatial_forecast(self, predictions: Dict, targets: Dict, 
                            variable: str = '2t', save_path: Optional[str] = None):
        """Plot spatial forecast comparison"""
        if variable not in predictions or variable not in targets:
            print(f"Variable {variable} not found in data")
            return
        
        pred = predictions[variable]
        target = targets[variable]
        
        # Take mean over time if multiple time steps
        if len(pred.shape) > 2:
            pred = np.mean(pred, axis=0)
            target = np.mean(target, axis=0)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Define colormap and limits based on variable
        if variable == '2t':
            vmin, vmax = target.min(), target.max()
            cmap = 'RdYlBu_r'
            units = 'K'
            title_var = 'Temperature'
        elif variable == 'tp':
            vmin, vmax = 0, max(target.max(), pred.max())
            cmap = 'Blues'
            units = 'm/s'
            title_var = 'Precipitation'
        else:
            vmin, vmax = min(target.min(), pred.min()), max(target.max(), pred.max())
            cmap = 'viridis'
            units = ''
            title_var = variable.upper()
        
        # Plot target
        im1 = axes[0].imshow(target, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[0].set_title(f'{title_var} - Observed')
        plt.colorbar(im1, ax=axes[0], label=units)
        
        # Plot prediction
        im2 = axes[1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        axes[1].set_title(f'{title_var} - Predicted')
        plt.colorbar(im2, ax=axes[1], label=units)
        
        # Plot difference
        diff = pred - target
        diff_max = max(abs(diff.min()), abs(diff.max()))
        im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, aspect='auto')
        axes[2].set_title(f'{title_var} - Difference (Pred - Obs)')
        plt.colorbar(im3, ax=axes[2], label=units)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved spatial forecast plot to {save_path}")
        
        plt.show()
    
    def plot_time_series(self, predictions: Dict, targets: Dict, 
                        variable: str = '2t', location: Tuple[int, int] = None,
                        dates: List = None, save_path: Optional[str] = None):
        """Plot time series comparison at specific location"""
        if variable not in predictions or variable not in targets:
            print(f"Variable {variable} not found in data")
            return
        
        pred = predictions[variable]
        target = targets[variable]
        
        # Select location (default to center of domain)
        if location is None:
            location = (pred.shape[-2] // 2, pred.shape[-1] // 2)
        
        # Extract time series
        if len(pred.shape) > 2:
            pred_ts = pred[:, location[0], location[1]]
            target_ts = target[:, location[0], location[1]]
        else:
            pred_ts = pred.flatten()
            target_ts = target.flatten()
        
        # Create time axis
        if dates is not None:
            x_axis = pd.to_datetime(dates)
        else:
            x_axis = range(len(pred_ts))
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Time series
        ax1.plot(x_axis, target_ts, label='Observed', color='blue', linewidth=2)
        ax1.plot(x_axis, pred_ts, label='Predicted', color='red', linewidth=2, alpha=0.8)
        ax1.set_ylabel(f'{variable} [{self._get_units(variable)}]')
        ax1.set_title(f'{variable.upper()} Time Series at Location {location}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error time series
        error = pred_ts - target_ts
        ax2.plot(x_axis, error, color='green', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(x_axis, error, alpha=0.3, color='green')
        ax2.set_ylabel(f'Error [{self._get_units(variable)}]')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        # Format dates if available
        if dates is not None:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved time series plot to {save_path}")
        
        plt.show()
    
    def _get_units(self, variable: str) -> str:
        """Get units for variable"""
        units_map = {
            '2t': 'K',
            'tp': 'm/s',
            '10u': 'm/s',
            '10v': 'm/s',
            'msl': 'Pa',
            'sp': 'Pa',
            'tcc': '-',
            'skt': 'K'
        }
        return units_map.get(variable, '')

class RwandaInference:
    """
    Production inference system for Rwanda weather forecasting
    """
    
    def __init__(self, config: RwandaConfig, model_path: str, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or config.get_device()
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize metrics and visualization
        self.metrics = RwandaMetrics(config)
        self.visualizer = RwandaVisualizer(config)
        
        print("Rwanda inference system ready")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        # Load model architecture
        model = create_rwanda_model(self.config, model_type='lightning', pretrained=False)
        
        # Load checkpoint
        checkpoint = load_model_checkpoint(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def predict(self, input_data: Union[Batch, Dict], lead_times: List[int]) -> Dict:
        """
        Generate weather forecast
        
        Args:
            input_data: Input weather data
            lead_times: Forecast lead times in hours
        
        Returns:
            Dictionary with predictions and metadata
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(input_data, dict):
                # Convert dict to batch format if needed
                input_batch = self._dict_to_batch(input_data)
            else:
                input_batch = input_data
            
            # Ensure data is on correct device
            input_batch = self._batch_to_device(input_batch)
            
            # Generate prediction
            predictions = self.model(input_batch)
            
            # Convert to numpy for processing
            pred_dict = self._batch_to_dict(predictions)
        
        # Add metadata
        result = {
            'predictions': pred_dict,
            'lead_times': lead_times,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'config': str(self.config),
                'device': str(self.device)
            }
        }
        
        return result
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test dataset
        
        Args:
            test_loader: DataLoader with test data
        
        Returns:
            Comprehensive evaluation metrics
        """
        print("Starting model evaluation...")
        
        all_predictions = []
        all_targets = []
        all_metadata = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{len(test_loader)}")
                
                # Move to device
                inputs = self._batch_to_device(inputs)
                targets = self._batch_to_device(targets)
                
                # Predict
                predictions = self.model(inputs)
                
                # Convert to numpy and store
                pred_dict = self._batch_to_dict(predictions)
                target_dict = self._batch_to_dict(targets)
                
                all_predictions.append(pred_dict)
                all_targets.append(target_dict)
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    cleanup_memory()
        
        print("Computing evaluation metrics...")
        
        # Combine all predictions and targets
        combined_predictions = self._combine_predictions(all_predictions)
        combined_targets = self._combine_predictions(all_targets)
        
        # Compute metrics
        lead_times = list(range(6, 121, 6))  # 6-hour intervals up to 120 hours
        metrics = self.metrics.evaluate_forecasts(
            predictions=combined_predictions,
            targets=combined_targets,
            lead_times=lead_times
        )
        
        print("Evaluation completed!")
        return metrics
    
    def generate_forecast_report(self, metrics: Dict, output_dir: str):
        """
        Generate comprehensive forecast evaluation report
        
        Args:
            metrics: Evaluation metrics dictionary
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating forecast report in {output_dir}")
        
        # Summary plot
        summary_path = output_path / "forecast_summary.png"
        self.visualizer.plot_forecast_summary(metrics, str(summary_path))
        
        # Detailed metrics report
        report_path = output_path / "detailed_metrics.json"
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = self._metrics_to_json(metrics)
            json.dump(json_metrics, f, indent=2)
        
        # Text summary
        summary_path = output_path / "summary_report.txt"
        self._write_text_summary(metrics, str(summary_path))
        
        print(f"Report generated successfully in {output_dir}")
    
    def _dict_to_batch(self, data_dict: Dict) -> Batch:
        """Convert dictionary to Aurora Batch format"""
        # This is a placeholder implementation
        # In practice, you'd need to properly construct Aurora Batch objects
        batch = Batch()
        batch.surf_vars = {}
        batch.atmos_vars = {}
        
        for key, value in data_dict.items():
            if key.startswith('atmos_'):
                batch.atmos_vars[key[6:]] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
            else:
                batch.surf_vars[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value
        
        return batch
    
    def _batch_to_device(self, batch: Batch) -> Batch:
        """Move batch to correct device"""
        if hasattr(batch, 'surf_vars'):
            for key, value in batch.surf_vars.items():
                if isinstance(value, torch.Tensor):
                    batch.surf_vars[key] = value.to(self.device)
        
        if hasattr(batch, 'atmos_vars'):
            for key, value in batch.atmos_vars.items():
                if isinstance(value, torch.Tensor):
                    batch.atmos_vars[key] = value.to(self.device)
        
        return batch
    
    def _batch_to_dict(self, batch: Batch) -> Dict[str, np.ndarray]:
        """Convert Aurora Batch to dictionary format"""
        result = {}
        
        if hasattr(batch, 'surf_vars'):
            for var_name, var_data in batch.surf_vars.items():
                if isinstance(var_data, torch.Tensor):
                    result[var_name] = var_data.detach().cpu().numpy()
                else:
                    result[var_name] = var_data
        
        if hasattr(batch, 'atmos_vars'):
            for var_name, var_data in batch.atmos_vars.items():
                if isinstance(var_data, torch.Tensor):
                    result[f'atmos_{var_name}'] = var_data.detach().cpu().numpy()
                else:
                    result[f'atmos_{var_name}'] = var_data
        
        return result
    
    def _combine_predictions(self, prediction_list: List[Dict]) -> Dict[str, np.ndarray]:
        """Combine list of prediction dictionaries"""
        if not prediction_list:
            return {}
        
        combined = {}
        for key in prediction_list[0].keys():
            arrays = [pred[key] for pred in prediction_list if key in pred]
            if arrays:
                combined[key] = np.concatenate(arrays, axis=0)
        
        return combined
    
    def _metrics_to_json(self, metrics: Dict) -> Dict:
        """Convert metrics dictionary to JSON-serializable format"""
        def convert_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            else:
                return value
        
        return convert_value(metrics)
    
    def _write_text_summary(self, metrics: Dict, filepath: str):
        """Write text summary of evaluation metrics"""
        with open(filepath, 'w') as f:
            f.write("RWANDA AURORA WEATHER FORECAST EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall performance
            if 'overall' in metrics:
                f.write("OVERALL PERFORMANCE:\n")
                f.write("-" * 20 + "\n")
                for var_name, var_metrics in metrics['overall'].items():
                    f.write(f"\n{var_name.upper()}:\n")
                    f.write(f"  RMSE: {var_metrics.get('rmse', 'N/A'):.4f}\n")
                    f.write(f"  MAE:  {var_metrics.get('mae', 'N/A'):.4f}\n")
                    f.write(f"  Bias: {var_metrics.get('bias', 'N/A'):.4f}\n")
                    f.write(f"  Corr: {var_metrics.get('correlation', 'N/A'):.4f}\n")
            
            # Skill scores
            if 'skill_scores' in metrics:
                f.write("\n\nSKILL SCORES:\n")
                f.write("-" * 20 + "\n")
                for var_name, skill_metrics in metrics['skill_scores'].items():
                    f.write(f"\n{var_name.upper()}:\n")
                    f.write(f"  MSESS: {skill_metrics.get('msess', 'N/A'):.4f}\n")
                    f.write(f"  NSE:   {skill_metrics.get('nse', 'N/A'):.4f}\n")
                    f.write(f"  IOA:   {skill_metrics.get('index_of_agreement', 'N/A'):.4f}\n")
            
            # Extreme events
            if 'extreme_events' in metrics:
                f.write("\n\nEXTREME EVENT PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                for var_name, extreme_metrics in metrics['extreme_events'].items():
                    if isinstance(extreme_metrics, dict):
                        f.write(f"\n{var_name.upper()}:\n")
                        for event_type, event_metrics in extreme_metrics.items():
                            f.write(f"  {event_type}:\n")
                            if isinstance(event_metrics, dict):
                                f.write(f"    Hit Rate: {event_metrics.get('hit_rate', 'N/A'):.4f}\n")
                                f.write(f"    FAR:      {event_metrics.get('false_alarm_rate', 'N/A'):.4f}\n")
                                f.write(f"    CSI:      {event_metrics.get('critical_success_index', 'N/A'):.4f}\n")

# Main evaluation function
def evaluate_rwanda_model(config: RwandaConfig,
                         model_path: str,
                         test_data_path: str,
                         output_dir: str) -> Dict:
    """
    Complete evaluation pipeline for Rwanda Aurora model
    
    Args:
        config: Rwanda configuration
        model_path: Path to trained model
        test_data_path: Path to test dataset
        output_dir: Directory for evaluation outputs
    
    Returns:
        Evaluation metrics dictionary
    """
    print("="*60)
    print("RWANDA AURORA MODEL EVALUATION")
    print("="*60)
    
    # Initialize inference system
    inference = RwandaInference(config, model_path)
    
    # Create test data loader
    from .data_processing import create_dataloaders
    # Use create_dataloaders but only return validation loader as test loader
    _, test_loader = create_dataloaders(config, test_data_path)
    
    # Evaluate model
    metrics = inference.evaluate_model(test_loader)
    
    # Generate report
    inference.generate_forecast_report(metrics, output_dir)
    
    print("Evaluation completed successfully!")
    return metrics

if __name__ == "__main__":
    # Test evaluation system
    config = RwandaConfig()
    
    # This would be used with actual model and data
    # metrics = evaluate_rwanda_model(
    #     config=config,
    #     model_path="/kaggle/input/rwanda-model/best_model.pt",
    #     test_data_path="/kaggle/input/test-data",
    #     output_dir="/kaggle/working/evaluation"
    # )
    
    print("Evaluation system ready for use!")