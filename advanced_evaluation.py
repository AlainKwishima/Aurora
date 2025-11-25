"""
Comprehensive Model Evaluation Pipeline
======================================

Advanced evaluation for Rwanda Aurora model including:
- Physical metrics (RMSE, MAE, Bias, Correlation)
- Baseline comparisons (Persistence, Climatology)
- Skill scores
- Per-variable and per-lead-time analysis
- Visualization suite

Usage:
    python advanced_evaluation.py --model working/best_rwanda_aurora.pt --data data/rwanda_era5_test.nc
"""

import argparse
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from scipy import stats

# Import model
import sys
sys.path.append('.')
from notebooks.rwanda_aurora_training import RwandaAuroraLite, KaggleConfig


class AdvancedEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.variables = ['2t', 'tp', '10u', '10v', 'msl']
        self.var_names = ['Temperature', 'Precipitation', 'U-Wind', 'V-Wind', 'Pressure']
        
    def calculate_metrics(self, predictions, targets):
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Ensure numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Per-variable metrics
        for i, var in enumerate(self.variables):
            pred_var = predictions[..., i]
            target_var = targets[..., i]
            
            # RMSE
            rmse = np.sqrt(np.mean((pred_var - target_var) ** 2))
            
            # MAE
            mae = np.mean(np.abs(pred_var - target_var))
            
            # Bias
            bias = np.mean(pred_var - target_var)
            
            # Correlation
            corr = np.corrcoef(pred_var.flatten(), target_var.flatten())[0, 1]
            
            # R²
            ss_res = np.sum((target_var - pred_var) ** 2)
            ss_tot = np.sum((target_var - np.mean(target_var)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics[var] = {
                'RMSE': rmse,
                'MAE': mae,
                'Bias': bias,
                'Correlation': corr,
                'R2': r2
            }
        
        return metrics
    
    def calculate_skill_score(self, model_metrics, baseline_metrics):
        """Calculate skill score relative to baseline."""
        skill_scores = {}
        
        for var in self.variables:
            model_rmse = model_metrics[var]['RMSE']
            baseline_rmse = baseline_metrics[var]['RMSE']
            
            # Skill score: 1 - (model_error / baseline_error)
            # > 0 means better than baseline
            # 1.0 means perfect
            skill = 1 - (model_rmse / baseline_rmse)
            skill_scores[var] = skill
        
        return skill_scores
    
    def persistence_forecast(self, input_data):
        """Persistence baseline: repeat last timestep."""
        # input_data: [batch, seq_len, height, width, channels]
        last_timestep = input_data[:, -1:, :, :, :]  # Get last timestep
        
        # Repeat for forecast length (20 timesteps)
        forecast = np.repeat(last_timestep, 20, axis=1)
        
        return forecast
    
    def climatology_forecast(self, historical_data, time_indices):
        """Climatology baseline: historical average for each day/time."""
        # Simplified: use overall mean
        climatology = np.mean(historical_data, axis=(0, 1), keepdims=True)
        climatology = np.repeat(climatology, historical_data.shape[0], axis=0)
        climatology = np.repeat(climatology, 20, axis=1)
        
        return climatology
    
    def evaluate_all(self, test_loader, output_dir='evaluation_results'):
        """Run comprehensive evaluation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("🔬 Advanced Model Evaluation")
        print("=" * 60)
        
        all_predictions = []
        all_targets = []
        all_inputs = []
        
        # Collect predictions
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                
                predictions = self.model(inputs, use_vectorized=True)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
                all_inputs.append(inputs.cpu().numpy())
                
                if batch_idx >= 100:  # Limit for speed
                    break
        
        # Concatenate
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_inputs = np.concatenate(all_inputs, axis=0)
        
        print(f"✓ Collected {len(all_predictions)} samples")
        
        # Model metrics
        print("\n📊 Model Performance:")
        model_metrics = self.calculate_metrics(all_predictions, all_targets)
        self._print_metrics(model_metrics)
        
        # Baseline: Persistence
        print("\n📈 Persistence Baseline:")
        persistence_preds = self.persistence_forecast(all_inputs)
        persistence_metrics = self.calculate_metrics(persistence_preds, all_targets)
        self._print_metrics(persistence_metrics)
        
        # Baseline: Climatology
        print("\n📈 Climatology Baseline:")
        climatology_preds = self.climatology_forecast(all_inputs, None)
        climatology_metrics = self.calculate_metrics(climatology_preds, all_targets)
        self._print_metrics(climatology_metrics)
        
        # Skill scores
        print("\n🎯 Skill Scores (vs Persistence):")
        persistence_skill = self.calculate_skill_score(model_metrics, persistence_metrics)
        for var, skill in persistence_skill.items():
            print(f"  {var}: {skill:.3f}")
        
        print("\n🎯 Skill Scores (vs Climatology):")
        climatology_skill = self.calculate_skill_score(model_metrics, climatology_metrics)
        for var, skill in climatology_skill.items():
            print(f"  {var}: {skill:.3f}")
        
        # Save results
        results = {
            'model_metrics': model_metrics,
            'persistence_metrics': persistence_metrics,
            'climatology_metrics': climatology_metrics,
            'persistence_skill': persistence_skill,
            'climatology_skill': climatology_skill,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        self._plot_metrics_comparison(model_metrics, persistence_metrics, climatology_metrics, output_dir)
        self._plot_skill_scores(persistence_skill, climatology_skill, output_dir)
        self._plot_per_lead_time(all_predictions, all_targets, output_dir)
        self._plot_spatial_errors(all_predictions, all_targets, output_dir)
        
        print(f"\n✅ Evaluation complete! Results saved to {output_dir}")
        
        return results
    
    def _print_metrics(self, metrics):
        """Print metrics table."""
        for var, vals in metrics.items():
            print(f"  {var:4s}: RMSE={vals['RMSE']:.4f}, MAE={vals['MAE']:.4f}, "
                  f"Corr={vals['Correlation']:.3f}, R²={vals['R2']:.3f}")
    
    def _plot_metrics_comparison(self, model, persistence, climatology, output_dir):
        """Plot metrics comparison bar chart."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = ['RMSE', 'MAE', 'Correlation', 'R2']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            vars = list(model.keys())
            x = np.arange(len(vars))
            width = 0.25
            
            model_vals = [model[v][metric] for v in vars]
            pers_vals = [persistence[v][metric] for v in vars]
            clim_vals = [climatology[v][metric] for v in vars]
            
            ax.bar(x - width, model_vals, width, label='Model', color='#2ecc71')
            ax.bar(x, pers_vals, width, label='Persistence', color='#e74c3c')
            ax.bar(x + width, clim_vals, width, label='Climatology', color='#3498db')
            
            ax.set_xlabel('Variable')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(vars)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_comparison.png', dpi=150)
        plt.close()
    
    def _plot_skill_scores(self, pers_skill, clim_skill, output_dir):
        """Plot skill scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        vars = list(pers_skill.keys())
        x = np.arange(len(vars))
        width = 0.35
        
        ax.bar(x - width/2, list(pers_skill.values()), width, label='vs Persistence', color='#9b59b6')
        ax.bar(x + width/2, list(clim_skill.values()), width, label='vs Climatology', color='#f39c12')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Skill Score')
        ax.set_title('Model Skill Scores (>0 = Better than Baseline)')
        ax.set_xticks(x)
        ax.set_xticklabels(vars)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'skill_scores.png', dpi=150)
        plt.close()
    
    def _plot_per_lead_time(self, predictions, targets, output_dir):
        """Plot error vs lead time."""
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        for i, (var, ax) in enumerate(zip(self.variables, axes)):
            lead_times = range(1, 21)  # 1-20 (in 6-hour steps)
            rmses = []
            
            for t in range(20):
                pred_t = predictions[:, t, :, :, i]
                target_t = targets[:, t, :, :, i]
                rmse = np.sqrt(np.mean((pred_t - target_t) ** 2))
                rmses.append(rmse)
            
            hours = [t * 6 for t in lead_times]
            ax.plot(hours, rmses, marker='o', linewidth=2, markersize=4, color='#3498db')
            ax.set_xlabel('Forecast Hour')
            ax.set_ylabel('RMSE')
            ax.set_title(f'{self.var_names[i]}')
            ax.grid(alpha=0.3)
        
        plt.suptitle('Forecast Error vs Lead Time')
        plt.tight_layout()
        plt.savefig(output_dir / 'error_vs_lead_time.png', dpi=150)
        plt.close()
    
    def _plot_spatial_errors(self, predictions, targets, output_dir):
        """Plot spatial distribution of errors."""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        for i, var in enumerate(self.variables):
            # Mean absolute error map
            mae_map = np.mean(np.abs(predictions[..., i] - targets[..., i]), axis=(0, 1))
            
            im1 = axes[0, i].imshow(mae_map, cmap='Reds')
            axes[0, i].set_title(f'{self.var_names[i]} - MAE')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Mean bias map
            bias_map = np.mean(predictions[..., i] - targets[..., i], axis=(0, 1))
            
            im2 = axes[1, i].imshow(bias_map, cmap='RdBu_r', vmin=-abs(bias_map).max(), vmax=abs(bias_map).max())
            axes[1, i].set_title(f'{self.var_names[i]} - Bias')
            plt.colorbar(im2, ax=axes[1, i])
        
        plt.suptitle('Spatial Error Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / 'spatial_errors.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='working/best_rwanda_aurora.pt')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--output', type=str, default='evaluation_results')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    config = KaggleConfig()
    model = RwandaAuroraLite(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create test loader
    from notebooks.rwanda_aurora_training import create_dataloaders
    _, test_loader = create_dataloaders(config)
    
    # Run evaluation
    evaluator = AdvancedEvaluator(model, device)
    results = evaluator.evaluate_all(test_loader, args.output)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
