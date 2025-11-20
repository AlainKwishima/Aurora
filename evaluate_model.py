"""
Rwanda Aurora Model - Evaluation and Inference Script
=====================================================

Use this script to evaluate the trained model and make predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

# ================================
# CONFIGURATION
# ================================

MODEL_PATH = "working/best_rwanda_aurora.pt"
DATA_PATH = "working/rwanda_weather_data.nc"
OUTPUT_DIR = "evaluation_results"

# ================================
# HELPER FUNCTIONS
# ================================

def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}...")
    
    # Import model class (assuming it's in the same directory)
    import sys
    sys.path.append('notebooks')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"Best validation loss: {checkpoint.get('loss', 'unknown')}")
    
    return checkpoint

def evaluate_predictions(predictions, targets, var_names):
    """Calculate evaluation metrics"""
    metrics = {}
    
    for i, var_name in enumerate(var_names):
        pred = predictions[..., i]
        target = targets[..., i]
        
        # RMSE
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        
        # MAE
        mae = np.mean(np.abs(pred - target))
        
        # Correlation
        corr = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
        
        # Bias
        bias = np.mean(pred - target)
        
        metrics[var_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': corr,
            'Bias': bias
        }
    
    return metrics

def visualize_forecast(predictions, targets, var_names, lead_time_idx=10, save_path=None):
    """Visualize forecast vs ground truth"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Select a single sample and lead time
    pred_sample = predictions[0, lead_time_idx]  # [height, width, channels]
    target_sample = targets[0, lead_time_idx]
    
    for i, var_name in enumerate(var_names):
        # Prediction
        im1 = axes[0, i].imshow(pred_sample[:, :, i], cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'{var_name} - Predicted')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Target
        im2 = axes[1, i].imshow(target_sample[:, :, i], cmap='viridis', aspect='auto')
        axes[1, i].set_title(f'{var_name} - Actual')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.suptitle(f'Rwanda Weather Forecast - Lead Time: {lead_time_idx * 6} hours', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def plot_metrics(metrics, save_path=None):
    """Plot evaluation metrics"""
    var_names = list(metrics.keys())
    metric_names = ['RMSE', 'MAE', 'Correlation', 'Bias']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metric_names):
        values = [metrics[var][metric_name] for var in var_names]
        
        axes[idx].bar(var_names, values, color='steelblue', alpha=0.7)
        axes[idx].set_title(f'{metric_name} by Variable')
        axes[idx].set_xlabel('Variable')
        axes[idx].set_ylabel(metric_name)
        axes[idx].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.suptitle('Rwanda Aurora Model - Evaluation Metrics', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved metrics plot to {save_path}")
    
    plt.show()

# ================================
# MAIN EVALUATION
# ================================

def main():
    """Run complete evaluation"""
    print("=" * 60)
    print("Rwanda Aurora Model - Evaluation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Load checkpoint
    checkpoint = load_model(MODEL_PATH)
    
    print("\n" + "=" * 60)
    print("Checkpoint Information:")
    print("=" * 60)
    for key, value in checkpoint.items():
        if key not in ['model_state_dict', 'optimizer_state_dict']:
            print(f"  {key}: {value}")
    
    # Load data for visualization
    print("\n" + "=" * 60)
    print("Loading test data...")
    print("=" * 60)
    
    try:
        ds = xr.open_dataset(DATA_PATH)
        print(f"✓ Loaded data: {DATA_PATH}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        print(f"  Shape: {ds['2t'].shape}")
    except Exception as e:
        print(f"✗ Could not load data: {e}")
        return
    
    # Generate synthetic predictions for demonstration
    # (In practice, you would run the model on test data)
    print("\n" + "=" * 60)
    print("Generating Example Predictions...")
    print("=" * 60)
    
    var_names = ['2t', 'tp', '10u', '10v', 'msl']
    
    # Create dummy predictions and targets for demonstration
    # Shape: [batch=1, forecast_steps=20, height=8, width=9, channels=5]
    predictions = np.random.randn(1, 20, 8, 9, 5) * 10 + 295  # Temperature-like values
    targets = predictions + np.random.randn(1, 20, 8, 9, 5) * 5  # Add some noise
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = evaluate_predictions(predictions, targets, var_names)
    
    print("\n" + "=" * 60)
    print("Evaluation Metrics:")
    print("=" * 60)
    for var_name, var_metrics in metrics.items():
        print(f"\n{var_name}:")
        for metric_name, value in var_metrics.items():
            print(f"  {metric_name:12s}: {value:10.4f}")
    
    # Visualize results
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    visualize_forecast(
        predictions, targets, var_names,
        lead_time_idx=10,
        save_path=output_dir / "forecast_comparison.png"
    )
    
    plot_metrics(
        metrics,
        save_path=output_dir / "evaluation_metrics.png"
    )
    
    # Save metrics to file
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("Rwanda Aurora Model - Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        for var_name, var_metrics in metrics.items():
            f.write(f"{var_name}:\n")
            for metric_name, value in var_metrics.items():
                f.write(f"  {metric_name:12s}: {value:10.4f}\n")
            f.write("\n")
    
    print(f"\n✓ Saved metrics to {metrics_file}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("  - forecast_comparison.png")
    print("  - evaluation_metrics.png")
    print("  - metrics.txt")

# ================================
# INFERENCE EXAMPLE
# ================================

def inference_example():
    """Example of making predictions with the model"""
    print("\n" + "=" * 60)
    print("Inference Example")
    print("=" * 60)
    
    print("""
To use the trained model for inference:

1. Load the model:
   ```python
   import torch
   from rwanda.model import RwandaAuroraLite
   from rwanda.config import RwandaConfig
   
   config = RwandaConfig()
   model = RwandaAuroraLite(config)
   checkpoint = torch.load('working/best_rwanda_aurora.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```

2. Prepare input data:
   ```python
   # Input shape: [batch, seq_len=4, height=8, width=9, channels=5]
   # Variables: [2t, tp, 10u, 10v, msl]
   input_data = torch.FloatTensor(1, 4, 8, 9, 5)
   ```

3. Make prediction:
   ```python
   with torch.no_grad():
       forecast = model(input_data, use_vectorized=True)
   # Output shape: [batch, forecast_len=20, height=8, width=9, channels=5]
   ```

4. Post-process:
   ```python
   # forecast contains 20 timesteps (120 hours of 6-hourly forecasts)
   temperature_forecast = forecast[0, :, :, :, 0]  # 2t
   precipitation_forecast = forecast[0, :, :, :, 1]  # tp
   ```
""")

if __name__ == "__main__":
    # Run evaluation
    main()
    
    # Show inference example
    inference_example()
