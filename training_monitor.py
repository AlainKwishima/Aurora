"""
Real-Time Training Monitoring Dashboard
======================================

Interactive dashboard for monitoring Rwanda Aurora training in real-time.
Uses TensorBoard for visualization.

Usage:
    # In training script, add:
    from training_monitor import TrainingMonitor
    monitor = TrainingMonitor(log_dir='runs/experiment_1')
    
    # In training loop:
    monitor.log_metrics(epoch, train_loss, val_loss, learning_rate)
    monitor.log_model_stats(model)
    
    # View dashboard:
    tensorboard --logdir=runs
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import io
from PIL import Image


class TrainingMonitor:
    """Real-time training monitoring with TensorBoard."""
    
    def __init__(self, log_dir='runs', experiment_name=None):
        if experiment_name is None:
            experiment_name = f"rwanda_aurora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"📊 Training Monitor initialized")
        print(f"Log directory: {self.log_dir}")
        print(f"\nTo view dashboard, run:")
        print(f"  tensorboard --logdir={log_dir}")
        print(f"  Then open: http://localhost:6006\n")
        
        self.step = 0
        self.best_val_loss = float('inf')
        
    def log_metrics(self, epoch, train_loss=None, val_loss=None, learning_rate=None, **kwargs):
        """Log training metrics."""
        
        if train_loss is not None:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            
        if val_loss is not None:
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            # Track best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.writer.add_scalar('Loss/Best_Validation', val_loss, epoch)
        
        if learning_rate is not None:
            self.writer.add_scalar('Training/Learning_Rate', learning_rate, epoch)
        
        # Log any additional kwargs
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        self.step += 1
    
    def log_model_stats(self, model):
        """Log model statistics (gradients, weights)."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, self.step)
            self.writer.add_histogram(f'Weights/{name}', param, self.step)
    
    def log_predictions(self, inputs, predictions, targets, epoch, num_samples=4):
        """Log sample predictions as images."""
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
        
        variables = ['2t', 'tp', '10u', '10v', 'msl']
        
        for i in range(min(num_samples, inputs.shape[0])):
            # Input (last timestep)
            im1 = axes[i, 0].imshow(inputs[i, -1, :, :, 0].cpu().numpy(), cmap='viridis')
            axes[i, 0].set_title('Input (Temperature)')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Target (first forecast step)
            im2 = axes[i, 1].imshow(targets[i, 0, :, :, 0].cpu().numpy(), cmap='viridis')
            axes[i, 1].set_title('Target')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Prediction
            im3 = axes[i, 2].imshow(predictions[i, 0, :, :, 0].cpu().numpy(), cmap='viridis')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        image_array = np.array(image)
        
        # Log to tensorboard
        self.writer.add_image('Predictions/Samples', image_array, epoch, dataformats='HWC')
        
        plt.close()
    
    def log_gpu_stats(self):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            
            self.writer.add_scalar('System/GPU_Memory_Allocated_GB', memory_allocated, self.step)
            self.writer.add_scalar('System/GPU_Memory_Reserved_GB', memory_reserved, self.step)
    
    def log_data_distribution(self, data, tag, epoch):
        """Log data distribution histogram."""
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        
        self.writer.add_histogram(tag, data, epoch)
    
    def log_hparams(self, hparams, metrics):
        """Log hyperparameters and final metrics."""
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """Close the writer."""
        self.writer.close()


# Integration example for rwanda_aurora_training.py
def integrate_with_training():
    """Example integration code to add to training script."""
    code_example = '''
# Add to imports:
from training_monitor import TrainingMonitor

# Add to main() or train_model():
monitor = TrainingMonitor(log_dir='runs', experiment_name='rwanda_aurora_v1')

# In training loop (inside epoch loop):
monitor.log_metrics(
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    learning_rate=optimizer.param_groups[0]['lr']
)

# Periodically (every N epochs):
if epoch % 5 == 0:
    monitor.log_predictions(inputs, predictions, targets, epoch)
    monitor.log_model_stats(model)
    monitor.log_gpu_stats()

# After training:
monitor.log_hparams(
    hparams={
        'learning_rate': config.training_config()['learning_rate'],
        'batch_size': config.training_config()['batch_size'],
        'hidden_dim': config.model_config()['hidden_dim'],
    },
    metrics={
        'final_val_loss': best_val_loss,
        'final_train_loss': train_loss,
    }
)
monitor.close()
'''
    return code_example


if __name__ == '__main__':
    print("Training Monitor Setup")
    print("=" * 60)
    print("\nTo integrate with your training:")
    print(integrate_with_training())
    print("\n" + "=" * 60)
    print("\nTo view dashboard:")
    print("  1. Run training with monitor enabled")
    print("  2. In terminal: tensorboard --logdir=runs")
    print("  3. Open: http://localhost:6006")
