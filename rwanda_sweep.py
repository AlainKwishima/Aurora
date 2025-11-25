"""
Rwanda Aurora Hyperparameter Sweep Configuration
==============================================

Optimized sweep configuration for the Rwanda Aurora model.
Uses Weights & Biases for tracking.

Usage:
    python rwanda_sweep.py --trials 20 --gpu
"""

import argparse
import torch
import wandb
from pathlib import Path
import numpy as np

# Import training components
from notebooks.rwanda_aurora_training import KaggleConfig, RwandaAuroraLite, train_model, create_dataloaders


# Rwanda-specific search space
RWANDA_SEARCH_SPACE = {
    # Model architecture
    'hidden_dim': {
        'values': [128, 256, 384, 512]
    },
    'num_layers': {
        'values': [2, 3, 4, 5, 6]
    },
    'dropout': {
        'min': 0.0,
        'max': 0.3
    },
    
    # Training hyperparameters
    'learning_rate': {
        'min': 1e-5,
        'max': 1e-3,
        'distribution': 'log_uniform'
    },
    'batch_size': {
        'values': [2, 4, 8, 16]
    },
    'weight_decay': {
        'min': 1e-6,
        'max': 1e-3,
        'distribution': 'log_uniform'
    },
    
    # Optimizer settings
    'optimizer': {
        'values': ['AdamW', 'Adam', 'RAdam']
    },
    'scheduler': {
        'values': ['CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'OneCycleLR']
    },
    
    # Data augmentation
    'noise_std': {
        'min': 0.0,
        'max': 0.05
    },
    'temporal_shift': {
        'values': [0, 1, 2]
    },
}


def create_sweep_config(method='bayes'):
    """Create W&B sweep configuration."""
    return {
        'method': method,  # 'bayes', 'random', 'grid'
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,
            's': 2
        },
        'parameters': RWANDA_SEARCH_SPACE
    }


def train_with_config(config=None):
    """Training function for W&B sweep."""
    
    # Initialize W&B run
    with wandb.init(config=config):
        config = wandb.config
        
        # Update KaggleConfig with sweep parameters
        class SweepConfig(KaggleConfig):
            @staticmethod
            def model_config():
                base = KaggleConfig.model_config()
                base.update({
                    'hidden_dim': config.hidden_dim,
                    'num_layers': config.num_layers,
                    'dropout': config.dropout,
                })
                return base
            
            @staticmethod
            def training_config():
                base = KaggleConfig.training_config()
                base.update({
                    'learning_rate': config.learning_rate,
                    'batch_size': config.batch_size,
                    'weight_decay': config.weight_decay,
                    'num_epochs': 50,  # Shorter for sweep
                    'early_stopping_patience': 10,
                })
                return base
        
        # Create model
        model = RwandaAuroraLite(SweepConfig())
        
        # Get device
        device = SweepConfig.get_device()
        model = model.to(device)
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(SweepConfig())
        
        # Train model
        try:
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=SweepConfig(),
                device=device
            )
            
            # Log best metrics
            best_val_loss = min(history['val_loss'])
            wandb.log({'best_val_loss': best_val_loss})
            
        except Exception as e:
            print(f"Training failed: {e}")
            wandb.log({'error': str(e)})
            raise


def main():
    parser = argparse.ArgumentParser(description='Rwanda Aurora Hyperparameter Sweep')
    parser.add_argument('--project', type=str, default='rwanda-aurora-sweep',
                       help='W&B project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='W&B entity name')
    parser.add_argument('--method', type=str, default='bayes',
                       choices=['bayes', 'random', 'grid'],
                       help='Sweep method')
    parser.add_argument('--count', type=int, default=20,
                       help='Number of runs')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    print("🔍 Rwanda Aurora Hyperparameter Sweep")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Count: {args.count}")
    print(f"GPU: {args.gpu and torch.cuda.is_available()}")
    print("=" * 60)
    
    # Create sweep
    sweep_config = create_sweep_config(method=args.method)
    sweep_id = wandb.sweep(
        sweep_config,
        project=args.project,
        entity=args.entity
    )
    
    print(f"\n✓ Sweep created: {sweep_id}")
    print(f"\nStarting sweep with {args.count} runs...")
    print("To view progress: https://wandb.ai\n")
    
    # Run sweep
    wandb.agent(sweep_id, function=train_with_config, count=args.count)
    
    print("\n✅ Sweep complete!")
    print(f"View results at: https://wandb.ai/{args.entity or 'your-entity'}/{args.project}")


if __name__ == '__main__':
    main()
