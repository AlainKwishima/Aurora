"""
Rwanda Aurora Weather Forecasting System
========================================

A comprehensive, production-ready implementation of Aurora weather forecasting 
model fine-tuned for Rwanda's unique geographic and meteorological characteristics.

Main Components:
- config: Rwanda-specific configuration
- data_processing: Data pipeline and synthetic generation  
- model: Enhanced Aurora model with Rwanda optimizations
- trainer: Advanced training system with Kaggle optimization
- evaluation: Comprehensive evaluation and inference
- utils: Utilities and helper functions

Quick Start:
    from rwanda.config import RwandaConfig
    from rwanda.trainer import train_rwanda_model
    
    config = RwandaConfig()
    history = train_rwanda_model(config, data_path, model_type='lightning')
"""

__version__ = "1.0.0"
__author__ = "Rwanda Aurora Development Team"
__email__ = "rwanda-aurora@example.com"

# Core imports for easy access
from .config import RwandaConfig

try:
    from .model import (
        RwandaAurora,
        RwandaAuroraLightning,  
        RwandaAuroraEnsemble,
        create_rwanda_model,
        load_model_checkpoint,
        save_model_checkpoint
    )
    
    from .trainer import (
        KaggleTrainer,
        train_rwanda_model
    )
    
    from .evaluation import (
        RwandaMetrics,
        RwandaInference,
        evaluate_rwanda_model
    )
    
    from .data_processing import (
        RwandaWeatherDataset,
        create_dataloaders,
        generate_rwanda_synthetic_data
    )
    
    from .utils import (
        PerformanceMonitor,
        RwandaVisualizer,
        DataManager,
        cleanup_memory,
        setup_kaggle_environment
    )
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some Rwanda Aurora components not available: {e}")

# Package metadata
__all__ = [
    # Core
    'RwandaConfig',
    
    # Model
    'RwandaAurora',
    'RwandaAuroraLightning',
    'RwandaAuroraEnsemble',
    'create_rwanda_model',
    'load_model_checkpoint',
    'save_model_checkpoint',
    
    # Training
    'KaggleTrainer',
    'train_rwanda_model',
    
    # Evaluation
    'RwandaMetrics',
    'RwandaInference', 
    'evaluate_rwanda_model',
    
    # Data Processing
    'RwandaWeatherDataset',
    'create_dataloaders',
    'generate_rwanda_synthetic_data',
    
    # Utils
    'PerformanceMonitor',
    'RwandaVisualizer',
    'DataManager',
    'cleanup_memory',
    'setup_kaggle_environment'
]

def get_version():
    """Get package version"""
    return __version__

def print_system_info():
    """Print system information for debugging"""
    import sys
    import torch
    
    print("Rwanda Aurora Weather Forecasting System")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA: Not available")
    
    print("=" * 40)

# Show version on import
print(f"Rwanda Aurora v{__version__} - Weather forecasting for Rwanda")
print("For documentation and examples, see RWANDA_README.md")