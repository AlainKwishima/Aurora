"""
Aurora Hyperparameter Sweeps Driver

Unified driver for grid/random/Bayesian sweeps on Aurora variants.
- Uses Optuna for Bayesian optimization
- Tracks experiments via MLflow
- Supports early pruning and resource monitoring
- Generates benchmark reports

Usage:
    python aurora_sweeps.py --variant Aurora --trials 30 --sampler TPE
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import GridSampler, RandomSampler, TPESampler

# Import Aurora components
try:
    import aurora
    from aurora import Batch, rollout
    from rwanda.config import RwandaConfig
    from rwanda.trainer import KaggleTrainer
    from rwanda.data_processing import create_dataloaders
    from rwanda.evaluation import RwandaMetrics
except ImportError as e:
    print(f"Aurora import error: {e}")
    print("Using fallback implementations")
    aurora = None


# Search spaces per variant
SEARCH_SPACES = {
    "Aurora": {
        "learning_rate": {"type": "loguniform", "low": 8e-5, "high": 4e-4},
        "weight_decay": {"type": "categorical", "choices": [0.0, 1e-3, 1e-2]},
        "encoder_depths": {"type": "categorical", "choices": [(6, 10, 8), (6, 8, 8), (8, 10, 8)]},
        "decoder_depths": {"type": "categorical", "choices": [(8, 10, 6), (8, 8, 6), (10, 8, 6)]},
        "num_heads": {"type": "categorical", "choices": [12, 16]},
        "embed_dim": {"type": "categorical", "choices": [384, 512]},
        "mlp_ratio": {"type": "categorical", "choices": [3.0, 4.0]},
        "dec_mlp_ratio": {"type": "categorical", "choices": [2.0, 3.0]},
        "drop_path": {"type": "uniform", "low": 0.0, "high": 0.2},
        "drop_rate": {"type": "uniform", "low": 0.0, "high": 0.1},
        "max_history_size": {"type": "categorical", "choices": [2, 3, 4]},
        "lora_mode": {"type": "categorical", "choices": ["single", "from_second", "all"]},
        "lora_steps": {"type": "int", "low": 20, "high": 60},
        "accumulation_steps": {"type": "categorical", "choices": [2, 4, 8]},
        "batch_size": {"type": "categorical", "choices": [1, 2]},
        "prefetch_factor": {"type": "categorical", "choices": [2, 4]},
    },
    "AuroraHighRes": {
        "patch_size": {"type": "categorical", "choices": [8, 10, 12]},
        "encoder_depths": {"type": "categorical", "choices": [(6, 8, 8), (6, 10, 8)]},
        "decoder_depths": {"type": "categorical", "choices": [(8, 8, 6), (10, 8, 6)]},
        "embed_dim": {"type": "categorical", "choices": [384, 512]},
        "num_heads": {"type": "categorical", "choices": [12, 16]},
        "drop_path": {"type": "uniform", "low": 0.0, "high": 0.2},
        "max_history_size": {"type": "categorical", "choices": [2, 3]},
        "accumulation_steps": {"type": "categorical", "choices": [2, 4]},
        "decoder_mlp_prune_ratio": {"type": "categorical", "choices": [1.5, 2.0]},
    },
    "AuroraAirPollution": {
        "patch_size": {"type": "categorical", "choices": [2, 3, 4]},
        "separate_perceiver": {"type": "categorical", "choices": [
            ("co", "no", "no2", "go3", "so2"),
            ("co", "no2", "go3", "so2"),
            ("co", "no2", "go3"),
        ]},
        "modulation_heads": {"type": "categorical", "choices": [True, False]},
        "log_space_loss": {"type": "categorical", "choices": [True, False]},
        "drop_rate": {"type": "uniform", "low": 0.0, "high": 0.1},
        "drop_path": {"type": "uniform", "low": 0.0, "high": 0.15},
        "lora_mode": {"type": "categorical", "choices": ["single", "from_second"]},
        "lora_steps": {"type": "int", "low": 20, "high": 60},
    },
    "AuroraWave": {
        "embed_dim": {"type": "categorical", "choices": [384, 512]},
        "num_heads": {"type": "categorical", "choices": [12, 16]},
        "mlp_ratio": {"type": "categorical", "choices": [3.0, 4.0]},
        "drop_path": {"type": "uniform", "low": 0.0, "high": 0.15},
        "lora_mode": {"type": "categorical", "choices": ["from_second", "all"]},
        "coastal_conditioning": {"type": "categorical", "choices": [True, False]},
        "spectral_loss_weight": {"type": "uniform", "low": 0.0, "high": 0.1},
    },
}


def sample_param(trial: optuna.Trial, spec: Dict[str, Any]) -> Any:
    """Sample a parameter from Optuna trial based on spec."""
    if spec["type"] == "categorical":
        return trial.suggest_categorical("", spec["choices"])
    elif spec["type"] == "int":
        return trial.suggest_int("", spec["low"], spec["high"])
    elif spec["type"] == "uniform":
        return trial.suggest_float("", spec["low"], spec["high"])
    elif spec["type"] == "loguniform":
        return trial.suggest_float("", spec["low"], spec["high"], log=True)
    else:
        raise ValueError(f"Unknown param type: {spec['type']}")


def build_config(variant: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Build RwandaConfig from trial."""
    config = RwandaConfig()
    space = SEARCH_SPACES[variant]
    
    # Sample parameters
    params = {}
    for name, spec in space.items():
        params[name] = sample_param(trial, spec)
    
    # Update model config
    for key in ["encoder_depths", "decoder_depths", "num_heads", "embed_dim", "mlp_ratio", "dec_mlp_ratio", "drop_path", "drop_rate", "patch_size"]:
        if key in params:
            config.MODEL_CONFIG[key] = params[key]
    
    # Update training config
    for key in ["learning_rate", "weight_decay", "accumulation_steps", "batch_size", "prefetch_factor", "max_history_size", "lora_mode", "lora_steps"]:
        if key in params:
            config.TRAINING_CONFIG[key] = params[key]
    
    # Special handling
    if "decoder_mlp_prune_ratio" in params:
        config.MODEL_CONFIG["dec_mlp_ratio"] = params["decoder_mlp_prune_ratio"]
    if "log_space_loss" in params:
        config.LOSS_CONFIG["log_space_loss"] = params["log_space_loss"]
    if "spectral_loss_weight" in params:
        config.LOSS_CONFIG["spectral_loss_weight"] = params["spectral_loss_weight"]
    if "coastal_conditioning" in params:
        config.MODEL_CONFIG["coastal_conditioning"] = params["coastal_conditioning"]
    
    return config, params


def create_model(variant: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Create model instance for variant."""
    if variant == "Aurora":
        model = aurora.Aurora(**config.MODEL_CONFIG)
    elif variant == "AuroraHighRes":
        model = aurora.AuroraHighRes(**config.MODEL_CONFIG)
    elif variant == "AuroraAirPollution":
        model = aurora.AuroraAirPollution(**config.MODEL_CONFIG)
    elif variant == "AuroraWave":
        model = aurora.AuroraWave(**config.MODEL_CONFIG)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return model


def objective(trial: optuna.Trial, variant: str, data_path: str, device: torch.device, max_epochs: int = 5) -> float:
    """Optuna objective function."""
    
    # Build config
    config, params = build_config(variant, trial)
    
    # MLflow tracking
    mlflow.set_experiment(f"aurora_{variant}")
    with mlflow.start_run(run_name=f"trial_{trial.number}_{uuid.uuid4().hex[:8]}"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("variant", variant)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("device", str(device))
        
        try:
            # Create model
            model = create_model(variant, config)
            model = model.to(device)
            
            # Enable activation checkpointing for memory efficiency
            if hasattr(model, 'configure_activation_checkpointing'):
                model.configure_activation_checkpointing()
            
            # Create data loaders
            train_loader, val_loader = create_dataloaders(config, data_path)
            
            # Create trainer
            trainer = KaggleTrainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device
            )
            
            # Train for limited epochs (smoke test)
            start_time = time.time()
            history = trainer.train()
            training_time = time.time() - start_time
            
            # Get best validation loss
            best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            
            # Log metrics
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("training_time", training_time)
            
            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
                mlflow.log_metric("peak_gpu_memory_gb", memory_allocated)
            
            # Early pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return best_val_loss
            
        except Exception as e:
            mlflow.log_param("error", str(e))
            raise optuna.TrialPruned()


def run_sweep(variant: str, data_path: str, n_trials: int = 30, sampler_name: str = "TPE", max_epochs: int = 5):
    """Run hyperparameter sweep for variant."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create sampler
    if sampler_name == "TPE":
        sampler = TPESampler(seed=42)
    elif sampler_name == "Random":
        sampler = RandomSampler(seed=42)
    elif sampler_name == "Grid":
        # For grid, we need to create a limited space
        space = SEARCH_SPACES[variant]
        grid_params = {}
        for name, spec in space.items():
            if spec["type"] == "categorical":
                grid_params[name] = spec["choices"][:3]  # Limit to 3 choices
            elif spec["type"] in ["int", "uniform"]:
                grid_params[name] = [spec["low"], (spec["low"] + spec["high"]) / 2, spec["high"]]
            elif spec["type"] == "loguniform":
                grid_params[name] = [spec["low"], np.sqrt(spec["low"] * spec["high"]), spec["high"]]
        sampler = GridSampler(grid_params)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=f"aurora_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, variant, data_path, device, max_epochs),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Save results
    results_dir = Path("sweep_results") / variant
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best value: {best_trial.value}")
    print(f"Best params: {best_trial.params}")
    
    # Save study
    study_df = study.trials_dataframe()
    study_df.to_csv(results_dir / "trials.csv", index=False)
    
    # Save best params
    with open(results_dir / "best_params.json", "w") as f:
        json.dump(best_trial.params, f, indent=2)
    
    # Save study object
    joblib.dump(study, results_dir / "study.joblib")
    
    print(f"Sweep completed. Results saved to {results_dir}")
    return study


def main():
    parser = argparse.ArgumentParser(description="Aurora Hyperparameter Sweeps")
    parser.add_argument("--variant", type=str, required=True, choices=list(SEARCH_SPACES.keys()),
                       help="Model variant to tune")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--trials", type=int, default=30,
                       help="Number of trials")
    parser.add_argument("--sampler", type=str, default="TPE", choices=["TPE", "Random", "Grid"],
                       help="Sampler type")
    parser.add_argument("--max_epochs", type=int, default=5,
                       help="Max epochs per trial (smoke test)")
    parser.add_argument("--mlflow_uri", type=str, default="./mlruns",
                       help="MLflow tracking URI")
    
    args = parser.parse_args()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    # Run sweep
    study = run_sweep(
        variant=args.variant,
        data_path=args.data_path,
        n_trials=args.trials,
        sampler_name=args.sampler,
        max_epochs=args.max_epochs,
    )
    
    return study


if __name__ == "__main__":
    main()