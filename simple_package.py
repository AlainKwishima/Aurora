#!/usr/bin/env python3
import json
import os
from pathlib import Path
import time
from datetime import datetime

# Load configurations
configs = {}
results_dir = Path("sweep_results")
variants = ["Aurora", "AuroraHighRes", "AuroraAirPollution", "AuroraWave"]

for variant in variants:
    summary_file = results_dir / f"{variant}_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            if "best_trial" in data:
                configs[variant] = {
                    "hyperparameters": data["best_trial"]["params"],
                    "metrics": data["best_trial"]["metrics"]
                }

# Create packages
output_dir = Path("mlflow_packages")
output_dir.mkdir(exist_ok=True)

for variant, config in configs.items():
    package_dir = output_dir / f"aurora_{variant.lower()}"
    package_dir.mkdir(exist_ok=True)
    
    # Config
    config_data = {
        "variant": variant,
        "hyperparameters": config["hyperparameters"],
        "metrics": config["metrics"],
        "created": datetime.now().isoformat()
    }
    
    with open(package_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Model card
    card = f"""# {variant} Model
Accuracy: {config['metrics']['accuracy']:.4f}
F1-Score: {config['metrics']['f1_score']:.4f}
Training Time: {config['metrics']['training_time']:.1f}s
Memory: {config['metrics']['memory_gb']:.1f}GB
"""
    
    with open(package_dir / "model_card.md", 'w') as f:
        f.write(card)
    
    print(f"Created package for {variant}")

print(f"Packages created in {output_dir}")