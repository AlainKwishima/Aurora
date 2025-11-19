#!/usr/bin/env python3
"""
Minimal Aurora Hyperparameter Sweep Driver
Runs smoke tests without MLflow dependencies
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweeps"""
    variant: str
    n_trials: int
    smoke_test: bool = False
    output_dir: str = "sweep_results"
    
class MinimalAuroraSweeper:
    """Minimal Aurora hyperparameter sweeper without MLflow dependencies"""
    
    def __init__(self, config: SweepConfig):
        self.config = config
        self.results = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Define search spaces for each variant
        self.search_spaces = {
            "Aurora": {
                "learning_rate": [1e-4, 5e-4, 1e-3],
                "encoder_depths": [[2, 2, 6, 2], [2, 2, 8, 2], [2, 4, 8, 4]],
                "decoder_depths": [[2, 2, 6, 2], [2, 2, 8, 2], [2, 4, 8, 4]],
                "num_heads": [8, 12, 16],
                "embed_dim": [384, 512, 768],
                "patch_size": [4, 8, 16],
                "window_size": [4, 6, 8],
                "drop_path_rate": [0.1, 0.2, 0.3],
                "batch_size": [4, 8, 16]
            },
            "AuroraHighRes": {
                "learning_rate": [5e-5, 1e-4, 5e-4],
                "patch_size": [2, 4, 8],
                "embed_dim": [512, 768, 1024],
                "num_heads": [12, 16, 24],
                "encoder_depths": [[2, 2, 8, 2], [2, 4, 12, 4], [4, 4, 16, 8]],
                "decoder_depths": [[2, 2, 8, 2], [2, 4, 12, 4], [4, 4, 16, 8]],
                "window_size": [4, 6, 8],
                "drop_path_rate": [0.1, 0.2, 0.3],
                "batch_size": [2, 4, 8]
            },
            "AuroraAirPollution": {
                "learning_rate": [1e-4, 5e-4, 1e-3],
                "patch_size": [4, 8, 16],
                "embed_dim": [384, 512, 768],
                "num_heads": [8, 12, 16],
                "encoder_depths": [[2, 2, 6, 2], [2, 4, 8, 4], [4, 4, 12, 8]],
                "decoder_depths": [[2, 2, 6, 2], [2, 4, 8, 4], [4, 4, 12, 8]],
                "separate_perceiver": [True, False],
                "log_space_loss": [True, False],
                "window_size": [4, 6, 8],
                "drop_path_rate": [0.1, 0.2, 0.3],
                "batch_size": [4, 8, 16]
            },
            "AuroraWave": {
                "learning_rate": [5e-5, 1e-4, 5e-4],
                "embed_dim": [384, 512, 768],
                "num_heads": [8, 12, 16],
                "encoder_depths": [[2, 2, 6, 2], [2, 4, 8, 4], [4, 4, 12, 8]],
                "decoder_depths": [[2, 2, 6, 2], [2, 4, 8, 4], [4, 4, 12, 8]],
                "coastal_conditioning": [True, False],
                "spectral_loss_weight": [0.1, 0.5, 1.0],
                "window_size": [4, 6, 8],
                "drop_path_rate": [0.1, 0.2, 0.3],
                "batch_size": [4, 8, 16]
            }
        }
    
    def generate_random_params(self, variant: str) -> Dict[str, Any]:
        """Generate random hyperparameters for a variant"""
        import random
        
        search_space = self.search_spaces.get(variant, self.search_spaces["Aurora"])
        params = {}
        
        for param_name, choices in search_space.items():
            if isinstance(choices[0], (int, float)) and len(choices) > 2:
                # For numeric ranges, interpolate between min/max
                if isinstance(choices[0], float):
                    params[param_name] = random.uniform(choices[0], choices[-1])
                else:
                    params[param_name] = random.randint(choices[0], choices[-1])
            else:
                # For discrete choices, pick randomly
                params[param_name] = random.choice(choices)
        
        return params
    
    def simulate_objective(self, params: Dict[str, Any], variant: str) -> Dict[str, float]:
        """Simulate objective function with physics-aware metrics"""
        
        # Simulate training time based on model complexity
        complexity_score = (
            params.get("embed_dim", 512) / 100 +
            sum(params.get("encoder_depths", [2, 2, 6, 2])) +
            sum(params.get("decoder_depths", [2, 2, 6, 2])) +
            params.get("num_heads", 12) / 2 +
            params.get("batch_size", 8) / 2
        )
        
        # Simulate performance metrics (higher is better for accuracy, lower for error)
        base_accuracy = 0.75
        accuracy = base_accuracy + (0.2 * (1 - complexity_score / 50)) + (0.1 * (1 - params.get("learning_rate", 0.001) / 0.01))
        
        # Variant-specific adjustments
        if variant == "AuroraHighRes":
            accuracy += 0.05  # High-res models typically better accuracy
        elif variant == "AuroraAirPollution":
            accuracy += 0.03  # Specialized for pollution prediction
        elif variant == "AuroraWave":
            accuracy += 0.02  # Specialized for wave prediction
        
        # Add some noise to simulate real-world variance
        import random
        accuracy += random.uniform(-0.05, 0.05)
        accuracy = max(0.1, min(0.99, accuracy))  # Clamp to reasonable range
        
        # Calculate related metrics
        precision = accuracy * 0.95
        recall = accuracy * 0.92
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Simulate training time (seconds)
        training_time = 300 + complexity_score * 20 + random.uniform(-50, 50)
        
        # Simulate memory usage (GB)
        memory_gb = 2 + complexity_score * 0.5 + random.uniform(-0.5, 0.5)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "training_time": training_time,
            "memory_gb": max(0.5, memory_gb)
        }
    
    def run_trial(self, trial_num: int) -> Dict[str, Any]:
        """Run a single hyperparameter trial"""
        logger.info(f"Starting trial {trial_num + 1}/{self.config.n_trials}")
        
        # Generate random hyperparameters
        params = self.generate_random_params(self.config.variant)
        
        # Simulate objective evaluation
        start_time = time.time()
        metrics = self.simulate_objective(params, self.config.variant)
        end_time = time.time()
        
        trial_result = {
            "trial_num": trial_num + 1,
            "params": params,
            "metrics": metrics,
            "duration": end_time - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Trial {trial_num + 1} completed: accuracy={metrics['accuracy']:.4f}, "
                   f"f1_score={metrics['f1_score']:.4f}, training_time={metrics['training_time']:.1f}s")
        
        return trial_result
    
    def run_sweep(self):
        """Run the complete hyperparameter sweep"""
        logger.info(f"Starting {self.config.n_trials} trials for {self.config.variant}")
        
        sweep_start = time.time()
        
        for trial_num in range(self.config.n_trials):
            try:
                result = self.run_trial(trial_num)
                self.results.append(result)
                
                # Save intermediate results
                self.save_results()
                
            except Exception as e:
                logger.error(f"Trial {trial_num + 1} failed: {e}")
                continue
        
        sweep_end = time.time()
        total_time = sweep_end - sweep_start
        
        # Generate summary
        summary = self.generate_summary()
        
        logger.info(f"Sweep completed in {total_time:.1f}s")
        logger.info(f"Best trial: #{summary['best_trial']['trial_num']} "
                   f"(accuracy={summary['best_trial']['metrics']['accuracy']:.4f})")
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate sweep summary"""
        if not self.results:
            return {"error": "No successful trials"}
        
        # Find best trial by accuracy
        best_trial = max(self.results, key=lambda x: x["metrics"]["accuracy"])
        
        # Calculate statistics
        accuracies = [r["metrics"]["accuracy"] for r in self.results]
        f1_scores = [r["metrics"]["f1_score"] for r in self.results]
        training_times = [r["metrics"]["training_time"] for r in self.results]
        
        summary = {
            "variant": self.config.variant,
            "total_trials": len(self.results),
            "best_trial": best_trial,
            "statistics": {
                "accuracy": {
                    "mean": sum(accuracies) / len(accuracies),
                    "min": min(accuracies),
                    "max": max(accuracies),
                    "std": (sum((x - sum(accuracies)/len(accuracies))**2 for x in accuracies) / len(accuracies))**0.5
                },
                "f1_score": {
                    "mean": sum(f1_scores) / len(f1_scores),
                    "min": min(f1_scores),
                    "max": max(f1_scores),
                    "std": (sum((x - sum(f1_scores)/len(f1_scores))**2 for x in f1_scores) / len(f1_scores))**0.5
                },
                "training_time": {
                    "mean": sum(training_times) / len(training_times),
                    "min": min(training_times),
                    "max": max(training_times),
                    "total": sum(training_times)
                }
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary
    
    def save_results(self):
        """Save results to JSON files"""
        # Save all results
        results_file = os.path.join(self.config.output_dir, f"{self.config.variant}_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary
        summary = self.generate_summary()
        summary_file = os.path.join(self.config.output_dir, f"{self.config.variant}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Minimal Aurora Hyperparameter Sweeps")
    parser.add_argument("--variant", type=str, required=True, 
                       choices=["Aurora", "AuroraHighRes", "AuroraAirPollution", "AuroraWave"],
                       help="Aurora model variant to sweep")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--output-dir", type=str, default="sweep_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Adjust for smoke test
    if args.smoke_test:
        args.n_trials = min(args.n_trials, 5)
        logger.info("Running in smoke test mode")
    
    # Create config
    config = SweepConfig(
        variant=args.variant,
        n_trials=args.n_trials,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir
    )
    
    # Run sweep
    sweeper = MinimalAuroraSweeper(config)
    summary = sweeper.run_sweep()
    
    # Print final results
    print("\n" + "="*60)
    print(f"SWEEP COMPLETED FOR {args.variant}")
    print("="*60)
    print(f"Total trials: {summary['total_trials']}")
    print(f"Best accuracy: {summary['best_trial']['metrics']['accuracy']:.4f}")
    print(f"Best F1-score: {summary['best_trial']['metrics']['f1_score']:.4f}")
    print(f"Best params: {json.dumps(summary['best_trial']['params'], indent=2)}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()