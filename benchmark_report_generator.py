#!/usr/bin/env python3
"""
Aurora Hyperparameter Sweep Benchmark Report Generator
Creates comprehensive benchmark reports with performance and efficiency metrics
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkReportGenerator:
    """Generates comprehensive benchmark reports from sweep results"""
    
    def __init__(self, results_dir: str = "sweep_results"):
        self.results_dir = Path(results_dir)
        self.variants = ["Aurora", "AuroraHighRes", "AuroraAirPollution", "AuroraWave"]
        
    def load_sweep_results(self) -> Dict[str, Dict]:
        """Load all sweep summary results"""
        results = {}
        
        for variant in self.variants:
            summary_file = self.results_dir / f"{variant}_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        results[variant] = json.load(f)
                    logger.info(f"Loaded results for {variant}")
                except Exception as e:
                    logger.error(f"Failed to load {variant} results: {e}")
                    results[variant] = None
            else:
                logger.warning(f"No results found for {variant}")
                results[variant] = None
                
        return results
    
    def generate_performance_comparison(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate performance comparison across variants"""
        comparison = {
            "best_accuracy": {},
            "best_f1_score": {},
            "efficiency_metrics": {},
            "training_efficiency": {},
            "rankings": {}
        }
        
        # Collect best metrics for each variant
        for variant, data in results.items():
            if data and "best_trial" in data:
                metrics = data["best_trial"]["metrics"]
                comparison["best_accuracy"][variant] = metrics["accuracy"]
                comparison["best_f1_score"][variant] = metrics["f1_score"]
                comparison["efficiency_metrics"][variant] = {
                    "training_time": metrics["training_time"],
                    "memory_gb": metrics["memory_gb"],
                    "accuracy_per_hour": metrics["accuracy"] / (metrics["training_time"] / 3600),
                    "accuracy_per_gb": metrics["accuracy"] / metrics["memory_gb"]
                }
        
        # Generate rankings
        if comparison["best_accuracy"]:
            comparison["rankings"]["accuracy"] = sorted(
                comparison["best_accuracy"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
        if comparison["best_f1_score"]:
            comparison["rankings"]["f1_score"] = sorted(
                comparison["best_f1_score"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
        if comparison["efficiency_metrics"]:
            comparison["rankings"]["efficiency"] = sorted(
                [(variant, metrics["accuracy_per_hour"]) for variant, metrics in comparison["efficiency_metrics"].items()],
                key=lambda x: x[1], 
                reverse=True
            )
        
        return comparison
    
    def generate_hyperparameter_analysis(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze hyperparameter trends and optimal configurations"""
        analysis = {
            "optimal_configs": {},
            "hyperparameter_trends": {},
            "architecture_insights": {}
        }
        
        for variant, data in results.items():
            if data and "best_trial" in data:
                params = data["best_trial"]["params"]
                metrics = data["best_trial"]["metrics"]
                
                # Store optimal configuration
                analysis["optimal_configs"][variant] = {
                    "hyperparameters": params,
                    "performance": {
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["f1_score"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"]
                    },
                    "efficiency": {
                        "training_time": metrics["training_time"],
                        "memory_gb": metrics["memory_gb"]
                    }
                }
                
                # Architecture analysis
                encoder_depths = params.get("encoder_depths", [])
                decoder_depths = params.get("decoder_depths", [])
                embed_dim = params.get("embed_dim", 0)
                num_heads = params.get("num_heads", 0)
                
                analysis["architecture_insights"][variant] = {
                    "total_encoder_layers": sum(encoder_depths) if encoder_depths else 0,
                    "total_decoder_layers": sum(decoder_depths) if decoder_depths else 0,
                    "model_complexity_score": embed_dim * (sum(encoder_depths) + sum(decoder_depths)),
                    "attention_heads": num_heads,
                    "embedding_dimension": embed_dim,
                    "efficiency_ratio": metrics["accuracy"] / (embed_dim * (sum(encoder_depths) + sum(decoder_depths))) if embed_dim > 0 else 0
                }
        
        return analysis
    
    def generate_recommendations(self, comparison: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Performance recommendations
        if comparison["rankings"].get("accuracy"):
            best_accuracy = comparison["rankings"]["accuracy"][0]
            recommendations.append({
                "category": "Performance",
                "priority": "High",
                "recommendation": f"Deploy {best_accuracy[0]} for highest accuracy ({best_accuracy[1]:.4f})",
                "rationale": f"{best_accuracy[0]} achieved the best accuracy in hyperparameter sweeps"
            })
        
        # Efficiency recommendations
        if comparison["rankings"].get("efficiency"):
            most_efficient = comparison["rankings"]["efficiency"][0]
            recommendations.append({
                "category": "Efficiency", 
                "priority": "Medium",
                "recommendation": f"Use {most_efficient[0]} for best training efficiency",
                "rationale": f"{most_efficient[0]} provides highest accuracy per training hour"
            })
        
        # Resource recommendations
        for variant, metrics in comparison["efficiency_metrics"].items():
            if metrics["memory_gb"] > 20:
                recommendations.append({
                    "category": "Resource Management",
                    "priority": "High",
                    "recommendation": f"Consider memory optimization for {variant} ({metrics['memory_gb']:.1f}GB)",
                    "rationale": f"{variant} requires significant memory resources"
                })
                
        # Architecture recommendations
        for variant, insights in analysis["architecture_insights"].items():
            if insights["efficiency_ratio"] < 0.001:
                recommendations.append({
                    "category": "Architecture",
                    "priority": "Medium",
                    "recommendation": f"Consider reducing model complexity for {variant}",
                    "rationale": f"Low efficiency ratio suggests over-parameterization"
                })
        
        return recommendations
    
    def generate_markdown_report(self, results: Dict[str, Dict], comparison: Dict[str, Any], 
                                analysis: Dict[str, Any], recommendations: List[Dict]) -> str:
        """Generate comprehensive markdown report"""
        
        report = f"""# Aurora Hyperparameter Sweep Benchmark Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents the results of comprehensive hyperparameter sweeps across four Aurora model variants:
- **Aurora** (Medium-resolution weather prediction)
- **AuroraHighRes** (High-resolution weather prediction)  
- **AuroraAirPollution** (Air pollution prediction)
- **AuroraWave** (Ocean wave prediction)

## Performance Rankings

### Accuracy Ranking
"""
        
        if comparison["rankings"].get("accuracy"):
            for i, (variant, accuracy) in enumerate(comparison["rankings"]["accuracy"], 1):
                report += f"{i}. **{variant}**: {accuracy:.4f}\n"
        
        report += "\n### F1-Score Ranking\n"
        if comparison["rankings"].get("f1_score"):
            for i, (variant, f1_score) in enumerate(comparison["rankings"]["f1_score"], 1):
                report += f"{i}. **{variant}**: {f1_score:.4f}\n"
        
        report += "\n### Training Efficiency Ranking\n"
        if comparison["rankings"].get("efficiency"):
            for i, (variant, efficiency) in enumerate(comparison["rankings"]["efficiency"], 1):
                report += f"{i}. **{variant}**: {efficiency:.4f} accuracy/hour\n"
        
        report += """
## Detailed Results by Variant

"""
        
        for variant, data in results.items():
            if data and "best_trial" in data:
                params = data["best_trial"]["params"]
                metrics = data["best_trial"]["metrics"]
                stats = data.get("statistics", {})
                
                report += f"""### {variant}

**Best Configuration:**
- Learning Rate: {params.get('learning_rate', 'N/A')}
- Embedding Dimension: {params.get('embed_dim', 'N/A')}
- Number of Heads: {params.get('num_heads', 'N/A')}
- Encoder Depths: {params.get('encoder_depths', 'N/A')}
- Decoder Depths: {params.get('decoder_depths', 'N/A')}

**Performance Metrics:**
- Accuracy: {metrics['accuracy']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}

**Efficiency Metrics:**
- Training Time: {metrics['training_time']:.1f} seconds
- Memory Usage: {metrics['memory_gb']:.1f} GB

**Statistics (across all trials):**
- Mean Accuracy: {stats.get('accuracy', {}).get('mean', 0):.4f} ± {stats.get('accuracy', {}).get('std', 0):.4f}
- Min Accuracy: {stats.get('accuracy', {}).get('min', 0):.4f}
- Max Accuracy: {stats.get('accuracy', {}).get('max', 0):.4f}

"""
        
        report += """## Architecture Analysis

"""
        
        for variant, insights in analysis["architecture_insights"].items():
            report += f"""### {variant}

- Total Encoder Layers: {insights['total_encoder_layers']}
- Total Decoder Layers: {insights['total_decoder_layers']}
- Model Complexity Score: {insights['model_complexity_score']:,}
- Efficiency Ratio: {insights['efficiency_ratio']:.6f}
- Embedding Dimension: {insights['embedding_dimension']}
- Attention Heads: {insights['attention_heads']}

"""
        
        report += """## Recommendations

"""
        
        for rec in recommendations:
            report += f"""### {rec['category']} ({rec['priority']} Priority)

**Recommendation:** {rec['recommendation']}

**Rationale:** {rec['rationale']}

"""
        
        report += """## Deployment Recommendations

Based on the sweep results, here are the recommended deployment strategies:

1. **Production Deployment**: Use the variant with highest accuracy that meets your latency requirements
2. **Development/Testing**: Use the most efficient variant for faster iteration
3. **Resource-Constrained Environments**: Consider the variant with best accuracy/memory ratio
4. **High-Resolution Applications**: AuroraHighRes provides superior performance for detailed predictions

## Next Steps

1. Validate top-performing configurations on held-out test sets
2. Conduct longer training runs with optimal hyperparameters
3. Implement model compression techniques for resource-constrained deployments
4. Set up continuous monitoring and A/B testing in production

---

*Report generated by Aurora Hyperparameter Sweep System*
"""
        
        return report
    
    def generate_json_report(self, results: Dict[str, Dict], comparison: Dict[str, Any], 
                           analysis: Dict[str, Any], recommendations: List[Dict]) -> Dict[str, Any]:
        """Generate machine-readable JSON report"""
        
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "variants_tested": list(results.keys()),
                "total_trials": sum(data.get("total_trials", 0) for data in results.values() if data)
            },
            "performance_rankings": comparison["rankings"],
            "best_configurations": analysis["optimal_configs"],
            "efficiency_metrics": comparison["efficiency_metrics"],
            "architecture_analysis": analysis["architecture_insights"],
            "recommendations": recommendations,
            "deployment_strategy": {
                "highest_accuracy": comparison["rankings"].get("accuracy", [["unknown", 0]])[0][0] if comparison["rankings"].get("accuracy") else "unknown",
                "most_efficient": comparison["rankings"].get("efficiency", [["unknown", 0]])[0][0] if comparison["rankings"].get("efficiency") else "unknown",
                "best_balance": self._find_best_balance_variant(comparison)
            }
        }
    
    def _find_best_balance_variant(self, comparison: Dict[str, Any]) -> str:
        """Find variant with best balance of accuracy and efficiency"""
        if not comparison["best_accuracy"] or not comparison["efficiency_metrics"]:
            return "unknown"
        
        # Simple scoring: accuracy * efficiency_ratio
        scores = {}
        for variant in comparison["best_accuracy"].keys():
            accuracy = comparison["best_accuracy"][variant]
            efficiency = comparison["efficiency_metrics"][variant]["accuracy_per_hour"]
            scores[variant] = accuracy * efficiency
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else "unknown"
    
    def generate_reports(self):
        """Generate all report formats"""
        logger.info("Loading sweep results...")
        results = self.load_sweep_results()
        
        if not results or all(v is None for v in results.values()):
            logger.error("No valid results found to generate report")
            return
        
        logger.info("Generating performance comparison...")
        comparison = self.generate_performance_comparison(results)
        
        logger.info("Analyzing hyperparameter trends...")
        analysis = self.generate_hyperparameter_analysis(results)
        
        logger.info("Generating recommendations...")
        recommendations = self.generate_recommendations(comparison, analysis)
        
        logger.info("Creating reports...")
        
        # Generate markdown report
        markdown_report = self.generate_markdown_report(results, comparison, analysis, recommendations)
        markdown_file = self.results_dir / "benchmark_report.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        logger.info(f"Markdown report saved to {markdown_file}")
        
        # Generate JSON report
        json_report = self.generate_json_report(results, comparison, analysis, recommendations)
        json_file = self.results_dir / "benchmark_report.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved to {json_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK REPORT GENERATED")
        print("="*60)
        print(f"Best Accuracy: {comparison['rankings'].get('accuracy', [['unknown', 0]])[0][0] if comparison['rankings'].get('accuracy') else 'unknown'}")
        print(f"Most Efficient: {comparison['rankings'].get('efficiency', [['unknown', 0]])[0][0] if comparison['rankings'].get('efficiency') else 'unknown'}")
        print(f"Best Balance: {json_report['deployment_strategy']['best_balance']}")
        print(f"Total Recommendations: {len(recommendations)}")
        print("="*60)
        
        return {
            "results": results,
            "comparison": comparison,
            "analysis": analysis,
            "recommendations": recommendations,
            "reports": {
                "markdown": str(markdown_file),
                "json": str(json_file)
            }
        }

def main():
    """Main function to generate benchmark reports"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Aurora benchmark reports")
    parser.add_argument("--results-dir", type=str, default="sweep_results", 
                       help="Directory containing sweep results")
    
    args = parser.parse_args()
    
    generator = BenchmarkReportGenerator(args.results_dir)
    reports = generator.generate_reports()
    
    return reports

if __name__ == "__main__":
    main()