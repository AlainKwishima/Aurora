## Scope

Optimize four Aurora variants (medium-res, high-res, air pollution, ocean waves) via targeted grid/random/BO sweeps, track experiments comprehensively, benchmark across datasets and efficiency metrics, package optimized weights to MLflow, and publish to Foundry with full metadata and model cards.

## Code Anchors

- Core models: `aurora/model/aurora.py:40` (`Aurora`), `:619` (`AuroraHighRes`), `:641` (`AuroraAirPollution`), `:799` (`AuroraWave`)
- Rollout: `aurora/rollout.py:14`
- Trainer: `rwanda/trainer.py:266` (`KaggleTrainer`)
- Data loaders: `rwanda/data_processing.py:643`
- Evaluation metrics: `rwanda/evaluation.py:1`
- Config: `rwanda/config.py:121` (TRAINING_CONFIG), `:161` (DATA_CONFIG), `:228` (METRICS_CONFIG)
- MLflow packaging: `package_mlflow.py:28` (save_model)
- Foundry registry: `aurora/foundry/common/model.py:21` (models), server wrapper: `aurora/foundry/server/mlflow_wrapper.py:115`

## Variant-Specific Search Spaces

Use physics- and memory-aware bounds; combine grid + random + Bayesian (Optuna) sampling.

- Medium-Res (Aurora)
  - Optimizer: `learning_rate` loguniform [8e-5, 4e-4], `weight_decay` {0, 1e-3, 1e-2}
  - Architecture: `encoder_depths` {(6,10,8),(6,8,8),(8,10,8)}, `decoder_depths` reversal; `num_heads` {12,16}; `embed_dim` {384,512}; `mlp_ratio` {3.0,4.0}; `dec_mlp_ratio` {2.0,3.0}
  - Regularization: `drop_path` [0.0,0.2], `drop_rate` [0.0,0.1]
  - Rollout/LoRA: `max_history_size` {2,3,4}, `lora_mode` {"single","from_second","all"}, `lora_steps` [20,60], `accumulation_steps` {2,4,8}
  - Batch/loader: `batch_size` {1,2} (if memory allows), `prefetch_factor` {2,4}

- High-Res (AuroraHighRes)
  - Patch/layout: `patch_size` {8,10,12}
  - Architecture: `encoder_depths` {(6,8,8),(6,10,8)}, `decoder_depths` {(8,8,6),(10,8,6)}, `embed_dim` {384,512}, `num_heads` {12,16}
  - Regularization/speed: `drop_path` [0.0,0.2], AMP on, optional decoder MLP pruning ratio {1.5,2.0}
  - Rollout: `max_history_size` {2,3}, `accumulation_steps` {2,4}

- Air Pollution (AuroraAirPollution)
  - Time/patch: `timestep` fixed 12h, `patch_size` {2,3,4}
  - Decoder: `separate_perceiver` subsets {species sets}, `modulation_heads` on/off
  - Positivity/log-loss: enable `positive_*`; test `log-space loss` flag {on,off}
  - Regularization: `drop_rate` [0.0,0.1], `drop_path` [0.0,0.15]
  - LoRA: `lora_mode` {"single","from_second"}, `lora_steps` [20,60]

- Ocean Waves (AuroraWave)
  - Angle/density kept; `lora_mode` {"from_second","all"}
  - Architecture: `embed_dim` {384,512}, `num_heads` {12,16}, `mlp_ratio` {3.0,4.0}
  - Regularization: `drop_path` [0.0,0.15]
  - Coastal conditioning flag {on,off}; spectral loss weight [0.0,0.1]

## Metrics & Classification Targets

- Regression: RMSE, MAE, bias, correlation, lead-time curves.
- Event classification (for accuracy/precision/recall/F1): thresholds from `METRICS_CONFIG` `rwanda/config.py:253` (e.g., precipitation >25mm/day, wind >15 m/s). Compute per-grid, per-time binary metrics.
- Efficiency: training duration, throughput; GPU memory (`torch.cuda.memory_allocated`), CPU RSS; inference latency (per-step rollout time).

## Sweeps Implementation

- Add a sweeps driver (Optuna): define `objective(config)` that instantiates model + `KaggleTrainer` (`rwanda/trainer.py:266`), runs limited epochs, returns validation metric (lead-time weighted RMSE + event F1). Support grid/random samplers.
- Parameter injection: pass trial params into `MODEL_CONFIG` and `TRAINING_CONFIG` (from `rwanda/config.py:121` and `:80`).
- Sampling strategies: grid (few architecture combos), random (10–30 trials), Bayesian (Optuna TPE, 30–60 trials).
- Early stopping: monitor validation; prune trials when plateau.

## Experiment Tracking

- MLflow logging: log params, metrics per epoch, final validation, resource utilization, training duration; attach artifacts (configs, plots, confusion matrices, lead-time curves).
- W&B optional: re-use existing hooks in `KaggleTrainer` to duplicate logs.
- Metadata: include dataset splits, seed, commit hash, variant name, compute environment.

## Benchmarking Reports

- Build report generator: compare best trials per variant across multiple test sets (ERA5, HRES, CAMS, WAM examples) with metrics tables and plots.
- Efficiency table: memory footprint (peak GPU), throughput (samples/sec), inference latency (ms/step), CPU fallback behavior.
- Save as artifacts: JSON + HTML/PNG plots; summarize into Markdown for publishing.

## Packaging to MLflow

- Use `package_mlflow.py:28` to produce pyfunc models for each optimized variant; embed `artifacts_versions.json` and best-config JSON.
- Artifacts: full configs (model/training/data), preprocessor description, normalization stats, usage examples.
- Versioning: name models with variant + date + trial ID; tag with dataset, thresholds, commit hash.

## Publishing to Foundry

- Configure `MLFLOW_TRACKING_URI` and credentials; register models with tags (variant, horizon, resolution, domain).
- Access control: set team/project ACLs in Foundry registry.
- Documentation: attach Markdown model cards per variant (capabilities, limitations, datasets, metrics, usage snippets).
- Optional A/B rollout: use server wrapper routing `aurora/foundry/server/mlflow_wrapper.py:89` (`nameA|nameB@p`).

## Deliverables

- Search configs and code to run sweeps per variant.
- Experiment logs with full metadata.
- Benchmark reports and plots.
- MLflow-packaged models and artifacts.
- Published models in Foundry with tags + docs.

## Execution Plan & Milestones

- Week 1: Implement sweeps driver; define search spaces; wire MLflow tracking; run smoke sweeps.
- Week 2–3: Full sweeps per variant; select top checkpoints; generate reports.
- Week 4: Package and publish to MLflow/Foundry; finalize docs + model cards.

## Notes

- Classification metrics apply to event detection; regression remains primary for continuous variables.
- Memory constraints dictate small batch sizes and AMP; prune search ranges if OOM occurs.

Confirm to proceed; I will implement the sweeps, run tuned experiments, produce reports, and publish all MLflow artifacts with documentation.