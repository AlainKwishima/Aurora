## Scope and Objectives

* Optimize Aurora’s four specialized versions for forecasting accuracy, robustness, and efficiency.

* Strengthen data pipeline quality, add augmentation and validation, and systematize benchmarking.

* Design scalable deployment, monitoring, versioning, and A/B testing.

* Deliver tuned weights, benchmark reports, architecture diagrams, documentation, and automated pipelines.

## Current Codebase Anchors

* Core model entry: `aurora/model/aurora.py:40` (`class Aurora`)

* Specialized variants: `AuroraHighRes` `aurora/model/aurora.py:619`, `AuroraAirPollution` `aurora/model/aurora.py:641`, `AuroraWave` `aurora/model/aurora.py:799`

* Multi-step forecasting: `aurora/rollout.py:14` (`def rollout`)

* Training (production-style): `rwanda/trainer.py:266` (`class KaggleTrainer`)

* Data loaders and preprocessing: `rwanda/data_processing.py:643` (`def create_dataloaders`)

* Foundry model registry/mapping: `aurora/foundry/common/model.py:21` (`class Model` + variants)

* MLflow packaging: `package_mlflow.py:28-30`

## Phase 1: Model Architecture Analysis & Benchmarking

* Audit architecture/hyperparameters per variant in `aurora/model/aurora.py` (window sizes, depths, heads, patch size, LoRA usage, level conditioning, modulation heads).

* Extract current metrics and loss composition from `rwanda/trainer.py` and `rwanda/evaluation.py` for a uniform metrics suite: RMSE, MAE, bias, correlation, extreme-event skills, lead-time curves.

* Establish baselines against state-of-the-art (GraphCast, Pangu-Weather, FourCastNet, CAMS baselines, WAVEWATCH III for waves) with matched datasets and lead times.

* Define variant-specific target metrics (e.g., 3–5 day CRPS improvements for medium-res; local extreme detection metrics for high-res; chemical species RMSE/bias for pollution; coastal wave height/period MAE).

## Phase 2: Data Pipeline Optimization

* Data audit: quantify coverage, gaps, NaNs, outliers, time/space inconsistencies; integrate checks in `rwanda/data_processing.py:643` pipeline and `aurora/batch.py` normalization.

* Preprocessing refinements: consistent regridding, de-seasonalization options, robust scaling with climatology-aware statistics, angle handling (sin/cos) aligned to `AuroraWave` design.

* Augmentation (physics-aware): spatial jitter/crop, temporal stride jitter, noise within physical bounds, wind-direction circular perturbations, balanced extreme events sampling.

* Automated data validation: implement dataset validators (shape, variable presence, ranges, monotonic time, NaN thresholds) that fail-fast in data loader construction.

## Phase 3: Specialized Version Enhancements

* Medium-resolution (3–5 day):

  * Tune `window_size`, `encoder/decoder_depths`, and `latent_levels` for longer horizons; evaluate increasing `max_history_size` in `Aurora.rollout`.

  * Add curriculum rollout (short→long horizon) and loss reweighting by lead time for stability.

* High-resolution (local features):

  * Optimize `patch_size`/Swin3D stride for sharper feature extraction; introduce local attention bias; refine `encoder_num_heads` for spatial granularity.

  * Distill/quantize for real-time: enable mixed precision thoroughly; prune decoder MLPs; optional LoRA for on-edge adaptation.

* Air pollution (chemical transport):

  * Strengthen separation heads (keep `separate_perceiver` for species); add positivity constraints and log-space loss variants; difference-prediction for spikey channels is retained.

  * Emission attribution: multi-source conditioning (inventory indices) and spatial regularizers; evaluate temporal lag features.

* Ocean waves (coastal):

  * Improve angle/density handling consistent with `AuroraWave`; add coastal mask conditioning; refine period estimation via spectral loss.

  * Couple wind-driven features with shallow-water corrections near coast.

## Phase 4: Training Strategy & Hyperparameter Tuning

* Unified training harness: reuse `KaggleTrainer` (`rwanda/trainer.py:266`) with configurable loss mixes (MSE/MAE/Huber + spectral/physics) and AMP/accumulation/clipping.

* Hyperparameter sweeps (Optuna or internal search) on learning rate, weight decay, scheduler parameters (`CosineAnnealingWarmRestarts`), LoRA modes, depth/head counts.

* Early stopping by lead-time skill, checkpointing best per variant, and ensembling top checkpoints for robustness.

## Phase 5: Deployment Infrastructure

* MLflow model packaging and registry: extend `package_mlflow.py:28-30` to package each variant; add version tags and metadata.

* Foundry integration: ensure `aurora/foundry/common/model.py:21` mappings include versioned names; add A/B routing toggles and canary deploy logic in server wrapper.

* Scalable serving: batch inference and streaming paths; define autoscaling policies; cache static injections; enable mixed-precision inference.

* Monitoring dashboards: collect inference latencies, error rates, and live metrics (RMSE/MAE by lead time) with alerts.

## Phase 6: Testing & Validation

* Unit tests: batch normalization/regridding, angle transforms, positivity enforcement, NaN handling.

* Integration tests: multi-step `rollout` (`aurora/rollout.py:14`), variant-specific forward passes, checkpoint load tests.

* Continuous evaluation: nightly jobs on real-world data; rolling metrics and drift detection; edge case suites (extremes, missing channels, irregular grids).

* Explainability: saliency/occlusion and attention map extraction for operational forecasting; per-variable contribution summaries.

## Phase 7: Documentation & Maintenance

* Technical docs per variant (architecture, inputs/outputs, training settings, metrics, deployment notes).

* Model cards: capabilities, limitations, datasets, fairness/ethics notes, intended use.

* Automated retraining pipelines: data refresh, validation, training, evaluation, packaging, and deployment promotion gates.

## Deliverables

* Optimized weights for medium-res, high-res, air pollution, and ocean waves.

* Performance benchmark reports with baselines and lead-time curves.

* Deployment architecture diagrams and A/B testing design.

* Documentation suite and model cards.

* Automated training and evaluation pipelines with scheduled runs.

## Milestones

* Week 1–2: Analysis, baselines, data audit; initial pipeline validators.

* Week 3–4: Augmentations and training harness unification; first tuning sweeps.

* Week 5–6: Variant-specific enhancements; produce optimized weights and benchmarks.

* Week 7: Deployment packaging, A/B setup, monitoring.

* Week 8: Tests, explainability, documentation, automation.

## Approval

* Confirm this plan to proceed with implementation and tuning across the four specialized models.

