# Production Training Guide - Rwanda Aurora

Complete guide for training a production-quality Rwanda weather forecasting model.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Data Acquisition](#data-acquisition)
3. [Training Options](#training-options)
4. [Hyperparameter Optimization](#hyperparameter-optimization)
5. [Model Evaluation](#model-evaluation)
6. [Monitoring](#monitoring)
7. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### Option 1: Kaggle (Recommended for GPU)

1. **Upload to Kaggle:**
   ```bash
   # Create Kaggle notebook from kaggle_gpu_training.ipynb
   # Enable GPU: Settings → Accelerator → GPU T4 x2
   ```

2. **Run training:**
   - Just run all cells in the notebook
   - Training time: ~10-30 minutes with GPU

3. **Download results:**
   - `best_rwanda_aurora.pt`
   - `training_history.png`

### Option 2: Local Training

```bash
# With CPU (slow, ~7 hours)
python notebooks/rwanda_aurora_training.py

# With GPU (if you have NVIDIA GPU)
CUDA_VISIBLE_DEVICES=0 python notebooks/rwanda_aurora_training.py
```

---

##️ Data Acquisition

### Get Real ERA5 Data

1. **Register for CDS:**
   - Sign up: https://cds.climate.copernicus.eu/user/register
   - Get API key: https://cds.climate.copernicus.eu/api-how-to

2. **Configure credentials:**
   ```bash
   # Create ~/.cdsapirc
   echo "url: https://cds.climate.copernicus.eu/api/v2" > ~/.cdsapirc
   echo "key: YOUR_UID:YOUR_API_KEY" >> ~/.cdsapirc
   ```

3. **Download data:**
   ```bash
   # Install CDS API
   pip install cdsapi
   
   # Download Rwanda data (2020-2024)
   python scripts/download_era5_data.py --years 2020-2024 --output data/rwanda_era5.nc --validate --split
   ```

4. **Expected result:**
   - `data/rwanda_era5.nc` - Full dataset (~500 MB)
   - `data/rwanda_era5_train.nc` - Training set (70%)
   - `data/rwanda_era5_val.nc` - Validation set (15%)
   - `data/rwanda_era5_test.nc` - Test set (15%)

---

## 🎓 Training Options

### Standard Training

```bash
python notebooks/rwanda_aurora_training.py
```

**Configuration:**
- Epochs: 100
- Batch size: 2 (CPU) / 8 (GPU)
- Learning rate: 5e-5
- Early stopping: 15 epochs

### Training with Monitoring

```bash
# Enable TensorBoard monitoring
# Modify training script to include:
from training_monitor import TrainingMonitor
monitor = TrainingMonitor(log_dir='runs', experiment_name='my_experiment')

# View dashboard
tensorboard --logdir=runs
# Open: http://localhost:6006
```

---

## 🔍 Hyperparameter Optimization

### Using Weights & Biases

1. **Setup W&B:**
   ```bash
   pip install wandb
   wandb login
   ```

2. **Run sweep:**
   ```bash
   # Bayesian optimization (20 trials)
   python rwanda_sweep.py --method bayes --count 20 --gpu
   
   # Random search (faster)
   python rwanda_sweep.py --method random --count 10 --gpu
   ```

3. **View results:**
   - Dashboard: https://wandb.ai/your-username/rwanda-aurora-sweep
   - Best parameters saved to sweep results

### Search Space

Optimizes:
- **Architecture**: hidden_dim (128-512), num_layers (2-6), dropout (0-0.3)
- **Training**: learning_rate (1e-5 to 1e-3), batch_size (2-16), weight_decay
- **Optimizer**: AdamW, Adam, RAdam
- **Scheduler**: CosineAnnealing, ReduceLROnPlateau, OneCycleLR

---

## 📊 Model Evaluation

### Quick Evaluation

```bash
# Basic metrics
python evaluate_model.py --model working/best_rwanda_aurora.pt
```

### Comprehensive Evaluation

```bash
# Advanced metrics with baselines
python advanced_evaluation.py \
    --model working/best_rwanda_aurora.pt \
    --data data/rwanda_era5_test.nc \
    --output evaluation_results
```

**Metrics Calculated:**
- **Per-variable**: RMSE, MAE, Bias, Correlation, R²
- **Skill scores**: vs Persistence, vs Climatology
- **Temporal**: Error vs lead time
- **Spatial**: Error maps

**Output:**
- `evaluation_results/evaluation_results.json` - All metrics
- `evaluation_results/metrics_comparison.png` - Bar charts
- `evaluation_results/skill_scores.png` - Skill visualization
- `evaluation_results/error_vs_lead_time.png` - Temporal performance
- `evaluation_results/spatial_errors.png` - Spatial error maps

---

## 📈 Monitoring

### Real-time Training Monitor

**Setup:**
```python
from training_monitor import TrainingMonitor

monitor = TrainingMonitor(log_dir='runs', experiment_name='experiment_1')

# In training loop:
monitor.log_metrics(epoch, train_loss, val_loss, learning_rate)
monitor.log_predictions(inputs, predictions, targets, epoch)
monitor.log_gpu_stats()
```

**View Dashboard:**
```bash
tensorboard --logdir=runs
```

**Features:**
- Real-time loss curves
- Learning rate tracking
- Sample predictions visualization
- GPU memory usage
- Gradient/weight distributions

---

## ✅ Best Practices

### 1. Data Quality

- ✅ Use real ERA5 data (not synthetic)
- ✅ Validate data integrity after download
- ✅ Check for missing values
- ✅ Use proper train/val/test splits (temporal, no data leakage)

### 2. Training Strategy

**Start Simple:**
```
1. Baseline training (50 epochs)
2. Evaluate vs baselines
3. Identify weaknesses
4. Hyperparameter sweep
5. Final training (100-200 epochs)
```

**Key Settings:**
- Early stopping: Prevent overfitting
- Gradient clipping: Prevent exploding gradients
- Learning rate schedule: Improve convergence
- Mixed precision: Faster training on GPU

### 3. Evaluation

**Always compare against:**
- ✅ Persistence forecast (repeat last observation)
- ✅ Climatology (historical average)
- ✅ GFS/ECMWF (if available)

**Target Skill Scores:**
- Temperature: > 0.4
- Precipitation: > 0.3
- Wind: > 0.2

### 4. Model Selection

**Choose best model based on:**
```python
# Weighted average of metrics
score = (
    0.3 * temp_skill +
    0.3 * precip_skill +
    0.2 * wind_skill +
    0.2 * overall_correlation
)
```

### 5. Production Deployment

**Before deploying:**
- [ ] Train on full dataset (train + val)
- [ ] Validate on held-out test set
- [ ] Test inference speed
- [ ] Check memory requirements
- [ ] Create model card/documentation

---

## 🎯 Success Criteria

### Minimum Viable Model

- ✅ Better than persistence (skill > 0.1)
- ✅ RMSE < baseline
- ✅ No catastrophic failures
- ✅ Reasonable inference time (< 5 sec)

### Production-Ready Model

- ✅ Skill score > 0.3 (vs persistence)
- ✅ Skill score > 0.2 (vs climatology)
- ✅ Temperature RMSE < 2°C
- ✅ Consistent performance across seasons
- ✅ Stable training (no NaN/Inf)

### State-of-the-Art

- ✅ Comparable to GFS/ECMWF
- ✅ Skill score > 0.5
- ✅ Good extreme event prediction
- ✅ Reliable uncertainty estimates

---

## 📚 Troubleshooting

### Training Issues

**Loss is NaN:**
- Check learning rate (too high?)
- Enable gradient clipping
- Verify data normalization

**No convergence:**
- Reduce learning rate
- Increase batch size
- Check data quality

**Overfitting:**
- Reduce model size
- Add dropout
- More data augmentation
- Early stopping

### GPU Issues

**Out of memory:**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision (FP16)

**MPS trace trap (Apple Silicon):**
- Use CPU or CUDA
- Disable torch.compile
- Use vectorized normalization

---

## 📞 Next Steps

1. **Download ERA5 data** → `scripts/download_era5_data.py`
2. **Train on Kaggle GPU** → `notebooks/kaggle_gpu_training.ipynb`
3. **Run hyperparameter sweep** → `rwanda_sweep.py`
4. **Evaluate comprehensively** → `advanced_evaluation.py`
5. **Monitor training** → `training_monitor.py`
6. **Deploy model** → See `PRODUCTION_README.md`

---

**Good luck with your training! 🌤️🚀**
