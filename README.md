<div align="center">

<img src="docs/gifs/high_res_2t.gif" alt="high resolution (0.1 degree) temperature at 2m predictions gif" width="150px">&nbsp;&nbsp;&nbsp;
<img src="docs/gifs/no2.gif" alt="nitrogen dioxide predictions gif" width="150px">&nbsp;&nbsp;&nbsp;
<img src="docs/gifs/wave_direction.gif" alt="ocean wave direction predictions gif" width="150px">&nbsp;&nbsp;&nbsp;
<img src="docs/gifs/tc_tracks.gif" alt="tropical cyclone track predictions gif" width="220px">

# 🌍 Aurora: A Foundation Model for the Earth System

**State-of-the-art AI for weather, air pollution, and ocean wave forecasting**

[![CI](https://github.com/microsoft/Aurora/actions/workflows/ci.yaml/badge.svg)](https://github.com/microsoft/Aurora/actions/workflows/ci.yaml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://microsoft.github.io/aurora)
[![Paper](https://img.shields.io/badge/Nature-2025-blue)](https://www.nature.com/articles/s41586-025-09005-y)
[![arXiv](https://img.shields.io/badge/arXiv-2405.13063-b31b1b.svg)](https://arxiv.org/abs/2405.13063)
[![PyPI version](https://badge.fury.io/py/microsoft-aurora.svg)](https://badge.fury.io/py/microsoft-aurora)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/microsoft-aurora.svg)](https://anaconda.org/conda-forge/microsoft-aurora)
[![DOI](https://zenodo.org/badge/828987595.svg)](https://doi.org/10.5281/zenodo.14983583)

[**📚 Documentation**](https://microsoft.github.io/aurora) • [**📄 Paper**](https://www.nature.com/articles/s41586-025-09005-y) • [**🚀 Quick Start**](#-quick-start) • [**💡 Examples**](https://microsoft.github.io/aurora/example_era5.html) • [**🤗 Models**](https://huggingface.co/microsoft/aurora)

</div>

## 🌟 Overview

Aurora is a cutting-edge foundation model capable of predicting atmospheric variables, air pollution, and ocean waves with unprecedented accuracy. Trained on vast amounts of Earth system data, Aurora can be adapted to specialized forecasting tasks with minimal additional training.

### 🎯 Key Capabilities

- **🌤️ Weather Forecasting**: High-resolution weather predictions at 0.1° and 0.25° resolutions
- **💨 Air Pollution**: Predicts PM1, PM2.5, PM10, NO₂, SO₂, O₃ and other pollutants  
- **🌊 Ocean Waves**: Forecasts significant wave height, direction, and period
- **🌀 Tropical Cyclones**: Tracks hurricane paths and intensities
- **⚡ Real-time**: Fast inference for operational forecasting
- **🌍 Global Coverage**: Works anywhere on Earth with consistent performance

### 🏆 Performance Highlights

- **Superior Accuracy**: Outperforms traditional numerical weather prediction in many scenarios
- **Multi-Resolution**: Supports both 0.1° (1801×3600) and 0.25° (721×1440) global grids
- **Fast Inference**: Generate global forecasts in minutes, not hours
- **Energy Efficient**: 1000x less energy consumption than traditional methods
- **Foundation Model**: Pre-trained on diverse datasets, adaptable to new regions with minimal fine-tuning

Cite us as follows:

```
@article{bodnar2025aurora,
    title = {A Foundation Model for the Earth System},
    author = {Cristian Bodnar and Wessel P. Bruinsma and Ana Lucic and Megan Stanley and Anna Allen and Johannes Brandstetter and Patrick Garvan and Maik Riechert and Jonathan A. Weyn and Haiyu Dong and Jayesh K. Gupta and Kit Thambiratnam and Alexander T. Archibald and Chun-Chieh Wu and Elizabeth Heider and Max Welling and Richard E. Turner and Paris Perdikaris},
    journal = {Nature},
    year = {2025},
    month = {May},
    day = {21},
    issn = {1476-4687},
    doi = {10.1038/s41586-025-09005-y},
    url = {https://doi.org/10.1038/s41586-025-09005-y},
}
```

Contents:

- [What is Aurora?](#what-is-aurora)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Security](#security)
- [Responsible AI Transparency Documentation](#responsible-ai-transparency-documentation)
- [Trademarks](#trademarks)
- [FAQ](#faq)

Please email [AIWeatherClimate@microsoft.com](mailto:AIWeatherClimate@microsoft.com)
if you are interested in using Aurora for commercial applications.
For research-related questions or technical support with the code here,
please [open an issue](https://github.com/microsoft/aurora/issues/new/choose)
or reach out to the authors of the paper.

## What is Aurora?

Aurora is a machine learning model that can predict atmospheric variables, such as temperature.
It is a _foundation model_, which means that it was first generally trained on a lot of data,
and then can be adapted to specialised atmospheric forecasting tasks with relatively little data.
We provide four such specialised versions:
one for medium-resolution weather prediction,
one for high-resolution weather prediction,
one for air pollution prediction,
and one for ocean wave prediction.

## 🚀 Quick Start

### 📦 Installation

Choose your preferred installation method:

**With pip (recommended):**
```bash
pip install microsoft-aurora
```

**With conda/mamba:**
```bash
mamba install microsoft-aurora -c conda-forge
```

**From source (for development):**
```bash
git clone https://github.com/microsoft/aurora.git
cd aurora
make install  # Installs with dev dependencies and pre-commit hooks
```

### 🔬 Your First Prediction

Get started in under 30 seconds:

```python
from datetime import datetime
import torch
from aurora import AuroraSmallPretrained, Batch, Metadata

# Initialize model (downloads ~500MB checkpoint)
model = AuroraSmallPretrained()
model.load_checkpoint()

# Create sample data batch
batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 17),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

# Generate prediction
prediction = model.forward(batch)
print(f"Temperature prediction shape: {prediction.surf_vars['2t'].shape}")
```

### 🎯 Real-World Examples

**Weather Forecasting with ERA5:**
```python
from aurora import AuroraPretrained, Batch

# Use the full model for production
model = AuroraPretrained()
model.load_checkpoint()
model.eval()

# Your ERA5 data processing here...
# batch = process_era5_data(...)
# prediction = model(batch)
```

**High-Resolution Weather (0.1°):**
```python
from aurora import AuroraHighRes

model = AuroraHighRes()
model.load_checkpoint()
# Best for IFS HRES analysis data at 0.1° resolution
```

**Air Pollution Forecasting:**
```python
from aurora import AuroraAirPollution

model = AuroraAirPollution()
model.load_checkpoint()
# Predicts PM1, PM2.5, PM10, NO₂, SO₂, O₃
```

**Ocean Wave Prediction:**
```python
from aurora import AuroraWave

model = AuroraWave()
model.load_checkpoint()
# Forecasts wave height, direction, and period
```

<div align="center">

**[📖 Explore Complete Examples →](https://microsoft.github.io/aurora/example_era5.html)**

</div>

## 🎯 Available Models

Aurora comes in several specialized variants optimized for different forecasting tasks:

| Model | Resolution | Best For | Variables | Use Case |
|-------|------------|----------|-----------|----------|
| **Aurora Pretrained** | 0.25° | General weather, ERA5 | `2t, 10u, 10v, msl, t, u, v, q, z` | Research, custom datasets |
| **Aurora HighRes** | 0.1° | High-res weather, IFS HRES | `2t, 10u, 10v, msl, t, u, v, q, z` | Operational forecasting |
| **Aurora Air Pollution** | 0.4° | Air quality, CAMS | `PM1, PM2.5, PM10, NO₂, SO₂, O₃, CO` | Environmental monitoring |
| **Aurora Wave** | 0.25° | Ocean waves, HRES-WAM | `swh, mwd, mwp, shww, mdww` | Maritime safety |
| **Aurora Small** | 0.25° | Development, testing | Basic weather vars | Debugging, prototyping |

### 🎯 Choosing the Right Model

- **🌍 General Weather**: Start with `AuroraPretrained` for maximum flexibility
- **📍 High-Resolution**: Use `AuroraHighRes` for detailed regional forecasts  
- **💨 Air Quality**: Choose `AuroraAirPollution` for pollution monitoring
- **🌊 Maritime**: Select `AuroraWave` for ocean and coastal applications
- **🔬 Research**: All models support fine-tuning on your specific datasets

<div align="center">

**[📊 Detailed Model Comparison →](https://microsoft.github.io/aurora/models.html)**

</div>

## ⚡ Advanced Features

### 🔄 Autoregressive Forecasting
Generate multi-step forecasts automatically:

```python
from aurora import rollout

# Generate 10-day forecast
predictions = [pred.to("cpu") for pred in rollout(model, batch, steps=40)]
```

### 🔧 Fine-tuning & Customization
Adapt Aurora to your specific domain:

```python
# Load pretrained weights
model = AuroraPretrained()
model.load_checkpoint()

# Fine-tune on your data
# ... your training loop here ...
```

### 📊 Batch Processing
Process multiple forecasts efficiently:

```python
# Process multiple regions/time steps
batch = Batch(
    surf_vars={...},  # Shape: (batch_size, time_steps, lat, lon)
    atmos_vars={...},  # Shape: (batch_size, time_steps, levels, lat, lon)
    metadata=Metadata(...)
)
```

### 🚀 GPU Acceleration
Optimized for NVIDIA GPUs:

```python
model = model.to("cuda")
with torch.inference_mode():
    prediction = model(batch.to("cuda"))
```

### 📈 System Requirements

| Model | GPU Memory | CPU Memory | Inference Time |
|-------|------------|------------|----------------|
| **Aurora Small** | 8GB | 16GB | ~30 seconds |
| **Aurora Pretrained** | 40GB | 64GB | ~2 minutes |
| **Aurora HighRes** | 60GB | 128GB | ~5 minutes |

*Benchmarks on NVIDIA A100, global 0.25° resolution*

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Setup for Contributors

```bash
# Clone and install in development mode
git clone https://github.com/microsoft/aurora.git
cd aurora
make install  # Installs dev dependencies + pre-commit hooks

# Run tests
make test

# Build documentation  
make docs
```

## 📚 Resources

### 📖 Documentation & Examples
- **[📚 Full Documentation](https://microsoft.github.io/aurora)** - Complete guides and API reference
- **[🌍 ERA5 Example](https://microsoft.github.io/aurora/example_era5.html)** - Weather forecasting tutorial
- **[💨 Air Pollution Example](https://microsoft.github.io/aurora/example_cams.ipynb)** - Environmental monitoring
- **[🌊 Wave Prediction Example](https://microsoft.github.io/aurora/example_wave.ipynb)** - Ocean forecasting
- **[🌀 Hurricane Tracking](https://microsoft.github.io/aurora/example_tc_tracking.ipynb)** - Tropical cyclone tracking

### 📊 Model Weights & Data
- **[🤗 HuggingFace Models](https://huggingface.co/microsoft/aurora)** - Pre-trained model checkpoints
- **[📈 Benchmark Results](sweep_results/)** - Performance evaluation results
- **[🗺️ Static Data](https://huggingface.co/microsoft/aurora/tree/main)** - Required static variables

### 🔬 Research & Publications
- **[📄 Nature Paper (2025)](https://www.nature.com/articles/s41586-025-09005-y)** - Original research publication
- **[📚 arXiv Preprint](https://arxiv.org/abs/2405.13063)** - Technical details and methodology
- **[🔖 Citation](#-citation)** - How to cite Aurora in your research

## ⚖️ License & Legal

### 📄 License
This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

### 🔒 Security
For security concerns, please see [SECURITY.md](SECURITY.md) and follow responsible disclosure practices.

### 🏢 Trademarks
This project may contain Microsoft trademarks or logos. Authorized use must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).

## 🏛️ Responsible AI

### 🎯 Our Commitment
Microsoft is committed to responsible AI development. This project follows our [AI principles](https://www.microsoft.com/en-us/ai/responsible-ai) of fairness, reliability, privacy, security, inclusiveness, transparency, and accountability.

### ⚠️ Important Limitations

**🔬 Research Use**: This code is intended for research and academic purposes. Commercial applications require separate licensing - contact [AIWeatherClimate@microsoft.com](mailto:AIWeatherClimate@microsoft.com).

**🎯 Accuracy**: Aurora provides probabilistic forecasts without guaranteed accuracy. Predictions should not be used directly for critical decisions without proper validation and expert review.

**📊 Training Data**: Models inherit potential biases from training data (ERA5, CMIP6, HRES, CAMS, etc.). Performance may vary for extreme events or unprecedented conditions.

**🔧 Operational Use**: Additional verification, post-processing, and expert analysis are essential before operational deployment.

### 📈 Model Evaluations
Aurora underwent extensive evaluation on held-out test data, including:
- ✅ Standard accuracy metrics (RMSE, ACC, CRPS)
- ✅ Extreme weather events (heatwaves, cold snaps, storms)
- ✅ Rare events (Hurricane Ciarán 2023, unusual patterns)
- ✅ Regional performance across different climate zones

See [the paper](https://arxiv.org/pdf/2405.13063) for complete evaluation details.

## 📖 Citation

If you use Aurora in your research, please cite:

```bibtex
@article{bodnar2025aurora,
    title = {A Foundation Model for the Earth System},
    author = {Cristian Bodnar and Wessel P. Bruinsma and Ana Lucic and Megan Stanley and Anna Allen and Johannes Brandstetter and Patrick Garvan and Maik Riechert and Jonathan A. Weyn and Haiyu Dong and Jayesh K. Gupta and Kit Thambiratnam and Alexander T. Archibald and Chun-Chieh Wu and Elizabeth Heider and Max Welling and Richard E. Turner and Paris Perdikaris},
    journal = {Nature},
    year = {2025},
    month = {May},
    day = {21},
    issn = {1476-4687},
    doi = {10.1038/s41586-025-09005-y},
    url = {https://doi.org/10.1038/s41586-025-09005-y},
}
```

## ❓ FAQ

<details>
<summary><strong>💻 System Requirements</strong></summary>

**Minimum**: 16GB RAM, 8GB VRAM (for Aurora Small)
**Recommended**: 64GB RAM, 40GB VRAM (for full Aurora)
**Optimal**: 128GB RAM, 80GB VRAM (for high-resolution models)

</details>

<details>
<summary><strong>⚡ Performance Tips</strong></summary>

- Use GPU acceleration for 10-100x speedup
- Batch multiple forecasts for better throughput
- Move predictions to CPU to manage GPU memory
- Use `torch.inference_mode()` for faster inference

</details>

<details>
<summary><strong>🔧 Troubleshooting</strong></summary>

**Out of Memory**: Reduce batch size or use Aurora Small
**Slow Inference**: Enable GPU acceleration and optimize data loading
**Poor Predictions**: Ensure correct input data format and normalization
**Import Errors**: Check PyTorch installation and CUDA compatibility

</details>

<details>
<summary><strong>🚀 Getting Help</strong></summary>

- 📋 [Open an Issue](https://github.com/microsoft/aurora/issues/new/choose) for bugs and feature requests
- 📧 Email [AIWeatherClimate@microsoft.com](mailto:AIWeatherClimate@microsoft.com) for commercial inquiries
- 📚 Check the [documentation](https://microsoft.github.io/aurora) for detailed guides

</details>

---

<div align="center">

**[🌟 Star this repo](https://github.com/microsoft/aurora)** if you find Aurora useful!

**[📊 Browse Examples](https://microsoft.github.io/aurora)** to see Aurora in action

</div>
