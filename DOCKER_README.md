# Aurora Rwanda - Docker Guide

This guide explains how to build and run the Aurora Rwanda project using Docker. This ensures a consistent environment and easy deployment.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your machine.
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop).

## Quick Start

### 1. Build the Image

```bash
docker-compose build
```

### 2. Run Quick Inference

This runs the `quick_inference.py` script to generate a sample forecast.

```bash
docker-compose run --rm inference
```

Results will be saved to the `working/` directory on your host machine.

### 3. Train the Model

This runs the training script `notebooks/rwanda_aurora_training.py`.

```bash
docker-compose run --rm train
```

**Note:** Training on CPU (default in Docker unless NVIDIA GPU is configured) will be slower than on your Mac with MPS. If you have an NVIDIA GPU machine, uncomment the `deploy` section in `docker-compose.yml` to enable GPU support.

### 4. Evaluate the Model

Run the full evaluation suite.

```bash
docker-compose run --rm evaluate
```

Results will be saved to `evaluation_results/`.

### 5. Interactive Shell

Open a bash shell inside the container to run custom commands.

```bash
docker-compose run --rm shell
```

## Directory Structure

The Docker container maps the following directories from your host:

- `./working` -> `/app/working`: Where model checkpoints and training logs are saved.
- `./evaluation_results` -> `/app/evaluation_results`: Where evaluation plots and metrics are saved.

This means any file generated inside these directories in the container will persist on your machine.

## Troubleshooting

### "No module named rwanda"
Ensure the `rwanda/` directory exists in the project root and is being copied in the `Dockerfile`. The current configuration handles this.

### "Out of Memory"
If training crashes, try increasing the memory allocated to Docker in Docker Desktop settings.

### GPU Support
To use NVIDIA GPUs, you need:
1. NVIDIA drivers installed on the host.
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.
3. Uncomment the `deploy` section in `docker-compose.yml`.
