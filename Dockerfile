# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0

# Set working directory
WORKDIR /app

# Install system dependencies
# git: required for hatch-vcs to determine version
# build-essential: for compiling some python packages
# libnetcdf-dev: for netcdf4
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration files
COPY pyproject.toml README.md LICENSE.txt ./

# Copy the aurora package
COPY aurora/ aurora/

# Install dependencies and the aurora package
# We use --no-cache-dir to keep image size down
# Increased timeout to prevent read timeouts on slow connections
RUN pip install --no-cache-dir --upgrade pip --default-timeout=1000 && \
    pip install --no-cache-dir ".[dev]" --default-timeout=1000

# Copy the rwanda module (local package)
COPY rwanda/ rwanda/

# Copy notebooks and scripts
COPY notebooks/ notebooks/
COPY evaluate_model.py quick_inference.py ./
COPY PRODUCTION_README.md ./

# Create directories for artifacts
RUN mkdir -p working evaluation_results

# Default command (can be overridden)
# Runs the quick inference script by default
CMD ["python", "quick_inference.py"]
