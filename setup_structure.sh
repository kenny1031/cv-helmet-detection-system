#!/bin/bash

# Exit if any command fails
set -e

echo "Creating project structure..."

# Root files
touch README.md requirements.txt pyproject.toml .env docker-compose.yml

# Configs
mkdir -p configs
touch configs/dataset.yaml configs/train_config.yaml configs/inference_config.yaml

# Data
mkdir -p data/raw data/processed data/annotations data/splits
touch data/prepare_dataset.py
touch data/validate_annotations.py
touch data/dataset_stats.py

# Training
mkdir -p training/experiments
touch training/train.py
touch training/evaluate.py
touch training/export_model.py
touch training/utils.py

# Registry
mkdir -p registry
touch registry/model_registry.py
touch registry/schema.sql
touch registry/db.sqlite

# Inference
mkdir -p inference
touch inference/predictor.py
touch inference/onnx_inference.py
touch inference/video_stream.py
touch inference/benchmark.py

# API
mkdir -p api
touch api/main.py
touch api/schemas.py
touch api/service.py

# Monitoring
mkdir -p monitoring/logs
touch monitoring/metrics.py
touch monitoring/drift_detector.py

# Notebooks
mkdir -p notebooks
touch notebooks/exploratory_analysis.ipynb

# Tests
mkdir -p tests
touch tests/test_registry.py
touch tests/test_inference.py
touch tests/test_data_pipeline.py

echo "Project structure created successfully!"