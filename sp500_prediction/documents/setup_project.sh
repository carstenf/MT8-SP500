#!/bin/bash

# run it with
# bash setup_project.sh
# or
# sh setup_project.sh

# Create main project directory
mkdir -p sp500_prediction

# Create subdirectories
mkdir -p sp500_prediction/data/raw
mkdir -p sp500_prediction/data/processed
mkdir -p sp500_prediction/notebooks
mkdir -p sp500_prediction/src/data
mkdir -p sp500_prediction/src/features
mkdir -p sp500_prediction/src/models
mkdir -p sp500_prediction/src/visualization
mkdir -p sp500_prediction/src/explanation
mkdir -p sp500_prediction/tests
mkdir -p sp500_prediction/configs
mkdir -p sp500_prediction/results/models
mkdir -p sp500_prediction/results/plots
mkdir -p sp500_prediction/results/metrics

# Create initial files
touch sp500_prediction/src/__init__.py
touch sp500_prediction/src/data/__init__.py
touch sp500_prediction/src/features/__init__.py
touch sp500_prediction/src/models/__init__.py
touch sp500_prediction/src/visualization/__init__.py
touch sp500_prediction/src/explanation/__init__.py
touch sp500_prediction/README.md
touch sp500_prediction/requirements.txt
touch sp500_prediction/setup.py