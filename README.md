# Latent Behavior Analysis - Flies

## Setup

### 1. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate lat-beh
```

### 2. Install AnimalPoseForecasting

This repo depends on plotting utilities from the AnimalPoseForecasting repository:
```bash

git clone https://github.com/kristinbranson/AnimalPoseForecasting.git
cd AnimalPoseForecasting
python -m pip install -e .
python -c "import apf; import flyllm; print('Success!')"
```

**Note:** We don't include AnimalPoseForecasting in this repo because it's an external dependency.

## Project Structure
```
In Progress...
```