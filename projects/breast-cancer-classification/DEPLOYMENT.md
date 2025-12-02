# Breast Cancer Classification - Deployment

This folder contains deployment files for the Breast Cancer Classification project.

## Project Overview

This project implements an ensemble machine learning pipeline for binary classification of breast cancer tumors using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The best-performing model (AdaBoost) achieved **99.12% accuracy**, **100% precision**, **98.59% recall**, and **0.9987 ROC-AUC**.

## Files

- **Dockerfile** - Container image definition for the application
- **docker-compose.yml** - Docker Compose configuration for local development and deployment
- **requirements.txt** - Python dependencies
- **.gitignore** - Git ignore patterns for the project
- **.dockerignore** - Docker ignore patterns to optimize build context
- **Breast_Cancer_Classification_Report.md** - Full technical analysis report
- **Breast_Cancer_Classification_Publication.pdf** - Publication-ready document

## Quick Start

### Using Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# Stop the application
docker-compose down
```

### Using Docker

```bash
# Build the image
docker build -t breast-cancer-classification .

# Run the container
docker run -p 8000:8000 breast-cancer-classification
```

## Configuration

Environment variables can be configured in a `.env` file:

```env
# Example .env file
MODEL_PATH=models/adaboost_model.pkl
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Volumes

The following directories are mounted as volumes for persistent data:

- `./data` - Input data and datasets
- `./models` - Trained models and checkpoints

## API Endpoints

Once deployed, the API provides:

- `POST /predict` - Submit features for classification
- `GET /health` - Health check endpoint
- `GET /model/info` - Model metadata and version

## Technologies

- Python 3.11
- scikit-learn, XGBoost, LightGBM, CatBoost
- FastAPI + Uvicorn
- MLflow for experiment tracking
- SHAP for explainability
