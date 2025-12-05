# LLM-Augmented Medical Diagnosis - Deployment

This folder contains deployment files for the LLM-Augmented Medical Diagnosis project, which integrates ensemble machine learning with LLM narrative analysis.

## Project Overview

This project combines methodologies from two previous portfolio works:
- **Breast Cancer Classification**: Ensemble ML methods achieving 99.12% accuracy
- **LLM Bias Detection**: LLM ensemble analysis with Bayesian hierarchical modeling

The integrated framework achieves **99.56% accuracy** by combining:
- AdaBoost classification of structured cytological features
- LLM ensemble (GPT-4, Claude-3, Llama-3) analysis of pathology narratives
- Bayesian fusion for uncertainty-aware predictions

## Files

- **Dockerfile** - Container image definition for the application
- **docker-compose.yml** - Docker Compose configuration for local development and deployment
- **requirements.txt** - Python dependencies (combines both project requirements)
- **.gitignore** - Git ignore patterns for the project
- **.dockerignore** - Docker ignore patterns to optimize build context
- **LLM_Augmented_Medical_Diagnosis_Report.md** - Full technical analysis report

## Quick Start

### Using Docker Compose

```bash
# Set environment variables for LLM API access
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key

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
docker build -t llm-medical-diagnosis .

# Run the container
docker run -p 8002:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  llm-medical-diagnosis
```

## Configuration

Environment variables can be configured in a `.env` file:

```env
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model configuration
ML_MODEL_PATH=models/adaboost_model.pkl
FUSION_MODEL_PATH=models/bayesian_fusion.pkl

# MLflow tracking
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Volumes

The following directories are mounted as volumes for persistent data:

- `./data` - Input data and datasets
- `./models` - Trained ML models and Bayesian posterior samples

## API Endpoints

Once deployed, the API provides:

- `POST /predict` - Submit features AND narrative for combined prediction
- `POST /predict/ml-only` - Submit features for ML-only classification
- `POST /predict/llm-only` - Submit narrative for LLM-only analysis
- `GET /health` - Health check endpoint
- `GET /model/info` - Model metadata and version

### Example Request

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "radius_mean": 17.99,
      "texture_mean": 10.38,
      "perimeter_mean": 122.8,
      "area_mean": 1001.0,
      ...
    },
    "narrative": "The aspirate shows discohesive malignant epithelial cells with marked nuclear pleomorphism..."
  }'
```

### Example Response

```json
{
  "prediction": "Malignant",
  "probability": 0.903,
  "credible_interval": [0.867, 0.941],
  "confidence": "HIGH",
  "components": {
    "ml_probability": 0.892,
    "llm_consensus_probability": 0.917,
    "llm_ratings": {
      "gpt4": 5,
      "claude3": 5,
      "llama3": 4
    },
    "llm_agreement": "HIGH"
  },
  "recommendation": "Histological confirmation and staging workup recommended.",
  "disclaimer": "Decision-support tool. Final diagnosis by qualified medical professional."
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
│                    (FastAPI + Uvicorn)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│   ML Classification  │         │   LLM Ensemble      │
│   (AdaBoost)        │         │   (GPT/Claude/Llama)│
└──────────┬──────────┘         └──────────┬──────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
              ┌─────────────────────┐
              │   Bayesian Fusion   │
              │   (PyMC + ArviZ)    │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  Uncertainty-Aware  │
              │     Prediction      │
              └─────────────────────┘
```

## Technologies

### From Breast Cancer Classification
- Python 3.11
- scikit-learn, XGBoost, LightGBM
- SMOTE for class balancing
- RFE for feature selection
- SHAP for explainability

### From LLM Bias Detection
- PyMC for Bayesian modeling
- ArviZ for posterior analysis
- Krippendorff's alpha for reliability
- OpenAI, Anthropic API clients

### Shared Infrastructure
- FastAPI + Uvicorn
- MLflow for experiment tracking
- Docker for containerization

## Related Projects

- `/projects/breast-cancer-classification/` - ML classification methodology source
- `/projects/llm-bias-detection/` - LLM ensemble and Bayesian methodology source
