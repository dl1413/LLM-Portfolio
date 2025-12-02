# LLM Bias Detection - Deployment

This folder contains deployment files for the LLM Bias Detection project.

## Project Overview

This project presents a novel computational framework for detecting and quantifying political bias in educational textbooks using an ensemble of three frontier Large Language Models (GPT-4, Claude-3-Opus, and Llama-3-70B) combined with Bayesian hierarchical modeling. The analysis achieved excellent inter-rater reliability (Krippendorff's α = 0.84) across **67,500 bias ratings**.

## Files

- **Dockerfile** - Container image definition for the application
- **docker-compose.yml** - Docker Compose configuration for local development and deployment
- **requirements.txt** - Python dependencies
- **.gitignore** - Git ignore patterns for the project
- **.dockerignore** - Docker ignore patterns to optimize build context
- **LLM_Ensemble_Bias_Detection_Report.md** - Full technical analysis report
- **LLM_Bias_Detection_Publication.pdf** - Publication-ready document

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
docker build -t llm-bias-detection .

# Run the container
docker run -p 8000:8000 llm-bias-detection
```

## Configuration

Environment variables can be configured in a `.env` file:

```env
# Example .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
TOGETHER_API_KEY=your-together-key
```

## Volumes

The following directories are mounted as volumes for persistent data:

- `./data` - Input data and datasets
- `./models` - Trained models and checkpoints

## API Endpoints

Once deployed, the API provides:

- `POST /analyze` - Submit textbook passage for bias analysis
- `GET /health` - Health check endpoint
- `GET /model/info` - Model metadata and ensemble configuration

## Technologies

- Python 3.11
- PyMC for Bayesian modeling
- OpenAI, Anthropic, Together APIs
- FastAPI + Uvicorn
- MLflow for experiment tracking
