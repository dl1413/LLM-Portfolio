# LLM Bias Detection - Deployment

This folder contains deployment files for the LLM Bias Detection project.

## Files

- **Dockerfile** - Container image definition for the application
- **docker-compose.yml** - Docker Compose configuration for local development and deployment
- **requirements.txt** - Python dependencies
- **.gitignore** - Git ignore patterns for the project
- **.dockerignore** - Docker ignore patterns to optimize build context

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
MODEL_NAME=bert-base-uncased
BATCH_SIZE=32
```

## Volumes

The following directories are mounted as volumes for persistent data:

- `./data` - Input data and datasets
- `./models` - Trained models and checkpoints
