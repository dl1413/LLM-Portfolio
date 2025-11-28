# LLM-Portfolio

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)
![PyMC](https://img.shields.io/badge/PyMC-5.0+-orange.svg)
![License](https://img.shields.io/badge/License-All_Rights_Reserved-yellow.svg)

**Production ML Systems & Large Language Model Portfolio**

A portfolio showcasing production-grade Machine Learning systems and Large Language Model (LLM) projects, demonstrating scalable architecture, fault-tolerant design, and rigorous statistical methods.

---

## 🎯 Featured Projects

### 1. Production Cancer Classification Pipeline

**99.12% accuracy** with 10x optimization through ensemble methods and parallel processing.

| Metric | Value |
|--------|-------|
| Accuracy | 99.12% |
| Precision | 100% |
| Recall | 98.59% |
| ROC-AUC | 0.9987 |
| Inference Latency | < 1ms |

**Key Features:**
- 8 ensemble algorithms evaluated at scale (AdaBoost, XGBoost, LightGBM, etc.)
- Comprehensive preprocessing pipeline (VIF, SMOTE, RFE)
- Production deployment with FastAPI and Docker
- 10x training speedup through vectorization and parallel processing

📄 [Technical Report](./Breast_Cancer_Classification_Report.md) | 📊 [Performance Benchmarks](./PERFORMANCE.md)

---

### 2. Distributed LLM Ensemble Platform

Orchestrating **67,500 API calls** across frontier models with Bayesian hierarchical inference.

| Metric | Value |
|--------|-------|
| Total API Calls | 67,500 |
| Krippendorff's α | 0.84 (excellent) |
| Throughput | ~150 passages/min |
| Success Rate | 99.5% |

**Key Features:**
- Multi-model ensemble (GPT-4, Claude-3, Llama-3)
- Fault-tolerant API orchestration with circuit breakers
- Bayesian hierarchical modeling with PyMC
- Production-grade error handling and rate limiting

📄 [Technical Report](./LLM_Ensemble_Bias_Detection_Report.md) | 🏗️ [Architecture](./ARCHITECTURE.md)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production ML Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │   Data   │───▶│  Model   │───▶│   API    │───▶│ Monitor│ │
│  │ Pipeline │    │ Training │    │ (FastAPI)│    │ (Logs) │ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       │              │                │               │      │
│       ▼              ▼                ▼               ▼      │
│   [Parallel]    [Ensemble]      [Docker]        [Metrics]   │
│   [Processing]  [Methods]       [Container]     [Alerts]    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system design documentation.

---

## ⚡ Performance Highlights

| Optimization | Before | After | Speedup |
|--------------|--------|-------|---------|
| Data Preprocessing | 0.8s | 0.08s | **10x** |
| Cross-Validation | 45s | 11s | **4.1x** |
| Model Inference | 15ms | 1.2ms | **12.5x** |
| LLM API Processing | Sequential | Parallel | **4x** |

See [PERFORMANCE.md](./PERFORMANCE.md) for detailed benchmarks.

---

## 🚀 Production Deployment

### Quick Start with Docker

```bash
# Build and run the API
cd deployment
docker build -t breast-cancer-api:v1.0 -f Dockerfile ..
docker run -d -p 8000:8000 --name bc-api breast-cancer-api:v1.0

# Test the endpoint
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, ...]}'
```

### Docker Compose

```bash
cd deployment
docker-compose up -d
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Make diagnosis prediction |
| `/health` | GET | Service health check |
| `/metrics` | GET | Model performance metrics |
| `/features` | GET | Expected feature names |
| `/docs` | GET | Interactive API documentation |

---

## 🛠️ Technologies

### Production ML Systems
- **Python 3.10+** - Core language
- **FastAPI** - High-performance API framework
- **Docker** - Containerization
- **scikit-learn** - ML algorithms
- **XGBoost / LightGBM** - Gradient boosting

### LLM Technologies
- **OpenAI GPT-4** - Frontier language model
- **Anthropic Claude-3** - Constitutional AI
- **Meta Llama-3** - Open-weights LLM
- **LangChain** - LLM orchestration

### Statistical Methods
- **PyMC** - Bayesian modeling
- **Bayesian Hierarchical Models** - Uncertainty quantification
- **MCMC Sampling** - Posterior inference
- **Krippendorff's Alpha** - Inter-rater reliability

### Infrastructure
- **Prometheus** - Metrics collection
- **Structured Logging** - Observability
- **Health Checks** - Container orchestration
- **Circuit Breakers** - Fault tolerance

---

## 📊 Project Documentation

| Document | Description |
|----------|-------------|
| [Breast Cancer Report](./Breast_Cancer_Classification_Report.md) | Full technical analysis |
| [LLM Bias Detection Report](./LLM_Ensemble_Bias_Detection_Report.md) | Bayesian ensemble study |
| [Architecture Guide](./ARCHITECTURE.md) | System design documentation |
| [Performance Benchmarks](./PERFORMANCE.md) | Optimization results |

---

## 🌐 Portfolio Website

Open `index.html` in a web browser to view the interactive portfolio, or deploy to GitHub Pages for online access.

---

## 📁 Repository Structure

```
LLM-Portfolio/
├── index.html                           # Portfolio website
├── styles.css                           # Website styling
├── README.md                            # This file
├── ARCHITECTURE.md                      # System architecture
├── PERFORMANCE.md                       # Performance benchmarks
├── Breast_Cancer_Classification_Report.md    # Technical report
├── LLM_Ensemble_Bias_Detection_Report.md     # Technical report
├── deployment/                          # Production deployment
│   ├── api.py                          # FastAPI application
│   ├── Dockerfile                      # Container definition
│   ├── docker-compose.yml              # Service orchestration
│   └── requirements.txt                # Python dependencies
└── reports/                            # Additional reports
```

---

## 📬 Contact

**Derek Lankeaux**  
MS Applied Statistics, Rochester Institute of Technology

- GitHub: [github.com/dl1413](https://github.com/dl1413)
- Portfolio: [LLM-Portfolio](https://github.com/dl1413/LLM-Portfolio)

---

## License

All rights reserved.
