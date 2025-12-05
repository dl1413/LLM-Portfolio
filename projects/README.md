# Projects

This folder contains three data science projects in this portfolio, each with their own deployment files and final materials. The third project integrates methodologies from the first two.

## Project Structure

```
projects/
├── breast-cancer-classification/    # Medical ML Classification Project
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── .gitignore
│   ├── .dockerignore
│   ├── DEPLOYMENT.md
│   ├── Breast_Cancer_Classification_Report.md
│   └── Breast_Cancer_Classification_Publication.pdf
│
├── llm-bias-detection/              # LLM Ensemble Bias Detection Project
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── .gitignore
│   ├── .dockerignore
│   ├── DEPLOYMENT.md
│   ├── LLM_Ensemble_Bias_Detection_Report.md
│   └── LLM_Bias_Detection_Publication.pdf
│
└── llm-medical-diagnosis/           # Integrated Project (Combines Projects 1 & 2)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .gitignore
    ├── .dockerignore
    ├── DEPLOYMENT.md
    └── LLM_Augmented_Medical_Diagnosis_Report.md
```

## Projects Overview

### 1. Breast Cancer Classification

**Goal:** Binary classification of breast cancer tumors using ensemble machine learning methods.

**Key Results:**
- 99.12% accuracy with AdaBoost classifier
- 100% precision, 98.59% recall
- 0.9987 ROC-AUC

**Technologies:** scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, MLflow

### 2. LLM Bias Detection

**Goal:** Detect and quantify political bias in educational textbooks using LLM ensemble and Bayesian methods.

**Key Results:**
- Krippendorff's α = 0.84 (excellent inter-rater reliability)
- 67,500 bias ratings across 4,500 passages
- 3/5 publishers showed statistically credible bias

**Technologies:** GPT-4, Claude-3, Llama-3, PyMC, Bayesian hierarchical modeling, MLflow

### 3. LLM-Augmented Medical Diagnosis ⭐ *NEW*

**Goal:** Integrate ensemble ML classification with LLM narrative analysis for enhanced cancer diagnosis. This project ties together the methodologies from Projects 1 and 2.

**Key Results:**
- 99.56% combined accuracy (vs. 99.12% ML-only)
- Krippendorff's α = 0.87 for LLM narrative assessment
- Bayesian fusion with full uncertainty quantification
- Zero false negatives with appropriate uncertainty flagging

**Integration:**
- From Project 1: AdaBoost classifier, preprocessing pipeline (VIF, SMOTE, RFE)
- From Project 2: LLM ensemble (GPT-4, Claude-3, Llama-3), Bayesian hierarchical modeling, inter-rater reliability

**Technologies:** scikit-learn, PyMC, ArviZ, GPT-4, Claude-3, Llama-3, Bayesian fusion, MLflow

## Getting Started

Each project folder contains a `DEPLOYMENT.md` file with detailed instructions for building and running the application using Docker.

### Quick Start (Any Project)

```bash
cd projects/<project-name>

# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t <project-name> .
docker run -p 8000:8000 <project-name>
```

## Final Materials

Each project includes:
- **Technical Report (`.md`)** - Comprehensive analysis with methodology, results, and discussion
- **Publication (`.pdf`)** - Publication-ready document for academic or professional sharing
