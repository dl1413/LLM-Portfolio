# LLM-Portfolio

A portfolio showcasing Data Science and Large Language Model (LLM) projects.

## Overview

This portfolio features three main data science projects that demonstrate machine learning, natural language processing, and AI applications:

### Featured Projects

1. **Breast Cancer Classification** - Ensemble ML methods for binary classification achieving 99.12% accuracy with AdaBoost on the WDBC dataset.

2. **LLM Bias Detection** - LLM ensemble (GPT-4, Claude-3, Llama-3) with Bayesian hierarchical modeling for detecting textbook bias. Achieved Krippendorff's α = 0.84.

3. **LLM-Augmented Medical Diagnosis** ⭐ *NEW* - Integrated framework combining ensemble ML classification with LLM narrative analysis. Achieves 99.56% accuracy with Bayesian fusion and uncertainty quantification. *This project ties together the first two projects.*

## Technologies

- Python
- Scikit-learn, XGBoost, LightGBM
- PyMC & ArviZ (Bayesian modeling)
- GPT-4, Claude-3, Llama-3 (LLM ensemble)
- FastAPI & Docker
- MLflow

## Project Structure

```
LLM-Portfolio/
├── index.html                    # Portfolio website
├── styles.css                    # Styling
├── projects/
│   ├── breast-cancer-classification/
│   ├── llm-bias-detection/
│   └── llm-medical-diagnosis/    # NEW: Integrated project
└── reports/
    ├── Breast_Cancer_Classification_Report.md
    └── LLM_Bias_Detection_Report.md
```

## Getting Started

Open `index.html` in a web browser to view the portfolio, or deploy to GitHub Pages for online access.

Each project in `/projects/` contains:
- `DEPLOYMENT.md` - Deployment instructions
- `Dockerfile` & `docker-compose.yml` - Container configuration
- `requirements.txt` - Python dependencies
- Technical report (`.md`) - Full analysis

## License

All rights reserved.
