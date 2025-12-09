# Derek Lankeaux - Research Engineer Portfolio

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-dl1413-black)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live-green)](https://dl1413.github.io/LLM-Portfolio/)

## 👋 About Me

Research Engineer and Data Scientist specializing in **Large Language Models (LLMs)**, **Ensemble Machine Learning**, and **Bayesian Statistical Methods**. MS in Applied Statistics from Rochester Institute of Technology with demonstrated expertise in production-ready ML systems, responsible AI governance, and scalable inference pipelines.

**Currently Seeking:** Research Engineer roles (2026) focusing on LLM applications, AI safety, model evaluation, and production ML systems.

## 🎯 Core Competencies

### Machine Learning & AI
- **LLM Systems**: Multi-model ensemble architectures (GPT-4o, Claude-3.5-Sonnet, Llama-3.2), prompt engineering, reliability assessment
- **Ensemble Methods**: Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting, Stacking, Voting classifiers
- **Bayesian Methods**: Hierarchical modeling, MCMC sampling (PyMC), posterior inference, uncertainty quantification
- **Responsible AI**: Explainability (SHAP, LIME), fairness auditing, model governance, IEEE 2830-2025 compliance

### Technical Stack (2026)
```python
# Core ML/AI
PyTorch 2.0+ | TensorFlow 2.15+ | scikit-learn 1.5+ | XGBoost 2.1+ | LightGBM 4.5+

# LLM & NLP
OpenAI API | Anthropic Claude | Llama 3.2 | HuggingFace Transformers | LangChain 0.3+

# Bayesian & Statistics  
PyMC 5.15+ | ArviZ 0.18+ | NumPyro | Stan

# Data Engineering
Pandas 2.2+ | Polars 1.0+ | NumPy 2.0+ | Dask

# MLOps & Production
MLflow 2.15+ | FastAPI 0.110+ | Docker | Redis | Prometheus

# Visualization
Matplotlib | Seaborn | Plotly | Streamlit | Gradio
```

### Research & Engineering Skills
- Production ML pipeline development with comprehensive testing
- Statistical validation: cross-validation, hypothesis testing, power analysis
- Inter-rater reliability assessment (Krippendorff's α, Cohen's κ)
- MCMC diagnostics and convergence analysis (R-hat, ESS)
- API integration with robust error handling and rate limiting
- Technical documentation and reproducible research

## 🚀 Featured Projects

### 1. LLM Ensemble Bias Detection with Bayesian Hierarchical Modeling
**[📄 Full Report](./LLM_Ensemble_Bias_Detection_Report.md)** | **[📊 Publication](./LLM_Bias_Detection_Publication.pdf)**

Novel computational framework for detecting political bias in educational content using frontier LLMs combined with rigorous Bayesian statistical inference.

**Key Achievements:**
- ✨ **67,500 bias ratings** across 4,500 textbook passages from 150 textbooks
- 🎯 **Krippendorff's α = 0.84** (excellent inter-rater reliability among GPT-4o, Claude-3.5-Sonnet, Llama-3.2)
- 📊 **Statistically significant publisher differences** (Friedman χ² = 42.73, p < 0.001)
- 🔬 **Bayesian uncertainty quantification** with 95% HDI for all parameters
- ⚡ **Production-ready pipeline** processing 2.5M tokens with circuit breakers and exponential backoff

**Technical Highlights:**
- Multi-LLM ensemble architecture with diversity across training paradigms (OpenAI, Anthropic, Meta)
- Prompt engineering for reliable structured outputs (JSON) with temperature control
- PyMC hierarchical model with partial pooling across publishers and textbooks
- MCMC sampling with excellent convergence (R-hat < 1.01, ESS > 3,000)
- MLflow experiment tracking with full provenance and artifact versioning
- IEEE 2830-2025 and EU AI Act compliance framework

**Technologies:** `GPT-4o` `Claude-3.5-Sonnet` `Llama-3.2-90B` `PyMC` `ArviZ` `MLflow` `Krippendorff's Alpha` `Bayesian Hierarchical Modeling` `FastAPI` `LangChain`

**Impact:** Established first-of-its-kind methodology for scalable educational content auditing with rigorous statistical validation.

---

### 2. Breast Cancer Classification with Enhanced Ensemble Methods
**[📄 Full Report](./Breast_Cancer_Classification_Report.md)** | **[📊 Publication](./Breast_Cancer_Classification_Publication.pdf)**

Production-grade ML pipeline for binary classification of breast cancer tumors using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset with comprehensive preprocessing and 8-algorithm benchmarking.

**Key Achievements:**
- 🏆 **99.12% accuracy** with AdaBoost (best-in-class performance)
- 💯 **100% precision** — zero false positives (no unnecessary biopsies)
- 🎯 **98.59% recall** — minimal missed malignancies
- 📈 **ROC-AUC: 0.9987** — near-perfect discrimination
- ✅ **10-fold CV: 98.46% ± 1.12%** — robust generalization confirmed
- 🏥 **Exceeds human inter-observer agreement** (90-95% in cytopathology)

**Technical Highlights:**
- Comprehensive preprocessing: VIF analysis, SMOTE class balancing, RFE feature selection
- Systematic evaluation of 8 ensemble algorithms (RF, GB, AdaBoost, XGBoost, LightGBM, Bagging, Voting, Stacking)
- SHAP-based explainability for clinical transparency
- Fairness auditing per IEEE 2830-2025 requirements
- MLflow model registry with versioned artifacts and signatures
- FastAPI production inference with <100ms p95 latency

**Technologies:** `scikit-learn` `XGBoost` `LightGBM` `AdaBoost` `SMOTE` `RFE` `SHAP` `MLflow` `FastAPI` `Responsible AI` `Model Governance`

**Impact:** Demonstrated clinical viability for computer-aided diagnosis with performance exceeding human pathologist concordance.

---

## 📊 Performance Metrics Summary

| Project | Key Metric | Value | Significance |
|---------|-----------|-------|--------------|
| **LLM Bias Detection** | Krippendorff's α | 0.84 | Excellent reliability (≥0.80 threshold) |
| | Publishers w/ Credible Bias | 3/5 (60%) | Statistically significant findings |
| | MCMC Convergence (R-hat) | < 1.01 | Perfect convergence across all parameters |
| | Total API Calls Processed | 67,500 | Production-scale deployment |
| **Breast Cancer Classification** | Accuracy | 99.12% | Exceeds human performance |
| | Precision | 100.00% | Zero false positives |
| | ROC-AUC | 0.9987 | Near-perfect discrimination |
| | Cross-Validation Stability | 98.46% ± 1.12% | Robust generalization |

## 🔬 Research Approach & Methodology

### Statistical Rigor
- Comprehensive validation: k-fold cross-validation, bootstrap resampling, permutation testing
- Bayesian uncertainty quantification with credible intervals
- Multiple hypothesis testing correction (Bonferroni, FDR)
- Power analysis and effect size reporting (Cohen's d, η²)

### Reproducibility Standards
- ✅ Fixed random seeds for all stochastic operations
- ✅ Version-controlled code with requirements.txt (pinned dependencies)
- ✅ MLflow experiment tracking with full lineage
- ✅ Comprehensive documentation (model cards, technical reports)
- ✅ IEEE 2830-2025 transparency compliance
- ✅ Carbon footprint estimation and logging

### Production Engineering
- Robust API handling: circuit breakers, exponential backoff, rate limiting
- Comprehensive error handling with structured logging (structlog)
- Monitoring dashboards with drift detection
- A/B testing frameworks for model evaluation
- Container-based deployment (Docker, Kubernetes)

## 📚 Publications & Reports

1. **LLM Ensemble Textbook Bias Detection: Technical Analysis Report**  
   *Version 3.0.0 | January 2026*  
   [Full Report](./LLM_Ensemble_Bias_Detection_Report.md) | [PDF](./LLM_Bias_Detection_Publication.pdf)

2. **Breast Cancer Classification: Technical Analysis Report**  
   *Version 3.0.0 | January 2026*  
   [Full Report](./Breast_Cancer_Classification_Report.md) | [PDF](./Breast_Cancer_Classification_Publication.pdf)

## 🎓 Education

**Master of Science in Applied Statistics**  
Rochester Institute of Technology | Expected 2026  
*Focus: Bayesian Methods, Machine Learning, Experimental Design*

## 💼 What I'm Looking For

**Target Roles (2026):**
- Research Engineer (LLMs, AI Safety, Model Evaluation)
- Machine Learning Engineer (Production ML Systems)
- Applied Research Scientist (Bayesian Methods, Ensemble Learning)
- AI/ML Research Engineer (Responsible AI, Model Governance)

**Key Interests:**
- Frontier LLM evaluation and reliability assessment
- Multi-model ensemble architectures and calibration
- Bayesian approaches to uncertainty quantification in AI
- Responsible AI: explainability, fairness, governance
- Production ML systems with MLOps best practices
- AI safety and alignment research

## 📫 Contact

- **GitHub:** [@dl1413](https://github.com/dl1413)
- **LinkedIn:** [Derek Lankeaux](https://linkedin.com/in/derek-lankeaux)
- **Portfolio:** [https://dl1413.github.io/LLM-Portfolio/](https://dl1413.github.io/LLM-Portfolio/)
- **Email:** Available upon request

---

## 🛠️ Repository Structure

```
LLM-Portfolio/
├── README.md                                      # This file
├── index.html                                     # Portfolio website
├── styles.css                                     # Styling
├── Breast_Cancer_Classification_Report.md         # Detailed technical report
├── Breast_Cancer_Classification_Publication.pdf   # Publication-ready PDF
├── LLM_Ensemble_Bias_Detection_Report.md          # Detailed technical report  
├── LLM_Bias_Detection_Publication.pdf             # Publication-ready PDF
└── reports/                                       # Additional reports
```

## 🏷️ Keywords

Large Language Models | GPT-4 | Claude | Llama | Ensemble Learning | XGBoost | LightGBM | AdaBoost | Bayesian Statistics | Hierarchical Modeling | PyMC | MCMC | Krippendorff's Alpha | SHAP | Explainable AI | Responsible AI | MLOps | MLflow | FastAPI | scikit-learn | Production ML | Model Governance | AI Safety | Research Engineering | Applied Statistics | Statistical Inference | Uncertainty Quantification | Cross-Validation | Hypothesis Testing

---

*Last Updated: January 2026*  
*Portfolio compliant with IEEE 2830-2025 (Transparent ML) and ISO/IEC 23894:2025 (AI Risk Management)*
