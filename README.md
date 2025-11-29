# Data Analyst Portfolio

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)
![Statistics](https://img.shields.io/badge/Statistics-Advanced-orange.svg)
![SQL](https://img.shields.io/badge/SQL-Proficient-blue.svg)
![License](https://img.shields.io/badge/License-All_Rights_Reserved-yellow.svg)

**Data Analysis & Statistical Insights Portfolio**

A portfolio showcasing data analysis projects demonstrating statistical rigor, research methodology, and the ability to transform complex data into actionable insights.

---

## 🎯 Featured Projects

### 1. Healthcare Analytics: Cancer Classification

**99.12% accuracy** diagnostic model using comprehensive statistical analysis and feature engineering.

| Metric | Value |
|--------|-------|
| Accuracy | 99.12% |
| Precision | 100% |
| Recall | 98.59% |
| ROC-AUC | 0.9987 |

**Key Analysis:**
- Analyzed 30 clinical features across 569 patient samples
- Statistical preprocessing with VIF analysis and class balancing (SMOTE)
- Feature selection using Recursive Feature Elimination (RFE)
- Comprehensive model evaluation with 10-fold cross-validation

📄 [Project Folder](./projects/breast-cancer-classification/) | 📊 [Technical Report](./projects/breast-cancer-classification/Breast_Cancer_Classification_Report.md)

---

### 2. Large-Scale Content Analysis Study

Analyzed **67,500 data points** with Bayesian hierarchical modeling and inter-rater reliability analysis.

| Metric | Value |
|--------|-------|
| Total Observations | 67,500 |
| Inter-Rater Reliability (α) | 0.84 (excellent) |
| Statistical Significance | p < 0.001 |
| Sources Analyzed | 150 |

**Key Analysis:**
- Designed systematic sampling strategy for 4,500 text passages
- Multi-source validation with Krippendorff's alpha reliability analysis
- Bayesian hierarchical modeling with full uncertainty quantification
- Hypothesis testing with Bonferroni-corrected post-hoc comparisons

📄 [Project Folder](./projects/llm-bias-detection/) | 📊 [Technical Report](./projects/llm-bias-detection/LLM_Ensemble_Bias_Detection_Report.md)

---

## 📊 Data Analysis Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Analysis Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │   Data   │───▶│  Clean & │───▶│ Analyze  │───▶│ Report │ │
│  │ Collect  │    │ Prepare  │    │ & Model  │    │ Insights│ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       │              │                │               │      │
│       ▼              ▼                ▼               ▼      │
│   [Sources]     [Validation]    [Statistics]    [Visualize] │
│   [Sampling]    [Transform]     [Testing]       [Explain]   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed methodology documentation.

---

## ⚡ Analysis Highlights

| Analysis | Dataset Size | Key Finding | Impact |
|----------|--------------|-------------|--------|
| Cancer Classification | 569 samples | 99.12% accuracy | Clinical-grade diagnostic model |
| Content Analysis | 67,500 ratings | α = 0.84 reliability | Validated multi-source methodology |
| Feature Engineering | 30 → 15 features | 50% reduction | Improved model interpretability |
| Uncertainty Quantification | Bayesian HDI | 95% credible intervals | Robust decision support |

See [PERFORMANCE.md](./PERFORMANCE.md) for detailed benchmarks.

---

## 🛠️ Technical Skills

### Data Analysis & Programming
- **Python** - pandas, NumPy, scipy
- **SQL** - Data querying and manipulation
- **Excel** - Advanced analysis and reporting
- **Data Wrangling** - Cleaning, transformation, validation

### Statistical Methods
- **Descriptive Statistics** - Summary metrics, distributions
- **Inferential Statistics** - Hypothesis testing, confidence intervals
- **Bayesian Analysis** - PyMC, uncertainty quantification
- **Regression & Classification** - scikit-learn, model evaluation

### Visualization & Reporting
- **matplotlib / seaborn** - Statistical visualizations
- **Data Storytelling** - Clear, actionable insights
- **Technical Documentation** - Reproducible analysis reports

### Research Methods
- **Experimental Design** - Sampling, controls, validation
- **Inter-Rater Reliability** - Krippendorff's alpha
- **A/B Testing** - Statistical significance testing

---

## 📊 Project Documentation

| Document | Description |
|----------|-------------|
| [Cancer Classification Report](./projects/breast-cancer-classification/Breast_Cancer_Classification_Report.md) | Healthcare analytics technical analysis |
| [Content Analysis Report](./projects/llm-bias-detection/LLM_Ensemble_Bias_Detection_Report.md) | Large-scale research study |
| [Methodology Guide](./ARCHITECTURE.md) | Analysis methodology documentation |
| [Performance Metrics](./PERFORMANCE.md) | Detailed results and benchmarks |

---

## 🌐 Portfolio Website

Open `index.html` in a web browser to view the interactive portfolio, or deploy to GitHub Pages for online access.

---

## 📁 Repository Structure

```
Data-Analyst-Portfolio/
├── index.html                              # Portfolio website
├── styles.css                              # Website styling
├── README.md                               # This file
├── ARCHITECTURE.md                         # Analysis methodology
├── PERFORMANCE.md                          # Performance metrics
├── projects/
│   ├── breast-cancer-classification/       # Healthcare analytics project
│   │   ├── README.md                       # Project overview
│   │   ├── Breast_Cancer_Classification_Report.md
│   │   ├── Breast_Cancer_Classification_Publication.pdf
│   │   └── deployment/                     # API implementation
│   └── llm-bias-detection/                 # Content analysis project
│       ├── README.md                       # Project overview
│       ├── LLM_Ensemble_Bias_Detection_Report.md
│       └── LLM_Bias_Detection_Publication.pdf
└── .gitignore
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
