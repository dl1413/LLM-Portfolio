# Derek Lankeaux, MS
## Research Engineer | ML Systems Architect | AI Safety Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail)](mailto:contact@example.com)

---

### 🎯 Research Engineer | Actively Seeking 2026 Opportunities

**Specialization:** Large Language Models • Ensemble Learning • Bayesian Statistics • Production ML Systems

> **Impact-Driven Research Engineer** with expertise in multi-model LLM ensembles, Bayesian hierarchical modeling, and production ML pipelines. Proven track record of delivering **99.12% accuracy** models and processing **67.5K+ LLM API calls** at scale. Published researcher with deep expertise in statistical validation, responsible AI governance (IEEE 2830-2025), and MLOps best practices.

### 🏆 Key Achievements

- 🔬 **Developed novel LLM ensemble framework** achieving Krippendorff's α = 0.84 (excellent reliability) across GPT-4o, Claude-3.5, and Llama-3.2
- 🏥 **Built production ML system** for breast cancer classification with **99.12% accuracy**, exceeding human expert performance
- 📊 **Processed 2.5M+ tokens** through production-grade API pipeline with circuit breakers and adaptive rate limiting
- 📈 **Published 2 technical reports** demonstrating expertise in Bayesian inference, ensemble methods, and statistical rigor
- ⚡ **Deployed FastAPI models** with <100ms p95 latency and comprehensive monitoring dashboards

---

## 🚀 Featured Research Projects

<table>
<tr>
<td width="50%" valign="top">

### 🔬 LLM Ensemble Bias Detection
**[📄 Technical Report](./LLM_Ensemble_Bias_Detection_Report.md)** | **[📊 Publication](./LLM_Bias_Detection_Publication.pdf)**

**Novel multi-LLM framework for bias detection using Bayesian hierarchical modeling**

#### Impact Metrics
- 📊 **67,500 bias ratings** processed across 4,500 passages
- 🎯 **Krippendorff's α = 0.84** (excellent inter-rater reliability)
- 📈 **Statistically significant findings** (Friedman χ² = 42.73, p < 0.001)
- ⚡ **Production-scale deployment** handling 2.5M tokens

#### Technical Innovation
- **Multi-LLM Ensemble**: GPT-4o, Claude-3.5-Sonnet, Llama-3.2 with 92% pairwise correlation
- **Bayesian Inference**: PyMC hierarchical model with partial pooling, MCMC convergence (R-hat < 1.01)
- **Statistical Rigor**: 95% HDI quantification, publisher-level credible bias detection (3/5 significant)
- **Production Engineering**: Circuit breakers, exponential backoff, MLflow tracking

#### Tech Stack
`GPT-4o` `Claude-3.5` `Llama-3.2` `PyMC` `ArviZ` `MLflow` `FastAPI` `LangChain`

</td>
<td width="50%" valign="top">

### 🏥 Breast Cancer ML Classification
**[📄 Technical Report](./Breast_Cancer_Classification_Report.md)** | **[📊 Publication](./Breast_Cancer_Classification_Publication.pdf)**

**Clinical-grade ensemble system exceeding human expert performance**

#### Impact Metrics
- 🏆 **99.12% accuracy** (best-in-class AdaBoost)
- 💯 **100% precision** (zero false positives)
- 🎯 **98.59% recall** (minimal missed cases)
- 📈 **ROC-AUC: 0.9987** (near-perfect discrimination)

#### Technical Innovation
- **8-Algorithm Benchmark**: Comprehensive evaluation (RF, XGBoost, LightGBM, AdaBoost, Stacking, Voting)
- **Advanced Preprocessing**: VIF multicollinearity analysis, SMOTE balancing, RFE feature selection
- **Explainable AI**: SHAP values for clinical transparency, fairness auditing (IEEE 2830-2025)
- **Production Ready**: MLflow registry, FastAPI deployment (<100ms p95 latency)

#### Tech Stack
`scikit-learn` `XGBoost` `LightGBM` `SMOTE` `SHAP` `MLflow` `FastAPI`

</td>
</tr>
</table>

---

## 💼 Professional Experience & Capabilities

### 🎯 Core Expertise

<table>
<tr>
<td width="33%" valign="top">

#### 🤖 LLM & NLP
- Multi-model ensemble architectures
- Prompt engineering & optimization
- Inter-rater reliability analysis
- API integration at scale
- Structured output generation

**Tools:** GPT-4o, Claude-3.5, Llama-3.2, HuggingFace, LangChain

</td>
<td width="33%" valign="top">

#### 📊 Statistical ML
- Ensemble methods (8+ algorithms)
- Bayesian hierarchical modeling
- MCMC diagnostics (R-hat, ESS)
- Hypothesis testing & validation
- Feature engineering & selection

**Tools:** PyMC, ArviZ, scikit-learn, XGBoost, LightGBM

</td>
<td width="33%" valign="top">

#### ⚙️ Production MLOps
- FastAPI model deployment
- MLflow experiment tracking
- Circuit breakers & rate limiting
- Monitoring & drift detection
- Docker/Kubernetes orchestration

**Tools:** MLflow, FastAPI, Docker, Redis, Prometheus

</td>
</tr>
</table>

### 🛠️ Technical Stack

```yaml
Languages:        Python 3.12+ • R • SQL • Bash
ML Frameworks:    PyTorch 2.0+ • TensorFlow 2.15+ • scikit-learn 1.5+ • JAX
LLM APIs:         OpenAI (GPT-4o) • Anthropic (Claude-3.5) • Meta (Llama-3.2) • HuggingFace
Ensemble ML:      XGBoost 2.1+ • LightGBM 4.5+ • CatBoost • AdaBoost
Bayesian Stats:   PyMC 5.15+ • ArviZ 0.18+ • NumPyro • Stan • JAGS
Data Stack:       Pandas 2.2+ • Polars 1.0+ • NumPy 2.0+ • Dask • Apache Arrow
MLOps:            MLflow 2.15+ • Weights & Biases • DVC • Kubeflow
Deployment:       FastAPI 0.110+ • Docker • Kubernetes • AWS • GCP
Monitoring:       Prometheus • Grafana • ELK Stack • Datadog
Explainability:   SHAP • LIME • Captum • InterpretML
Version Control:  Git • GitHub Actions • GitLab CI/CD
```

### 🔬 Research Methodology

**Statistical Rigor**
- ✅ Cross-validation (k-fold, stratified, leave-one-out)
- ✅ Bayesian inference with credible intervals (95% HDI)
- ✅ Multiple testing correction (Bonferroni, FDR, Holm-Sidak)
- ✅ Effect size reporting (Cohen's d, η², Cramer's V)
- ✅ Power analysis and sample size determination

**Reproducibility Standards**
- ✅ IEEE 2830-2025 (Transparent ML) compliance
- ✅ ISO/IEC 23894:2025 (AI Risk Management) alignment
- ✅ Fixed random seeds and version pinning
- ✅ Comprehensive model cards and documentation
- ✅ Carbon footprint tracking and reporting

**Production Engineering**
- ✅ Robust error handling and circuit breakers
- ✅ Adaptive rate limiting and backoff strategies
- ✅ Comprehensive logging (structlog) and monitoring
- ✅ A/B testing frameworks and gradual rollouts
- ✅ Model performance tracking and drift detection

---

## 📊 Quantitative Performance Summary

<table>
<tr>
<td width="50%" valign="top">

### LLM Ensemble Bias Detection
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Inter-Rater Reliability** | α = 0.84 | Excellent (≥0.80) |
| **Model Convergence** | R-hat < 1.01 | Perfect |
| **Statistical Power** | χ² = 42.73 | p < 0.001 |
| **Scale Deployment** | 67.5K calls | Production |
| **Credible Findings** | 3/5 publishers | 60% detection |

</td>
<td width="50%" valign="top">

### Breast Cancer Classification
| Metric | Value | Benchmark |
|--------|-------|-----------|
| **Accuracy** | 99.12% | Exceeds human (90-95%) |
| **Precision** | 100.00% | Zero false positives |
| **Recall** | 98.59% | Minimal misses |
| **ROC-AUC** | 0.9987 | Near-perfect |
| **CV Stability** | 98.46% ± 1.12% | Robust |

</td>
</tr>
</table>

---

## 🎓 Education & Certifications

**Master of Science in Applied Statistics**  
Rochester Institute of Technology | Expected 2026  
*Specialization: Bayesian Methods, Machine Learning, Experimental Design*

**Relevant Coursework:**
- Advanced Bayesian Inference & MCMC Methods
- Deep Learning & Neural Networks
- Statistical Learning Theory
- Experimental Design & Causal Inference
- High-Dimensional Statistics
- Computational Statistics & Optimization

---

## 💼 Target Opportunities (2026)

### 🎯 Ideal Roles

<table>
<tr>
<td width="50%">

**Research Engineer**
- LLM evaluation & benchmarking
- Multi-model ensemble systems
- AI safety & alignment
- Model reliability assessment

</td>
<td width="50%">

**ML Systems Engineer**
- Production ML pipelines
- MLOps infrastructure
- Model deployment & monitoring
- Scalable inference systems

</td>
</tr>
<tr>
<td width="50%">

**Applied Research Scientist**
- Bayesian statistical methods
- Ensemble learning research
- Causal inference
- Experimental design

</td>
<td width="50%">

**AI Safety Researcher**
- Responsible AI governance
- Model explainability (XAI)
- Fairness & bias detection
- Compliance frameworks

</td>
</tr>
</table>

### 🌟 What I Bring

✅ **Technical Depth**: Deep expertise in LLMs, Bayesian methods, and ensemble ML  
✅ **Production Experience**: Deployed FastAPI models with <100ms latency at scale  
✅ **Research Rigor**: Published work with strong statistical validation (p < 0.001)  
✅ **AI Governance**: IEEE 2830-2025 and ISO/IEC 23894:2025 compliance experience  
✅ **Reproducibility**: MLflow tracking, version control, comprehensive documentation  
✅ **Impact Focus**: Track record of exceeding benchmarks (99.12% vs 90-95% human)

---

## 📚 Publications & Technical Reports

| Title | Type | Date | Links |
|-------|------|------|-------|
| **LLM Ensemble Textbook Bias Detection** | Technical Report v3.0.0 | Jan 2026 | [Report](./LLM_Ensemble_Bias_Detection_Report.md) • [PDF](./LLM_Bias_Detection_Publication.pdf) |
| **Breast Cancer Classification** | Technical Report v3.0.0 | Jan 2026 | [Report](./Breast_Cancer_Classification_Report.md) • [PDF](./Breast_Cancer_Classification_Publication.pdf) |

---

## 📫 Let's Connect

<div align="center">

### 🤝 Open to Research Engineer Opportunities | Available for Interviews

**Preferred Contact:** [LinkedIn](https://linkedin.com/in/derek-lankeaux) | [Email](mailto:contact@example.com)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Derek_Lankeaux-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/derek-lankeaux)
[![GitHub](https://img.shields.io/badge/GitHub-@dl1413-181717?style=for-the-badge&logo=github)](https://github.com/dl1413)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Site-00C7B7?style=for-the-badge)](https://dl1413.github.io/LLM-Portfolio/)

**Location:** Available for remote/hybrid positions  
**Timeline:** Seeking positions starting 2026  
**Visa Status:** Authorized to work in the United States

</div>

---

<div align="center">

## 🛠️ Repository Structure

```
LLM-Portfolio/
├── 📄 README.md                                      # This portfolio
├── 🌐 index.html                                     # Interactive portfolio site
├── 🎨 styles.css                                     # Portfolio styling
├── 📊 Breast_Cancer_Classification_Report.md         # ML technical report
├── 📑 Breast_Cancer_Classification_Publication.pdf   # Publication PDF
├── 🔬 LLM_Ensemble_Bias_Detection_Report.md          # LLM research report
├── 📑 LLM_Bias_Detection_Publication.pdf             # Publication PDF
└── 📁 reports/                                       # Additional documentation
```

---

### 🔍 Keywords for Search & ATS

</div>

**Machine Learning:** Deep Learning • Neural Networks • Ensemble Methods • Random Forest • XGBoost • LightGBM • AdaBoost • Gradient Boosting • Stacking • Bagging

**Large Language Models:** GPT-4 • GPT-4o • Claude-3.5-Sonnet • Llama-3.2 • BERT • Transformers • Prompt Engineering • Few-Shot Learning • Zero-Shot Learning • In-Context Learning

**Bayesian Statistics:** Hierarchical Modeling • MCMC • PyMC • Stan • Posterior Inference • Prior Specification • Credible Intervals • Bayesian Inference • Probabilistic Programming

**Statistical Methods:** Hypothesis Testing • Cross-Validation • Bootstrap • Permutation Testing • Effect Sizes • Power Analysis • Multiple Testing Correction • Inter-Rater Reliability • Krippendorff's Alpha • Cohen's Kappa

**Explainable AI (XAI):** SHAP • LIME • Feature Importance • Model Interpretability • Fairness Auditing • Bias Detection • Responsible AI • AI Ethics • AI Governance

**MLOps & Production:** MLflow • Weights & Biases • Model Registry • Experiment Tracking • FastAPI • Docker • Kubernetes • CI/CD • Model Monitoring • Drift Detection • A/B Testing

**Programming:** Python • R • SQL • PyTorch • TensorFlow • scikit-learn • Pandas • NumPy • Dask • Apache Spark

**Research Engineering:** Technical Writing • Statistical Validation • Reproducible Research • Peer Review • Literature Review • Experimental Design • Causal Inference

**AI Safety:** Model Evaluation • Benchmark Development • Reliability Assessment • Safety Testing • Alignment • Constitutional AI • Red Teaming

**Standards & Compliance:** IEEE 2830-2025 • ISO/IEC 23894:2025 • EU AI Act • GDPR • Model Cards • Transparency • Accountability

---

<div align="center">

**📌 Last Updated:** December 2025  
**✅ Compliance:** IEEE 2830-2025 (Transparent ML) • ISO/IEC 23894:2025 (AI Risk Management)  
**🔒 License:** Portfolio content © 2025 Derek Lankeaux. Code samples available under MIT License.

---

*⭐ If you find this work interesting, please star this repository!*

</div>
