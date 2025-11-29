# ⚡ Analysis Results & Performance

## Overview

This document details the analysis results and key findings across both portfolio projects, demonstrating rigorous statistical methods and data-driven insights.

---

## Healthcare Analytics: Cancer Classification

### Key Analysis Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 99.12% | Exceptional classification performance |
| Precision | 100% | No false positives |
| Recall | 98.59% | Minimal missed cases |
| ROC-AUC | 0.9987 | Near-perfect discrimination |
| Cross-Validation | 98.46% ± 1.12% | Robust generalization |

### Data Processing Summary

| Component | Details | Outcome |
|-----------|---------|---------|
| Data Preprocessing | VIF analysis, SMOTE balancing | Clean, balanced dataset |
| Feature Selection | RFE with 15 features selected | 50% dimensionality reduction |
| Model Comparison | 8 ensemble algorithms evaluated | AdaBoost selected as best |
| Validation | 10-fold stratified cross-validation | Confirmed model stability |

### Statistical Analysis Methods

```python
# Feature importance analysis
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Identify most predictive features
selector = RFE(RandomForestClassifier(n_estimators=100), 
               n_features_to_select=15)
selector.fit(X_train, y_train)
important_features = X.columns[selector.support_]

# Cross-validation for robust evaluation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print(f"Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Confidence interval calculation
from scipy import stats
ci_95 = stats.t.interval(0.95, len(cv_scores)-1, 
                         loc=cv_scores.mean(), 
                         scale=stats.sem(cv_scores))
```

### Key Findings

- **Top Predictive Features:** worst concave points, worst perimeter, mean concave points
- **Feature Reduction:** 30 → 15 features without accuracy loss
- **Model Stability:** Low variance across cross-validation folds (±1.12%)

---

## Large-Scale Content Analysis Study

### Research Study Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Observations | 67,500 | Large-scale analysis |
| Inter-Rater Reliability (α) | 0.84 | Excellent agreement |
| Friedman Test | χ² = 42.73 | Significant differences (p < 0.001) |
| Bayesian R-hat | < 1.01 | Proper convergence |
| Effect Size Range | -0.48 to +0.38 | Moderate effects detected |

### Data Collection Summary

| Component | Specification |
|-----------|--------------|
| Sources Analyzed | 150 |
| Passages per Source | 30 |
| Total Text Passages | 4,500 |
| Ratings per Passage | 3 (multi-source validation) |
| Total Data Points | 67,500 |

### Statistical Analysis Results

| Source | Posterior Mean | 95% HDI | Classification |
|--------|----------------|---------|----------------|
| Source C | -0.48 | [-0.62, -0.34] | Credible effect |
| Source A | -0.29 | [-0.41, -0.17] | Credible effect |
| Source E | +0.02 | [-0.10, +0.14] | Neutral |
| Source B | +0.08 | [-0.04, +0.20] | Neutral |
| Source D | +0.38 | [+0.26, +0.50] | Credible effect |

### Statistical Methods Applied

```python
# Inter-rater reliability analysis
import krippendorff
alpha = krippendorff.alpha(ratings_matrix, level_of_measurement='interval')
print(f"Krippendorff's Alpha: {alpha:.2f}")

# Bayesian hierarchical modeling
import pymc as pm
import arviz as az

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_global = pm.Normal('mu_global', mu=0, sigma=1)
    sigma_group = pm.HalfNormal('sigma_group', sigma=0.5)
    
    # Group effects
    group_effect = pm.Normal('group_effect', mu=0, sigma=sigma_group, shape=n_groups)
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu_global + group_effect[group_idx], 
                      sigma=sigma_obs, observed=y)
    
    # Sample with diagnostics
    trace = pm.sample(2000, chains=4, return_inferencedata=True)
    
# Convergence diagnostics
print(az.summary(trace, var_names=['group_effect']))
```

### Key Findings

- **Reliability Validated:** Multi-source agreement (α = 0.84) confirms data quality
- **Significant Differences:** Friedman test confirms variation across sources (p < 0.001)
- **Uncertainty Quantified:** 95% HDI provides probabilistic bounds on effects
- **3/5 Sources:** Show statistically credible effects

---

## Analysis Quality Metrics

### Data Quality Assessment

| Quality Dimension | Project 1 | Project 2 |
|-------------------|-----------|-----------|
| Completeness | 100% (no missing) | 99.5% |
| Accuracy | Validated against source | Multi-source triangulation |
| Consistency | Cross-validation confirmed | Reliability α = 0.84 |
| Timeliness | Current dataset | Recent data collection |

### Statistical Rigor

| Aspect | Implementation |
|--------|----------------|
| Cross-Validation | 10-fold stratified |
| Confidence Intervals | 95% CI for all estimates |
| Multiple Comparisons | Bonferroni correction applied |
| Effect Sizes | Cohen's d / posterior effects reported |
| Uncertainty | Bayesian HDI for probabilistic inference |

---

## Methodology & Reproducibility

All analyses were conducted using:

- **Python:** 3.10+
- **Statistical Libraries:** pandas, NumPy, scipy, scikit-learn
- **Bayesian Analysis:** PyMC 5.0+, ArviZ
- **Visualization:** matplotlib, seaborn
- **Environment:** Conda for dependency management

### Reproducibility Steps

```bash
# Clone repository and set up environment
git clone https://github.com/dl1413/LLM-Portfolio.git
cd LLM-Portfolio

# Create environment
conda env create -f environment.yml
conda activate data-analysis

# Run analysis notebooks
jupyter notebook
```

---

*Analysis results documented as of November 2025*
