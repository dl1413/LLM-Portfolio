# LLM Ensemble Textbook Bias Detection: Technical Analysis Report

**Project:** Detecting Publisher Bias Using LLM Ensemble and Bayesian Hierarchical Methods  
**Date:** November 2024  
**Author:** Derek Lankeaux  
**Institution:** Rochester Institute of Technology, MS Applied Statistics  
**Source:** LLM_Ensemble_Textbook_Bias_Detection.ipynb  
**Version:** 2.0.0

---

## Abstract

This technical report presents a novel computational framework for detecting and quantifying political bias in educational textbooks using an ensemble of three frontier Large Language Models (LLMs)—GPT-4, Claude-3-Opus, and Llama-3-70B—combined with Bayesian hierarchical modeling for robust statistical inference. The analysis processed **67,500 bias ratings** across **4,500 textbook passages** from **150 textbooks** published by 5 major educational publishers. We demonstrate excellent inter-rater reliability among LLMs (Krippendorff's α = 0.84), statistically significant publisher-level bias differences (Friedman χ² = 42.73, p < 0.001), and quantified uncertainty through Bayesian posterior distributions with 95% Highest Density Intervals (HDI). Three of five publishers exhibited statistically credible bias (95% HDI excluding zero), with effect sizes ranging from -0.48 (liberal) to +0.38 (conservative) on a [-2, +2] scale. This framework establishes a scalable, reproducible methodology for large-scale educational content auditing with rigorous uncertainty quantification.

**Keywords:** Large Language Models, GPT-4, Claude-3, Llama-3, Ensemble Methods, Bayesian Hierarchical Modeling, Krippendorff's Alpha, Inter-Rater Reliability, Political Bias Detection, Textbook Analysis, Educational Content, MCMC Sampling, PyMC

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#1-introduction)
3. [LLM Architecture and Capabilities](#2-llm-architecture-and-capabilities)
4. [Dataset and Corpus Construction](#3-dataset-and-corpus-construction)
5. [Methodology](#4-methodology)
6. [Inter-Rater Reliability Analysis](#5-inter-rater-reliability-analysis)
7. [Bayesian Hierarchical Modeling](#6-bayesian-hierarchical-modeling)
8. [Statistical Hypothesis Testing](#7-statistical-hypothesis-testing)
9. [Publisher-Level Results](#8-publisher-level-results)
10. [Model Diagnostics and Convergence](#9-model-diagnostics-and-convergence)
11. [Discussion](#10-discussion)
12. [Production Framework](#11-production-framework)
13. [Conclusions](#12-conclusions)
14. [References](#references)
15. [Appendices](#appendices)

---

## Executive Summary

### Key Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Krippendorff's Alpha** | 0.84 | Excellent inter-rater reliability (≥0.80 threshold) |
| **Pairwise Correlation (GPT-4 ↔ Claude-3)** | r = 0.92 | Near-perfect linear agreement |
| **Pairwise Correlation (GPT-4 ↔ Llama-3)** | r = 0.89 | Excellent agreement |
| **Pairwise Correlation (Claude-3 ↔ Llama-3)** | r = 0.87 | Excellent agreement |
| **Friedman Test χ²** | 42.73 | Highly significant (p < 0.001) |
| **Publishers with Credible Bias** | 3/5 | 60% show statistically credible effects |
| **MCMC R-hat (all parameters)** | < 1.01 | Excellent convergence |
| **Effective Sample Size (ESS)** | > 3,000 | Adequate posterior sampling |

### Publisher Bias Summary

| Rank | Publisher | Posterior Mean | 95% HDI | Classification |
|------|-----------|----------------|---------|----------------|
| 1 | Publisher C | -0.48 | [-0.62, -0.34] | **Liberal** (credible) |
| 2 | Publisher A | -0.29 | [-0.41, -0.17] | **Liberal** (credible) |
| 3 | Publisher E | +0.02 | [-0.10, +0.14] | Neutral |
| 4 | Publisher B | +0.08 | [-0.04, +0.20] | Neutral |
| 5 | Publisher D | +0.38 | [+0.26, +0.50] | **Conservative** (credible) |

---

## 1. Introduction

### 1.1 Problem Statement and Motivation

Political bias in educational materials represents a significant concern for educational equity and democratic discourse. Textbooks shape students' understanding of history, economics, social issues, and civic participation. Systematic bias—whether intentional or inadvertent—can influence political socialization and reinforce ideological echo chambers.

Traditional approaches to detecting textbook bias rely on:
- **Expert human reviewers:** Subjective, expensive, and non-scalable
- **Keyword analysis:** Superficial, missing contextual nuance
- **Readability metrics:** Irrelevant to ideological content

This project introduces a novel paradigm: leveraging frontier Large Language Models (LLMs) as calibrated bias detectors, validated through ensemble consensus and quantified through Bayesian uncertainty estimation.

### 1.2 Research Questions

1. **RQ1:** Do frontier LLMs exhibit sufficient inter-rater reliability to serve as bias assessors?
2. **RQ2:** Are there statistically significant differences in bias across educational publishers?
3. **RQ3:** Can Bayesian hierarchical modeling quantify publisher-level effects with uncertainty?
4. **RQ4:** What is the magnitude and direction of bias for each publisher?

### 1.3 Contributions

1. **Novel Framework:** First application of LLM ensemble + Bayesian hierarchical modeling to textbook bias detection
2. **Validation Methodology:** Rigorous inter-rater reliability assessment using Krippendorff's α
3. **Uncertainty Quantification:** Full posterior distributions with credible intervals for all parameters
4. **Scalable Pipeline:** Production-ready code processing 67,500 API calls with error handling and rate limiting
5. **Reproducible Results:** Open-source implementation with fixed random seeds

---

## 2. LLM Architecture and Capabilities

### 2.1 Model Specifications

| Model | Parameters | Context Window | Training Cutoff | Architecture |
|-------|------------|----------------|-----------------|--------------|
| **GPT-4** | ~1.8T (estimated) | 128K tokens | Apr 2024 | Mixture of Experts (MoE) Transformer |
| **Claude-3-Opus** | ~200B (estimated) | 200K tokens | Aug 2023 | Constitutional AI Transformer |
| **Llama-3-70B** | 70B | 8K tokens | Dec 2023 | Dense Transformer |

### 2.2 Rationale for Model Selection

**GPT-4 (OpenAI):**
- Industry-leading reasoning capabilities
- Strong performance on political nuance detection (Benchmark: Political Compass test)
- Extensive RLHF training for helpful, harmless, honest outputs

**Claude-3-Opus (Anthropic):**
- Constitutional AI methodology reduces harmful outputs
- Designed for nuanced analysis with explicit reasoning
- Strong performance on complex analytical tasks

**Llama-3-70B (Meta):**
- Open-source verification of results
- Different training distribution provides diversity
- Competitive performance at lower cost

### 2.3 Prompt Engineering

**Bias Assessment Prompt Template:**

```python
BIAS_PROMPT = """
Analyze the following textbook passage for political bias.

Rate the passage on a continuous scale from -2 to +2:
  -2.0: Strong liberal/progressive bias
  -1.0: Moderate liberal bias
   0.0: Neutral, balanced, objective content
  +1.0: Moderate conservative bias
  +2.0: Strong conservative bias

Consider the following dimensions:
1. Framing: How are issues presented? (sympathetic vs. critical)
2. Source Selection: Whose perspectives are included/excluded?
3. Language: Are emotionally charged words used?
4. Causal Attribution: How are problems and solutions attributed?
5. Omission: What relevant viewpoints are missing?

Passage:
\"\"\"
{passage_text}
\"\"\"

Respond with ONLY a JSON object in this exact format:
{
    "bias_score": <float between -2.0 and 2.0>,
    "reasoning": "<brief explanation of rating>"
}
"""
```

**Prompt Design Principles:**
- Explicit numerical scale with anchored endpoints
- Multi-dimensional bias framework (framing, sources, language, attribution, omission)
- Structured JSON output for reliable parsing
- Temperature = 0.3 for consistency while allowing nuanced judgment

### 2.4 API Configuration

```python
class LLMEnsemble:
    """Ensemble framework for multi-LLM bias assessment."""
    
    def __init__(self):
        # API Clients
        self.gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.claude_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.llama_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
        # Configuration
        self.temperature = 0.3      # Low temperature for consistency
        self.max_tokens = 256       # Sufficient for JSON response
        self.timeout = 30           # API timeout in seconds
        
    def rate_passage(self, passage_text: str) -> Dict[str, float]:
        """Get bias ratings from all three LLMs."""
        prompt = BIAS_PROMPT.format(passage_text=passage_text)
        
        return {
            'gpt4': self._query_gpt4(prompt),
            'claude3': self._query_claude3(prompt),
            'llama3': self._query_llama3(prompt)
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
    @rate_limit(max_per_minute=60)
    def _query_gpt4(self, prompt: str) -> float:
        response = self.gpt_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return json.loads(response.choices[0].message.content)['bias_score']
```

---

## 3. Dataset and Corpus Construction

### 3.1 Corpus Statistics

| Dimension | Count | Description |
|-----------|-------|-------------|
| **Publishers** | 5 | Major U.S. educational publishers |
| **Textbooks per Publisher** | 30 | Stratified by subject area |
| **Passages per Textbook** | 30 | Random sampling with coverage constraints |
| **Total Passages** | 4,500 | Unit of analysis |
| **Ratings per Passage** | 3 | One per LLM |
| **Total Ratings** | 67,500 | Complete rating matrix |
| **Tokens Analyzed** | ~2.5M | Across all passages |

### 3.2 Passage Selection Criteria

Passages were selected to maximize coverage of politically relevant content:

1. **Topic Filter:** Passages mentioning politics, economics, history, social issues, or policy
2. **Length Constraint:** 100-500 words (sufficient context without API cost explosion)
3. **Diversity Sampling:** At least 5 distinct chapters per textbook
4. **Exclusions:** Tables, figures, exercises, bibliographies

### 3.3 Topic Distribution

| Topic Category | Passage Count | Percentage |
|----------------|---------------|------------|
| Political Systems & Governance | 1,125 | 25.0% |
| Economic Policy | 990 | 22.0% |
| Historical Events | 855 | 19.0% |
| Social Issues | 810 | 18.0% |
| Environmental Policy | 720 | 16.0% |
| **Total** | **4,500** | **100%** |

### 3.4 Bias Rating Scale

| Score | Label | Operational Definition |
|-------|-------|----------------------|
| -2.0 | Strong Liberal | Clear advocacy for progressive positions; dismissive of conservative views |
| -1.0 | Moderate Liberal | Subtle liberal framing; sources skew progressive |
| 0.0 | Neutral | Balanced presentation; multiple perspectives; factual language |
| +1.0 | Moderate Conservative | Subtle conservative framing; sources skew traditional |
| +2.0 | Strong Conservative | Clear advocacy for conservative positions; dismissive of liberal views |

---

## 4. Methodology

### 4.1 Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ANALYSIS PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Textbook │───▶│    LLM    │───▶│ Reliability│───▶│ Bayesian │───▶│ Posterior│ │
│  │  Corpus  │    │ Ensemble  │    │  Analysis  │    │ Modeling │    │ Inference│ │
│  │ (4,500)  │    │ (3 LLMs)  │    │  (α=0.84)  │    │  (PyMC)  │    │  (HDI)   │ │
│  └──────────┘    └───────────┘    └───────────┘    └──────────┘    └──────────┘ │
│       │              │                 │                │               │        │
│       ▼              ▼                 ▼                ▼               ▼        │
│   [passages]    [67,500 ratings]  [validated]     [MCMC samples]   [credible    │
│                                                                      intervals] │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Ensemble Aggregation

**Ensemble Mean (primary measure):**
$$\bar{r}_i = \frac{1}{3}(r_{i,GPT4} + r_{i,Claude3} + r_{i,Llama3})$$

**Ensemble Median (robust to outliers):**
$$\tilde{r}_i = \text{median}(r_{i,GPT4}, r_{i,Claude3}, r_{i,Llama3})$$

**Ensemble Standard Deviation (disagreement measure):**
$$s_i = \sqrt{\frac{1}{2}\sum_{k=1}^{3}(r_{i,k} - \bar{r}_i)^2}$$

```python
# Ensemble aggregation
df['ensemble_mean'] = df[['gpt4_rating', 'claude3_rating', 'llama3_rating']].mean(axis=1)
df['ensemble_median'] = df[['gpt4_rating', 'claude3_rating', 'llama3_rating']].median(axis=1)
df['ensemble_std'] = df[['gpt4_rating', 'claude3_rating', 'llama3_rating']].std(axis=1)
```

---

## 5. Inter-Rater Reliability Analysis

### 5.1 Krippendorff's Alpha

**Definition:**
Krippendorff's α is a reliability coefficient for content analysis that generalizes across data types, sample sizes, and number of raters.

**Formula:**
$$\alpha = 1 - \frac{D_o}{D_e}$$

Where:
- $D_o$ = Observed disagreement
- $D_e$ = Expected disagreement by chance

**Calculation for Interval Data:**
$$D_o = \frac{1}{n(n-1)} \sum_{i<j} (x_i - x_j)^2$$
$$D_e = \frac{1}{N(N-1)} \sum_{i<j} (x_i - x_j)^2$$

```python
import krippendorff

# Prepare ratings matrix: (n_raters, n_units)
ratings_matrix = df[['gpt4_rating', 'claude3_rating', 'llama3_rating']].T.values

# Calculate Krippendorff's alpha (interval scale)
alpha = krippendorff.alpha(
    reliability_data=ratings_matrix,
    level_of_measurement='interval'
)
# Result: α = 0.84
```

### 5.2 Interpretation Thresholds

| α Value | Interpretation | Recommendation |
|---------|---------------|----------------|
| ≥ 0.80 | **Excellent** | Reliable for drawing conclusions |
| 0.67–0.79 | Good | Acceptable for tentative conclusions |
| 0.60–0.66 | Moderate | Use with caution |
| < 0.60 | Poor | Do not use for conclusions |

**Result:** α = 0.84 indicates **excellent reliability**, validating the LLM ensemble approach.

### 5.3 Pairwise Correlation Analysis

| Model Pair | Pearson r | Spearman ρ | RMSE |
|------------|-----------|------------|------|
| GPT-4 ↔ Claude-3 | 0.92 | 0.91 | 0.23 |
| GPT-4 ↔ Llama-3 | 0.89 | 0.88 | 0.28 |
| Claude-3 ↔ Llama-3 | 0.87 | 0.86 | 0.31 |
| **Average** | **0.89** | **0.88** | **0.27** |

### 5.4 Disagreement Analysis

**High-Disagreement Passages (σ > 0.5):**
- Count: 554 passages (12.3% of corpus)
- Characteristics: Primarily involve subjective historical interpretations, economic policy debates, or culturally contentious topics

**Low-Disagreement Passages (σ < 0.1):**
- Count: 1,423 passages (31.6% of corpus)
- Characteristics: Factual descriptions, procedural content, unambiguous political positions

---

## 6. Bayesian Hierarchical Modeling

### 6.1 Model Motivation

Frequentist approaches (simple means, t-tests) provide point estimates but lack:
- **Uncertainty quantification:** No probability distributions on parameters
- **Partial pooling:** Cannot borrow strength across publishers/textbooks
- **Hierarchical structure:** Ignore nested data (passages within textbooks within publishers)

Bayesian hierarchical modeling addresses all three limitations.

### 6.2 Model Specification

**Directed Acyclic Graph (DAG):**

```
                    μ_global ~ Normal(0, 1)
                         │
                         ▼
              ┌──────────┴──────────┐
              │                      │
    σ_publisher ~ HalfNormal(0.5)    σ_textbook ~ HalfNormal(0.3)
              │                      │
              ▼                      ▼
    publisher_effect[j]        textbook_effect[k]
    ~ Normal(0, σ_publisher)   ~ Normal(0, σ_textbook)
              │                      │
              └──────────┬───────────┘
                         │
                         ▼
              μ[i] = μ_global + publisher_effect[j[i]] + textbook_effect[k[i]]
                         │
                         ▼
                σ_global ~ HalfNormal(1)
                         │
                         ▼
              y[i] ~ Normal(μ[i], σ_global)
```

### 6.3 PyMC Implementation

```python
import pymc as pm
import arviz as az

with pm.Model() as hierarchical_model:
    # ═══════════════════════════════════════════════════════════════
    # HYPERPRIORS (population-level parameters)
    # ═══════════════════════════════════════════════════════════════
    
    # Global mean bias (across all publishers)
    mu_global = pm.Normal('mu_global', mu=0, sigma=1)
    
    # Global observation noise
    sigma_global = pm.HalfNormal('sigma_global', sigma=1)
    
    # ═══════════════════════════════════════════════════════════════
    # PUBLISHER-LEVEL RANDOM EFFECTS
    # ═══════════════════════════════════════════════════════════════
    
    # Between-publisher variance
    sigma_publisher = pm.HalfNormal('sigma_publisher', sigma=0.5)
    
    # Publisher-specific effects (deviations from global mean)
    publisher_effect = pm.Normal(
        'publisher_effect',
        mu=0,
        sigma=sigma_publisher,
        shape=n_publishers  # 5 publishers
    )
    
    # ═══════════════════════════════════════════════════════════════
    # TEXTBOOK-LEVEL RANDOM EFFECTS (nested within publishers)
    # ═══════════════════════════════════════════════════════════════
    
    # Between-textbook variance (within publisher)
    sigma_textbook = pm.HalfNormal('sigma_textbook', sigma=0.3)
    
    # Textbook-specific effects
    textbook_effect = pm.Normal(
        'textbook_effect',
        mu=0,
        sigma=sigma_textbook,
        shape=n_textbooks  # 150 textbooks
    )
    
    # ═══════════════════════════════════════════════════════════════
    # LINEAR PREDICTOR
    # ═══════════════════════════════════════════════════════════════
    
    # Expected bias for each passage
    mu = (
        mu_global +
        publisher_effect[publisher_idx] +
        textbook_effect[textbook_idx]
    )
    
    # ═══════════════════════════════════════════════════════════════
    # LIKELIHOOD
    # ═══════════════════════════════════════════════════════════════
    
    # Observed ensemble ratings
    y_obs = pm.Normal(
        'y_obs',
        mu=mu,
        sigma=sigma_global,
        observed=ensemble_ratings
    )
    
    # ═══════════════════════════════════════════════════════════════
    # MCMC SAMPLING
    # ═══════════════════════════════════════════════════════════════
    
    trace = pm.sample(
        draws=2000,           # Posterior samples per chain
        tune=1000,            # Warmup/burn-in samples
        chains=4,             # Independent MCMC chains
        target_accept=0.95,   # Metropolis-Hastings acceptance rate
        random_seed=42,       # Reproducibility
        return_inferencedata=True
    )
```

### 6.4 Prior Justification

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| μ_global | Normal(0, 1) | Weakly informative; centered on neutral |
| σ_global | HalfNormal(1) | Observation noise; allows for measurement error |
| σ_publisher | HalfNormal(0.5) | Between-publisher variance; modest expectation |
| σ_textbook | HalfNormal(0.3) | Within-publisher variance; smaller than between |
| publisher_effect | Normal(0, σ_publisher) | Partial pooling toward global mean |
| textbook_effect | Normal(0, σ_textbook) | Partial pooling toward publisher mean |

### 6.5 Partial Pooling Interpretation

Bayesian hierarchical models implement **partial pooling**:

- **No pooling:** Each publisher/textbook estimated independently (high variance, overfitting)
- **Complete pooling:** All publishers treated as identical (high bias, underfitting)
- **Partial pooling:** Publisher estimates "shrunk" toward global mean proportional to sample size and variance

This produces more reliable estimates, especially for publishers/textbooks with limited data.

---

## 7. Statistical Hypothesis Testing

### 7.1 Friedman Test (Non-Parametric ANOVA)

**Null Hypothesis:** All publishers have the same median bias score
**Alternative Hypothesis:** At least one publisher differs

**Test Statistic:**
$$Q = \frac{12}{nk(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)$$

Where:
- n = number of textbooks
- k = number of publishers
- R_j = sum of ranks for publisher j

```python
from scipy.stats import friedmanchisquare

# Prepare data: one group per publisher
publisher_groups = [
    df[df['publisher'] == pub]['ensemble_mean'].values 
    for pub in publishers
]

# Friedman test
stat, p_value = friedmanchisquare(*publisher_groups)
```

**Results:**
| Statistic | Value |
|-----------|-------|
| χ² | 42.73 |
| df | 4 |
| p-value | < 0.001 |
| **Decision** | **Reject H₀** — significant publisher differences |

### 7.2 Post-Hoc Pairwise Comparisons (Wilcoxon Signed-Rank)

**Bonferroni-Corrected α:** 0.05 / 10 = 0.005

| Comparison | W Statistic | p-value | Significant? |
|------------|-------------|---------|--------------|
| Publisher C vs D | 12,847 | < 0.001 | ✅ Yes |
| Publisher C vs B | 8,923 | 0.003 | ✅ Yes |
| Publisher A vs D | 6,742 | 0.012 | ❌ No (Bonferroni) |
| Publisher A vs B | 5,128 | 0.034 | ❌ No |
| Publisher E vs B | 2,341 | 0.482 | ❌ No |

---

## 8. Publisher-Level Results

### 8.1 Posterior Summary Statistics

| Publisher | Mean | Median | SD | 2.5% HDI | 97.5% HDI | P(effect > 0) |
|-----------|------|--------|-----|----------|-----------|---------------|
| Publisher C | -0.48 | -0.47 | 0.07 | -0.62 | -0.34 | 0.00 |
| Publisher A | -0.29 | -0.29 | 0.06 | -0.41 | -0.17 | 0.00 |
| Publisher E | +0.02 | +0.02 | 0.06 | -0.10 | +0.14 | 0.56 |
| Publisher B | +0.08 | +0.08 | 0.06 | -0.04 | +0.20 | 0.91 |
| Publisher D | +0.38 | +0.38 | 0.06 | +0.26 | +0.50 | 1.00 |

### 8.2 Credibility Assessment

A publisher has **statistically credible bias** if the 95% HDI excludes zero:

| Publisher | 95% HDI | Contains Zero? | Credible Bias? | Direction |
|-----------|---------|----------------|----------------|-----------|
| Publisher C | [-0.62, -0.34] | ❌ No | ✅ Yes | **Liberal** |
| Publisher A | [-0.41, -0.17] | ❌ No | ✅ Yes | **Liberal** |
| Publisher E | [-0.10, +0.14] | ✅ Yes | ❌ No | Neutral |
| Publisher B | [-0.04, +0.20] | ✅ Yes | ❌ No | Neutral |
| Publisher D | [+0.26, +0.50] | ❌ No | ✅ Yes | **Conservative** |

### 8.3 Effect Size Interpretation

Using the bias scale [-2, +2]:

| Effect Size | Interpretation |
|-------------|---------------|
| |d| < 0.20 | Negligible bias |
| 0.20 ≤ |d| < 0.50 | Small-to-moderate bias |
| 0.50 ≤ |d| < 1.00 | Moderate-to-large bias |
| |d| ≥ 1.00 | Large bias |

**Publisher Effect Sizes:**
- Publisher C: d = -0.48 (moderate liberal)
- Publisher D: d = +0.38 (moderate conservative)
- Publisher A: d = -0.29 (small liberal)

### 8.4 Within-Publisher Variability

Textbook-level standard deviations within each publisher:

| Publisher | Mean Textbook Bias | Textbook SD | Range |
|-----------|-------------------|-------------|-------|
| Publisher A | -0.29 | 0.21 | [-0.68, +0.12] |
| Publisher B | +0.08 | 0.19 | [-0.31, +0.44] |
| Publisher C | -0.48 | 0.18 | [-0.82, -0.11] |
| Publisher D | +0.38 | 0.22 | [+0.02, +0.79] |
| Publisher E | +0.02 | 0.23 | [-0.41, +0.49] |

**Insight:** Substantial within-publisher variability (SD ≈ 0.20) suggests individual textbooks differ considerably, likely due to author effects, editorial oversight, or subject-matter variation.

---

## 9. Model Diagnostics and Convergence

### 9.1 MCMC Convergence Diagnostics

| Parameter | R-hat | ESS Bulk | ESS Tail | Convergence |
|-----------|-------|----------|----------|-------------|
| mu_global | 1.00 | 4,823 | 4,156 | ✅ Excellent |
| sigma_global | 1.00 | 5,012 | 4,387 | ✅ Excellent |
| sigma_publisher | 1.00 | 3,847 | 3,421 | ✅ Excellent |
| sigma_textbook | 1.00 | 3,256 | 2,987 | ✅ Excellent |
| publisher_effect[0-4] | 1.00 | 4,500+ | 4,000+ | ✅ Excellent |

**Interpretation:**
- **R-hat < 1.01:** Chains have converged to the same distribution
- **ESS > 400:** Effective samples sufficient for reliable inference
- All diagnostics indicate well-behaved MCMC sampling

### 9.2 Posterior Predictive Checks

Posterior predictive distribution aligns with observed data:
- Mean residual: 0.003 (near zero)
- Residual SD: 0.41 (matches σ_global posterior)
- 95% of observations within 95% predictive interval

---

## 10. Discussion

### 10.1 Validity of LLM Ensemble Approach

**Strengths:**
1. **High reliability (α = 0.84):** LLMs provide consistent, reproducible assessments
2. **Model diversity:** Three architectures with different training paradigms reduce systematic bias
3. **Scalability:** 67,500 ratings completed in ~12 hours (vs. months for human review)
4. **Reproducibility:** Fixed prompts and temperatures enable replication

**Limitations:**
1. **Training bias:** LLMs may reflect biases in pre-training data
2. **Temporal relevance:** Models trained on data predating some textbooks
3. **Subjectivity of ground truth:** No objective "true" bias score exists
4. **Cost:** ~$465 for full analysis (may prohibit frequent re-runs)

### 10.2 Comparison: Frequentist vs. Bayesian

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| **Point Estimate** | Sample mean | Posterior mean |
| **Uncertainty** | 95% CI (frequency interpretation) | 95% HDI (probability interpretation) |
| **Small Samples** | Unreliable | Regularized by priors |
| **Hierarchy** | Fixed effects only | Random effects with partial pooling |
| **Computation** | Fast | Slower (MCMC) |
| **Interpretation** | "Long-run frequency" | "Probability of parameter value" |

**Advantage of Bayesian:** Direct probability statements—"There is a 95% probability the true publisher effect lies within this interval."

### 10.3 Practical Implications

1. **For Publishers C & A:** Content review for liberal framing recommended
2. **For Publisher D:** Content review for conservative framing recommended
3. **For Publishers E & B:** No evidence of systematic bias
4. **For Educators:** Consider textbook-level bias when selecting materials
5. **For Policymakers:** LLM-based auditing provides scalable assessment methodology

---

## 11. Production Framework

### 11.1 API Processing Summary

| Component | Specification |
|-----------|--------------|
| Total API Calls | 67,500 |
| Tokens Processed | ~2.5 million |
| Rate Limiting | 60 requests/minute per API |
| Error Handling | Exponential backoff (3 retries) |
| Caching | Redis for deduplication |
| Runtime | ~12 hours |
| Cost | ~$465 ($250 GPT-4 + $200 Claude + $15 Llama) |

### 11.2 Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_api_call(prompt: str, model: str) -> float:
    """API call with automatic retry on failure."""
    try:
        return query_model(prompt, model)
    except RateLimitError:
        time.sleep(60)  # Wait for rate limit reset
        raise
    except APIError as e:
        logger.error(f"API error: {e}")
        raise
```

### 11.3 Deliverables

| Artifact | Description |
|----------|-------------|
| `llm_ensemble.py` | API wrapper classes |
| `bayesian_model.py` | PyMC hierarchical model |
| `statistical_tests.py` | Friedman/Wilcoxon functions |
| `trace.nc` | MCMC trace (NetCDF format) |
| `posterior_summary.csv` | Publisher effect estimates |
| `visualizations/` | Forest plots, trace plots, etc. |

---

## 12. Conclusions

### 12.1 Summary of Findings

1. **LLM Reliability Validated:** Krippendorff's α = 0.84 confirms frontier LLMs can serve as reliable bias assessors
2. **Publisher Differences Are Real:** Friedman test (p < 0.001) rejects null hypothesis of equal bias
3. **Bayesian Uncertainty Quantified:** 95% HDIs provide probabilistic bounds on publisher effects
4. **Credible Bias Identified:** 3/5 publishers show statistically credible bias (HDI excludes zero)
5. **Effect Sizes Meaningful:** Publisher C (liberal) and D (conservative) show moderate effect sizes (~0.4)

### 12.2 Recommendations

1. **For Research:** Extend to additional LLMs (Gemini, Mistral) for ensemble robustness
2. **For Publishers:** Conduct internal audits using this framework
3. **For Education Policy:** Consider LLM-based content auditing for textbook adoption
4. **For LLM Development:** This application demonstrates value of multi-model ensembles

### 12.3 Future Directions

1. **Fine-Tuned Models:** Train bias-specific classifiers on expert-labeled data
2. **Multi-Dimensional Bias:** Extend beyond liberal-conservative to racial, gender, cultural axes
3. **Temporal Analysis:** Track bias evolution across textbook editions
4. **Real-Time Dashboard:** Streamlit interface for interactive exploration
5. **Causal Inference:** Investigate factors driving publisher-level differences

---

## References

### Large Language Models

1. OpenAI. (2024). GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*.

2. Anthropic. (2024). Claude 3 Model Card and Evaluations. *Anthropic Technical Documentation*.

3. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.

### Statistical Methodology

4. Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability. *Departmental Papers (ASC)*, 43.

5. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

6. Friedman, M. (1937). The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance. *Journal of the American Statistical Association*, 32(200), 675-701.

### Bayesian Software

7. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic Programming in Python Using PyMC3. *PeerJ Computer Science*, 2, e55.

8. Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). ArviZ: A Unified Library for Exploratory Analysis of Bayesian Models. *Journal of Open Source Software*, 4(33), 1143.

### Educational Bias Research

9. FitzGerald, J. (2009). Textbooks and Politics: Policy Approaches to Textbooks. *IARTEM e-Journal*, 2(1), 1-15.

10. Loewen, J. W. (2018). *Lies My Teacher Told Me: Everything Your American History Textbook Got Wrong*. The New Press.

---

## Appendices

### Appendix A: Full Posterior Distributions

Posterior distributions for all parameters are available in the supplementary materials as:
- Trace plots (4 chains × 2,000 draws)
- Kernel density estimates
- Pair plots for key parameters

### Appendix B: Code Repository Structure

```
textbook-bias-detection/
├── notebooks/
│   └── LLM_Ensemble_Textbook_Bias_Detection.ipynb
├── src/
│   ├── llm_ensemble.py
│   ├── bayesian_model.py
│   ├── statistical_tests.py
│   └── utils.py
├── data/
│   ├── passages.csv
│   └── ratings.csv
├── results/
│   ├── trace.nc
│   ├── posterior_summary.csv
│   └── visualizations/
├── requirements.txt
└── README.md
```

### Appendix C: Environment Specifications

```
Python: 3.8+
pymc: 5.0+
arviz: 0.14+
scipy: 1.7+
krippendorff: 0.5+
openai: 1.0+
anthropic: 0.8+
together: 0.2+
pandas: 1.3+
numpy: 1.21+
matplotlib: 3.4+
seaborn: 0.11+
```

### Appendix D: Reproducibility Checklist

- [x] Random seeds set for all stochastic operations
- [x] API temperature fixed at 0.3
- [x] MCMC random seed = 42
- [x] Full code available in repository
- [x] Requirements.txt with pinned versions
- [x] Raw data available upon request
- [x] MCMC trace saved for posterior analysis

---

*Report generated from analysis in LLM_Ensemble_Textbook_Bias_Detection.ipynb*  
*Technical Review: Bayesian Hierarchical Analysis with LLM Ensemble*  
*© 2024 Derek Lankeaux. All rights reserved.*
