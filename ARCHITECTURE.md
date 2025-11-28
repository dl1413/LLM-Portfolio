# 🏗️ System Architecture

## Overview

This document describes the production ML system architecture for both portfolio projects, demonstrating scalable, fault-tolerant design principles applicable to frontier AI applications.

---

## Project 1: Breast Cancer Classification Pipeline

### Production ML Pipeline Architecture

```
┌─────────────────┐
│  Raw Data Input │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Preprocessing Pipeline     │
│  • VIF Analysis             │
│  • SMOTE Balancing          │
│  • Feature Scaling          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Parallel Model Training    │
│  • 8 Ensemble Methods       │
│  • Stratified 5-Fold CV     │
│  • Hyperparameter Tuning    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Model Evaluation & Selection│
│  • ROC-AUC Analysis         │
│  • Clinical Metrics         │
│  • Performance Profiling    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Production Deployment      │
│  • Model Persistence        │
│  • REST API (FastAPI)       │
│  • Docker Container         │
│  • Monitoring & Logging     │
└─────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Modularity** | Each pipeline stage is independently testable |
| **Fault Tolerance** | Graceful handling of missing data, model failures |
| **Scalability** | Parallel cross-validation, efficient data structures |
| **Observability** | Comprehensive logging, performance metrics |
| **Reproducibility** | Fixed random seeds, versioned dependencies |

### API Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
│                   (nginx / ALB)                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐          ┌───────────────┐
│   API Pod 1   │          │   API Pod 2   │
│   (FastAPI)   │          │   (FastAPI)   │
└───────┬───────┘          └───────┬───────┘
        │                           │
        └─────────────┬─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Registry                            │
│                   (MLflow / S3)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Project 2: LLM Ensemble Bias Detection

### Distributed LLM Ensemble Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Orchestration Layer                      │
│  • Rate Limiting (60 req/min per endpoint)               │
│  • Circuit Breakers (automatic failover)                 │
│  • Retry Logic (exponential backoff)                     │
└────────────┬─────────────────────────┬───────────────────┘
             │                         │
    ┌────────▼────────┐      ┌────────▼────────┐
    │  Load Balancer  │      │  Cache Layer    │
    │  (Round Robin)  │      │  (Redis/Memory) │
    └────────┬────────┘      └─────────────────┘
             │
    ┌────────▼────────────────────────────────┐
    │      Parallel Processing Pool            │
    │  ThreadPoolExecutor (max_workers=10)    │
    └─┬──────┬──────┬──────────────────────────┘
      │      │      │
   ┌──▼──┐ ┌▼───┐ ┌▼────┐
   │GPT-4│ │Claude│ │Llama│
   │ API │ │ API │ │ API │
   └─────┘ └────┘ └─────┘
```

### Components

#### 1. API Orchestration Layer

```python
class LLMOrchestrator:
    """Manages multi-model API calls with fault tolerance."""
    
    def __init__(self):
        self.clients = {
            'gpt4': OpenAIClient(rate_limit=60),
            'claude3': AnthropicClient(rate_limit=60),
            'llama3': TogetherClient(rate_limit=60)
        }
        self.circuit_breakers = {
            model: CircuitBreaker(failure_threshold=5)
            for model in self.clients
        }
        
    async def call_with_retry(self, model, prompt, max_retries=3):
        """Call LLM with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                if self.circuit_breakers[model].is_open():
                    raise CircuitBreakerOpen(f"{model} unavailable")
                    
                response = await self.clients[model].complete(prompt)
                self.circuit_breakers[model].record_success()
                return response
                
            except RateLimitError:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
            except APIError as e:
                self.circuit_breakers[model].record_failure()
                raise
                
        raise MaxRetriesExceeded()
```

#### 2. Parallel Processing Pipeline

```python
def process_passages_parallel(passages, models, max_workers=10):
    """Process passages in parallel across all models."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_passage = {
            executor.submit(rate_passage, passage, model): (passage, model)
            for passage in passages
            for model in models
        }
        
        for future in tqdm(as_completed(future_to_passage)):
            passage, model = future_to_passage[future]
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed: {e}")
                results.append(None)
                
    return results
```

#### 3. Bayesian Inference Engine

```python
class BayesianInferencePipeline:
    """Production Bayesian hierarchical modeling with PyMC."""
    
    def __init__(self, n_chains=4, n_iterations=2000):
        self.n_chains = n_chains
        self.n_iterations = n_iterations
        self.convergence_threshold = 1.01  # R-hat
        
    def sample_with_diagnostics(self, model):
        """Sample with comprehensive diagnostics."""
        with model:
            trace = pm.sample(
                draws=self.n_iterations,
                chains=self.n_chains,
                cores=self.n_chains,
                return_inferencedata=True
            )
            
            # Validate convergence
            rhat = az.rhat(trace)
            ess = az.ess(trace)
            self.validate_convergence(rhat, ess)
            
            return trace
```

---

## Data Flow Diagrams

### Breast Cancer Classification

```
Input Layer           Processing Layer         Output Layer
───────────────────────────────────────────────────────────
                                              
[Patient       ]     [StandardScaler  ]     [Prediction    ]
[Features (30) ] ──▶ [Transform       ] ──▶ [Probability   ]
                     [               ]     [Confidence    ]
                            │
                            ▼
                     [RFE Selector    ]
                     [15 Features     ]
                            │
                            ▼
                     [AdaBoost Model  ]
                     [99.12% Accuracy ]
```

### LLM Ensemble Bias Detection

```
Input Layer           Processing Layer         Output Layer
───────────────────────────────────────────────────────────

[Textbook      ]     [LLM Ensemble    ]     [Bias Score    ]
[Passage       ] ──▶ [GPT-4, Claude,  ] ──▶ [Confidence    ]
[              ]     [Llama           ]     [Explanation   ]
                            │
                            ▼
                     [Aggregation     ]
                     [Krippendorff α  ]
                            │
                            ▼
                     [Bayesian Model  ]
                     [Publisher Effects]
```

---

## Monitoring & Observability

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prediction_counter = Counter(
    'predictions_total', 
    'Total number of predictions',
    ['model', 'prediction']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

model_confidence = Histogram(
    'model_confidence',
    'Prediction confidence distribution',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)
```

### Logging Configuration

```python
import structlog

logger = structlog.get_logger()
logger.info("Prediction made", extra={
    "prediction": "Benign",
    "confidence": 0.95,
    "latency_ms": 0.8,
    "model_version": "v1.0"
})
```

### Alerting Rules

| Alert | Condition | Action |
|-------|-----------|--------|
| High Error Rate | 5xx > 5% (5min) | Page on-call |
| Slow Predictions | p95 latency > 100ms | Investigate |
| Low Confidence | confidence < 0.7 rate > 10% | Review model |
| API Quota | usage > 80% daily | Scale up |

---

## Deployment Architecture

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: breast-cancer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: breast-cancer-api
  template:
    spec:
      containers:
      - name: api
        image: breast-cancer-api:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
```

### Production Considerations

| Aspect | Implementation |
|--------|----------------|
| **Horizontal Scaling** | Stateless API design enables load balancing |
| **Vertical Scaling** | 10x speedup through vectorization |
| **Fault Tolerance** | Automatic retry with exponential backoff |
| **Monitoring** | Prometheus metrics, structured logging |
| **Security** | Input validation, rate limiting, secrets management |

---

*Architecture documentation updated November 2025*
