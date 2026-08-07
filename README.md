# VoxIntel — Error-Aware Speech Understanding with ASR Adaptation, Intent Robustness, Error Taxonomy, and MLOps

## Overview

VoxIntel is an end-to-end Speech AI project that investigates how Automatic Speech Recognition (ASR) transcription errors propagate into downstream Natural Language Understanding (NLU), specifically intent classification.

Traditional ASR systems are evaluated almost exclusively using **Word Error Rate (WER)** and **Character Error Rate (CER)**. These metrics treat every transcription error as equally harmful, but downstream language understanding does not:

- "turn **off** the lights" → "turn **of** the lights" — high textual similarity, intent unchanged.
- "turn **off** the lights" → "turn **on** the lights" — similar WER increase, but the resulting action is completely reversed.

VoxIntel moves beyond the question **"Can we reduce WER?"** and instead asks:

> **Which ASR errors actually matter for downstream understanding, which intents are most fragile, and how can production Speech AI systems become more robust to these failures?**

The project is designed as both:

- an **applied machine learning research system**, studying error propagation and intent robustness, and
- a **production-oriented ML platform**, with MLOps components such as experiment tracking, model versioning, deployment, and monitoring.

---

## Problem Statement

Modern voice assistants and conversational AI systems typically work in multiple stages:

```text
Speech
  ↓
Automatic Speech Recognition (ASR)
  ↓
Transcript
  ↓
Natural Language Understanding (NLU)
  ↓
Intent Prediction
```

ASR systems are usually evaluated with WER, while intent models are evaluated with Accuracy, Precision, Recall, and F1-score. These two evaluation worlds rarely meet. VoxIntel connects them directly by asking:

> If ASR transcription quality improves, does downstream intent understanding improve as well — and if not, which *kinds* of errors are responsible?

---

## Research Objectives

1. Fine-tune a Wav2Vec2 ASR model on the SLURP dataset.
2. Compare baseline and fine-tuned ASR performance using WER and CER.
3. Train and evaluate an intent classification model using:
   - Ground-truth transcripts
   - Baseline ASR transcripts
   - Fine-tuned ASR transcripts
4. Quantify how transcription errors affect downstream intent prediction.
5. Build an **Error Taxonomy** to categorize ASR mistakes and analyze their downstream impact.
6. Develop a production-ready speech-to-intent pipeline with deployment, monitoring, and reproducibility using modern MLOps practices.

---

## Research Questions

**RQ1** — How much can Wav2Vec2 fine-tuning improve ASR performance on the SLURP dataset?

**RQ2** — Do reductions in WER and CER consistently translate into improvements in downstream intent classification?

**RQ3** — Which intent classes are most sensitive to transcription errors?

**RQ4** — Which categories of ASR errors have the greatest impact on intent understanding?

**RQ5** — Can downstream intent models remain robust despite imperfect ASR transcripts?

---

## Error Taxonomy

Instead of treating all ASR errors equally, VoxIntel categorizes transcription mistakes into meaningful groups and studies their downstream impact independently.

| Error Type          | Example                     | Potential Impact |
|---------------------|------------------------------|-------------------|
| Substitution        | "off" → "on"                 | High              |
| Deletion            | Missing important keyword    | High              |
| Insertion           | Extra filler words            | Low–Medium        |
| Named Entity Error  | "John" → "Joan"               | High              |
| Number Error        | "five" → "nine"               | High              |
| Negation Error      | "don't" → "do"                | Critical          |
| Location Error      | "kitchen" → "bedroom"         | High              |
| Function Word Error | "the" omitted                  | Usually Low       |

Each category is analyzed for how often it occurs and how often it flips or degrades the predicted intent, producing a ranked view of which error types are most responsible for downstream failures — rather than a single aggregate WER number.

---

## Version 1 Scope

For Version 1, VoxIntel uses a single dataset to keep experiments controlled and interpretable:

- **Dataset:** SLURP
- **ASR Model:** `facebook/wav2vec2-base-960h`
- **Intent Model:** `distilbert-base-uncased`

SLURP is chosen because it provides all required components in one dataset — audio, ground-truth transcript, and intent labels — making comparisons across transcript sources fair and clean.

---

## Project Pipeline

```text
                    ┌────────────────────┐
                    │    SLURP Dataset   │
                    │ audio + transcript │
                    │   + intent labels  │
                    └─────────┬──────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
    ┌─────────────────────┐       ┌────────────────────────┐
    │   ASR Training      │       │ Intent Model Training  │
    │ Wav2Vec2 Fine-Tune  │       │ DistilBERT Fine-Tune   │
    └─────────┬───────────┘       └────────────┬───────────┘
              │                                │
              ▼                                │
    ┌─────────────────────┐                    │
    │  Generated          │                    │
    │  Transcripts        │────────────────────┘
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │ Error Taxonomy &    │
    │ ASR Error Analysis  │
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │ Intent Inference    │
    │ Transcript → Intent │
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │ Intent Robustness & │
    │ Error Propagation   │
    │ Analysis            │
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │ Deployment + MLOps  │
    │ FastAPI, MLflow,    │
    │ DVC, Docker, CI/CD  │
    └─────────────────────┘
```

---

## Tech Stack

### Core ML / NLP / Speech
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- torchaudio
- jiwer
- evaluate
- scikit-learn

### Data and Visualization
- pandas
- numpy
- matplotlib
- seaborn

### MLOps / Production
- MLflow
- DVC
- FastAPI
- Uvicorn
- Docker
- GitHub Actions

### Optional Extensions
- Evidently (monitoring / drift analysis)
- Prefect (pipeline orchestration)

---

## Planned Experiments

### Experiment 1 — Upper Bound Intent Classification
Use the clean ground-truth transcript from SLURP.

```text
Ground Truth Transcript → DistilBERT → Intent
```
**Goal:** establish the best possible intent classification performance.

### Experiment 2 — Baseline ASR → Intent Pipeline
Use a pretrained ASR model without task-specific fine-tuning.

```text
Audio → Pretrained Wav2Vec2 → Transcript → DistilBERT → Intent
```
**Goal:** measure how transcription errors reduce downstream intent accuracy.

### Experiment 3 — Fine-Tuned ASR → Intent Pipeline
Fine-tune the ASR model on SLURP and regenerate transcripts.

```text
Audio → Fine-Tuned Wav2Vec2 → Transcript → DistilBERT → Intent
```
**Goal:** test whether lowering WER/CER improves downstream intent prediction.

### Experiment 4 — Error Taxonomy & Sensitivity Analysis
Categorize every ASR error (substitution, deletion, insertion, named entity, number, negation, location, function word) and measure its individual effect on intent accuracy.

**Goal:** identify which specific error categories, not just error rate, drive downstream failures — and which intent classes are most fragile.

### Experiment 5 — Optimization and Inference Efficiency
Apply model optimization to improve deployment efficiency.

Possible techniques:
- Dynamic INT8 Quantization
- CPU latency benchmarking
- Model size comparison

**Goal:** evaluate the tradeoff between speed, model size, and accuracy.

---

## Evaluation Metrics

### ASR Metrics
- Word Error Rate (WER)
- Character Error Rate (CER)
- Training / Validation Loss
- Inference Latency

### Intent Metrics
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

### Error Taxonomy Metrics
- Frequency per error category
- Intent-flip rate per error category
- Confidence degradation per error category
- Correlation between WER and intent accuracy

### Production / Optimization Metrics
- Inference Latency
- Model Size
- Memory Usage
- WER after Quantization
- API response time

---

## MLOps Components

VoxIntel is also intended to function as a production-grade ML system. Planned MLOps features include:

- Experiment Tracking with MLflow
- Data & Artifact Versioning with DVC
- Model Registry for trained ASR and intent models
- Evaluation Gates before model promotion
- FastAPI Inference Service
- Dockerized Deployment
- CI/CD with GitHub Actions
- Monitoring and Logging for latency, confidence, and drift
- Retraining Workflow for future model updates

---

## Research Contribution

VoxIntel contributes an end-to-end evaluation framework that combines:

- ASR fine-tuning
- Downstream intent classification
- Error propagation analysis
- Error taxonomy and sensitivity analysis
- Intent robustness evaluation
- Production deployment
- MLOps and monitoring

into a single, coherent study of *which* ASR errors matter for real-world spoken language understanding, rather than a narrow focus on reducing WER.

---

## Expected Deliverables

By the end of the project, the repository should contain:

- A fine-tuned Wav2Vec2 ASR model
- A fine-tuned DistilBERT intent classifier
- Controlled comparison between ground-truth, baseline ASR, and fine-tuned ASR transcripts
- A complete evaluation report with WER/CER, Accuracy/Precision/Recall/Macro F1, confusion matrix, and error analysis
- An Error Taxonomy report ranking error categories by downstream impact
- A quantized or optimized inference-ready model
- A FastAPI endpoint for speech-to-intent prediction
- MLOps setup with experiment tracking, data versioning, model versioning, and reproducible runs
- A documented GitHub repository with scripts, configs, reports, and visualizations

---

## Suggested Repository Structure

```text
VoxIntel/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── notebooks/
├── configs/
├── src/
│   ├── data/
│   ├── asr/
│   ├── intent/
│   ├── error_taxonomy/
│   ├── evaluation/
│   ├── optimization/
│   ├── pipeline/
│   ├── serving/
│   └── monitoring/
├── experiments/
├── reports/
├── models/
├── tests/
├── requirements.txt
├── Dockerfile
├── dvc.yaml
├── Makefile
└── README.md
```

---

## Initial Milestones

### Phase 1 — Setup
- Set up repository, write README, create requirements
- Initialize Git, DVC, and MLflow structure

### Phase 2 — Data & Baseline
- Load and inspect SLURP
- Create train/validation/test manifests
- Run baseline ASR inference
- Compute baseline WER/CER

### Phase 3 — ASR Fine-Tuning
- Fine-tune Wav2Vec2
- Evaluate ASR performance
- Log results to MLflow

### Phase 4 — Intent Modeling
- Train DistilBERT on clean transcripts
- Train DistilBERT on ASR-generated transcripts
- Compare downstream performance

### Phase 5 — Error Taxonomy
- Categorize transcription errors per taxonomy
- Measure intent-flip rate and confidence degradation per category
- Identify most fragile intent classes

### Phase 6 — Deployment
- Optimize inference (quantization, latency benchmarking)
- Build FastAPI endpoint
- Containerize and document deployment workflow

---

## Long-Term Extensions

Future versions may include:

- Cross-dataset generalization testing
- Noise robustness experiments
- Confidence calibration
- Drift monitoring on live inference logs
- Retraining and model promotion workflows
- Support for additional speech datasets such as Fluent Speech Commands or Common Voice

---

## Project Goal

VoxIntel aims to demonstrate not only that speech and language models can be trained effectively, but that transcription errors can be understood, categorized, and diagnosed — and that the resulting speech-to-intent system can be evaluated, optimized, versioned, deployed, and maintained as a real production ML system.

---

## Installation

```bash
git clone https://github.com/<your-username>/VoxIntel.git
cd VoxIntel
pip install -r requirements.txt
```

## Requirements

```txt
# Core ML / Speech / NLP
torch
torchaudio
transformers
datasets
accelerate
evaluate
jiwer
scikit-learn

# Data / Visualization
numpy
pandas
matplotlib
seaborn

# Audio + utilities
soundfile
librosa

# API / Serving
fastapi
uvicorn
python-multipart
pydantic

# MLOps
mlflow
dvc

# Experiment / config / utils
pyyaml
tqdm
joblib

# Optional monitoring / drift
evidently
```