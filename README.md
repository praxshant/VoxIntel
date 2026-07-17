# VoxIntel — Production-Grade Speech Understanding with ASR Adaptation, Intent Learning, and MLOps

## Overview

VoxIntel is an end-to-end Speech AI project that studies how improvements in Automatic Speech Recognition (ASR) affect downstream Natural Language Understanding (NLU), specifically intent classification.

The project is designed as both:

- an **applied machine learning research system**, and
- a **production-oriented ML platform** with MLOps components such as experiment tracking, model versioning, deployment, and monitoring.

Instead of only using pretrained speech models, VoxIntel focuses on training, fine-tuning, evaluating, optimizing, and deploying a complete speech-to-intent pipeline.

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

In practice, ASR systems are usually evaluated using metrics like Word Error Rate (WER), while intent models are evaluated using Accuracy, Precision, Recall, and F1-score.

However, an important real-world question remains:

> If ASR transcription quality improves, does downstream intent understanding improve as well?

VoxIntel is built to answer this question through controlled experiments using a single spoken language understanding dataset.

---

## Research Question

### Primary Question
How much does improving speech transcription quality improve downstream intent classification?

### Extended Research Question
What is the relationship between ASR transcription quality (WER/CER) and downstream intent classification accuracy across different intent categories?

---

## Version 1 Scope

For Version 1, VoxIntel uses a single dataset:

- **Dataset:** SLURP
- **ASR Model:** `facebook/wav2vec2-base-960h`
- **Intent Model:** `distilbert-base-uncased`

SLURP is chosen because it provides all required components in one dataset:

- audio
- ground-truth transcript
- intent labels

This makes the experiments cleaner, fairer, and easier to interpret.

---

## Architecture Diagram

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
    │ Intent Inference    │
    │ Transcript → Intent │
    └─────────┬───────────┘
              ▼
    ┌─────────────────────┐
    │ Evaluation          │
    │ WER, CER, F1, Acc   │
    │ Latency, Size       │
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
Ground Truth Transcript
        ↓
DistilBERT
        ↓
Intent
```

**Goal:** establish the best possible intent classification performance.

### Experiment 2 — Baseline ASR → Intent Pipeline
Use a pretrained ASR model without task-specific fine-tuning.

```text
Audio
  ↓
Pretrained Wav2Vec2
  ↓
Transcript
  ↓
DistilBERT
  ↓
Intent
```

**Goal:** measure how transcription errors reduce downstream intent accuracy.

### Experiment 3 — Fine-Tuned ASR → Intent Pipeline
Fine-tune the ASR model on SLURP and regenerate transcripts.

```text
Audio
  ↓
Fine-Tuned Wav2Vec2
  ↓
Transcript
  ↓
DistilBERT
  ↓
Intent
```

**Goal:** test whether lowering WER/CER improves downstream intent prediction.

### Experiment 4 — Optimization and Inference Efficiency
Apply model optimization to improve deployment efficiency.

Possible techniques:
- Dynamic INT8 Quantization
- CPU latency benchmarking
- Model size comparison

**Goal:** evaluate the tradeoff between speed, model size, and accuracy.

### Experiment 5 — Error and Sensitivity Analysis
Analyze where ASR errors matter most.

Examples:
- intent classes most sensitive to transcription noise
- examples where one wrong word changes the intent
- correlation between WER and intent accuracy
- confidence degradation under noisy transcripts

---

## Evaluation Metrics

### ASR Metrics
- Word Error Rate (WER)
- Character Error Rate (CER)
- Training Loss
- Validation Loss
- Inference Latency

### Intent Metrics
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

### Production / Optimization Metrics
- Inference Latency
- Model Size
- Memory Usage
- WER after Quantization
- API response time

---

## MLOps Components

VoxIntel is also intended to function as a production-grade ML system.

Planned MLOps features include:

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

## Expected Deliverables

By the end of the project, the repository should contain:

- A fine-tuned Wav2Vec2 ASR model
- A fine-tuned DistilBERT intent classifier
- Controlled comparison between:
  - ground-truth transcripts
  - baseline ASR transcripts
  - fine-tuned ASR transcripts
- A complete evaluation report with:
  - WER / CER
  - Accuracy / Precision / Recall / Macro F1
  - confusion matrix
  - error analysis
- A quantized or optimized inference-ready model
- A FastAPI endpoint for speech-to-intent prediction
- MLOps setup with:
  - experiment tracking
  - data versioning
  - model versioning
  - reproducible runs
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

### Phase 1
- Set up repository
- Write README
- Create requirements
- Initialize Git, DVC, and MLflow structure

### Phase 2
- Load and inspect SLURP
- Create train/validation/test manifests
- Run baseline ASR inference
- Compute baseline WER/CER

### Phase 3
- Fine-tune Wav2Vec2
- Evaluate ASR performance
- Log results to MLflow

### Phase 4
- Train DistilBERT on clean transcripts
- Train DistilBERT on ASR-generated transcripts
- Compare downstream performance

### Phase 5
- Optimize inference
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

VoxIntel aims to demonstrate not only that speech and language models can be trained effectively, but also that they can be evaluated, optimized, versioned, deployed, and maintained as a real production ML system.

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