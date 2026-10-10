# VoxIntel Reliability Dashboard — deployment, MLOps & free-tier plan

A CPU-only FastAPI app (no torch in the default image): it serves the reliability
dashboard UI at `/` and the API at `/eval/*`, `/inspect`, `/reliability`, `/health`.
Everything runs off two committed artifacts — the pickled risk model and one SLURP
feature CSV — so it boots in seconds with no GPU and no model downloads. The dashboard
reproduces the frozen-split evaluation live (ROC-AUC, Brier, ECE, risk–coverage / AURC
vs the MSP baseline, coverage@risk) and the inspector scores a typed command on the
same model.

## 0. Complete local runbook (conda `torch26`)

The `torch26` env has everything (torch + transformers + sklearn + fastapi), so one
env runs the research, the Whisper experiment, and the app. With torch present, the
inspector uses the **real** DistilBERT intent model.

    conda activate torch26
    cd C:\Users\ACER\OneDrive\Desktop\VoxIntel
    # (fresh env only) pip install -r requirements.txt

    # 1. Build the reliability model the dashboard loads (seed 42).
    python src/analysis/frozen_split_eval.py            # -> models/voxintel_r_intent_rf.joblib

    # 2. Self-checks + tests (all should pass).
    python src/analysis/reliability_metrics.py
    python tests/test_reliability.py
    python tests/test_eval.py

    # 3. Launch the dashboard, then open http://localhost:8000
    python -m uvicorn src.serving.app:create_app --factory --port 8000

    # 4. Whisper cross-architecture experiment (phase 9, GPU).
    python src/analysis/whisper_asr_eval.py --limit 8      # smoke (seconds)
    python src/analysis/whisper_asr_eval.py                # full 8,688-clip decode + H1 (resumable)
    python src/analysis/whisper_asr_eval.py --eval         # re-run H1 on the cached CSV only
    python notebooks/_gen_whisper.py                       # (re)generate the notebook
    python -m nbconvert --to notebook --execute --inplace notebooks/retraining_on_whisper.ipynb

    # 5. Container (optional; needs Docker Desktop running).
    docker build -t voxintel .
    docker run -p 8000:8000 voxintel                       # http://localhost:8000

Note: regenerating the model in `torch26` pickles it with that env's scikit-learn; the
Docker image pins scikit-learn 1.6.1 to match the shipped joblib, so rebuild the image
from a 1.6.1 model (or accept a version-mismatch warning on load) if you re-pickle it.

## 1. Run locally (no Docker)

    python src/analysis/frozen_split_eval.py          # one-time: writes the risk model joblib
    pip install fastapi uvicorn pandas scikit-learn joblib
    python -m uvicorn src.serving.app:create_app --factory --port 8000
    # open http://localhost:8000

Self-checks / tests:

    python src/serving/app.py            # reliability demo
    python tests/test_reliability.py     # reliability-path unit tests
    python tests/test_eval.py            # dashboard metrics + FastAPI e2e tests

Config (environment variables):

| var | default | meaning |
|---|---|---|
| `VOXINTEL_FEATURES` | `reports/phase6_voxintel_r_slurp/voxintel_r_features.csv` | eval source table |
| `VOXINTEL_RISK_LOW` | `0.33` | risk < low → recommend **execute** |
| `VOXINTEL_RISK_HIGH` | `0.66` | risk ≥ high → recommend **abstain** (between → **clarify**) |
| `VOXINTEL_INTENT_MODEL` | auto | path to the DistilBERT intent model for the live inspector; `off` forces demo |

<!-- placeholder:rest -->
## 2. Run with Docker

    python src/analysis/frozen_split_eval.py          # the risk model must exist locally
    docker build -t voxintel .
    docker run -p 8000:8000 voxintel                  # http://localhost:8000

The image is CPU-only (`python:3.11-slim` + scikit-learn/pandas/fastapi), runs as a
non-root user (`appuser`, uid 10001), and ships a `HEALTHCHECK` that polls `/health`.
Live typed-command inference (DistilBERT) is intentionally **not** in this image — it
would pull torch+transformers and multiply the image size; the dashboard metrics are
fully real without it, and the inspector falls back to a clearly-labelled demo.
(Verified: `docker build` + `docker run` succeed; `/health` healthy, non-root,
endpoints respond; image ≈ 844 MB.)

## 3. Free-tier deployment (verified Oct 2026)

The app is a single Docker container that scales to zero, so it fits a genuine free
tier. Constraints change often — these were checked Oct 2026; re-verify before relying.

**Primary — Google Cloud Run (recommended).** Real Docker, scale-to-zero (no cost when
idle), perpetual monthly free allowance (≈2M requests + 360k vCPU-seconds + 180k
GiB-seconds, mid-2026). Deploy the Dockerfile as-is:

    gcloud run deploy voxintel --source . --region <r> --allow-unauthenticated --memory 512Mi

**Secondary — Render free "Hobby" web service.** $0, 512 MB RAM / 0.1 CPU, 750
instance-hours/month, spins down after 15 min idle (≈1 min cold start), 5 GB
bandwidth/month (reduced Apr 2026). The sklearn RF + one CSV fit the envelope.

**Not free anymore:** Hugging Face Spaces **Docker/Gradio** Spaces require a paid PRO
plan (mid-2026); only **Static** Spaces are free, which can't run this backend.

In every case the app ships with **no auth or rate limiting** — deploy private, or put
an auth gateway / Cloud Run IAM in front before exposing it publicly.

## 4. MLOps plan (free, no paid infra)

- **Model & experiment versioning.** The risk model is a self-describing joblib bundle;
  the phase-8 bake-off persists `nb20_best_model.json`. Reproducible regeneration:
  `python src/analysis/frozen_split_eval.py` (seed 42, frozen 70/30 split). MLflow
  (local file store, free) is listed optional in `requirements.txt`; not needed now.
- **Evaluation (the quality gate).** `tests/test_reliability.py` asserts the
  frozen-split invariants and `tests/test_eval.py` the dashboard metrics (model beats
  the MSP baseline on AURC) + serving contract. Both run in CI
  (`.github/workflows/tests.yml`) on every push/PR — free on GitHub Actions.
- **Monitoring.** `/health` exposes model + dashboard status for an uptime check;
  `/eval/summary` serves the live held-out metrics. The shipped Mahalanobis support
  gate (`nb22_cross_dataset.json`) is the drift primitive — log inputs and alert when
  the out-of-support flag rate climbs. (Honest caveat from the research: a density gate
  alone misses concept shift; pair it with output-distribution monitoring.)
- **Reproducibility.** Pinned `requirements.txt`, fixed seed 42 everywhere, a frozen
  split shared across scripts and notebooks, headline numbers regenerated from cached
  features with no audio/GPU.

## 5. Limitations

- Dashboard metrics are on the **SLURP-dev** frozen split — a held-out test estimate,
  not a live production stream.
- The default image computes metrics + demo inspector; **live** typed ASR→intent needs
  the torch env (the `torch26` runbook), not the lean container.
- Free tiers spin down / are rate-limited; expect cold starts, not production SLAs.

