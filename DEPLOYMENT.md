# VoxIntel Command Arena — deployment, MLOps & free-tier plan

The arena is a CPU-only FastAPI app (no torch in the default image): it serves the
reliability game UI at `/` and the API at `/arena/*`, `/reliability`, `/health`.
Everything runs off two committed artifacts — the pickled risk model and one SLURP
feature CSV — so it boots in seconds with no GPU and no model downloads.

## 0. Complete local runbook (conda `torch26`)

The `torch26` env has everything (torch + transformers + sklearn + fastapi), so one
env runs the research, the Whisper experiment, and the app.

    conda activate torch26
    cd C:\Users\ACER\OneDrive\Desktop\VoxIntel
    # (fresh env only) pip install -r requirements.txt

    # 1. Build the reliability model the serving layer + arena load (seed 42).
    python src/analysis/frozen_split_eval.py            # -> models/voxintel_r_intent_rf.joblib

    # 2. Self-checks + tests (all should pass).
    python src/analysis/reliability_metrics.py
    python tests/test_reliability.py
    python tests/test_arena.py

    # 3. Launch the VoxIntel Command Arena, then open http://localhost:8000
    python -m uvicorn src.serving.app:create_app --factory --port 8000

    # 4. Whisper cross-architecture experiment (phase 9, GPU).
    python src/analysis/whisper_asr_eval.py --limit 8      # smoke (seconds)
    python src/analysis/whisper_asr_eval.py                # full 8,688-clip decode + H1 (resumable)
    python src/analysis/whisper_asr_eval.py --eval         # re-run H1 on the cached CSV only
    python notebooks/_gen_whisper.py                       # (re)generate the notebook
    python -m nbconvert --to notebook --execute --inplace notebooks/retraining_on_whisper.ipynb

    # 5. Container (optional; needs Docker Desktop running).
    docker build -t voxintel-arena .
    docker run -p 8000:8000 voxintel-arena                 # http://localhost:8000

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
    python tests/test_arena.py           # arena unit + FastAPI e2e tests

Config (environment variables):

| var | default | meaning |
|---|---|---|
| `VOXINTEL_FEATURES` | `reports/phase6_voxintel_r_slurp/voxintel_r_features.csv` | round source table |
| `VOXINTEL_RISK_LOW` | `0.33` | risk < low → recommend **execute** |
| `VOXINTEL_RISK_HIGH` | `0.66` | risk ≥ high → recommend **abstain** (between → **clarify**) |

## 2. Run with Docker

    python src/analysis/frozen_split_eval.py          # the risk model must exist locally
    docker build -t voxintel-arena .
    docker run -p 8000:8000 voxintel-arena            # http://localhost:8000

The image is CPU-only (`python:3.11-slim` + scikit-learn/pandas/fastapi), runs as a
non-root user (`appuser`, uid 10001), and ships a `HEALTHCHECK` that polls `/health`.
Live typed-command inference (DistilBERT) is intentionally **not** in this image —
it would pull torch+transformers and multiply the image size; the arena's core loop
is fully real without it, and the "type your own command" box falls back to a
clearly-labelled demo. (Build/run was authored and reviewed here but not executed:
no Docker daemon was available in the audit environment — see the run report.)

<!-- placeholder:deploy -->
## 3. Free-tier deployment (verified Oct 2026)

The app is a single Docker container that scales to zero, so it fits a genuine
free tier. Constraints change often — these were checked in Oct 2026; re-verify
before you rely on them.

**Primary — Google Cloud Run (recommended).** Real Docker, scale-to-zero (no cost
when idle), and a perpetual monthly free allowance (≈2M requests + 360k vCPU-seconds
+ 180k GiB-seconds, mid-2026). Deploy the Dockerfile as-is:

    gcloud run deploy voxintel-arena --source . --region <r> --allow-unauthenticated --memory 512Mi

**Secondary — Render free "Hobby" web service.** $0, 512 MB RAM / 0.1 CPU, 750
instance-hours/month, spins down after 15 min idle (≈1 min cold start), 5 GB
bandwidth/month (reduced Apr 2026). Point it at this repo; it builds the Dockerfile.
The 512 MB / 0.1 CPU envelope is enough for the sklearn RF + the one CSV.

**Not free anymore:** Hugging Face Spaces **Docker/Gradio** Spaces now require a
paid PRO plan (as of mid-2026); only **Static** Spaces are free, which cannot run
this backend. Don't use HF Spaces for the API.

In every case the app ships with **no auth or rate limiting** — deploy private, or
put an auth gateway / Cloud Run IAM in front before exposing it publicly.

## 4. MLOps plan (free, no paid infra)

- **Model & experiment versioning.** The risk model is a self-describing joblib
  bundle (`{model, features, threshold, trained_on, target}`); the phase-8 bake-off
  persists `nb20_best_model.json` next to it. Reproducible regeneration:
  `python src/analysis/frozen_split_eval.py` (seed 42, frozen 70/30 split). For
  heavier tracking, MLflow (local file store, free) is listed optional in
  `requirements.txt`; the current pipeline doesn't need it.
- **Evaluation (the quality gate).** `tests/test_reliability.py` asserts the
  frozen-split invariants (intent-only AUC > 0.80, H1 verdict consistent with its
  CI) and `tests/test_arena.py` the serving/scoring contract. Both run in CI
  (`.github/workflows/tests.yml`) on every push/PR — free on GitHub Actions.
- **Monitoring.** `/health` exposes model + arena status for an uptime check; the
  arena already aggregates live reliability metrics (coverage, execute-accuracy,
  failure-catch rate) at `/arena/metrics`. The shipped Mahalanobis support gate
  (`nb22_cross_dataset.json`) is the drift primitive — log inputs and alert when the
  out-of-support flag rate climbs. (Honest caveat from the research: a density gate
  alone misses concept shift; pair it with output-distribution monitoring.)
- **Reproducibility.** Pinned `requirements.txt`, fixed seed 42 everywhere, a frozen
  split shared across scripts and notebooks, and headline numbers that regenerate
  from cached features with no audio/GPU.

## 5. Limitations

- Arena rounds are drawn from **SLURP-dev** only; the score reflects reliability
  behaviour on that distribution, not a live production stream.
- The default image scores cached commands with the real risk model; **live** ASR→
  intent on arbitrary typed/spoken input needs the torch image (not built here).
- In-memory session history (one process) — fine for a demo; add SQLite only if you
  need cross-process persistence.
- Free tiers spin down / are rate-limited; expect cold starts, not production SLAs.

