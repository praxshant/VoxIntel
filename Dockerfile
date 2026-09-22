# Lean image for the reference-free VoxIntel-R reliability endpoint.
#
# Serves POST /reliability (+ /health). /predict is present but returns an
# "attach a model" error here: scoring an HF intent model would additionally
# need torch+transformers — add them below if you serve /predict.
#
# SECURITY: the app ships with NO auth or rate limiting (see src/serving/app.py).
# This container is for local/eval use — put an auth gateway in front before
# exposing it beyond localhost.
#
# Build (from repo root; the risk model must exist locally — it is gitignored
# and produced by `python src/analysis/frozen_split_eval.py`):
#     docker build -t voxintel-r .
#     docker run -p 8000:8000 voxintel-r
FROM python:3.11-slim

WORKDIR /app

# Versions pinned to the env the RF was pickled in, so joblib.load stays
# compatible across rebuilds.
RUN pip install --no-cache-dir \
    scikit-learn==1.6.1 numpy==2.1.3 pandas==2.2.3 joblib==1.4.2 scipy==1.15.3 \
    fastapi==0.115.0 "uvicorn==0.53.0" pydantic==2.9.2

COPY src/ ./src/
COPY models/voxintel_r_intent_rf.joblib ./models/voxintel_r_intent_rf.joblib

EXPOSE 8000

# --factory: app.py exposes create_app(), not a module-level `app`.
CMD ["uvicorn", "src.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
