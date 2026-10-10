# VoxIntel Command Arena + reference-free reliability endpoint.
#
# Serves the arena UI (/) and the reliability API (/arena/*, /reliability, /health).
# /predict returns an "attach a model" error here: live ASR/intent inference would
# add torch+transformers (a much larger image). The arena's core loop is fully real
# without them — it scores cached SLURP commands with the pickled RandomForest.
#
# SECURITY: no auth or rate limiting (see src/serving/app.py). For local / eval /
# a private free-tier Space only — put an auth gateway in front before public use.
#
# Build + run (from repo root; the risk model + feature CSV must exist locally):
#     python src/analysis/frozen_split_eval.py        # produces the risk model
#     docker build -t voxintel-arena .
#     docker run -p 8000:8000 voxintel-arena          # open http://localhost:8000
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VOXINTEL_RISK_LOW=0.33 \
    VOXINTEL_RISK_HIGH=0.66

WORKDIR /app

# Versions pinned to the env the RF was pickled in, so joblib.load stays
# compatible across rebuilds. CPU-only — no torch.
RUN pip install --no-cache-dir \
    scikit-learn==1.6.1 numpy==2.1.3 pandas==2.2.3 joblib==1.4.2 scipy==1.15.3 \
    fastapi==0.115.0 "uvicorn==0.53.0" pydantic==2.9.2

# App code (incl. the arena static UI), the risk model, and the feature table.
COPY src/ ./src/
COPY models/voxintel_r_intent_rf.joblib ./models/voxintel_r_intent_rf.joblib
COPY reports/phase6_voxintel_r_slurp/voxintel_r_features.csv \
     ./reports/phase6_voxintel_r_slurp/voxintel_r_features.csv

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness: hit /health with the stdlib (no curl in slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

# --factory: app.py exposes create_app(), not a module-level `app`.
CMD ["uvicorn", "src.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
