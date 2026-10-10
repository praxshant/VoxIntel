# VoxIntel-R — portfolio & résumé

A reference-free reliability layer for a speech-to-intent (SLU) pipeline: it predicts,
at inference time and without a reference transcript, when an ASR-derived intent is
likely wrong, and turns that calibrated risk into an execute / clarify / abstain
decision. Research is frozen across 22 notebooks; a live reliability dashboard and a
reproducible, tested serving stack sit on top. All numbers below reproduce from
committed artifacts (`python src/analysis/frozen_split_eval.py`, seed 42).

## Résumé bullets (factual)

- Built **VoxIntel-R**, a reference-free reliability model for an ASR→intent cascade
  (Wav2Vec2-CTC + DistilBERT) on the SLURP benchmark: predicts downstream intent
  failure from inference-time uncertainty at **ROC-AUC 0.912** on a frozen held-out
  split, calibrated to **ECE 0.014** (isotonic), with a selective-prediction policy
  that beats confidence thresholding (**AURC 0.058 vs 0.083**).
- Established a rigorously-defended **negative result** — ASR-native uncertainty adds
  no signal beyond intent-native uncertainty (paired-bootstrap **ΔAUC −0.037**, 95% CI
  entirely negative, **0/20** repeated splits) — that held under a tuned six-model
  bake-off and richer N-best + neural-LM features, and replicated in direction on a
  second corpus (Fluent Speech Commands).
- Shipped the model as a **FastAPI** reliability service and a **reliability dashboard**
  (live held-out metrics + risk–coverage curve vs the MSP baseline, plus a real-time
  command inspector: ASR→intent→risk→execute/clarify/abstain), **Dockerized** (CPU-only,
  non-root, healthcheck) with **GitHub Actions CI** and unit + end-to-end tests covering
  success, uncertain, invalid, and failure paths.
- Engineered a **reproducible, leakage-audited** pipeline (22 notebooks, frozen seed-42
  split, dependency-free self-checks) whose headline numbers regenerate from cached
  features with no GPU; caught, fixed, and archived a taxonomy pipeline bug as a
  transparency artifact rather than hiding it.
- **Stress-tested the negative across ASR architectures** by swapping Wav2Vec2-CTC
  for a zero-shot **Whisper** front-end (new `retraining_on_whisper.ipynb`; **full
  8,688-clip** GPU decode; same intent model, protocol, and frozen split): H1 is **NOT
  SUPPORTED on Whisper too** — ΔAUC(C−B) **−0.020, 95% CI [−0.033, −0.007], 2/20
  splits** — the **same direction as Wav2Vec2** (−0.037), so the result **generalizes
  across CTC and seq2seq ASR**. A 2,500-clip pilot had looked borderline-positive
  (+0.022); the full run resolved it to negative — reported as-is, not stopped at the
  favorable subset.

## One-paragraph portfolio description

VoxIntel-R asks whether a voice assistant can know when not to trust its own
ASR-derived intent. On SLURP, the intent model's own uncertainty predicts downstream
failure well (ROC-AUC 0.912) and calibrates cleanly, while ASR-native confidence adds
nothing on top — a clean, reproducible negative result confirmed across models, a
richer feature family, and a second dataset. The calibrated risk drives a selective
execute/clarify/abstain policy, exposed through a FastAPI endpoint and a reliability
dashboard (held-out metrics, risk–coverage curve, command inspector), with Docker, CI,
and end-to-end tests. The project is
deliberately scoped to one ASR/NLU family and is honest about its limits (simulated
action-severity for the cost layer; single-pair generality is the open question).

## 60-second demo

    python src/analysis/frozen_split_eval.py
    python -m uvicorn src.serving.app:create_app --factory --port 8000   # http://localhost:8000

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker + free-tier hosting, and the
[README](README.md) / [reports/INDEX.md](reports/INDEX.md) for the full research trail.
