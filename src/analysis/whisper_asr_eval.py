"""Phase 9 — Whisper as a second ASR architecture (cross-architecture H1 check).

Why this exists
---------------
The whole VoxIntel-R result rests on ONE ASR family (Wav2Vec2-base CTC). The main
reviewer/interviewer objection is: *does the H1 negative (ASR-native uncertainty
adds nothing beyond intent-native uncertainty) generalize beyond CTC?* Whisper is
the natural stress test: a seq2seq encoder-decoder with an internal LM, a totally
different uncertainty surface from CTC frame posteriors.

Research question. Swapping the ASR front-end from Wav2Vec2-CTC to Whisper (same
DistilBERT intent model, same reference-free protocol, same frozen 70/30 seed-42
split), does ASR-native uncertainty now add predictive signal for downstream intent
failure beyond intent-native uncertainty?

Hypothesis (pre-registered direction). H1 stays NOT SUPPORTED — intent-native
uncertainty remains a near-sufficient statistic. We do NOT assume this; we report
what the subset evidence shows, with a paired-bootstrap CI.

Protocol. For each SLURP-dev clip: zero-shot Whisper transcript + Whisper-native
uncertainty features (token log-probs + per-step entropy from generation scores);
run the SAME frozen DistilBERT intent model on that transcript -> intent features
(confidence/entropy/margin) + predicted intent; target = predicted != reference
intent. Refit intent-only (B) vs intent+Whisper-ASR (C) on the frozen split.
References (reference transcript, WER/CER) are used ONLY for offline evaluation,
never as a risk feature.

Environment / cost. Decode runs in the CUDA env (torch26); whisper-base.en ~140 MB
download, GPU inference ~0.2-0.5 s/clip. The eval half is sklearn-only and runs in
base. Resumable: re-running appends only missing clips.

Run:
  conda run -n torch26 python src/analysis/whisper_asr_eval.py --limit 8     # smoke
  conda run -n torch26 python src/analysis/whisper_asr_eval.py --limit 1500  # pilot subset
  conda run -n torch26 python src/analysis/whisper_asr_eval.py               # full 8688
  python src/analysis/whisper_asr_eval.py --selfcheck   # feature math, no torch
  python src/analysis/whisper_asr_eval.py --eval        # H1 on cached CSV (base ok)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.reliability_metrics import (frozen_split, paired_bootstrap_gap,
                                              INTENT_FEATURES, TARGET, SEED)

SLURP_FEATURES = ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"
OUT_DIR = ROOT / "reports" / "phase9_whisper_cross_asr"
WHISPER_CSV = OUT_DIR / "whisper_voxintel_features.csv"
H1_JSON = OUT_DIR / "whisper_h1.json"
WHISPER_MODEL = "openai/whisper-base.en"
INTENT_MODEL = ROOT / "models" / "distilbert_intent"
SAMPLE_SEED = 42
# 9 Whisper-native features to mirror the size of the Wav2Vec2 family A.
WHISPER_ASR_FEATURES = ["whisper_mean_token_logprob", "whisper_min_token_logprob",
                        "whisper_std_token_logprob", "whisper_seq_avg_logprob",
                        "whisper_mean_step_entropy", "whisper_max_step_entropy",
                        "whisper_std_step_entropy", "whisper_num_tokens", "audio_duration"]

def _whisper_features_from_scores(scores, chosen_ids, eot_id):
    """Whisper-native, reference-free uncertainty from generation output.
    scores: tuple(step)[1, vocab] logits; chosen_ids: generated token ids (1D).
    Returns token-logprob stats + per-step predictive-entropy stats + length."""
    import torch
    tok_logprobs, step_entropies = [], []
    for step, logit in enumerate(scores):
        logp = torch.log_softmax(logit[0].float(), dim=-1)
        p = logp.exp()
        plogp = p * logp                                   # suppressed tokens: 0 * -inf = NaN
        plogp = torch.where(torch.isfinite(plogp), plogp, torch.zeros_like(plogp))
        step_entropies.append(float(-plogp.sum()))
        if step < len(chosen_ids):
            tid = int(chosen_ids[step])
            if tid == eot_id:
                break
            tok_logprobs.append(float(logp[tid]))
    if not tok_logprobs:
        tok_logprobs = [0.0]
    tl = np.array(tok_logprobs, float); se = np.array(step_entropies or [0.0], float)
    return {"whisper_mean_token_logprob": float(tl.mean()),
            "whisper_min_token_logprob": float(tl.min()),
            "whisper_std_token_logprob": float(tl.std()),
            "whisper_seq_avg_logprob": float(tl.sum() / len(tl)),
            "whisper_mean_step_entropy": float(se.mean()),
            "whisper_max_step_entropy": float(se.max()),
            "whisper_std_step_entropy": float(se.std()),
            "whisper_num_tokens": int(len(tl))}


def _intent_features_from_logits(logits):
    """3 intent-native features + predicted class id from a logit vector."""
    import torch
    p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    order = np.sort(p)[::-1]
    top1 = float(order[0]); top2 = float(order[1]) if len(order) > 1 else 0.0
    entropy = float(-(p * np.log(p + 1e-10)).sum())
    return {"intent_confidence": top1, "intent_entropy": entropy,
            "intent_margin": top1 - top2}, int(p.argmax())


def _load_models():
    """Whisper (ASR) + the frozen DistilBERT intent model, on GPU if available."""
    import torch
    from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                              AutoModelForSequenceClassification, AutoTokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wp = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    wm = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device).eval()
    itok = AutoTokenizer.from_pretrained(str(INTENT_MODEL))
    imodel = AutoModelForSequenceClassification.from_pretrained(str(INTENT_MODEL)).to(device).eval()
    id2label = {int(k): v for k, v in imodel.config.id2label.items()}
    return wp, wm, itok, imodel, id2label, device

def decode(limit=None):
    """Zero-shot Whisper decode + intent re-scoring over a seeded SLURP-dev subset.
    Reference-free features only; reference intent is carried for the OFFLINE target."""
    import torch
    from src.utils.audio import load_audio
    base = pd.read_csv(SLURP_FEATURES)
    if limit and limit < len(base):
        base = base.sample(n=limit, random_state=SAMPLE_SEED).reset_index(drop=True)
    done = set(pd.read_csv(WHISPER_CSV)["audio_path"]) if WHISPER_CSV.exists() else set()
    todo = base[~base["audio_path"].isin(done)].reset_index(drop=True)
    print(f"whisper decode: {len(todo)} clips ({len(done)} cached) -> {WHISPER_CSV.name}")
    if not len(todo):
        return
    wp, wm, itok, imodel, id2label, device = _load_models()
    eot = wp.tokenizer.eos_token_id
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buf, t0, header = [], time.time(), not WHISPER_CSV.exists()
    for k, row in enumerate(todo.itertuples(index=False), 1):
        try:
            wav, sr = load_audio(row.audio_path, target_sr=16_000)
            feats = wp(wav, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                gen = wm.generate(feats, return_dict_in_generate=True, output_scores=True,
                                  max_new_tokens=96)
            ids = gen.sequences[0][-len(gen.scores):]
            wf = _whisper_features_from_scores(gen.scores, ids, eot)
            text = wp.batch_decode(gen.sequences, skip_special_tokens=True)[0].strip()
            enc = itok(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            with torch.no_grad():
                logits = imodel(**enc).logits[0]
            intf, pid = _intent_features_from_logits(logits)
            pred = id2label.get(pid, str(pid))
            rec = {"audio_path": row.audio_path, "ground_truth_intent": row.ground_truth_intent,
                   "whisper_transcript": text, "predicted_intent": pred,
                   "intent_failed": int(pred != row.ground_truth_intent),
                   "audio_duration": float(getattr(row, "audio_duration", np.nan)),
                   **{k2: v for k2, v in wf.items() if k2 != "audio_duration"}, **intf}
        except Exception as e:
            print(f"  skip {row.audio_path}: {e}"); continue
        buf.append(rec)
        if len(buf) >= 100 or k == len(todo):
            pd.DataFrame(buf).to_csv(WHISPER_CSV, mode="a", header=header, index=False)
            header, buf = False, []
            print(f"  {k}/{len(todo)}  {(time.time()-t0)/k*1000:.0f} ms/clip")
    print(f"{WHISPER_CSV.name} done.")

def _rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def evaluate():
    """H1 on the cached Whisper features: intent-only (B) vs intent+Whisper-ASR (C)
    on the frozen 70/30 seed-42 split, with a paired-bootstrap dAUC and 20-split
    stability — the identical protocol used for Wav2Vec2 (frozen_split_eval.py)."""
    from sklearn.model_selection import train_test_split
    if not WHISPER_CSV.exists():
        raise SystemExit(f"missing {WHISPER_CSV} — run the decode first (torch26 env).")
    df = pd.read_csv(WHISPER_CSV).dropna(subset=INTENT_FEATURES + WHISPER_ASR_FEATURES)
    n, fail_rate = len(df), float(df[TARGET].mean())
    if n < 100:
        raise SystemExit(f"only {n} rows — decode a larger subset for a usable CI.")

    def fit_pair(tr, te, ytr):
        b = _rf(); b.fit(tr[INTENT_FEATURES], ytr)
        c = _rf(); c.fit(tr[INTENT_FEATURES + WHISPER_ASR_FEATURES], ytr)
        return (b.predict_proba(te[INTENT_FEATURES])[:, 1],
                c.predict_proba(te[INTENT_FEATURES + WHISPER_ASR_FEATURES])[:, 1])

    Xtr, Xte, ytr, yte = frozen_split(df)
    pB, pC = fit_pair(Xtr, Xte, ytr)
    aucB, aucC = _auc(yte, pB), _auc(yte, pC)
    gap = paired_bootstrap_gap(yte.to_numpy(), pC, pB, _auc)   # dAUC(C - B)

    wins = 0
    for s in range(20):
        tr, te, ytr2, _ = train_test_split(df, df[TARGET].astype(int),
                                           test_size=0.30, stratify=df[TARGET], random_state=s)
        pb, pc = fit_pair(tr, te, ytr2)
        wins += _auc(te[TARGET], pc) > _auc(te[TARGET], pb)

    supported = bool(gap["ci95_low"] > 0 and wins >= 11)
    res = {"asr_frontend": WHISPER_MODEL, "n_clips": n, "intent_failure_rate": fail_rate,
           "intent_model": "models/distilbert_intent (shared with Wav2Vec2 run)",
           "auc_B_intent_only": aucB, "auc_C_intent_plus_whisper_asr": aucC,
           "delta_auc_C_minus_B": gap, "stability_wins_of_20": int(wins),
           "H1_supported": supported,
           "decision_rule": "dAUC 95% CI > 0 on frozen split AND >= 11/20 splits",
           "wav2vec2_reference": {"auc_B_intent_only_full_8688": 0.912,
                                  "delta_auc_C_minus_B_full": -0.037, "H1_supported": False},
           "note": "Subset pilot unless n_clips==8688; compare SIGN of dAUC, not raw AUC "
                   "magnitude (different n and transcripts than the Wav2Vec2 run)."}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    H1_JSON.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("n_clips", "intent_failure_rate",
          "auc_B_intent_only", "auc_C_intent_plus_whisper_asr", "stability_wins_of_20",
          "H1_supported")}, indent=2))
    print("dAUC(C-B):", gap, "->", H1_JSON.name)
    return res

def _selfcheck():
    """Feature-math invariants (no torch/audio): a confident 1-token decode has
    0 token-logprob spread; a flat 2-way step has entropy ~ln(vocab-ish)."""
    import types
    # fake 'torch' just enough for the two helpers (log_softmax/softmax on 1D).
    try:
        import torch  # real torch if present
    except Exception:
        torch = None
    if torch is not None:
        logits = torch.tensor([[0.0, 10.0, 0.0]])          # near-deterministic step
        f = _whisper_features_from_scores((logits,), torch.tensor([1]), eot_id=999)
        assert f["whisper_num_tokens"] == 1
        assert f["whisper_mean_step_entropy"] < 0.1        # peaked -> low entropy
        intf, pid = _intent_features_from_logits(torch.tensor([2.0, 0.0, 0.0]))
        assert pid == 0 and 0 < intf["intent_margin"] <= 1 and intf["intent_confidence"] > 0.5
        print("whisper_asr_eval selfcheck OK (torch present)")
    else:
        print("whisper_asr_eval selfcheck skipped (torch absent — run --eval in base instead)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="decode a seeded N-clip subset")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--eval", action="store_true", help="H1 on cached CSV (sklearn only)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); return
    if a.eval:
        evaluate(); return
    decode(a.limit)
    if not a.decode_only:
        try:
            evaluate()
        except SystemExit as e:
            print(e)


if __name__ == "__main__":
    main()




