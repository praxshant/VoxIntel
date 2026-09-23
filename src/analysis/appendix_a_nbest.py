"""Appendix A — N-best / decoder uncertainty features for H1 (confirmatory).

H1 (does ASR uncertainty add signal beyond intent uncertainty?) was NOT SUPPORTED
with 9 CTC frame-level ASR features. A reviewer may ask whether *richer* ASR
uncertainty — from an N-best beam list — changes it. This computes the LM-free
subset of the Appendix-A family A' and re-runs the H1 test on the frozen split.

Two phases:
  decode : GPU beam search (pyctcdecode, no LM) over the 8,688 SLURP-dev clips
           -> 4 reference-free N-best features per clip -> appendix_a_nbest_features.csv
  eval   : refit B (intent-only) vs B u A' on the SAME frozen 70/30 seed-42 split,
           paired-bootstrap dAUC + 20-split stability -> appendix_a_nbest_h1.json

The LM half of family A (LM score, acoustic-LM disagreement) is specified with a
KenLM n-gram, which has no Windows wheel. We compute it with a neural LM
(distilgpt2) as a documented substitute — a stronger fluency model than a 4-gram,
scoring the same N-best texts — via two more phases:
  lm-decode : re-run beam search, score each hypothesis' text with distilgpt2
              -> 2 LM features per clip -> appendix_a_lm_features.csv
  lm-eval   : refit B vs B u A (all 6 features) on the frozen split -> appendix_a_full_h1.json

Run (in the CUDA env):
  conda run -n torch26 python src/analysis/appendix_a_nbest.py            # decode all + eval (LM-free A')
  conda run -n torch26 python src/analysis/appendix_a_nbest.py --limit 20 # smoke
  conda run -n torch26 python src/analysis/appendix_a_nbest.py --eval-only
  conda run -n torch26 python src/analysis/appendix_a_nbest.py --lm-decode # + LM features
  conda run -n torch26 python src/analysis/appendix_a_nbest.py --lm-eval   # full 6-feature H1
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

FEATURES = ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"
OUT_DIR = ROOT / "reports" / "phase8_hypothesis_validation"
NBEST_CSV = OUT_DIR / "appendix_a_nbest_features.csv"
H1_JSON = OUT_DIR / "appendix_a_nbest_h1.json"
CKPT = str(ROOT / "models" / "wav2vec2_slurp_debug" / "checkpoint-41500")  # best_metric ckpt
APRIME = ["nbest_entropy", "nbest_top_gap", "mean_token_posterior", "nbest_lev_dispersion"]
LM_FEATURES = ["lm_score", "acoustic_lm_disagreement"]  # LM-based half of family A
BEAM_WIDTH, NBEST = 100, 10
LM_MODEL = "distilgpt2"        # KenLM has no Windows wheel; a neural LM is the substitute
LM_CSV = OUT_DIR / "appendix_a_lm_features.csv"
FULL_H1_JSON = OUT_DIR / "appendix_a_full_h1.json"


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if not la or not lb:
        return max(la, lb)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _nbest_features(beams, logp):
    """4 reference-free, inference-time N-best features from one clip's beams."""
    scores = np.array([b[-1] for b in beams[:NBEST]], float)  # seq log-likelihoods
    q = np.exp(scores - scores.max()); q = q / q.sum()
    entropy = float(-(q * np.log(q + 1e-12)).sum()) if len(q) > 1 else 0.0
    gap = float(q[0] - (q[1] if len(q) > 1 else 0.0))          # 1.0 if a single hyp
    mean_tok = float(np.mean(np.exp(logp.max(axis=1))))        # mean top-1 frame posterior
    uniq = list(dict.fromkeys(b[0] for b in beams[:NBEST]))
    disp = float(np.mean([_lev(a, c) / max(len(a), len(c), 1)
                          for i, a in enumerate(uniq) for c in uniq[i + 1:]])) if len(uniq) > 1 else 0.0
    return {"nbest_entropy": entropy, "nbest_top_gap": gap,
            "mean_token_posterior": mean_tok, "nbest_lev_dispersion": disp}


def _disagreement(acoustic, lm):
    """1 - Spearman rank corr between the acoustic and LM scores over the N-best.
    0 = LM ranks the hypotheses like acoustics; ~2 = fully reversed; 0 if <2 hyps."""
    from scipy.stats import spearmanr
    if len(acoustic) < 2:
        return 0.0
    rho = spearmanr(acoustic, lm).correlation
    return 0.0 if not np.isfinite(rho) else float(1.0 - rho)


def _gpt2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), device


def _lm_scores(texts, tok, model, device):
    """Per-token mean log-prob of each hypothesis under the neural LM (higher =
    more fluent; a garbled ASR hypothesis scores lower). Empty/1-token -> 0."""
    import torch
    out = []
    for t in texts:
        ids = tok((t or "").lower(), return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:
            out.append(0.0); continue
        with torch.no_grad():
            out.append(float(-model(ids, labels=ids).loss))
    return np.array(out, float)


def _lm_features(beams, tok, model, device):
    """2 LM-based family-A features from one clip's N-best: LM score of the
    acoustic-best hypothesis, and acoustic-vs-LM rank disagreement over the list."""
    texts = [b[0] for b in beams[:NBEST]]
    acoustic = np.array([b[-1] for b in beams[:NBEST]], float)   # LM-free beam score
    lm = _lm_scores(texts, tok, model, device)
    return {"lm_score": float(lm[0]) if len(lm) else 0.0,
            "acoustic_lm_disagreement": _disagreement(acoustic, lm)}


def _load_decoder():
    """ASR model + a pyctcdecode decoder built from its tokenizer vocab (no LM)."""
    from pyctcdecode import build_ctcdecoder
    from src.asr.inference import load_asr_model
    processor, model, device = load_asr_model(CKPT)
    vocab = processor.tokenizer.get_vocab()
    labels = [None] * len(vocab)
    for tok, idx in vocab.items():
        labels[idx] = tok
    pad, wd = processor.tokenizer.pad_token, processor.tokenizer.word_delimiter_token
    decoder = build_ctcdecoder(["" if t == pad else (" " if t == wd else t) for t in labels])
    return processor, model, device, decoder


def _decode_loop(out_csv, feat_fn, limit=None):
    """Resumable beam-search decode over the SLURP-dev clips; feat_fn(beams, logp)
    -> a feature dict per clip, appended to out_csv. Shared by decode / decode_lm."""
    import torch
    from src.utils.audio import load_audio
    df = pd.read_csv(FEATURES)
    paths = df["audio_path"].tolist()[: limit or len(df)]
    done = set(pd.read_csv(out_csv)["audio_path"]) if out_csv.exists() else set()
    todo = [p for p in paths if p not in done]
    print(f"decode: {len(todo)} clips ({len(done)} already done) -> {out_csv.name}")

    processor, model, device, decoder = _load_decoder()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buf, t0, header = [], time.time(), not out_csv.exists()
    for k, p in enumerate(todo, 1):
        try:
            wav, sr = load_audio(p, target_sr=16_000)
            inp = processor(wav, sampling_rate=sr, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                logits = model(inp).logits[0]
            logp = torch.log_softmax(logits, dim=-1).cpu().numpy().astype("float32")
            beams = decoder.decode_beams(logp, beam_width=BEAM_WIDTH)
            row = {"audio_path": p, **feat_fn(beams, logp)}
        except Exception as e:
            print(f"  skip {p}: {e}"); continue
        buf.append(row)
        if len(buf) >= 200 or k == len(todo):
            pd.DataFrame(buf).to_csv(out_csv, mode="a", header=header, index=False)
            header, buf = False, []
            print(f"  {k}/{len(todo)}  {(time.time()-t0)/k*1000:.0f} ms/clip")
    print(f"{out_csv.name} done.")


def decode(limit=None):
    """N-best (LM-free A') features -> appendix_a_nbest_features.csv."""
    _decode_loop(NBEST_CSV, _nbest_features, limit)


def decode_lm(limit=None):
    """LM-based family-A features (distilgpt2 substitute for KenLM) -> LM_CSV."""
    lt, lm, ldev = _gpt2()
    _decode_loop(LM_CSV, lambda beams, logp: _lm_features(beams, lt, lm, ldev), limit)


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def _fit_pair(tr, te, ytr, yte, extra=APRIME):
    """Return (p_B, p_{B u extra}) test-set risk scores from two RFs on the same split."""
    from sklearn.ensemble import RandomForestClassifier
    def rf(cols):
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1)
        m.fit(tr[cols], ytr); return m.predict_proba(te[cols])[:, 1]
    return rf(INTENT_FEATURES), rf(INTENT_FEATURES + extra)


def _h1_test(df, extra, out, meta):
    """Frozen-split dAUC(B u extra - B) + 20-split stability; write out, return res."""
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = frozen_split(df)
    p_b, p_ba = _fit_pair(Xtr, Xte, ytr, yte, extra)
    auc_b, auc_ba = _auc(yte, p_b), _auc(yte, p_ba)
    gap = paired_bootstrap_gap(yte.to_numpy(), p_ba, p_b, _auc)  # dAUC(B u extra - B)

    wins = 0  # 20-split stability (Appendix A rule: replicate in >= 11/20)
    for s in range(20):
        tr, te, ytr2, yte2 = train_test_split(df, df[TARGET].astype(int),
                                              test_size=0.30, stratify=df[TARGET], random_state=s)
        pb, pba = _fit_pair(tr, te, ytr2, yte2, extra)
        wins += _auc(yte2, pba) > _auc(yte2, pb)

    supported = bool(gap["ci95_low"] > 0 and wins >= 11)
    res = {"n": int(len(df)), **meta,
           "auc_B_intent_only": auc_b, "auc_B_union": auc_ba,
           "delta_auc_union_minus_B": gap, "stability_wins_of_20": int(wins),
           "H1_supported": supported,
           "decision_rule": "dAUC 95% CI > 0 on frozen split AND >= 11/20 splits"}
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("auc_B_intent_only", "auc_B_union",
          "stability_wins_of_20", "H1_supported")}, indent=2))
    print("gap:", gap, "->", out.name)
    return res


def evaluate():
    """H1 confirmatory: does B u A' (LM-free) beat intent-only B on the frozen split?"""
    df = pd.read_csv(FEATURES).merge(pd.read_csv(NBEST_CSV), on="audio_path", how="inner")
    if len(df) < 100:
        raise SystemExit(f"only {len(df)} rows merged — run decode first.")
    return _h1_test(df, APRIME, H1_JSON,
                    {"features_union": APRIME, "lm_features_omitted": LM_FEATURES,
                     "kenlm_available": False})


def evaluate_full():
    """H1 with the FULL family A (A' + LM features scored by distilgpt2)."""
    df = (pd.read_csv(FEATURES)
          .merge(pd.read_csv(NBEST_CSV), on="audio_path", how="inner")
          .merge(pd.read_csv(LM_CSV), on="audio_path", how="inner"))
    if len(df) < 100:
        raise SystemExit(f"only {len(df)} rows merged — run --lm-decode first.")
    return _h1_test(df, APRIME + LM_FEATURES, FULL_H1_JSON,
                    {"features_union": APRIME + LM_FEATURES, "lm_model": LM_MODEL,
                     "lm_note": "distilgpt2 substitute for KenLM (no Windows wheel)"})


def _selfcheck():
    """Feature-math invariants (no GPU/audio): the silent-bug surface is here."""
    assert _lev("abc", "abc") == 0 and _lev("abc", "abd") == 1 and _lev("", "abc") == 3
    logp = np.log(np.array([[0.9, 0.1], [0.8, 0.2]]))            # top-1 posteriors 0.9, 0.8
    one = _nbest_features([("HI", None, [], -1.0, -1.0)], logp)  # single hypothesis
    assert one["nbest_entropy"] == 0.0 and one["nbest_top_gap"] == 1.0 and one["nbest_lev_dispersion"] == 0.0
    assert abs(one["mean_token_posterior"] - 0.85) < 1e-9
    two = _nbest_features([("HI", None, [], -1.0, -1.0), ("HO", None, [], -1.0, -1.0)], logp)
    assert abs(two["nbest_entropy"] - np.log(2)) < 1e-9 and two["nbest_top_gap"] == 0.0
    assert two["nbest_lev_dispersion"] > 0.0                      # HI vs HO differ
    # LM disagreement: same ranking -> 0; reversed -> ~2; <2 hyps -> 0.
    assert _disagreement([3.0, 2.0, 1.0], [3.0, 2.0, 1.0]) == 0.0
    assert _disagreement([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) > 1.9
    assert _disagreement([1.0], [1.0]) == 0.0
    print("appendix_a_nbest selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--lm-decode", action="store_true", help="compute LM features (GPU)")
    ap.add_argument("--lm-eval", action="store_true", help="full 6-feature H1 test")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck(); sys.exit(0)
    if a.lm_decode:
        decode_lm(a.limit)
        if not a.limit:
            evaluate_full()
        sys.exit(0)
    if a.lm_eval:
        evaluate_full(); sys.exit(0)
    if not a.eval_only:
        decode(a.limit)
    if not a.decode_only and not a.limit:
        evaluate()
