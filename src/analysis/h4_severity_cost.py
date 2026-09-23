"""Appendix B harness — turns human severity annotations into the H4 verdict.

H4 (cost-sensitive deferral) is INCONCLUSIVE in the paper because SLURP/FSC
carry no action-severity labels, so every misfire is weighted equally. This
module is the machinery that makes H4 a real test once tiers are annotated:

  make_template()      -> write the intent-level annotation task (real SLURP
                          intents, blank tier columns) for 2-3 annotators.
  load_labels(path)    -> read a filled template (intent -> tier_final).
  run_h4(labels)       -> refit the frozen-split intent risk model and price
                          three deferral policies on the frozen TEST set,
                          returning the H4 verdict with paired-bootstrap CIs.

Reference-free: runs off reports/.../voxintel_r_features.csv, no audio/GPU, and
reuses the frozen split + bootstrap from frozen_split_eval / reliability_metrics
so H4 shares the H1/H3 risk model. It does NOT invent severities — Appendix B is
explicit that guessed tiers manufacture the ground truth H4 needs; the honest
status until the template is annotated (Cohen's/Fleiss' kappa >= 0.7) is
"labels not yet collected". demo() runs on SYNTHETIC/illustrative tiers purely
to verify the arithmetic, and asserts no verdict.

Run: python src/analysis/h4_severity_cost.py                 # self-check + write template
     python src/analysis/h4_severity_cost.py --illustrative  # rubric + robustness (NOT verdict)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from src.analysis.reliability_metrics import (paired_bootstrap_gap,
                                                  frozen_split, INTENT_FEATURES, SEED)
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.analysis.reliability_metrics import (paired_bootstrap_gap,
                                                  frozen_split, INTENT_FEATURES, SEED)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "reports" / "phase6_voxintel_r_slurp" / "voxintel_r_features.csv"
OUT_DIR = ROOT / "reports" / "phase8_hypothesis_validation"
TEMPLATE = OUT_DIR / "severity_labels_TEMPLATE.csv"

TIERS = ("benign", "moderate", "critical")
# Illustrative cost units (Appendix B: "optionally a continuous cost per tier").
# REAL costs come from the annotation exercise; these only let the harness run.
DEFAULT_COST = {"benign": 1.0, "moderate": 5.0, "critical": 25.0}
DEFER_COST = 2.0  # cost of abstaining / routing one utterance to a human
def _rf():  # same config as frozen_split_eval / notebook 16
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


def _risk_scores(df):
    """Refit the deployed intent-only RF on the frozen TRAIN split; return the
    test frame, its labels, test risk scores, and the train frame/labels (for
    tuning the confidence threshold off the test set)."""
    Xtr, Xte, ytr, yte = frozen_split(df)
    m = _rf(); m.fit(Xtr[INTENT_FEATURES], ytr)
    return Xte, yte, m.predict_proba(Xte[INTENT_FEATURES])[:, 1], Xtr, ytr


def _tune_conf_threshold(conf, fail, tier_cost, defer_cost):
    """Confidence threshold (defer iff conf < tau) minimising cost on train."""
    best_tau, best = float(conf.min()), np.inf
    for tau in np.quantile(conf, np.linspace(0, 1, 51)):
        cost = np.where(conf >= tau, fail * tier_cost, defer_cost).mean()
        if cost < best:
            best, best_tau = cost, float(tau)
    return best_tau


def _price_policies(Xte, yte, p_risk, Xtr, ytr, tier_map, cost, defer_cost):
    """Realized per-utterance cost of the three policies on the frozen TEST set:
    (a) always-execute, (b) confidence-threshold deferral, (c) Bayes risk deferral.
    Shared by run_h4 (adds a bootstrap CI) and robustness_sweep (point decision)."""
    tc_te = Xte["ground_truth_intent"].map(tier_map).map(cost).to_numpy(float)
    tc_tr = Xtr["ground_truth_intent"].map(tier_map).map(cost).to_numpy(float)
    fail_te, fail_tr = yte.to_numpy(float), ytr.to_numpy(float)
    conf_te, conf_tr = Xte["intent_confidence"].to_numpy(), Xtr["intent_confidence"].to_numpy()
    cost_a = fail_te * tc_te
    tau = _tune_conf_threshold(conf_tr, fail_tr, tc_tr, defer_cost)
    cost_b = np.where(conf_te >= tau, fail_te * tc_te, defer_cost)
    execute_c = p_risk * tc_te <= defer_cost          # execute only if worth the risk
    cost_c = np.where(execute_c, fail_te * tc_te, defer_cost)
    return cost_a, cost_b, cost_c, tau, float((~execute_c).mean())


def run_h4(labels, cost=None, defer_cost=DEFER_COST, n_boot=2000):
    """Price three deferral policies on the frozen TEST set given an
    intent->tier map. H4 supported iff risk-deferral is cheaper than BOTH
    always-execute and confidence-deferral with a bootstrap CI excluding 0."""
    cost = cost or DEFAULT_COST
    df = pd.read_csv(FEATURES)
    missing = set(df["ground_truth_intent"].unique()) - set(labels)
    if missing:
        raise ValueError(f"{len(missing)} intents unannotated, e.g. {sorted(missing)[:5]}")
    Xte, yte, p_risk, Xtr, ytr = _risk_scores(df)
    cost_a, cost_b, cost_c, tau, defer_rate = _price_policies(
        Xte, yte, p_risk, Xtr, ytr, labels, cost, defer_cost)

    fail_te = yte.to_numpy(float)
    tiers_te = Xte["ground_truth_intent"].map(labels)
    mean_ = lambda _y, s: float(np.mean(s))
    gap_ca = paired_bootstrap_gap(fail_te, cost_c, cost_a, mean_, n=n_boot)
    gap_cb = paired_bootstrap_gap(fail_te, cost_c, cost_b, mean_, n=n_boot)
    supported = (gap_ca["ci95_high"] < 0) and (gap_cb["ci95_high"] < 0)
    return {"n_test": int(len(fail_te)),
            "tier_counts": {t: int((tiers_te == t).sum()) for t in TIERS},
            "cost_always_execute": float(cost_a.sum()),
            "cost_confidence_defer": float(cost_b.sum()),
            "cost_risk_defer": float(cost_c.sum()),
            "conf_threshold": tau, "defer_rate_risk": defer_rate,
            "gap_risk_minus_always": gap_ca, "gap_risk_minus_confidence": gap_cb,
            "H4_supported": bool(supported), "cost_units": cost, "defer_cost": defer_cost}
def make_template(path=TEMPLATE):
    """Write the intent-level annotation task: real SLURP intents, blank tiers."""
    df = pd.read_csv(FEATURES)
    rows = [{"intent": intent, "scenario": grp["scenario"].mode().iloc[0],
             "n_utterances": len(grp), "example_transcript": grp["asr_transcript"].iloc[0],
             "tier_annotator_1": "", "tier_annotator_2": "", "tier_annotator_3": "",
             "tier_final": "", "notes": ""}
            for intent, grp in df.groupby("ground_truth_intent")]
    out = pd.DataFrame(rows).sort_values("n_utterances", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def illustrative_tier_map(intents):
    """HEURISTIC tiers from the intent name — a smoke-test substitute, NOT
    annotation. Appendix B forbids using guessed tiers as H4 ground truth."""
    crit = ("send", "post", "email", "pay", "order", "book", "delete", "remove",
            "unlock", "lock", "taxi", "ticket", "cleaning")
    mod = ("set", "add", "create", "update", "alarm", "calendar", "list", "reminder")
    m = {}
    for i in intents:
        s = i.lower()
        m[i] = "critical" if any(k in s for k in crit) else \
               "moderate" if any(k in s for k in mod) else "benign"
    return m


# --- Appendix B, short of human labels: a single-rater rubric pass + robustness ---
# The real H4 verdict needs a kappa>=0.7 multi-annotator pass we cannot fabricate.
# rubric_tier_map is ONE rater applying severity_annotation_rubric.md (verb, not
# domain: a query is benign, a send/pay/appliance is critical). AMBIGUOUS lists the
# intents the rubric leaves genuinely two-sided, with the alternate tier the sweep
# flips them to. This is an illustration + sensitivity analysis, NOT ground truth.
def rubric_tier_map(intents):
    """Single-rater tiers per severity_annotation_rubric.md (NOT the kappa>=0.7
    ground truth). Query=benign; send/post/order/ticket/taxi/appliance=critical;
    set/add/remove/volume/alarm/calendar/light-toggle=moderate; else benign."""
    crit = ("sendemail", "send", "social_post", "post", "taxi", "ticket", "order",
            "cleaning", "coffee", "wemo", "querycontact")  # to-a-person / pays / appliance
    mod = ("_set", "createoradd", "addcontact", "remove", "volume", "alarm",
           "calendar", "likeness", "dislikeness", "settings", "light")
    out = {}
    for i in intents:
        s = i.lower()
        if "query" in s and "querycontact" not in s:      # any query is read-only
            out[i] = "benign"
        elif any(k in s for k in crit):
            out[i] = "critical"
        elif s == "set" or any(k in s for k in mod):
            out[i] = "moderate"
        else:
            out[i] = "benign"
    return out


# intent -> alternate tier the rubric could also justify (money/appliance vs trivial)
AMBIGUOUS = {"email_querycontact": "benign", "iot_coffee": "moderate",
             "iot_wemo_on": "moderate", "iot_wemo_off": "moderate",
             "iot_cleaning": "moderate", "cleaning": "moderate",
             "iot_hue_lightchange": "benign", "iot_hue_lightdim": "benign",
             "iot_hue_lightoff": "benign", "iot_hue_lightup": "benign",
             "iot_hue_lighton": "benign", "hue_lightoff": "benign",
             "hue_lightup": "benign", "audio_volume_up": "benign",
             "audio_volume_down": "benign", "audio_volume_mute": "benign",
             "music_likeness": "benign", "music_dislikeness": "benign",
             "music_settings": "benign", "email_addcontact": "benign"}


def load_labels(path):
    """Read a filled template: intent -> tier_final. Validates the tier vocab."""
    t = pd.read_csv(path)
    lab = {str(i): str(v) for i, v in zip(t["intent"], t["tier_final"])}
    bad = {i: v for i, v in lab.items() if v not in TIERS}
    if bad:
        raise ValueError(f"unfilled/invalid tier_final for {len(bad)} intents, e.g. {list(bad)[:5]}")
    return lab


def robustness_sweep(n=1000, seed=SEED):
    """How robust is the H4 decision to the two things we can't pin down without
    human labels — the borderline tier assignments and the cost weights? Fits the
    risk model ONCE, then for n trials flips each AMBIGUOUS intent (p=0.5) and
    samples a cost vector (moderate~U[3,10], critical~U[10,50], defer~U[1,4]),
    recording whether risk-deferral beats BOTH baselines on realized total cost.
    Point decision only (no per-trial bootstrap) — breadth, not a single verdict."""
    rng = np.random.default_rng(seed)
    df = pd.read_csv(FEATURES)
    Xte, yte, p_risk, Xtr, ytr = _risk_scores(df)
    base = rubric_tier_map(df["ground_truth_intent"].unique())
    beats_a = beats_b = both = 0
    savings = []
    for _ in range(n):
        tmap = dict(base)
        for i, alt in AMBIGUOUS.items():
            if i in tmap and rng.random() < 0.5:
                tmap[i] = alt
        cost = {"benign": 1.0, "moderate": float(rng.uniform(3, 10)),
                "critical": float(rng.uniform(10, 50))}
        ca, cb, cc, _, _ = _price_policies(Xte, yte, p_risk, Xtr, ytr, tmap,
                                           cost, float(rng.uniform(1, 4)))
        A, B, C = ca.sum(), cb.sum(), cc.sum()
        beats_a += C < A; beats_b += C < B; both += (C < A) and (C < B)
        savings.append((min(A, B) - C) / min(A, B))
    savings = np.asarray(savings)
    return {"n_trials": n, "p_beats_always": beats_a / n,
            "p_beats_confidence": beats_b / n, "p_H4_holds_point": both / n,
            "savings_vs_best_baseline": {"mean": float(savings.mean()),
                "p05": float(np.percentile(savings, 5)),
                "p95": float(np.percentile(savings, 95))},
            "n_ambiguous_intents": len(AMBIGUOUS)}


def run_illustrative(out=OUT_DIR / "appendix_b_h4_illustrative.json", n_sweep=1000):
    """Single-rater rubric point estimate (full bootstrap) + robustness sweep.
    Writes the rubric labels (auditable) and the result. NOT the H4 verdict."""
    df = pd.read_csv(FEATURES)
    tmap = rubric_tier_map(df["ground_truth_intent"].unique())
    point = run_h4(tmap)                       # DEFAULT_COST, full 2000-boot CI
    sweep = robustness_sweep(n=n_sweep)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"intent": list(tmap), "tier_rubric": list(tmap.values()),
                  "ambiguous_alt": [AMBIGUOUS.get(i, "") for i in tmap]}
                 ).sort_values("intent").to_csv(
        OUT_DIR / "severity_labels_rubric_singlerater.csv", index=False)
    res = {"status": "ILLUSTRATIVE single-rater rubric pass + robustness sweep — "
           "NOT the kappa>=0.7 multi-annotator H4 verdict Appendix B requires",
           "rubric_point_estimate": point, "robustness": sweep}
    out.write_text(json.dumps(res, indent=2))
    print(f"rubric point: always {point['cost_always_execute']:.0f} | conf "
          f"{point['cost_confidence_defer']:.0f} | risk {point['cost_risk_defer']:.0f} "
          f"| H4(point,CI)={point['H4_supported']}")
    print(f"robustness over {n_sweep} tier/cost draws: risk beats both baselines in "
          f"{sweep['p_H4_holds_point']*100:.0f}% (vs-always {sweep['p_beats_always']*100:.0f}%, "
          f"vs-conf {sweep['p_beats_confidence']*100:.0f}%) -> {out.name}")
    return res


def demo():
    """Self-check. (1) synthetic: the Bayes deferral rule minimises expected
    cost elementwise, so its total can never exceed always-execute. (2) end-to-
    end: run_h4 executes on illustrative tiers and returns finite, non-negative
    costs — with NO verdict asserted, which needs real labels."""
    rng = np.random.default_rng(0)
    n = 5000
    p_fail = rng.random(n)
    tier_cost = rng.choice([1.0, 5.0, 25.0], n)
    expected_always = p_fail * tier_cost
    expected_bayes = np.minimum(p_fail * tier_cost, DEFER_COST)  # defer iff cheaper
    assert expected_bayes.sum() <= expected_always.sum() + 1e-9
    execute = p_fail * tier_cost <= DEFER_COST
    assert np.allclose(np.where(execute, expected_always, DEFER_COST), expected_bayes)

    if FEATURES.exists():
        df = pd.read_csv(FEATURES)
        res = run_h4(illustrative_tier_map(df["ground_truth_intent"].unique()), n_boot=300)
        for k in ("cost_always_execute", "cost_confidence_defer", "cost_risk_defer"):
            assert np.isfinite(res[k]) and res[k] >= 0
        assert 0.0 <= res["defer_rate_risk"] <= 1.0
        print(f"illustrative H4 (NOT the paper result): always {res['cost_always_execute']:.0f} | "
              f"conf {res['cost_confidence_defer']:.0f} | risk {res['cost_risk_defer']:.0f} | "
              f"defer-rate {res['defer_rate_risk']:.2f}")
    print("h4_severity_cost demo OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--illustrative", action="store_true",
                    help="run single-rater rubric + robustness sweep (NOT the H4 verdict)")
    a = ap.parse_args()
    if a.illustrative:
        run_illustrative()
        return
    out = make_template()
    print(f"wrote annotation template ({len(out)} intents) -> {TEMPLATE}")
    print("Next: 2-3 annotators fill tier_final in {benign,moderate,critical} to kappa>=0.7,")
    print("then  run_h4(load_labels(TEMPLATE))  for the real H4 verdict.")
    demo()


if __name__ == "__main__":
    main()


