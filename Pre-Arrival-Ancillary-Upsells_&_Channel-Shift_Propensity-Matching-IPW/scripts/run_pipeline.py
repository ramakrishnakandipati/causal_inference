"""End-to-end orchestrator: hardware -> data -> propensity -> cate -> evaluate -> score.

Usage:  python scripts/run_pipeline.py [--fast] [--stage data propensity ...]
Set CAUSAL_OUT=<dir> to redirect outputs (used by tests).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.common.config import load_config, out_dir
from src.common.hardware import get_device_report
from src.data import synthetic_generator as sg
from src.evaluation import metrics_and_plots as mp
from src.models import propensity_ate as pa
from src.models.cate_learners import make_learners
from src.models.deep_causal import _DeepCATE

STAGES = ["hardware", "data", "propensity", "cate", "evaluate", "score"]
OUTCOMES = {"ancillary": ("y_ancillary", "tau_anc", False), "direct": ("y_direct", "tau_direct", True)}
log = logging.getLogger("pipeline")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def setup_logging():
    fh = logging.FileHandler(out_dir("logs") / "pipeline.log", mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh], force=True,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def gpu_telemetry(tag):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
        log.info("GPU telemetry [%s]: util%%, mem used, mem total, temp = %s", tag, out)
    except Exception:
        log.info("GPU telemetry [%s]: nvidia-smi unavailable", tag)


def save_json(obj, name):
    with open(out_dir("metrics") / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


# ------------------------------------------------------------------ stages
def stage_hardware(fast):
    rep = get_device_report()
    log.info("Device report: %s", rep)
    save_json(rep, "device_report.json")
    gpu_telemetry("start")


def stage_data(fast):
    cfg = load_config()
    n = cfg["data"]["n_fast"] if fast else cfg["data"]["n_full"]
    df = sg.generate(n, cfg["seed"])
    tr, te = sg.split(df, cfg["data"]["test_frac"], cfg["seed"])
    tr.to_parquet(out_dir("data") / "train.parquet"); te.to_parquet(out_dir("data") / "test.parquet")
    log.info("Data: train=%d test=%d treated share=%.3f", len(tr), len(te), df.treatment.mean())
    save_json(dict(n=n, n_train=len(tr), n_test=len(te), treated_share=float(df.treatment.mean()),
                   true_ate_ancillary=float(df.tau_anc.mean()), true_ate_direct=float(df.tau_direct.mean()),
                   segment_share=df.segment.value_counts(normalize=True).to_dict(),
                   channel_share=df.channel.value_counts(normalize=True).to_dict()), "data_summary.json")


def _load():
    return pd.read_parquet(out_dir("data") / "train.parquet"), pd.read_parquet(out_dir("data") / "test.parquet")


def stage_propensity(fast):
    cfg = load_config()
    tr, te = _load()
    X, T = sg.build_features(tr), tr.treatment.values
    e, model, cal = pa.fit_propensity(X, T, fast, cfg["seed"])
    keep = pa.common_support(e, T)
    log.info("Propensity: range [%.3f, %.3f], common-support keeps %.1f%%", e.min(), e.max(), 100 * keep.mean())
    joblib.dump(dict(model=model, calibrator=cal, clip=cfg["propensity"]["clip"]), out_dir("models") / "propensity.joblib")
    # test-set propensity (full model + calibration)
    from src.models.propensity_ate import _logit
    raw = model.predict_proba(sg.build_features(te))[:, 1]
    e_te = np.clip(cal.predict_proba(_logit(raw).reshape(-1, 1))[:, 1], *cfg["propensity"]["clip"])
    pd.DataFrame({"e_hat": e_te}).to_parquet(out_dir("data") / "test_propensity.parquet")
    d, Xk, ek = tr[keep].reset_index(drop=True), X[keep].reset_index(drop=True), e[keep]
    res = pa.estimate_all(d, Xk, ek, fast, cfg["seed"])
    res["common_support_kept"] = float(keep.mean())
    res["propensity_auc_note"] = "cross-fitted calibrated XGBoost"
    bal = pa.balance_table(Xk, d.treatment.values, ek, cfg["propensity"]["weight_trim_q"], cfg["propensity"]["caliper_sd"])
    bal.to_csv(out_dir("metrics") / "balance_smd.csv")
    res["balance"] = dict(max_abs_smd_before=float(bal.smd_before.abs().max()), max_abs_smd_psm=float(bal.smd_psm.abs().max()),
                          max_abs_smd_ipw=float(bal.smd_ipw.abs().max()),
                          share_below_0p1_before=float((bal.smd_before.abs() < 0.1).mean()),
                          share_below_0p1_psm=float((bal.smd_psm.abs() < 0.1).mean()),
                          share_below_0p1_ipw=float((bal.smd_ipw.abs() < 0.1).mean()))
    save_json(res, "ate_comparison.json")
    pd.DataFrame({"e_hat": e, "treatment": T, "channel": tr.channel.values, "e_true": tr.e_true.values}).to_parquet(
        out_dir("data") / "train_propensity.parquet")
    mp.love_plot(bal); mp.overlap_plot(e, T, tr.channel.values); mp.ate_comparison_plot(res)
    ref = {o: pa.refute(d, Xk, col, fast, cfg["seed"]) for o, (col, _, _) in OUTCOMES.items()}
    save_json(ref, "refutation.json")
    log.info("Propensity stage done; refutation dowhy available=%s", ref["ancillary"]["dowhy"].get("available"))


def stage_cate(fast):
    cfg = load_config()
    tr, te = _load()
    X, Xt, T = sg.build_features(tr), sg.build_features(te), tr.treatment.values
    preds = pd.DataFrame(index=te.index)
    meta = {}
    for outcome, (col, tcol, binary) in OUTCOMES.items():
        learners = make_learners(binary, fast, cfg["seed"])
        for name, m in learners.items():
            try:
                m.fit(X, T, tr[col].values)
                p = m.predict_cate(Xt)
                m.time_inference(Xt)
                preds[f"{outcome}|{name}"] = p
                if m.name.startswith("Causal Forest"):
                    lo, hi = m.predict_interval(Xt)
                    preds[f"{outcome}|{name}|lo"], preds[f"{outcome}|{name}|hi"] = lo, hi
                meta[f"{outcome}|{name}"] = dict(fit_seconds=m.fit_seconds, latency_ms_per_1k=m.latency_ms_per_1k, family=m.family)
                path = out_dir("models") / f"{outcome}__{slug(name)}"
                if isinstance(m, _DeepCATE):
                    m.save(path.with_suffix(".pt"))
                else:
                    m.save(path.with_suffix(".joblib"))
                log.info("[%s] %-28s fit %.1fs  latency %.1f ms/1k", outcome, name, m.fit_seconds, m.latency_ms_per_1k)
            except Exception as ex:  # keep pipeline alive; record failure
                log.exception("Learner %s failed: %s", name, ex)
                meta[f"{outcome}|{name}"] = dict(error=str(ex)[:200])
        gpu_telemetry(f"after cate {outcome}")
    preds.to_parquet(out_dir("data") / "test_preds.parquet")
    save_json(meta, "learner_meta.json")


def stage_evaluate(fast):
    tr, te = _load()
    P = pd.read_parquet(out_dir("data") / "test_preds.parquet")
    e_te = pd.read_parquet(out_dir("data") / "test_propensity.parquet")["e_hat"].values
    meta = json.load(open(out_dir("metrics") / "learner_meta.json"))
    lbs, cvs, deciles = [], [], []
    for outcome, (col, tcol, binary) in OUTCOMES.items():
        pre = f"{outcome}|"
        preds = {c[len(pre):]: P[c].values for c in P.columns if c.startswith(pre) and not c.endswith(("|lo", "|hi"))}
        m = {k[len(pre):]: v for k, v in meta.items() if k.startswith(pre) and "error" not in v}
        lb, cv = mp.leaderboard(preds, te, e_te, outcome, m)
        lbs.append(lb); cvs.append(cv.assign(outcome=outcome))
        real = lb[~lb.model.str.startswith("Oracle")]
        best = real.iloc[0].model
        dec = mp.decile_uplift(preds[best], te.treatment.values, te[col].values.astype(float), e_te).assign(outcome=outcome, model=best)
        deciles.append(dec)
        mp.qini_plot(cv, outcome); mp.policy_plot(cv, outcome); mp.decile_plot(dec, best, outcome)
        top3 = list(real.sort_values("pehe").model[:3])
        mp.cate_dist_plot(preds, te[tcol].values, outcome, top3)
    pd.concat(lbs).to_csv(out_dir("metrics") / "cate_leaderboard.csv", index=False)
    pd.concat(cvs).to_csv(out_dir("metrics") / "uplift_curves.csv", index=False)
    pd.concat(deciles).to_csv(out_dir("metrics") / "uplift_deciles.csv", index=False)
    log.info("Leaderboards written")


def stage_score(fast):
    """scored_guests.parquet: test guests with every model's CATE, best-model scores, combined value, budget selection."""
    cfg = load_config(); b = cfg["business"]
    tr, te = _load()
    P = pd.read_parquet(out_dir("data") / "test_preds.parquet")
    lb = pd.read_csv(out_dir("metrics") / "cate_leaderboard.csv")
    sc = te.copy()
    for c in P.columns:
        o, n, *rest = c.split("|")
        sc[f"cate_{o}__{slug(n)}" + (f"__{rest[0]}" if rest else "")] = P[c].values
    best = {}
    for o in OUTCOMES:
        sub = lb[(lb.outcome == o) & ~lb.model.str.startswith("Oracle")].sort_values("qini", ascending=False)
        best[o] = sub.iloc[0].model
        sc[f"tau_hat_{o}"] = P[f"{o}|{best[o]}"].values
    sc["incremental_value_hat"] = b["ancillary_margin"] * sc["tau_hat_ancillary"] + b["direct_booking_margin"] * sc["tau_hat_direct"] - b["cost_per_offer"]
    thr = np.quantile(sc["incremental_value_hat"], 1 - b["budget_frac"])
    sc["target_flag"] = (sc["incremental_value_hat"] >= thr).astype(int)
    sc["incremental_value_true"] = b["ancillary_margin"] * sc.tau_anc + b["direct_booking_margin"] * sc.tau_direct - b["cost_per_offer"]
    sc.to_parquet(out_dir("data") / "scored_guests.parquet")
    save_json(dict(best_model=best, budget_frac=b["budget_frac"],
                   est_value_targeted=float(sc.loc[sc.target_flag == 1, "incremental_value_hat"].sum()),
                   true_value_targeted=float(sc.loc[sc.target_flag == 1, "incremental_value_true"].sum()),
                   true_value_all=float(sc["incremental_value_true"].sum()),
                   true_value_oracle_top=float(np.sort(sc["incremental_value_true"].values)[::-1][: int(b["budget_frac"] * len(sc))].sum())),
              "campaign_summary.json")
    gpu_telemetry("end")


FUNCS = dict(hardware=stage_hardware, data=stage_data, propensity=stage_propensity, cate=stage_cate,
             evaluate=stage_evaluate, score=stage_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--stage", nargs="*", default=STAGES, choices=STAGES)
    a = ap.parse_args()
    setup_logging()
    for s in STAGES:
        if s in a.stage:
            t0 = time.time()
            log.info("=== stage %s (fast=%s) ===", s, a.fast)
            FUNCS[s](a.fast)
            log.info("=== stage %s done in %.1fs ===", s, time.time() - t0)


if __name__ == "__main__":
    main()
