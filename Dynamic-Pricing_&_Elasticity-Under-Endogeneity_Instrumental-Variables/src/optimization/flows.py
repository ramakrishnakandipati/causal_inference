"""End-to-end decision flows with metrics at every stage and overall.

Four flows, each run on several worked examples:
  1. Elasticity identification : context -> instrument relevance -> naive estimate -> IV estimate -> validation
  2. Price optimisation        : elasticity -> demand at rack -> ADR sweep -> prescription -> realised RevPAR
  3. Promo CATE decision       : baseline -> CATE ensemble -> conversion uplift -> economics -> offer decision
  4. Batch CRM targeting       : ingest -> score -> segment -> value -> targeting quality

Every "truth" number comes from the synthetic data-generating process, so each stage can be checked.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.optimization.revenue_optimizer import (build_profile_df, compare_policies, crm_segments, rack_rate,
                                                status_quo_adr, truth_log_demand, truth_occupancy)
from src.utils.registry import load_bookings, load_json
from src.utils.structural import expected_occupancy

CATE_MODELS = ["S-Learner", "T-Learner", "X-Learner", "R-Learner (NonParamDML)", "DR-Learner", "Causal Forest DML",
               "Uplift RF (causalml)", "TARNet", "DragonNet", "CEVAE", "DeepIV (PyTorch)", "DR-IV (LinearDRIV)"]
IV_MODELS = ["2SLS", "DML-IV", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)"]

FLOW_META = {
    "elasticity": ("Flow 1 · Elasticity identification",
                   "Can we recover the true price elasticity despite the hidden demand shock?"),
    "pricing": ("Flow 2 · Price optimisation",
                "Turn the elasticity into an ADR and check the realised RevPAR against the truth."),
    "promo": ("Flow 3 · Promo CATE decision",
              "Should this guest get a 15% promotion? Ensemble of 12 causal models vs the truth."),
    "crm": ("Flow 4 · Batch CRM targeting",
            "Score a reservation batch, split guests into uplift quadrants and value the targeting."),
}


def _usd(v: float) -> str:
    return f"${v:,.0f}"


def _scn(cfg: Dict[str, Any], key: str, alt: bool = False) -> Dict[str, Any]:
    s = cfg["scenarios"][key]
    return {"profile": s["profile_alt" if alt else "profile"], "instruments": s["instruments"], "u": s["demand_shock_u"]}


def examples(cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    A, Bl, Bc, C = _scn(cfg, "A"), _scn(cfg, "B"), _scn(cfg, "B", True), _scn(cfg, "C")
    return {
        "elasticity": [{"label": "A · Convention surge (group, urban)", **A},
                       {"label": "B1 · Leisure early-bird (resort)", **Bl},
                       {"label": "B2 · Corporate traveller (resort)", **Bc}],
        "pricing": [{"label": "A · Convention surge", **A},
                    {"label": "B1 · Leisure shoulder season", **Bl},
                    {"label": "C · Airline -25% seats", **C}],
        "promo": [{"label": "B1 · Leisure early-bird", **Bl},
                  {"label": "B2 · Corporate traveller", **Bc},
                  {"label": "C · Leisure, airline cut", **C}],
        "crm": [{"label": "Batch 1 · 500 mixed reservations · Causal Forest DML", "model": "Causal Forest DML", "n": 500, "seed": 7, "segment": None},
                {"label": "Batch 2 · 400 leisure reservations · DragonNet", "model": "DragonNet", "n": 400, "seed": 11, "segment": "leisure"}],
    }


def _stage(name: str, detail: str, metrics: List[tuple]) -> Dict[str, Any]:
    return {"name": name, "detail": detail, "metrics": [{"label": a, "value": b} for a, b in metrics]}


# ------------------------------------------------------------------------------------------------ flow 1
def elasticity_flow(models, cfg, ex) -> Dict[str, Any]:
    row = build_profile_df(ex["profile"], ex["instruments"], cfg=cfg)
    diag = load_json("iv_first_stage_diagnostics.json")
    truth = float(row["true_elasticity"].iloc[0])
    ref = float(np.exp(row["ln_adr"].iloc[0]))
    z_only = status_quo_adr(row, cfg, 0.0)
    with_u = status_quo_adr(row, cfg, ex["u"])
    est = {n: float(models[n].elasticity(row)[0]) for n in ["Naive OLS", "Naive XGBoost"] + IV_MODELS if n in models}
    iv_cons = float(np.median([est[n] for n in IV_MODELS if n in est]))
    err_naive, err_iv = abs(est["Naive OLS"] - truth), abs(iv_cons - truth)
    red = (1 - err_iv / max(err_naive, 1e-9)) * 100
    fs = diag.get("first_stage", {})
    stages = [
        _stage("1 · Context", "Guest profile and market state",
               [("Segment", f"{ex['profile']['segment']} / {ex['profile']['property_tier']}"), ("Reference ADR", _usd(ref)),
                ("Hidden shock U", f"{ex['u']:+.1f}")]),
        _stage("2 · Instrument relevance", "Z moves the rate without moving demand",
               [("First-stage F", f"{fs.get('f_statistic_robust', float('nan')):,.0f}"),
                ("Partial R² of Z", f"{fs.get('partial_r2', float('nan')):.2f}"),
                ("Rate shift from Z alone", f"{(z_only / ref - 1) * 100:+.0f}%")]),
        _stage("3 · Naive estimate", "Ignores the hidden shock",
               [("OLS elasticity", f"{est['Naive OLS']:+.2f}"), ("XGBoost elasticity", f"{est['Naive XGBoost']:+.2f}"),
                ("Rate shift incl. U", f"{(with_u / ref - 1) * 100:+.0f}%")]),
        _stage("4 · IV estimate", "2SLS, DML-IV, DR-IV, DeepIV",
               [("2SLS", f"{est['2SLS']:+.2f}"), ("DeepIV", f"{est['DeepIV (PyTorch)']:+.2f}"),
                ("IV median", f"{iv_cons:+.2f}")]),
        _stage("5 · Validation", "Against the ground truth",
               [("True elasticity", f"{truth:+.2f}"), ("Naive |error|", f"{err_naive:.2f}"), ("IV |error|", f"{err_iv:.2f}")]),
    ]
    return {"label": ex["label"], "stages": stages,
            "overall": {"bias_reduction_pct": red, "naive_error": err_naive, "iv_error": err_iv,
                        "sign_correct": bool(iv_cons < 0)},
            "headline": f"Naive error {err_naive:.2f} → IV error {err_iv:.2f} ({red:.0f}% of the bias removed)"}


# ------------------------------------------------------------------------------------------------ flow 2
def pricing_flow(models, cfg, ex) -> Dict[str, Any]:
    keys = ["Naive OLS", "2SLS", "DML-IV", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)"]
    row = build_profile_df(ex["profile"], ex["instruments"], cfg=cfg)
    res = compare_policies({k: models[k] for k in keys}, row, cfg, ex["u"])
    pm = res["per_model"]
    nv, dp = pm["Naive OLS"], pm["DeepIV (PyTorch)"]
    rack = res["rack_rate"]
    eps = cfg["dgp"]["eps_sd"]
    occ = {k: float(expected_occupancy(models[k].predict_log_demand(row, np.log([rack])), eps)[0]) for k in ("Naive OLS", "DeepIV (PyTorch)")}
    occ_t = float(truth_occupancy(row, cfg, np.array([rack]), None)[0])
    truth_best = res["truth_optimal_net_revpar"]
    lift = res.get("revpar_lift_pct_causal_vs_naive") or 0.0
    capture = dp["realised_net_revpar"] / max(truth_best, 1e-9) * 100
    n_grid = cfg["data"]["price_grid_points"]
    stages = [
        _stage("1 · Elasticity", "From the fitted estimators",
               [("Naive OLS", f"{nv['estimated_elasticity']:+.2f}"), ("DeepIV", f"{dp['estimated_elasticity']:+.2f}"),
                ("Truth", f"{res['true_elasticity']:+.2f}")]),
        _stage("2 · Demand at rack", f"Conversion at {_usd(rack)}",
               [("Naive OLS", f"{occ['Naive OLS']:.0%}"), ("DeepIV", f"{occ['DeepIV (PyTorch)']:.0%}"), ("Truth", f"{occ_t:.0%}")]),
        _stage("3 · ADR sweep", f"{n_grid} ADRs from $95 to $650",
               [("Naive OLS optimum", _usd(nv["optimal_adr"])), ("DeepIV optimum", _usd(dp["optimal_adr"])),
                ("True optimum", _usd(res["truth_optimal_adr"]))]),
        _stage("4 · Prescription", "Rack ADR plus promo depth",
               [("Offered ADR", _usd(dp["optimal_adr"])), ("Promo depth", f"{dp['optimal_promo_pct']:.0f}%"),
                ("Status-quo ADR", _usd(res["status_quo_adr"]))]),
        _stage("5 · Realised RevPAR", "On the true demand, averaged over U",
               [("Naive policy", _usd(nv["realised_net_revpar"])), ("DeepIV policy", _usd(dp["realised_net_revpar"])),
                ("Best possible", _usd(truth_best))]),
    ]
    return {"label": ex["label"], "stages": stages,
            "overall": {"lift_vs_naive_pct": lift, "lift_vs_status_quo_pct": res.get("revpar_lift_pct_causal_vs_status_quo") or 0.0,
                        "capture_of_best_pct": capture, "adr_gap_vs_truth": abs(dp["optimal_adr"] - res["truth_optimal_adr"])},
            "headline": f"{lift:+.0f}% RevPAR vs naive; captures {capture:.0f}% of the best achievable"}


# ------------------------------------------------------------------------------------------------ flow 3
def promo_flow(models, cfg, ex) -> Dict[str, Any]:
    row = build_profile_df(ex["profile"], ex["instruments"], cfg=cfg)
    rack = float(rack_rate(row, cfg)[0])
    row["ln_adr"], row["adr"] = np.log(rack), rack
    eps, d, unit = cfg["dgp"]["eps_sd"], cfg["dgp"]["promo"]["discount"], cfg["optimization"]["unit_cost_per_room"]
    names = [n for n in CATE_MODELS if n in models]
    taus = np.array([float(models[n].promo_cate(row)[0]) for n in names])
    cons, spread = float(np.median(taus)), float(np.std(taus))
    truth_tau = float(row["true_elasticity"].iloc[0] * np.log(1 - d))
    base = models["DeepIV (PyTorch)"].predict_log_demand(row, np.log([rack]))
    o0, o1 = float(expected_occupancy(base, eps)[0]), float(expected_occupancy(base + cons, eps)[0])
    t0 = float(truth_occupancy(row, cfg, np.array([rack]), None)[0])
    t1 = float(truth_occupancy(row, cfg, np.array([rack * (1 - d)]), None)[0])
    nv_hat = o1 * (rack * (1 - d) - unit) - o0 * (rack - unit)
    nv_true = t1 * (rack * (1 - d) - unit) - t0 * (rack - unit)
    offer_hat, offer_true = nv_hat > 0, nv_true > 0
    stages = [
        _stage("1 · Baseline", f"Booking probability at rack {_usd(rack)}",
               [("Model baseline", f"{o0:.0%}"), ("Truth baseline", f"{t0:.0%}"), ("Segment", ex["profile"]["segment"])]),
        _stage("2 · CATE ensemble", f"{len(names)} causal models, effect on log demand",
               [("Median CATE", f"{cons:+.3f}"), ("Spread across models", f"±{spread:.3f}"), ("True CATE", f"{truth_tau:+.3f}")]),
        _stage("3 · Conversion uplift", f"A {d:.0%} discount",
               [("Estimated uplift", f"{(o1 - o0) * 100:+.1f} pp"), ("True uplift", f"{(t1 - t0) * 100:+.1f} pp"),
                ("CATE |error|", f"{abs(cons - truth_tau):.3f}")]),
        _stage("4 · Economics", "Net RevPAR change after the discount",
               [("Estimated", f"{nv_hat:+,.1f}"), ("True", f"{nv_true:+,.1f}"), ("Unit cost", _usd(unit))]),
        _stage("5 · Decision", "Offer the promo?",
               [("Model says", "Offer" if offer_hat else "Do not offer"), ("Truth says", "Offer" if offer_true else "Do not offer"),
                ("Correct", "Yes" if offer_hat == offer_true else "No")]),
    ]
    return {"label": ex["label"], "stages": stages,
            "overall": {"decision_correct": bool(offer_hat == offer_true), "cate_abs_error": abs(cons - truth_tau),
                        "value_error_usd": abs(nv_hat - nv_true), "model_spread": spread},
            "headline": f"{'Offer' if offer_hat else 'Withhold'} the promo ({'matches' if offer_hat == offer_true else 'misses'} the truth); CATE error {abs(cons - truth_tau):.3f}"}


# ------------------------------------------------------------------------------------------------ flow 4
def crm_flow(models, cfg, ex) -> Dict[str, Any]:
    df = load_bookings()
    if ex.get("segment"):
        df = df[df["segment"] == ex["segment"]]
    batch = df.sample(ex["n"], random_state=ex["seed"]).reset_index(drop=True)
    scored = crm_segments(batch, models[ex["model"]], cfg)
    d, unit, eps = cfg["dgp"]["promo"]["discount"], cfg["optimization"]["unit_cost_per_room"], cfg["dgp"]["eps_sd"]
    price = batch["adr"].to_numpy(float)
    ld0 = truth_log_demand(batch, cfg, price, batch["demand_shock_u"].to_numpy(float))
    o0 = expected_occupancy(ld0, eps)
    o1 = expected_occupancy(ld0 + batch["true_promo_cate"].to_numpy(float), eps)
    true_nv = o1 * (price * (1 - d) - unit) - o0 * (price - unit)
    seg = scored["uplift_segment"].to_numpy()
    pers = seg == "Persuadables"
    targeted, blanket, oracle = float(true_nv[pers].sum()), float(true_nv.sum()), float(true_nv[true_nv > 0].sum())
    precision = float((true_nv[pers] > 0).mean()) if pers.any() else float("nan")
    capture = targeted / oracle * 100 if oracle > 0 else float("nan")
    rank = float(pd.Series(scored["net_value_usd"]).corr(pd.Series(true_nv), method="spearman"))
    counts = {s: int((seg == s).sum()) for s in ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]}
    stages = [
        _stage("1 · Ingest", "Reservation batch",
               [("Reservations", f"{len(batch):,}"), ("Mean offered ADR", _usd(price.mean())), ("Model", {"Causal Forest DML": "Forest DML"}.get(ex["model"], ex["model"]))]),
        _stage("2 · Score", "Heterogeneous CATE per guest",
               [("Mean CATE", f"{scored['cate_log_demand'].mean():+.3f}"), ("Mean true CATE", f"{batch['true_promo_cate'].mean():+.3f}"),
                ("Rank corr. with true value", f"{rank:.2f}")]),
        _stage("3 · Segment", "Uplift quadrants",
               [("Persuadables", f"{counts['Persuadables']}"), ("Sure Things / Lost Causes", f"{counts['Sure Things']} / {counts['Lost Causes']}"),
                ("Sleeping Dogs", f"{counts['Sleeping Dogs']}")]),
        _stage("4 · Value", "True net RevPAR change of each policy",
               [("Target Persuadables", f"{targeted:+,.0f}"), ("Discount everyone", f"{blanket:+,.0f}"), ("Oracle targeting", f"{oracle:+,.0f}")]),
        _stage("5 · Quality", "Targeting against the truth",
               [("Precision", f"{precision:.0%}" if precision == precision else "n/a"),
                ("Oracle value captured", f"{capture:.0f}%" if capture == capture else "n/a"),
                ("Gain vs blanket", f"{targeted - blanket:+,.0f}")]),
    ]
    return {"label": ex["label"], "stages": stages,
            "overall": {"precision": precision, "oracle_capture_pct": capture, "gain_vs_blanket_usd": targeted - blanket, "rank_corr": rank},
            "headline": f"Targeting {int(pers.sum())} guests earns {targeted:+,.0f} vs {blanket:+,.0f} for discounting all"}


RUNNERS = {"elasticity": elasticity_flow, "pricing": pricing_flow, "promo": promo_flow, "crm": crm_flow}


def _nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def run_all_flows(models: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run every flow on every example and compute per-flow and programme-level metrics."""
    exs = examples(cfg)
    flows: Dict[str, Any] = {}
    for key, runner in RUNNERS.items():
        flows[key] = {"title": FLOW_META[key][0], "question": FLOW_META[key][1],
                      "examples": [runner(models, cfg, e) for e in exs[key]]}
    o = {k: [e["overall"] for e in flows[k]["examples"]] for k in flows}
    flows["scorecard"] = {
        "elasticity": {"Bias removed (mean)": f"{_nanmean([x['bias_reduction_pct'] for x in o['elasticity']]):.0f}%",
                       "Mean |error| naive → IV": f"{_nanmean([x['naive_error'] for x in o['elasticity']]):.2f} → {_nanmean([x['iv_error'] for x in o['elasticity']]):.2f}"},
        "pricing": {"RevPAR vs naive (median)": f"{float(np.median([x['lift_vs_naive_pct'] for x in o['pricing']])):+.0f}%",
                    "Best achievable captured": f"{_nanmean([x['capture_of_best_pct'] for x in o['pricing']]):.0f}%"},
        "promo": {"Correct decisions": f"{sum(x['decision_correct'] for x in o['promo'])}/{len(o['promo'])}",
                  "Mean CATE |error|": f"{_nanmean([x['cate_abs_error'] for x in o['promo']]):.3f}"},
        "crm": {"Targeting precision (mean)": f"{_nanmean([x['precision'] for x in o['crm']]):.0%}",
                "Oracle value captured (mean)": f"{_nanmean([x['oracle_capture_pct'] for x in o['crm']]):.0f}%"},
    }
    flows["raw_overall"] = o
    return flows
