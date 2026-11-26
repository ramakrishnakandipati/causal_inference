"""End-to-end scenario flows (UI-independent): every flow is a list of stages, each stage with its own metrics.

Four flow types, two worked examples each (8 flows in total):

  A  Campaign read-out      - was the campaign incremental?     Miami (all 24 donors) | Las Vegas convention
  B  Validity / negative    - can we trust a null or a thin     Tampa placebo (true effect 0) | New York spring break
                              donor pool?
  C  Which days work        - Geo-CATE what-if for one day      Holiday weekend + flight surge | Rainy weekday + OTA price war
  D  Budget and pacing      - plan a flight, then audit it      Weekend-weighted $3M | "Optimised" $3M

A and B share the same seven panel stages (setup -> parallel trends -> DiD -> synthetic control -> placebo ->
economics -> verdict); C and D have their own. Everything is computed from the same `Engine` the FastAPI service uses,
and `report()` returns plain dicts so the Streamlit tab and `scripts/capture_screenshots.py` show identical numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.serving import Engine
from src.utils.config import resolve

HEADLINE = "ASCM (Ridge)"
DID_FAMILY = ["TWFE DiD", "Event Study", "CS-DiD (DR)"]
SC_FAMILY = ["SCM", "ASCM (Ridge)", "SDID", "Bayesian SC"]
P_ALPHA = 0.10          # smallest p reachable with 24 donors is 0.04; with 4 donors it is 0.20 - so no flow can "pass" on a thin pool

FLOW_TYPES = {
    "A": {"title": "Campaign read-out", "question": "Was the regional campaign incremental, and was it worth the money?", "kind": "panel"},
    "B": {"title": "Validity & negative control", "question": "Can we trust a null result, or a thin donor pool?", "kind": "panel"},
    "C": {"title": "Which days work hardest", "question": "How much lift does a specific market-day deliver?", "kind": "cate"},
    "D": {"title": "Budget & pacing", "question": "How should a budget be flighted, and does the plan survive the truth?", "kind": "pacing"},
}

FLOWS = [
    {"id": "A1", "type": "A", "short": "A1 - Miami winter peak", "title": "Miami flagship, all 24 control DMAs",
     "scenario": "mia_all24",
     "story": "Aurelia's headline question: 60 days of CTV, airport and billboard in Miami against every other DMA in the panel. "
              "Miami's winter ramp is steeper than the average market, so a naive DiD should overstate the lift."},
    {"id": "A2", "type": "A", "short": "A2 - Las Vegas convention", "title": "Las Vegas convention & billboard blitz",
     "scenario": "las_convention",
     "story": "A $2.2M billboard-led push against only four donors (ORL, SAN, PHX, AUS). The effect is real, but a four-donor "
              "placebo distribution cannot give a p-value below 0.20."},
    {"id": "B1", "type": "B", "short": "B1 - Tampa placebo", "title": "Tampa: an unexposed market (true effect 0)",
     "scenario": "tpa_placebo",
     "story": "Tampa was never advertised to. A trustworthy method must report 'no effect'; any estimator that returns a "
              "significant lift here is producing false positives."},
    {"id": "B2", "type": "B", "short": "B2 - New York spring break", "title": "New York: real but small effect, three donors",
     "scenario": "nyc_spring_break",
     "story": "A modest awareness push with three donors (BOS, CHI, DCA). The point estimates are positive but the evidence "
              "cannot separate the campaign from placebo markets."},
    {"id": "C1", "type": "C", "short": "C1 - Holiday weekend + flight surge", "title": "Holiday weekend with a flight-search surge",
     "cond": {"is_weekend": 1, "is_holiday_window": 1, "flight_search_index": 140, "weather_severity_index": 0,
              "ota_promo_pressure_index": 22, "competitor_adr_usd": 400, "days_since_launch": 15},
     "story": "A holiday-window Saturday two weeks in: flight searches into Miami are spiking, weather is clear and OTAs are quiet."},
    {"id": "C2", "type": "C", "short": "C2 - Rainy weekday + OTA price war", "title": "Rainy weekday under heavy OTA promotion",
     "cond": {"is_weekend": 0, "is_holiday_window": 0, "flight_search_index": 85, "weather_severity_index": 17,
              "ota_promo_pressure_index": 75, "competitor_adr_usd": 330, "days_since_launch": 45},
     "story": "A stormy Tuesday late in the flight: searches are soft, OTAs are discounting hard and rivals have cut rates."},
    {"id": "D1", "type": "D", "short": "D1 - Weekend-weighted $3M", "title": "Weekend-weighted flight, $3M",
     "budget": 3_000_000, "strategy": "weekend_weighted",
     "story": "Put more of the same $3M behind weekend days, where the CATE learners say media works hardest."},
    {"id": "D2", "type": "D", "short": "D2 - 'Optimised' $3M", "title": "Curve-optimised flight, $3M",
     "budget": 3_000_000, "strategy": "optimized",
     "story": "The response-curve optimiser promises a big gain. The panel is simulated, so we can check that promise against the true response."},
]
FLOW_BY_ID = {f["id"]: f for f in FLOWS}


def _f(x, d=0, pre="", suf=""):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{pre}{x:,.{d}f}{suf}"


def _metric(label, value, help=None):
    return {"label": label, "value": value, "help": help}


def _metrics_row(sub: pd.DataFrame, estimator: str):
    return sub[sub.estimator == estimator].iloc[0]


# ----------------------------------------------------------------------------- A / B : panel flows
def panel_report(eng: Engine, flow: dict) -> dict:
    sc = flow["scenario"]
    spec = eng.scenario_spec(sc)
    mdir = eng.mdir
    S = pd.read_csv(mdir / "panel_att_summary.csv")
    S = S[(S.scenario == sc) & (S.outcome == "room_nights_booked")]
    E = pd.read_csv(mdir / "economics.csv")
    E = E[E.scenario == sc].set_index("estimator")
    PT = pd.read_csv(mdir / "event_study_pretrend.csv")
    PT = PT[(PT.scenario == sc) & (PT.outcome == "room_nights_booked")].iloc[0]
    PP = pd.read_csv(mdir / "placebo_pvalues.csv")
    PP = PP[(PP.scenario == sc) & (PP.outcome == "room_nights_booked")].set_index("estimator")
    h = _metrics_row(S, HEADLINE)
    tw = _metrics_row(S, "TWFE DiD")
    truth = float(h.att_true_mean)
    T1 = 240 - (spec["start_day"] - 1)
    p = float(PP.loc[HEADLINE, "p_exact_ratio"])
    pmin = float(PP.loc[HEADLINE, "min_possible_p"])
    rank, n_units = int(PP.loc[HEADLINE, "rank_of_treated"]), int(PP.loc[HEADLINE, "n_units"])
    has_econ = HEADLINE in E.index                  # the Tampa placebo has no spend, so no economics
    nan = float("nan")
    ir = E.loc[HEADLINE] if has_econ else pd.Series({"spend_usd": 0.0, "incremental_room_nights": nan, "incremental_net_revenue_usd": nan, "iroas": nan,
                                                     "iroas_lo": nan, "iroas_hi": nan, "cpirn_usd": nan, "true_cpirn_usd": nan, "true_iroas": 0.0,
                                                     "inc_rn_lo": nan, "inc_rn_hi": nan})
    spend = float(ir.spend_usd)
    sig = (h.ci_lo > 0) and (p <= P_ALPHA)
    true_ir = float(ir.true_iroas)

    # --- verdict logic (the same rule for every panel flow)
    if truth == 0:
        verdict = "NO DETECTABLE EFFECT" if not sig else "FALSE POSITIVE"
        action = "Do not credit the campaign; the market was never exposed." if not sig else "Distrust this estimator on this market."
    elif not has_econ:
        verdict, action = "NOT EVALUATED", "No spend, no economics."
    elif sig and ir.iroas >= 1:
        verdict, action = "INCREMENTAL AND PROFITABLE", "Keep the programme; scale with a holdout re-test."
    elif sig:
        verdict, action = "INCREMENTAL BUT BELOW BREAK-EVEN", "Real lift but iROAS < 1: renegotiate cost or retarget."
    elif p >= 0.99 * pmin and p <= 0.5:
        verdict, action = "SUGGESTIVE, NOT CONCLUSIVE", "Point estimate is positive but interval spans zero and the donor pool is too thin to prove it; add donors or extend the test."
    else:
        verdict, action = "INCONCLUSIVE", "Cannot separate the campaign from placebo markets."

    steps = [
        {"key": "setup", "title": "1. Experiment design", "chart": "none",
         "metrics": [_metric("Treated market", spec["treated"]), _metric("Donor DMAs", str(len(spec["donors"]))),
                     _metric("Pre / campaign days", f"{spec['start_day'] - 1} / {T1}"), _metric("Spend", _f(spend / 1e6, 1, "$", "M")),
                     _metric("Smallest possible p", _f(pmin, 3), "1 / (donors + 1): the floor set by the donor pool")],
         "note": f"{spec['label']}. Ground truth is known (simulated panel): true mean effect {_f(truth, 0)} room nights / day."},
        {"key": "trends", "title": "2. Parallel-trends diagnostic", "chart": "trends",
         "metrics": [_metric("Pre-trend RMS", _f(PT.pretrend_rms, 1), "RMS of event-study leads k = -14..-2"),
                     _metric("Pre-trend slope / day", _f(PT.pretrend_slope, 2)),
                     _metric("Permutation p (pre-trend)", _f(PT.p_pretrend_perm, 2), "Share of placebo markets with an equal or larger pre-trend"),
                     _metric("Early vs late effect", f"{PT.post_early:,.0f} -> {PT.post_late:,.0f}", "Event-study mean, first 20 vs last 20 campaign days")],
         "note": ("Leads that drift or an effect that decays to zero are the warning signs of a violated parallel-trends assumption. "
                  "Here the pre-period gap is " + ("large" if PT.pretrend_rms > 100 else "moderate") + " relative to the donors, so DiD estimates deserve scepticism.")},
        {"key": "did", "title": "3. DiD family (assumes parallel trends)", "chart": "did",
         "metrics": [_metric(nm, f"{_metrics_row(S, nm).att_mean:,.0f}", f"bias vs truth {_metrics_row(S, nm).att_bias_pct:+.0f}%" if truth else None) for nm in DID_FAMILY]
                    + [_metric("Truth", _f(truth, 0))],
         "note": f"Plain TWFE returns {tw.att_mean:,.0f} / day against a truth of {truth:,.0f}" + (f" ({tw.att_bias_pct:+.0f}%)." if truth else ".")},
        {"key": "sc", "title": "4. Synthetic-control family (re-weighted donors)", "chart": "sc",
         "metrics": [_metric("Pre-fit RMSPE (ASCM)", _f(h.pre_rmspe, 1), "Pre-period tracking error: lower is a better counterfactual"),
                     _metric("ASCM ATT / day", _f(h.att_mean, 0), f"95% CI {_f(h.ci_lo)} to {_f(h.ci_hi)}"),
                     _metric("SCM ATT / day", _f(_metrics_row(S, 'SCM').att_mean, 0)), _metric("SDID ATT / day", _f(_metrics_row(S, 'SDID').att_mean, 0)),
                     _metric("ASCM bias vs truth", _f(h.att_bias_pct, 0, suf="%") if truth else "n/a")],
         "note": f"Total incremental room nights over {T1} days: {h.att_total:,.0f} (truth {float(h.total_true):,.0f})."},
        {"key": "placebo", "title": "5. Placebo inference", "chart": "placebo",
         "metrics": [_metric("Treated rank", f"{rank} of {n_units}", "Rank of the treated market by post/pre RMSPE ratio"),
                     _metric("Exact p-value", _f(p, 3)), _metric("Smallest attainable p", _f(pmin, 3)),
                     _metric("95% CI contains 0?", "yes" if h.ci_lo <= 0 <= h.ci_hi else "no"),
                     _metric("CI covers truth?", "yes" if bool(h.covered) else "no")],
         "note": ("A treated market that ranks first among placebos is as extreme as the design can show." if rank == 1
                  else "The treated market is not an outlier among placebo markets.") + f" With {len(spec['donors'])} donors the p-value cannot go below {pmin:.3f}."},
        {"key": "econ", "title": "6. Economics", "chart": "econ",
         "metrics": [_metric("Incremental room nights", _f(ir.incremental_room_nights), f"95% {_f(ir.inc_rn_lo)} to {_f(ir.inc_rn_hi)}"),
                     _metric("CPIRN", _f(ir.cpirn_usd, 0, "$"), f"truth {_f(ir.true_cpirn_usd, 0, '$')}"),
                     _metric("iROAS", _f(ir.iroas, 2, suf="x"), f"95% {_f(ir.iroas_lo, 2)} to {_f(ir.iroas_hi, 2)}"),
                     _metric("True iROAS", _f(true_ir, 2, suf="x")),
                     _metric("Break-even room nights", _f(spend / (ir.incremental_net_revenue_usd / ir.incremental_room_nights)) if has_econ else "n/a")],
         "note": ("iROAS = incremental net revenue / spend; the interval is propagated from the ATT interval." if has_econ else
                  "No campaign spend in this market, so there is no iROAS to compute: the only question is whether any lift is detected (it should not be).")},
    ]
    overall = [_metric("Verdict", verdict), _metric("Headline ATT / day", _f(h.att_mean, 0), f"truth {_f(truth, 0)}"),
               _metric("iROAS", _f(ir.iroas, 2, suf="x"), f"truth {_f(true_ir, 2)}x"),
               _metric("Placebo p", _f(p, 3), f"floor {pmin:.3f}"),
               _metric("TWFE bias", _f(tw.att_bias_pct, 0, suf="%") if truth else "n/a", "naive DiD vs ASCM")]
    steps.append({"key": "verdict", "title": "7. Verdict", "chart": "none", "metrics": overall[:4], "note": action, "verdict": verdict})
    return {"flow": flow, "kind": "panel", "scenario": sc, "steps": steps, "overall": overall, "verdict": verdict, "action": action,
            "headline": {"att_mean": float(h.att_mean), "att_total": float(h.att_total), "iroas": float(ir.iroas), "p": p, "truth": truth, "true_iroas": true_ir},
            "has_econ": has_econ}


# ----------------------------------------------------------------------------- C : geo-CATE what-if
def cate_row(eng: Engine, cond: dict) -> dict:
    win = eng.window
    row = {f: float(win[f].median()) for f in eng.feature_names}
    row.update({k: float(v) for k, v in cond.items()})
    row["dow"] = 5.0 if row["is_weekend"] else 2.0
    return row


def cate_report(eng: Engine, flow: dict) -> dict:
    win = eng.window
    base_row = {f: float(win[f].median()) for f in eng.feature_names}
    row = cate_row(eng, flow["cond"])
    learners = [k for k in ["S-Learner", "R-Learner", "DR-Learner", "TARNet", "CFRNet-DragonNet", "Causal Forest DML", "Linear DML", "T-Learner", "X-Learner", "Domain Adaptation"] if k in eng.cate_models]
    preds = {}
    for ln in learners:
        o = eng.cate([row], ln)
        preds[ln] = float(o["incremental_room_nights"][0])
    avg = {ln: float(np.mean(eng.cate(eng.window_features().to_dict("records"), ln)["incremental_room_nights"])) for ln in learners}
    med, lo, hi = float(np.median(list(preds.values()))), min(preds.values()), max(preds.values())
    avg_med = float(np.median(list(avg.values())))
    index = med / avg_med if avg_med else float("nan")
    feat = eng.feature_names
    Z = (win[feat] - win[feat].mean()) / win[feat].std().replace(0, 1)
    z = (pd.Series(row)[feat] - win[feat].mean()) / win[feat].std().replace(0, 1)
    nn = ((Z - z) ** 2).sum(axis=1).idxmin()
    near = win.loc[nn]
    truth = float(near["tau_true"])
    best = eng.best_learner
    best_pred = preds.get(best, med)
    err = best_pred - truth
    if index >= 1.10:
        verdict, action = "LEAN IN", f"Weight spend up on days like this (about {index:.2f}x the campaign-average lift)."
    elif index <= 0.90:
        verdict, action = "PULL BACK", f"Shift spend away from days like this (about {index:.2f}x the campaign-average lift)."
    else:
        verdict, action = "HOLD EVEN", f"Lift is close to the campaign average ({index:.2f}x); no reason to re-weight."
    diffs = {k: (row[k], base_row[k]) for k in flow["cond"] if k in base_row and abs(row[k] - base_row[k]) > 1e-9}
    steps = [
        {"key": "cond", "title": "1. Market conditions", "chart": "none",
         "metrics": [_metric(k.replace("_", " "), f"{v:g}", f"typical campaign day: {b:g}") for k, (v, b) in diffs.items()][:6],
         "note": flow["story"]},
        {"key": "learners", "title": "2. Learner predictions", "chart": "learners",
         "metrics": [_metric(ln, _f(v, 0), "incremental room nights / day") for ln, v in list(preds.items())[:5]],
         "note": f"{len(preds)} learners score this day; the median is {med:,.0f} room nights, a spread of {lo:,.0f} to {hi:,.0f}."},
        {"key": "agree", "title": "3. Agreement & uncertainty", "chart": "none",
         "metrics": [_metric("Median lift (room nights)", _f(med, 0)), _metric("Learner spread", _f(hi - lo, 0), "max - min across learners"),
                     _metric("Spread / median", _f((hi - lo) / med * 100 if med else float("nan"), 0, suf="%")),
                     _metric("Best learner", best or "n/a"), _metric("Best-learner lift", _f(best_pred, 0))],
         "note": "A wide spread means the conditions sit in a thinly populated corner of the 60 treated days; treat the point estimate with caution."},
        {"key": "index", "title": "4. Versus the campaign average", "chart": "index",
         "metrics": [_metric("Campaign-average lift", _f(avg_med, 0), "median across learners over the 60 campaign days"),
                     _metric("This day", _f(med, 0)), _metric("Relative index", _f(index, 2, suf="x")),
                     _metric("Lift %", _f(100 * med / eng.meta["base"], 1, suf="%"), f"of a {eng.meta['base']:,.0f} room-night baseline")],
         "note": "Index > 1 means media works harder than average on this kind of day."},
        {"key": "audit", "title": "5. Audit vs the nearest real day", "chart": "none",
         "metrics": [_metric("Nearest campaign day", f"day {int(near['day_index'])}"), _metric("True effect that day", _f(truth, 0)),
                     _metric("Best-learner error", _f(err, 0), "prediction - truth on that day"),
                     _metric("Verdict", verdict)],
         "note": "The nearest simulated day is the only ground truth available for a hypothetical scenario; it is a sanity check, not a score."},
    ]
    overall = [_metric("Verdict", verdict), _metric("Median lift / day", _f(med, 0), f"{lo:,.0f} to {hi:,.0f}"),
               _metric("Index vs average", _f(index, 2, suf="x")), _metric("Best-learner error", _f(err, 0), "vs nearest real day")]
    return {"flow": flow, "kind": "cate", "steps": steps, "overall": overall, "verdict": verdict, "action": action, "preds": preds, "avg": avg_med,
            "row": row, "headline": {"median": med, "index": index, "lo": lo, "hi": hi, "best_learner": best}}


# ----------------------------------------------------------------------------- D : budget and pacing
def pacing_report(eng: Engine, flow: dict) -> dict:
    budget, strat = flow["budget"], flow["strategy"]
    sim = eng.simulate(budget, strat)
    PV = pd.read_csv(eng.mdir / "pacing_validation.csv").set_index("strategy")
    cmp_ = pd.DataFrame(sim["strategy_comparison"]).set_index("strategy")
    even = cmp_.loc["even"]
    sel = cmp_.loc[strat]
    proj = float(PV.loc[strat, "projected_vs_even_pct"])
    tru = float(PV.loc[strat, "true_vs_even_pct"])
    gap = proj - tru
    cv = eng.curve
    trust = abs(gap) <= 2.0
    if strat == "even":
        verdict, action = "BASELINE", "Even pacing is the reference plan."
    elif trust and tru >= -1.0:
        verdict, action = "SAFE TO RUN", f"Projected {proj:+.1f}% and true {tru:+.1f}% vs even agree within 2 points: a harmless, defensible re-weighting."
    elif gap > 2.0:
        verdict, action = "DO NOT TRUST THE PROJECTION", f"The curve projects {proj:+.1f}% vs even but the true response gives {tru:+.1f}%: the gain is a model artefact."
    else:
        verdict, action = "REVIEW", f"Projected {proj:+.1f}%, true {tru:+.1f}%."
    steps = [
        {"key": "inputs", "title": "1. Plan inputs", "chart": "none",
         "metrics": [_metric("Budget", _f(budget / 1e6, 2, "$", "M")), _metric("Strategy", strat), _metric("Campaign days", str(sim["days"])),
                     _metric("Net revenue / room night", _f(sim["net_revenue_per_room_night"], 0, "$"))],
         "note": flow["story"]},
        {"key": "curve", "title": "2. Response-curve quality", "chart": "none",
         "metrics": [_metric("Curve R2", _f(cv.r2, 2), "Fit of adstock + Hill to the estimated daily lift"), _metric("Adstock theta", _f(cv.theta, 2), "sits on its prior bound 0.90"),
                     _metric("Half-saturation K", _f(cv.K, 0, suf="k")), _metric("Weak fit?", "yes" if sim["curve_reliability"]["weak"] else "no")],
         "note": sim["curve_reliability"]["note"]},
        {"key": "plan", "title": "3. Daily plan & projection", "chart": "plan",
         "metrics": [_metric("Projected incremental RN", _f(sel.incremental_room_nights, 0), f"even: {even.incremental_room_nights:,.0f}"),
                     _metric("CPIRN", _f(sel.cpirn_usd, 0, "$")), _metric("iROAS", _f(sel.iroas, 2, suf="x")),
                     _metric("Net revenue", _f(sel.net_revenue_usd / 1e6, 2, "$", "M"))],
         "note": "Projections come from the fitted response curve x the best learner's day-by-day multiplier."},
        {"key": "compare", "title": "4. Strategy comparison", "chart": "compare",
         "metrics": [_metric("Best projected strategy", sim["best_strategy"]), _metric("Projected vs even", _f(proj, 1, suf="%")),
                     _metric("Spread across plans", _f(cmp_.incremental_room_nights.max() - cmp_.incremental_room_nights.min(), 0, suf=" RN")),
                     _metric("Back-loaded vs even (true)", _f(float(PV.loc["back_loaded", "true_vs_even_pct"]), 1, suf="%"))],
         "note": "All eight plans spend the same budget."},
        {"key": "audit", "title": "5. Audit against the true response", "chart": "audit",
         "metrics": [_metric("Projected vs even", _f(proj, 1, suf="%")), _metric("True vs even", _f(tru, 1, suf="%")),
                     _metric("Projection error", _f(gap, 1, suf=" pts")), _metric("True incremental RN", _f(float(PV.loc[strat, "true_incremental_rn"]), 0))],
         "note": "Possible only because the panel is simulated; a real campaign would need a pacing holdout."},
        {"key": "verdict", "title": "6. Verdict", "chart": "none",
         "metrics": [_metric("Verdict", verdict), _metric("Projected", _f(proj, 1, suf="%")), _metric("True", _f(tru, 1, suf="%")), _metric("iROAS (projected)", _f(sel.iroas, 2, suf="x"))],
         "note": action, "verdict": verdict},
    ]
    overall = [_metric("Verdict", verdict), _metric("Projected vs even", _f(proj, 1, suf="%")), _metric("True vs even", _f(tru, 1, suf="%")),
               _metric("iROAS (projected)", _f(sel.iroas, 2, suf="x"))]
    return {"flow": flow, "kind": "pacing", "steps": steps, "overall": overall, "verdict": verdict, "action": action, "sim": sim,
            "headline": {"projected": proj, "true": tru, "iroas": float(sel.iroas), "total_rn": float(sel.incremental_room_nights)}}


def report(eng: Engine, flow: dict) -> dict:
    kind = FLOW_TYPES[flow["type"]]["kind"]
    return {"panel": panel_report, "cate": cate_report, "pacing": pacing_report}[kind](eng, flow)
