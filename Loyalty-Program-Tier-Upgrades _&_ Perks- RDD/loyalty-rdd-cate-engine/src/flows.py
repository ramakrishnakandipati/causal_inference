"""End-to-end scenario flows (UI independent). Every flow is a list of stages, each with its own metrics.

Four flow types, two worked examples each (8 flows):

  A  Gold effect read-out        - does Gold change behaviour, and what is it worth?   A1 fuzzy design (headline) | A2 sharp design
  B  Validity & negative control - can we trust the design and a null result?          B1 placebo programme (true effect 0) | B2 gamed threshold
  C  Guest what-if (CATE)        - how much does Gold do for THIS guest?               C1 bleisure couple | C2 corporate road warrior
  D  Threshold policy            - what if the Gold threshold moved?                   D1 lower to 17 nights | D2 raise to 24 nights

A and B share six panel stages (setup, density, balance, estimates, robustness, economics), B adds a seventh
(replication), and every flow ends in a verdict. C has five stages (profile, context, models, evidence, economics) and D six
(today, candidate, who, impact, net value, sensitivity). `build_all()` returns plain dicts (and `save()` writes them to
outputs/metrics/flows_report.json) so the Streamlit tab and scripts/capture_screenshots.py show identical numbers.
Money is on a revenue basis scaled by an assumed contribution margin; both are printed next to every money figure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.data import synthetic_generator as gen
from src.models import rdd_estimators as R

ROOT = Path(__file__).resolve().parents[1]
REPORT_NAME = "flows_report.json"
OUTCOMES = ["next_year_room_nights", "next_year_ancillary_spend", "next_year_direct_booking_ratio"]
OC_LABEL = {"next_year_room_nights": "Room nights", "next_year_ancillary_spend": "Ancillary spend",
            "next_year_direct_booking_ratio": "Direct-booking ratio"}

PERSONAS = {
    "Leisure family traveller": dict(historical_adr=170.0, business_travel_ratio=0.08, lead_time_avg=70.0, ancillary_spend_t=900.0,
                                     direct_booking_ratio_t=0.30, tenure_years=5.0, co_brand_card_holder=False,
                                     home_region="North America", age_group="40-49", weekend_stay_ratio=0.55),
    "Bleisure couple": dict(historical_adr=175.0, business_travel_ratio=0.40, lead_time_avg=40.0, ancillary_spend_t=700.0,
                            direct_booking_ratio_t=0.35, tenure_years=4.0, co_brand_card_holder=False,
                            home_region="Europe", age_group="30-39", weekend_stay_ratio=0.40),
    "Corporate road warrior": dict(historical_adr=210.0, business_travel_ratio=0.88, lead_time_avg=8.0, ancillary_spend_t=350.0,
                                   direct_booking_ratio_t=0.40, tenure_years=9.0, co_brand_card_holder=True,
                                   home_region="North America", age_group="40-49", weekend_stay_ratio=0.05),
    "Budget solo explorer": dict(historical_adr=75.0, business_travel_ratio=0.10, lead_time_avg=25.0, ancillary_spend_t=150.0,
                                 direct_booking_ratio_t=0.15, tenure_years=1.5, co_brand_card_holder=False,
                                 home_region="Asia-Pacific", age_group="18-29", weekend_stay_ratio=0.45),
    "Luxury resort loyalist": dict(historical_adr=420.0, business_travel_ratio=0.20, lead_time_avg=90.0, ancillary_spend_t=2600.0,
                                   direct_booking_ratio_t=0.60, tenure_years=12.0, co_brand_card_holder=True,
                                   home_region="Middle East & Africa", age_group="50-64", weekend_stay_ratio=0.50),
}

FLOW_TYPES = {
    "A": {"title": "Gold effect read-out", "question": "Does reaching Gold change behaviour, and is it worth the perks?", "kind": "panel"},
    "B": {"title": "Validity & negative control", "question": "Can we trust the design, and can it tell 'no effect' from an effect?", "kind": "panel"},
    "C": {"title": "Guest what-if", "question": "How much does Gold do for this particular guest?", "kind": "cate"},
    "D": {"title": "Threshold policy", "question": "What happens to nights and money if the Gold threshold moves?", "kind": "policy"},
}

FLOWS = [
    {"id": "A1", "type": "A", "short": "A1 - Fuzzy design (headline)", "title": "The programme as run: co-brand fast-track and status matches",
     "data": {"regime": "fuzzy", "manipulation": False, "effect_scale": 1.0},
     "story": "12% of members below 20 nights already hold Gold (credit-card fast-track, status match) and 94% above it do (sync lag). "
              "Eligibility at 20 nights instruments actual Gold, so the estimate is for guests whose status follows the threshold."},
    {"id": "A2", "type": "A", "short": "A2 - Sharp design", "title": "Strict night rule: Gold if and only if 20+ nights",
     "data": {"regime": "sharp", "manipulation": False, "effect_scale": 1.0},
     "story": "The same population under a clean rule with no fast-tracks. Take-up jumps from 0% to 100%, so sharp and fuzzy "
              "RDD should agree on the effect and the sharp one should be tighter."},
    {"id": "B1", "type": "B", "short": "B1 - Placebo programme", "title": "Gold perks that do nothing (true effect 0)",
     "data": {"regime": "fuzzy", "manipulation": False, "effect_scale": 0.0},
     "story": "A programme whose perks change nothing. A trustworthy design must say 'no detectable effect'; a significant "
              "jump here would be a false positive."},
    {"id": "B2", "type": "B", "short": "B2 - Gamed threshold", "title": "Members top up stays to just reach 20 nights",
     "data": {"regime": "fuzzy", "manipulation": True, "effect_scale": 1.0},
     "story": "40% of guests 1 to 3 nights short buy extra stays to land at 20 or 21 nights. The guests who reach Gold are no longer "
              "comparable to those who miss it, so the density test should fire and the estimate should not be trusted."},
    {"id": "C1", "type": "C", "short": "C1 - Bleisure couple", "title": "Bleisure couple, mid-priced stays",
     "persona": "Bleisure couple",
     "story": "Mixed business and leisure, mid-tier ADR, books well ahead: the profile the generator makes most responsive."},
    {"id": "C2", "type": "C", "short": "C2 - Corporate road warrior", "title": "Corporate road warrior, last-minute bookings",
     "persona": "Corporate road warrior",
     "story": "88% business travel, 8-day lead time, rigid itinerary, co-brand card: stays are set by the employer, not by perks."},
    {"id": "D1", "type": "D", "short": "D1 - Lower to 17 nights", "title": "Open Gold to guests with 17 to 19 nights",
     "new_cutoff": 17, "perk_cost": 120.0,
     "story": "Three nights below today's rule, perks cost $120 per activated member. Close enough to 20 that the RDD effect still applies."},
    {"id": "D2", "type": "D", "short": "D2 - Raise to 24 nights", "title": "Tighten Gold to 24 nights to cut perk cost",
     "new_cutoff": 24, "perk_cost": 300.0,
     "story": "Guests with 20 to 23 nights would lose Gold. Perks are expensive ($300 per member), so the question is whether the saving "
              "exceeds the lost stays."},
]
FLOW_BY_ID = {f["id"]: f for f in FLOWS}
GOOD = {"GOLD PAYS OFF", "NO DETECTABLE EFFECT (CORRECT)", "LEAN IN", "ADOPT"}
BAD = {"DO NOT TRUST THE ESTIMATE", "CAUSAL BUT UNPROFITABLE", "PULL BACK", "DO NOT CHANGE", "FALSE POSITIVE"}


# ----------------------------------------------------------------------------- helpers
def load_assumptions() -> dict:
    a = {"contribution_margin": 0.40, "perk_cost_per_member": 120.0, "seed": 42}
    p = ROOT / "configs" / "config.yaml"
    if p.exists():
        a.update((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("flows", {}) or {})
    return a


def _rdd_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))["rdd"]


def _f(x, d: int = 0, pre: str = "", suf: str = "") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{pre}{x:,.{d}f}{suf}"


def _money(x, d: int = 0) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{'-' if x < 0 else ''}${abs(x):,.{d}f}"


def _fmt_oc(oc: str, v: Optional[float], signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    s = "+" if signed and v >= 0 else ""
    if oc == "next_year_ancillary_spend":
        return f"{s}{_money(v)}" if v >= 0 else _money(v)
    if oc == "next_year_direct_booking_ratio":
        return f"{s}{v:.3f}"
    return f"{s}{v:.2f}"


def _m(label: str, value: str, help: str = "") -> dict:
    return {"label": label, "value": value, "help": help}


def _stage(key: str, title: str, question: str, status: str, metrics: List[dict], takeaway: str,
           chart: Optional[dict] = None, table: Optional[List[dict]] = None) -> dict:
    return {"key": key, "title": title, "question": question, "status": status, "metrics": metrics,
            "takeaway": takeaway, "chart": chart or {"type": "none"}, "table": table}


def _py(o):
    """Recursively convert numpy / pandas objects to JSON-safe Python (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.ndarray, pd.Index, pd.Series)):
        return _py(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return _py(o.to_dict("records"))
    return o


def _economics(d_nights: float, d_anc: float, adr: float, perk: float, margin: float) -> dict:
    room_rev = d_nights * adr
    gross = room_rev + d_anc
    contrib = gross * margin
    net = contrib - perk
    return {"room_rev": room_rev, "anc_rev": d_anc, "gross": gross, "contrib": contrib, "perk": perk, "net": net,
            "roi": net / perk if perk else None, "break_even_margin": perk / gross if gross > 0 else None}


# ============================================================================= B : replication evidence
REPLICATIONS = 30


def _replication(flow: dict, c: float, h: float, n: int) -> dict:
    """Re-draw the panel REPLICATIONS times: how often would this design cry wolf, and how often catch gaming?"""
    d = flow["data"]
    fuzzy = d["regime"] == "fuzzy"
    truth0 = gen.TARGETS[OUTCOMES[0]] * d["effect_scale"]
    ests, ses, flagged = [], [], []
    for s in range(1000, 1000 + REPLICATIONS):
        dd = gen.generate(n, d["regime"], d["manipulation"], int(c), s, effect_scale=d["effect_scale"])
        e = R.rdd_estimate(dd, OUTCOMES[0], c, h, 1, "triangular", fuzzy)
        ests.append(e["estimate"])
        ses.append(e["se"])
        flagged.append(R.mccrary_density_test(dd["qualifying_nights_t"], c)["reject_continuity_5pct"])
    ests, ses = np.array(ests), np.array(ses)
    z = (ests - truth0) / ses
    out = {"k": REPLICATIONS, "truth": truth0, "mean_est": float(ests.mean()), "mean_se": float(ses.mean()), "sd_est": float(ests.std()),
           "reject_zero": float(np.mean(np.abs(ests / ses) > 1.96)), "coverage": float(np.mean(np.abs(z) <= 1.96)),
           "density_flag_rate": float(np.mean(flagged)), "est": ests.tolist(), "lo": (ests - 1.96 * ses).tolist(), "hi": (ests + 1.96 * ses).tolist()}
    if d["manipulation"]:     # companion: the same test on clean data (its false-alarm rate)
        clean = []
        for s in range(1000, 1000 + REPLICATIONS):
            dd = gen.generate(n, d["regime"], False, int(c), s, effect_scale=d["effect_scale"])
            clean.append(R.mccrary_density_test(dd["qualifying_nights_t"], c)["reject_continuity_5pct"])
        out["clean_density_flag_rate"] = float(np.mean(clean))
    return out


def _replication_stage(flow: dict, rep: dict) -> dict:
    k = rep["k"]
    if not flow["data"]["manipulation"]:
        ok = rep["reject_zero"] <= 0.15
        return _stage(
            "replication", "Replication: is this one draw typical?", f"If we re-drew the placebo programme {k} times, how often would the design cry wolf?",
            "PASS" if ok else "WARN",
            [_m("Replications", str(k), "fresh 50,000-member panels, true effect 0"),
             _m("False-positive rate", _f(rep["reject_zero"] * 100, 0, suf="%"), "room-night 95% CI excludes 0; nominal 5%"),
             _m("Mean estimate", f"{rep['mean_est']:+.3f} nights", "should be about 0"),
             _m("Mean SE vs spread of estimates", f"{rep['mean_se']:.3f} vs {rep['sd_est']:.3f}", "equal when the standard errors are honest"),
             _m("CI covers the truth", _f(rep["coverage"] * 100, 0, suf="%"), "nominal 95%")],
            f"Across {k} placebo draws the estimates centre on {rep['mean_est']:+.3f} and the interval excludes zero {rep['reject_zero']:.0%} of the time "
            f"(nominal 5%): the method is unbiased, and one significant draw in twenty is expected.",
            {"type": "caterpillar", "est": rep["est"], "lo": rep["lo"], "hi": rep["hi"], "truth": rep["truth"]})
    ok = rep["density_flag_rate"] >= 0.9 and rep["clean_density_flag_rate"] <= 0.15
    return _stage(
        "replication", "Replication: does the validity test reliably catch gaming?", f"Over {k} re-draws, how often does the density test fire?",
        "PASS" if ok else "WARN",
        [_m("Replications", str(k), "gamed panels and, for comparison, clean panels"),
         _m("Density test fires on gamed data", _f(rep["density_flag_rate"] * 100, 0, suf="%"), "power"),
         _m("Density test fires on clean data", _f(rep["clean_density_flag_rate"] * 100, 0, suf="%"), "false-alarm rate; nominal 5%"),
         _m("Mean estimate on gamed data", f"{rep['mean_est']:+.2f} nights", f"true effect {rep['truth']:+.2f}"),
         _m("CI covers the truth", _f(rep["coverage"] * 100, 0, suf="%"))],
        f"The density test fires in {rep['density_flag_rate']:.0%} of gamed draws and {rep['clean_density_flag_rate']:.0%} of clean ones. "
        f"The gamed estimate ({rep['mean_est']:+.2f}) still looks plausible, so only the validity tests reveal the problem.",
        {"type": "rates", "labels": ["Fires on gamed data", "Fires on clean data"], "values": [rep["density_flag_rate"], rep["clean_density_flag_rate"]]})


# ============================================================================= A / B : panel flows
def panel_report(flow: dict, svc, assump: dict) -> dict:
    from src.evaluation.visualizer import _count_fits, local_poly_curve
    d, c = flow["data"], float(svc.cutoff)
    seed = int(assump["seed"])
    df = gen.generate(int(svc.meta["n_panel"]), d["regime"], d["manipulation"], int(c), seed, effect_scale=d["effect_scale"])
    suite = R.run_rdd_suite(df, _rdd_cfg())
    truth = gen.local_truth(df)
    main = suite["main"].set_index("outcome")
    h = float(suite["bandwidths"]["used"])
    r = df["qualifying_nights_t"].to_numpy(float)
    z = (r >= c)
    fuzzy = d["regime"] == "fuzzy"
    margin, perk = float(assump["contribution_margin"]), float(assump["perk_cost_per_member"])
    stages: List[dict] = []

    # ---- 1 setup ----------------------------------------------------------------------------
    t_below, t_above = float(df.loc[~z, "gold_status"].mean()), float(df.loc[z, "gold_status"].mean())
    first_stage = float(main.iloc[0].get("first_stage_jump", np.nan)) if fuzzy else 1.0
    n_win = int((np.abs(r - c) <= h).sum())
    take = df.groupby("qualifying_nights_t")["gold_status"].mean()
    take = take[(take.index >= 4) & (take.index <= 36)]
    stages.append(_stage(
        "setup", "Setup: who is in the data?", "What does the design look like before any effect is estimated?", "INFO",
        [_m("Members", _f(len(df))), _m("Design", "Fuzzy (fast-tracks)" if fuzzy else "Sharp (strict rule)"),
         _m("Gold below 20 nights", _f(t_below * 100, 1, suf="%")), _m("Gold at 20+ nights", _f(t_above * 100, 1, suf="%")),
         _m("Take-up jump", _f(first_stage, 2), "First-stage jump in Gold attainment at the cutoff"),
         _m("RDD bandwidth", f"±{h:.1f} nights", f"CCT {suite['bandwidths']['cct']:.1f}, IK {suite['bandwidths']['ik']:.1f}"),
         _m("Members in window", _f(n_win))],
        (f"Gold take-up jumps from {_f(t_below * 100, 0, suf='%')} to {_f(t_above * 100, 0, suf='%')}" if fuzzy else "Gold is assigned strictly by the 20-night rule")
        + f"; the RDD compares the {_f(n_win)} members within ±{h:.1f} nights of the cutoff.",
        {"type": "takeup", "r": take.index.tolist(), "t": take.values.tolist(), "cutoff": c}))

    # ---- 2 density --------------------------------------------------------------------------
    mc = suite["mccrary"]
    dens_ok = not mc["reject_continuity_5pct"]
    x, cnt, curves = _count_fits(df["qualifying_nights_t"], c, 10, 2)
    keep = (x >= c - 10) & (x <= c + 10)
    stages.append(_stage(
        "density", "Density: is anyone gaming the threshold?", "Do guests pile up just above 20 nights?",
        "PASS" if dens_ok else "FAIL",
        [_m("Log density ratio", f"{mc['log_density_ratio']:+.3f}", "log(right density / left density) at the cutoff; 0 = continuous"),
         _m("p-value", _f(mc["p_value"], 3), "McCrary-style test, quadratic fit with Poisson variance"),
         _m("Density left / right", f"{mc['density_left']:.0f} / {mc['density_right']:.0f}", "fitted guests per night at the boundary"),
         _m("Verdict", "No bunching" if dens_ok else "BUNCHING DETECTED")],
        "The density of qualifying nights is smooth through 20." if dens_ok else
        f"Guests are {math.exp(mc['log_density_ratio']):.1f}x more numerous just above the cutoff than the trend predicts: the threshold is being gamed.",
        {"type": "density", "x": x[keep].tolist(), "n": cnt[keep].tolist(), "cutoff": c,
         "left": {"x": curves["left"][0].tolist(), "y": curves["left"][1].tolist()},
         "right": {"x": curves["right"][0].tolist(), "y": curves["right"][1].tolist()}}))

    # ---- 3 balance --------------------------------------------------------------------------
    bal = suite["balance"]
    bal_ok = bool(bal["balanced"].all())
    zs = (bal["estimate"] / bal["se"]).fillna(0)
    worst = bal.iloc[int(zs.abs().argmax())]
    stages.append(_stage(
        "balance", "Balance: are guests alike on either side?", "Do any pre-treatment covariates jump at the cutoff?",
        "PASS" if bal_ok else "FAIL",
        [_m("Balanced covariates", f"{int(bal['balanced'].sum())} of {len(bal)}"),
         _m("Smallest Holm p", _f(float(bal['p_holm'].min()), 3), "multiple-testing adjusted"),
         _m("Largest jump", f"{worst['covariate']} ({zs.abs().max():.1f} SE)")],
        "No pre-treatment characteristic changes at the threshold, so guests just below are a fair comparison group." if bal_ok else
        "At least one covariate jumps at the cutoff, so the two sides are not comparable.",
        {"type": "balance", "covariate": bal["covariate"].tolist(), "z": zs.tolist(), "ok": bal["balanced"].tolist()}))

    # ---- 4 estimates ------------------------------------------------------------------------
    rows, covered, errs = [], 0, []
    for oc in OUTCOMES:
        e = main.loc[oc]
        cover = bool(e["ci_low"] <= truth[oc] <= e["ci_high"])
        err_se = (e["estimate"] - truth[oc]) / e["se"] if e["se"] else float("nan")
        covered += cover
        errs.append(err_se)
        rows.append({"Outcome": OC_LABEL[oc], "Estimate": _fmt_oc(oc, e["estimate"], True),
                     "95% CI": f"{_fmt_oc(oc, e['ci_low'])} to {_fmt_oc(oc, e['ci_high'])}",
                     "True effect": _fmt_oc(oc, truth[oc], True), "Error (SE)": f"{err_se:+.1f}",
                     "CI covers truth": "yes" if cover else "NO", "p": _f(e["p_value"], 4)})
    prim = main.loc["next_year_room_nights"]
    sig = bool(prim["ci_low"] > 0 or prim["ci_high"] < 0)
    r_chart = {}
    for oc in OUTCOMES:
        y = df[oc].to_numpy(float)
        sel = (r >= 4) & (r <= 36)
        bins = pd.DataFrame({"r": r[sel], "y": y[sel]}).groupby("r")["y"].mean()
        curves_oc = {}
        for side_name, side in (("left", r < c), ("right", r >= c)):
            grid = np.linspace(r[sel & side].min() + 0.5, c, 40) if side_name == "left" else np.linspace(c, r[sel & side].max() - 0.5, 40)
            f_, lo_, hi_ = local_poly_curve(r[sel & side], y[sel & side], grid, hb=5.0)
            curves_oc[side_name] = {"x": grid, "f": f_, "lo": lo_, "hi": hi_}
        r_chart[oc] = {"bins_x": bins.index, "bins_y": bins.values, **curves_oc}
    stages.append(_stage(
        "estimates", "Estimates: the jump at 20 nights", "How much does Gold move each outcome?",
        "INFO",
        [_m("Room nights", f"{_fmt_oc(OUTCOMES[0], prim['estimate'], True)} (SE {prim['se']:.2f})"),
         _m("Ancillary spend", _fmt_oc(OUTCOMES[1], main.loc[OUTCOMES[1], 'estimate'], True)),
         _m("Direct-booking ratio", _fmt_oc(OUTCOMES[2], main.loc[OUTCOMES[2], 'estimate'], True)),
         _m("CIs covering truth", f"{covered} of 3"),
         _m("First-stage F", _f(float(prim.get('first_stage_f', np.nan)), 0) if fuzzy else "n/a (sharp)")],
        (f"Gold raises room nights by {_fmt_oc(OUTCOMES[0], prim['estimate'], True)} (95% CI {_fmt_oc(OUTCOMES[0], prim['ci_low'])} to "
         f"{_fmt_oc(OUTCOMES[0], prim['ci_high'])}); the true value is {_fmt_oc(OUTCOMES[0], truth[OUTCOMES[0]], True)}." if sig else
         f"No detectable effect on room nights (95% CI {_fmt_oc(OUTCOMES[0], prim['ci_low'])} to {_fmt_oc(OUTCOMES[0], prim['ci_high'])} includes 0); "
         f"the true effect is {_fmt_oc(OUTCOMES[0], truth[OUTCOMES[0]], True)}."),
        {"type": "rdd", "outcomes": {oc: r_chart[oc] for oc in OUTCOMES}, "cutoff": c,
         "labels": {oc: OC_LABEL[oc] for oc in OUTCOMES}}, rows))

    # ---- 5 robustness -----------------------------------------------------------------------
    sens = suite["sensitivity"]
    base_sens = sens[(sens.outcome == OUTCOMES[0]) & (sens.kernel == "triangular") & (sens.poly_order == 1)].dropna(subset=["estimate"])
    dev_se = float(((base_sens["estimate"] - prim["estimate"]).abs() / base_sens["se"]).max())
    plc = suite["placebo"]
    plc_prim = plc[plc.outcome == OUTCOMES[0]]
    plc_null = int(plc_prim["null_not_rejected"].sum())
    don = suite["donut"].set_index("outcome").loc[OUTCOMES[0]]
    don_dev = abs(don["donut"] - don["baseline"]) / don["donut_se"] if don["donut_se"] else float("nan")
    checks = [dev_se <= 3.0, plc_null >= 3, don_dev <= 3.0]
    rob_status = "PASS" if all(checks) else ("WARN" if sum(checks) == 2 else "FAIL")
    stages.append(_stage(
        "robustness", "Robustness: does the answer survive other choices?", "Is the estimate stable across bandwidths, placebo cutoffs and a donut hole?",
        rob_status,
        [_m("Bandwidth sweep", f"{base_sens['estimate'].min():.2f} to {base_sens['estimate'].max():.2f}", "room nights, h = 3 to 12, triangular, p = 1"),
         _m("Max deviation", f"{dev_se:.1f} SE", "largest gap to the headline estimate; pass if <= 3"),
         _m("Placebo cutoffs null", f"{plc_null} of {len(plc_prim)}", "false cutoffs at 12, 15, 25, 30; pass if >= 3"),
         _m("Donut hole (19, 20 dropped)", f"{don['donut']:.2f} vs {don['baseline']:.2f}", f"{don_dev:.1f} SE apart; pass if <= 3")],
        "The estimate holds across bandwidths, kernels and a donut hole, and false cutoffs show nothing." if rob_status == "PASS" else
        "At least one robustness check disagrees with the headline estimate.",
        {"type": "robust", "sens": base_sens[["bandwidth", "estimate", "se"]].to_dict("records"),
         "placebo": plc_prim[["placebo_cutoff", "estimate", "se"]].to_dict("records"), "headline": float(prim["estimate"])}))

    # ---- 5b replication (B flows only) ------------------------------------------------------
    rep = None
    if flow["type"] == "B":
        rep = _replication(flow, c, h, int(svc.meta["n_panel"]))
        stages.append(_replication_stage(flow, rep))

    # ---- 6 economics ------------------------------------------------------------------------
    adr = float(df.loc[np.abs(r - c) <= h, "historical_adr"].mean())
    eco = _economics(float(prim["estimate"]), float(main.loc[OUTCOMES[1], "estimate"]), adr, perk, margin)
    eco_t = _economics(float(truth[OUTCOMES[0]]), float(truth[OUTCOMES[1]]), adr, perk, margin)
    stages.append(_stage(
        "economics", "Economics: is Gold worth its perks?", "Per Gold member per year: incremental revenue against perk cost",
        "PASS" if eco["net"] > 0 else "WARN",
        [_m("Extra room revenue", _money(eco["room_rev"]), f"{prim['estimate']:.2f} nights x ${adr:,.0f} window ADR"),
         _m("Extra ancillary spend", _money(eco["anc_rev"])),
         _m("Gross revenue", _money(eco["gross"])),
         _m(f"Contribution at {margin:.0%}", _money(eco["contrib"]), "assumed contribution margin"),
         _m("Perk cost", _money(perk), "assumed per Gold member"),
         _m("Net per member", _money(eco["net"]), f"at the true effect: {_money(eco_t['net'])}"),
         _m("Break-even margin", _f(eco["break_even_margin"] * 100, 0, suf="%") if eco["break_even_margin"] else "n/a",
            "contribution margin at which perks exactly pay for themselves")],
        (f"Each Gold member returns {_money(eco['gross'])} of extra revenue; after a {margin:.0%} margin and {_money(perk)} of perks the net is {_money(eco['net'])} "
         f"(ROI {eco['roi']:+.0%}). Direct-booking gains are not monetised." if eco["gross"] > 0 else
         "There is no measurable extra revenue to set against the perk cost."),
        {"type": "waterfall", "labels": ["Gross revenue", f"Contribution ({margin:.0%})", "Perk cost", "Net"],
         "values": [eco["gross"], eco["contrib"], -perk, eco["net"]]}))

    # ---- verdict ----------------------------------------------------------------------------
    gates = [dens_ok, bal_ok, rob_status != "FAIL"]
    if not (dens_ok and bal_ok):
        verdict = "DO NOT TRUST THE ESTIMATE"
        reason = ("The design fails a validity check (" + ("density" if not dens_ok else "balance") + "), so the jump cannot be read as a causal effect"
                  + (f"; the density test fires in {rep['density_flag_rate']:.0%} of gamed re-draws and {rep['clean_density_flag_rate']:.0%} of clean ones." if rep else "."))
    elif sig and abs(truth[OUTCOMES[0]]) < 1e-9:
        verdict = "FALSE POSITIVE"
        reason = (f"The interval excludes zero (p = {prim['p_value']:.3f}) but the true effect is 0. At the 5% level about 1 in 20 placebo programmes looks like this"
                  + (f"; over {rep['k']} re-draws the rate was {rep['reject_zero']:.0%}." if rep else "."))
    elif not sig:
        zero = abs(truth[OUTCOMES[0]]) < 1e-9
        verdict = "NO DETECTABLE EFFECT (CORRECT)" if zero else "NO DETECTABLE EFFECT"
        reason = "The design is valid and the room-night interval includes zero" + (": this matches the true effect of 0." if zero else ", although the true effect is not zero.")
    elif eco["net"] > 0:
        verdict, reason = "GOLD PAYS OFF", f"Valid design, a significant effect, and {_money(eco['net'])} net per member at a {margin:.0%} margin."
    else:
        verdict, reason = "CAUSAL BUT UNPROFITABLE", "The effect is real but too small to cover the perk cost."
    summary = {
        "verdict": verdict, "reason": reason,
        "metrics": [_m("Validity gates passed", f"{sum(gates)} of 3", "density, balance, robustness"),
                    _m("Room-night effect", _fmt_oc(OUTCOMES[0], prim["estimate"], True)),
                    _m("True room-night effect", _fmt_oc(OUTCOMES[0], truth[OUTCOMES[0]], True)),
                    _m("Mean |error| (SE)", f"{np.nanmean(np.abs(errs)):.1f}"),
                    _m("CIs covering truth", f"{covered} of 3"),
                    _m("Net per Gold member", _money(eco["net"])),
                    _m("ROI", _f(eco["roi"] * 100, 0, suf="%") if eco["roi"] is not None else "n/a")]
                   + ([_m("Replication", f"{rep['reject_zero']:.0%} false positives" if not flow["data"]["manipulation"] else f"{rep['density_flag_rate']:.0%} detection")] if rep else [])}
    kpi = {"replication": rep and {k: v for k, v in rep.items() if k not in ("est", "lo", "hi")}, "headline_effect": float(prim["estimate"]), "true_effect": float(truth[OUTCOMES[0]]), "ci_covers_truth": covered,
           "mean_abs_err_se": float(np.nanmean(np.abs(errs))), "net_per_member": eco["net"], "roi": eco["roi"],
           "gates_passed": int(sum(gates)), "density_p": float(mc["p_value"]), "balanced": bool(bal_ok),
           "robust_status": rob_status, "significant": sig, "first_stage": first_stage,
           "per_outcome": {oc: {"estimate": float(main.loc[oc, "estimate"]), "se": float(main.loc[oc, "se"]),
                                "truth": float(truth[oc]), "covered": bool(main.loc[oc, "ci_low"] <= truth[oc] <= main.loc[oc, "ci_high"]),
                                "err_se": float((main.loc[oc, "estimate"] - truth[oc]) / main.loc[oc, "se"])} for oc in OUTCOMES}}
    return {"stages": stages, "summary": summary, "kpi": kpi}


# ============================================================================= C : guest what-if
def cate_report(flow: dict, svc, assump: dict) -> dict:
    profile = PERSONAS[flow["persona"]]
    margin, perk = float(assump["contribution_margin"]), float(assump["perk_cost_per_member"])
    out = svc.cate(profile, "room_nights", None, True)
    tau = out["predicted_effect"]
    best = out["learner"]
    lb = svc.leaderboards["room_nights"].set_index("learner")
    alt = pd.Series(out["all_learner_estimates"]).sort_values()
    corr = lb["pearson"].reindex(alt.index)
    w = (corr.clip(lower=0) ** 2).fillna(0)
    weighted = float((alt * w).sum() / w.sum()) if w.sum() > 0 else float(alt.mean())
    good = lb[(lb["pearson"] >= 0.85)].index
    core = alt.reindex([l for l in alt.index if l in good])
    ref = svc._reference("room_nights", best)
    p10, p50, p90 = (float(np.quantile(ref, q)) for q in (0.1, 0.5, 0.9))
    stages: List[dict] = []

    stages.append(_stage(
        "profile", "Profile: who is this guest?", "What do we know about the guest, and which segment do they fall in?", "INFO",
        [_m("Segment", out["segment"]), _m("ADR", _money(profile["historical_adr"])),
         _m("Business share", _f(profile["business_travel_ratio"] * 100, 0, suf="%")),
         _m("Lead time", f"{profile['lead_time_avg']:.0f} days"), _m("Weekend share", _f(profile["weekend_stay_ratio"] * 100, 0, suf="%")),
         _m("Tenure", f"{profile['tenure_years']:.0f} years"), _m("Co-brand card", "yes" if profile["co_brand_card_holder"] else "no"),
         _m("Direct-booking share", _f(profile["direct_booking_ratio_t"] * 100, 0, suf="%"))],
        f"{flow['persona']} ({out['segment'].lower()}) with a {_money(profile['historical_adr'])} average rate and {profile['business_travel_ratio']:.0%} business travel.",
        {"type": "profile", "profile": {k: (v if not isinstance(v, bool) else int(v)) for k, v in profile.items()}}))

    pct = out["percentile_among_guests"]
    hist, edges = np.histogram(ref, bins=24)
    stages.append(_stage(
        "context", "Context: where does the guest sit among all guests?", "Is the predicted effect high or low compared with the average guest near the cutoff?", "INFO",
        [_m("Predicted Gold effect", f"{tau:+.2f} nights", f"learner: {best}"), _m("Average guest", f"{out['population_average_effect']:+.2f} nights"),
         _m("Percentile", f"{pct:.0f}th"), _m("10th / 50th / 90th", f"{p10:.1f} / {p50:.1f} / {p90:.1f}", "predicted effects of guests near the cutoff")],
        f"This guest's predicted effect is {'above' if tau > out['population_average_effect'] else 'below'} the {out['population_average_effect']:+.2f}-night average, "
        f"at the {pct:.0f}th percentile of guests.",
        {"type": "distribution", "edges": edges.tolist(), "counts": hist.tolist(), "guest": tau, "mean": out["population_average_effect"]}))

    same_sign = int((np.sign(alt) == np.sign(tau)).sum())
    cf = None
    try:
        cf = svc.cate(profile, "room_nights", "causal_forest_dml", False).get("interval_95")
    except Exception:
        pass
    spread = float(core.std()) if len(core) > 1 else float(alt.std())
    stages.append(_stage(
        "models", "Models: do the learners agree?", "How consistent is the prediction across the 12 learners?",
        "PASS" if spread < 0.6 else "WARN",
        [_m("Headline learner", f"{tau:+.2f} ({best})"), _m("Accuracy-weighted mean", f"{weighted:+.2f}", "weights = correlation² with the true effect on the leaderboard"),
         _m("Learners with corr ≥ 0.85", f"{len(core)} of {len(alt)}"), _m("Spread of those", f"±{spread:.2f} (SD)", "agreement among accurate learners"),
         _m("Same sign as headline", f"{same_sign} of {len(alt)}"), _m("Range, all learners", f"{alt.min():+.2f} to {alt.max():+.2f}"),
         _m("Causal-forest 95% interval", f"{cf[0]:+.2f} to {cf[1]:+.2f}" if cf else "n/a")],
        f"Accurate learners agree within ±{spread:.2f} nights; the weakest ones (distilled orthogonal forest, linear DML) are the outliers." if spread < 0.6 else
        "The learners disagree noticeably for this profile; treat the point estimate with caution.",
        {"type": "learners", "learners": alt.index.tolist(), "values": alt.values.tolist(), "corr": corr.fillna(0).tolist(),
         "headline": tau, "weighted": weighted}))

    # evidence from out-of-fold predictions on look-alike guests
    oof = pd.read_csv(svc.out / "metrics" / "oof_cate_room_nights.csv.gz")
    near = oof[(np.abs(oof["business_travel_ratio"] - profile["business_travel_ratio"]) <= 0.12) &
               (np.abs(oof["historical_adr"] - profile["historical_adr"]) <= 0.25 * profile["historical_adr"])]
    if len(near) >= 20:
        est_n, true_n = float(np.average(near[best], weights=near["w"])), float(np.average(near["tau_true"], weights=near["w"]))
        ev_status = "PASS" if abs(est_n - true_n) <= 0.5 else "WARN"
        ev_metrics = [_m("Look-alike guests", _f(len(near)), "business share ±0.12 and ADR ±25%, near the cutoff"),
                      _m(f"Estimated effect ({best})", f"{est_n:+.2f} nights", "out-of-fold predictions"),
                      _m("True effect (synthetic)", f"{true_n:+.2f} nights"), _m("Absolute error", f"{abs(est_n - true_n):.2f} nights")]
        ev_text = f"Among {len(near):,} look-alikes the model's out-of-fold average ({est_n:+.2f}) is within {abs(est_n - true_n):.2f} nights of the truth ({true_n:+.2f})."
    else:
        est_n = true_n = float("nan")
        ev_status = "WARN"
        ev_metrics = [_m("Look-alike guests", _f(len(near)))]
        ev_text = "Too few look-alike guests to check the prediction."
    stages.append(_stage("evidence", "Evidence: does the model work for guests like this?", "Is the estimate accurate for similar guests it did not train on?",
                         ev_status, ev_metrics, ev_text))

    adr = float(profile["historical_adr"])
    eco = _economics(tau, 0.0, adr, perk, margin)
    roi = eco["roi"]
    stages.append(_stage(
        "economics", "Economics: is Gold worth it for this guest?", "Predicted extra stay revenue per year against the cost of the perks",
        "PASS" if eco["net"] > 0 else "WARN",
        [_m("Extra room nights", f"{tau:+.2f}"), _m("Extra room revenue", _money(eco["room_rev"]), f"{tau:.2f} nights x {_money(adr)}"),
         _m(f"Contribution at {margin:.0%}", _money(eco["contrib"])), _m("Perk cost", _money(perk)),
         _m("Net per year", _money(eco["net"])), _m("ROI", _f(roi * 100, 0, suf="%")),
         _m("Break-even margin", _f(eco["break_even_margin"] * 100, 0, suf="%") if eco["break_even_margin"] else "n/a")],
        f"Gold adds {_money(eco['room_rev'])} of room revenue for this guest; after margin and perks the net is {_money(eco['net'])}. "
        "Ancillary and direct-booking effects are not modelled per guest.",
        {"type": "waterfall", "labels": ["Room revenue", f"Contribution ({margin:.0%})", "Perk cost", "Net"],
         "values": [eco["room_rev"], eco["contrib"], -perk, eco["net"]]}))

    if roi is not None and roi >= 0.5 and pct >= 60:
        verdict, reason = "LEAN IN", "Above-average responder and the perks clearly pay for themselves."
    elif roi is not None and roi <= -0.25:
        verdict, reason = "PULL BACK", "The expected extra stays do not cover the perk cost."
    elif roi is not None and roi >= 0.5:
        verdict, reason = "LEAN IN", "The perks pay for themselves, although this guest is not an above-average responder."
    else:
        verdict, reason = "BREAK-EVEN", "Roughly neutral: keep the standard programme and avoid costly extras."
    summary = {"verdict": verdict, "reason": reason, "metrics": [
        _m("Predicted effect", f"{tau:+.2f} nights"), _m("Percentile", f"{pct:.0f}th"), _m("Learner spread", f"±{spread:.2f}"),
        _m("Look-alike error", f"{abs(est_n - true_n):.2f}" if not math.isnan(est_n) else "n/a"), _m("Net per year", _money(eco["net"])),
        _m("ROI", _f(roi * 100, 0, suf="%"))]}
    kpi = {"effect": tau, "percentile": pct, "spread": spread, "weighted": weighted, "net": eco["net"], "roi": roi,
           "lookalike_err": None if math.isnan(est_n) else abs(est_n - true_n), "persona": flow["persona"],
           "learner_estimates": alt.to_dict()}
    return {"stages": stages, "summary": summary, "kpi": kpi}


# ============================================================================= D : threshold policy
def policy_report(flow: dict, svc, assump: dict) -> dict:
    c, new_c = float(svc.cutoff), float(flow["new_cutoff"])
    perk, margin = float(flow["perk_cost"]), float(assump["contribution_margin"])
    sim = svc.simulate_threshold(int(new_c), "room_nights", None, perk)
    panel = svc.panel
    r = panel["qualifying_nights_t"].to_numpy(float)
    lowering = new_c < c
    mask = (r >= new_c) & (r < c) if lowering else (r >= c) & (r < new_c)
    aff = panel[mask]
    fs = float(sim["take_up_factor"])
    nm, model = svc.model("room_nights", None)
    tau_aff = np.asarray(model.predict(aff[svc.covariates].to_numpy(float))).ravel() * fs if len(aff) else np.array([])
    pop_avg = float(svc._reference("room_nights", nm).mean())
    stages: List[dict] = []

    elig = int((r >= c).sum())
    stages.append(_stage(
        "today", "Today: how many members hold Gold?", "What does the current 20-night rule produce?", "INFO",
        [_m("Members", _f(len(panel))), _m("Eligible today (20+ nights)", _f(elig)), _m("Share eligible", _f(elig / len(panel) * 100, 1, suf="%")),
         _m("Take-up when eligible", _f(float(panel.loc[r >= c, "gold_status"].mean()) * 100, 0, suf="%")),
         _m("Gold effect at the cutoff (RDD)", f"{svc.late('next_year_room_nights')['estimate']:+.2f} nights")],
        f"{_f(elig)} of {_f(len(panel))} members ({elig / len(panel):.0%}) are eligible for Gold today.",
        {"type": "nights_hist", "x": list(range(1, 46)), "n": np.bincount(r.astype(int), minlength=46)[1:46].tolist(),
         "cutoff": c, "new_cutoff": new_c}))

    stages.append(_stage(
        "candidate", "Candidate: what changes?", "Who gains or loses eligibility?", "INFO",
        [_m("Candidate threshold", f"{new_c:.0f} nights", f"today {c:.0f}"), _m("Direction", "Lower: more guests reach Gold" if lowering else "Raise: fewer guests reach Gold"),
         _m("Guests affected", _f(sim["guests_affected"])), _m("Share of members", _f(sim["share_of_members_affected"] * 100, 1, suf="%")),
         _m("Gold take-up factor", _f(fs, 2), "first-stage jump: share of newly eligible guests who actually take Gold up"),
         _m("Distance from cutoff", f"{abs(new_c - c):.0f} nights")],
        (f"{_f(sim['guests_affected'])} guests with {new_c:.0f} to {c - 1:.0f} nights would gain Gold." if lowering else
         f"{_f(sim['guests_affected'])} guests with {c:.0f} to {new_c - 1:.0f} nights would lose Gold eligibility."),
        None))

    if len(aff):
        b = aff["business_travel_ratio"].to_numpy()
        wk = aff["weekend_stay_ratio"].to_numpy()
        corp, leis = float((b >= 0.6).mean()), float((b < 0.3).mean())
        bl = float(((b >= 0.3) & (b < 0.6) & (wk >= 0.25)).mean())
        avg_tau = float(tau_aff.mean())
        avg_adr = float(aff["historical_adr"].mean())
        stages.append(_stage(
            "who", "Who: what kind of guests are they?", "Are the affected guests responsive or rigid?",
            "PASS" if avg_tau / fs >= pop_avg * 0.9 else "WARN",
            [_m("Leisure share", _f(leis * 100, 0, suf="%")), _m("Bleisure share", _f(bl * 100, 0, suf="%")), _m("Corporate share", _f(corp * 100, 0, suf="%")),
             _m("Average ADR", _money(avg_adr)), _m("Avg effect per guest", f"{avg_tau:+.2f} nights", "includes the take-up factor"),
             _m("Population average", f"{pop_avg:+.2f} nights", "effect if Gold is held")],
            f"The affected group is {leis:.0%} leisure and {corp:.0%} corporate; their average effect if they take Gold is {avg_tau / fs:+.2f} nights versus {pop_avg:+.2f} for the population.",
            {"type": "mix", "labels": ["Leisure", "Bleisure", "Corporate", "Other"], "values": [leis, bl, corp, max(0.0, 1 - leis - bl - corp)]}))
    else:
        stages.append(_stage("who", "Who: what kind of guests are they?", "Are the affected guests responsive or rigid?", "INFO",
                             [_m("Guests affected", "0")], "No guests are affected.", None))

    rev = float(sim["incremental_revenue_usd"] or 0.0)
    nights = float(sim["incremental_total"])
    stages.append(_stage(
        "impact", "Impact: nights and revenue", "How many room nights and how much revenue move?",
        "INFO",
        [_m("Incremental room nights", f"{nights:+,.0f}"), _m("Incremental revenue", _money(rev), "nights x each guest's ADR"),
         _m("Avg per affected guest", f"{sim['avg_effect_per_guest']:+.2f} nights"), _m("Model", sim["learner"])],
        f"Net room nights {'gained' if nights >= 0 else 'lost'}: {abs(nights):,.0f} (about {_money(abs(rev))} of room revenue).",
        None))

    cost_signed = float(sim["perk_cost_usd"])           # negative when the threshold is raised (a saving)
    net_rev = float(sim["net_value_usd"] or 0.0)
    net_m = rev * margin - cost_signed
    be_cost = margin * abs(rev) / (sim["guests_affected"] * fs) if sim["guests_affected"] else float("nan")
    stages.append(_stage(
        "net", "Net value: revenue, margin and perk cost", "Does the change pay once perks and margin are counted?",
        "PASS" if net_m > 0 else "WARN",
        [_m("Perk cost " + ("added" if lowering else "saved"), _money(abs(cost_signed))),
         _m("Net, revenue basis", _money(net_rev), "same number as the Threshold simulator tab"),
         _m(f"Net at {margin:.0%} margin", _money(net_m), "contribution of the revenue change minus perk cost (or plus the saving)"),
         _m("Break-even perk cost", _money(be_cost), "per member; " + ("adopt if actual cost is below this" if lowering else "adopt if actual cost is above this")),
         _m("Perk cost assumed", _money(perk))],
        f"At a {margin:.0%} margin the change is worth {_money(net_m)}; the break-even perk cost is {_money(be_cost)} per member "
        f"versus the {_money(perk)} assumed.",
        {"type": "waterfall", "labels": ["Revenue change", f"Contribution ({margin:.0%})", "Perk cost" + (" added" if lowering else " saved"), "Net"],
         "values": [rev, rev * margin, -cost_signed, net_m]}))

    grid = []
    for cc in range(12, 31):
        s = svc.simulate_threshold(cc, "room_nights", None, perk)
        grid.append({"threshold": cc, "net_m": float((s["incremental_revenue_usd"] or 0.0) * margin - s["perk_cost_usd"]), "affected": s["guests_affected"]})
    cost_grid = []
    for pc in (60, 120, 240, 360):
        s = svc.simulate_threshold(int(new_c), "room_nights", None, float(pc))
        cost_grid.append({"perk_cost": pc, "net_m": float((s["incremental_revenue_usd"] or 0.0) * margin - s["perk_cost_usd"])})
    far = abs(new_c - c) > 5
    by_c = {g["threshold"]: g["net_m"] for g in grid}
    toward = int(new_c) + (1 if lowering else -1)         # one night closer to today's cutoff
    away = int(new_c) - (1 if lowering else -1)           # one night further from it
    same_sign = sum(1 for g in cost_grid if (g["net_m"] > 0) == (net_m > 0))
    stages.append(_stage(
        "sensitivity", "Sensitivity: how fragile is the answer?", "Does the decision survive other thresholds, perk costs and extrapolation?",
        "WARN" if far or same_sign < len(cost_grid) else "PASS",
        [_m("Distance from cutoff", f"{abs(new_c - c):.0f} nights", "RDD evidence is local; more than 5 nights is extrapolation"),
         _m("Perk costs that keep the sign", f"{same_sign} of {len(cost_grid)}", "tested $60, $120, $240, $360"),
         _m(f"Net at {toward} nights (closer)", _money(by_c.get(toward)), "at the assumed margin and perk cost"),
         _m(f"Net at {away} nights (further)", _money(by_c.get(away)), "net grows with distance only because the effect is extrapolated"),
         _m("Warnings", "; ".join(sim["warnings"]) if sim["warnings"] else "none")],
        ("The candidate is more than 5 nights from the cutoff, so the effects are extrapolated and the result is directional only." if far else
         "The candidate is within the range where the cutoff evidence applies; the perk-cost grid shows where the decision would flip."),
        {"type": "policy_grid", "grid": grid, "cost_grid": cost_grid, "cutoff": c, "new_cutoff": new_c}))

    if far:
        verdict, reason = "DIRECTIONAL ONLY", "The candidate is far from 20 nights, so effects are extrapolated."
    elif net_m > 0:
        verdict, reason = "ADOPT", f"Worth {_money(net_m)} a year at a {margin:.0%} margin and {_money(perk)} perk cost."
    else:
        verdict, reason = "DO NOT CHANGE", f"Loses {_money(abs(net_m))} a year at a {margin:.0%} margin and {_money(perk)} perk cost."
    summary = {"verdict": verdict, "reason": reason, "metrics": [
        _m("Guests affected", _f(sim["guests_affected"])), _m("Incremental nights", f"{nights:+,.0f}"), _m("Incremental revenue", _money(rev)),
        _m(f"Net at {margin:.0%} margin", _money(net_m)), _m("Break-even perk cost", _money(be_cost)),
        _m("Distance from cutoff", f"{abs(new_c - c):.0f} nights")]}
    kpi = {"guests_affected": sim["guests_affected"], "nights": nights, "revenue": rev, "net_margin": net_m,
           "break_even_cost": be_cost, "distance": abs(new_c - c), "net_revenue_basis": net_rev}
    return {"stages": stages, "summary": summary, "kpi": kpi}


# ============================================================================= programme level
def programme_report(reports: List[dict], svc, assump: dict) -> dict:
    by = {r["id"]: r for r in reports}
    A1, A2, B1, B2 = (by[k]["kpi"] for k in ("A1", "A2", "B1", "B2"))
    lb = svc.leaderboards["room_nights"]
    ok = lb[lb["status"] == "ok"].sort_values("pehe_rmse")
    best = ok.iloc[0]
    clean_pass = [A1["gates_passed"] == 3, A2["gates_passed"] == 3, B1["gates_passed"] == 3]
    meta = svc.meta
    margin, perk = assump["contribution_margin"], assump["perk_cost_per_member"]
    groups = [
        {"title": "Data and design", "metrics": [
            _m("Members", _f(meta["n_panel"])), _m("Design (headline)", meta["regime"]), _m("Gold take-up jump", _f(A1["first_stage"], 2)),
            _m("RDD bandwidth", f"±{meta['bandwidths']['used']:.1f} nights"), _m("Window for CATE learners", f"±{meta['cate_bandwidth']:.0f} nights")]},
        {"title": "Causal estimates (A1 fuzzy, A2 sharp)", "metrics": [
            _m(OC_LABEL[oc], f"{_fmt_oc(oc, v['estimate'], True)} vs true {_fmt_oc(oc, v['truth'], True)}",
               f"error {v['err_se']:+.1f} SE; CI {'covers' if v['covered'] else 'misses'} the truth") for oc, v in A1["per_outcome"].items()] +
            [_m("Sharp design, room nights", f"{_fmt_oc(OUTCOMES[0], A2['headline_effect'], True)} vs true {_fmt_oc(OUTCOMES[0], A2['true_effect'], True)}")]},
        {"title": "Validity suite", "metrics": [
            _m("Clean flows passing all gates", f"{sum(clean_pass)} of 3", "A1, A2, B1"),
            _m("Gamed threshold flagged (B2)", "yes" if B2["gates_passed"] < 3 and B2["density_p"] < 0.05 else "NO", f"density p = {B2['density_p']:.1e}"),
            _m("Placebo draw significant (B1, seed 42)", "yes (false positive)" if B1["significant"] else "no", f"estimate {_fmt_oc(OUTCOMES[0], B1['headline_effect'], True)} nights"),
            _m("CIs covering truth (A1, A2, B1)", f"{A1['ci_covers_truth'] + A2['ci_covers_truth'] + B1['ci_covers_truth']} of 9"),
            _m("Placebo false-positive rate", _f(B1["replication"]["reject_zero"] * 100, 0, suf="%"), f"{B1['replication']['k']} re-draws, nominal 5%"),
            _m("Gaming detection rate", _f(B2["replication"]["density_flag_rate"] * 100, 0, suf="%"),
               f"false alarm on clean data {B2['replication']['clean_density_flag_rate']:.0%}")]},
        {"title": "Heterogeneity models (room nights, out-of-fold)", "metrics": [
            _m("Best learner by RMSE", f"{best['learner']}", f"RMSE {best['pehe_rmse']:.2f} nights"),
            _m("Median correlation with truth", f"{ok['pearson'].median():.2f}"),
            _m("Learners with corr ≥ 0.85", f"{int((ok['pearson'] >= 0.85).sum())} of {len(ok)}"),
            _m("Median average-effect error", f"{(ok['ate_hat'] - ok['ate_true']).abs().median():.2f} nights", "median |estimated - true average|"),
            _m("Best uplift@10%", f"{ok['uplift_at_10'].max():+.2f} nights", "mean effect among the top-10% targeted"),
            _m("Best AUUC over random", f"{ok['auuc_over_random'].max():.3f}")]},
        {"title": "Decisions and money", "metrics": [
            _m("Net per Gold member (A1)", _money(A1["net_per_member"]), f"{margin:.0%} margin, {_money(perk)} perks"),
            _m("Net per member (A2 sharp)", _money(A2["net_per_member"])),
            _m("C1 vs C2 effect gap", f"{by['C1']['kpi']['effect'] - by['C2']['kpi']['effect']:+.2f} nights", "bleisure minus corporate"),
            _m("D1 lower to 17", _money(by["D1"]["kpi"]["net_margin"]), "net at assumed margin"),
            _m("D2 raise to 24", _money(by["D2"]["kpi"]["net_margin"]), "net at assumed margin")]},
        {"title": "Compute", "metrics": [
            _m("Training time", f"{meta['train_seconds'] / 60:.1f} min"), _m("GPU", str(meta["hardware"].get("gpu_name") or "CPU")),
            _m("Learners trained", str(len(ok)))]},
    ]
    compare = []
    for f in FLOWS:
        rep = by[f["id"]]
        k = rep["kpi"]
        if f["type"] in ("A", "B"):
            headline = (f"{k['headline_effect']:+.2f} nights (true {k['true_effect']:+.2f}); validity gates {k['gates_passed']} of 3; "
                        f"net {_money(k['net_per_member'])} per member")
        elif f["type"] == "C":
            headline = f"{k['effect']:+.2f} nights, {k['percentile']:.0f}th percentile; net {_money(k['net'])} per year"
        else:
            headline = f"{k['guests_affected']:,} guests, {k['nights']:+,.0f} nights; net {_money(k['net_margin'])} per year at assumed margin"
        compare.append({"Flow": f["short"], "Type": FLOW_TYPES[f["type"]]["title"], "Verdict": rep["summary"]["verdict"], "Headline": headline})
    models = {"personas": {by[k]["persona"]: by[k]["kpi"]["learner_estimates"] for k in ("C1", "C2")},
              "corr": ok.set_index("learner")["pearson"].to_dict()}
    return {"groups": groups, "compare": compare, "models": models}


# ============================================================================= orchestration
def build_flow(flow: dict, svc, assump: dict) -> dict:
    kind = FLOW_TYPES[flow["type"]]["kind"]
    body = {"panel": panel_report, "cate": cate_report, "policy": policy_report}[kind](flow, svc, assump)
    return {"id": flow["id"], "type": flow["type"], "short": flow["short"], "title": flow["title"], "story": flow["story"],
            "persona": flow.get("persona"), "kind": kind, **body}


def build_all(svc) -> dict:
    assump = load_assumptions()
    reports = [build_flow(f, svc, assump) for f in FLOWS]
    return _py({"assumptions": assump, "flow_types": FLOW_TYPES, "flows": reports,
                "programme": programme_report(reports, svc, assump)})


def save(report: dict, outputs: Path) -> Path:
    p = Path(outputs) / "metrics" / REPORT_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return p


def load(outputs: Path) -> Optional[dict]:
    p = Path(outputs) / "metrics" / REPORT_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_and_save(outputs: Path) -> Path:
    from src.api.service import ModelService
    return save(build_all(ModelService(outputs)), Path(outputs))


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    a = ap.parse_args()
    t0 = time.time()
    path = build_and_save(Path(a.outputs))
    rep = json.loads(path.read_text(encoding="utf-8"))
    for f in rep["flows"]:
        print(f"{f['id']}  {f['summary']['verdict']:<34} {f['short']}")
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    from src.utils.device import hard_exit
    hard_exit(0)
