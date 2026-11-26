"""End-to-end scenario flows (UI-independent). Every flow is a list of stages, each stage carrying its own metrics.

Four flow types, two worked examples each (8 flows):

  A  Holdout read-out     - was branded search incremental, and was it worth the money?
                            A1 all 10 holdout DMAs | A2 the two Tier-1 hubs only (thin evidence)
  B  Validity / refutation - can the estimates be trusted?
                            B1 direct bookings | B2 OTA bookings
  C  DMA what-if (CATE)   - what does pausing do in ONE market?
                            C1 Miami Beach flagship (Tier 1) | C2 Asheville drive-to (Tier 3)
  D  Bidding policy       - which DMAs should pause, and does the rule pay?
                            D1 evidence-margin rule (k=1, fixed a priori) | D2 naive rule (k=0)

`build_all()` computes everything once from the same code the API uses and writes `outputs/metrics/flows.json`; the
Streamlit tab, the FastAPI `/flows` endpoints and the README all read that file, so every surface shows identical numbers.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.api import service
from src.api.presets import PRESETS
from src.data import feature_engineering as fe
from src.data.synthetic_geo_generator import generate_panel, load_panel
from src.economics import cannibalization_roi as econ
from src.models import macro_geolift as mg
from src.utils.config import path

log = logging.getLogger(__name__)
ALPHA = 0.10                      # smallest p reachable depends on the number of holdout DMAs (2 pairs -> 0.5)
BSTS_FAST = dict(num_warmup=300, num_samples=400)
POLICY_KS = (0, 0.5, 1, 1.5, 2, 3)

FLOW_TYPES = {
    "A": {"title": "Holdout read-out", "question": "Was branded paid search incremental, and was it worth its spend?"},
    "B": {"title": "Validity & refutation", "question": "Can we trust these estimates, and which estimator should we believe?"},
    "C": {"title": "DMA what-if (CATE)", "question": "What does pausing branded search do in one specific market?"},
    "D": {"title": "Bidding policy", "question": "Which DMAs should pause, and does the rule actually pay?"},
}
FLOWS = [
    {"id": "A1", "type": "A", "short": "A1 - Full holdout (10 DMAs)", "title": "Programme read-out: all 10 holdout DMAs",
     "tiers": [1, 2, 3],
     "story": "The headline question: 4 weeks with branded paid search paused in 10 stratified-random DMAs (2 hubs, 4 regional, 4 drive-to), "
              "against 40 controls. All three macro estimators, then dollars."},
    {"id": "A2", "type": "A", "short": "A2 - Tier-1 hubs only (2 DMAs)", "title": "Read-out for the two Tier-1 hub holdouts",
     "tiers": [1],
     "story": "Same experiment, but only the two major hubs, the markets where OTA conquesting is fiercest. Two treated DMAs can "
              "never produce a permutation p below 0.5, so the stage metrics show what thin evidence looks like."},
    {"id": "B1", "type": "B", "short": "B1 - Validity: direct bookings", "title": "Refutation suite on direct bookings",
     "outcome": "direct_bookings",
     "story": "Placebo-in-time, placebo-in-space and a Monte-Carlo over fresh experiments, to decide which of MMT, SDID and BSTS to believe."},
    {"id": "B2", "type": "B", "short": "B2 - Validity: OTA bookings", "title": "Refutation suite on OTA bookings",
     "outcome": "ota_bookings",
     "story": "The same suite on the OTA-diversion outcome. OTA volume is smaller and noisier, so the placebo bands are wider."},
    {"id": "C1", "type": "C", "short": "C1 - Miami Beach flagship", "title": "What if Miami Beach (Tier 1) paused?",
     "preset": "A",
     "story": "Heavy OTA conquesting, modest organic cannibalization: the profile where pausing is most likely to leak bookings to OTAs."},
    {"id": "C2", "type": "C", "short": "C2 - Asheville drive-to", "title": "What if Asheville (Tier 3) paused?",
     "preset": "C",
     "story": "Top organic rank and almost no OTA pressure: nearly every branded click would have arrived for free."},
    {"id": "D1", "type": "D", "short": "D1 - Evidence-margin rule (k=1)", "title": "Pause iff gain > 1 x learner disagreement",
     "k": 1.0,
     "story": "The default policy, with its margin fixed before looking at the truth: pause a DMA only when the nine learners' mean gain "
              "exceeds one standard deviation of their disagreement."},
    {"id": "D2", "type": "D", "short": "D2 - Naive rule (k=0)", "title": "Pause iff the mean prediction is positive",
     "k": 0.0,
     "story": "The tempting rule: trust the ensemble mean with no margin. Noisy DMA-level CATEs turn into confident-looking mistakes."},
]
FLOW_BY_ID = {f["id"]: f for f in FLOWS}


# ----------------------------------------------------------------------------- helpers
def _num(x):
    return x is not None and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))


def _f(x, d=0, pre="", suf=""):
    if not _num(x):
        return "n/a"
    s = f"{abs(x):,.{d}f}"
    return f"{'-' if x < 0 else ''}{pre}{s}{suf}"


def _usd(x, d=0):
    return _f(x, d, "$")


def _m(label, value, help=None):
    return {"label": label, "value": value, "help": help}


def _stage(key, title, metrics, note="", chart=None, flag=None):
    return {"key": key, "title": title, "metrics": metrics, "note": note, "chart": chart, "flag": flag}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _rj(name):
    return json.loads((path("metrics") / name).read_text())


def _csv(name, **kw):
    return pd.read_csv(path("metrics") / name, **kw)


def _hist(vals, marks: dict, bins=24):
    vals = np.asarray(vals, float)
    lo, hi = min(vals.min(), *marks.values()) - 0.2, max(vals.max(), *marks.values()) + 0.2
    cnt, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    return {"kind": "hist", "x": [round(float((a + b) / 2), 2) for a, b in zip(edges[:-1], edges[1:])],
            "y": cnt.tolist(), "marks": {k: float(v) for k, v in marks.items()}}


# ----------------------------------------------------------------------------- A: holdout read-out
def _estimates(gp, pairs, sub, outcome, reuse_saved):
    """MMT / SDID / BSTS on the treated subset `sub`; the full holdout reuses the headline results saved by the pipeline."""
    gd = mg.GeoData(gp.panel, gp.dma, outcome)
    cut = gd.T - 28
    ids = gp.dma.dma_id.values[sub]
    ps = pairs[pairs.treated_dma_id.isin(ids)]
    tr = gp.dma[gp.dma.dma_id.isin(ids)]
    truth = float(tr.true_post_tau_direct.mean() if outcome == "direct_bookings" else tr.true_post_tau_ota.mean())
    if reuse_saved:
        mac = _rj("macro_results.json")[outcome]
        ser = _csv("macro_series.csv")
        ser = ser[ser.outcome == outcome]
        res = {k: dict(mac[k]) for k in ("MMT", "SDID", "BSTS")}
        cf = {k: ser[ser.method == k].set_index("day").counterfactual.values for k in res}
        obs = ser[ser.method == "SDID"].set_index("day").observed.values
        return gd, cut, ps, truth, res, cf, obs
    r = {"MMT": mg.mmt(gd, ps, cut),
         "SDID": mg.sdid(gd, cut, treated=sub, n_placebo=150), "BSTS": mg.bsts(gd, cut, treated=sub, **BSTS_FAST)}
    res = {k: {kk: v for kk, v in x.items() if np.ndim(v) == 0} for k, x in r.items()}
    for k in res:
        res[k]["covers_truth"] = bool(res[k]["ci_low"] <= truth <= res[k]["ci_high"])
    res["MMT"]["p_perm"] = float(r["MMT"]["p_perm"])
    cf = {k: r[k]["counterfactual"] for k in r}
    return gd, cut, ps, truth, res, cf, r["SDID"]["observed"]


def _prefit(obs, cf, cut):
    """Pre-period tracking error of a counterfactual, as % of the treated mean."""
    return float(np.sqrt(np.mean((obs[:cut] - cf[:cut]) ** 2)) / obs[:cut].mean() * 100)


def panel_flow(flow, gp, pairs, reuse_saved):
    dma = gp.dma
    sub = ((dma["T"] == 1) & dma.tier.isin(flow["tiers"])).values
    n = int(sub.sum())
    dsub = dma[sub]
    gd_d, cut, ps, t_d, rd, cf_d, obs_d = _estimates(gp, pairs, sub, "direct_bookings", reuse_saved)
    _, _, _, t_o, ro, _, _ = _estimates(gp, pairs, sub, "ota_bookings", reuse_saved)
    tiers = dsub.tier.value_counts().sort_index()
    spend_day = float(gp.panel[(gp.panel.post == 0) & gp.panel.dma_id.isin(dsub.dma_id)].groupby("dma_id").branded_spend.mean().sum())
    pmin = 2.0 / 2 ** n

    # placebo-in-time (pre-data only, 4 weeks early) for the direct outcome
    pit_cut, pit_end = cut - 28, cut
    pm = mg.mmt(gd_d, ps, pit_cut, pit_end)
    ps_ = mg.sdid(gd_d, pit_cut, pit_end, n_placebo=100, treated=sub)
    drift = max(abs(pm["att"]), abs(ps_["att"]))
    sdid_cov0 = ps_["ci_low"] <= 0 <= ps_["ci_high"]

    # economics, headline = SDID
    d2 = dma.copy()
    d2["T"] = sub.astype(int)
    h, ho = rd["SDID"], ro["SDID"]
    e = econ.programme_economics(gp.panel, d2, h["att"], ho["att"], (h["ci_low"], h["ci_high"]), (ho["ci_low"], ho["ci_high"]), "SDID")
    et = econ.programme_economics(gp.panel, d2, t_d, t_o, label="Truth")
    nv, nv_rng = e["net_value_of_branded_search_usd"], [-x for x in e["net_margin_change_if_paused_usd_range"]][::-1]
    sig = h["ci_high"] < 0 and (h.get("p_value") or 1) <= ALPHA
    if pmin > ALPHA:
        verdict = "INCONCLUSIVE: TOO FEW HOLDOUT DMAS"
        action = (f"With {n} treated DMAs the exact permutation test cannot go below p = {pmin:.2f}, and the three estimators disagree "
                  f"({min(rd[k]['att'] for k in ('MMT', 'SDID', 'BSTS')):.1f} to {max(rd[k]['att'] for k in ('MMT', 'SDID', 'BSTS')):.1f} per DMA-day). "
                  f"The point estimate leans KEEP ({_usd(nv)}); the simulated truth, {_usd(et['net_value_of_branded_search_usd'])}, shows why not to act on it.")
    elif sig and nv > 0 and min(nv_rng) > 0:
        verdict, action = "KEEP BRANDED SEARCH", "Direct loss is significant and the whole interval of net value is positive."
    elif sig and nv > 0:
        verdict, action = "KEEP (LEAN), RE-TEST", "The direct loss is significant, but net value's interval spans zero: cannibalization is too uncertain to be decisive."
    elif sig:
        verdict, action = "PAUSE", "Significant effect but pausing nets positive margin."
    else:
        verdict, action = "INCONCLUSIVE", "The direct-bookings loss is not distinguishable from zero with this many holdout DMAs; extend the test or add holdouts."

    bias = lambda r, t: (r["att"] - t) / abs(t) * 100 if t else float("nan")
    est_rows = [(k, rd[k]) for k in ("MMT", "SDID", "BSTS")]
    ser_n = len(obs_d)
    smooth = lambda a: pd.Series(a).rolling(7, min_periods=1).mean().round(3).tolist()
    chart_ts = {"kind": "line", "x": list(range(ser_n)), "cut": cut, "title": "Direct bookings per holdout DMA-day (7-day mean)",
                "series": {"observed": smooth(obs_d), "MMT counterfactual": smooth(cf_d["MMT"]),
                           "SDID counterfactual": smooth(cf_d["SDID"]), "BSTS counterfactual": smooth(cf_d["BSTS"])}}
    chart_roas = {"kind": "bar", "title": "ROAS: what last-click credits vs what is incremental",
                  "x": ["Last-click ROAS", "iROAS (SDID)", "iROAS (truth)", "iROAS incl. OTA diversion"],
                  "y": [e["last_click_roas"], e["iroas"], et["iroas"], e["iroas_incl_ota_diversion"]]}
    stages = [
        _stage("design", "1. Experiment design", [
            _m("Holdout DMAs", f"{n}", "branded SEM paused for 4 weeks"), _m("Tier mix", " / ".join(f"T{t}:{c}" for t, c in tiers.items())),
            _m("Matched pairs", f"{len(ps)}", "DTW + Mahalanobis, Hungarian one-to-one"),
            _m("Mean DTW / Mahalanobis", f"{ps.dtw.mean():.2f} / {ps.mahalanobis.mean():.2f}", "lower = closer pre-period behaviour and covariates"),
            _m("Branded spend (4 wk)", _usd(spend_day * 28)), _m("Smallest possible exact p", _f(pmin, 3), "2 / 2^pairs: the floor for the sign-flip test")],
            f"Truth (known, simulated): direct {_f(t_d, 2)} and OTA {_f(t_o, 2)} bookings per DMA-day from pausing."),
        _stage("prefit", "2. Pre-period fit & placebo-in-time", [
            _m("MMT pre-fit error", _f(_prefit(obs_d, cf_d["MMT"], cut), 1, suf="%"), "RMSE of pre-period counterfactual vs observed, % of mean"),
            _m("SDID pre-fit error", _f(_prefit(obs_d, cf_d["SDID"], cut), 1, suf="%")),
            _m("Placebo-in-time MMT", _f(pm["att"], 2), f"CI {_f(pm['ci_low'], 2)} to {_f(pm['ci_high'], 2)} (truth is 0)"),
            _m("Placebo-in-time SDID", _f(ps_["att"], 2), f"CI {_f(ps_['ci_low'], 2)} to {_f(ps_['ci_high'], 2)} (truth is 0)"),
            _m("Worst placebo drift", f"{drift:.2f}", "bookings per DMA-day of effect found where none exists")],
            "A counterfactual that finds an effect 4 weeks before the pause is mis-calibrated by about that much.",
            chart_ts, "pass" if sdid_cov0 and pm["ci_low"] <= 0 <= pm["ci_high"] and drift < 0.5 * abs(t_d) else "warn"),
        _stage("mmt", "3. Matched-market test (MMT)", [
            _m("Direct ATT / DMA-day", _f(rd["MMT"]["att"], 2), f"95% CI {_f(rd['MMT']['ci_low'], 2)} to {_f(rd['MMT']['ci_high'], 2)}"),
            _m("Exact permutation p", _f(rd["MMT"].get("p_perm"), 3), "paired sign-flip, two-sided"),
            _m("Bias vs truth", _f(bias(rd["MMT"], t_d), 0, suf="%")), _m("OTA ATT / DMA-day", _f(ro["MMT"]["att"], 2), f"truth {_f(t_o, 2)}")],
            "Ratio-adjusted matched control; no model, so it inherits any pre-period drift between a pair.", None,
            "pass" if rd["MMT"]["covers_truth"] else "fail"),
        _stage("sdid", "4. Synthetic DiD (SDID)", [
            _m("Direct ATT / DMA-day", _f(h["att"], 2), f"95% CI {_f(h['ci_low'], 2)} to {_f(h['ci_high'], 2)}"),
            _m("Placebo-space p", _f(h.get("p_value"), 3), "re-estimate after fake-treating controls"),
            _m("Bias vs truth", _f(bias(h, t_d), 0, suf="%")), _m("OTA ATT / DMA-day", _f(ho["att"], 2), f"truth {_f(t_o, 2)}")],
            "Re-weights donors on their pre-period trend and re-weights pre-period days; the headline estimator.", None,
            "pass" if h["covers_truth"] else "fail"),
        _stage("bsts", "5. Bayesian structural time series (BSTS)", [
            _m("Direct ATT / DMA-day", _f(rd["BSTS"]["att"], 2), f"95% CrI {_f(rd['BSTS']['ci_low'], 2)} to {_f(rd['BSTS']['ci_high'], 2)}"),
            _m("P(effect < 0)", _f(rd["BSTS"].get("prob_negative"), 2)), _m("Max R-hat", _f(rd["BSTS"].get("rhat_max"), 3), "< 1.05 = chains agree"),
            _m("Bias vs truth", _f(bias(rd["BSTS"], t_d), 0, suf="%")), _m("OTA ATT / DMA-day", _f(ro["BSTS"]["att"], 2), f"truth {_f(t_o, 2)}")],
            "Forecasts the counterfactual from control-DMA principal components; shrinks toward zero here (see flow B1).", None,
            "pass" if rd["BSTS"]["covers_truth"] and abs(bias(rd["BSTS"], t_d)) < 40 else "warn"),
        _stage("econ", "6. Economics (headline: SDID)", [
            _m("Last-click ROAS", _f(e["last_click_roas"], 1, suf="x")),
            _m("Incremental ROAS", _f(e["iroas"], 1, suf="x"), f"truth {_f(et['iroas'], 1, suf='x')}"),
            _m("Cannibalization", _f(e["cannibalization_rate"] * 100, 0, suf="%"),
               f"range {e['cannibalization_rate_range'][0]:.0%} to {e['cannibalization_rate_range'][1]:.0%}; truth {et['cannibalization_rate']:.0%}"),
            _m("OTA commission leakage", _usd(e["ota_commission_leakage_usd"]), "commission paid on bookings diverted to OTAs"),
            _m("Branded spend avoided", _usd(e["branded_spend_avoided_usd"])),
            _m("Net value of keeping SEM", _usd(nv), f"interval {_usd(nv_rng[0])} to {_usd(nv_rng[1])}; truth {_usd(et['net_value_of_branded_search_usd'])}")],
            "Net contribution margin = direct GBV - SEM spend + OTA GBV x (1 - commission). Value of branded search = -(effect of pausing).",
            chart_roas, "pass" if min(nv_rng) > 0 else "warn"),
        _stage("verdict", "7. Verdict", [
            _m("Decision", verdict), _m("Direct loss significant?", "yes" if sig else "no"),
            _m("Net value of keeping", _usd(nv)), _m("iROAS vs last-click", f"{e['iroas']:.1f}x vs {e['last_click_roas']:.1f}x")], action, None,
            "pass" if verdict.startswith("KEEP BR") else "warn"),
    ]
    flow_metrics = [_m("Net value of keeping SEM (4 wk)", _usd(nv)), _m("iROAS (vs last-click)", f"{e['iroas']:.1f}x ({e['last_click_roas']:.1f}x)"),
                    _m("Cannibalization", f"{e['cannibalization_rate']:.0%}"), _m("Direct ATT (SDID)", _f(h["att"], 2)),
                    _m("Truth", _f(t_d, 2)), _m("Smallest possible p", _f(pmin, 3))]
    return stages, verdict, action, flow_metrics, {"direct_att": h["att"], "truth": t_d, "net_value_keep_usd": nv, "iroas": e["iroas"]}


# ----------------------------------------------------------------------------- B: validity
def _mc_ota(n=10):
    rows = []
    for sd in range(1000, 1000 + n):
        g = generate_panel(seed=sd)
        p, _ = fe.match_markets(g.panel, g.dma)
        gd = mg.GeoData(g.panel, g.dma, "ota_bookings")
        cut = gd.T - 28
        t = float(g.dma[g.dma["T"] == 1].true_post_tau_ota.mean())
        for k, r in (("MMT", mg.mmt(gd, p, cut)), ("SDID", mg.sdid(gd, cut, n_placebo=60))):
            rows.append({"seed": sd, "method": k, "truth": t, "est": r["att"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    df = pd.DataFrame(rows)
    df["err"], df["covered"] = df.est - df.truth, (df.ci_low <= df.truth) & (df.truth <= df.ci_high)
    df.to_csv(path("metrics") / "macro_montecarlo_ota.csv", index=False)
    return df


def validity_flow(flow, gp):
    oc = flow["outcome"]
    lab = "direct" if oc == "direct_bookings" else "OTA"
    mac = _rj("macro_results.json")[oc]
    truth = mac["truth"]
    pit = _csv("placebo_in_time.csv")
    pit = pit[pit.outcome == oc].set_index("method")
    sp = _csv(f"placebo_space_{oc}.csv")
    mcp = path("metrics") / "macro_montecarlo_ota.csv"
    mc = _csv("macro_montecarlo.csv") if oc == "direct_bookings" else (_csv("macro_montecarlo_ota.csv") if mcp.exists() else _mc_ota())
    mcs = mc.groupby("method").agg(bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt((e ** 2).mean()))), cov=("covered", "mean"), n=("seed", "count"))
    methods = [m for m in ("MMT", "SDID", "BSTS") if m in mac]
    pl = sp.placebo_att.values
    sd = float(pl.std(ddof=1))
    rank_p = {m: float(((np.abs(pl) >= abs(mac[m]["att"])).sum() + 1) / (len(pl) + 1)) for m in ("MMT", "SDID")}
    tru = abs(truth)
    status = {}
    for m in methods:
        why = []
        if m in pit.index and not bool(pit.loc[m, "covers_zero"]):
            why.append("placebo-in-time finds a false effect")
        if m in mcs.index and abs(mcs.loc[m, "bias"]) > 0.4 * tru:
            why.append(f"Monte-Carlo bias {mcs.loc[m, 'bias']:+.2f} is large vs truth {truth:.2f}")
        if m in mcs.index and mcs.loc[m, "cov"] < 0.7:
            why.append(f"CI coverage only {mcs.loc[m, 'cov']:.0%}")
        if m not in mcs.index:
            hb = abs(mac[m]["att"] - truth)
            why.append(f"no Monte-Carlo run; headline-seed error {mac[m]['att'] - truth:+.2f} ({hb / tru:.0%} of truth)" if hb > 0.4 * tru else "not Monte-Carlo tested")
            status[m] = ("CAUTION" if hb > 0.4 * tru else "UNVERIFIED", why)
            continue
        status[m] = ("FAIL" if len(why) >= 2 or (why and "Monte-Carlo bias" in why[0]) else "CAUTION" if why else "PASS", why)
    passing = [m for m in methods if status[m][0] == "PASS"]
    verdict = ("TRUST " + " + ".join(passing)) if passing else "NO ESTIMATOR PASSES"
    action = "; ".join(f"{m}: {status[m][0]}" + (f" ({'; '.join(status[m][1])})" if status[m][1] else "") for m in methods)
    chart_hist = _hist(pl, {"SDID estimate": mac["SDID"]["att"], "MMT estimate": mac["MMT"]["att"], "truth": truth})
    chart_mc = {"kind": "bar", "title": f"Monte-Carlo bias in {lab} bookings per DMA-day (estimate - truth)", "x": list(mcs.index), "y": [float(v) for v in mcs.bias]}
    stages = [
        _stage("scope", "1. What is being refuted", [
            _m("Outcome", f"{lab} bookings / DMA-day"), _m("Truth", _f(truth, 2)),
            _m("Estimators on trial", " / ".join(methods)), _m("Refutations", "placebo-time, placebo-space, Monte-Carlo")],
            "Each estimator must find nothing where nothing happened, and must be roughly unbiased across fresh experiments."),
        _stage("pit", "2. Placebo-in-time (pause 4 weeks early, pre-data only)",
               [_m(f"{m} placebo ATT", _f(pit.loc[m, "placebo_att"], 2), f"CI {_f(pit.loc[m, 'ci_low'], 2)} to {_f(pit.loc[m, 'ci_high'], 2)}; p {_f(pit.loc[m, 'p_value'], 3)}")
                for m in methods if m in pit.index],
               "The true placebo effect is 0. An interval excluding zero means pre-period drift the estimator cannot absorb.", None,
               "pass" if all(bool(pit.loc[m, "covers_zero"]) for m in methods if m in pit.index) else "warn"),
        _stage("pis", "3. Placebo-in-space (fake-treat control DMAs)", [
            _m("Placebo ATT sd", _f(sd, 2), f"{len(pl)} fake holdouts drawn from control DMAs"),
            _m("p (SDID estimate)", _f(rank_p["SDID"], 3), "share of placebo effects at least as extreme"),
            _m("p (MMT estimate)", _f(rank_p["MMT"], 3)), _m("Real SDID / sd", _f(mac["SDID"]["att"] / sd, 1, suf=" sd"), "signal-to-noise vs placebo spread")],
            "If the real effect sits far in the tail of the placebo distribution it is unlikely to be luck.", chart_hist,
            "pass" if rank_p["SDID"] <= ALPHA else "warn"),
        _stage("mc", "4. Monte-Carlo over fresh experiments", [
            _m(f"{m} bias / RMSE", f"{mcs.loc[m, 'bias']:+.2f} / {mcs.loc[m, 'rmse']:.2f}", f"CI coverage {mcs.loc[m, 'cov']:.0%} over {int(mcs.loc[m, 'n'])} experiments")
            for m in methods if m in mcs.index],
            "New DMAs and new randomisation each time; bias and coverage are the honest scorecard.", chart_mc,
            "pass" if all(abs(mcs.loc[m, "bias"]) < 0.4 * tru for m in methods if m in mcs.index) else "warn"),
        _stage("truth", "5. Headline seed against truth", [
            _m(f"{m} (covers truth: {'yes' if mac[m]['covers_truth'] else 'NO'})", _f(mac[m]["att"], 2), f"CI {_f(mac[m]['ci_low'], 2)} to {_f(mac[m]['ci_high'], 2)}")
            for m in methods], f"Truth {truth:.2f}. One draw proves little; stages 2-4 are what decide trust."),
        _stage("verdict", "6. Verdict", [_m("Estimator status", f"{m}: {status[m][0]}") for m in methods] + [_m("Decision", verdict)], action, None,
               "pass" if passing else "fail"),
    ]
    fm = [_m("Estimators passing", f"{len(passing)} / {len(methods)}"), _m("Placebo-space sd", _f(sd, 2)),
          *[_m(f"{m} MC bias", f"{mcs.loc[m, 'bias']:+.2f}") for m in methods if m in mcs.index]]
    return stages, verdict, action, fm, {"passing": passing, "status": {m: status[m][0] for m in methods}}


# ----------------------------------------------------------------------------- C: DMA what-if
def _preset_profile(k):
    p = PRESETS[k]
    return {x: p[x] for x in p if x != "label"} | {"flight_idx": 100.0}


def cate_flow(flow):
    k = flow["preset"]
    pre = PRESETS[k]
    prof = _preset_profile(k)
    sim_keys = ["baseline_direct_bookings_day", "ota_to_direct_ratio", "lastclick_share", "cannibalization_rate", "ota_capture_rate",
                "diversion_rate", "adr", "commission_rate", "cpc"]
    sim = service.simulate_holdout_policy({x: pre[x] for x in sim_keys})
    pr = service.predict_dma_cate(prof)
    lbd, lbo = _csv("cate_leaderboard_direct.csv", index_col=0), _csv("cate_leaderboard_ota.csv", index_col=0)
    dv, ov = pr["per_model"]["direct"], pr["per_model"]["ota"]
    names = list(dv)
    vd, vo = np.array([dv[a] for a in names]), np.array([ov[a] for a in names])
    ens_d, ens_o = pr["ensemble_models"]["direct"], pr["ensemble_models"]["ota"]
    mean_d, mean_o = float(vd.mean()), float(vo.mean())
    base_d = prof["baseline_direct_bookings_day"]
    g_d = prof["adr"] * econ.NIGHTS
    g_o = g_d * econ.OTA_ADR_FACTOR
    H = 28
    # nine-learner mean (more stable than the top-3) in dollars
    spend = base_d * prof["lastclick_share"] / econ.DEFAULT_CVR * prof["cpc"]
    d_ncm9 = mean_d * base_d * g_d + mean_o * base_d * prof["ota_to_direct_ratio"] * g_o * (1 - prof["commission_rate"]) + spend
    cate_dec = "Pause" if d_ncm9 > 0 else "Keep"
    struct_dec = "Pause" if sim["decision"].startswith("Pause") else "Keep"
    agree = cate_dec == struct_dec
    n_neg = int((vd < 0).sum())
    const_pehe = float(lbd.loc["Constant ATE (DiD)", "pehe_norm"])
    ens_pehe = float(np.mean([lbd.loc[a, "pehe_norm"] for a in ens_d]))
    ens_rank = float(np.mean([lbd.loc[a, "rank_corr"] for a in ens_d]))
    reliable = ens_pehe < const_pehe
    if agree:
        verdict = f"{struct_dec.upper()} BRANDED SEARCH"
        action = "The structural simulator and the nine CATE learners agree."
    else:
        verdict = f"{struct_dec.upper()} (CATE DISAGREES)"
        action = "CATE learners and the structural model disagree. DMA-level CATE is noisy (it does not beat the constant-effect reference), so follow the experiment-calibrated structural answer and re-test."
    chart_models = {"kind": "bar", "title": "Direct-booking lift if paused, by learner (%)", "x": names, "y": [float(v * 100) for v in vd]}
    chart_econ = {"kind": "bar", "title": f"{H}-day net margin change if paused ($)", "x": ["Structural simulation", "CATE (9 learners)"],
                  "y": [sim["net_margin_change_if_paused_usd"], d_ncm9 * H]}
    stages = [
        _stage("profile", "1. DMA profile", [
            _m("Market", pre["label"].split(" (")[0]), _m("Tier", f"{pre['tier']}"), _m("Baseline direct bookings / day", f"{base_d:.0f}"),
            _m("OTA : direct ratio", f"{prof['ota_to_direct_ratio']:.2f}"), _m("OTA auction pressure", f"{prof['ota_pressure']:.2f}"),
            _m("Organic SERP rank", f"{prof['organic_rank']:.1f}", "1 = top of the free listings"), _m("ADR / commission", f"${prof['adr']:.0f} / {prof['commission_rate']:.1%}")],
            "Inputs come from the tier-calibrated preset; nothing here is a measured DMA."),
        _stage("sim", "2. Structural simulation (experiment-calibrated)", [
            _m("Last-click ROAS", _f(sim["last_click_roas"], 1, suf="x")), _m("Incremental ROAS", _f(sim["iroas"], 1, suf="x")),
            _m("Cannibalization", f"{sim['cannibalization_rate']:.0%}"), _m("Break-even cannibalization", f"{sim['break_even_cannibalization']:.0%}", "pausing pays above this"),
            _m(f"Net margin change if paused ({H}d)", _usd(sim["net_margin_change_if_paused_usd"])), _m("OTA commission leakage", _usd(sim["ota_commission_leakage_usd"]))],
            "Closed-form: direct loss = lastclick x [(1-c) + c.q]; OTA gain = lastclick x [(1-c).v + c.q].", None, "pass"),
        _stage("cate", "3. Nine CATE learners (relative lift if paused)", [
            _m("Direct lift, mean of 9", _f(mean_d * 100, 1, suf="%"), f"range {vd.min() * 100:.1f}% to {vd.max() * 100:.1f}%"),
            _m("OTA lift, mean of 9", _f(mean_o * 100, 1, suf="%"), f"range {vo.min() * 100:.1f}% to {vo.max() * 100:.1f}%"),
            _m("Learners predicting a direct loss", f"{n_neg} / {len(vd)}"), _m("Top-3 direct lift", _f(pr["direct_lift_pct"], 1, suf="%"), ", ".join(ens_d)),
            _m("Learner disagreement (sd)", _f(vd.std() * 100, 1, suf=" pts"))],
            "Each learner predicts relative lift in bookings; spread across learners is itself an uncertainty signal.", chart_models,
            "pass" if n_neg == len(vd) else "warn"),
        _stage("reliab", "4. How reliable are DMA-level CATEs?", [
            _m("Top-3 PEHE / sd(true CATE)", _f(ens_pehe, 2), "1.0 = no better than knowing nothing about heterogeneity"),
            _m("Constant-ATE reference", _f(const_pehe, 2)), _m("Top-3 rank correlation", _f(ens_rank, 2), "Spearman of predicted vs true CATE across DMAs"),
            _m("Best learner (direct)", lbd.drop("Constant ATE (DiD)").pehe_norm.idxmin())],
            "A learner that does not beat the constant-effect reference should not drive a single-DMA decision." if not reliable else "The ensemble beats the constant-effect reference.",
            None, "pass" if reliable else "warn"),
        _stage("econ", f"5. Dollars over {H} days (CATE route)", [
            _m("Direct bookings change / day", _f(mean_d * base_d, 1)), _m("OTA bookings change / day", _f(mean_o * base_d * prof["ota_to_direct_ratio"], 1)),
            _m("Branded spend saved", _usd(spend * H)), _m("Net margin change if paused", _usd(d_ncm9 * H)),
            _m("OTA commission leakage", _usd(mean_o * base_d * prof["ota_to_direct_ratio"] * g_o * prof["commission_rate"] * H))],
            "Same dollar bridge as the programme read-out, fed by learner predictions instead of an experiment ATT.", chart_econ),
        _stage("verdict", "6. Verdict", [_m("Structural", struct_dec), _m("CATE (9 learners)", cate_dec), _m("Agreement", "yes" if agree else "NO"), _m("Decision", verdict)],
               action, None, "pass" if agree else "warn"),
    ]
    fm = [_m("Structural net change if paused", _usd(sim["net_margin_change_if_paused_usd"])), _m("CATE net change if paused", _usd(d_ncm9 * H)),
          _m("iROAS (last-click)", f"{sim['iroas']:.1f}x ({sim['last_click_roas']:.1f}x)"), _m("Agreement", "yes" if agree else "no")]
    return stages, verdict, action, fm, {"structural_usd": sim["net_margin_change_if_paused_usd"], "cate_usd": d_ncm9 * H, "agree": agree}


# ----------------------------------------------------------------------------- D: policy
def policy_flow(flow, gp):
    k = flow["k"]
    dma = gp.dma
    oof = {t: _csv(f"cate_oof_predictions_{t}.csv") for t in ("direct", "ota")}
    algos = [c.replace("tau_hat_", "") for c in oof["direct"].columns if c.startswith("tau_hat_")]
    mats = {t: oof[t].groupby("dma_id")[[f"tau_hat_{a}" for a in algos]].mean().reindex(dma.dma_id).values for t in oof}
    G = np.stack([econ.gain_per_day(dma, mats["direct"][:, j], mats["ota"][:, j]) for j in range(len(algos))], 1)
    mean, sd = G.mean(1), G.std(1)
    pause = (mean - k * sd) > 0
    true28 = dma.true_post_tau_ncm.values * 28
    oracle = true28 > 0
    value = float(true28[pause].sum())
    orc = float(true28[oracle].sum())
    always = float(true28.sum())
    false_p = pause & ~oracle
    missed = ~pause & oracle
    unanimous = float(((G > 0).all(1) | (G < 0).all(1)).mean())
    rules = {kk: float(true28[(mean - kk * sd) > 0].sum()) for kk in POLICY_KS}
    by_tier = pd.DataFrame({"tier": dma.tier.values, "pause": pause, "val": true28, "orc": oracle})
    tier_rows = by_tier.groupby("tier").apply(lambda d: pd.Series({"n": len(d), "paused": int(d.pause.sum()), "oracle": int(d.orc.sum()),
                                                                    "value": float(d.val[d.pause].sum()), "all": float(d.val.sum())}), include_groups=False)
    if value > 0 and value >= 0.25 * orc:
        verdict, action = "PILOT THE RULE", "Positive realised value; validate on a fresh holdout first."
    elif value > 0:
        verdict, action = "MARGINAL", "Positive but a small share of the oracle."
    else:
        verdict, action = "DO NOT ADOPT: KEEP EVERYWHERE", f"The rule loses {_usd(-value)} against always-keep ($0); the experiment-level answer (keep branded search) is safer."
    chart_k = {"kind": "bar", "title": "Realised 4-week value of the rule by margin k ($)", "x": [f"k={kk:g}" for kk in POLICY_KS], "y": [rules[kk] for kk in POLICY_KS]}
    chart_t = {"kind": "bar", "title": "True value of pausing, by tier ($, 4 weeks)", "x": [f"Tier {int(t)}" for t in tier_rows.index], "y": [float(v) for v in tier_rows["all"]]}
    stages = [
        _stage("rule", "1. Rule definition", [
            _m("Rule", f"pause iff mean gain - {k:g} x sd > 0"), _m("Learners in ensemble", f"{len(algos)}"),
            _m("Out-of-fold", "DMA-grouped", "each DMA is scored by models that never saw it"), _m("k fixed", "a priori" if k == 1 else "naive baseline")],
            "Gain = predicted d(net contribution margin) per day from pausing, computed per learner then combined."),
        _stage("agree", "2. Learner agreement", [
            _m("Mean disagreement (sd, $/day)", _usd(float(sd.mean()))), _m("Unanimous on sign", f"{unanimous:.0%}", "DMAs where all 9 learners agree pause vs keep"),
            _m("Mean predicted gain ($/day)", _usd(float(mean.mean()))), _m("DMAs with positive mean gain", f"{int((mean > 0).sum())} / {len(dma)}")],
            "Large disagreement relative to the gain is the signal the margin k is meant to respect."),
        _stage("decide", "3. Decisions", [
            _m("DMAs to pause", f"{int(pause.sum())} / {len(dma)}"), _m("Oracle would pause", f"{int(oracle.sum())}"),
            *[_m(f"Tier {int(t)} paused", f"{int(r.paused)} of {int(r.n)}", f"oracle {int(r.oracle)}") for t, r in tier_rows.iterrows()]],
            "Counts of Pause decisions against the ground-truth oracle."),
        _stage("value", "4. Realised value vs benchmarks (4 weeks, ground truth)", [
            _m("Rule value", _usd(value)), _m("Always keep", "$0"), _m("Always pause", _usd(always)), _m("Oracle", _usd(orc)),
            _m("Decision accuracy", f"{float((pause == oracle).mean()):.0%}"), _m("Share of oracle captured", _f(value / orc * 100, 0, suf="%"))],
            "Scoring uses simulated truth that a live test never has; in practice use a doubly robust policy-value estimate.", chart_k,
            "pass" if value > 0 else "fail"),
        _stage("err", "5. Cost of being wrong", [
            _m("False pauses", f"{int(false_p.sum())}", "paused where keeping was better"), _m("Money lost to false pauses", _usd(float(true28[false_p].sum()))),
            _m("Missed pauses", f"{int(missed.sum())}", "kept where pausing was better"), _m("Margin forgone on missed pauses", _usd(float(true28[missed].sum())))],
            "False pauses are expensive because the average DMA has strongly negative pause value.", chart_t),
        _stage("verdict", "6. Verdict", [_m("Decision", verdict), _m("Rule value", _usd(value)), _m("Always keep", "$0"), _m("Oracle", _usd(orc))],
               action, None, "pass" if value > 0 else "fail"),
    ]
    fm = [_m("Rule value (4 wk)", _usd(value)), _m("Pauses", f"{int(pause.sum())}"), _m("Oracle", _usd(orc)), _m("Accuracy", f"{float((pause == oracle).mean()):.0%}")]
    return stages, verdict, action, fm, {"value_usd": value, "n_pause": int(pause.sum()), "oracle_usd": orc}


# ----------------------------------------------------------------------------- assemble
def _programme(gp, results):
    mac, prog, val = _rj("macro_results.json"), _rj("programme_economics.json"), _rj("policy_value.json")
    hw, meta = _rj("hardware.json"), _rj("experiment_meta.json")
    lbd = _csv("cate_leaderboard_direct.csv", index_col=0)
    ml = lbd.drop("Constant ATE (DiD)")
    const = float(lbd.loc["Constant ATE (DiD)", "pehe_norm"])
    td, to = mac["direct_bookings"]["truth"], mac["ota_bookings"]["truth"]
    s, tr = prog["SDID"], prog["Truth"]
    b = results["B1"]["summary"]["status"]
    groups = [
        {"group": "Experiment", "metrics": [_m("DMAs / holdout", f"{meta.get('n_dmas', 50)} / {meta['n_holdout']}"), _m("Window", "20 wk pre + 4 wk pause"),
                                             _m("Device (torch)", str(hw.get("torch", hw.get("torch_device", "n/a"))))]},
        {"group": "Macro lift (direct bookings / DMA-day)", "metrics": [_m("Truth", _f(td, 2)), _m("SDID", _f(mac["direct_bookings"]["SDID"]["att"], 2)),
                                                                         _m("MMT", _f(mac["direct_bookings"]["MMT"]["att"], 2)), _m("BSTS", _f(mac["direct_bookings"]["BSTS"]["att"], 2))]},
        {"group": "Macro lift (OTA bookings / DMA-day)", "metrics": [_m("Truth", _f(to, 2)), _m("SDID", _f(mac["ota_bookings"]["SDID"]["att"], 2)),
                                                                      _m("MMT", _f(mac["ota_bookings"]["MMT"]["att"], 2)), _m("BSTS", _f(mac["ota_bookings"]["BSTS"]["att"], 2))]},
        {"group": "Economics (10 holdout DMAs, 4 wk)", "metrics": [_m("Last-click ROAS", f"{s['last_click_roas']:.1f}x"), _m("iROAS (SDID / truth)", f"{s['iroas']:.1f}x / {tr['iroas']:.1f}x"),
                                                                    _m("Cannibalization", f"{s['cannibalization_rate']:.0%} / {tr['cannibalization_rate']:.0%}"),
                                                                    _m("Net value of keeping SEM", _usd(s["net_value_of_branded_search_usd"])), _m("OTA commission leakage", _usd(s["ota_commission_leakage_usd"]))]},
        {"group": "Validity", "metrics": [_m(f"{m} status (direct)", v) for m, v in b.items()] + [_m("Estimators passing (OTA)", f"{len(results['B2']['summary']['passing'])} / 3")]},
        {"group": "CATE (direct lift)", "metrics": [_m("Best learner", ml.pehe_norm.idxmin()), _m("Best PEHE / sd", _f(ml.pehe_norm.min(), 2)), _m("Constant-ATE PEHE / sd", _f(const, 2)),
                                                    _m("Learners beating constant", f"{int((ml.pehe_norm < const).sum())} / {len(ml)}"), _m("Best rank corr", _f(ml.rank_corr.max(), 2))]},
        {"group": "Policy", "metrics": [_m("Rule value (k=1)", _usd(val["model_policy_usd"])), _m("Oracle", _usd(val["oracle_usd"])), _m("Always pause", _usd(val["always_pause_usd"])),
                                        _m("Decision accuracy", f"{val['decision_accuracy']:.0%}")]},
    ]
    rollup = [{"flow": fid, "type": FLOW_TYPES[FLOW_BY_ID[fid]["type"]]["title"], "scenario": FLOW_BY_ID[fid]["short"], "verdict": r["verdict"],
               "headline": " | ".join(f"{m['label']}: {m['value']}" for m in r["flow_metrics"][:2])} for fid, r in results.items()]
    return {"groups": groups, "rollup": rollup}


def build_all() -> dict:
    gp = load_panel()
    pairs = _csv("matched_pairs.csv")
    results = {}
    for f in FLOWS:
        log.info("flow %s", f["id"])
        if f["type"] == "A":
            st, v, a, fm, summ = panel_flow(f, gp, pairs, reuse_saved=(f["id"] == "A1"))
        elif f["type"] == "B":
            st, v, a, fm, summ = validity_flow(f, gp)
        elif f["type"] == "C":
            st, v, a, fm, summ = cate_flow(f)
        else:
            st, v, a, fm, summ = policy_flow(f, gp)
        results[f["id"]] = {k: f[k] for k in ("id", "type", "short", "title", "story")} | {"type_title": FLOW_TYPES[f["type"]]["title"],
                                                                                           "question": FLOW_TYPES[f["type"]]["question"], "stages": st,
                                                                                           "verdict": v, "action": a, "flow_metrics": fm, "summary": summ}
    out = {"flows": results, "order": [f["id"] for f in FLOWS], "types": FLOW_TYPES, "programme": _programme(gp, results)}
    out = _jsonable(out)
    (path("metrics") / "flows.json").write_text(json.dumps(out, indent=2))
    log.info("flows.json written: %d flows", len(results))
    return out


def load() -> dict:
    p = path("metrics") / "flows.json"
    return json.loads(p.read_text()) if p.exists() else build_all()
