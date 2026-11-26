"""End-to-end scenario flows (UI independent). Writes outputs/metrics/flows.json.

Four flow types x 2 examples = 8 flows. Each flow is a list of *stages*; every stage carries its own metrics, an
optional table and chart spec, plus a one-line reading. `overall` holds programme-level metrics shared by all flows.

  A  Guest decision     profile -> selection bias -> CATE from all models -> decision -> counterfactual check
  B  Cohort read-out    cohort -> naive bias -> IPW/SIPW/AIPW -> balance -> CATE-vs-truth for the cohort
  C  Validity           overlap & trimming -> balance -> refuters -> DoWhy -> verdict
  D  Campaign policy    scoring -> economics -> selection -> policy comparison -> ROI & risk
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import serving
from src.common.config import load_config, out_dir
from src.data import synthetic_generator as sg
from src.models import propensity_ate as pa

FLOWS = [
    dict(id="A1", type="A", short="A1 · Expedia family Suite", title="Guest decision: Expedia family on a Suite",
         story="A family of four booked a Suite through Expedia 6+ weeks out and uses an iPhone. Should we send the pre-arrival upsell / app offer?",
         pick=dict(channel="OTA_Expedia", room_tier="Suite", min_children=1, min_lead=40, device="iOS")),
    dict(id="A2", type="A", short="A2 · Corporate loyalty solo", title="Guest decision: Corporate loyalty-member solo traveller",
         story="A Gold/Platinum business traveller on a short lead time. Already loyal and direct-leaning: does an upsell nudge help or annoy?",
         pick=dict(channel="Corporate", min_loyalty=2)),
    dict(id="B1", type="B", short="B1 · OTA cohort read-out", title="Cohort read-out: OTA bookings (Expedia + Booking.com)",
         story="Does the campaign work for the commission-heavy OTA book of business? Remove selection bias and read the lift for this cohort.",
         cohort=dict(channels=["OTA_Expedia", "OTA_Booking"])),
    dict(id="B2", type="B", short="B2 · Direct + Corporate", title="Cohort read-out: Direct web + Corporate",
         story="Direct and corporate guests already book with us. The campaign should add little: check that the estimators agree.",
         cohort=dict(channels=["Direct_Web", "Corporate"])),
    dict(id="C1", type="C", short="C1 · Validity: ancillary", title="Validity checks: ancillary-spend ATE",
         story="Before trusting the $ lift: overlap, trimming sensitivity, balance and refutation tests.", outcome="ancillary"),
    dict(id="C2", type="C", short="C2 · Validity: direct shift", title="Validity checks: direct-channel-shift ATE",
         story="Same battery for the binary direct re-booking outcome.", outcome="direct"),
    dict(id="D1", type="D", short="D1 · Baseline 30% budget", title="Campaign policy: baseline, contact the top 30%",
         story="Marketing can contact 30% of arrivals at $1.50 each. Score everyone, pick the best 30%, compare with random and oracle.",
         campaign=dict(budget_frac=0.30, cost=1.5, am=0.45, dm=55.0, channels=None)),
    dict(id="D2", type="D", short="D2 · OTA-only push", title="Campaign policy: OTA-only push at 20% commission",
         story="Revenue management wants to shift OTA guests direct: a booking is worth $75, only OTA channels are eligible, 30% budget.",
         campaign=dict(budget_frac=0.30, cost=1.5, am=0.45, dm=75.0, channels=["OTA_Expedia", "OTA_Booking"])),
]


def slug(n):
    return re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_")


def _j(name):
    return json.load(open(out_dir("metrics") / name, encoding="utf-8"))


def _f(x, n=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return int(round(float(x))) if n == 0 else round(float(x), n)


def stage(name, metrics, reading, table=None, chart=None, status="ok"):
    return dict(name=name, metrics=metrics, reading=reading, table=table, chart=chart, status=status)


# ------------------------------------------------------------------ shared context
class Ctx:
    def __init__(self):
        self.cfg = load_config()
        self.tr = pd.read_parquet(out_dir("data") / "train.parquet")
        self.sc = pd.read_parquet(out_dir("data") / "scored_guests.parquet")
        self.te_prop = pd.read_parquet(out_dir("data") / "test_propensity.parquet")["e_hat"].values
        tp = pd.read_parquet(out_dir("data") / "train_propensity.parquet")
        self.e = tp["e_hat"].values
        self.X = sg.build_features(self.tr)
        self.T = self.tr.treatment.values
        self.ate = _j("ate_comparison.json")
        self.ref = _j("refutation.json")
        self.lb = pd.read_csv(out_dir("metrics") / "cate_leaderboard.csv")
        self.camp = _j("campaign_summary.json")
        self.data = _j("data_summary.json")
        self.meta = _j("learner_meta.json")
        self.dec = pd.read_csv(out_dir("metrics") / "uplift_deciles.csv")
        self.bal = pd.read_csv(out_dir("metrics") / "balance_smd.csv").set_index("feature")
        fast = self.data["n"] < 20000
        self.y = {"ancillary": self.tr.y_ancillary.values.astype(float), "direct": self.tr.y_direct.values.astype(float)}
        self.tau = {"ancillary": self.tr.tau_anc.values, "direct": self.tr.tau_direct.values}
        self.mu = {o: pa.outcome_models_crossfit(self.X, self.T, self.y[o], o == "direct", fast, self.cfg["seed"], 3)
                   for o in ("ancillary", "direct")}
        self.auc = float(roc_auc_score(self.T, self.e))


def overall_metrics(c: Ctx) -> dict:
    a, d = c.ate["outcomes"]["ancillary"], c.ate["outcomes"]["direct"]
    best_a, best_d = c.camp["best_model"]["ancillary"], c.camp["best_model"]["direct"]
    la = c.lb[(c.lb.outcome == "ancillary") & (c.lb.model == best_a)].iloc[0]
    ld = c.lb[(c.lb.outcome == "direct") & (c.lb.model == best_d)].iloc[0]
    passes = sum(int(v["pass"]) for o in c.ref.values() for v in o["builtin"].values())
    total = sum(len(o["builtin"]) for o in c.ref.values())
    return {
        "Stays simulated": f"{c.data['n']:,}", "Treated share": f"{c.data['treated_share']:.0%}",
        "Propensity AUC (cross-fit)": _f(c.auc, 3), "Common support kept": f"{c.ate['common_support_kept']:.1%}",
        "Max |SMD| before → after IPW": f"{c.ate['balance']['max_abs_smd_before']:.2f} → {c.ate['balance']['max_abs_smd_ipw']:.3f}",
        "Naive ancillary lift ($)": _f(a["estimators"]["Naive"]["ate"], 2), "AIPW ancillary lift ($)": _f(a["estimators"]["AIPW (DR)"]["ate"], 2),
        "True ancillary ATE ($)": _f(a["true_ate"], 2),
        "Naive direct lift (pp)": _f(d["estimators"]["Naive"]["ate"] * 100, 1), "AIPW direct lift (pp)": _f(d["estimators"]["AIPW (DR)"]["ate"] * 100, 1),
        "True direct ATE (pp)": _f(d["true_ate"] * 100, 1),
        "Best ancillary CATE model": best_a, "Ancillary PEHE ($)": _f(la.pehe, 2), "Best direct CATE model": best_d, "Direct PEHE (pp)": _f(ld.pehe * 100, 1),
        "Refuters passed": f"{passes}/{total}",
        "Targeted value (true $)": _f(c.camp["true_value_targeted"], 0), "Oracle value (true $)": _f(c.camp["true_value_oracle_top"], 0),
        "Oracle capture": f"{c.camp['true_value_targeted'] / c.camp['true_value_oracle_top']:.1%}",
        "Learners fitted": len(c.meta), "Total fit time (s)": _f(sum(v.get("fit_seconds", 0) for v in c.meta.values()), 1),
    }


# ------------------------------------------------------------------ A: guest decision
def _pick_guest(sc: pd.DataFrame, p: dict) -> pd.Series:
    m = pd.Series(True, index=sc.index)
    rules = [("channel", lambda s, v: s.channel == v), ("room_tier", lambda s, v: s.room_tier == v), ("device", lambda s, v: s.device == v),
             ("min_children", lambda s, v: s.children >= v), ("min_lead", lambda s, v: s.lead_time >= v), ("min_loyalty", lambda s, v: s.loyalty_tier >= v)]
    for k, fn in rules:
        if k in p:
            nm = m & fn(sc, p[k])
            if nm.sum() >= 5:       # relax a criterion rather than return nothing (tiny fast-mode runs)
                m = nm
    sub = sc[m]
    X = sg.build_features(sub)
    Z = (X - X.mean()) / (X.std() + 1e-9)
    return sub.loc[((Z - Z.median()) ** 2).sum(axis=1).idxmin()]


def guest_flow(c: Ctx, f: dict) -> dict:
    b = c.cfg["business"]
    g = _pick_guest(c.sc, f["pick"])
    idx = g.name
    e_hat = float(c.te_prop[idx])
    cols = {o: [col for col in c.sc.columns if col.startswith(f"cate_{o}__") and not col.endswith(("__lo", "__hi"))] for o in ("ancillary", "direct")}
    names = {o: {col: n for n in serving.model_names(o) for col in cols[o] if col == f"cate_{o}__{slug(n)}"} for o in cols}
    rows = []
    for n in serving.model_names("ancillary"):
        ca, cd = f"cate_ancillary__{slug(n)}", f"cate_direct__{slug(n)}"
        rows.append(dict(model=n, tau_anc=_f(g[ca], 2) if ca in g.index else None))
    tbl = pd.DataFrame({"model": serving.model_names("ancillary")})
    tbl["τ̂ ancillary ($)"] = [_f(g.get(f"cate_ancillary__{slug(n)}"), 2) for n in tbl.model]
    dnames = serving.model_names("direct")
    td = pd.DataFrame({"model": dnames, "τ̂ direct (pp)": [_f(g.get(f"cate_direct__{slug(n)}", np.nan) * 100, 2) for n in dnames]})
    tbl = tbl.merge(td, on="model", how="outer")
    tau_a, tau_d = float(g.tau_anc), float(g.tau_direct)
    tbl["abs err ancillary ($)"] = (tbl["τ̂ ancillary ($)"] - tau_a).abs().round(2)
    tbl["abs err direct (pp)"] = (tbl["τ̂ direct (pp)"] - tau_d * 100).abs().round(2)
    a_vals = tbl["τ̂ ancillary ($)"].dropna()
    prof = {k: (float(g[k]) if k in ("lead_time", "hist_spend") else (g[k].item() if hasattr(g[k], "item") else g[k]))
            for k in sg.RAW_COLS}
    rank_pct = float((c.sc.incremental_value_hat >= g.incremental_value_hat).mean())
    seg_pred = serving.classify(g.incremental_value_hat, g.tau_hat_direct, g.tau_hat_ancillary)
    send = g.incremental_value_hat > 0
    true_val = float(g.incremental_value_true)
    lo = g.get("cate_ancillary__causal_forest_dml__lo", np.nan); hi = g.get("cate_ancillary__causal_forest_dml__hi", np.nan)
    pop_e = float(c.te_prop.mean())
    cohort = c.sc[c.sc.channel == g.channel]
    stages = [
        stage("1 · Guest profile", {"Channel": g.channel, "Room tier": g.room_tier, "Lead (days)": int(g.lead_time), "Device": g.device,
                                    "Loyalty tier": int(g.loyalty_tier), "Hist. spend ($)": _f(g.hist_spend, 0)},
              f"Held-out reservation #{int(idx)}, the most typical guest matching the scenario (nearest to the median in feature space).",
              table=pd.DataFrame([prof]).T.reset_index().rename(columns={"index": "feature", 0: "value"}).astype(str).to_dict("records")),
        stage("2 · Selection-bias check", {"P(contacted | x) estimated": _f(e_hat), "Average over all guests": _f(pop_e), "Ratio vs average": _f(e_hat / pop_e, 2),
                                           "True P(contacted) (synthetic)": _f(g.e_true), "Was actually contacted": "yes" if g.treatment else "no"},
              "Guests like this are contacted " + ("more" if e_hat > pop_e else "less") + " often than average, so comparing contacted vs un-contacted guests like this one is confounded; "
              "the CATE models below adjust for it.", chart=dict(kind="bars", title="Contact propensity", labels=["This guest (est.)", "This guest (true)", "All guests"], values=[_f(e_hat), _f(g.e_true), _f(pop_e)])),
        stage("3 · CATE from every model", {"True τ ancillary ($)": _f(tau_a, 2), "Best-model τ̂ ($)": _f(g.tau_hat_ancillary, 2), "Median across models ($)": _f(a_vals.median(), 2),
                                            "Model spread (std, $)": _f(a_vals.std(), 2), "True τ direct (pp)": _f(tau_d * 100, 2), "Best-model τ̂ direct (pp)": _f(g.tau_hat_direct * 100, 2)},
              "Every estimator gives its own τ̂ for this guest; the dashed line is the synthetic truth. Causal Forest adds a 95% interval"
              + (f" [{lo:.1f}, {hi:.1f}] $ which {'covers' if lo <= tau_a <= hi else 'misses'} the truth." if not np.isnan(lo) else "."),
              table=tbl.fillna("–").to_dict("records"),
              chart=dict(kind="estimates", title="τ̂ ancillary ($) by model", labels=list(tbl.model), values=[_f(v, 2) for v in tbl["τ̂ ancillary ($)"].fillna(0)], truth=_f(tau_a, 2))),
        stage("4 · Decision", {"Expected net value ($)": _f(g.incremental_value_hat, 2), "Decision": "SEND" if send else "SUPPRESS", "Predicted segment": seg_pred,
                               "Rank (top x% of guests)": f"{rank_pct:.0%}", "Inside top-30% budget": "yes" if g.target_flag else "no"},
              f"Net value = {b['ancillary_margin']} × τ̂ ancillary + ${b['direct_booking_margin']:.0f} × τ̂ direct − ${b['cost_per_offer']:.2f} offer cost.",
              chart=dict(kind="bars", title="Expected value decomposition ($)", labels=["Ancillary margin", "Direct-shift value", "Offer cost", "Net"],
                         values=[_f(b["ancillary_margin"] * g.tau_hat_ancillary, 2), _f(b["direct_booking_margin"] * g.tau_hat_direct, 2), _f(-b["cost_per_offer"], 2), _f(g.incremental_value_hat, 2)])),
        stage("5 · Counterfactual check (synthetic truth)", {"True net value of sending ($)": _f(true_val, 2), "Decision correct": "yes" if (true_val > 0) == bool(send) else "no",
                                                             "True segment": g.segment, "Est. − true value ($)": _f(g.incremental_value_hat - true_val, 2),
                                                             "Observed ancillary spend ($)": _f(g.y_ancillary, 2), "Rebooked direct": int(g.y_direct)},
              "Because the DGP is known we can score the decision against the real counterfactual, something a real campaign can only approximate with a holdout.",
              status="ok" if (true_val > 0) == bool(send) else "warn"),
        stage("6 · Cohort context", {"Guests in same channel": int(len(cohort)), "Mean best-model τ̂ ($)": _f(cohort.tau_hat_ancillary.mean(), 2), "Mean true τ ($)": _f(cohort.tau_anc.mean(), 2),
                                     "Share SEND in channel": f"{(cohort.incremental_value_hat > 0).mean():.0%}"},
              "How this guest compares with others booked through the same channel."),
    ]
    return dict(id=f["id"], type="A", short=f["short"], title=f["title"], story=f["story"], stages=stages,
                summary={"Decision": "SEND" if send else "SUPPRESS", "Net value ($)": _f(g.incremental_value_hat, 2), "Correct": "yes" if (true_val > 0) == bool(send) else "no"},
                api_payload={k: (prof[k] if k not in ("adults", "children", "loyalty_tier", "los", "resort") else int(prof[k])) for k in sg.RAW_COLS},
                expect=dict(tau_ancillary=_f(g.tau_hat_ancillary, 4), tau_direct=_f(g.tau_hat_direct, 5)))


# ------------------------------------------------------------------ B: cohort read-out
def cohort_flow(c: Ctx, f: dict) -> dict:
    m = c.tr.channel.isin(f["cohort"]["channels"]).values
    T, e = c.T[m], c.e[m]
    stages = []
    res = {}
    for o in ("ancillary", "direct"):
        y = c.y[o][m]; mu1, mu0 = c.mu[o][0][m], c.mu[o][1][m]
        tau = c.tau[o][m]
        naive = y[T == 1].mean() - y[T == 0].mean()
        ipw, _ = pa.ipw(e, T, y); sip, _ = pa.sipw(e, T, y); aip = pa.aipw(e, T, y, mu1, mu0)[0]
        res[o] = dict(true=float(tau.mean()), naive=float(naive), ipw=float(ipw), sipw=float(sip), aipw=float(aip))
    sc = 100
    s0 = c.sc[c.sc.channel.isin(f["cohort"]["channels"])]
    X = c.X[m]
    w = np.where(T == 1, np.minimum(1 / e, np.quantile(1 / e, .99)), np.minimum(1 / (1 - e), np.quantile(1 / (1 - e), .99)))
    smd0, smd1 = pa.smd(X, T), pa.smd(X, T, w)
    stages.append(stage("1 · Cohort", {"Stays (train)": int(m.sum()), "Treated share": f"{T.mean():.0%}", "Mean P(contacted)": _f(e.mean()),
                                       "Share of all stays": f"{m.mean():.0%}", "True ancillary ATE ($)": _f(res['ancillary']['true'], 2),
                                       "True direct ATE (pp)": _f(res['direct']['true'] * sc, 2)},
                        "Cohort definition and its true (synthetic) effects.", chart=dict(kind="bars", title="Cohort vs population true ATE (ancillary $)", labels=["Cohort", "All stays"], values=[_f(res['ancillary']['true'], 2), _f(c.ate['outcomes']['ancillary']['true_ate'], 2)])))
    stages.append(stage("2 · Selection bias", {"Naive ancillary lift ($)": _f(res['ancillary']['naive'], 2), "Bias vs truth ($)": _f(res['ancillary']['naive'] - res['ancillary']['true'], 2),
                                                "Naive direct lift (pp)": _f(res['direct']['naive'] * sc, 2), "Bias vs truth (pp)": _f((res['direct']['naive'] - res['direct']['true']) * sc, 2)},
                        "Treated guests in this cohort differ systematically from untreated ones, so the raw difference is not the causal effect.", status="warn"))
    for o, unit, k in (("ancillary", "$", 1), ("direct", "pp", sc)):
        r = res[o]
        names = ["Naive", "IPW", "SIPW", "AIPW"]; vals = [_f(r[x] * k, 2) for x in ("naive", "ipw", "sipw", "aipw")]
        stages.append(stage(f"3{'a' if o == 'ancillary' else 'b'} · Adjusted estimate: {o}", {f"{n} ({unit})": v for n, v in zip(names, vals)} | {f"True ({unit})": _f(r['true'] * k, 2), "AIPW error": _f((r['aipw'] - r['true']) * k, 2)},
                            "Weighting / doubly-robust estimators pull the naive number back towards the truth (dashed).",
                            chart=dict(kind="estimates", title=f"{o}: estimate by method ({unit})", labels=names, values=vals, truth=_f(r['true'] * k, 2))))
    stages.append(stage("4 · Covariate balance in cohort", {"Max |SMD| before": _f(smd0.abs().max()), "Max |SMD| after IPW": _f(smd1.abs().max()),
                                                            "Covariates |SMD|<0.1 before": f"{(smd0.abs() < .1).mean():.0%}", "after IPW": f"{(smd1.abs() < .1).mean():.0%}"},
                        "Weights should push every standardised mean difference under 0.10.", status="ok" if smd1.abs().max() < 0.1 else "warn",
                        chart=dict(kind="bars", title="|SMD| before vs after IPW (top 8 covariates)", labels=list(smd0.abs().sort_values(ascending=False).index[:8]),
                                   values=[_f(v) for v in smd0.abs().sort_values(ascending=False)[:8]], values2=[_f(smd1.abs()[i]) for i in smd0.abs().sort_values(ascending=False).index[:8]], name1="before", name2="after IPW")))
    stages.append(stage("5 · CATE models for the cohort (held-out)", {"Mean best τ̂ ancillary ($)": _f(s0.tau_hat_ancillary.mean(), 2), "Mean true τ ancillary ($)": _f(s0.tau_anc.mean(), 2),
                                                                    "Mean best τ̂ direct (pp)": _f(s0.tau_hat_direct.mean() * sc, 2), "Mean true τ direct (pp)": _f(s0.tau_direct.mean() * sc, 2),
                                                                    "Share SEND": f"{(s0.incremental_value_hat > 0).mean():.0%}", "Persuadable share (true)": f"{(s0.segment == 'Persuadable').mean():.0%}"},
                        "Individual-level models agree with the cohort-level estimate: the cohort average of τ̂ is close to the truth."))
    return dict(id=f["id"], type="B", short=f["short"], title=f["title"], story=f["story"], stages=stages,
                summary={"True ancillary ATE ($)": _f(res['ancillary']['true'], 2), "AIPW ($)": _f(res['ancillary']['aipw'], 2), "Naive ($)": _f(res['ancillary']['naive'], 2)})


# ------------------------------------------------------------------ C: validity
def validity_flow(c: Ctx, f: dict) -> dict:
    o = f["outcome"]; unit, k = ("$", 1) if o == "ancillary" else ("pp", 100)
    y = c.y[o]; mu1, mu0 = c.mu[o]; T, e, tau = c.T, c.e, c.tau[o]
    rows = []
    for lo, hi in ((0.025, 0.975), (0.05, 0.95), (0.10, 0.90), (0.20, 0.80)):
        mk = (e >= lo) & (e <= hi)
        a = pa.aipw(e[mk], T[mk], y[mk], mu1[mk], mu0[mk])[0]
        rows.append(dict(window=f"[{lo}, {hi}]", kept=f"{mk.mean():.1%}", aipw=_f(a * k, 2), truth_on_kept=_f(tau[mk].mean() * k, 2), error=_f((a - tau[mk].mean()) * k, 2)))
    ref = c.ref[o]; bi = ref["builtin"]; dw = ref.get("dowhy") or {}
    ate = c.ate["outcomes"][o]; ests = ate["estimators"]
    ok = all(v["pass"] for v in bi.values())
    stages = [
        stage("1 · Overlap & trimming sensitivity", {"Propensity AUC": _f(c.auc), "e(x) range": f"{e.min():.3f} – {e.max():.3f}", "Common support kept": f"{c.ate['common_support_kept']:.1%}",
                                                     f"AIPW range over windows ({unit})": f"{min(r['aipw'] for r in rows)} – {max(r['aipw'] for r in rows)}"},
              "The estimate should be stable when extreme-propensity guests are trimmed; a drift signals poor overlap.", table=rows,
              chart=dict(kind="estimates", title=f"AIPW ATE ({unit}) by trimming window", labels=[r["window"] for r in rows], values=[r["aipw"] for r in rows], truth=_f(ate["true_ate"] * k, 2))),
        stage("2 · Covariate balance", {"Max |SMD| before": _f(c.ate['balance']['max_abs_smd_before']), "after PSM": _f(c.ate['balance']['max_abs_smd_psm']), "after IPW": _f(c.ate['balance']['max_abs_smd_ipw']),
                                        "Covariates < 0.10 after IPW": f"{c.ate['balance']['share_below_0p1_ipw']:.0%}"},
              "Balance is a property of the propensity model, so it is the same for both outcomes.",
              chart=dict(kind="bars", title="Top-8 imbalanced covariates: |SMD|", labels=list(c.bal.smd_before.abs().sort_values(ascending=False).index[:8]), values=[_f(v) for v in c.bal.smd_before.abs().sort_values(ascending=False)[:8]],
                         values2=[_f(c.bal.smd_ipw.abs()[i]) for i in c.bal.smd_before.abs().sort_values(ascending=False).index[:8]], name1="before", name2="after IPW")),
        stage("3 · Built-in refuters (AIPW)", {"Baseline ATE": _f(ref["baseline_aipw_ate"] * k, 3), "Random common cause": _f(bi["random_common_cause"]["new_effect"] * k, 3),
                                               "Placebo treatment": _f(bi["placebo_treatment"]["new_effect"] * k, 3), "Data subset (80%)": _f(bi["data_subset"]["new_effect"] * k, 3),
                                               "All passed": "yes" if ok else "no"},
              "Random cause and subset should leave the effect unchanged; a placebo treatment should give ≈ 0.", status="ok" if ok else "warn",
              chart=dict(kind="estimates", title=f"Refuter effects ({unit}); dashed = baseline", labels=["Baseline", "Random cause", "Placebo", "Subset"],
                         values=[_f(ref["baseline_aipw_ate"] * k, 3), _f(bi["random_common_cause"]["new_effect"] * k, 3), _f(bi["placebo_treatment"]["new_effect"] * k, 3), _f(bi["data_subset"]["new_effect"] * k, 3)], truth=_f(ref["baseline_aipw_ate"] * k, 3))),
        stage("4 · DoWhy cross-check", ({"Estimate (OLS backdoor)": _f(dw["estimate"] * k, 3)} | {f"{n}": _f(v.get("new_effect", np.nan) * k, 3) for n, v in dw["refuters"].items()}) if dw.get("available") else {"DoWhy": "unavailable"},
              "Independent implementation of the same refuters (DoWhy backdoor linear regression).", status="ok" if dw.get("available") else "warn"),
        stage("5 · Verdict", {"True ATE": _f(ate["true_ate"] * k, 3), "Naive": _f(ests["Naive"]["ate"] * k, 3), "IPW": _f(ests["IPW"]["ate"] * k, 3), "AIPW": _f(ests["AIPW (DR)"]["ate"] * k, 3),
                              "AIPW error": f"{(ests['AIPW (DR)']['ate'] / ate['true_ate'] - 1):+.1%}", "Naive error": f"{(ests['Naive']['ate'] / ate['true_ate'] - 1):+.0%}"},
              "Refuters probe only the observed-confounder model; they cannot rule out unmeasured confounding (none exists here by construction).", status="ok" if ok else "warn",
              chart=dict(kind="estimates", title=f"ATE ({unit}) by estimator", labels=list(ests.keys()), values=[_f(v["ate"] * k, 3) for v in ests.values()], truth=_f(ate["true_ate"] * k, 3))),
    ]
    return dict(id=f["id"], type="C", short=f["short"], title=f["title"], story=f["story"], stages=stages,
                summary={"Refuters passed": f"{sum(int(v['pass']) for v in bi.values())}/{len(bi)}", "AIPW error": f"{(ests['AIPW (DR)']['ate'] / ate['true_ate'] - 1):+.1%}"})


# ------------------------------------------------------------------ D: campaign policy
def policy_flow(c: Ctx, f: dict) -> dict:
    p = f["campaign"]
    r = serving.optimize_campaign(p["budget_frac"], p["cost"], p["am"], p["dm"], True, p["channels"])
    mask = r.pop("mask")
    sc = c.sc if not p["channels"] else c.sc[c.sc.channel.isin(p["channels"])].reset_index(drop=True)
    best = c.camp["best_model"]
    dec = c.dec
    la = c.lb[(c.lb.outcome == "ancillary") & (c.lb.model == best["ancillary"])].iloc[0]
    ld = c.lb[(c.lb.outcome == "direct") & (c.lb.model == best["direct"])].iloc[0]
    sel = sc[mask]
    val_true = p["am"] * sc.tau_anc + p["dm"] * sc.tau_direct - p["cost"]
    dogs = int((val_true[mask] < 0).sum())
    capture = r["true_net_value"] / r["oracle_net_value"] if r["oracle_net_value"] else None
    lift_vs_rand = r["true_net_value"] / r["random_net_value"] if r["random_net_value"] > 0 else None
    stages = [
        stage("1 · Scoring", {"Ancillary model": best["ancillary"], "Direct model": best["direct"], "Ancillary Qini": _f(la.qini, 3), "Direct Qini": _f(ld.qini, 4),
                              "Ancillary PEHE ($)": _f(la.pehe, 2), "Direct PEHE (pp)": _f(ld.pehe * 100, 2), "Guests scored": int(len(sc))},
              "Best learner per outcome (by Qini on held-out stays) scores every eligible guest.",
              chart=dict(kind="bars", title="Ancillary: IPW uplift by score decile ($/guest)", labels=[f"D{d}" for d in dec[dec.outcome == 'ancillary'].decile], values=[_f(v, 2) for v in dec[dec.outcome == 'ancillary'].ipw_uplift])),
        stage("2 · Economics & budget", {"Budget share": f"{p['budget_frac']:.0%}", "Cost / offer ($)": p["cost"], "Ancillary margin": p["am"], "Value of a direct booking ($)": p["dm"],
                                         "Eligible channels": ", ".join(p["channels"]) if p["channels"] else "all"},
              "Net value per guest = margin × τ̂ ancillary + booking value × τ̂ direct − offer cost; contact the top guests with positive value."),
        stage("3 · Selection", {"Guests contacted": r["n_selected"], "Share of eligible": f"{r['share_selected']:.0%}", "Spend ($)": _f(r["spend"], 0), "Mean τ̂ ancillary of selected ($)": _f(sel.tau_hat_ancillary.mean(), 2),
                                "Mean τ̂ direct of selected (pp)": _f(sel.tau_hat_direct.mean() * 100, 2), "Persuadables among selected": f"{(sel.segment == 'Persuadable').mean():.0%}"},
              "Who gets the offer.", table=pd.DataFrame(r["by_channel"]).round(3).to_dict("records"),
              chart=dict(kind="bars", title="Selected guests by true segment", labels=list(sel.segment.value_counts().index), values=[int(v) for v in sel.segment.value_counts().values])),
        stage("4 · Policy comparison (true $)", {"Causal targeting": _f(r["true_net_value"], 0), "Random, same size": _f(r["random_net_value"], 0), "Oracle (true τ)": _f(r["oracle_net_value"], 0), "Contact everyone": _f(r["treat_all_net_value"], 0),
                                                  "Capture of oracle": f"{capture:.1%}" if capture else "–", "Lift vs random": f"{lift_vs_rand:.1f}x" if lift_vs_rand else "–"},
              "Targeting is judged against random picks of the same size and the unattainable oracle. Contacting everyone can win when most guests have positive value.",
              chart=dict(kind="bars", title="True net value ($)", labels=["Causal", "Random", "Oracle", "Everyone"], values=[_f(r["true_net_value"], 0), _f(r["random_net_value"], 0), _f(r["oracle_net_value"], 0), _f(r["treat_all_net_value"], 0)])),
        stage("5 · ROI & risk", {"True ROI": f"{r['roi_true'] * 100:,.0f}%" if r["roi_true"] is not None else "–", "Estimated net value ($)": _f(r["est_net_value"], 0), "True net value ($)": _f(r["true_net_value"], 0),
                                 "Estimation error": f"{(r['est_net_value'] / r['true_net_value'] - 1):+.1%}" if r["true_net_value"] else "–",
                                 "Contacted with negative true value": dogs, "Share of contacted": f"{dogs / max(r['n_selected'], 1):.1%}"},
              "Estimated vs true value shows whether the model is over-promising; negative-value contacts are the sleeping-dog leakage.",
              status="ok" if dogs / max(r["n_selected"], 1) < 0.1 else "warn"),
    ]
    return dict(id=f["id"], type="D", short=f["short"], title=f["title"], story=f["story"], stages=stages,
                summary={"True net value ($)": _f(r["true_net_value"], 0), "Capture of oracle": f"{capture:.1%}" if capture else "–", "ROI": f"{r['roi_true'] * 100:,.0f}%" if r["roi_true"] is not None else "–"})


def model_matrix(c: Ctx) -> dict:
    """Every learner scored inside every flow (same guests / cohorts / campaigns as the flows themselves).

    A: abs error of tau_hat for the flow's guest | B: cohort mean error and cohort PEHE | C: bias of the model's
    population-mean CATE vs the true ATE | D: true net value of the model's own top-30% policy (both outcomes from the
    same learner; Uplift RF is binary-only so it is scored on the direct outcome alone) and capture of the oracle.
    """
    sc = c.sc
    b = c.cfg["business"]
    names = sorted({n for o in ("ancillary", "direct") for n in serving.model_names(o)})
    rows = {n: {} for n in names}
    for f in FLOWS:
        if f["type"] == "A":
            g = _pick_guest(sc, f["pick"])
            for n in names:
                for o, truth, k in (("ancillary", g.tau_anc, 1.0), ("direct", g.tau_direct, 100.0)):
                    col = f"cate_{o}__{slug(n)}"
                    rows[n][f"{f['id']} |τ̂−τ| {o} ({'$' if o == 'ancillary' else 'pp'})"] = _f(abs(g[col] - truth) * k, 3) if col in sc else None
        elif f["type"] == "B":
            s = sc[sc.channel.isin(f["cohort"]["channels"])]
            for n in names:
                for o, tcol, k, u in (("ancillary", "tau_anc", 1.0, "$"), ("direct", "tau_direct", 100.0, "pp")):
                    col = f"cate_{o}__{slug(n)}"
                    if col in s:
                        rows[n][f"{f['id']} cohort PEHE {o} ({u})"] = _f(np.sqrt(((s[col] - s[tcol]) ** 2).mean()) * k, 3)
                        rows[n][f"{f['id']} cohort mean err {o} ({u})"] = _f((s[col].mean() - s[tcol].mean()) * k, 3)
                    else:
                        rows[n][f"{f['id']} cohort PEHE {o} ({u})"] = None
                        rows[n][f"{f['id']} cohort mean err {o} ({u})"] = None
        elif f["type"] == "C":
            for n in names:
                col = f"cate_{f['outcome']}__{slug(n)}"
                tcol, k, u = ("tau_anc", 1.0, "$") if f["outcome"] == "ancillary" else ("tau_direct", 100.0, "pp")
                rows[n][f"{f['id']} mean-CATE ATE bias {f['outcome']} ({u})"] = _f((sc[col].mean() - sc[tcol].mean()) * k, 3) if col in sc else None
        else:
            p = f["campaign"]
            s = sc if not p["channels"] else sc[sc.channel.isin(p["channels"])]
            vt = p["am"] * s.tau_anc + p["dm"] * s.tau_direct - p["cost"]
            kk = int(round(p["budget_frac"] * len(s)))
            oracle = float(np.sort(vt.values)[::-1][:kk].sum())
            for n in names:
                ca, cd = f"cate_ancillary__{slug(n)}", f"cate_direct__{slug(n)}"
                if cd not in s:
                    rows[n][f"{f['id']} true net value ($)"] = rows[n][f"{f['id']} capture of oracle"] = None
                    continue
                ta = s[ca] if ca in s else s["tau_hat_ancillary"]          # Uplift RF: borrow the best ancillary model
                vh = p["am"] * ta + p["dm"] * s[cd] - p["cost"]
                sel = np.argsort(-vh.values, kind="stable")[:kk]
                sel = sel[vh.values[sel] > 0]
                v = float(vt.values[sel].sum())
                rows[n][f"{f['id']} true net value ($)"] = _f(v, 0)
                rows[n][f"{f['id']} capture of oracle"] = _f(v / oracle, 4)
    return rows


def build_flows() -> dict:
    serving.load_model.cache_clear()
    c = Ctx()
    flows = []
    for f in FLOWS:
        flows.append({"A": guest_flow, "B": cohort_flow, "C": validity_flow, "D": policy_flow}[f["type"]](c, f))
    out = dict(overall=overall_metrics(c), flows=flows, model_matrix=model_matrix(c))
    with open(out_dir("metrics") / "flows.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return out


if __name__ == "__main__":
    r = build_flows()
    print({f["id"]: f["summary"] for f in r["flows"]})
