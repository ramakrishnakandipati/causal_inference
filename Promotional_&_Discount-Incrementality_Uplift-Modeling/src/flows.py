"""End-to-end scenario flows (UI-independent): guest selection, per-guest report and campaign-level report.

A *guest flow* walks one real held-out guest through: profile -> selection-bias check (propensity) -> CATE from all
11 estimators -> persona + CRM decision -> business impact checked against the ground-truth counterfactual.
The *campaign flow* does the same for a whole CRM list (batch scoring -> decisions -> policy comparison -> ROI).

Both the Streamlit app and scripts/capture_screenshots.py import from here, so the screenshots are produced by exactly
the numbers the UI shows.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.common import FEATURES, LOYALTY_LEVELS, load_config, resolve

FLOWS = [
    {"id": "flow1", "slug": "persuadable", "kind": "guest", "persona": "Persuadable", "short": "1 · Persuadable",
     "title": "Weekend Leisure Browser", "preset": "Weekend Leisure Browser",
     "story": "A price-sensitive guest who books mostly weekend stays, moderately recent, opens emails. A genuine promo "
              "responder: she books because of the offer."},
    {"id": "flow2", "slug": "always_taker", "kind": "guest", "persona": "Always-Taker", "short": "2 · Always-Taker",
     "title": "Frequent Corporate Platinum", "preset": "Frequent Corporate Platinum",
     "story": "A Platinum road warrior with short lead times who books many times a year at full rate. He will book "
              "anyway, so a discount is pure margin leakage."},
    {"id": "flow3", "slug": "lost_cause", "kind": "guest", "persona": "Lost Cause", "short": "3 · Lost Cause",
     "title": "Dormant 2-Year Lapsed Guest", "preset": "Dormant 2-Year Lapsed Guest",
     "story": "A non-member who has not stayed in well over a year and ignores emails. No offer will move this guest; "
              "mailing only adds cost."},
    {"id": "flow4", "slug": "sleeping_dog", "kind": "guest", "persona": "Sleeping Dog", "short": "4 · Sleeping Dog",
     "title": "Full-Rate Loyal Guest with Active Booking", "preset": "Full-Rate Loyal Guest with Active Booking",
     "story": "A loyal, rate-insensitive guest with a high cancellation rate and a booking already in place. A promo "
              "invites cancel-and-rebook at 20% off, or an unsubscribe."},
    {"id": "flow5", "slug": "campaign", "kind": "campaign", "persona": None, "short": "5 · Campaign",
     "title": "Score a CRM list, target, and measure ROI",
     "story": "Run the whole pipeline on a CRM list: batch-score guests, turn CATEs into SEND / SUPPRESS decisions, and "
              "compare the causal policy with mass blast and propensity targeting using the ground-truth counterfactuals."},
]

QUADRANT_WHY = {
    "Persuadable": "Estimated conversion lift is clearly positive ({tc:+.1f} pp) and the extra bookings outweigh the offer cost.",
    "Always-Taker": "Baseline conversion is already high ({p0:.0f}%) and the lift is small ({tc:+.1f} pp): the offer mostly gives away margin.",
    "Lost Cause": "Baseline conversion is only {p0:.0f}% and there is no real lift ({tc:+.1f} pp): mailing would only add cost.",
    "Sleeping Dog": "Estimated conversion goes DOWN ({tc:+.1f} pp): a promo risks cancel-and-rebook at a discount or an unsubscribe.",
}


# ------------------------------------------------------------------------------------------------------------
def load_flow_data(cfg=None):
    cfg = cfg or load_config()
    logs = resolve(cfg["paths"]["logs"])
    te = pd.read_parquet(logs / "test_set.parquet")
    dec = pd.read_parquet(logs / "test_decisions.parquet")
    return te, dec


def pick_guest(te: pd.DataFrame, persona: str) -> pd.Series:
    """Most 'typical' held-out guest of a persona: nearest to the persona's median in standardised feature space."""
    X = te[FEATURES].astype(float)
    Z = (X - X.mean()) / X.std()
    m = (te["persona"] == persona).values
    med = Z[m].median()
    d = ((Z[m] - med) ** 2).sum(axis=1)
    return te.loc[d.idxmin()]


def guest_payload(row: pd.Series) -> dict:
    """API-ready request body for one guest (loyalty tier as its label)."""
    g = {k: float(row[k]) for k in FEATURES if k != "loyalty_tier"}
    g["loyalty_tier"] = LOYALTY_LEVELS[int(row["loyalty_tier"])]
    return {k: (round(g[k], 3) if k != "loyalty_tier" else g[k]) for k in FEATURES}


def _p0_true(dconv, R, tau_margin_true, off):
    """Recover the true baseline conversion probability from the DGP identity tau = dConv*R*(m-d) - p0*R*d."""
    return (dconv * R * (off["margin_rate"] - off["discount_pct"]) - tau_margin_true) / (R * off["discount_pct"])


def true_values(df: pd.DataFrame, cfg) -> dict[str, np.ndarray]:
    """Ground-truth net value per guest of each action (discount arm exact; points arm modelled)."""
    o = cfg["offer"]
    R = df["stay_value"].values
    dconv = df["tau_conv_true"].values
    p0 = _p0_true(dconv, R, df["tau_margin_true"].values, o)
    P = o["points_offer"] * o["points_cost_per_point"]
    cc = o["comm_cost"]
    return {"disc": df["tau_margin_true"].values - cc,
            "pts": o["points_conv_effect_ratio"] * dconv * (R * o["margin_rate"] - P) - p0 * P - cc,
            "p0_true": p0}


# ------------------------------------------------------------------------------------------------------------
def guest_report(scorer, row: pd.Series, cfg=None, estimator: str | None = None) -> dict:
    cfg = cfg or scorer.cfg
    est = estimator or scorer.best
    payload = guest_payload(row)
    per_est = []
    for name in scorer.order:
        r = scorer.score([payload], name)[0]
        per_est.append({"estimator": name, "tau": r["tau_margin_hat"], "lo": r["ci95_low"], "hi": r["ci95_high"]})
    head = scorer.score([payload], est)[0]
    e = scorer.propensity([payload])[0]
    one = row.to_frame().T
    tv = true_values(one.astype({c: float for c in one.columns if c in ("tau_margin_true", "tau_conv_true", "stay_value")}), cfg)
    values = {"SEND 20% OFF": float(tv["disc"][0]), "SEND 2,000 POINTS": float(tv["pts"][0]), "SUPPRESS": 0.0}
    return {
        "payload": payload, "estimator": est, "per_estimator": per_est, "head": head, "propensity": float(e),
        "w_if_mailed": float(1.0 / e), "w_if_not_mailed": float(1.0 / (1.0 - e)),
        "truth": {"persona": str(row["persona"]), "tau_margin": float(row["tau_margin_true"]),
                  "tau_conv": float(row["tau_conv_true"]), "stay_value": float(row["stay_value"]),
                  "p0": float(tv["p0_true"][0])},
        "values": values, "chosen_value": values[head["decision"]], "mass_blast_value": values["SEND 20% OFF"],
    }


def campaign_report(scorer, te: pd.DataFrame, dec_log: pd.DataFrame, est: str, n: int, thr: float, seed: int = 7) -> dict:
    cfg = scorer.cfg
    o = cfg["offer"]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(te), size=min(n, len(te)), replace=False))
    s = te.iloc[idx].reset_index(drop=True)
    t0 = time.perf_counter()
    res = pd.DataFrame(scorer.score(s[FEATURES].to_dict("records"), est, offer={"cate_threshold": thr}))
    ms = 1000 * (time.perf_counter() - t0)
    tv = true_values(s, cfg)
    R, d0 = s["stay_value"].values, o["discount_pct"]
    P = o["points_offer"] * o["points_cost_per_point"]
    y0 = s["Y0_conv"].values
    dec = res["decision"].values
    is_d, is_p = dec == "SEND 20% OFF", dec == "SEND 2,000 POINTS"
    send = is_d | is_p
    k = int(send.sum())
    resp = dec_log["resp_prob"].values[idx]
    prop_mask = np.zeros(len(s), bool)
    prop_mask[np.argsort(-resp)[:k]] = True
    cc = o["comm_cost"]

    def cann(disc_mask, pts_mask=None):
        v = float((y0[disc_mask] * R[disc_mask] * d0).sum())
        if pts_mask is not None:
            v += float((y0[pts_mask] * P).sum())
        return v

    pol = pd.DataFrame([
        {"Policy": "Mass Blast (20% off)", "Mailed": len(s), "Net incr. margin ($)": float(tv["disc"].sum()),
         "Wasted offers ($)": cann(np.ones(len(s), bool))},
        {"Policy": "No Promo", "Mailed": 0, "Net incr. margin ($)": 0.0, "Wasted offers ($)": 0.0},
        {"Policy": "Propensity Targeting (same budget)", "Mailed": k, "Net incr. margin ($)": float(tv["disc"][prop_mask].sum()),
         "Wasted offers ($)": cann(prop_mask)},
        {"Policy": f"Causal Uplift ({est})", "Mailed": k,
         "Net incr. margin ($)": float(tv["disc"][is_d].sum() + tv["pts"][is_p].sum()), "Wasted offers ($)": cann(is_d, is_p)},
    ])
    conf = pd.crosstab(s["persona"], res["quadrant"], normalize="index").round(3)
    mailed_mix = s.loc[send, "persona"].value_counts(normalize=True).round(3)
    supp_bad = s["persona"].isin(["Always-Taker", "Sleeping Dog"]).values
    return {"sample": s, "results": res, "policies": pol, "confusion": conf, "n": len(s), "n_send": k,
            "latency_ms_total": ms, "decision_counts": res["decision"].value_counts().to_dict(),
            "mailed_mix": mailed_mix.to_dict(),
            "harmful_suppressed": float((~send[supp_bad]).mean()) if supp_bad.any() else float("nan"),
            "cc": cc}
