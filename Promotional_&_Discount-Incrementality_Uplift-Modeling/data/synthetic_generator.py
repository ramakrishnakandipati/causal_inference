"""Hospitality causal data-generating process with ground-truth counterfactuals.

Five latent guest personas drive both the covariates and the heterogeneous treatment effect:
  Persuadable   : moderate recency, high weekend ratio, high price sensitivity  -> tau > 0
  Always-Taker  : frequent Platinum guests with short lead times                -> tau_conv ~ 0, tau_margin < 0
  Lost Cause    : dormant (>500 days)                                           -> tau ~ 0
  Sleeping Dog  : high cancellation / low price sensitivity                     -> tau < 0
  Neutral       : everyone else, small positive effect

Treatment assignment is confounded (engaged, loyal, price-sensitive guests are more likely to be mailed)
except for a randomized holdout (`rct == 1`, T ~ Bernoulli(0.5)).
Counterfactual conversions share a common uniform draw (monotone coupling), so Y1 and Y0 are
consistent potential outcomes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import FEATURES, PERSONAS, load_config, resolve, get_logger  # noqa: E402

log = get_logger()


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n: int = 50_000, seed: int = 42, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    off = cfg["offer"]
    rng = np.random.default_rng(seed)

    # ---- latent persona -------------------------------------------------------------
    probs = np.array([0.22, 0.18, 0.22, 0.10, 0.28])
    pid = rng.choice(len(PERSONAS), size=n, p=probs)
    P = {name: pid == i for i, name in enumerate(PERSONAS)}

    def mix(per_persona: dict, default, sd):
        mu = np.full(n, float(default))
        for k, v in per_persona.items():
            mu[P[k]] = v
        return mu + rng.normal(0, sd, n)

    # ---- covariates -----------------------------------------------------------------
    freq = np.clip(mix({"Persuadable": 3.0, "Always-Taker": 11.0, "Lost Cause": 0.9,
                        "Sleeping Dog": 5.0, "Neutral": 3.5}, 3.5, 1.2) * rng.lognormal(0, 0.2, n), 0.1, 40)
    days = np.where(P["Lost Cause"], rng.uniform(500, 1100, n),
           np.where(P["Persuadable"], rng.gamma(6, 25, n) + 60,
           np.where(P["Always-Taker"], rng.gamma(2, 20, n) + 5,
           np.where(P["Sleeping Dog"], rng.gamma(3, 40, n) + 20, rng.gamma(3, 70, n) + 30))))
    days = np.clip(days, 1, 1500)

    tier_p = {"Persuadable": [0.30, 0.40, 0.22, 0.08], "Always-Taker": [0.02, 0.10, 0.28, 0.60],
              "Lost Cause": [0.60, 0.28, 0.09, 0.03], "Sleeping Dog": [0.10, 0.30, 0.40, 0.20],
              "Neutral": [0.40, 0.32, 0.20, 0.08]}
    tier = np.zeros(n, dtype=int)
    for name, p in tier_p.items():
        m = P[name]
        tier[m] = rng.choice(4, size=m.sum(), p=p)

    sens = np.clip(mix({"Persuadable": 0.75, "Always-Taker": 0.20, "Lost Cause": 0.40,
                        "Sleeping Dog": 0.12, "Neutral": 0.40}, 0.4, 0.12), 0, 1)
    lead = np.clip(mix({"Persuadable": 30, "Always-Taker": 6, "Lost Cause": 35,
                        "Sleeping Dog": 40, "Neutral": 25}, 25, 8), 0, 180)
    adr = np.clip(mix({"Persuadable": 150, "Always-Taker": 240, "Lost Cause": 120,
                       "Sleeping Dog": 210, "Neutral": 160}, 160, 30) + 15 * tier, 60, 600)
    wk = np.clip(mix({"Persuadable": 0.75, "Always-Taker": 0.25, "Lost Cause": 0.40,
                      "Sleeping Dog": 0.35, "Neutral": 0.40}, 0.4, 0.12), 0, 1)
    canc = np.clip(mix({"Persuadable": 0.08, "Always-Taker": 0.05, "Lost Cause": 0.10,
                        "Sleeping Dog": 0.38, "Neutral": 0.10}, 0.1, 0.05), 0, 0.9)
    openr = np.clip(mix({"Persuadable": 0.45, "Always-Taker": 0.55, "Lost Cause": 0.08,
                         "Sleeping Dog": 0.35, "Neutral": 0.30}, 0.3, 0.12), 0, 1)
    pts = np.clip(rng.gamma(2, 1500, n) * (1 + 0.8 * tier) * (0.3 + freq / 6), 0, 250_000)

    X = pd.DataFrame({
        "hist_booking_freq": freq.round(2), "days_since_last_stay": days.round(0),
        "loyalty_tier": tier, "hist_price_sensitivity": sens.round(3), "avg_lead_time": lead.round(1),
        "avg_adr": adr.round(2), "weekend_stay_ratio": wk.round(3), "cancellation_rate": canc.round(3),
        "email_open_rate_90d": openr.round(3), "points_balance": pts.round(0),
    })[FEATURES]

    # ---- potential conversion probabilities (30 days) --------------------------------
    base = _sigmoid(-2.0 + 0.25 * np.log1p(freq) * 2 - 0.004 * np.minimum(days, 700) / 5
                    + 0.3 * tier - 0.5 * canc + 0.8 * openr)
    p0 = np.select(
        [P["Persuadable"], P["Always-Taker"], P["Lost Cause"], P["Sleeping Dog"]],
        [0.07 + 0.12 * base, 0.50 + 0.25 * base, 0.01 + 0.03 * base, 0.18 + 0.15 * base],
        default=0.06 + 0.20 * base)
    tau_c = np.select(
        [P["Persuadable"], P["Always-Taker"], P["Lost Cause"], P["Sleeping Dog"]],
        [0.12 + 0.20 * sens * wk + 0.08 * (openr > 0.3),   # clear positive lift
         0.015 + 0.02 * sens,                              # would have booked anyway
         0.004 + 0.01 * openr,
         -(0.05 + 0.15 * canc) - 0.04 * (1 - sens)],        # cancels full-rate stay and rebooks / unsubscribes
        default=0.03 + 0.04 * sens)
    p1 = np.clip(p0 + tau_c, 0.001, 0.99)
    p0 = np.clip(p0, 0.001, 0.99)
    tau_conv = p1 - p0

    # ---- stay value & margin -----------------------------------------------------------
    nights = 1 + (wk > 0.5).astype(int) + rng.poisson(0.8, n)
    revenue = adr * nights                                    # full-rate stay revenue
    m, d = off["margin_rate"], off["discount_pct"]
    u = rng.uniform(size=n)                                   # shared uniform => coupled counterfactuals
    conv0 = (u < p0).astype(int)
    conv1 = (u < p1).astype(int)
    margin0 = conv0 * revenue * m
    margin1 = conv1 * revenue * (m - d)                       # discounted booking nets (m - d) of full rate
    noise = rng.normal(0, 2.0, n)                              # small incidental noise
    y0_margin, y1_margin = margin0 + noise, margin1 + noise
    tau_margin = y1_margin - y0_margin                         # realised individual effect
    tau_margin_expected = p1 * revenue * (m - d) - p0 * revenue * m   # conditional expectation (primary truth)

    # ---- confounded treatment assignment + randomized holdout --------------------------------
    logit_e = (-0.9 + 2.0 * (openr - 0.3) + 0.35 * tier + 0.9 * (sens - 0.4)
               + 0.04 * np.minimum(freq, 15) - 0.0006 * np.minimum(days, 700))
    e_x = np.clip(_sigmoid(logit_e), 0.05, 0.95)
    rct = (rng.uniform(size=n) < cfg["data"]["rct_fraction"]).astype(int)
    e_true = np.where(rct == 1, 0.5, e_x)
    T = (rng.uniform(size=n) < e_true).astype(int)

    df = X.copy()
    df["T"] = T
    df["Y_margin"] = np.where(T == 1, y1_margin, y0_margin).round(2)
    df["Y_conv"] = np.where(T == 1, conv1, conv0)
    df["rct"] = rct
    df["e_true"] = e_true
    df["Y0_margin"], df["Y1_margin"] = y0_margin.round(2), y1_margin.round(2)
    df["Y0_conv"], df["Y1_conv"] = conv0, conv1
    df["tau_margin_true"] = tau_margin_expected.round(4)       # E[Y1-Y0|X,persona]; used for PEHE
    df["tau_margin_realised"] = tau_margin.round(2)
    df["tau_conv_true"] = tau_conv.round(4)
    df["stay_value"] = revenue.round(2)
    df["persona"] = np.array(PERSONAS)[pid]                   # NEVER a model feature
    return df


def self_check(df: pd.DataFrame) -> dict:
    g = df.groupby("persona")[["tau_conv_true", "tau_margin_true"]].mean()
    assert g.loc["Persuadable", "tau_margin_true"] > 0, "Persuadables must have positive margin uplift"
    assert g.loc["Always-Taker", "tau_conv_true"] < 0.05 and g.loc["Always-Taker", "tau_margin_true"] < 0
    assert abs(g.loc["Lost Cause", "tau_margin_true"]) < 2.0
    assert g.loc["Sleeping Dog", "tau_conv_true"] < 0
    rct = df[df.rct == 1]
    assert abs(rct["T"].mean() - 0.5) < 0.03
    assert df.e_true.between(0.05, 0.95).all()
    return g.round(4).to_dict()


def main(n: int | None = None, seed: int | None = None, out: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    n = n or cfg["data"]["n_samples"]
    seed = seed if seed is not None else cfg["seed"]
    df = generate(n, seed, cfg)
    summary = self_check(df)
    path = resolve(out or cfg["data"]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Generated %d guests -> %s | treated=%.1f%% | true ATE margin=$%.2f | conv ATE=%.3f",
             len(df), path.name, 100 * df["T"].mean(), df.tau_margin_true.mean(), df.tau_conv_true.mean())
    log.info("Persona mean tau (conv, margin): %s", summary)
    return df


if __name__ == "__main__":
    main()
