"""Hospitality DGP with confounded treatment assignment, non-linear CATE and ground truth.

Structural equations (all confounders observed => unconfoundedness holds given X):
  X  : channel, room_tier, lead_time, adults, children, device, hist_spend, loyalty, los, resort
  T  ~ Bernoulli(e(X)),  logit e = a0 + engagement(X) + channel/lead/loyalty terms
  Y_anc    = mu0(X) + T * tau_anc(X) + eps          (continuous, $)
  Y_direct = 1[u < p0(X) + T * tau_dir(X)]           (binary, monotone coupling in u)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["OTA_Expedia", "OTA_Booking", "Direct_Web", "Corporate"]
TIERS = ["Standard", "Deluxe", "Suite"]
DEVICES = ["iOS", "Android", "Desktop"]
NUM_COLS = ["lead_time", "adults", "children", "hist_spend", "loyalty_tier", "los", "resort"]
CAT_COLS = ["channel", "room_tier", "device"]
RAW_COLS = CAT_COLS + NUM_COLS
TRUTH_COLS = ["mu0_anc", "mu1_anc", "tau_anc", "p0_direct", "p1_direct", "tau_direct", "e_true", "segment"]
SEGMENTS = ["Persuadable", "Always-Taker", "Sleeping Dog", "Lost Cause"]


def feature_columns() -> list[str]:
    cols = list(NUM_COLS)
    cols += [f"channel_{c}" for c in CHANNELS]
    cols += [f"tier_{c}" for c in TIERS]
    cols += [f"device_{c}" for c in DEVICES]
    cols += ["log_lead", "log_hist", "party_size"]
    return cols


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic one-hot design matrix with a fixed column order (used by training AND the API)."""
    out = pd.DataFrame(index=df.index)
    for c in NUM_COLS:
        out[c] = df[c].astype(float)
    for c in CHANNELS:
        out[f"channel_{c}"] = (df["channel"] == c).astype(float)
    for c in TIERS:
        out[f"tier_{c}"] = (df["room_tier"] == c).astype(float)
    for c in DEVICES:
        out[f"device_{c}"] = (df["device"] == c).astype(float)
    out["log_lead"] = np.log1p(df["lead_time"].astype(float))
    out["log_hist"] = np.log1p(df["hist_spend"].astype(float))
    out["party_size"] = (df["adults"] + df["children"]).astype(float)
    return out[feature_columns()]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def structural(df: pd.DataFrame) -> dict:
    """Deterministic parts of the DGP as a function of covariates (also used to build truth for personas)."""
    ch = df["channel"].values
    tier = df["room_tier"].values
    dev = df["device"].values
    lead = df["lead_time"].values.astype(float)
    hist = df["hist_spend"].values.astype(float)
    loy = df["loyalty_tier"].values.astype(float)
    los = df["los"].values.astype(float)
    kids = df["children"].values.astype(float)
    adults = df["adults"].values.astype(float)
    resort = df["resort"].values.astype(float)
    is_ota = np.isin(ch, ["OTA_Expedia", "OTA_Booking"]).astype(float)
    ios = (dev == "iOS").astype(float)
    andr = (dev == "Android").astype(float)
    tier_v = pd.Series(tier).map({"Standard": 0, "Deluxe": 1, "Suite": 2}).values.astype(float)
    llead = np.log1p(lead)
    lhist = np.log1p(hist)

    # ---- propensity: engaged / tech-savvy / affluent / long-lead guests are far likelier to be reached
    engage = 0.55 * ios + 0.25 * andr + 0.45 * (lhist - 5.0) + 0.45 * (llead - 3.0)
    logit_e = (-0.75 + 0.7 * engage + 0.35 * loy + 0.45 * is_ota * (ch == "OTA_Expedia")
               - 0.35 * (ch == "Corporate") + 0.2 * tier_v)
    e = np.clip(_sigmoid(logit_e), 0.04, 0.96)

    # ---- ancillary spend ($): baseline also driven by the confounders
    mu0 = (12 + 28 * tier_v + 6.5 * (lhist - 4) + 9 * loy + 2.5 * los + 5 * resort
           + 2.0 * kids + 4 * (llead - 3))
    mu0 = np.maximum(mu0, 5.0)
    # non-linear CATE: big for OTA guests on suites with long lead & mobile; ~0 for direct/corporate
    lead_sat = 1 / (1 + np.exp(-(lead - 21) / 12))            # saturating in lead time
    tau_anc = (is_ota * (8 + 16 * tier_v * lead_sat + 6 * (ios + 0.6 * andr) + 3 * resort * (los >= 3))
               + (ch == "Direct_Web") * (3 + 2 * tier_v)
               + (ch == "Corporate") * (-4.0 + 1.0 * tier_v)           # mildly annoyed business travellers
               - 5 * (kids >= 2) * (tier_v == 0)                        # family on a budget room: dislike upsell
               + 2.0 * np.minimum(los, 7) / 7 * is_ota)
    tau_anc = tau_anc * (1 - 0.08 * loy)                                # loyalty members already buy ancillaries

    # ---- direct re-booking probability within 180 days
    p0 = _sigmoid(-2.0 + 1.9 * (ch == "Direct_Web") + 1.2 * (ch == "Corporate") + 0.45 * loy
                  + 0.25 * (lhist - 5) + 0.15 * (llead - 3) + 0.25 * adults / 2)
    tau_dir_raw = (is_ota * (0.06 + 0.07 * ios + 0.04 * andr + 0.025 * loy + 0.03 * (lead > 30))
                   * (0.4 + 0.6 * p0 * 4)                                # more room to move if already warm
                   + 0.02 * (ch == "Direct_Web")
                   - 0.07 * (ch == "Corporate") * (1 + 0.5 * (loy >= 2))
                   - 0.03 * (kids >= 2) * (dev == "Desktop"))
    p1 = np.clip(p0 + tau_dir_raw, 0.005, 0.995)
    tau_dir = p1 - p0
    return dict(e=e, mu0=mu0, tau_anc=tau_anc, p0=p0, p1=p1, tau_dir=tau_dir)


def _segment(p0, tau_dir):
    seg = np.full(len(p0), "Lost Cause", dtype=object)
    seg[(p0 >= 0.45) & (tau_dir > -0.02)] = "Always-Taker"
    seg[tau_dir >= 0.08] = "Persuadable"
    seg[tau_dir <= -0.015] = "Sleeping Dog"
    return seg


def generate(n: int = 60000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channel = rng.choice(CHANNELS, n, p=[0.34, 0.30, 0.22, 0.14])
    ota = np.isin(channel, ["OTA_Expedia", "OTA_Booking"])
    tier_p = np.where(ota[:, None], [[0.55, 0.33, 0.12]], [[0.35, 0.40, 0.25]])
    tier = _vec_choice(rng, TIERS, tier_p)
    lead = np.where(channel == "Corporate", rng.gamma(2.0, 6.0, n), rng.gamma(2.2, 17.0, n)).round().clip(0, 365)
    adults = np.where(channel == "Corporate", 1, rng.choice([1, 2, 3, 4], n, p=[0.2, 0.55, 0.15, 0.10]))
    children = np.where(channel == "Corporate", 0, rng.choice([0, 1, 2, 3], n, p=[0.55, 0.2, 0.18, 0.07]))
    dev_p = np.where(ota[:, None], [[0.38, 0.32, 0.30]], [[0.30, 0.25, 0.45]])
    device = _vec_choice(rng, DEVICES, dev_p)
    loyalty = rng.choice([0, 1, 2, 3], n, p=[0.45, 0.25, 0.20, 0.10])
    loyalty = np.where(channel == "Direct_Web", np.minimum(loyalty + rng.integers(0, 2, n), 3), loyalty)
    has_hist = rng.random(n) < (0.35 + 0.18 * loyalty)
    hist = has_hist * rng.lognormal(4.2, 0.8, n) * (1 + 0.4 * loyalty)
    los = np.clip(rng.poisson(2.6, n) + 1, 1, 14)
    resort = rng.binomial(1, 0.45, n)

    df = pd.DataFrame(dict(channel=channel, room_tier=tier, lead_time=lead, adults=adults, children=children,
                           device=device, hist_spend=hist.round(2), loyalty_tier=loyalty, los=los, resort=resort))
    s = structural(df)
    T = (rng.random(n) < s["e"]).astype(int)
    eps = rng.normal(0, 14.0, n) * (0.6 + 0.4 * (df["room_tier"].map({"Standard": 0, "Deluxe": 1, "Suite": 2}).values))
    u = rng.random(n)
    y0a = s["mu0"] + eps
    y1a = s["mu0"] + s["tau_anc"] + eps
    y0d = (u < s["p0"]).astype(int)
    y1d = (u < s["p1"]).astype(int)

    df["treatment"] = T
    df["y_ancillary"] = np.where(T == 1, y1a, y0a).round(2)
    df["y_direct"] = np.where(T == 1, y1d, y0d).astype(int)
    df["mu0_anc"] = s["mu0"]
    df["mu1_anc"] = s["mu0"] + s["tau_anc"]
    df["tau_anc"] = s["tau_anc"]
    df["p0_direct"] = s["p0"]
    df["p1_direct"] = s["p1"]
    df["tau_direct"] = s["tau_dir"]
    df["e_true"] = s["e"]
    df["segment"] = _segment(s["p0"], s["tau_dir"])
    return df.reset_index(drop=True)


def _vec_choice(rng, labels, probs):
    r = rng.random(len(probs))[:, None]
    idx = (r > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, len(labels) - 1)
    return np.array(labels)[idx]


def split(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 42):
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(len(df)) < test_frac
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def true_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth structural quantities for arbitrary covariate rows (personas)."""
    s = structural(df)
    return pd.DataFrame({"e_true": s["e"], "tau_anc": s["tau_anc"], "tau_direct": s["tau_dir"], "p0_direct": s["p0"],
                         "segment": _segment(s["p0"], s["tau_dir"])})
