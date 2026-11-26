"""Synthetic hotel-loyalty panel with a known Gold-tier effect at the 20-night cutoff.

Design notes
------------
* Running variable R (qualifying nights in year t) is drawn from a smooth lognormal
  mixture, so the density is continuous at the cutoff unless ``manipulation=True``.
* Covariates are smooth functions of R, so they show no jump at the cutoff, but they
  correlate with both R and the outcomes (selection bias a naive comparison would pick up).
* Treatment effects tau(X) are calibrated so the kernel-weighted average in a window
  around the cutoff equals the target LATE exactly (+3.4 nights, +$420, +0.11).
  In the fuzzy regime the average is weighted by complier probability, so the target is
  the LATE for compliers.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

CUTOFF = 20

TARGETS: Dict[str, float] = {
    "next_year_room_nights": 3.4,
    "next_year_ancillary_spend": 420.0,
    "next_year_direct_booking_ratio": 0.11,
}

COVARIATES = [
    "historical_adr", "business_travel_ratio", "lead_time_avg", "ancillary_spend_t",
    "direct_booking_ratio_t", "tenure_years", "co_brand_card_holder", "home_region",
    "age_group", "weekend_stay_ratio",
]

HOME_REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"]
AGE_GROUPS = ["18-29", "30-39", "40-49", "50-64", "65+"]

# Fuzzy regime: P(T=1 | R < c) is ~12% on average (co-brand fast track / status match),
# P(T=1 | R >= c) is 94% (enrollment sync lag).
P_ABOVE = 0.94
P_BELOW_BASE = 0.055
P_BELOW_COBRAND = 0.245   # mean(cobrand)~0.27 -> overall ~0.12


def _tri_weights(r: np.ndarray, c: float, h: float) -> np.ndarray:
    return np.clip(1.0 - np.abs(r - c) / h, 0.0, None)


def generate(
    n: int = 50_000,
    regime: str = "fuzzy",
    manipulation: bool = False,
    cutoff: int = CUTOFF,
    seed: Optional[int] = 42,
    r_min: int = 1,
    r_max: int = 45,
) -> pd.DataFrame:
    """Generate the loyalty panel. ``regime`` is 'sharp' or 'fuzzy'."""
    if regime not in ("sharp", "fuzzy"):
        raise ValueError("regime must be 'sharp' or 'fuzzy'")
    rng = np.random.default_rng(seed)

    # ---- guest covariates ------------------------------------------------------------
    business = rng.beta(2.0, 4.0, n)                                # mean ~0.33
    home_region = rng.choice(5, n, p=[0.38, 0.27, 0.20, 0.08, 0.07])
    age_group = rng.choice(5, n, p=[0.14, 0.27, 0.27, 0.24, 0.08])
    tenure = rng.gamma(2.2, 2.0, n).clip(0.1, 30)
    cobrand = (rng.random(n) < (0.18 + 0.05 * (age_group >= 1) + 0.03 * np.minimum(tenure, 10) / 10 * 3)).astype(int)
    region_adr = np.array([1.0, 1.08, 0.92, 0.80, 1.12])[home_region]
    adr = np.exp(rng.normal(np.log(140), 0.30, n) + 0.35 * business) * region_adr
    adr = adr.clip(45, 900)
    lead = rng.gamma(2.0, 12.0, n) * (1.0 + 0.9 * (1 - business))   # leisure books earlier
    lead = lead.clip(0, 300)
    weekend = np.clip(rng.beta(2.0, 3.0, n) * (1.5 - business) * 0.9, 0, 1)

    # ---- running variable: smooth lognormal with seasonal modulation -------------------
    season = 1.0 + 0.12 * np.sin(2 * np.pi * rng.random(n))
    mu = np.log(13.5) + 0.38 * business + 0.015 * np.minimum(tenure, 15) + 0.10 * cobrand
    r_cont = np.exp(rng.normal(mu, 0.52)) * season
    r = np.rint(r_cont).astype(int)
    # truncate to [r_min, r_max] by redrawing (avoids artificial piles at the edges)
    for _ in range(20):
        bad = (r < r_min) | (r > r_max)
        if not bad.any():
            break
        k = int(bad.sum())
        r_cont_new = np.exp(rng.normal(mu[bad], 0.52)) * (1.0 + 0.12 * np.sin(2 * np.pi * rng.random(k)))
        r[bad] = np.rint(r_cont_new).astype(int)
    r = np.clip(r, r_min, r_max)

    if manipulation:
        # guests 1-3 nights short pay for extra stays that land them just above the cutoff
        near = (r >= cutoff - 3) & (r < cutoff)
        shift = near & (rng.random(n) < 0.40)
        r[shift] = cutoff + rng.integers(0, 2, int(shift.sum()))

    # ---- year-t behaviour, smooth in R -------------------------------------------------
    ancillary_t = np.clip(
        60 + 14.0 * r + 1.6 * adr * (1 - 0.3 * business) + rng.gamma(2.0, 90.0, n), 0, None)
    direct_t = np.clip(
        0.22 + 0.012 * np.minimum(tenure, 12) + 0.10 * cobrand + 0.004 * r
        + 0.06 * business + rng.normal(0, 0.10, n), 0.0, 0.95)

    # ---- treatment ---------------------------------------------------------------------
    z = (r >= cutoff).astype(int)
    p0 = P_BELOW_BASE + P_BELOW_COBRAND * cobrand
    if regime == "sharp":
        t = z.copy()
        compliance = np.zeros(n)
        comp_w = np.ones(n)
        ctype = np.where(z == 1, "complier", "complier")
    else:
        u = rng.random(n)
        t = np.where(z == 1, (u < P_ABOVE), (u < p0)).astype(int)
        comp_w = P_ABOVE - p0                       # share of compliers given X
        ctype = np.where(u < p0, "always_taker", np.where(u < P_ABOVE, "complier", "never_taker"))
        compliance = comp_w

    # ---- heterogeneity: high for mid-tier leisure / bleisure, low for rigid corporate ----
    mid_adr = np.exp(-(((adr - 165.0) / 75.0) ** 2))
    leisure = (1.0 - business) ** 1.5
    bleisure_bump = np.exp(-(((business - 0.40) / 0.15) ** 2))
    h = 0.15 + 0.9 * mid_adr * (0.35 * leisure + 0.65 * bleisure_bump + 0.25) \
        + 0.25 * weekend + 0.10 * (age_group <= 2) - 0.10 * cobrand
    h = np.clip(h, 0.05, None)
    h_anc = h * (adr / 150.0) ** 0.8
    h_dir = h ** 0.7 * (1.0 - direct_t) ** 0.8

    cal_w = _tri_weights(r, cutoff, 3.0) * comp_w
    def calibrated(shape: np.ndarray, target: float) -> np.ndarray:
        return target * shape / np.average(shape, weights=cal_w)

    tau_rn = calibrated(h, TARGETS["next_year_room_nights"])
    tau_anc = calibrated(h_anc, TARGETS["next_year_ancillary_spend"])
    tau_dir = calibrated(h_dir, TARGETS["next_year_direct_booking_ratio"])

    # ---- outcomes (smooth baseline + tau * T + noise) ----------------------------------
    rn0 = (2.0 + 0.62 * r - 0.004 * r ** 2 + 3.0 * business + 0.12 * np.minimum(tenure, 15)
           + 0.9 * cobrand + 0.008 * (adr - 140) + 0.9 * weekend + rng.normal(0, 3.6, n))
    rn0 = np.clip(rn0, 0, None)
    anc0 = (150 + 38.0 * r + 0.55 * ancillary_t + 1.1 * (adr - 140) + 60 * cobrand
            + 120 * weekend + rng.normal(0, 420, n))
    anc0 = np.clip(anc0, 0, None)
    dir0 = (0.06 + 0.55 * direct_t + 0.0025 * r + 0.07 * cobrand + 0.003 * np.minimum(tenure, 12)
            + rng.normal(0, 0.07, n))
    dir0 = np.clip(dir0, 0.0, 0.88)

    y_rn = rn0 + tau_rn * t
    y_anc = anc0 + tau_anc * t
    y_dir = np.clip(dir0 + tau_dir * t, 0.0, 1.0)

    df = pd.DataFrame({
        "member_id": np.arange(n),
        "qualifying_nights_t": r,
        "gold_eligible": z,
        "gold_status": t,
        "historical_adr": adr.round(2),
        "business_travel_ratio": business.round(4),
        "lead_time_avg": lead.round(1),
        "ancillary_spend_t": ancillary_t.round(2),
        "direct_booking_ratio_t": direct_t.round(4),
        "tenure_years": tenure.round(2),
        "co_brand_card_holder": cobrand,
        "home_region": home_region,
        "age_group": age_group,
        "weekend_stay_ratio": weekend.round(4),
        "next_year_room_nights": y_rn.round(2),
        "next_year_ancillary_spend": y_anc.round(2),
        "next_year_direct_booking_ratio": y_dir.round(4),
        "tau_true_room_nights": tau_rn,
        "tau_true_ancillary_spend": tau_anc,
        "tau_true_direct_booking_ratio": tau_dir,
        "compliance_prob": compliance,
        "compliance_type": ctype,
    })
    df.attrs.update({"regime": regime, "cutoff": cutoff, "manipulation": manipulation,
                     "targets": dict(TARGETS)})
    return df


def local_truth(df: pd.DataFrame, bandwidth: float = 3.0) -> Dict[str, float]:
    """Kernel-weighted true LATE near the cutoff (what an RDD should recover)."""
    c = df.attrs.get("cutoff", CUTOFF)
    w = _tri_weights(df["qualifying_nights_t"].to_numpy(float), c, bandwidth)
    if df.attrs.get("regime") == "fuzzy":
        w = w * df["compliance_prob"].to_numpy()
    cols = {"next_year_room_nights": "tau_true_room_nights",
            "next_year_ancillary_spend": "tau_true_ancillary_spend",
            "next_year_direct_booking_ratio": "tau_true_direct_booking_ratio"}
    return {y: float(np.average(df[c_], weights=w)) for y, c_ in cols.items()}


TAU_COLUMN = {
    "next_year_room_nights": "tau_true_room_nights",
    "next_year_ancillary_spend": "tau_true_ancillary_spend",
    "next_year_direct_booking_ratio": "tau_true_direct_booking_ratio",
}


if __name__ == "__main__":
    for reg in ("sharp", "fuzzy"):
        d = generate(regime=reg)
        print(reg, d.shape, local_truth(d))
        print(d.groupby(d.qualifying_nights_t >= CUTOFF)["gold_status"].mean().round(3).to_dict())
