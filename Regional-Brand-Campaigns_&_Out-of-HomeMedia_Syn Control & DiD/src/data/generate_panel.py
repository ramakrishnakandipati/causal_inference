"""Synthetic 25-DMA x 240-day hospitality geo panel with known ground-truth counterfactuals.

Design
------
* Demand (room nights) follows a linear 3-factor model in levels
      y0_it = mu_i + sum_k L_ik f_kt + mu_i * sum_m B_im x_mit + local_trend_it + eps_it
  with factors  f1 = national demand, f2 = Sunbelt winter peak, f3 = urban / business cycle.
* Treated-candidate markets (MIA, LAS, NYC) are built as a *planted convex combination* of donor
  demand structure (+ a small level shift), so a synthetic control exists in principle, while their
  own weather / flight-search / competitor shocks stay idiosyncratic.
* The campaign effect passes spend -> geometric adstock -> Hill saturation -> multiplicative
  modifiers (weekend, holiday, flight-search surge, competitor ADR compression, weather, OTA pressure).
* y0 (no campaign) is generated once; scenarios only differ in which market is exposed and how.
  That gives exact ground truth for every preset, including a zero-effect placebo (TPA).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.config import ensure_dir, load_config, resolve, save_json
from src.utils.logger import get_logger

OUTCOMES = ["room_nights_booked", "direct_web_sessions", "branded_search_index", "incremental_net_revenue_usd"]
COVARIATES = ["competitor_adr_usd", "flight_search_index", "temp_max_f", "precipitation_in",
              "weather_severity_index", "is_weekend", "is_holiday_window", "seasonality_sin",
              "seasonality_cos", "ota_promo_pressure_index"]

# (code, name, region, tier, mu, (a1,a2,a3), comp_adr, own_adr, tmean, tamp, tsd, p_wet, wet_peak_doy,
#  wet_amp, mean_precip_in, storm_kind, leisure_index)
_DONORS = [
    ("FLL", "Fort Lauderdale", "Southeast", 2, 1400, (0.10, 0.20, 0.00), 330, 440, 83, 8, 3.5, 0.30, 250, 0.45, 0.55, "trop", 1.0),
    ("TPA", "Tampa", "Southeast", 2, 1100, (0.10, 0.17, 0.01), 300, 405, 82, 10, 4.5, 0.28, 200, 0.55, 0.50, "trop", 1.0),
    ("ORL", "Orlando", "Southeast", 1, 1700, (0.12, 0.07, 0.02), 290, 420, 83, 10, 4.5, 0.30, 200, 0.55, 0.55, "trop", 0.8),
    ("SAN", "San Diego", "West", 2, 1200, (0.10, 0.08, 0.04), 340, 455, 71, 6, 3.5, 0.10, 20, 0.60, 0.35, "none", 0.8),
    ("PHX", "Phoenix", "Southwest", 2, 1100, (0.10, 0.22, 0.02), 270, 400, 86, 22, 5.0, 0.07, 215, 0.60, 0.30, "none", 0.8),
    ("HNL", "Honolulu", "Pacific", 1, 1300, (0.08, 0.09, 0.00), 400, 540, 85, 3, 2.0, 0.35, 10, 0.30, 0.35, "none", 1.0),
    ("MSY", "New Orleans", "South Central", 3, 700, (0.10, 0.08, 0.05), 250, 380, 80, 14, 5.0, 0.33, 200, 0.40, 0.60, "trop", 0.7),
    ("CHS", "Charleston", "Southeast", 3, 450, (0.12, 0.04, 0.00), 280, 390, 76, 14, 5.0, 0.28, 210, 0.40, 0.50, "trop", 1.0),
    ("SAV", "Savannah", "Southeast", 3, 350, (0.12, 0.03, 0.00), 250, 360, 77, 14, 5.0, 0.28, 210, 0.40, 0.50, "trop", 1.0),
    ("BOS", "Boston", "Northeast", 1, 1500, (0.12, -0.05, 0.22), 300, 420, 59, 22, 5.0, 0.30, 150, 0.10, 0.50, "winter", -0.6),
    ("CHI", "Chicago", "Midwest", 1, 1900, (0.12, -0.07, 0.22), 280, 400, 58, 24, 6.0, 0.30, 170, 0.20, 0.50, "winter", -0.6),
    ("DCA", "Washington DC", "Mid-Atlantic", 1, 1700, (0.12, -0.03, 0.24), 300, 410, 66, 21, 5.0, 0.30, 150, 0.10, 0.50, "winter", -0.7),
    ("AUS", "Austin", "South Central", 2, 800, (0.12, 0.04, 0.14), 260, 380, 80, 17, 5.0, 0.22, 140, 0.30, 0.50, "none", -0.2),
    ("DFW", "Dallas-Fort Worth", "South Central", 1, 1500, (0.10, 0.02, 0.20), 250, 360, 77, 19, 5.0, 0.22, 130, 0.30, 0.50, "none", -0.5),
    ("IAH", "Houston", "South Central", 1, 1300, (0.10, 0.03, 0.18), 240, 350, 80, 16, 4.5, 0.28, 200, 0.30, 0.55, "trop", -0.5),
    ("LAX", "Los Angeles", "West", 1, 2600, (0.10, 0.06, 0.12), 330, 430, 75, 9, 4.0, 0.10, 20, 0.50, 0.40, "none", 0.3),
    ("SFO", "San Francisco", "West", 1, 1900, (0.10, -0.02, 0.20), 350, 450, 65, 6, 4.0, 0.20, 15, 0.60, 0.35, "none", -0.5),
    ("SEA", "Seattle", "Northwest", 2, 1200, (0.12, -0.05, 0.14), 290, 410, 60, 14, 4.5, 0.38, 350, 0.70, 0.30, "none", -0.2),
    ("DEN", "Denver", "Mountain", 2, 1100, (0.12, -0.02, 0.12), 260, 370, 64, 20, 7.0, 0.15, 130, 0.50, 0.30, "winter", 0.0),
    ("ATL", "Atlanta", "Southeast", 1, 1600, (0.10, 0.02, 0.18), 250, 370, 73, 17, 5.0, 0.28, 70, 0.30, 0.50, "none", -0.4),
    ("CLT", "Charlotte", "Southeast", 2, 700, (0.10, 0.01, 0.16), 230, 340, 71, 18, 5.0, 0.28, 200, 0.30, 0.50, "none", -0.4),
    ("BNA", "Nashville", "South Central", 2, 800, (0.12, 0.02, 0.12), 260, 370, 71, 19, 5.0, 0.28, 90, 0.30, 0.50, "winter", 0.2),
]
# markets built from planted donor combinations: (name, region, tier, comp_adr, tmean, tamp, tsd, p_wet,
#  wet_peak_doy, wet_amp, mean_precip, storm, leisure_index)
_DERIVED = {
    "MIA": ("Miami", "Southeast", 1, 360, 84, 7, 3.5, 0.32, 230, 0.50, 0.60, "trop", 1.0),
    "LAS": ("Las Vegas", "Southwest", 1, 240, 80, 24, 5.0, 0.05, 215, 0.60, 0.30, "none", 0.9),
    "NYC": ("New York", "Northeast", 1, 410, 63, 22, 5.5, 0.32, 200, 0.20, 0.50, "winter", -0.4),
}
DMA_ORDER = ["MIA", "FLL", "TPA", "ORL", "LAS", "SAN", "PHX", "NYC", "BOS", "CHI", "DFW", "IAH", "LAX", "SFO",
             "SEA", "DEN", "ATL", "CLT", "BNA", "AUS", "MSY", "CHS", "SAV", "HNL", "DCA"]

_HOLIDAYS = [  # inclusive date windows inside Jul 2025 - Feb 2026
    ("2025-07-02", "2025-07-06"), ("2025-08-29", "2025-09-02"), ("2025-10-11", "2025-10-14"),
    ("2025-11-24", "2025-11-30"), ("2025-12-20", "2026-01-03"), ("2026-01-16", "2026-01-20"),
    ("2026-02-13", "2026-02-17"),
]
_OTA_EVENTS = [  # (start, end, magnitude) national OTA sale pressure
    ("2025-07-08", "2025-07-11", 10), ("2025-08-28", "2025-09-02", 15), ("2025-11-24", "2025-12-01", 45),
    ("2025-12-26", "2026-01-03", 20), ("2026-02-12", "2026-02-17", 25),
]
NET_FACTOR = None  # set from cfg economics at runtime


# ----------------------------------------------------------------------------- helpers
def _ar1(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) with marginal standard deviation `sd`."""
    out = np.empty(n)
    out[0] = rng.normal(0, sd)
    innov = sd * np.sqrt(1 - rho**2)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + rng.normal(0, innov)
    return out


def _events(rng: np.random.Generator, T: int, rate: float, dur: tuple[int, int], mag: tuple[float, float]) -> np.ndarray:
    """Smooth hump-shaped episodes (rate cuts, search surges) at random times."""
    out = np.zeros(T)
    t = 0
    while t < T:
        if rng.random() < rate:
            d = int(rng.integers(dur[0], dur[1] + 1))
            m = rng.uniform(*mag)
            hump = np.sin(np.linspace(0, np.pi, d + 2))[1:-1]
            e = min(T, t + d)
            out[t:e] = np.maximum(out[t:e], m * hump[: e - t])
            t += d
        else:
            t += 1
    return out


def _storm_flags(rng: np.random.Generator, doy: np.ndarray, kind: str) -> np.ndarray:
    T = len(doy)
    if kind == "trop":
        hazard = np.where((doy >= 182) & (doy <= 305), 0.012, 0.0015)
    elif kind == "winter":
        hazard = np.where((doy <= 75) | (doy >= 335), 0.012, 0.0015)
    else:
        hazard = np.full(T, 0.0015)
    flags = np.zeros(T)
    t = 0
    while t < T:
        if rng.random() < hazard[t]:
            d = int(rng.integers(1, 4))
            flags[t: t + d] = 1.0
            t += d
        else:
            t += 1
    return flags


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _std(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def _sens(lei: float, scale: float) -> dict:
    """Relative demand sensitivities (fraction of mu per unit of driver)."""
    return {
        "weekend": 0.10 * lei, "holiday": 0.12 * lei,
        "flight": (0.016 + 0.006 * lei) * scale, "weather": -(0.010 + 0.004 * max(lei, 0.0)) * scale,
        "ota": -0.008 * scale, "cadr": 0.005 * scale,
    }


# ----------------------------------------------------------------------------- market table
def build_dma_table(cfg: dict) -> pd.DataFrame:
    """25-row market table incl. derived treated candidates (MIA, LAS, NYC)."""
    scale = cfg["dgp"]["covariate_effect_scale"]
    shift = cfg["dgp"]["derived_level_shift_frac"]
    rows: dict[str, dict] = {}
    for (code, name, region, tier, mu, a, cadr, adr, tm, ta, ts, pw, wpk, wamp, pamt, storm, lei) in _DONORS:
        rows[code] = dict(dma_code=code, dma_name=name, region=region, tier=tier, mu=float(mu),
                          L=np.array(a) * mu, cadr=cadr, adr=adr, tmean=tm, tamp=ta, tsd=ts, pwet=pw,
                          wpk=wpk, wamp=wamp, pamt=pamt, storm=storm, lei=lei, derived=False,
                          sens=_sens(lei, scale))
    for code, (name, region, tier, cadr, tm, ta, ts, pw, wpk, wamp, pamt, storm, lei) in _DERIVED.items():
        w = cfg["dgp"]["planted"][code]
        mu0 = sum(wj * rows[j]["mu"] for j, wj in w.items())
        L = sum(wj * rows[j]["L"] for j, wj in w.items())
        adr = sum(wj * rows[j]["adr"] for j, wj in w.items())
        sens = {k: sum(wj * rows[j]["sens"][k] for j, wj in w.items()) for k in rows["FLL"]["sens"]}
        rows[code] = dict(dma_code=code, dma_name=name, region=region, tier=tier, mu=mu0 * (1 + shift),
                          L=L, cadr=cadr, adr=adr, tmean=tm, tamp=ta, tsd=ts, pwet=pw, wpk=wpk, wamp=wamp,
                          pamt=pamt, storm=storm, lei=lei, derived=True, sens=sens)
    return pd.DataFrame([rows[c] for c in DMA_ORDER])


def build_calendar(cfg: dict) -> pd.DataFrame:
    T = cfg["panel"]["n_days"]
    start = dt.date.fromisoformat(cfg["panel"]["start_date"])
    dates = pd.to_datetime([start + dt.timedelta(days=i) for i in range(T)])
    cal = pd.DataFrame({"date": dates, "day_index": np.arange(1, T + 1)})
    cal["doy"] = cal["date"].dt.dayofyear
    cal["is_weekend"] = (cal["date"].dt.dayofweek >= 5).astype(float)
    hol = np.zeros(T)
    for a, b in _HOLIDAYS:
        hol[(cal["date"] >= a) & (cal["date"] <= b)] = 1.0
    cal["is_holiday_window"] = hol
    cal["seasonality_sin"] = np.sin(2 * np.pi * cal["doy"] / 365.25)
    cal["seasonality_cos"] = np.cos(2 * np.pi * cal["doy"] / 365.25)
    ota = np.zeros(T)
    for a, b, m in _OTA_EVENTS:
        ota[(cal["date"] >= a) & (cal["date"] <= b)] = m
    cal["ota_national"] = ota
    return cal


# ----------------------------------------------------------------------------- latent panel
@dataclass
class Latent:
    """Everything that does not depend on the campaign."""
    cfg: dict
    dmas: pd.DataFrame
    cal: pd.DataFrame
    cov: dict[str, np.ndarray]          # name -> (N,T)
    z: dict[str, np.ndarray]            # standardised effect drivers (N,T)
    y0: dict[str, np.ndarray]           # outcome -> (N,T)
    nrpr: np.ndarray                    # net revenue per room night (N,T)
    factors: np.ndarray                 # (T,3)
    seed: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def codes(self) -> list[str]:
        return list(self.dmas["dma_code"])


def build_latent(cfg: dict, seed: int | None = None) -> Latent:
    seed = cfg["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    dmas = build_dma_table(cfg)
    cal = build_calendar(cfg)
    N, T = len(dmas), len(cal)
    d = cfg["dgp"]
    doy = cal["doy"].to_numpy(float)

    # latent factors (standardised over the window)
    f1 = 0.8 * np.cos(2 * np.pi * (doy - 200) / 365.25) + _ar1(rng, T, 0.97, 0.9)
    f2 = np.cos(2 * np.pi * (doy - 20) / 365.25) + 0.35 * _ar1(rng, T, 0.92, 1.0)
    f3 = 0.8 * np.cos(2 * np.pi * (doy - 285) / 182.6) + 0.40 * _ar1(rng, T, 0.92, 1.0)
    F = np.column_stack([_std(f1), _std(f2), _std(f3)])

    regions = dmas["region"].unique()
    regional_temp = {r: _ar1(rng, T, 0.75, 3.0) for r in regions}
    nat_flight = _ar1(rng, T, 0.88, 1.0)

    cov = {k: np.zeros((N, T)) for k in COVARIATES}
    z = {k: np.zeros((N, T)) for k in ["flight", "cadr_comp", "weather", "ota"]}
    compress = np.zeros((N, T))
    y0 = {o: np.zeros((N, T)) for o in OUTCOMES}
    nrpr = np.zeros((N, T))

    net_factor = (1 + cfg["dgp"]["economics"]["ancillary_uplift"]) * (1 - cfg["dgp"]["economics"]["distribution_cost"])
    web_c, web_a, bs_a = 28.0, 1.25, 1.40
    mia_mu = float(dmas.loc[dmas["dma_code"] == "MIA", "mu"].iloc[0])
    bs_c = 100.0 / mia_mu

    for i, r in dmas.iterrows():
        L = np.asarray(r["L"])
        mu = r["mu"]
        demand_ix = (F @ L) / (np.abs(L).sum() + 1e-9)          # seasonal demand fundamentals in [-1,1]-ish

        # --- weather
        storm = _storm_flags(rng, doy, r["storm"])
        temp = (r["tmean"] - r["tamp"] * np.cos(2 * np.pi * (doy - 20) / 365.25)
                + 0.6 * regional_temp[r["region"]] + _ar1(rng, T, 0.7, r["tsd"])
                - np.where(storm > 0, 12.0 if r["storm"] == "winter" else 3.0, 0.0))
        p_wet = np.clip(r["pwet"] * (1 + r["wamp"] * np.cos(2 * np.pi * (doy - r["wpk"]) / 365.25)), 0.02, 0.75)
        wet = rng.random(T) < p_wet
        precip = np.where(wet, rng.gamma(0.9, r["pamt"] / 0.9, T), 0.0)
        precip = precip + storm * rng.uniform(1.0 if r["storm"] == "winter" else 2.0, 3.0 if r["storm"] == "winter" else 6.0, T)
        comfy = 78.0 if r["lei"] > 0 else 68.0
        raw = 1.8 * precip + 0.04 * np.maximum(0, np.abs(temp - comfy) - 15) ** 2 + 4.0 * storm
        severity = 100 * (1 - np.exp(-raw / 8.0))

        # --- competitor ADR with rate-cut (compression) episodes
        comp = _events(rng, T, 1 / 60, (5, 14), (0.06, 0.18))
        cadr = (r["cadr"] * (1 + 0.15 * demand_ix) * np.exp(_ar1(rng, T, 0.8, 0.02)) * (1 - comp))
        # --- flight search index: national sale shocks + DMA surges
        surge = _events(rng, T, 1 / 40, (4, 10), (15, 45))
        flight = np.clip(100 * (1 + 0.20 * demand_ix + 0.03 * nat_flight) + surge + _ar1(rng, T, 0.7, 4.0), 40, None)
        # --- OTA promo pressure
        ota = np.clip(30 + cal["ota_national"].to_numpy() * rng.uniform(0.8, 1.2) + _ar1(rng, T, 0.8, 6.0), 0, 100)

        cov["competitor_adr_usd"][i] = cadr
        cov["flight_search_index"][i] = flight
        cov["temp_max_f"][i] = temp
        cov["precipitation_in"][i] = precip
        cov["weather_severity_index"][i] = severity
        cov["is_weekend"][i] = cal["is_weekend"]
        cov["is_holiday_window"][i] = cal["is_holiday_window"]
        cov["seasonality_sin"][i] = cal["seasonality_sin"]
        cov["seasonality_cos"][i] = cal["seasonality_cos"]
        cov["ota_promo_pressure_index"][i] = ota
        z["flight"][i] = _zscore(flight)
        z["cadr_comp"][i] = _zscore(comp)
        z["weather"][i] = _zscore(severity)
        z["ota"][i] = _zscore(ota)
        compress[i] = comp

        # --- room nights (primary)
        s = r["sens"]
        rel = (s["weekend"] * (cal["is_weekend"].to_numpy() - 2 / 7)
               + s["holiday"] * (cal["is_holiday_window"].to_numpy() - cal["is_holiday_window"].mean())
               + s["flight"] * z["flight"][i] + s["weather"] * z["weather"][i]
               + s["ota"] * z["ota"][i] + s["cadr"] * _zscore(cadr))
        local = _ar1(rng, T, d["local_trend_ar"], d["local_trend_sd_frac"] * mu)
        eps = _ar1(rng, T, d["noise_ar"], d["noise_sd_frac"] * mu)
        rn = np.maximum(mu + F @ L + mu * rel + local + eps, 0.2 * mu)
        y0["room_nights_booked"][i] = rn

        # --- secondary outcomes share the demand structure (so a synthetic control exists for each)
        y0["direct_web_sessions"][i] = web_c * (mu + web_a * (rn - mu)) + _ar1(rng, T, 0.4, 0.035 * web_c * mu)
        y0["branded_search_index"][i] = bs_c * (mu + bs_a * (rn - mu)) + _ar1(rng, T, 0.4, 0.040 * bs_c * mu)
        dev = (rn - mu) / mu
        price = r["adr"] * (1 + 0.35 * dev) * np.exp(_ar1(rng, T, 0.8, 0.015))
        nrpr[i] = price * net_factor
        y0["incremental_net_revenue_usd"][i] = rn * nrpr[i]

    return Latent(cfg=cfg, dmas=dmas, cal=cal, cov=cov, z=z, y0=y0, nrpr=nrpr, factors=F, seed=seed,
                  extra={"compress": compress})


# ----------------------------------------------------------------------------- campaign
def spend_path(pattern: str, total: float, cal: pd.DataFrame, start_day: int, seed: int) -> np.ndarray:
    """Daily OOH+CTV spend (USD). Zero before `start_day`; flighted according to the pattern."""
    T = len(cal)
    out = np.zeros(T)
    if pattern == "none" or total <= 0:
        return out
    rng = np.random.default_rng(seed + 17)
    idx = np.arange(start_day - 1, T)
    n = len(idx)
    dates = cal["date"].iloc[idx]
    w = np.ones(n)

    def in_range(a: str, b: str) -> np.ndarray:
        return ((dates >= a) & (dates <= b)).to_numpy()

    k = np.arange(n)
    if pattern == "winter_airport":
        w += 1.6 * in_range("2025-12-28", "2026-01-04") + 1.6 * in_range("2026-02-12", "2026-02-17")   # airport spectaculars
        w += 0.6 * (((k // 7) % 2) == 1)                                                             # billboard flights
        w *= np.where(k < 7, 1.35, 1.0)                                                              # launch ramp
    elif pattern == "convention_billboard":
        w = 0.8 + 2.0 * in_range("2026-01-02", "2026-01-09") + 1.2 * in_range("2026-02-05", "2026-02-12")
        w += 0.3 * (dates.dt.dayofweek.to_numpy() < 4)
    elif pattern == "awareness_push":
        w = 0.5 + 1.6 * np.exp(-k / 35.0) + 0.8 * in_range("2026-02-13", "2026-02-17")
    else:
        raise ValueError(f"unknown pattern {pattern}")
    w = w * np.exp(rng.normal(0, 0.06, n))
    out[idx] = total * w / w.sum()
    return out


def _hill(A: np.ndarray, K: float, s: float) -> np.ndarray:
    A = np.maximum(A, 0.0)
    return A**s / (A**s + K**s)


def adstock(spend_k: np.ndarray, theta: float) -> np.ndarray:
    A = np.zeros_like(spend_k, dtype=float)
    carry = 0.0
    for t, sp in enumerate(spend_k):
        carry = sp + theta * carry
        A[t] = carry
    return A


def effect_modifier(lat: Latent, i: int) -> np.ndarray:
    g = lat.cfg["dgp"]["interactions"]
    cal = lat.cal
    m = np.exp(g["is_weekend"] * (cal["is_weekend"].to_numpy() - 2 / 7)
               + g["is_holiday_window"] * (cal["is_holiday_window"].to_numpy() - cal["is_holiday_window"].mean())
               + g["flight_search_z"] * lat.z["flight"][i]
               + g["competitor_compression_z"] * lat.z["cadr_comp"][i]
               + g["weather_severity_z"] * lat.z["weather"][i]
               + g["ota_promo_z"] * lat.z["ota"][i])
    return np.clip(m, 0.4, 2.2)


def campaign_effects(lat: Latent, treated: str, spend: np.ndarray, start_day: int,
                     target: dict[str, float]) -> dict[str, np.ndarray]:
    """True tau_t per outcome for the treated market (zeros before the campaign)."""
    T = len(lat.cal)
    i = lat.codes.index(treated)
    out = {o: np.zeros(T) for o in OUTCOMES}
    if spend.sum() <= 0:
        return out
    h = lat.cfg["dgp"]["hill"]
    th = lat.cfg["dgp"]["adstock_theta"]
    M = effect_modifier(lat, i)
    post = np.arange(T) >= start_day - 1
    for o in ["room_nights_booked", "direct_web_sessions", "branded_search_index"]:
        A = adstock(spend / 1000.0, th[o])
        S = _hill(A, h["K"], h["s"])
        base = lat.y0[o][i]
        shape = S * M * base
        k = target[o] * base[post].sum() / max(shape[post].sum(), 1e-9)
        out[o] = np.where(post, k * shape, 0.0)
    out["incremental_net_revenue_usd"] = out["room_nights_booked"] * lat.nrpr[i]
    return out


def assemble_panel(lat: Latent, scenario: dict, seed: int | None = None) -> pd.DataFrame:
    """Long panel (N*T rows) for one scenario, incl. ground-truth potential outcomes."""
    cfg = lat.cfg
    seed = lat.seed if seed is None else seed
    cal, dmas = lat.cal, lat.dmas
    N, T = len(dmas), len(cal)
    treated = scenario["treated"]
    start_day = int(scenario["start_day"])
    spend = spend_path(scenario["pattern"], float(scenario["total_spend_usd"]), cal, start_day, seed)
    lift = float(scenario["target_mean_lift"])
    target = {"room_nights_booked": lift, "direct_web_sessions": lift * cfg["dgp"]["target_mean_lift"]["direct_web_sessions"] / cfg["dgp"]["target_mean_lift"]["room_nights_booked"],
              "branded_search_index": lift * cfg["dgp"]["target_mean_lift"]["branded_search_index"] / cfg["dgp"]["target_mean_lift"]["room_nights_booked"]}
    tau = campaign_effects(lat, treated, spend, start_day, target)
    ti = lat.codes.index(treated)

    frames = []
    for i, r in dmas.iterrows():
        is_t = i == ti
        active = (np.arange(T) >= start_day - 1) & is_t & (spend.sum() > 0)
        df = pd.DataFrame({
            "dma_code": r["dma_code"], "dma_name": r["dma_name"], "region": r["region"], "tier": int(r["tier"]),
            "date": cal["date"], "day_index": cal["day_index"],
            "post_period": (cal["day_index"] >= start_day).astype(int),
            "treated_unit": int(is_t),
            "cohort_start_day": start_day if is_t else 0,
            "treatment_active": active.astype(int),
            "ooh_ctv_spend_usd": np.round(spend if is_t else np.zeros(T), 2),
        })
        for o in OUTCOMES:
            att = np.where(active, tau[o], 0.0) if is_t else np.zeros(T)
            df[o] = np.round(lat.y0[o][i] + att, 3)
        for c in COVARIATES:
            df[c] = np.round(lat.cov[c][i], 4)
        att_rn = np.where(active, tau["room_nights_booked"], 0.0)
        df["y0_true"] = np.round(lat.y0["room_nights_booked"][i], 3)
        df["y1_true"] = np.round(lat.y0["room_nights_booked"][i] + att_rn, 3)
        df["true_att_it"] = np.round(att_rn, 3)
        for o in OUTCOMES[1:]:
            df[f"y0_true_{o}"] = np.round(lat.y0[o][i], 3)
            df[f"true_att_{o}"] = np.round(np.where(active, tau[o], 0.0) if is_t else np.zeros(T), 3)
        df["net_revenue_per_room_night"] = np.round(lat.nrpr[i], 3)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel["dma_code"] = panel["dma_code"].astype("category")
    return panel


def scenario_specs(cfg: dict) -> dict[str, dict]:
    sc = {}
    for name, s in cfg["dgp"]["scenarios"].items():
        s = dict(s)
        s.setdefault("start_day", cfg["panel"]["campaign_start_day"])
        sc[name] = s
    return sc


def generate_base_panel(cfg: dict | None = None, seed: int | None = None, lat: Latent | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    lat = lat or build_latent(cfg, seed)
    return assemble_panel(lat, scenario_specs(cfg)["mia_flagship"], seed)


def generate_scenarios(cfg: dict | None = None, seed: int | None = None, lat: Latent | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    lat = lat or build_latent(cfg, seed)
    return {name: assemble_panel(lat, spec, seed) for name, spec in scenario_specs(cfg).items()}


def wide(df: pd.DataFrame, col: str, codes: list[str] | None = None) -> pd.DataFrame:
    """DMA x day matrix (rows = DMA, columns = day_index)."""
    w = df.pivot_table(index="dma_code", columns="day_index", values=col, observed=True)
    if codes is not None:
        w = w.loc[codes]
    return w


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    log = get_logger("data")
    lat = build_latent(cfg)
    specs = scenario_specs(cfg)
    base = assemble_panel(lat, specs["mia_flagship"])
    ensure_dir(resolve(cfg["panel"]["parquet_path"]).parent)
    base.to_parquet(resolve(cfg["panel"]["parquet_path"]), index=False)
    base.to_csv(resolve(cfg["panel"]["csv_path"]), index=False)
    sdir = ensure_dir(cfg["panel"]["scenario_dir"])
    meta = {}
    for name, spec in specs.items():
        df = base if name == "mia_flagship" else assemble_panel(lat, spec)
        df.to_parquet(sdir / f"{name}.parquet", index=False)
        meta[name] = spec
    save_json(meta, sdir / "scenarios_meta.json")
    post = base[(base.treated_unit == 1) & (base.treatment_active == 1)]
    log.info("panel: %d rows (%d DMAs x %d days) -> %s", len(base), base.dma_code.nunique(), base.day_index.nunique(),
             cfg["panel"]["parquet_path"])
    log.info("MIA campaign: spend $%.2fM, mean true ATT %.1f room nights/day (%.1f%% of counterfactual), total %.0f room nights",
             post.ooh_ctv_spend_usd.sum() / 1e6, post.true_att_it.mean(),
             100 * post.true_att_it.sum() / post.y0_true.sum(), post.true_att_it.sum())
    return base


if __name__ == "__main__":
    main()
