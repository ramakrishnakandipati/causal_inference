"""Synthetic hospitality DGP with an unobserved demand shock U, three valid instruments Z
and ground-truth heterogeneous elasticity / promo CATE.

Run:  python -m data.generate_hospitality_data [--n 30000]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.gpu_utils import hardware_summary, load_config, resolve, set_seed  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import dump_json, first_stage_f  # noqa: E402
from src.utils.structural import expected_occupancy, net_revpar  # noqa: E402

SEGMENTS = ["leisure", "corporate", "group"]
TIERS = ["luxury", "urban_business", "resort"]
ROOM_MULT = np.array([1.0, 1.25, 1.70])  # standard, deluxe, suite

INSTRUMENTS = ["competitor_room_outage_pct", "airline_seat_capacity_shock", "pricing_algo_rounding_threshold"]
FEATURES = [
    "lead_time", "los", "dow", "is_weekend", "seasonality", "loyalty_tier", "hist_adr_sensitivity",
    "room_type", "tier_luxury", "tier_urban_business", "tier_resort",
    "seg_leisure", "seg_corporate", "seg_group",
]
TREATMENT = "ln_net_adr"
OUTCOME = "log_demand"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def add_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add the one-hot design columns used by every model from the categorical columns."""
    df = df.copy()
    df["is_weekend"] = df["dow"].isin([4, 5]).astype(float)
    for t in TIERS:
        df[f"tier_{t}"] = (df["property_tier"] == t).astype(float)
    for s in SEGMENTS:
        df[f"seg_{s}"] = (df["segment"] == s).astype(float)
    return df


def structural_theta(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    """Ground-truth price elasticity theta(X) of demand (always negative)."""
    d = cfg["dgp"]
    base = df["segment"].map(d["theta"]).to_numpy(float)
    lead_z = np.clip((df["lead_time"].to_numpy(float) - 30.0) / 30.0, -1.0, 3.0)
    tier_adj = df["property_tier"].map({"luxury": 0.25, "urban_business": 0.0, "resort": -0.10}).to_numpy(float)
    theta = (
        base
        - 0.25 * lead_z
        + 0.18 * df["loyalty_tier"].to_numpy(float)
        - 0.30 * df["hist_adr_sensitivity"].to_numpy(float)
        + tier_adj
        + 0.10 * df["room_type"].to_numpy(float)
    )
    lo, hi = d["theta_clip"]
    return np.clip(theta, lo, hi)


def ln_reference_price(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    ref = df["property_tier"].map(cfg["dgp"]["price_ref"]).to_numpy(float)
    return np.log(ref * ROOM_MULT[df["room_type"].to_numpy(int)])


def demand_intercept(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    """Structural log-demand at the reference price, with U = 0 and eps = 0."""
    d = cfg["dgp"]
    wk = df["is_weekend"].to_numpy(float)
    leis = df["seg_leisure"].to_numpy(float)
    corp = df["seg_corporate"].to_numpy(float)
    return (
        d["ln_base_demand"]
        + d["season_demand"] * df["seasonality"].to_numpy(float)
        + d["weekend_leisure_demand"] * wk * leis
        - 0.08 * wk * corp
        + 0.02 * (df["los"].to_numpy(float) - 3.0)
    )


def structural_log_demand(df: pd.DataFrame, cfg: Dict[str, Any], ln_price: np.ndarray, u: np.ndarray | float = 0.0) -> np.ndarray:
    """log D(X, p, U) without idiosyncratic noise."""
    theta = structural_theta(df, cfg)
    return demand_intercept(df, cfg) + theta * (ln_price - ln_reference_price(df, cfg)) + cfg["dgp"]["gamma_u"] * u


def optimal_policy(df: pd.DataFrame, cfg: Dict[str, Any], u: np.ndarray | float = 0.0) -> pd.DataFrame:
    """Row-wise ground-truth optimum (offered ADR, promo depth) by exhaustive grid search."""
    dg, dd, op = cfg["dgp"], cfg["data"], cfg["optimization"]
    grid = np.linspace(dd["price_min"], dd["price_max"], dd["price_grid_points"])
    promos = np.array(op["promo_grid"], float)
    n = len(df)
    best_rev = np.full(n, -np.inf)
    best_p = np.zeros(n)
    best_w = np.zeros(n)
    u_arr = np.broadcast_to(np.asarray(u, float), (n,))
    for p in grid:
        for w in promos:
            ld = structural_log_demand(df, cfg, np.full(n, np.log(p * (1.0 - w))), u_arr)
            rev = net_revpar(expected_occupancy(ld, dg["eps_sd"]), np.full(n, p), w, op["unit_cost_per_room"])
            better = rev > best_rev
            best_rev = np.where(better, rev, best_rev)
            best_p = np.where(better, p, best_p)
            best_w = np.where(better, w, best_w)
    return pd.DataFrame({"true_opt_adr": best_p, "true_opt_promo": best_w, "true_opt_net_revpar": best_rev}, index=df.index)


def generate(n: int | None = None, seed: int | None = None, cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    n = int(n or cfg["data"]["n_samples"])
    seed = int(seed if seed is not None else cfg["project"]["seed"])
    rng = np.random.default_rng(seed)
    d = cfg["dgp"]
    pe = d["price_equation"]

    # ---- observed confounders X ------------------------------------------------------------
    tier_idx = rng.choice(3, size=n, p=[0.25, 0.45, 0.30])
    tier = np.array(TIERS)[tier_idx]
    dow = rng.integers(0, 7, size=n)
    is_wk = np.isin(dow, [4, 5]).astype(float)
    urban = (tier == "urban_business").astype(float)

    logits = np.stack([
        0.3 + 1.2 * is_wk - 0.6 * urban,          # leisure
        0.9 - 1.8 * is_wk + 1.0 * urban,          # corporate
        -1.4 + 0.1 * is_wk + 0.1 * urban,         # group
    ], axis=1)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    seg_idx = (probs.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    segment = np.array(SEGMENTS)[seg_idx]

    lead_mean = np.array([62.0, 9.0, 85.0])[seg_idx]
    lead_time = np.clip(rng.gamma(shape=2.2, scale=lead_mean / 2.2), 0, 365).round()
    los = 1 + rng.poisson(np.array([2.6, 1.0, 2.0])[seg_idx])
    seasonality = np.clip(rng.normal(0, 1, n), -2.5, 2.5)
    loyalty = rng.choice(4, size=n, p=[0.50, 0.25, 0.15, 0.10])
    hist_sens = rng.normal(0, 1, n) + 0.4 * (seg_idx == 0) - 0.3 * (seg_idx == 1)
    room_type = rng.choice(3, size=n, p=[0.55, 0.30, 0.15])

    df = pd.DataFrame({
        "booking_id": np.arange(n),
        "property_tier": tier, "segment": segment, "dow": dow,
        "lead_time": lead_time, "los": los, "seasonality": seasonality,
        "loyalty_tier": loyalty, "hist_adr_sensitivity": hist_sens, "room_type": room_type,
    })
    df = add_dummies(df)

    # ---- unobserved demand shock U (latent city-wide event surge) ----------------------------
    u = rng.normal(0, 1, n)

    # ---- instruments Z (independent of U; mildly dependent on observed X only) ---------------
    lam = d["outage_lambda"]
    outage = np.clip(rng.exponential(lam * (1.0 + 0.4 * urban)), 0, 35.0)
    airline = rng.normal(0, 1, n)                         # units of 10% change in feeder-route seats
    rounding = rng.binomial(1, 0.35, n).astype(float)     # competitor feed crosses rounding trigger
    df["competitor_room_outage_pct"] = outage
    df["airline_seat_capacity_shock"] = airline
    df["pricing_algo_rounding_threshold"] = rounding

    # ---- treatment: offered ADR (endogenous via U) then promotional discount -----------------
    ln_ref = ln_reference_price(df, cfg)
    ln_p = (
        ln_ref
        + pe["season"] * seasonality + pe["weekend"] * is_wk + pe["loyalty"] * loyalty
        + pe["u_coef"] * u
        + pe["z_outage"] * (outage - lam) / lam
        + pe["z_airline"] * airline
        + pe["z_rounding"] * rounding
        + rng.normal(0, pe["noise_sd"], n)
    )
    adr = np.clip(np.exp(ln_p), cfg["data"]["price_min"], cfg["data"]["price_max"])

    promo_logit = (
        d["promo"]["base_logit"] + 0.5 * (seg_idx == 0) + 0.35 * hist_sens
        - 0.25 * (loyalty - 0.8) + 0.25 * np.clip((lead_time - 30) / 30, -1, 3)
    )
    promo = rng.binomial(1, _sigmoid(promo_logit)).astype(float)   # CRM targeting: depends on X only
    net_adr = adr * (1.0 - d["promo"]["discount"] * promo)

    # ---- outcomes ----------------------------------------------------------------------------
    eps = rng.normal(0, d["eps_sd"], n)
    log_demand = structural_log_demand(df, cfg, np.log(net_adr), u) + eps
    demand = np.exp(log_demand)
    occupancy = np.clip(1.0 - np.exp(-demand), 0.0, 0.999)
    converted = rng.binomial(1, occupancy)
    unit_cost = cfg["optimization"]["unit_cost_per_room"]

    df["demand_shock_u"] = u                     # kept for diagnostics / EDA only - never a model input
    df["adr"] = adr
    df["promo"] = promo
    df["net_adr"] = net_adr
    df["ln_adr"] = np.log(adr)
    df["ln_net_adr"] = np.log(net_adr)
    df["log_demand"] = log_demand
    df["occupancy_rate"] = occupancy
    df["converted"] = converted.astype(int)
    df["net_revpar"] = occupancy * (net_adr - unit_cost)

    # ---- ground truth -------------------------------------------------------------------------
    theta = structural_theta(df, cfg)
    df["true_elasticity"] = theta
    df["true_promo_cate"] = theta * np.log(1.0 - d["promo"]["discount"])   # effect of promo on log demand
    return df


def counterfactuals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Ground-truth structural quantities per booking (U = 0 i.e. an average market state)."""
    out = pd.DataFrame({
        "booking_id": df["booking_id"].to_numpy(),
        "true_elasticity": df["true_elasticity"].to_numpy(),
        "true_promo_cate": df["true_promo_cate"].to_numpy(),
        "ln_reference_price": ln_reference_price(df, cfg),
        "structural_intercept": demand_intercept(df, cfg),
    })
    pol = optimal_policy(df, cfg, 0.0).reset_index(drop=True)
    return pd.concat([out, pol], axis=1)


def summary_stats(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    import statsmodels.api as sm

    x = df[FEATURES].to_numpy(float)
    design = sm.add_constant(np.column_stack([df[TREATMENT].to_numpy(float), x]))
    ols = sm.OLS(df[OUTCOME].to_numpy(float), design).fit()
    fs = first_stage_f(df[TREATMENT].to_numpy(), df[INSTRUMENTS].to_numpy(), x)
    corr_zu = {z: float(np.corrcoef(df[z], df["demand_shock_u"])[0, 1]) for z in INSTRUMENTS}
    corr_zt = {z: float(np.corrcoef(df[z], df[TREATMENT])[0, 1]) for z in INSTRUMENTS}
    return {
        "n_rows": int(len(df)),
        "mean_adr": float(df["adr"].mean()),
        "mean_occupancy": float(df["occupancy_rate"].mean()),
        "mean_net_revpar": float(df["net_revpar"].mean()),
        "promo_rate": float(df["promo"].mean()),
        "segment_shares": df["segment"].value_counts(normalize=True).round(4).to_dict(),
        "true_mean_elasticity": float(df["true_elasticity"].mean()),
        "true_elasticity_by_segment": df.groupby("segment")["true_elasticity"].mean().round(4).to_dict(),
        "naive_ols_price_coefficient": float(ols.params[1]),
        "naive_ols_bias": float(ols.params[1] - df["true_elasticity"].mean()),
        "first_stage": fs,
        "corr_instrument_with_treatment": corr_zt,
        "corr_instrument_with_unobserved_shock": corr_zu,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("data_gen", "data_gen.log")
    set_seed(cfg["project"]["seed"])
    t0 = time.time()
    log.info("Hardware: %s", hardware_summary())

    df = generate(args.n, args.seed, cfg)
    cf = counterfactuals(df, cfg)
    stats = summary_stats(df, cfg)

    out = Path(args.out_dir) if args.out_dir else resolve(cfg["paths"]["data"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "hospitality_bookings.parquet", index=False)
    cf.to_parquet(out / "ground_truth_counterfactuals.parquet", index=False)
    dump_json(stats, out / "dgp_summary_stats.json")

    log.info("Dataset shape=%s, counterfactuals shape=%s", df.shape, cf.shape)
    log.info("Naive OLS price coefficient=%.3f vs true mean elasticity=%.3f (bias=%+.3f)",
             stats["naive_ols_price_coefficient"], stats["true_mean_elasticity"], stats["naive_ols_bias"])
    log.info("First-stage F=%.1f (partial R2=%.3f); corr(Z,U)=%s",
             stats["first_stage"]["f_stat"], stats["first_stage"]["partial_r2"], stats["corr_instrument_with_unobserved_shock"])
    log.info("Saved %s | %s | %s (%.1fs)", out / "hospitality_bookings.parquet",
             out / "ground_truth_counterfactuals.parquet", out / "dgp_summary_stats.json", time.time() - t0)

    assert stats["naive_ols_price_coefficient"] > stats["true_mean_elasticity"], "endogeneity bias not present"
    assert stats["first_stage"]["f_stat"] > 10, "instruments are weak"


if __name__ == "__main__":
    main()
