"""Structural RevPAR optimizer: optimal ADR and promotional discount from an estimated demand curve.

Net RevPAR(p, d) = E[occupancy(p, d)] * (p * (1 - d) - unit_cost),  occupancy = 1 - exp(-D)
Demand D(X, p) comes from a fitted estimator (naive OLS vs DeepIV / DR-IV ...) and is *scored* against the
ground-truth structural demand (with the scenario's unobserved shock U) to report realised RevPAR.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from data.generate_hospitality_data import (FEATURES, INSTRUMENTS, ROOM_MULT, add_dummies, demand_intercept,
                                            ln_reference_price, structural_theta)
from src.utils.gpu_utils import load_config
from src.utils.structural import expected_occupancy, net_revpar

PROFILE_DEFAULTS: Dict[str, Any] = {
    "segment": "leisure", "property_tier": "urban_business", "lead_time": 30, "los": 2, "dow": 5,
    "seasonality": 0.0, "loyalty_tier": 1, "hist_adr_sensitivity": 0.0, "room_type": 1,
}
INSTRUMENT_DEFAULTS: Dict[str, float] = {
    "competitor_room_outage_pct": 5.0, "airline_seat_capacity_shock": 0.0, "pricing_algo_rounding_threshold": 0,
}


def build_profile_df(profile: Dict[str, Any], instruments: Dict[str, Any] | None = None, n: int = 1,
                     cfg: Dict[str, Any] | None = None) -> pd.DataFrame:
    """One (repeated n times) booking-request row with every column the estimators need."""
    cfg = cfg or load_config()
    row = {**PROFILE_DEFAULTS, **(profile or {})}
    row.update({**INSTRUMENT_DEFAULTS, **(instruments or {})})
    df = pd.DataFrame([row] * n)
    df["booking_id"] = np.arange(n)
    df = add_dummies(df)
    df["true_elasticity"] = structural_theta(df, cfg)
    df["ln_adr"] = ln_reference_price(df, cfg)
    return df


def rack_rate(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    """Published best-available rate: reference price plus a premium that leaves headroom for targeted promos."""
    return np.exp(ln_reference_price(df, cfg)) * cfg["optimization"].get("rack_premium", 1.0)


def truth_log_demand(df: pd.DataFrame, cfg: Dict[str, Any], net_price: np.ndarray, u: float = 0.0) -> np.ndarray:
    theta = structural_theta(df, cfg)
    return demand_intercept(df, cfg) + theta * (np.log(net_price) - ln_reference_price(df, cfg)) + cfg["dgp"]["gamma_u"] * u


_UN, _UW = np.polynomial.hermite_e.hermegauss(9)
_UW = _UW / _UW.sum()


def truth_occupancy(row: pd.DataFrame, cfg: Dict[str, Any], net_price: np.ndarray, u: float | None) -> np.ndarray:
    """Expected occupancy under the true structural demand. u=None marginalises the unobserved shock U ~ N(0,1)."""
    eps = cfg["dgp"]["eps_sd"]
    if u is not None:
        return expected_occupancy(truth_log_demand(row, cfg, net_price, u), eps)
    return sum(w * expected_occupancy(truth_log_demand(row, cfg, net_price, float(x)), eps) for x, w in zip(_UN, _UW))


def _grid(cfg: Dict[str, Any]) -> np.ndarray:
    d = cfg["data"]
    return np.linspace(d["price_min"], d["price_max"], d["price_grid_points"])


def revpar_curve(model, row: pd.DataFrame, cfg: Dict[str, Any], promo: float = 0.0) -> pd.DataFrame:
    """Predicted occupancy and Net RevPAR over the ADR grid for a single guest profile (row has 1 line)."""
    grid = _grid(cfg)
    rep = pd.concat([row.iloc[[0]]] * len(grid), ignore_index=True)
    ln_net = np.log(grid * (1.0 - promo))
    log_d = model.predict_log_demand(rep, ln_net)
    occ = expected_occupancy(log_d, cfg["dgp"]["eps_sd"])
    rev = net_revpar(occ, grid, promo, cfg["optimization"]["unit_cost_per_room"])
    return pd.DataFrame({"adr": grid, "occupancy": occ, "net_revpar": rev})


def truth_curve(row: pd.DataFrame, cfg: Dict[str, Any], u: float | None = None, promo: float = 0.0) -> pd.DataFrame:
    grid = _grid(cfg)
    rep = pd.concat([row.iloc[[0]]] * len(grid), ignore_index=True)
    occ = truth_occupancy(rep, cfg, grid * (1.0 - promo), u)
    return pd.DataFrame({"adr": grid, "occupancy": occ,
                         "net_revpar": net_revpar(occ, grid, promo, cfg["optimization"]["unit_cost_per_room"])})


def realised_revpar(row: pd.DataFrame, cfg: Dict[str, Any], adr: float, promo: float, u: float | None = None) -> float:
    rep = row.iloc[[0]].reset_index(drop=True)
    occ = truth_occupancy(rep, cfg, np.array([adr * (1.0 - promo)]), u)
    return float(net_revpar(occ, np.array([adr]), promo, cfg["optimization"]["unit_cost_per_room"])[0])


def status_quo_adr(row: pd.DataFrame, cfg: Dict[str, Any], u: float = 0.0) -> float:
    """What a market-following revenue manager charges: the DGP's price rule (reacts to U and to Z)."""
    pe, dd = cfg["dgp"]["price_equation"], cfg["data"]
    r = row.iloc[0]
    lam = cfg["dgp"]["outage_lambda"]
    ln_p = (
        ln_reference_price(row, cfg)[0]
        + pe["season"] * r["seasonality"] + pe["weekend"] * r["is_weekend"] + pe["loyalty"] * r["loyalty_tier"]
        + pe["u_coef"] * u
        + pe["z_outage"] * (r["competitor_room_outage_pct"] - lam) / lam
        + pe["z_airline"] * r["airline_seat_capacity_shock"]
        + pe["z_rounding"] * r["pricing_algo_rounding_threshold"]
    )
    return float(np.clip(np.exp(ln_p), dd["price_min"], dd["price_max"]))


def optimize_for_model(model, row: pd.DataFrame, cfg: Dict[str, Any], u: float | None = 0.0,
                       rack: float | None = None) -> Dict[str, Any]:
    """Pick optimal net price using the model's own demand, then split it into rack ADR + promo depth.

    Because a promotion is equivalent to a lower net price for a single guest profile, the promo is the
    segment-specific discount off the published rack rate needed to reach the model-optimal net price.
    """
    rack = float(rack if rack is not None else rack_rate(row, cfg)[0])
    unit = cfg["optimization"]["unit_cost_per_room"]
    curve = revpar_curve(model, row, cfg, 0.0)
    i = int(curve["net_revpar"].to_numpy().argmax())
    p_star = float(curve["adr"].iloc[i])
    promos = np.asarray(cfg["optimization"]["promo_grid"], float)
    if p_star >= rack:
        adr, promo = p_star, 0.0
    else:
        need = 1.0 - p_star / rack
        promo = float(promos[np.abs(promos - need).argmin()])
        adr = rack if promo > 0 else p_star
        if promo == 0.0:
            adr = p_star
    net_price = adr * (1.0 - promo)
    rep = row.iloc[[0]].reset_index(drop=True)
    pred_occ = float(expected_occupancy(model.predict_log_demand(rep, np.log([net_price])), cfg["dgp"]["eps_sd"])[0])
    return {
        "model": model.name,
        "optimal_net_price": float(net_price),
        "optimal_adr": float(adr),
        "optimal_promo_pct": float(promo * 100.0),
        "predicted_occupancy": pred_occ,
        "predicted_net_revpar": float(pred_occ * (net_price - unit)),
        "realised_net_revpar": realised_revpar(row, cfg, adr, promo, None),      # averaged over U ~ N(0,1)
        "realised_net_revpar_at_scenario_u": realised_revpar(row, cfg, adr, promo, u),
        "estimated_elasticity": float(model.elasticity(rep)[0]),
    }


def compare_policies(models: Dict[str, Any], row: pd.DataFrame, cfg: Dict[str, Any], u: float = 0.0,
                     naive_key: str = "Naive OLS", causal_key: str = "DeepIV (PyTorch)") -> Dict[str, Any]:
    """Optimal prices for several models and the realised RevPAR lift of the causal policy over the naive one.

    Headline RevPAR is the *expected* realised value over the unobserved shock U (the decision is made before U is
    known); `..._at_scenario_u` is the value if the scenario's specific shock materialises.
    """
    rack = float(rack_rate(row, cfg)[0])
    results = {k: optimize_for_model(m, row, cfg, u, rack) for k, m in models.items()}
    truth_best = truth_curve(row, cfg, None)
    ib = int(truth_best["net_revpar"].to_numpy().argmax())
    out: Dict[str, Any] = {
        "rack_rate": rack,
        "scenario_u": float(u),
        "per_model": results,
        "truth_optimal_adr": float(truth_best["adr"].iloc[ib]),
        "truth_optimal_net_revpar": float(truth_best["net_revpar"].iloc[ib]),
        "true_elasticity": float(row["true_elasticity"].iloc[0]),
    }
    sq = status_quo_adr(row, cfg, u)
    out["status_quo_adr"] = sq
    out["status_quo_net_revpar"] = realised_revpar(row, cfg, sq, 0.0, None)
    out["status_quo_net_revpar_at_scenario_u"] = realised_revpar(row, cfg, sq, 0.0, u)
    if causal_key in results:
        c = results[causal_key]
        out["revpar_lift_pct_causal_vs_status_quo"] = float((c["realised_net_revpar"] / max(out["status_quo_net_revpar"], 1e-9) - 1.0) * 100.0)
    if naive_key in results and causal_key in results:
        n_, c_ = results[naive_key], results[causal_key]
        out["revpar_lift_pct_causal_vs_naive"] = float((c_["realised_net_revpar"] / max(n_["realised_net_revpar"], 1e-9) - 1.0) * 100.0)
        out["revpar_lift_pct_causal_vs_naive_at_scenario_u"] = float(
            (c_["realised_net_revpar_at_scenario_u"] / max(n_["realised_net_revpar_at_scenario_u"], 1e-9) - 1.0) * 100.0)
    return out


def crm_segments(df: pd.DataFrame, model, cfg: Dict[str, Any], discount: float | None = None) -> pd.DataFrame:
    """Score a batch of reservations: net value of the standard promotion and uplift quadrants.

    baseline_conv  : expected booking probability at the offered rate without the promo
    uplift_conv    : promo-induced change in booking probability
    net_value      : change in expected net RevPAR (promo revenue gain minus discount given away)
    """
    discount = cfg["dgp"]["promo"]["discount"] if discount is None else discount
    unit = cfg["optimization"]["unit_cost_per_room"]
    eps = cfg["dgp"]["eps_sd"]
    ln_p = np.log(df["adr"].to_numpy(float)) if "adr" in df else df["ln_adr"].to_numpy(float)
    price = np.exp(ln_p)
    tau = model.promo_cate(df)                                    # effect of the promo on log demand
    base_ld = model.predict_log_demand(df, ln_p)
    occ0 = expected_occupancy(base_ld, eps)
    occ1 = expected_occupancy(base_ld + tau, eps)
    net_value = occ1 * (price * (1 - discount) - unit) - occ0 * (price - unit)
    thr = max(0.02 * float(np.mean(occ0 * (price - unit))), 0.5)
    seg = np.where(net_value > thr, "Persuadables",
                   np.where(net_value < -thr, "Sleeping Dogs",
                            np.where(occ0 >= 0.70, "Sure Things", "Lost Causes")))
    out = df[[c for c in ["booking_id", "segment", "property_tier", "lead_time", "los"] if c in df]].copy()
    out["offered_adr"] = price
    out["baseline_conv"] = occ0
    out["uplift_conv"] = occ1 - occ0
    out["cate_log_demand"] = tau
    out["net_value_usd"] = net_value
    out["uplift_segment"] = seg
    out["recommended_promo_pct"] = np.where(seg == "Persuadables", discount * 100.0, 0.0)
    return out
