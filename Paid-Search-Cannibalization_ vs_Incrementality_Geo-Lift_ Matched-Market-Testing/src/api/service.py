"""Serving logic shared by the FastAPI app and the Streamlit dashboard (single source of truth)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.synthetic_geo_generator import CATE_FEATURES
from src.economics import cannibalization_roi as econ
from src.models.model_registry import load_models
from src.utils.config import path


def _read_json(name):
    p = path("metrics") / name
    return json.loads(p.read_text()) if p.exists() else None


@lru_cache(maxsize=1)
def _models():
    """target -> (list of (algo, model) for the top-3 learners by PEHE)."""
    out = {}
    for target in ("direct", "ota"):
        lb = pd.read_csv(path("metrics") / f"cate_leaderboard_{target}.csv", index_col=0)
        ms = load_models(target)
        top = [a for a in lb.index if a in ms][:3]
        out[target] = {"top": top, "all": ms}
    return out


def models_loaded() -> dict:
    return {t: {"ensemble": v["top"], "n_models": len(v["all"])} for t, v in _models().items()}


def design_row(p: dict, dow: int) -> np.ndarray:
    """Build the CATE feature vector (same order/transform as src.data.feature_engineering.cate_matrix)."""
    g_d = p["adr"] * econ.NIGHTS
    g_o = g_d * econ.OTA_ADR_FACTOR
    spend = p["baseline_direct_bookings_day"] * p["lastclick_share"] / econ.DEFAULT_CVR * p["cpc"]
    ncm = (p["baseline_direct_bookings_day"] * g_d - spend
           + p["baseline_direct_bookings_day"] * p["ota_to_direct_ratio"] * g_o * (1 - p["commission_rate"]))
    row = {"tier": p["tier"], "organic_rank": p["organic_rank"], "ota_pressure": p["ota_pressure"],
           "competitor_adr": p["competitor_adr"], "loyalty_pen": p["loyalty_pen"], "lead_time": p["lead_time"],
           "flight_idx": p.get("flight_idx", 100.0), "dow": dow, "holiday": 0, "season": 1.0,
           "pre_direct_mean": np.log1p(p["baseline_direct_bookings_day"]),
           "pre_ota_mean": np.log1p(p["baseline_direct_bookings_day"] * p["ota_to_direct_ratio"]),
           "pre_ncm_mean": ncm}
    return np.array([float(row[c]) for c in CATE_FEATURES])


def predict_dma_cate(p: dict) -> dict:
    """CATE of pausing branded SEM for one DMA profile, per learner and as the top-3 ensemble."""
    X = np.vstack([design_row(p, d) for d in range(7)])      # average over the week
    ms = _models()
    out = {"per_model": {"direct": {}, "ota": {}}}
    lifts = {}
    for t in ("direct", "ota"):
        for a, m in ms[t]["all"].items():
            out["per_model"][t][a] = float(np.mean(m.predict(X)))
        lifts[t] = float(np.mean([out["per_model"][t][a] for a in ms[t]["top"]]))
    base_d = p["baseline_direct_bookings_day"]
    base_o = base_d * p["ota_to_direct_ratio"]
    g_d = p["adr"] * econ.NIGHTS
    g_o = g_d * econ.OTA_ADR_FACTOR
    spend = base_d * p["lastclick_share"] / econ.DEFAULT_CVR * p["cpc"]
    d_direct, d_ota = lifts["direct"] * base_d, lifts["ota"] * base_o
    d_ncm = d_direct * g_d + d_ota * g_o * (1 - p["commission_rate"]) + spend
    out.update({
        "ensemble_models": {t: ms[t]["top"] for t in ms},
        "direct_lift_pct": lifts["direct"] * 100, "ota_lift_pct": lifts["ota"] * 100,
        "delta_direct_bookings_per_day": d_direct, "delta_ota_bookings_per_day": d_ota,
        "ota_commission_leakage_per_day": d_ota * g_o * p["commission_rate"],
        "branded_spend_saved_per_day": spend, "ncm_change_if_paused_per_day": d_ncm,
        "decision": "Pause Branded SEM" if d_ncm > 0 else "Keep Branded SEM"})
    return out


def simulate_holdout_policy(p: dict) -> dict:
    return econ.simulate_policy(p)


def metrics_summary() -> dict:
    m = path("metrics")
    lb = {t: pd.read_csv(m / f"cate_leaderboard_{t}.csv", index_col=0).reset_index().astype(object)
          .where(lambda d: d.notna(), None).to_dict("records")
          for t in ("direct", "ota") if (m / f"cate_leaderboard_{t}.csv").exists()}
    return {"experiment": _read_json("experiment_meta.json"), "hardware": _read_json("hardware.json"),
            "macro_geolift": _read_json("macro_results.json"), "programme_economics": _read_json("programme_economics.json"),
            "policy_value": _read_json("policy_value.json"), "cate_leaderboard": lb}


def archetypes() -> dict:
    return _read_json("archetypes.json") or {}


def flows_index() -> dict:
    """Programme scorecard plus one line per end-to-end flow (full stage detail: flow_report)."""
    from src import flows
    d = flows.load()
    return {"programme": d["programme"], "flows": [{k: d["flows"][i][k] for k in ("id", "type", "type_title", "short", "verdict", "flow_metrics")}
                                                    for i in d["order"]]}


def flow_report(flow_id: str) -> dict | None:
    from src import flows
    return flows.load()["flows"].get(flow_id.upper())
