"""FastAPI service.   Run: uvicorn src.api.main:app --port 8000   (Swagger UI at /docs)"""
from __future__ import annotations

import sys
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.optimization.revenue_optimizer import (build_profile_df, compare_policies, rack_rate, revpar_curve,  # noqa: E402
                                                truth_curve)
from src.utils.gpu_utils import cuda_available, hardware_summary, load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import to_jsonable  # noqa: E402
from src.utils.registry import load_json, load_models  # noqa: E402
from src.utils.structural import expected_occupancy  # noqa: E402

log = get_logger("api_server", "api_server.log")
STATE: Dict[str, Any] = {"models": {}, "cfg": None, "started": time.time()}

PRICING_MODELS = ["Naive OLS", "Naive XGBoost", "2SLS", "DML-IV", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)"]
ELASTICITY_MODELS = ["Naive OLS", "Naive XGBoost", "2SLS", "DML-IV", "DR-IV (LinearDRIV)", "DeepIV (PyTorch)"]
CATE_MODELS = ["S-Learner", "T-Learner", "X-Learner", "R-Learner (NonParamDML)", "DR-Learner", "Causal Forest DML",
               "Uplift RF (causalml)", "TARNet", "DragonNet", "CEVAE", "DeepIV (PyTorch)", "DR-IV (LinearDRIV)"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["cfg"] = load_config()
    STATE["models"] = load_models()
    STATE["started"] = time.time()
    log.info("Startup: %d models loaded | hardware=%s", len(STATE["models"]), hardware_summary())
    yield


app = FastAPI(
    title="Grand Horizon Hotels - Dynamic Pricing & Elasticity under Endogeneity",
    description="Instrumental-variable and CATE engine that removes demand-shock (omitted variable) bias from "
                "price-elasticity estimates and prescribes optimal ADR and promotional discounts.",
    version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    log.info("%s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, (time.time() - t0) * 1000)
    return response


class GuestRequest(BaseModel):
    """Booking request context: observed confounders X and market instruments Z."""

    segment: Literal["leisure", "corporate", "group"] = Field("leisure", examples=["leisure"], description="Guest segment")
    property_tier: Literal["luxury", "urban_business", "resort"] = Field("urban_business", examples=["urban_business"])
    lead_time: float = Field(30, ge=0, le=365, examples=[30], description="Days between booking and arrival")
    los: int = Field(2, ge=1, le=21, examples=[2], description="Length of stay (nights)")
    dow: int = Field(5, ge=0, le=6, examples=[5], description="Arrival day of week, Mon=0")
    seasonality: float = Field(0.0, ge=-3, le=3, examples=[0.0], description="Standardised seasonality index")
    loyalty_tier: int = Field(1, ge=0, le=3, examples=[1])
    hist_adr_sensitivity: float = Field(0.0, ge=-3, le=3, examples=[0.0], description="Guest's historical ADR sensitivity (z-score)")
    room_type: int = Field(1, ge=0, le=2, examples=[1], description="0 standard, 1 deluxe, 2 suite")
    competitor_room_outage_pct: float = Field(5.0, ge=0, le=35, examples=[5.0], description="Instrument Z1")
    airline_seat_capacity_shock: float = Field(0.0, ge=-4, le=4, examples=[0.0], description="Instrument Z2, units of 10% of feeder seats")
    pricing_algo_rounding_threshold: int = Field(0, ge=0, le=1, examples=[0], description="Instrument Z3")
    demand_shock_u: float = Field(0.0, ge=-3, le=3, examples=[0.0],
                                  description="Scenario's latent demand shock (used only to report realised RevPAR; models never see it)")
    offered_adr: Optional[float] = Field(None, ge=95, le=650, examples=[320.0], description="Rate on offer; defaults to the rack rate")

    def profile(self) -> Dict[str, Any]:
        keys = ["segment", "property_tier", "lead_time", "los", "dow", "seasonality", "loyalty_tier", "hist_adr_sensitivity", "room_type"]
        return {k: getattr(self, k) for k in keys}

    def instruments(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ["competitor_room_outage_pct", "airline_seat_capacity_shock", "pricing_algo_rounding_threshold"]}


def _models(names: List[str]) -> Dict[str, Any]:
    m = {n: STATE["models"][n] for n in names if n in STATE["models"]}
    if not m:
        raise HTTPException(503, "No trained models found. Run `make train` first.")
    return m


def _row(req: GuestRequest):
    return build_profile_df(req.profile(), req.instruments(), cfg=STATE["cfg"])


@app.get("/health", tags=["system"])
def health() -> Dict[str, Any]:
    return {"status": "ok", "gpu_available": cuda_available(), "hardware": to_jsonable(hardware_summary()),
            "models_loaded": len(STATE["models"]), "uptime_seconds": round(time.time() - STATE["started"], 1)}


@app.get("/api/v1/iv-diagnostics", tags=["diagnostics"])
def iv_diagnostics() -> Dict[str, Any]:
    """First-stage F, Wu-Hausman, Sargan-Hansen and the benchmark summary."""
    return {"iv_first_stage_diagnostics": load_json("iv_first_stage_diagnostics.json"),
            "benchmark_summary": load_json("benchmark_summary.json")}


@app.post("/api/v1/estimate-elasticity", tags=["elasticity"])
def estimate_elasticity(req: GuestRequest) -> Dict[str, Any]:
    """Causal elasticity (DeepIV, DML-IV, DR-IV, 2SLS) against the naive OLS bias gap."""
    row = _row(req)
    ms = _models(ELASTICITY_MODELS)
    est = {n: float(m.elasticity(row)[0]) for n, m in ms.items()}
    truth = float(row["true_elasticity"].iloc[0])
    naive = est.get("Naive OLS")
    return to_jsonable({
        "estimated_elasticity": est,
        "ground_truth_elasticity": truth,
        "bias_vs_truth": {n: v - truth for n, v in est.items()},
        "naive_ols_bias_gap": None if naive is None else {k: naive - v for k, v in est.items() if k != "Naive OLS"},
        "interpretation": "Naive models report a flat or positive elasticity because rates and demand rise together during "
                          "unobserved demand surges; IV estimators recover the negative structural elasticity.",
    })


@app.post("/api/v1/optimize-price", tags=["pricing"])
def optimize_price(req: GuestRequest) -> Dict[str, Any]:
    """Sweep the $95-$650 ADR grid with each model's structural demand and return optimal ADR / promo / RevPAR lift."""
    cfg, row = STATE["cfg"], _row(req)
    ms = _models(PRICING_MODELS)
    res = compare_policies(ms, row, cfg, req.demand_shock_u,
                           causal_key="DeepIV (PyTorch)" if "DeepIV (PyTorch)" in ms else next(iter(ms)))
    causal = res["per_model"].get("DeepIV (PyTorch)") or next(iter(res["per_model"].values()))
    curves = {}
    for n in ("Naive OLS", "DeepIV (PyTorch)"):
        if n in ms:
            c = revpar_curve(ms[n], row, cfg)
            curves[n] = {"adr": c["adr"].round(1).tolist()[::3], "net_revpar": c["net_revpar"].round(2).tolist()[::3]}
    t = truth_curve(row, cfg, None)
    curves["Ground truth"] = {"adr": t["adr"].round(1).tolist()[::3], "net_revpar": t["net_revpar"].round(2).tolist()[::3]}
    return to_jsonable({
        "optimal_adr": causal["optimal_adr"], "optimal_promo_pct": causal["optimal_promo_pct"],
        "optimal_net_price": causal["optimal_net_price"],
        "expected_revpar_lift_vs_naive_pct": res.get("revpar_lift_pct_causal_vs_naive"),
        "expected_revpar_lift_vs_status_quo_pct": res.get("revpar_lift_pct_causal_vs_status_quo"),
        "naive_ols_recommendation": res["per_model"].get("Naive OLS"),
        "per_model": res["per_model"], "rack_rate": res["rack_rate"], "status_quo_adr": res["status_quo_adr"],
        "ground_truth": {"optimal_adr": res["truth_optimal_adr"], "elasticity": res["true_elasticity"]},
        "curves": curves,
    })


@app.post("/api/v1/cate-benchmark", tags=["cate"])
def cate_benchmark(req: GuestRequest) -> Dict[str, Any]:
    """Promotional-discount CATE from every meta-learner, forest and deep causal net, plus consensus uplift."""
    cfg, row = STATE["cfg"], _row(req)
    adr = req.offered_adr or float(rack_rate(row, cfg)[0])
    row["ln_adr"] = np.log(adr)
    row["adr"] = adr
    ms = _models(CATE_MODELS)
    base_ref = ms.get("DeepIV (PyTorch)") or next(iter(ms.values()))
    base_ld = base_ref.predict_log_demand(row, np.log([adr]))
    eps = cfg["dgp"]["eps_sd"]
    occ0 = float(expected_occupancy(base_ld, eps)[0])
    d = cfg["dgp"]["promo"]["discount"]
    out, taus = {}, []
    for n, m in ms.items():
        tau = float(m.promo_cate(row)[0])
        taus.append(tau)
        out[n] = {"cate_log_demand": tau, "implied_elasticity": float(m.elasticity(row)[0]),
                  "uplift_conversion_pp": float((expected_occupancy(base_ld + tau, eps)[0] - occ0) * 100.0)}
    truth_tau = float(row["true_promo_cate"].iloc[0]) if "true_promo_cate" in row else float(row["true_elasticity"].iloc[0] * np.log(1 - d))
    cons = float(np.median(taus))
    return to_jsonable({
        "offered_adr": adr, "promo_discount_pct": d * 100, "baseline_conversion": occ0,
        "cate_estimates": out,
        "consensus": {"median_cate_log_demand": cons, "std_across_models": float(np.std(taus)),
                      "uplift_conversion_pp": float((expected_occupancy(base_ld + cons, eps)[0] - occ0) * 100.0)},
        "ground_truth_cate": truth_tau,
    })
