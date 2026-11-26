"""FastAPI service.  uvicorn src.api.main:app --port 8000"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api import service
from src.utils.device import get_hardware

app = FastAPI(title="Paid Search Incrementality Engine - Grand Horizon Hotels",
              description="Cannibalization vs incrementality of branded paid search: CATE prediction, "
                          "holdout policy simulation and experiment metrics.", version="1.0.0")


class DMAFeatures(BaseModel):
    tier: int = Field(2, ge=1, le=3, description="1 Major Hub, 2 Regional, 3 Drive-to")
    organic_rank: float = Field(2.0, ge=1, le=5, description="Average organic SERP rank for the brand term (1 = top)")
    ota_pressure: float = Field(0.35, ge=0, le=1, description="OTA auction impression share / conquesting pressure")
    competitor_adr: float = Field(210.0, gt=0)
    loyalty_pen: float = Field(0.24, ge=0, le=1)
    lead_time: float = Field(28.0, ge=0)
    flight_idx: float = Field(100.0, gt=0)
    baseline_direct_bookings_day: float = Field(25.0, gt=0)
    ota_to_direct_ratio: float = Field(0.55, ge=0)
    lastclick_share: float = Field(0.30, ge=0, le=1)
    adr: float = Field(205.0, gt=0)
    commission_rate: float = Field(0.175, ge=0.0, le=0.3)
    cpc: float = Field(2.2, gt=0)


class SimulationInputs(BaseModel):
    baseline_direct_bookings_day: float = Field(25.0, gt=0)
    ota_to_direct_ratio: float = Field(0.6, ge=0)
    lastclick_share: float = Field(0.30, ge=0, le=1)
    cannibalization_rate: float = Field(0.75, ge=0, le=1)
    ota_capture_rate: float = Field(0.20, ge=0, le=1)
    diversion_rate: float = Field(0.45, ge=0, le=1)
    adr: float = Field(205.0, gt=0)
    commission_rate: float = Field(0.175, ge=0.0, le=0.3)
    cpc: float = Field(2.2, gt=0)
    cvr: float = Field(0.025, gt=0, le=1)
    horizon_days: int = Field(28, ge=1, le=365)


@app.get("/health")
def health():
    hw = get_hardware().summary()
    try:
        loaded = service.models_loaded()
    except Exception as e:  # models not trained yet
        return {"status": "degraded", "detail": f"models not available: {e}", "hardware": hw}
    return {"status": "ok", "hardware": hw, "models": loaded}


@app.post("/predict/dma-cate")
def predict_dma_cate(body: DMAFeatures):
    try:
        return service.predict_dma_cate(body.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(503, f"models not trained: {e}")


@app.post("/simulate/holdout-policy")
def simulate_holdout_policy(body: SimulationInputs):
    return service.simulate_holdout_policy(body.model_dump())


@app.get("/metrics/summary")
def metrics_summary():
    return service.metrics_summary()
