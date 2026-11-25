"""FastAPI real-time CATE scoring & CRM decision service.   Run: uvicorn api.app:app --port 8000"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.serving import PRESETS, Scorer  # noqa: E402

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["scorer"] = Scorer()
    yield


app = FastAPI(title="Hospitality Uplift / CATE Scoring API", version="1.0", lifespan=lifespan,
              description="Scores guests with causal uplift estimators and returns a CRM decision "
                          "(SEND 20% OFF / SEND 2,000 POINTS / SUPPRESS).")


class Guest(BaseModel):
    hist_booking_freq: float = Field(ge=0, le=60, examples=[3.0], description="Historical bookings per year")
    days_since_last_stay: float = Field(ge=0, le=3000, examples=[170])
    loyalty_tier: Literal["Non-Member", "Silver", "Gold", "Platinum"] = "Silver"
    hist_price_sensitivity: float = Field(ge=0, le=1, examples=[0.8])
    avg_lead_time: float = Field(ge=0, le=365, examples=[30])
    avg_adr: float = Field(ge=20, le=2000, examples=[150])
    weekend_stay_ratio: float = Field(ge=0, le=1, examples=[0.8])
    cancellation_rate: float = Field(ge=0, le=1, examples=[0.07])
    email_open_rate_90d: float = Field(ge=0, le=1, examples=[0.5])
    points_balance: float = Field(ge=0, examples=[4000])


class Offer(BaseModel):
    discount_pct: Optional[float] = Field(None, ge=0, le=0.6)
    points_offer: Optional[float] = Field(None, ge=0)
    points_cost_per_point: Optional[float] = Field(None, ge=0)
    cate_threshold: Optional[float] = None


class ScoreRequest(BaseModel):
    guest: Guest
    estimator: Optional[str] = Field(None, examples=["CausalForestDML"])
    offer: Optional[Offer] = None
    explain: bool = False


class BatchRequest(BaseModel):
    guests: list[Guest] = Field(max_length=5000)
    estimator: Optional[str] = None


def _scorer() -> Scorer:
    return state["scorer"]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(_scorer().models), "default_estimator": _scorer().best}


@app.get("/models")
def models():
    return {"estimators": list(_scorer().models), "default": _scorer().best}


@app.get("/presets")
def presets():
    return PRESETS


@app.post("/score")
def score(req: ScoreRequest):
    try:
        offer = {k: v for k, v in (req.offer.model_dump() if req.offer else {}).items() if v is not None}
        return _scorer().score([req.guest.model_dump()], req.estimator, offer, req.explain)[0]
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/score/batch")
def score_batch(req: BatchRequest):
    try:
        res = _scorer().score([g.model_dump() for g in req.guests], req.estimator)
    except KeyError as e:
        raise HTTPException(404, str(e))
    counts: dict = {}
    for r in res:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    return {"n": len(res), "decision_counts": counts, "results": res}
