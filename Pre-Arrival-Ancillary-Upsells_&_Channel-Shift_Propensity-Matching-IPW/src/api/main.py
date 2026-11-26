"""FastAPI service: /health, /predict/cate, /optimize/campaign, /models/metrics."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import serving
from src.common.config import load_config, out_dir
from src.common.hardware import get_device_report
from src.data import synthetic_generator as sg


class Guest(BaseModel):
    channel: Literal["OTA_Expedia", "OTA_Booking", "Direct_Web", "Corporate"]
    room_tier: Literal["Standard", "Deluxe", "Suite"]
    lead_time: float = Field(ge=0, le=730, description="Booking lead time in days")
    adults: int = Field(ge=1, le=8)
    children: int = Field(ge=0, le=8)
    device: Literal["iOS", "Android", "Desktop"]
    hist_spend: float = Field(ge=0, description="Historical ancillary spend ($)")
    loyalty_tier: int = Field(ge=0, le=3, description="0=none,1=Silver,2=Gold,3=Platinum")
    los: int = Field(ge=1, le=30, description="Length of stay (nights)")
    resort: int = Field(ge=0, le=1)

    model_config = {"json_schema_extra": {"examples": [serving.PERSONAS["Expedia family, Suite, 60-day lead (iOS)"]]}}


class CateRequest(BaseModel):
    guests: list[Guest] = Field(min_length=1, max_length=2000)
    model_ancillary: str | None = Field(None, description="Model name; default = best by Qini")
    model_direct: str | None = None


class CampaignRequest(BaseModel):
    budget_frac: float = Field(0.30, gt=0, le=1)
    cost_per_offer: float = Field(1.5, ge=0)
    ancillary_margin: float = Field(0.45, ge=0, le=1)
    direct_margin: float = Field(55.0, ge=0)
    only_positive: bool = True
    channels: list[Literal["OTA_Expedia", "OTA_Booking", "Direct_Web", "Corporate"]] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = (out_dir("metrics") / "campaign_summary.json").exists()
    if app.state.ready:   # warm the best models
        for o in ("ancillary", "direct"):
            serving.load_model(o, serving.best_model(o))
    yield


app = FastAPI(title="Hospitality Causal Upsell Engine", version="1.0.0", lifespan=lifespan,
              description="CATE scoring and budget-constrained campaign optimisation for pre-arrival ancillary upsells and direct-channel shift.")


def _need_ready():
    if not (out_dir("metrics") / "campaign_summary.json").exists():
        raise HTTPException(503, "Artifacts missing: run `python scripts/run_pipeline.py` first.")


@app.get("/health")
def health():
    ready = (out_dir("metrics") / "campaign_summary.json").exists()
    return {"status": "ok" if ready else "degraded", "artifacts_ready": ready, "device": get_device_report(),
            "best_models": serving.campaign_summary()["best_model"] if ready else None}


@app.post("/predict/cate")
def predict_cate(req: CateRequest):
    _need_ready()
    df = pd.DataFrame([g.model_dump() for g in req.guests])
    for m, o in ((req.model_ancillary, "ancillary"), (req.model_direct, "direct")):
        if m and m not in serving.model_names(o):
            raise HTTPException(422, f"Unknown {o} model '{m}'. Options: {serving.model_names(o)}")
    res = serving.predict_guests(df, req.model_ancillary, req.model_direct)
    return {"models": {"ancillary": req.model_ancillary or serving.best_model("ancillary"),
                       "direct": req.model_direct or serving.best_model("direct")},
            "predictions": res.round(5).to_dict("records")}


@app.post("/optimize/campaign")
def optimize(req: CampaignRequest):
    _need_ready()
    r = serving.optimize_campaign(req.budget_frac, req.cost_per_offer, req.ancillary_margin, req.direct_margin,
                                  req.only_positive, req.channels)
    r.pop("mask")
    return r


@app.get("/models/metrics")
def models_metrics():
    _need_ready()
    m = out_dir("metrics")
    lb = pd.read_csv(m / "cate_leaderboard.csv")
    import json
    return {"leaderboard": lb.round(5).astype(object).where(lb.notna(), None).to_dict("records"),
            "ate_comparison": json.load(open(m / "ate_comparison.json")),
            "refutation": json.load(open(m / "refutation.json")),
            "campaign_summary": serving.campaign_summary(), "device": get_device_report()}
