"""FastAPI service: /health, /predict/att, /predict/cate, /simulate/campaign (+ /presets, /models helpers).

    uvicorn src.api.main:app --port 8000        # Swagger UI at /docs, ReDoc at /redoc
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from src.models.cate_learners import LEARNER_ORDER  # noqa: E402
from src.serving import Engine  # noqa: E402

state: dict = {}

Scenario = Literal["mia_all24", "mia_flagship", "las_convention", "nyc_spring_break", "tpa_placebo"]
Estimator = Literal["TWFE DiD", "Event Study", "CS-DiD (DR)", "SCM", "ASCM (Ridge)", "ASCM (ElasticNet)", "SDID", "Bayesian SC"]
Outcome = Literal["room_nights_booked", "direct_web_sessions", "branded_search_index", "incremental_net_revenue_usd"]
Learner = Literal["S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner", "Causal Forest DML", "Linear DML",
                  "Domain Adaptation", "TARNet", "CFRNet-DragonNet"]
Strategy = Literal["even", "front_loaded", "back_loaded", "pulsed_7_on_7_off", "holiday_weighted", "weekend_weighted",
                   "cate_weighted", "optimized"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = Engine()
    yield
    state.clear()


app = FastAPI(
    title="Aurelia Hotels - Regional Brand & OOH Incrementality API",
    version="1.0.0",
    description=("Geo-experiment incrementality for regional CTV / airport / billboard campaigns, where user-level holdouts "
                 "are impossible. **/predict/att** returns aggregate and dynamic ATT from DiD, synthetic-control and Bayesian "
                 "estimators; **/predict/cate** scores market conditions with ten heterogeneous-effect learners; "
                 "**/simulate/campaign** projects room nights, CPIRN and iROAS for budget and pacing what-ifs."),
    lifespan=lifespan,
)


def engine() -> Engine:
    if "engine" not in state:
        raise HTTPException(503, "engine not loaded")
    return state["engine"]


# ----------------------------------------------------------------------------- schemas
class AttRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"scenario": "mia_all24", "estimator": "ASCM (Ridge)",
                                                             "outcome": "room_nights_booked", "live": False}})
    scenario: Scenario = Field("mia_all24", description="Preset panel. 'mia_all24' is the headline analysis with all 24 control DMAs.")
    estimator: Estimator = Field("ASCM (Ridge)", description="Panel estimator. 'Bayesian SC' is cache-only.")
    outcome: Outcome = "room_nights_booked"
    live: bool = Field(False, description="Re-run the estimator now instead of reading the pipeline cache (not available for Bayesian SC).")
    treated: Optional[str] = Field(None, min_length=3, max_length=3, description="Override the exposed DMA code (custom analysis; implies live).")
    donors: Optional[list[str]] = Field(None, min_length=2, max_length=24, description="Custom donor pool of DMA codes (implies live).")
    include_series: bool = Field(True, description="Include daily observed / counterfactual / ATT paths.")


class MarketDay(BaseModel):
    """One market-day of conditions. Omitted fields default to the campaign-window median."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "competitor_adr_usd": 335.0, "flight_search_index": 125.0, "temp_max_f": 80.0, "precipitation_in": 0.0,
        "weather_severity_index": 3.0, "is_weekend": 1, "is_holiday_window": 0, "seasonality_sin": 0.40, "seasonality_cos": 0.92,
        "ota_promo_pressure_index": 28.0, "days_since_launch": 20, "dow": 5}})
    competitor_adr_usd: Optional[float] = Field(None, ge=50, le=1500)
    flight_search_index: Optional[float] = Field(None, ge=0, le=400)
    temp_max_f: Optional[float] = Field(None, ge=-20, le=125)
    precipitation_in: Optional[float] = Field(None, ge=0, le=20)
    weather_severity_index: Optional[float] = Field(None, ge=0, le=100)
    is_weekend: Optional[int] = Field(None, ge=0, le=1)
    is_holiday_window: Optional[int] = Field(None, ge=0, le=1)
    seasonality_sin: Optional[float] = Field(None, ge=-1, le=1)
    seasonality_cos: Optional[float] = Field(None, ge=-1, le=1)
    ota_promo_pressure_index: Optional[float] = Field(None, ge=0, le=100)
    days_since_launch: Optional[int] = Field(None, ge=0, le=400)
    dow: Optional[int] = Field(None, ge=0, le=6, description="Day of week, Monday = 0")


class CateRequest(BaseModel):
    rows: list[MarketDay] = Field(..., min_length=1, max_length=5000)
    learner: Optional[Learner] = Field(None, description="Defaults to the best learner by PEHE.")
    baseline_room_nights: Optional[float] = Field(None, gt=0, le=20000, description="Counterfactual daily room nights used to convert lift into room nights.")
    alpha: float = Field(0.05, gt=0.001, lt=0.5, description="CI level for learners with native intervals (Causal Forest DML, Linear DML).")


class CampaignRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"budget_usd": 3000000, "strategy": "optimized", "overrides": {"weather_severity_index": 5}}})
    budget_usd: float = Field(..., gt=0, le=100_000_000)
    strategy: Strategy = "even"
    learner: Optional[Learner] = None
    baseline_room_nights: Optional[float] = Field(None, gt=0, le=20000)
    net_revenue_per_room_night: Optional[float] = Field(None, gt=0, le=5000)
    overrides: dict[str, float] = Field(default_factory=dict, description="Force a market-condition feature to a constant over the window, e.g. weather_severity_index.")


# ----------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["service"], summary="Liveness and loaded-artifact summary")
def health() -> dict:
    return engine().health()


@app.get("/presets", tags=["service"], summary="The four one-click scenarios used by the workbench")
def presets() -> list[dict]:
    return engine().presets()


@app.get("/models", tags=["service"], summary="Available CATE learners and feature order")
def models() -> dict:
    e = engine()
    return {"learners": [k for k in LEARNER_ORDER if k in e.cate_models], "best": e.best_learner, "features": e.feature_names}


@app.post("/predict/att", tags=["prediction"], summary="Aggregate and dynamic ATT for an exposed market vs. a donor pool")
def predict_att(req: AttRequest) -> dict:
    try:
        out = engine().att(req.scenario, req.estimator, req.outcome, live=req.live or bool(req.treated or req.donors),
                           treated=req.treated, donors=req.donors)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if not req.include_series:
        out = {k: v for k, v in out.items() if k != "series"}
    return out


@app.post("/predict/cate", tags=["prediction"], summary="Conditional campaign lift for market-day conditions")
def predict_cate(req: CateRequest) -> dict:
    try:
        rows = [r.model_dump(exclude_none=True) for r in req.rows]
        return engine().cate(rows, req.learner, req.baseline_room_nights, req.alpha)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/simulate/campaign", tags=["simulation"], summary="Budget and pacing what-if: room nights, CPIRN and iROAS")
def simulate_campaign(req: CampaignRequest) -> dict:
    try:
        return engine().simulate(req.budget_usd, req.strategy, req.learner, req.baseline_room_nights, req.overrides,
                                 req.net_revenue_per_room_night)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
