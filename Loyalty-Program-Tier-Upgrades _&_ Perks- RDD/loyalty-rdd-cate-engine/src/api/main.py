"""FastAPI service: /health, /predict/late, /predict/cate, /simulate/threshold (+ /leaderboard)."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from src.api.schemas import (CateRequest, CateResponse, HealthResponse, LateRequest, LateResponse,
                             SimulateRequest, SimulateResponse)
from src.api.service import ModelService, ModelsNotTrained, OUTCOME_SHORT


def create_app(outputs: Optional[str] = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            state["svc"] = ModelService(outputs)
        except ModelsNotTrained as exc:
            state["error"] = str(exc)
        yield

    app = FastAPI(
        title="Loyalty Tier RDD + CATE Engine",
        description="Causal effect of reaching Gold status (20 qualifying nights) on next-year room nights, "
                    "ancillary spend and direct-booking ratio. Regression discontinuity for the average effect at "
                    "the cutoff; kernel-weighted CATE learners for who responds most.",
        version="0.1.0", lifespan=lifespan)

    def svc() -> ModelService:
        if "svc" not in state:
            raise HTTPException(503, state.get("error", "models not loaded"))
        return state["svc"]

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        return svc().health()

    @app.get("/leaderboard", tags=["system"], summary="CATE learner leaderboard (out-of-fold metrics)")
    def leaderboard(outcome: str = "room_nights"):
        lb = svc().leaderboards.get(outcome)
        if lb is None:
            raise HTTPException(404, f"no leaderboard for '{outcome}'")
        return lb.where(lb.notna(), None).to_dict("records")

    @app.post("/predict/late", response_model=LateResponse, tags=["RDD"],
              summary="Local average treatment effect of Gold status at the cutoff")
    def predict_late(req: LateRequest):
        res = svc().late(req.outcome, req.kernel, req.bandwidth, req.poly_order, req.design)
        if res["estimate"] is None:
            raise HTTPException(422, "not enough support points in this bandwidth for the chosen polynomial order")
        return res

    @app.post("/predict/cate", response_model=CateResponse, tags=["CATE"],
              summary="Individualised effect of Gold status for one guest profile")
    def predict_cate(req: CateRequest):
        try:
            return svc().cate(req.guest.model_dump(), req.outcome, req.learner, req.compare_all)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\""))

    @app.post("/simulate/threshold", response_model=SimulateResponse, tags=["simulation"],
              summary="What if the Gold threshold moved?")
    def simulate_threshold(req: SimulateRequest):
        try:
            return svc().simulate_threshold(req.new_cutoff, req.outcome, req.learner, req.perk_cost_per_member)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\""))

    return app


app = create_app(os.environ.get("LOYALTY_OUTPUTS"))
