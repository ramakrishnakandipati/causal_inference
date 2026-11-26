"""Pydantic v2 request / response schemas with hospitality-domain validation."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

HomeRegion = Literal["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa"]
AgeGroup = Literal["18-29", "30-39", "40-49", "50-64", "65+"]
OutcomeName = Literal["next_year_room_nights", "next_year_ancillary_spend", "next_year_direct_booking_ratio"]
OutcomeShort = Literal["room_nights", "ancillary_spend", "direct_booking_ratio"]
Kernel = Literal["triangular", "epanechnikov", "uniform"]


class GuestProfile(BaseModel):
    """A loyalty member's year-t profile (the effect modifiers X)."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "historical_adr": 165.0, "business_travel_ratio": 0.4, "lead_time_avg": 35.0, "ancillary_spend_t": 650.0,
        "direct_booking_ratio_t": 0.35, "tenure_years": 4.0, "co_brand_card_holder": False,
        "home_region": "North America", "age_group": "30-39", "weekend_stay_ratio": 0.3}})

    historical_adr: float = Field(..., ge=30, le=2000, description="Average daily rate paid, USD")
    business_travel_ratio: float = Field(..., ge=0, le=1, description="Share of stays that are business trips")
    lead_time_avg: float = Field(..., ge=0, le=365, description="Average booking lead time, days")
    ancillary_spend_t: float = Field(..., ge=0, le=100_000, description="F&B/spa/resort spend in year t, USD")
    direct_booking_ratio_t: float = Field(..., ge=0, le=1, description="Share of stays booked direct in year t")
    tenure_years: float = Field(..., ge=0, le=60, description="Years in the loyalty programme")
    co_brand_card_holder: bool = Field(..., description="Holds the co-branded credit card")
    home_region: HomeRegion
    age_group: AgeGroup
    weekend_stay_ratio: float = Field(..., ge=0, le=1, description="Share of nights that fall on weekends")

    @model_validator(mode="after")
    def _consistency(self):
        if self.tenure_years < 1 and self.co_brand_card_holder and self.ancillary_spend_t > 20_000:
            raise ValueError("implausible profile: first-year member with >$20k ancillary spend")
        return self


class HealthResponse(BaseModel):
    status: str
    cutoff: float
    regime: str
    outcomes_trained: List[str]
    best_learner: Dict[str, Optional[str]]
    cuda_available: bool
    gpu_name: Optional[str] = None
    n_panel: int


class LateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "outcome": "next_year_room_nights", "kernel": "triangular", "bandwidth": None, "poly_order": 1, "design": None}})

    outcome: OutcomeName = "next_year_room_nights"
    kernel: Kernel = "triangular"
    bandwidth: Optional[float] = Field(None, ge=4, le=20, description="nights; default = CCT-optimal")
    poly_order: Literal[1, 2] = 1
    design: Optional[Literal["sharp", "fuzzy"]] = Field(None, description="default = the regime in the data")


class LateResponse(BaseModel):
    outcome: str
    design: str
    kernel: str
    poly_order: int
    bandwidth: float
    estimate: Optional[float]
    se: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    p_value: Optional[float]
    n_left: int
    n_right: int
    true_local_late: float = Field(..., description="Known truth in the synthetic data")
    first_stage_jump: Optional[float] = None


class CateRequest(BaseModel):
    # explicit example: Swagger would otherwise pre-fill "learner": "string", which is not a model name
    model_config = ConfigDict(json_schema_extra={"example": {
        "guest": GuestProfile.model_config["json_schema_extra"]["example"],
        "outcome": "room_nights", "learner": None, "compare_all": True}})

    guest: GuestProfile
    outcome: OutcomeShort = "room_nights"
    learner: Optional[str] = Field(None, description="default = best learner for the outcome")
    compare_all: bool = True


class CateResponse(BaseModel):
    outcome: str
    learner: str
    predicted_effect: float
    unit: str
    interval_95: Optional[List[float]] = None
    percentile_among_guests: float
    population_average_effect: float
    segment: str
    recommendation: str
    all_learner_estimates: Optional[Dict[str, float]] = None


class SimulateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "new_cutoff": 17, "outcome": "room_nights", "learner": None, "perk_cost_per_member": 120.0}})

    new_cutoff: int = Field(..., ge=8, le=40, description="Candidate Gold threshold, nights")
    outcome: OutcomeShort = "room_nights"
    learner: Optional[str] = None
    perk_cost_per_member: float = Field(0.0, ge=0, le=5000, description="USD cost of Gold perks per activated member")


class SimulateResponse(BaseModel):
    new_cutoff: int
    direction: str
    guests_affected: int
    share_of_members_affected: float
    avg_effect_per_guest: float
    incremental_total: float
    unit: str
    incremental_revenue_usd: Optional[float]
    perk_cost_usd: float
    net_value_usd: Optional[float]
    learner: str
    take_up_factor: float
    warnings: List[str]
