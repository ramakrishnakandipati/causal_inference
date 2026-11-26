"""Model service shared by the FastAPI app and the Streamlit dashboard.

Loads the artifacts written by ``src.models.trainer`` (panel, RDD suite, CATE learners, metadata).
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data import synthetic_generator as gen
from src.models import rdd_estimators as R
from src.utils import device

warnings.filterwarnings("ignore")

OUTCOME_SHORT = {"next_year_room_nights": "room_nights", "next_year_ancillary_spend": "ancillary_spend",
                 "next_year_direct_booking_ratio": "direct_booking_ratio"}
SHORT_TO_OUTCOME = {v: k for k, v in OUTCOME_SHORT.items()}
UNITS = {"room_nights": "nights", "ancillary_spend": "USD", "direct_booking_ratio": "share"}


def to_cpu_inference(obj, _seen: Optional[set] = None) -> int:
    """Switch every XGBoost estimator inside a fitted learner to CPU prediction (returns how many were switched).

    Why: XGBoost 2.x keeps CUDA thread-local state; when a short-lived thread (a Streamlit script run, a test
    client portal) that predicted on the GPU exits, the process aborts natively (0xC0000409 on Windows).
    Serving single-guest predictions on CPU is fast and removes the hazard. Training still uses the GPU.
    """
    import xgboost as xgb
    seen = _seen if _seen is not None else set()
    if id(obj) in seen or obj is None or isinstance(obj, (str, bytes, int, float, bool, np.ndarray, pd.DataFrame)):
        return 0
    seen.add(id(obj))
    n = 0
    if isinstance(obj, xgb.XGBModel):
        obj.set_params(device="cpu")
        try:
            obj.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass
        return 1
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    else:
        return 0
    for ch in children:
        n += to_cpu_inference(ch, seen)
    return n


def default_outputs_dir() -> Path:
    return Path(os.environ.get("LOYALTY_OUTPUTS", "outputs"))


class ModelsNotTrained(RuntimeError):
    pass


class ModelService:
    def __init__(self, outputs: Optional[os.PathLike] = None):
        self.out = Path(outputs) if outputs else default_outputs_dir()
        meta_path = self.out / "models" / "meta.json"
        if not meta_path.exists():
            raise ModelsNotTrained(f"{meta_path} not found - run `make train` first")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.cutoff = float(self.meta["cutoff"])
        self.regime = self.meta["regime"]
        self.covariates: List[str] = self.meta["covariates"]
        self.panel = pd.read_csv(self.out / "models" / "panel.csv.gz")
        self.panel.attrs.update({"regime": self.regime, "cutoff": self.cutoff})
        self.rdd = joblib.load(self.out / "models" / "rdd_model.joblib")["suite"]
        self._models: Dict[str, Dict[str, object]] = {}
        self._ref: Dict[tuple, np.ndarray] = {}
        self.leaderboards = {p.stem.replace("cate_leaderboard_", ""): pd.read_csv(p)
                             for p in sorted((self.out / "metrics").glob("cate_leaderboard_*.csv"))}

    # ------------------------------------------------------------------ catalog
    @property
    def outcomes(self) -> List[str]:
        return list(self.leaderboards)

    def learners(self, outcome_short: str) -> List[str]:
        d = self.out / "models" / outcome_short
        return sorted(p.stem for p in d.glob("*.joblib")) if d.exists() else []

    def best_learner(self, outcome_short: str) -> Optional[str]:
        return self.meta.get("best_learner", {}).get(outcome_short)

    def model(self, outcome_short: str, learner: Optional[str] = None):
        learner = learner or self.best_learner(outcome_short)
        if learner is None or learner not in self.learners(outcome_short):
            raise KeyError(f"no model '{learner}' for outcome '{outcome_short}'")
        cache = self._models.setdefault(outcome_short, {})
        if learner not in cache:
            m = joblib.load(self.out / "models" / outcome_short / f"{learner}.joblib")
            if os.environ.get("LOYALTY_GPU_INFERENCE", "0") != "1":
                to_cpu_inference(m)
            cache[learner] = m
        return learner, cache[learner]

    def health(self) -> dict:
        d = device.detect()
        return {"status": "ok", "cutoff": self.cutoff, "regime": self.regime,
                "outcomes_trained": self.outcomes,
                "best_learner": self.meta.get("best_learner", {}),
                "cuda_available": bool(d["cuda"]), "gpu_name": d["gpu_name"], "n_panel": int(len(self.panel))}

    # ------------------------------------------------------------------ LATE
    def late(self, outcome: str, kernel: str = "triangular", bandwidth: Optional[float] = None,
             order: int = 1, design: Optional[str] = None) -> dict:
        h = float(bandwidth or self.meta["bandwidths"]["used"])
        fuzzy = (design or self.regime) == "fuzzy"
        res = R.rdd_estimate(self.panel, outcome, self.cutoff, h, order, kernel, fuzzy)
        truth = gen.local_truth(self.panel)[outcome]
        res = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res.items()}
        return {"outcome": outcome, "design": "fuzzy" if fuzzy else "sharp", "kernel": kernel, "poly_order": order,
                "bandwidth": h, "true_local_late": truth, **res}

    # ------------------------------------------------------------------ CATE
    def _vector(self, profile: dict) -> np.ndarray:
        regions, ages = self.meta["home_regions"], self.meta["age_groups"]
        row = dict(profile)
        if isinstance(row["home_region"], str):
            row["home_region"] = regions.index(row["home_region"])
        if isinstance(row["age_group"], str):
            row["age_group"] = ages.index(row["age_group"])
        row["co_brand_card_holder"] = int(bool(row["co_brand_card_holder"]))
        return np.array([[float(row[c]) for c in self.covariates]])

    def _reference(self, outcome_short: str, learner: str) -> np.ndarray:
        key = (outcome_short, learner)
        if key not in self._ref:
            h = float(self.meta["cate_bandwidth"])
            win = self.panel[(self.panel["qualifying_nights_t"] - self.cutoff).abs() <= h]
            win = win.sample(min(3000, len(win)), random_state=0)
            _, m = self.model(outcome_short, learner)
            self._ref[key] = np.sort(m.predict(win[self.covariates].to_numpy(float)))
        return self._ref[key]

    @staticmethod
    def segment(p: dict) -> str:
        b, wk = p["business_travel_ratio"], p["weekend_stay_ratio"]
        if b >= 0.6:
            return "Corporate road warrior"
        if b >= 0.3 and wk >= 0.25:
            return "Bleisure traveller"
        if b < 0.3:
            return "Leisure traveller"
        return "Mixed-purpose traveller"

    def cate(self, profile: dict, outcome_short: str = "room_nights", learner: Optional[str] = None,
             compare_all: bool = True) -> dict:
        x = self._vector(profile)
        name, m = self.model(outcome_short, learner)
        tau = float(np.asarray(m.predict(x)).ravel()[0])
        ref = self._reference(outcome_short, name)
        pct = float(np.searchsorted(ref, tau) / len(ref) * 100)
        out = {"outcome": SHORT_TO_OUTCOME[outcome_short], "learner": name, "predicted_effect": tau,
               "unit": UNITS[outcome_short], "percentile_among_guests": pct,
               "population_average_effect": float(ref.mean()), "segment": self.segment(profile)}
        out["interval_95"] = None
        est = getattr(m, "est_", None)
        if est is not None and hasattr(est, "effect_interval"):
            try:
                lo, hi = est.effect_interval(x, alpha=0.05)
                out["interval_95"] = [float(np.ravel(lo)[0]), float(np.ravel(hi)[0])]
            except Exception:
                pass
        if compare_all:
            alt = {}
            for l in self.learners(outcome_short):
                try:
                    alt[l] = float(np.asarray(self.model(outcome_short, l)[1].predict(x)).ravel()[0])
                except Exception:
                    continue
            out["all_learner_estimates"] = alt
        ratio = tau / out["population_average_effect"] if out["population_average_effect"] else 1.0
        out["recommendation"] = (
            "High responder: prioritise Gold perks and proactive status-match offers." if ratio >= 1.25 else
            "Average responder: standard Gold programme is appropriate." if ratio >= 0.75 else
            "Low responder: perks add little incremental stay behaviour; avoid costly upgrades.")
        return out

    # ------------------------------------------------------------------ threshold simulation
    def simulate_threshold(self, new_cutoff: float, outcome_short: str = "room_nights",
                           learner: Optional[str] = None, perk_cost_per_member: float = 0.0) -> dict:
        c = self.cutoff
        nm, m = self.model(outcome_short, learner)
        r = self.panel["qualifying_nights_t"].to_numpy(float)
        lowering = new_cutoff < c
        mask = (r >= new_cutoff) & (r < c) if lowering else (r >= c) & (r < new_cutoff)
        aff = self.panel[mask]
        main = self.rdd["main"].set_index("outcome")
        fs = float(main.iloc[0].get("first_stage_jump", np.nan)) if self.regime == "fuzzy" else 1.0
        fs = 1.0 if not np.isfinite(fs) else fs
        warnings_ = []
        if abs(new_cutoff - c) > 5:
            warnings_.append("Cutoff moved more than 5 nights: effects are extrapolated well beyond the RDD's local validity.")
        if len(aff) == 0:
            return {"new_cutoff": new_cutoff, "direction": "no change", "guests_affected": 0, "share_of_members_affected": 0.0,
                    "avg_effect_per_guest": 0.0, "incremental_total": 0.0, "incremental_revenue_usd": 0.0,
                    "perk_cost_usd": 0.0, "net_value_usd": 0.0, "learner": nm, "unit": UNITS[outcome_short],
                    "take_up_factor": fs, "warnings": warnings_}
        tau = np.asarray(m.predict(aff[self.covariates].to_numpy(float))).ravel() * fs
        sign = 1.0 if lowering else -1.0
        total = float(sign * tau.sum())
        revenue = float(sign * (tau * aff["historical_adr"].to_numpy()).sum()) if outcome_short == "room_nights" else None
        cost = float(perk_cost_per_member * len(aff) * fs * sign)
        return {"new_cutoff": new_cutoff, "direction": "lower (more guests reach Gold)" if lowering else "raise (fewer guests reach Gold)",
                "guests_affected": int(len(aff)), "share_of_members_affected": float(len(aff) / len(self.panel)),
                "avg_effect_per_guest": float(tau.mean()), "incremental_total": total, "unit": UNITS[outcome_short],
                "incremental_revenue_usd": revenue, "perk_cost_usd": cost,
                "net_value_usd": None if revenue is None else float(revenue - cost), "learner": nm,
                "take_up_factor": fs, "warnings": warnings_}
