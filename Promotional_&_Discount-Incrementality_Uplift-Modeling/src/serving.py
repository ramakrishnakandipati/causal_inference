"""Shared real-time scoring logic used by both the FastAPI service and the Streamlit simulator."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from src.common import FEATURES, LOYALTY_LEVELS, get_logger, load_config, resolve
from src.evaluate import assign_quadrant, decide, points_tau_margin, stay_value_proxy

log = get_logger()

PRESETS = {
    "Weekend Leisure Browser": {"persona": "Persuadable", "hist_booking_freq": 3.0, "days_since_last_stay": 170, "loyalty_tier": "Silver",
                                "hist_price_sensitivity": 0.82, "avg_lead_time": 30, "avg_adr": 150, "weekend_stay_ratio": 0.85,
                                "cancellation_rate": 0.07, "email_open_rate_90d": 0.50, "points_balance": 4000},
    "Frequent Corporate Platinum": {"persona": "Always-Taker", "hist_booking_freq": 14.0, "days_since_last_stay": 25, "loyalty_tier": "Platinum",
                                    "hist_price_sensitivity": 0.15, "avg_lead_time": 5, "avg_adr": 260, "weekend_stay_ratio": 0.15,
                                    "cancellation_rate": 0.04, "email_open_rate_90d": 0.60, "points_balance": 85000},
    "Dormant 2-Year Lapsed Guest": {"persona": "Lost Cause", "hist_booking_freq": 0.8, "days_since_last_stay": 780, "loyalty_tier": "Non-Member",
                                    "hist_price_sensitivity": 0.40, "avg_lead_time": 35, "avg_adr": 120, "weekend_stay_ratio": 0.40,
                                    "cancellation_rate": 0.10, "email_open_rate_90d": 0.04, "points_balance": 800},
    "Full-Rate Loyal Guest with Active Booking": {"persona": "Sleeping Dog", "hist_booking_freq": 5.0, "days_since_last_stay": 60, "loyalty_tier": "Gold",
                                                  "hist_price_sensitivity": 0.08, "avg_lead_time": 40, "avg_adr": 215, "weekend_stay_ratio": 0.35,
                                                  "cancellation_rate": 0.42, "email_open_rate_90d": 0.35, "points_balance": 22000},
}


def _force_cpu(obj, seen: set, depth: int = 0):
    """Recursively set every XGBoost estimator inside `obj` to device='cpu' and move torch modules to CPU."""
    if id(obj) in seen or depth > 6 or obj is None or isinstance(obj, (str, int, float, bool, np.ndarray, pd.DataFrame)):
        return
    seen.add(id(obj))
    mod = type(obj).__module__
    if mod.startswith("xgboost"):
        try:
            obj.set_params(device="cpu")
        except Exception:
            try:
                obj.set_param({"device": "cpu"})
            except Exception:
                pass
        return
    if mod.startswith("torch"):
        if hasattr(obj, "cpu"):
            try:
                obj.cpu()
            except Exception:
                pass
        return
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())
    else:
        return
    for c in children:
        _force_cpu(c, seen, depth + 1)


class Scorer:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        md = resolve(self.cfg["paths"]["models"])
        aux = ("baseline_conv", "conv_companion", "propensity")
        self.models = {p.stem: joblib.load(p) for p in md.glob("*.joblib") if p.stem not in aux}
        from src.models import ESTIMATORS
        self.order = [n for n in ESTIMATORS if n in self.models]
        self.prop = joblib.load(md / "propensity.joblib") if (md / "propensity.joblib").exists() else None
        self.conv = joblib.load(md / "conv_companion.joblib")
        self.base = joblib.load(md / "baseline_conv.joblib")
        self.surrogates = {p.stem: joblib.load(p) for p in (md / "surrogates").glob("*.joblib")}
        import json
        s = resolve(self.cfg["paths"]["logs"]) / "summary.json"
        self.best = json.load(open(s))["best_model"] if s.exists() else next(iter(self.models))
        # serve entirely on CPU: avoids CUDA/XGBoost calls from multiple server threads (hard-crashed Streamlit reruns)
        seen: set = set()
        for obj in [*self.models.values(), self.conv, self.base, self.prop, *self.surrogates.values()]:
            _force_cpu(obj, seen)

    @staticmethod
    def frame(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df["loyalty_tier"].dtype == object:
            df["loyalty_tier"] = df["loyalty_tier"].map({n: i for i, n in enumerate(LOYALTY_LEVELS)})
        return df[FEATURES].astype(float)

    def propensity(self, rows: list[dict]) -> np.ndarray:
        """Historical-CRM propensity e(x) = P(T=1 | X), clipped to the overlap range used for IPW."""
        if self.prop is None:
            raise RuntimeError("propensity.joblib missing - run `python main.py --stage evaluate`")
        lo, hi = self.cfg["evaluation"]["propensity_clip"]
        return np.clip(self.prop.predict_proba(self.frame(rows).values.astype(np.float32))[:, 1], lo, hi)

    def score(self, rows: list[dict], estimator: str | None = None, offer: dict | None = None, explain: bool = False) -> list[dict]:
        t0 = time.perf_counter()
        est = estimator or self.best
        if est not in self.models:
            raise KeyError(f"unknown estimator {est}; available={list(self.models)}")
        cfg = self.cfg
        if offer:
            cfg = {**cfg, "offer": {**cfg["offer"], **offer}}
        X = self.frame(rows)
        m = self.models[est]
        tau = m.predict(X)
        ci = m.predict_interval(X)
        tc = self.conv.predict(X)
        p0 = self.base.predict_proba(X.values.astype(np.float32))[:, 1]
        R = stay_value_proxy(X)
        quad = assign_quadrant(tc, tau, p0, cfg)
        tp = points_tau_margin(tc, p0, R, cfg["offer"])
        dec = decide(tau, tc, p0, R, quad, cfg, tau_points=tp)
        ms = 1000 * (time.perf_counter() - t0)
        out = []
        for i in range(len(X)):
            r = {"estimator": est, "tau_margin_hat": float(tau[i]), "tau_conv_hat": float(tc[i]), "baseline_conv_prob": float(p0[i]),
                 "tau_margin_points_hat": float(tp[i]), "expected_stay_value": float(R[i]),
                 "ci95_low": None if ci is None else float(ci[0][i]), "ci95_high": None if ci is None else float(ci[1][i]),
                 "quadrant": str(quad[i]), "decision": str(dec[i]), "latency_ms": ms / len(X)}
            if explain and est in self.surrogates:
                import shap
                ex = shap.TreeExplainer(self.surrogates[est])(X.iloc[[i]])
                r["shap"] = {"base_value": float(ex.base_values[0]), "values": dict(zip(FEATURES, map(float, ex.values[0]))),
                             "data": dict(zip(FEATURES, map(float, X.iloc[i])))}
            out.append(r)
        return out
