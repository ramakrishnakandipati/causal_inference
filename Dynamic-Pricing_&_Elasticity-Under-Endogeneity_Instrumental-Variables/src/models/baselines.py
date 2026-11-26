"""Biased baselines that ignore the unobserved demand shock U."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data.generate_hospitality_data import FEATURES, OUTCOME, TREATMENT
from src.models.base import LINEAR_FEATURES, BaseElasticityModel
from src.utils.gpu_utils import make_xgb_regressor


class NaiveOLS(BaseElasticityModel):
    """Y ~ ln p + X (Ridge with a tiny penalty). Expected: positive, upward-biased price coefficient."""

    name = "Naive OLS"
    family = "naive"

    def fit(self, train: pd.DataFrame) -> "NaiveOLS":
        t0 = time.time()
        self.scaler = StandardScaler().fit(train[LINEAR_FEATURES].to_numpy(float))
        x = self.scaler.transform(train[LINEAR_FEATURES].to_numpy(float))
        design = np.column_stack([train[TREATMENT].to_numpy(float), x])
        self.model = Ridge(alpha=self.cfg["models"]["ridge_alpha"] * 1e-3).fit(design, train[OUTCOME].to_numpy(float))
        self.theta = float(self.model.coef_[0])
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.theta)

    def get_diagnostics(self):
        return {"price_coefficient": self.theta}


class NaiveXGB(BaseElasticityModel):
    """Gradient-boosted Y ~ f(ln p, X). Elasticity = central finite difference in ln p. Ignores U."""

    name = "Naive XGBoost"
    family = "naive"

    def __init__(self, cfg=None, h: float = 0.10):
        super().__init__(cfg)
        self.h = h

    def _x(self, df: pd.DataFrame, ln_p: np.ndarray) -> np.ndarray:
        return np.column_stack([ln_p, df[FEATURES].to_numpy(float)])

    def fit(self, train: pd.DataFrame) -> "NaiveXGB":
        t0 = time.time()
        self.model = make_xgb_regressor(self.cfg)
        self.model.fit(self._x(train, train[TREATMENT].to_numpy(float)), train[OUTCOME].to_numpy(float))
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        # Differentiate around the *reference price* of each row so the slope is read at a common price level.
        from data.generate_hospitality_data import ln_reference_price

        p0 = ln_reference_price(df, self.cfg)
        hi = self.model.predict(self._x(df, p0 + self.h))
        lo = self.model.predict(self._x(df, p0 - self.h))
        return (hi - lo) / (2 * self.h)

    def predict_log_demand(self, df: pd.DataFrame, ln_price: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._x(df, np.asarray(ln_price, float))), float)
