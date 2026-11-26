"""Common interface: every estimator exposes theta_hat(X) = d log demand / d ln price."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from data.generate_hospitality_data import FEATURES, INSTRUMENTS, OUTCOME, TREATMENT, ln_reference_price
from src.utils.gpu_utils import load_config, make_xgb_regressor

# Collinearity-safe design for linear / econometric models (drops one level per dummy family).
LINEAR_FEATURES: List[str] = [f for f in FEATURES if f not in ("seg_group", "tier_resort")]
META_FEATURES: List[str] = FEATURES + ["ln_adr"]


class BaseElasticityModel:
    """Subclasses implement fit() and elasticity(); the structural demand curve is derived from them."""

    name: str = "base"
    family: str = "naive"          # naive | iv | deep_iv | cate_meta | cate_deep
    native_target: str = "elasticity"   # 'elasticity' (IV-style) or 'promo_cate' (meta-learner style)
    is_iv: bool = False

    def __init__(self, cfg: Dict[str, Any] | None = None):
        self.cfg = cfg or load_config()
        self.fit_seconds: float = 0.0
        self._intercept_model = None
        self.loss_history: Dict[str, List[float]] = {}

    # -- interface ---------------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "BaseElasticityModel":  # pragma: no cover - abstract
        raise NotImplementedError

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- derived quantities -------------------------------------------------------------------
    @property
    def ln_one_minus_d(self) -> float:
        return math.log(1.0 - self.cfg["dgp"]["promo"]["discount"])

    def promo_cate(self, df: pd.DataFrame) -> np.ndarray:
        """Effect of the standard promotion on log demand: theta(X) * ln(1 - d)."""
        return self.elasticity(df) * self.ln_one_minus_d

    def fit_structural_intercept(self, train: pd.DataFrame) -> None:
        """alpha(X) = E[Y - theta_hat(X) * T | X]; U has mean zero given X so alpha is the structural intercept."""
        theta = self.elasticity(train)
        # Anchor at the reference price for numerical stability: alpha(X) = E[Y - theta(X) (T - ln p_ref(X)) | X]
        resid = train[OUTCOME].to_numpy(float) - theta * (train[TREATMENT].to_numpy(float) - ln_reference_price(train, self.cfg))
        m = make_xgb_regressor(self.cfg, n_estimators=150, max_depth=3, learning_rate=0.08)
        m.fit(train[FEATURES].to_numpy(float), resid)
        self._intercept_model = m

    def intercept(self, df: pd.DataFrame) -> np.ndarray:
        if self._intercept_model is None:
            raise RuntimeError("fit_structural_intercept() has not been called")
        return np.asarray(self._intercept_model.predict(df[FEATURES].to_numpy(float)), float)

    def predict_log_demand(self, df: pd.DataFrame, ln_price: np.ndarray) -> np.ndarray:
        return self.intercept(df) + self.elasticity(df) * (np.asarray(ln_price, float) - ln_reference_price(df, self.cfg))

    def get_diagnostics(self) -> Dict[str, Any]:
        return {}

    def to_cpu(self) -> "BaseElasticityModel":
        """Move every torch module to CPU so the fitted object can be pickled / served on any machine."""
        import torch
        import torch.nn as nn

        for v in list(vars(self).values()):
            if isinstance(v, nn.Module):
                v.cpu()
        if hasattr(self, "device"):
            self.device = torch.device("cpu")
        return self
