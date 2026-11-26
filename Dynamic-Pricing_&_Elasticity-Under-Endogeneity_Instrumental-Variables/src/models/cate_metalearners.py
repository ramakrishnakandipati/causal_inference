"""CATE meta-learners, causal forest and uplift random forest for the promotional-discount treatment.

Treatment W = promo (binary), outcome Y = log demand, features = observed X + offered ADR.
Each model returns tau_hat(X): effect of the standard promotion on log demand. The implied price
elasticity is tau_hat / ln(1 - discount), which lets every model be scored against the same ground truth.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from data.generate_hospitality_data import OUTCOME
from src.models.base import META_FEATURES, BaseElasticityModel
from src.utils.gpu_utils import (make_catboost_regressor, make_lgbm_regressor, make_xgb_classifier,
                                 make_xgb_regressor)


def _xy(df: pd.DataFrame):
    return df[META_FEATURES].to_numpy(float), df["promo"].to_numpy(int), df[OUTCOME].to_numpy(float)


class CateMeta(BaseElasticityModel):
    """kind: s | t | x | r | dr | causal_forest | uplift_rf"""

    family = "cate_meta"
    native_target = "promo_cate"
    _NAMES = {"s": "S-Learner", "t": "T-Learner", "x": "X-Learner", "r": "R-Learner (NonParamDML)",
              "dr": "DR-Learner", "causal_forest": "Causal Forest DML", "uplift_rf": "Uplift RF (causalml)"}

    def __init__(self, kind: str, cfg=None, max_rows: int | None = None):
        super().__init__(cfg)
        self.kind = kind
        self.name = self._NAMES[kind]
        self.max_rows = max_rows if kind in ("causal_forest", "uplift_rf") else None

    # -- model factories ---------------------------------------------------------------------
    def _reg(self):
        return make_xgb_regressor(self.cfg)

    def _clf(self):
        return make_xgb_classifier(self.cfg)

    def _build(self):
        from econml.dml import CausalForestDML, NonParamDML
        from econml.dr import DRLearner
        from econml.metalearners import SLearner, TLearner, XLearner

        seed = self.cfg["project"]["seed"]
        if self.kind == "s":
            return SLearner(overall_model=self._reg())
        if self.kind == "t":
            return TLearner(models=[make_catboost_regressor(self.cfg), make_catboost_regressor(self.cfg)])
        if self.kind == "x":
            return XLearner(models=[self._reg(), self._reg()], propensity_model=self._clf(),
                            cate_models=[self._reg(), self._reg()])
        if self.kind == "r":
            return NonParamDML(model_y=self._reg(), model_t=self._clf(),
                               model_final=make_lgbm_regressor(self.cfg, n_estimators=150, max_depth=3),
                               discrete_treatment=True, cv=3, random_state=seed)
        if self.kind == "dr":
            return DRLearner(model_propensity=self._clf(), model_regression=self._reg(),
                             model_final=make_lgbm_regressor(self.cfg, n_estimators=150, max_depth=3),
                             cv=3, random_state=seed)
        if self.kind == "causal_forest":
            fo = self.cfg["models"]["forest"]
            return CausalForestDML(model_y=self._reg(), model_t=self._clf(), discrete_treatment=True,
                                   n_estimators=fo["n_estimators"], min_samples_leaf=fo["min_samples_leaf"],
                                   max_depth=12, cv=3, random_state=seed)
        raise ValueError(self.kind)

    # -- fit / predict -----------------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "CateMeta":
        t0 = time.time()
        if self.max_rows and len(train) > self.max_rows:
            train = train.sample(self.max_rows, random_state=self.cfg["project"]["seed"])
        x, w, y = _xy(train)
        if self.kind == "uplift_rf":
            self._fit_uplift(train)
        else:
            self.est = self._build()
            if self.kind in ("r", "causal_forest"):
                self.est.fit(y, w, X=x)
            else:
                self.est.fit(y, w, X=x)
        self.fit_seconds = time.time() - t0
        self.fit_structural_intercept(train)
        return self

    def _fit_uplift(self, train: pd.DataFrame) -> None:
        """causalml UpliftRandomForestClassifier on the binary conversion outcome.

        The uplift is a change in conversion *probability*; it is mapped back to the log-demand scale
        through the saturating occupancy link occ = 1 - exp(-D) using a control-group occupancy model.
        """
        from causalml.inference.tree import UpliftRandomForestClassifier

        x, w, _ = _xy(train)
        conv = train["converted"].to_numpy(int)
        fo = self.cfg["models"]["forest"]
        self.uplift = UpliftRandomForestClassifier(
            n_estimators=min(100, fo["n_estimators"]), max_depth=7, min_samples_leaf=max(50, fo["min_samples_leaf"] * 3),
            min_samples_treatment=50, n_reg=50, evaluationFunction="CTS", control_name="control",
            random_state=self.cfg["project"]["seed"], n_jobs=-1)
        self.uplift.fit(x, treatment=np.where(w == 1, "treatment", "control"), y=conv)
        ctrl = train["promo"].to_numpy() == 0
        self._occ0 = make_xgb_regressor(self.cfg)
        self._occ0.fit(x[ctrl], train["occupancy_rate"].to_numpy(float)[ctrl])

    def _uplift_to_log_effect(self, x: np.ndarray, uplift_prob: np.ndarray) -> np.ndarray:
        occ0 = np.clip(self._occ0.predict(x), 0.02, 0.97)
        occ1 = np.clip(occ0 + uplift_prob, 0.01, 0.995)
        return np.log(-np.log(1.0 - occ1)) - np.log(-np.log(1.0 - occ0))

    def promo_cate(self, df: pd.DataFrame) -> np.ndarray:
        x = df[META_FEATURES].to_numpy(float)
        if self.kind == "uplift_rf":
            raw = np.asarray(self.uplift.predict(x)).reshape(len(df), -1)[:, 0]
            return self._uplift_to_log_effect(x, raw)
        return np.asarray(self.est.effect(x), float).ravel()

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        return self.promo_cate(df) / self.ln_one_minus_d

    def uplift_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Effect of the promotion on conversion probability (used for CRM targeting displays)."""
        x = df[META_FEATURES].to_numpy(float)
        if self.kind == "uplift_rf":
            return np.asarray(self.uplift.predict(x)).reshape(len(df), -1)[:, 0]
        return np.asarray(self.est.effect(x), float).ravel()


def causalml_cross_check(train: pd.DataFrame, test: pd.DataFrame, cfg: Dict[str, Any] | None = None) -> Dict[str, np.ndarray]:
    """Re-estimate S/T/X/R promo effects with causalml's meta-learners (independent implementation)."""
    from causalml.inference.meta import BaseRRegressor, BaseSRegressor, BaseTRegressor, BaseXRegressor

    x, w, y = _xy(train)
    xt = test[META_FEATURES].to_numpy(float)
    out: Dict[str, np.ndarray] = {}
    mk = lambda: make_xgb_regressor(cfg, n_estimators=120)  # noqa: E731
    s = BaseSRegressor(learner=mk()); s.fit(x, w, y); out["S"] = np.asarray(s.predict(xt)).ravel()
    t = BaseTRegressor(learner=mk()); t.fit(x, w, y); out["T"] = np.asarray(t.predict(xt)).ravel()
    xl = BaseXRegressor(learner=mk()); xl.fit(x, w, y); out["X"] = np.asarray(xl.predict(xt, p=np.full(len(xt), w.mean()))).ravel()
    r = BaseRRegressor(learner=mk()); r.fit(x, w, y, p=np.full(len(x), w.mean())); out["R"] = np.asarray(r.predict(xt)).ravel()
    return out
