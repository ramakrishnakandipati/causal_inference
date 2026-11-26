"""EconML Double ML family: LinearDML, NonParamDML and CausalForestDML (GPU GBDT nuisances).

All estimators cross-fit with DMA-grouped folds (``groups=`` is passed to ``fit``).
LinearDML uses only a handful of named drivers: on the full, collinear feature set its linear
CATE is unstable.
"""
from __future__ import annotations

import warnings

import numpy as np

from src.data.synthetic_geo_generator import CATE_FEATURES
from src.models.meta_learners import BaseCATE, E_KNOWN, make_classifier, make_regressor
from src.utils.config import load_config

LINEAR_DRIVERS = ["tier", "organic_rank", "ota_pressure", "loyalty_pen", "pre_ncm_mean"]


class _Wrap(BaseCATE):
    def __init__(self, seed: int = 0, n_splits: int = 5):
        super().__init__("xgb", "estimated", seed, n_splits)

    def _cv(self, groups):
        from src.models.meta_learners import group_folds
        return group_folds(groups, self.n_splits, self.seed)

    def _est(self):  # pragma: no cover
        raise NotImplementedError

    def fit(self, X, T, Y, groups):
        warnings.filterwarnings("ignore")
        self.est = self._est(groups)
        self.est.fit(Y, T.astype(int), X=self._X(X), W=self._W(X), groups=groups)
        return self

    def _X(self, X):
        return X

    def _W(self, X):
        return None

    def predict(self, X):
        return np.asarray(self.est.effect(self._X(X))).ravel()

    def interval(self, X, alpha: float = 0.05):
        lo, hi = self.est.effect_interval(self._X(X), alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class LinearDMLModel(_Wrap):
    name = "LinearDML"

    def _idx(self):
        d = [CATE_FEATURES.index(c) for c in LINEAR_DRIVERS]
        w = [i for i in range(len(CATE_FEATURES)) if i not in d]
        return d, w

    def _X(self, X):
        d, _ = self._idx()
        x = X[:, d].copy()
        x[:, -1] = (x[:, -1] - self.mu_) / self.sd_      # pre_ncm_mean: standardise
        return x

    def _W(self, X):
        return X[:, self._idx()[1]]

    def _est(self, groups):
        from econml.dml import LinearDML
        return LinearDML(model_y=make_regressor("xgb", shallow=True, seed=self.seed),
                         model_t=make_classifier("xgb", self.seed), discrete_treatment=True,
                         cv=self._cv(groups), random_state=self.seed)

    def fit(self, X, T, Y, groups):
        d, _ = self._idx()
        self.mu_, self.sd_ = X[:, d[-1]].mean(), X[:, d[-1]].std() + 1e-9
        return super().fit(X, T, Y, groups)


class NonParamDMLModel(_Wrap):
    name = "NonParamDML"

    def _est(self, groups):
        from econml.dml import NonParamDML
        return NonParamDML(model_y=make_regressor("xgb", shallow=True, seed=self.seed),
                           model_t=make_classifier("xgb", self.seed),
                           model_final=make_regressor("xgb", shallow=True, seed=self.seed + 7),
                           discrete_treatment=True, cv=self._cv(groups), random_state=self.seed)

    def predict(self, X):
        return np.asarray(self.est.effect(X)).ravel()

    def interval(self, X, alpha: float = 0.05):
        raise NotImplementedError("NonParamDML has no closed-form intervals")


class CausalForestDMLModel(_Wrap):
    name = "CausalForestDML"

    def _est(self, groups):
        from econml.dml import CausalForestDML
        cf = load_config()["models"]["causal_forest"]
        return CausalForestDML(model_y=make_regressor("xgb", shallow=True, seed=self.seed),
                               model_t=make_classifier("xgb", self.seed), discrete_treatment=True,
                               n_estimators=cf["n_estimators"], min_samples_leaf=cf["min_samples_leaf"],
                               max_depth=cf["max_depth"], honest=True, cv=self._cv(groups), random_state=self.seed)

    def importances(self):
        return np.asarray(self.est.feature_importances_)
