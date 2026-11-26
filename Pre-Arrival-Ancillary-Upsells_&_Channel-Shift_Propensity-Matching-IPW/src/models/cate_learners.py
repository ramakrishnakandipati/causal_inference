"""Unified CATE interface + S/T/X/R/DR learners, CausalForestDML and Uplift RF."""
from __future__ import annotations

import time
import warnings

import joblib
import numpy as np
import pandas as pd

from src.common.config import load_config
from src.common.hardware import catboost_params, lgb_params, xgb_params

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class BaseCATEEstimator:
    """fit(X, T, y) -> self ; predict_cate(X) -> tau_hat ; predict_interval(X) -> (lo, hi) or None."""
    name = "base"
    family = "meta"

    def __init__(self, binary: bool = False, fast: bool = False, seed: int = 42):
        self.binary, self.fast, self.seed = binary, fast, seed
        self.fit_seconds = np.nan
        self.latency_ms_per_1k = np.nan

    def _fit(self, X, T, y): raise NotImplementedError
    def _predict(self, X) -> np.ndarray: raise NotImplementedError

    def fit(self, X: pd.DataFrame, T: np.ndarray, y: np.ndarray):
        t0 = time.time()
        self._fit(X, np.asarray(T).astype(int), np.asarray(y, float))
        self.fit_seconds = time.time() - t0
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._predict(X), float).ravel()

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.05):
        return None

    def time_inference(self, X: pd.DataFrame, reps: int = 3):
        Xs = X.iloc[: min(len(X), 2000)]
        self.predict_cate(Xs)  # warm-up
        t0 = time.time()
        for _ in range(reps):
            self.predict_cate(Xs)
        self.latency_ms_per_1k = (time.time() - t0) / reps / len(Xs) * 1000 * 1000
        return self.latency_ms_per_1k

    def save(self, path):
        joblib.dump(self, path)


# ------------------------------------------------------------------ base learner factory
def xgb_cfg(fast):
    c = load_config()["learners"]
    return dict(c["xgb_fast"] if fast else c["xgb"])


def make_base(kind: str, binary: bool, fast: bool, seed: int = 42):
    if kind == "xgb":
        import xgboost as xgb
        p = {**xgb_cfg(fast), **xgb_params(), "random_state": seed}
        return xgb.XGBClassifier(**p) if binary else xgb.XGBRegressor(**p)
    if kind == "lgb":
        import lightgbm as lgb
        c = xgb_cfg(fast)
        p = dict(n_estimators=c["n_estimators"], max_depth=c["max_depth"] + 2, learning_rate=c["learning_rate"],
                 subsample=c["subsample"], subsample_freq=1, colsample_bytree=c["colsample_bytree"],
                 random_state=seed, **lgb_params())
        return lgb.LGBMClassifier(**p) if binary else lgb.LGBMRegressor(**p)
    if kind == "cat":
        import catboost as cb
        c = xgb_cfg(fast)
        p = dict(iterations=c["n_estimators"], depth=c["max_depth"] + 2, learning_rate=c["learning_rate"] * 1.5,
                 random_seed=seed, **catboost_params())
        return cb.CatBoostClassifier(**p) if binary else cb.CatBoostRegressor(**p)
    raise ValueError(kind)


def _pred(m, X, binary):
    return m.predict_proba(X)[:, 1] if binary else m.predict(X)


# ------------------------------------------------------------------ meta-learners
class SLearner(BaseCATEEstimator):
    def __init__(self, binary=False, fast=False, seed=42, base="xgb"):
        super().__init__(binary, fast, seed)
        self.base = base
        self.name = f"S-Learner ({ {'xgb': 'XGB', 'lgb': 'LightGBM', 'cat': 'CatBoost'}[base] })"

    def _fit(self, X, T, y):
        Xa = X.copy(); Xa["_T"] = T
        self.m = make_base(self.base, self.binary, self.fast, self.seed).fit(Xa, y)

    def _predict(self, X):
        X1 = X.copy(); X1["_T"] = 1
        X0 = X.copy(); X0["_T"] = 0
        return _pred(self.m, X1, self.binary) - _pred(self.m, X0, self.binary)


class TLearner(BaseCATEEstimator):
    def __init__(self, binary=False, fast=False, seed=42, base="xgb"):
        super().__init__(binary, fast, seed)
        self.base = base
        self.name = f"T-Learner ({ {'xgb': 'XGB', 'lgb': 'LightGBM', 'cat': 'CatBoost'}[base] })"

    def _fit(self, X, T, y):
        self.m1 = make_base(self.base, self.binary, self.fast, self.seed).fit(X[T == 1], y[T == 1])
        self.m0 = make_base(self.base, self.binary, self.fast, self.seed).fit(X[T == 0], y[T == 0])

    def _predict(self, X):
        return _pred(self.m1, X, self.binary) - _pred(self.m0, X, self.binary)


class XLearner(BaseCATEEstimator):
    name = "X-Learner"

    def _fit(self, X, T, y):
        import xgboost as xgb
        b = self.binary
        m1 = make_base("xgb", b, self.fast, self.seed).fit(X[T == 1], y[T == 1])
        m0 = make_base("xgb", b, self.fast, self.seed).fit(X[T == 0], y[T == 0])
        d1 = y[T == 1] - _pred(m0, X[T == 1], b)          # imputed effects, treated
        d0 = _pred(m1, X[T == 0], b) - y[T == 0]          # imputed effects, control
        p = {**xgb_cfg(self.fast), **xgb_params(), "random_state": self.seed}
        self.t1 = xgb.XGBRegressor(**p).fit(X[T == 1], d1)
        self.t0 = xgb.XGBRegressor(**p).fit(X[T == 0], d0)
        self.g = make_base("xgb", True, self.fast, self.seed).fit(X, T)
        self.clip = load_config()["propensity"]["clip"]

    def _predict(self, X):
        g = np.clip(self.g.predict_proba(X)[:, 1], *self.clip)
        return g * self.t0.predict(X) + (1 - g) * self.t1.predict(X)


# ------------------------------------------------------------------ EconML wrappers
def _econml_nuis(fast, seed, binary_y=False):
    return make_base("xgb", binary_y, fast, seed), make_base("xgb", True, fast, seed)


class RLearner(BaseCATEEstimator):
    name = "R-Learner (NonParamDML)"
    family = "econml"

    def _fit(self, X, T, y):
        from econml.dml import NonParamDML
        my, mt = _econml_nuis(self.fast, self.seed)       # outcome nuisance as regressor even for binary y
        self.m = NonParamDML(model_y=my, model_t=mt, model_final=make_base("xgb", False, self.fast, self.seed),
                             discrete_treatment=True, cv=load_config()["learners"]["folds"], random_state=self.seed)
        self.m.fit(y, T, X=X.values)

    def _predict(self, X):
        return self.m.effect(X.values)


class DRLearner(BaseCATEEstimator):
    name = "DR-Learner"
    family = "econml"

    def _fit(self, X, T, y):
        from econml.dr import DRLearner as EDR
        self.m = EDR(model_propensity=make_base("xgb", True, self.fast, self.seed),
                     model_regression=make_base("xgb", False, self.fast, self.seed),
                     model_final=make_base("xgb", False, self.fast, self.seed),
                     cv=load_config()["learners"]["folds"], random_state=self.seed)
        self.m.fit(y, T, X=X.values)

    def _predict(self, X):
        return self.m.effect(X.values)


class CausalForest(BaseCATEEstimator):
    name = "Causal Forest DML"
    family = "econml"

    def _fit(self, X, T, y):
        from econml.dml import CausalForestDML
        c = load_config()["learners"]
        n_est = c["forest_trees_fast"] if self.fast else c["forest_trees"]
        n_est = max(n_est // 4 * 4, 8)
        my, mt = _econml_nuis(self.fast, self.seed)
        self.m = CausalForestDML(model_y=my, model_t=mt, discrete_treatment=True, n_estimators=n_est,
                                 min_samples_leaf=20 if not self.fast else 10, max_depth=None,
                                 cv=c["folds"], random_state=self.seed, n_jobs=-1)
        self.m.fit(y, T, X=X.values)

    def _predict(self, X):
        return self.m.effect(X.values)

    def predict_interval(self, X, alpha=0.05):
        lo, hi = self.m.effect_interval(X.values, alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class UpliftRF(BaseCATEEstimator):
    """CausalML uplift random forest; binary outcome only (KL / Chi-square / ED criteria)."""
    name = "Uplift RF (CausalML)"
    family = "causalml"

    def __init__(self, binary=True, fast=False, seed=42, criterion="KL"):
        super().__init__(True, fast, seed)
        self.criterion = criterion
        self.name = f"Uplift RF ({criterion})"

    def _fit(self, X, T, y):
        from causalml.inference.tree import UpliftRandomForestClassifier
        c = load_config()["learners"]
        n = c["uplift_trees_fast"] if self.fast else c["uplift_trees"]
        self.m = UpliftRandomForestClassifier(n_estimators=n, max_depth=6, min_samples_leaf=100,
                                              min_samples_treatment=50, evaluationFunction=self.criterion,
                                              control_name="control", random_state=self.seed, n_jobs=-1)
        self.m.fit(X=X.values.astype(float), treatment=np.where(T == 1, "treatment", "control"), y=y.astype(int))

    def _predict(self, X):
        return self.m.predict(X.values.astype(float)).ravel()


# ------------------------------------------------------------------ registry
def make_learners(binary: bool, fast: bool = False, seed: int = 42, only: list[str] | None = None) -> dict:
    from src.models.deep_causal import DragonNet, TARNet
    L = [SLearner(binary, fast, seed, "xgb"), SLearner(binary, fast, seed, "cat"),
         TLearner(binary, fast, seed, "xgb"), TLearner(binary, fast, seed, "lgb"),
         XLearner(binary, fast, seed), RLearner(binary, fast, seed), DRLearner(binary, fast, seed),
         CausalForest(binary, fast, seed), TARNet(binary, fast, seed), DragonNet(binary, fast, seed)]
    if binary:
        L.append(UpliftRF(True, fast, seed, "KL"))
    out = {m.name: m for m in L}
    if only:
        out = {k: v for k, v in out.items() if any(o.lower() in k.lower() for o in only)}
    return out
