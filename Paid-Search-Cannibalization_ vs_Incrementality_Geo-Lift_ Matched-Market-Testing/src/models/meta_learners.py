"""S/T/X/R/DR meta-learners with GPU gradient-boosted base learners (XGBoost / CatBoost).

Design notes
* Treatment is assigned at DMA level and repeated over 28 days, so every cross-fit split is
  *grouped by DMA*; otherwise nuisance models memorise a DMA's treatment status from its rows.
* The holdout is a stratified 20/80 randomisation, so the propensity is known (0.2). Learners use
  the known propensity by default (``propensity="known"``) and can estimate it (``"estimated"``).
"""
from __future__ import annotations

import numpy as np

from src.utils.config import load_config
from src.utils.device import get_hardware

E_KNOWN = 0.20


# ---------------------------------------------------------------- base-learner factories
def make_regressor(kind: str = "xgb", shallow: bool = False, seed: int = 0):
    hw, nu = get_hardware(), load_config()["models"]["nuisance"]
    n_est = 100 if shallow else nu["n_estimators"]
    depth = 2 if shallow else nu["max_depth"]
    mcw = 28 if shallow else 5   # a leaf must hold ~a DMA's worth of rows: rows of one DMA are not independent
    if kind == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=nu["learning_rate"],
                            subsample=nu["subsample"], colsample_bytree=nu["colsample_bytree"],
                            min_child_weight=mcw, random_state=seed, verbosity=0, **hw.xgb_params)
    if kind == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=n_est, depth=depth, learning_rate=0.08, random_seed=seed,
                                 verbose=0, allow_writing_files=False, **hw.catboost_params)
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=n_est, max_depth=depth, learning_rate=nu["learning_rate"],
                             subsample=nu["subsample"], subsample_freq=1, colsample_bytree=nu["colsample_bytree"],
                             random_state=seed, verbose=-1, **hw.lgbm_params)
    raise ValueError(kind)


def make_classifier(kind: str = "xgb", seed: int = 0):
    hw, nu = get_hardware(), load_config()["models"]["nuisance"]
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=150, depth=3, learning_rate=0.08, random_seed=seed, verbose=0,
                                  allow_writing_files=False, **hw.catboost_params)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=150, max_depth=3, learning_rate=nu["learning_rate"], random_state=seed,
                         verbosity=0, **hw.xgb_params)


def group_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Folds that keep every DMA's rows together. Returns list of (train_idx, test_idx)."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    parts = np.array_split(ug, min(n_splits, len(ug)))
    out = []
    for p in parts:
        te = np.isin(groups, p)
        out.append((np.where(~te)[0], np.where(te)[0]))
    return out


def _fit(model, X, y, w=None):
    if w is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def _predict(model, X):
    return np.asarray(model.predict(X), dtype=float)


def _cf_mu(X, T, Y, folds, t_val, kind, seed):
    """Cross-fitted E[Y | X, T=t_val] evaluated for every row."""
    out = np.zeros(len(Y))
    for tr, te in folds:
        m = tr[T[tr] == t_val]
        out[te] = _predict(_fit(make_regressor(kind, seed=seed), X[m], Y[m]), X[te])
    return out


def _cf_mean(X, Y, folds, kind, seed):
    out = np.zeros(len(Y))
    for tr, te in folds:
        out[te] = _predict(_fit(make_regressor(kind, seed=seed), X[tr], Y[tr]), X[te])
    return out


def _cf_prop(X, T, folds, kind, seed):
    out = np.zeros(len(T))
    for tr, te in folds:
        clf = make_classifier(kind, seed).fit(X[tr], T[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return np.clip(out, 0.05, 0.95)


class BaseCATE:
    name = "base"

    def __init__(self, base: str = "xgb", propensity: str = "known", seed: int = 0, n_splits: int = 5):
        self.base, self.propensity, self.seed, self.n_splits = base, propensity, seed, n_splits

    def _e(self, X, T, folds):
        return np.full(len(T), E_KNOWN) if self.propensity == "known" else _cf_prop(X, T, folds, self.base, self.seed)

    def fit(self, X, T, Y, groups):  # pragma: no cover
        raise NotImplementedError

    def predict(self, X):  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------- S-learner
class SLearner(BaseCATE):
    name = "S-Learner"

    def fit(self, X, T, Y, groups):
        self.m = _fit(make_regressor(self.base, seed=self.seed), np.column_stack([X, T]), Y)
        return self

    def predict(self, X):
        n = len(X)
        return _predict(self.m, np.column_stack([X, np.ones(n)])) - _predict(self.m, np.column_stack([X, np.zeros(n)]))


# ---------------------------------------------------------------- T-learner
class TLearner(BaseCATE):
    name = "T-Learner"

    def fit(self, X, T, Y, groups):
        self.m0 = _fit(make_regressor(self.base, seed=self.seed), X[T == 0], Y[T == 0])
        self.m1 = _fit(make_regressor(self.base, seed=self.seed + 1), X[T == 1], Y[T == 1])
        return self

    def predict(self, X):
        return _predict(self.m1, X) - _predict(self.m0, X)


# ---------------------------------------------------------------- X-learner
class XLearner(BaseCATE):
    name = "X-Learner"

    def fit(self, X, T, Y, groups):
        folds = group_folds(groups, self.n_splits, self.seed)
        m0 = _fit(make_regressor(self.base, seed=self.seed), X[T == 0], Y[T == 0])
        m1 = _fit(make_regressor(self.base, seed=self.seed + 1), X[T == 1], Y[T == 1])
        d1 = Y[T == 1] - _predict(m0, X[T == 1])          # imputed effects for treated
        d0 = _predict(m1, X[T == 0]) - Y[T == 0]          # imputed effects for controls
        self.t1 = _fit(make_regressor(self.base, shallow=True, seed=self.seed + 2), X[T == 1], d1)
        self.t0 = _fit(make_regressor(self.base, shallow=True, seed=self.seed + 3), X[T == 0], d0)
        if self.propensity == "known":
            self._clf = None
        else:
            self._clf = make_classifier(self.base, self.seed).fit(X, T)
        return self

    def predict(self, X):
        e = np.full(len(X), E_KNOWN) if self._clf is None else np.clip(self._clf.predict_proba(X)[:, 1], 0.05, 0.95)
        return e * _predict(self.t0, X) + (1 - e) * _predict(self.t1, X)


# ---------------------------------------------------------------- R-learner (Robinson)
class RLearner(BaseCATE):
    name = "R-Learner"

    def fit(self, X, T, Y, groups):
        folds = group_folds(groups, self.n_splits, self.seed)
        m = _cf_mean(X, Y, folds, self.base, self.seed)
        e = self._e(X, T, folds)
        Yr, Tr = Y - m, T - e
        Tr = np.where(np.abs(Tr) < 1e-3, 1e-3, Tr)
        self.final = _fit(make_regressor(self.base, shallow=True, seed=self.seed + 4), X, Yr / Tr, Tr ** 2)
        return self

    def predict(self, X):
        return _predict(self.final, X)


# ---------------------------------------------------------------- DR-learner (cross-fitted AIPW)
class DRLearner(BaseCATE):
    name = "DR-Learner"

    def fit(self, X, T, Y, groups):
        folds = group_folds(groups, self.n_splits, self.seed)
        mu0 = _cf_mu(X, T, Y, folds, 0, self.base, self.seed)
        mu1 = _cf_mu(X, T, Y, folds, 1, self.base, self.seed + 1)
        e = self._e(X, T, folds)
        phi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
        self.pseudo = phi
        self.final = _fit(make_regressor(self.base, shallow=True, seed=self.seed + 5), X, phi)
        return self

    def predict(self, X):
        return _predict(self.final, X)


def default_meta_learners(seed: int = 0) -> dict[str, BaseCATE]:
    """S/T/R/DR on GPU XGBoost, X on GPU CatBoost."""
    return {"S-Learner": SLearner("xgb", seed=seed), "T-Learner": TLearner("xgb", seed=seed),
            "X-Learner": XLearner("catboost", seed=seed), "R-Learner": RLearner("xgb", seed=seed),
            "DR-Learner": DRLearner("xgb", seed=seed)}
