"""GPU gradient-boosting factories, cross-fitting helpers and RDD window preparation."""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from src.data.synthetic_generator import COVARIATES
from src.models.rdd_estimators import kernel_weights
from src.utils import device

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

PROP_CLIP = (0.03, 0.97)


# ----------------------------------------------------------------------------- GBM factories
def gbm_defaults() -> Dict[str, float]:
    return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.8,
            "colsample_bytree": 0.9, "min_child_weight": 20.0, "reg_lambda": 5.0}


NUISANCE = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.8,
            "colsample_bytree": 0.9, "min_child_weight": 30.0, "reg_lambda": 5.0}


def gbm_space(trial, prefix: str = "") -> Dict[str, float]:
    return {
        "n_estimators": trial.suggest_int(prefix + "n_estimators", 60, 400, step=20),
        "max_depth": trial.suggest_int(prefix + "max_depth", 2, 6),
        "learning_rate": trial.suggest_float(prefix + "learning_rate", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float(prefix + "subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(prefix + "colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float(prefix + "min_child_weight", 5.0, 80.0, log=True),
        "reg_lambda": trial.suggest_float(prefix + "reg_lambda", 1.0, 30.0, log=True),
    }


def xgb_reg(p: Optional[dict] = None, use_gpu: bool = True, seed: int = 0):
    import xgboost as xgb
    q = {**NUISANCE, **(p or {})}
    q["n_estimators"] = int(q["n_estimators"]); q["max_depth"] = int(q["max_depth"])
    return xgb.XGBRegressor(random_state=seed, n_jobs=4, **device.xgb_params(use_gpu), **q)


def xgb_clf(p: Optional[dict] = None, use_gpu: bool = True, seed: int = 0):
    import xgboost as xgb
    q = {**NUISANCE, **(p or {})}
    q["n_estimators"] = int(q["n_estimators"]); q["max_depth"] = int(q["max_depth"])
    return xgb.XGBClassifier(random_state=seed, n_jobs=4, **device.xgb_params(use_gpu), **q)


def lgbm_reg(p: Optional[dict] = None, use_gpu: bool = False, seed: int = 0):
    import lightgbm as lgb
    q = {**gbm_defaults(), **(p or {})}
    return lgb.LGBMRegressor(
        n_estimators=int(q["n_estimators"]), max_depth=int(q["max_depth"]), learning_rate=q["learning_rate"],
        subsample=q["subsample"], subsample_freq=1, colsample_bytree=q["colsample_bytree"],
        min_child_weight=q["min_child_weight"] / 20.0, reg_lambda=q["reg_lambda"], random_state=seed,
        verbose=-1, n_jobs=4, **device.lgbm_params(use_gpu))


def catboost_reg(p: Optional[dict] = None, use_gpu: bool = True, seed: int = 0):
    from catboost import CatBoostRegressor
    q = {**gbm_defaults(), **(p or {})}
    return CatBoostRegressor(iterations=int(q["n_estimators"]), depth=int(min(q["max_depth"], 8)),
                             learning_rate=q["learning_rate"], l2_leaf_reg=q["reg_lambda"], random_seed=seed,
                             verbose=0, allow_writing_files=False, **device.catboost_params(use_gpu))


def clip_prop(e: np.ndarray) -> np.ndarray:
    return np.clip(e, *PROP_CLIP)


# ----------------------------------------------------------------------------- cross-fitting
def _folds(n: int, k: int, seed: int, strat: Optional[np.ndarray] = None):
    if strat is not None and len(np.unique(strat)) > 1:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), strat))
    return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n)))


def oof_regression(X, y, w, k: int = 3, seed: int = 0, use_gpu: bool = True, params=None) -> np.ndarray:
    out = np.zeros(len(y))
    for tr, te in _folds(len(y), k, seed):
        m = xgb_reg(params, use_gpu, seed).fit(X[tr], y[tr], sample_weight=w[tr])
        out[te] = m.predict(X[te])
    return out


def oof_propensity(X, T, w, k: int = 3, seed: int = 0, use_gpu: bool = True, params=None) -> np.ndarray:
    out = np.zeros(len(T))
    for tr, te in _folds(len(T), k, seed, T):
        m = xgb_clf(params, use_gpu, seed).fit(X[tr], T[tr], sample_weight=w[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return clip_prop(out)


def oof_potential_outcomes(X, T, Y, w, k: int = 3, seed: int = 0, use_gpu: bool = True,
                           params=None) -> Tuple[np.ndarray, np.ndarray]:
    mu0, mu1 = np.zeros(len(Y)), np.zeros(len(Y))
    for tr, te in _folds(len(Y), k, seed, T):
        for t, tgt in ((0, mu0), (1, mu1)):
            idx = tr[T[tr] == t]
            m = xgb_reg(params, use_gpu, seed).fit(X[idx], Y[idx], sample_weight=w[idx])
            tgt[te] = m.predict(X[te])
    return mu0, mu1


def dr_scores(X, T, Y, w, k: int = 3, seed: int = 0, use_gpu: bool = True) -> np.ndarray:
    """AIPW pseudo-outcomes with cross-fitted nuisances (used as an evaluation proxy)."""
    mu0, mu1 = oof_potential_outcomes(X, T, Y, w, k, seed, use_gpu)
    e = oof_propensity(X, T, w, k, seed, use_gpu)
    return mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)


# ----------------------------------------------------------------------------- RDD window prep
class CateData:
    """Local RDD sample for the CATE learners.

    * keeps guests with |R - c| <= h and kernel weights w = K((R-c)/h);
    * removes the side-specific running-variable trend from the outcome
      (Y_adj = Y - b_side * (R - c)); this keeps the level at the cutoff, so the
      treated/control gap at R=c is preserved while the confounding trend in R is removed;
    * effect modifiers X are the 10 guest covariates (R itself is not a modifier).
    """

    def __init__(self, df: pd.DataFrame, outcome: str, cutoff: float, h: float,
                 kernel: str = "triangular", running: str = "qualifying_nights_t"):
        r = df[running].to_numpy(float)
        w = kernel_weights(r, cutoff, h, kernel)
        sel = w > 0
        self.df = df.loc[sel].reset_index(drop=True)
        self.w = w[sel] / w[sel].mean()
        self.r = r[sel]
        self.cutoff, self.h, self.kernel, self.outcome = cutoff, h, kernel, outcome
        self.feature_names = list(COVARIATES)
        self.X = self.df[COVARIATES].to_numpy(float)
        self.T = self.df["gold_status"].to_numpy(float)
        self.Z = (self.r >= cutoff).astype(float)
        self.Y_raw = self.df[outcome].to_numpy(float)
        self.slopes = self._slopes()
        self.Y = self.Y_raw - np.where(self.Z == 1, self.slopes[1], self.slopes[0]) * (self.r - cutoff)

    def _slopes(self) -> Tuple[float, float]:
        out = []
        mu, sd = self.X.mean(0), self.X.std(0) + 1e-9
        for side in (0.0, 1.0):
            m = self.Z == side
            D = np.column_stack([np.ones(m.sum()), self.r[m] - self.cutoff, (self.X[m] - mu) / sd])
            W = self.w[m]
            beta = np.linalg.lstsq(D * np.sqrt(W)[:, None], self.Y_raw[m] * np.sqrt(W), rcond=None)[0]
            out.append(float(beta[1]))
        return out[0], out[1]

    @property
    def tau_true(self) -> Optional[np.ndarray]:
        from src.data.synthetic_generator import TAU_COLUMN
        col = TAU_COLUMN.get(self.outcome)
        return self.df[col].to_numpy(float) if col in self.df else None

    def __len__(self) -> int:
        return len(self.Y)
