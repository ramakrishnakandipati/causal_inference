"""Kernel-weighted CATE learners (12 estimators covering the 11 algorithm families).

Every learner follows ``fit(X, T, Y, w, Z=None)`` / ``predict(X)`` where
  X : guest covariates (effect modifiers)         T : actual Gold status (0/1)
  Y : slope-adjusted outcome (see CateData)       w : RDD kernel weights K((R-c)/h)
  Z : eligibility instrument 1(R >= c), used by the IV learner.
Predictions estimate tau(x) = E[Y(1) - Y(0) | X = x, R ~ c].

GPU use: XGBoost (cuda), CatBoost (task_type=GPU), PyTorch (cuda). LightGBM's pip wheel has no
GPU build, so it runs in CPU histogram mode. Everything falls back to CPU automatically.
"""
from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional, Type

import numpy as np

from src.utils import device
from src.utils.nuisance import (NUISANCE, catboost_reg, clip_prop, gbm_defaults, gbm_space, lgbm_reg,
                                oof_potential_outcomes, oof_propensity, oof_regression, xgb_clf, xgb_reg)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class BaseCATE:
    name = "base"
    needs_instrument = False
    tunable = True

    def __init__(self, params: Optional[dict] = None, use_gpu: bool = True, seed: int = 42):
        self.params = {**self.default_params(), **(params or {})}
        self.use_gpu, self.seed = use_gpu, seed
        self.history: Dict[str, list] = {}

    @classmethod
    def default_params(cls) -> dict:
        return gbm_defaults()

    @classmethod
    def search_space(cls, trial) -> dict:
        return gbm_space(trial)

    def fit(self, X, T, Y, w, Z=None):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def clone(self, params: Optional[dict] = None):
        return type(self)({**self.params, **(params or {})}, self.use_gpu, self.seed)


def _gbm_part(p: dict) -> dict:
    keys = gbm_defaults().keys()
    return {k: v for k, v in p.items() if k in keys}


# ============================================================================= meta-learners
class SLearner(BaseCATE):
    """Single model f(X, T, T*X). XGBoost (GPU) or LightGBM base learner."""
    name = "s_learner"

    @classmethod
    def default_params(cls):
        return {**gbm_defaults(), "base": "xgboost"}

    @classmethod
    def search_space(cls, trial):
        return {**gbm_space(trial), "base": trial.suggest_categorical("base", ["xgboost", "lightgbm"])}

    def _features(self, X, t):
        t = np.full(len(X), t, float) if np.isscalar(t) else t
        Xs = (X - self.mu_) / self.sd_
        return np.column_stack([X, t, Xs * t[:, None]])      # main effects + interaction terms

    def fit(self, X, T, Y, w, Z=None):
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        p = _gbm_part(self.params)
        self.model_ = (lgbm_reg(p, False, self.seed) if self.params["base"] == "lightgbm"
                       else xgb_reg(p, self.use_gpu, self.seed))
        self.model_.fit(self._features(X, T), Y, sample_weight=w)
        return self

    def predict(self, X):
        return self.model_.predict(self._features(X, 1.0)) - self.model_.predict(self._features(X, 0.0))


class TLearner(BaseCATE):
    """Separate outcome models for treated / control inside the window (XGBoost or CatBoost, GPU)."""
    name = "t_learner"

    @classmethod
    def default_params(cls):
        return {**gbm_defaults(), "base": "xgboost"}

    @classmethod
    def search_space(cls, trial):
        return {**gbm_space(trial), "base": trial.suggest_categorical("base", ["xgboost", "catboost"])}

    def _make(self):
        p = _gbm_part(self.params)
        return (catboost_reg(p, self.use_gpu, self.seed) if self.params["base"] == "catboost"
                else xgb_reg(p, self.use_gpu, self.seed))

    def fit(self, X, T, Y, w, Z=None):
        self.m0_, self.m1_ = self._make(), self._make()
        self.m0_.fit(X[T == 0], Y[T == 0], sample_weight=w[T == 0])
        self.m1_.fit(X[T == 1], Y[T == 1], sample_weight=w[T == 1])
        return self

    def predict(self, X):
        return self.m1_.predict(X) - self.m0_.predict(X)


class XLearner(BaseCATE):
    """Kunzel et al.: impute individual effects with the opposite arm's model, blend by propensity."""
    name = "x_learner"

    def fit(self, X, T, Y, w, Z=None):
        p = _gbm_part(self.params)
        m0 = xgb_reg(p, self.use_gpu, self.seed).fit(X[T == 0], Y[T == 0], sample_weight=w[T == 0])
        m1 = xgb_reg(p, self.use_gpu, self.seed).fit(X[T == 1], Y[T == 1], sample_weight=w[T == 1])
        d1 = Y[T == 1] - m0.predict(X[T == 1])           # imputed effects for treated
        d0 = m1.predict(X[T == 0]) - Y[T == 0]           # imputed effects for controls
        self.t1_ = xgb_reg(p, self.use_gpu, self.seed).fit(X[T == 1], d1, sample_weight=w[T == 1])
        self.t0_ = xgb_reg(p, self.use_gpu, self.seed).fit(X[T == 0], d0, sample_weight=w[T == 0])
        self.prop_ = xgb_clf(None, self.use_gpu, self.seed).fit(X, T, sample_weight=w)
        return self

    def predict(self, X):
        e = clip_prop(self.prop_.predict_proba(X)[:, 1])
        return e * self.t0_.predict(X) + (1 - e) * self.t1_.predict(X)     # propensity-weighted blend


class RLearner(BaseCATE):
    """Robinson / Nie-Wager: regress outcome residual on treatment residual, weighted by residual^2."""
    name = "r_learner"

    def fit(self, X, T, Y, w, Z=None):
        m = oof_regression(X, Y, w, 3, self.seed, self.use_gpu)
        e = oof_propensity(X, T, w, 3, self.seed, self.use_gpu)
        yr, tr = Y - m, T - e
        pseudo = yr / np.where(np.abs(tr) < 1e-2, 1e-2 * np.where(tr < 0, -1, 1), tr)
        wt = w * tr ** 2
        lim = np.quantile(np.abs(pseudo), 0.99)
        self.model_ = xgb_reg(_gbm_part(self.params), self.use_gpu, self.seed)
        self.model_.fit(X, np.clip(pseudo, -lim, lim), sample_weight=wt)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class DRLearner(BaseCATE):
    """AIPW pseudo-outcome regression with 3-fold cross-fitted nuisances."""
    name = "dr_learner"

    def fit(self, X, T, Y, w, Z=None):
        mu0, mu1 = oof_potential_outcomes(X, T, Y, w, 3, self.seed, self.use_gpu)
        e = oof_propensity(X, T, w, 3, self.seed, self.use_gpu)
        phi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
        lim = np.quantile(np.abs(phi), 0.995)
        self.model_ = xgb_reg(_gbm_part(self.params), self.use_gpu, self.seed)
        self.model_.fit(X, np.clip(phi, -lim, lim), sample_weight=w)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ============================================================================= econml learners
def _nuis_reg(use_gpu, seed):
    return xgb_reg(None, use_gpu, seed)


def _nuis_clf(use_gpu, seed):
    return xgb_clf(None, use_gpu, seed)


class _EconmlBase(BaseCATE):
    def predict(self, X):
        return np.asarray(self.est_.effect(X)).ravel()


class LinearDMLLearner(_EconmlBase):
    """econml LinearDML: GPU XGBoost nuisances, linear CATE in X."""
    name = "linear_dml"

    @classmethod
    def default_params(cls):
        return {"alpha_unused": 0.0}

    @classmethod
    def search_space(cls, trial):
        return {"degree": trial.suggest_int("degree", 1, 2)}

    def fit(self, X, T, Y, w, Z=None):
        from econml.dml import LinearDML
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        deg = int(self.params.get("degree", 1))
        self.feat_ = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False)).fit(X)
        self.est_ = LinearDML(model_y=_nuis_reg(self.use_gpu, self.seed), model_t=_nuis_clf(self.use_gpu, self.seed),
                              discrete_treatment=True, cv=3, random_state=self.seed)
        self.est_.fit(Y, T, X=self.feat_.transform(X), sample_weight=w)
        return self

    def predict(self, X):
        return np.asarray(self.est_.effect(self.feat_.transform(X))).ravel()


class NonParamDMLLearner(_EconmlBase):
    """econml NonParamDML: arbitrary (GPU XGBoost) final-stage CATE model."""
    name = "nonparam_dml"

    def fit(self, X, T, Y, w, Z=None):
        from econml.dml import NonParamDML
        self.est_ = NonParamDML(model_y=_nuis_reg(self.use_gpu, self.seed), model_t=_nuis_clf(self.use_gpu, self.seed),
                                model_final=xgb_reg(_gbm_part(self.params), self.use_gpu, self.seed),
                                discrete_treatment=True, cv=3, random_state=self.seed)
        self.est_.fit(Y, T, X=X, sample_weight=w)
        return self


class CausalForestDMLLearner(_EconmlBase):
    """econml CausalForestDML (honest forest on DML residuals), GPU XGBoost nuisances."""
    name = "causal_forest_dml"

    @classmethod
    def default_params(cls):
        return {"n_estimators": 300, "min_samples_leaf": 40, "max_depth": 10, "max_samples": 0.4}

    @classmethod
    def search_space(cls, trial):
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 400, step=100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 120, log=True),
                "max_depth": trial.suggest_int("max_depth", 5, 14),
                "max_samples": trial.suggest_float("max_samples", 0.2, 0.5)}

    def fit(self, X, T, Y, w, Z=None):
        from econml.dml import CausalForestDML
        p = self.params
        self.est_ = CausalForestDML(model_y=_nuis_reg(self.use_gpu, self.seed), model_t=_nuis_clf(self.use_gpu, self.seed),
                                    discrete_treatment=True, cv=3, n_estimators=int(p["n_estimators"]),
                                    min_samples_leaf=int(p["min_samples_leaf"]), max_depth=int(p["max_depth"]),
                                    max_samples=float(p["max_samples"]), random_state=self.seed, n_jobs=-1)
        self.est_.fit(Y, T, X=X, sample_weight=w)
        return self


class OrthogonalForestLearner(BaseCATE):
    """econml DMLOrthoForest (ORF), distilled into a fast surrogate.

    ORF prediction costs ~0.25 s per point, so it is fitted on a subsample, evaluated on a reference
    set, and that reference CATE is distilled into a small XGBoost model used for prediction.
    """
    name = "orthogonal_forest"
    tunable = False

    @classmethod
    def default_params(cls):
        return {"n_trees": 40, "min_leaf_size": 20, "max_depth": 8, "subsample_ratio": 0.3,
                "n_fit": 3000, "n_ref": 400}

    def fit(self, X, T, Y, w, Z=None):
        from econml.orf import DMLOrthoForest
        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        p, rng = self.params, np.random.default_rng(self.seed)
        # DMLOrthoForest has no sample_weight: importance-resample rows with probability proportional to w
        idx = rng.choice(len(Y), int(p["n_fit"]), replace=False, p=w / w.sum())
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu_) / self.sd_
        self.est_ = DMLOrthoForest(n_trees=int(p["n_trees"]), min_leaf_size=int(p["min_leaf_size"]),
                                   max_depth=int(p["max_depth"]), subsample_ratio=float(p["subsample_ratio"]),
                                   discrete_treatment=True, model_T=LogisticRegressionCV(max_iter=500),
                                   model_Y=LassoCV(max_iter=3000), random_state=self.seed)
        self.est_.fit(Y[idx], T[idx], X=Xs[idx])
        ref = rng.choice(len(Y), min(int(p["n_ref"]), len(Y)), replace=False)
        tau_ref = np.asarray(self.est_.effect(Xs[ref])).ravel()
        self.surrogate_ = xgb_reg({"n_estimators": 120, "max_depth": 3, "learning_rate": 0.08,
                                   "min_child_weight": 5}, self.use_gpu, self.seed).fit(X[ref], tau_ref)
        return self

    def predict(self, X):
        return self.surrogate_.predict(X)


class DMLIVLearner(BaseCATE):
    """econml DMLIV family (NonParamDMLIV: GPU XGBoost final stage) for the fuzzy-RDD setting:
    Z = 1(R >= c) instruments actual Gold attainment T, so the CATE is a complier effect."""
    name = "dmliv"
    needs_instrument = True

    def fit(self, X, T, Y, w, Z=None):
        from econml.iv.dml import NonParamDMLIV
        if Z is None:
            raise ValueError("DMLIV needs the eligibility instrument Z")
        self.est_ = NonParamDMLIV(model_y_xw=_nuis_reg(self.use_gpu, self.seed), model_t_xw=_nuis_clf(self.use_gpu, self.seed),
                          model_t_xwz=_nuis_clf(self.use_gpu, self.seed),
                          model_final=xgb_reg(_gbm_part(self.params), self.use_gpu, self.seed),
                          discrete_treatment=True, discrete_instrument=True, cv=3, random_state=self.seed)
        self.est_.fit(Y, T, Z=Z.astype(int), X=X, sample_weight=w)
        return self

    def predict(self, X):
        return np.asarray(self.est_.effect(X)).ravel()


# ============================================================================= PyTorch networks
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _TorchCATE(BaseCATE):
    """Shared training loop: weighted loss, validation early stopping, loss-curve history."""
    dragon = False

    @classmethod
    def default_params(cls):
        return {"hidden": 128, "depth": 2, "head": 64, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
                "epochs": 150, "patience": 15, "batch_size": 1024, "alpha": 1.0, "beta": 1.0}

    @classmethod
    def search_space(cls, trial):
        d = {"hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
             "depth": trial.suggest_int("depth", 1, 3),
             "head": trial.suggest_categorical("head", [32, 64, 128]),
             "dropout": trial.suggest_float("dropout", 0.0, 0.3),
             "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
             "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}
        if cls.dragon:
            d["alpha"] = trial.suggest_float("alpha", 0.1, 2.0)
            d["beta"] = trial.suggest_float("beta", 0.1, 2.0)
        return d

    def _build(self, d_in):
        torch, nn = _torch()
        p, dragon = self.params, self.dragon

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                layers, d = [], d_in
                for _ in range(int(p["depth"]) + (1 if dragon else 0)):
                    layers += [nn.Linear(d, int(p["hidden"])), nn.ELU(), nn.Dropout(p["dropout"])]
                    d = int(p["hidden"])
                s.rep = nn.Sequential(*layers)
                mk = lambda: nn.Sequential(nn.Linear(d, int(p["head"])), nn.ELU(), nn.Linear(int(p["head"]), 1))
                s.h0, s.h1 = mk(), mk()
                if dragon:
                    s.prop = nn.Linear(d, 1)
                    s.eps = nn.Parameter(torch.zeros(1))

            def forward(s, x):
                z = s.rep(x)
                logit = s.prop(z).squeeze(-1) if dragon else None
                return s.h0(z).squeeze(-1), s.h1(z).squeeze(-1), logit

        return Net()

    def fit(self, X, T, Y, w, Z=None):
        torch, nn = _torch()
        p = self.params
        dev = device.torch_device(self.use_gpu)
        torch.manual_seed(self.seed)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        self.ym_, self.ys_ = float(np.average(Y, weights=w)), float(Y.std() + 1e-9)
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32, device=dev)
        Yt = torch.tensor((Y - self.ym_) / self.ys_, dtype=torch.float32, device=dev)
        Tt = torch.tensor(T, dtype=torch.float32, device=dev)
        Wt = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(Y)); nv = max(int(0.15 * len(Y)), 50)
        va, tr = torch.tensor(perm[:nv], device=dev), torch.tensor(perm[nv:], device=dev)

        self.d_in_ = X.shape[1]
        self.net_ = self._build(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

        def loss_fn(idx):
            m0, m1, logit = self.net_(Xt[idx])
            t, y, ww = Tt[idx], Yt[idx], Wt[idx]
            mu_t = t * m1 + (1 - t) * m0
            out = (ww * (y - mu_t) ** 2).mean()
            if self.dragon:
                e = torch.sigmoid(logit).clamp(0.02, 0.98)
                bce = (ww * nn.functional.binary_cross_entropy(e, t, reduction="none")).mean()
                h = t / e - (1 - t) / (1 - e)
                targeted = (ww * (y - (mu_t + self.net_.eps * h)) ** 2).mean()
                return out + p["alpha"] * bce + p["beta"] * targeted
            return out

        best, best_state, bad = np.inf, None, 0
        self.history = {"train": [], "val": []}
        bs = int(p["batch_size"])
        for _ in range(int(p["epochs"])):
            self.net_.train()
            sh = tr[torch.randperm(len(tr), device=dev)]
            tot, nb = 0.0, 0
            for i in range(0, len(sh), bs):
                opt.zero_grad()
                loss = loss_fn(sh[i:i + bs])
                loss.backward(); opt.step()
                tot += float(loss); nb += 1
            self.net_.eval()
            with torch.no_grad():
                v = float(loss_fn(va))
            self.history["train"].append(tot / max(nb, 1)); self.history["val"].append(v)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= int(p["patience"]):
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def predict(self, X):
        torch, _ = _torch()
        dev = next(self.net_.parameters()).device
        with torch.no_grad():
            x = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32, device=dev)
            m0, m1, _ = self.net_(x)
        return ((m1 - m0).cpu().numpy()) * self.ys_

    def __getstate__(self):
        """Pickle weights only (the network class is built dynamically); portable across devices."""
        state = self.__dict__.copy()
        net = state.pop("net_", None)
        if net is not None:
            state["_net_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        return state

    def __setstate__(self, state):
        sd = state.pop("_net_state", None)
        self.__dict__.update(state)
        if sd is not None:
            self.net_ = self._build(self.d_in_)
            self.net_.load_state_dict(sd)
            self.net_.eval()                      # reloaded models predict on CPU


class TARNetLearner(_TorchCATE):
    """TARNet: shared representation + separate mu0 / mu1 heads, kernel-weighted loss (CUDA)."""
    name = "tarnet"
    dragon = False


class DragonNetLearner(_TorchCATE):
    """DragonNet: shared representation, mu0/mu1 heads, propensity head, targeted regularization (CUDA)."""
    name = "dragonnet"
    dragon = True


# ============================================================================= registry
LEARNERS: Dict[str, Type[BaseCATE]] = {c.name: c for c in (
    SLearner, TLearner, XLearner, RLearner, DRLearner, LinearDMLLearner, NonParamDMLLearner,
    CausalForestDMLLearner, OrthogonalForestLearner, DMLIVLearner, TARNetLearner, DragonNetLearner)}


def make_learner(name: str, params: Optional[dict] = None, use_gpu: bool = True, seed: int = 42) -> BaseCATE:
    return LEARNERS[name](params, use_gpu, seed)


# ============================================================================= Optuna tuning
class RLossScorer:
    """Ground-truth-free validation score for CATE models (Robinson R-loss, lower is better).

    Nuisances m(x)=E[Y|x] and e(x)=E[T|x] are fitted on the training part only, so the validation
    residuals are honest.
    """

    def __init__(self, X_tr, T_tr, Y_tr, w_tr, X_va, T_va, Y_va, w_va, use_gpu=True, seed=0):
        m = xgb_reg(None, use_gpu, seed).fit(X_tr, Y_tr, sample_weight=w_tr)
        e = xgb_clf(None, use_gpu, seed).fit(X_tr, T_tr, sample_weight=w_tr)
        self.y_res = Y_va - m.predict(X_va)
        self.t_res = T_va - clip_prop(e.predict_proba(X_va)[:, 1])
        self.w = w_va

    def __call__(self, tau_hat) -> float:
        from src.evaluation.causal_metrics import r_loss
        return r_loss(tau_hat, self.y_res, self.t_res, self.w)


def tune_learner(name: str, cd, n_trials: int = 8, timeout: Optional[float] = None, use_gpu: bool = True,
                 seed: int = 42, val_frac: float = 0.25, log=None) -> dict:
    """Optuna TPE search minimising validation R-loss. Returns best params + trial table."""
    import optuna
    cls = LEARNERS[name]
    if not cls.tunable or n_trials <= 0:
        return {"best_params": {}, "best_value": None, "trials": [], "tuned": False}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cd)); nv = int(val_frac * len(cd))
    va, tr = perm[:nv], perm[nv:]
    scorer = RLossScorer(cd.X[tr], cd.T[tr], cd.Y[tr], cd.w[tr], cd.X[va], cd.T[va], cd.Y[va], cd.w[va], use_gpu, seed)

    def objective(trial):
        params = cls.search_space(trial)
        try:
            L = cls(params, use_gpu, seed).fit(cd.X[tr], cd.T[tr], cd.Y[tr], cd.w[tr], cd.Z[tr])
            return scorer(L.predict(cd.X[va]))
        except Exception as exc:                       # a bad trial must not kill the study
            if log:
                log(f"  trial failed for {name}: {exc}")
            raise optuna.TrialPruned()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    # trial 0 = library defaults, so tuning can never be worse than the untuned model
    study.enqueue_trial({k: v for k, v in cls.default_params().items()
                         if k in _space_keys(cls)})
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    done = [t for t in study.trials if t.value is not None]
    if not done:
        return {"best_params": {}, "best_value": None, "trials": [], "tuned": False}
    return {"best_params": dict(study.best_params), "best_value": float(study.best_value), "tuned": True,
            "trials": [{"number": t.number, "value": t.value, **t.params} for t in done]}


def _space_keys(cls) -> set:
    import optuna

    class _Rec:
        keys: set = set()

        def _rec(self, name, *a, **k):
            self.keys.add(name)
            return a[0] if a else 0

        suggest_int = suggest_float = suggest_categorical = lambda self, name, *a, **k: (self.keys.add(name), a[0] if not isinstance(a[0], list) else a[0][0])[1]

    rec = _Rec(); rec.keys = set()
    try:
        cls.search_space(rec)
    except Exception:
        pass
    return rec.keys
