"""CATE estimator suite (11 headline estimators) with GPU acceleration + graceful CPU fallback.

Common interface
    est = Estimator(cfg, params, seed)
    est.fit(X_df, T, Y, aux=None)          # X_df: DataFrame with FEATURES; aux: {"Y_conv": ...}
    est.fit_with_ci(X_df, T, Y, aux)       # fit + subsample replicas for SE-based 95% CI (if no native CI)
    est.predict(X_df)       -> tau_hat (n,)
    est.predict_interval(X_df, alpha=.05) -> (lo, hi) | None
"""
from __future__ import annotations

import copy
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.common import FEATURES, get_logger, load_config
from src.features import Preprocessor

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger()

Z95 = 1.96


# --------------------------------------------------------------------------------------------
# Device handling
# --------------------------------------------------------------------------------------------
def device_info(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    want = bool(cfg["gpu"]["enabled"])
    cuda = False
    name = "cpu"
    try:
        import torch
        cuda = want and torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover
        pass
    return {"cuda": cuda, "gpu_name": name, "torch_device": "cuda" if cuda else "cpu",
            "xgb_device": "cuda" if cuda else "cpu", "catboost_task": cfg["gpu"]["catboost_task_type"] if cuda else "CPU",
            "lightgbm": "cpu (pip wheel has no CUDA build)"}


# --------------------------------------------------------------------------------------------
# Base-learner factories
# --------------------------------------------------------------------------------------------
def _xgb_params(cfg, params=None, small=False):
    p = dict(cfg["training"]["xgb"])
    p.update(params or {})
    if small:
        p.update(n_estimators=min(p["n_estimators"], 150), max_depth=min(p["max_depth"], 4))
    return p


def make_regressor(cfg, params=None, kind: str | None = None, small=False, seed=None):
    kind = kind or cfg["training"]["base_learner"]
    p = _xgb_params(cfg, params, small)
    seed = cfg["seed"] if seed is None else seed
    dev = device_info(cfg)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**p, device=dev["xgb_device"], tree_method=cfg["gpu"]["xgb_tree_method"],
                            random_state=seed, n_jobs=4, verbosity=0)
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                             subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                             random_state=seed, verbose=-1, n_jobs=4)
    if kind == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=p["n_estimators"], depth=p["max_depth"], learning_rate=p["learning_rate"],
                                 task_type=dev["catboost_task"], random_seed=seed, verbose=0,
                                 allow_writing_files=False)
    raise ValueError(kind)


def make_classifier(cfg, params=None, kind: str | None = None, small=False, seed=None):
    kind = kind or cfg["training"]["base_learner"]
    p = _xgb_params(cfg, params, small)
    seed = cfg["seed"] if seed is None else seed
    dev = device_info(cfg)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**p, device=dev["xgb_device"], tree_method=cfg["gpu"]["xgb_tree_method"],
                             random_state=seed, n_jobs=4, verbosity=0)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                              subsample=p["subsample"], subsample_freq=1, colsample_bytree=p["colsample_bytree"],
                              random_state=seed, verbose=-1, n_jobs=4)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=p["n_estimators"], depth=p["max_depth"], learning_rate=p["learning_rate"],
                                  task_type=dev["catboost_task"], random_seed=seed, verbose=0, allow_writing_files=False)
    raise ValueError(kind)


def _arr(X) -> np.ndarray:
    return X[FEATURES].values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)


# --------------------------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------------------------
class CATEModel:
    name = "base"
    uses_gpu = False
    native_ci = False
    ci_B_key = "default"

    def __init__(self, cfg=None, params=None, seed=None, kind=None):
        self.cfg = cfg or load_config()
        self.params = params or {}
        self.seed = self.cfg["seed"] if seed is None else seed
        self.kind = kind or self.cfg["training"]["base_learner"]
        self.replicas: list = []

    # ---- helpers
    def reg(self, small=False, seed_off=0):
        return make_regressor(self.cfg, self.params, self.kind, small, self.seed + seed_off)

    def clf(self, small=False, seed_off=0):
        return make_classifier(self.cfg, self.params, self.kind, small, self.seed + seed_off)

    def fit(self, X, T, Y, aux=None):  # pragma: no cover
        raise NotImplementedError

    def predict(self, X):  # pragma: no cover
        raise NotImplementedError

    # ---- subsample-replica CI (SE via subsampling, m = n/2 -> Var_n ~ 0.5 * Var_m)
    def fit_with_ci(self, X, T, Y, aux=None, B: int = 0, frac: float = 0.5):
        self.fit(X, T, Y, aux)
        if self.native_ci or B <= 0:
            return self
        rng = np.random.default_rng(self.seed)
        n = len(Y)
        for b in range(B):
            idx = np.sort(rng.choice(n, int(frac * n), replace=False))
            rep = type(self)(self.cfg, self.params, self.seed + 1000 + b, self.kind)
            rep.fit(X.iloc[idx].reset_index(drop=True), np.asarray(T)[idx], np.asarray(Y)[idx],
                    None if aux is None else {k: np.asarray(v)[idx] for k, v in aux.items()})
            rep.replicas = []
            self.replicas.append(rep)
        return self

    def predict_interval(self, X, alpha: float = 0.05):
        if len(self.replicas) < 2:
            return None
        preds = np.stack([r.predict(X) for r in self.replicas])
        se = preds.std(axis=0, ddof=1) * np.sqrt(0.5)
        tau = self.predict(X)
        z = Z95 if alpha == 0.05 else float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - alpha / 2))
        return tau - z * se, tau + z * se

    def to_cpu(self):
        return self


# --------------------------------------------------------------------------------------------
# A. Meta-learners
# --------------------------------------------------------------------------------------------
class SLearner(CATEModel):
    name = "S-Learner"
    uses_gpu = True

    def fit(self, X, T, Y, aux=None):
        A = np.column_stack([_arr(X), np.asarray(T, np.float32)])
        self.m = self.reg().fit(A, np.asarray(Y))
        return self

    def predict(self, X):
        A = _arr(X)
        n = len(A)
        return self.m.predict(np.column_stack([A, np.ones(n, np.float32)])) - \
            self.m.predict(np.column_stack([A, np.zeros(n, np.float32)]))


class TLearner(CATEModel):
    name = "T-Learner"
    uses_gpu = True

    def fit(self, X, T, Y, aux=None):
        A, T, Y = _arr(X), np.asarray(T), np.asarray(Y)
        self.m0 = self.reg(seed_off=1).fit(A[T == 0], Y[T == 0])
        self.m1 = self.reg(seed_off=2).fit(A[T == 1], Y[T == 1])
        return self

    def predict(self, X):
        A = _arr(X)
        return self.m1.predict(A) - self.m0.predict(A)


class XLearner(CATEModel):
    """Stage 1: mu0, mu1.  Stage 2: regress imputed effects D1 = Y1 - mu0(X1), D0 = mu1(X0) - Y0.
    Final: tau = e(x) tau0(x) + (1 - e(x)) tau1(x). The imputations use the opposite arm's model, so they are
    out-of-arm (cross-fitted) by construction."""
    name = "X-Learner"
    uses_gpu = True

    def fit(self, X, T, Y, aux=None):
        A, T, Y = _arr(X), np.asarray(T), np.asarray(Y)
        A0, A1, Y0, Y1 = A[T == 0], A[T == 1], Y[T == 0], Y[T == 1]
        mu0 = self.reg(seed_off=1).fit(A0, Y0)
        mu1 = self.reg(seed_off=2).fit(A1, Y1)
        D1 = Y1 - mu0.predict(A1)
        D0 = mu1.predict(A0) - Y0
        self.tau1 = self.reg(seed_off=3).fit(A1, D1)
        self.tau0 = self.reg(seed_off=4).fit(A0, D0)
        self.e = self.clf(small=True, seed_off=5).fit(A, T)
        return self

    def predict(self, X):
        A = _arr(X)
        e = np.clip(self.e.predict_proba(A)[:, 1], 0.02, 0.98)
        return e * self.tau0.predict(A) + (1 - e) * self.tau1.predict(A)


class RLearner(CATEModel):
    """Robinson residual-on-residual: minimise sum_i ((Y-m)-(T-e) tau(X))^2 using out-of-fold m, e.
    Implemented as weighted regression: pseudo-outcome (Y-m)/(T-e) with weight (T-e)^2."""
    name = "R-Learner"
    uses_gpu = True

    def fit(self, X, T, Y, aux=None):
        from sklearn.model_selection import KFold
        A, T, Y = _arr(X), np.asarray(T), np.asarray(Y, float)
        K = self.cfg["training"]["crossfit_folds"]
        m_hat, e_hat = np.zeros(len(Y)), np.zeros(len(Y))
        for tr, te in KFold(K, shuffle=True, random_state=self.seed).split(A):
            m_hat[te] = self.reg(small=True, seed_off=1).fit(A[tr], Y[tr]).predict(A[te])
            e_hat[te] = self.clf(small=True, seed_off=2).fit(A[tr], T[tr]).predict_proba(A[te])[:, 1]
        e_hat = np.clip(e_hat, 0.02, 0.98)
        res_t, res_y = T - e_hat, Y - m_hat
        w = res_t ** 2
        pseudo = res_y / np.where(np.abs(res_t) < 1e-3, 1e-3, res_t)
        lo, hi = np.quantile(pseudo, [0.005, 0.995])
        self.final = self.reg(seed_off=3).fit(A, np.clip(pseudo, lo, hi), sample_weight=w)
        return self

    def predict(self, X):
        return self.final.predict(_arr(X))


# --------------------------------------------------------------------------------------------
# B. econml / causalml estimators
# --------------------------------------------------------------------------------------------
class DRLearnerModel(CATEModel):
    name = "DR-Learner"
    uses_gpu = True
    ci_B_key = "dr"

    def fit(self, X, T, Y, aux=None):
        from econml.dr import DRLearner
        self.m = DRLearner(model_propensity=self.clf(small=True, seed_off=1), model_regression=self.reg(small=True, seed_off=2),
                           model_final=self.reg(seed_off=3), cv=self.cfg["training"]["crossfit_folds"],
                           min_propensity=0.02, random_state=self.seed)
        self.m.fit(np.asarray(Y, float), np.asarray(T), X=_arr(X))
        return self

    def predict(self, X):
        return np.asarray(self.m.effect(_arr(X))).ravel()


class LinearDMLModel(CATEModel):
    name = "LinearDML"
    native_ci = True

    def fit(self, X, T, Y, aux=None):
        from econml.dml import LinearDML
        from sklearn.preprocessing import PolynomialFeatures
        self.pre = Preprocessor().fit(X)
        Xs = self.pre.transform(X)
        self.m = LinearDML(model_y=self.reg(small=True, seed_off=1), model_t=self.clf(small=True, seed_off=2),
                           discrete_treatment=True, featurizer=PolynomialFeatures(2, include_bias=False),
                           cv=self.cfg["training"]["crossfit_folds"], random_state=self.seed)
        self.m.fit(np.asarray(Y, float), np.asarray(T), X=Xs)
        return self

    def predict(self, X):
        return np.asarray(self.m.effect(self.pre.transform(X))).ravel()

    def predict_interval(self, X, alpha=0.05):
        lo, hi = self.m.effect_interval(self.pre.transform(X), alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class NonParamDMLModel(CATEModel):
    name = "NonParamDML"
    uses_gpu = True
    ci_B_key = "dr"

    def fit(self, X, T, Y, aux=None):
        from econml.dml import NonParamDML
        self.m = NonParamDML(model_y=self.reg(small=True, seed_off=1), model_t=self.clf(small=True, seed_off=2),
                             model_final=self.reg(seed_off=3), discrete_treatment=True,
                             cv=self.cfg["training"]["crossfit_folds"], random_state=self.seed)
        self.m.fit(np.asarray(Y, float), np.asarray(T), X=_arr(X))
        return self

    def predict(self, X):
        return np.asarray(self.m.effect(_arr(X))).ravel()


class CausalForestDMLModel(CATEModel):
    """Honest causal forest on residualised Y/T, native asymptotic CIs (tau +/- 1.96 SE)."""
    name = "CausalForestDML"
    uses_gpu = True
    native_ci = True

    def fit(self, X, T, Y, aux=None):
        from econml.dml import CausalForestDML
        f = self.cfg["training"]["forest"]
        self.m = CausalForestDML(model_y=self.reg(small=True, seed_off=1), model_t=self.clf(small=True, seed_off=2),
                                 discrete_treatment=True, n_estimators=f["n_estimators"],
                                 min_samples_leaf=f["min_samples_leaf"], max_depth=f["max_depth"],
                                 cv=3, honest=True, random_state=self.seed, n_jobs=-1)
        self.m.fit(np.asarray(Y, float), np.asarray(T), X=_arr(X))
        return self

    def predict(self, X):
        return np.asarray(self.m.effect(_arr(X))).ravel()

    def predict_interval(self, X, alpha=0.05):
        lo, hi = self.m.effect_interval(_arr(X), alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class UpliftRFModel(CATEModel):
    """causalml UpliftRandomForest (CTS / KL / ED divergence) on the binary conversion outcome.
    `predict_conv` is the native conversion uplift; `predict` converts it to margin scale with the exact identity
        tau_margin = tau_conv * R * (m - d) - p0 * R * d
    where R is the expected stay value and p0 a baseline-conversion plug-in (GBM on control guests)."""
    name = "UpliftRF"
    ci_B_key = "dr"

    def _stay_value(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
        return X["avg_adr"].values * (1 + (X["weekend_stay_ratio"].values > 0.5) + 0.8)

    def fit(self, X, T, Y, aux=None):
        from causalml.inference.tree import UpliftRandomForestClassifier
        yc = np.asarray(aux["Y_conv"] if aux is not None and "Y_conv" in aux else Y).astype(int)
        T = np.asarray(T)
        u = self.cfg["training"]["uplift_rf"]
        self.m = UpliftRandomForestClassifier(control_name="control", n_estimators=u["n_estimators"], max_depth=u["max_depth"],
                                              min_samples_leaf=u["min_samples_leaf"], evaluationFunction=u["evaluationFunction"],
                                              random_state=self.seed, n_jobs=-1)
        self.m.fit(_arr(X), np.where(T == 1, "treatment", "control"), yc)
        self.p0 = self.clf(small=True, seed_off=7).fit(_arr(X)[T == 0], yc[T == 0])
        return self

    def predict_conv(self, X):
        return np.asarray(self.m.predict(_arr(X))).reshape(len(X), -1)[:, 0]

    def predict(self, X):
        o = self.cfg["offer"]
        R = self._stay_value(X)
        p0 = self.p0.predict_proba(_arr(X))[:, 1]
        return self.predict_conv(X) * R * (o["margin_rate"] - o["discount_pct"]) - p0 * R * o["discount_pct"]


# --------------------------------------------------------------------------------------------
# C. Deep causal networks (PyTorch, CUDA + AMP)
# --------------------------------------------------------------------------------------------
import torch.nn as _nn  # noqa: E402


class _Net(_nn.Module):
    """Shared representation Phi(x) + outcome heads h0/h1 (+ propensity head & epsilon for DragonNet)."""

    def __init__(self, d, hidden, rep_layers, dragon):
        super().__init__()
        import torch
        layers, din = [], d
        for _ in range(rep_layers):
            layers += [_nn.Linear(din, hidden), _nn.ELU()]
            din = hidden

        def head():
            return _nn.Sequential(_nn.Linear(hidden, hidden // 2), _nn.ELU(), _nn.Linear(hidden // 2, hidden // 2),
                                  _nn.ELU(), _nn.Linear(hidden // 2, 1))
        self.rep = _nn.Sequential(*layers)
        self.h0, self.h1 = head(), head()
        self.dragon = dragon
        if dragon:
            self.prop = _nn.Linear(hidden, 1)
            self.eps = _nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = self.rep(x)
        logit = self.prop(z).squeeze(-1) if self.dragon else None
        return self.h0(z).squeeze(-1), self.h1(z).squeeze(-1), logit


def _build_net(d, hidden, rep_layers, dragon):
    return _Net(d, hidden, rep_layers, dragon)


class _DeepCATE(CATEModel):
    dragon = False
    uses_gpu = True
    ci_B_key = "nn"

    def fit(self, X, T, Y, aux=None):
        import torch
        import torch.nn.functional as F
        nn_cfg = self.cfg["training"]["nn"]
        dev = torch.device(device_info(self.cfg)["torch_device"])
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.pre = Preprocessor().fit(X)
        Xs = self.pre.transform(X)
        T = np.asarray(T, np.float32)
        Y = np.asarray(Y, np.float32)
        self.y_mu, self.y_sd = float(Y.mean()), float(Y.std() + 1e-6)
        Ys = (Y - self.y_mu) / self.y_sd
        perm = rng.permutation(len(Y))
        nv = int(0.1 * len(Y))
        va, tr = perm[:nv], perm[nv:]
        tens = lambda a, i: torch.tensor(a[i], device=dev)  # noqa: E731
        Xt, Tt, Yt = tens(Xs, tr), tens(T, tr), tens(Ys, tr)
        Xv, Tv, Yv = tens(Xs, va), tens(T, va), tens(Ys, va)
        self.net = _build_net(Xs.shape[1], nn_cfg["hidden"], nn_cfg["rep_layers"], self.dragon).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=nn_cfg["lr"], weight_decay=nn_cfg["weight_decay"])
        use_amp = bool(nn_cfg["amp"]) and dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        alpha, beta = self.cfg["training"]["dragonnet_alpha"], self.cfg["training"]["dragonnet_beta"]

        def loss_fn(x, t, y):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                y0, y1, logit = self.net(x)
            y0, y1 = y0.float(), y1.float()
            q = torch.where(t > 0.5, y1, y0)
            l_y = F.mse_loss(q, y)
            if not self.dragon:
                return l_y, l_y
            logit = logit.float()
            l_t = F.binary_cross_entropy_with_logits(logit, t)
            p = torch.sigmoid(logit).clamp(0.02, 0.98)
            h = t / p - (1 - t) / (1 - p)
            l_reg = F.mse_loss(q + self.net.eps * h, y)
            return l_y + alpha * l_t + beta * l_reg, l_y

        self.history = {"train": [], "val": [], "train_outcome": [], "val_outcome": []}
        best, best_state, bad = np.inf, None, 0
        bs = nn_cfg["batch_size"]
        for ep in range(nn_cfg["epochs"]):
            self.net.train()
            order = torch.randperm(len(Yt), device=dev)
            tot, tot_y, nb = 0.0, 0.0, 0
            for i in range(0, len(order), bs):
                b = order[i:i + bs]
                opt.zero_grad(set_to_none=True)
                l, ly = loss_fn(Xt[b], Tt[b], Yt[b])
                scaler.scale(l).backward()
                scaler.step(opt)
                scaler.update()
                tot += l.item(); tot_y += ly.item(); nb += 1
            self.net.eval()
            with torch.no_grad():
                lv, lvy = loss_fn(Xv, Tv, Yv)
            self.history["train"].append(tot / nb); self.history["train_outcome"].append(tot_y / nb)
            self.history["val"].append(lv.item()); self.history["val_outcome"].append(lvy.item())
            if lv.item() < best - 1e-5:
                best, bad = lv.item(), 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= nn_cfg["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.best_epoch = int(np.argmin(self.history["val"])) + 1
        return self

    def predict(self, X):
        import torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.pre.transform(X), device=dev)
            y0, y1, _ = self.net(x)
        return ((y1 - y0).float().cpu().numpy()) * self.y_sd

    def to_cpu(self):
        self.net.cpu()
        for r in self.replicas:
            r.to_cpu()
        return self


class TARNetModel(_DeepCATE):
    name = "TARNet"
    dragon = False


class DragonNetModel(_DeepCATE):
    name = "DragonNet"
    dragon = True


# --------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------
ESTIMATORS: "OrderedDict[str, type]" = OrderedDict(
    (c.name, c) for c in [SLearner, TLearner, XLearner, RLearner, DRLearnerModel, LinearDMLModel, NonParamDMLModel,
                          CausalForestDMLModel, UpliftRFModel, TARNetModel, DragonNetModel])


def timed_fit(est: CATEModel, X, T, Y, aux=None, B: int = 0):
    t0 = time.perf_counter()
    est.fit_with_ci(X, T, Y, aux, B=B)
    return time.perf_counter() - t0
