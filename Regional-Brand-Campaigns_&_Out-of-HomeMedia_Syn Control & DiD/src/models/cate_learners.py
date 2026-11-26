"""Ten heterogeneous Geo-CATE learners with GPU backends and automatic CPU fallback.

Learning problem
----------------
One market (MIA) is exposed, so identification cannot come from covariates alone. Each DMA-day outcome is
therefore first *residualised against a leave-self-out *augmented* synthetic control* (fitted on the pre-period only) and
expressed as a fraction of that market's pre-period mean:

        Y~_it = (y_it - yhat0_it) / mean_pre(y_i)          D_it = treatment_active

For donors Y~ is a placebo residual (true effect 0); for MIA campaign days it contains the campaign effect.
The learners then estimate tau(x) = E[Y~ | D=1, x] - E[Y~ | D=0, x] (a lift in units of baseline demand). This
removes the unit / time structure that a pooled outcome model could not learn and avoids inheriting the
parallel-trends bias of a TWFE residual.

Learners:  S, T, X, R, DR, CausalForestDML, LinearDML, DomainAdaptation, TARNet, CFRNet+DragonNet.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models._solvers import simplex_ls
from src.utils.device import device_info, make_classifier, make_regressor, torch_device
from src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LEARNER_ORDER = ["S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner", "Causal Forest DML",
                 "Linear DML", "Domain Adaptation", "TARNet", "CFRNet-DragonNet"]


# ============================================================================ dataset
@dataclass
class CATEData:
    X: np.ndarray               # (n,p) float32 features
    D: np.ndarray               # (n,) 0/1
    Y: np.ndarray               # (n,) residualised relative outcome
    folds: np.ndarray           # (n,) cross-fitting fold ids
    feature_names: list[str]
    tau_true: np.ndarray        # (n,) true effect in room nights (0 where untreated)
    base: float                 # treated market pre-period mean (room nights): tau_rn = tau_rel * base
    eval_mask: np.ndarray       # (n,) MIA campaign rows
    dma: np.ndarray             # (n,) dma codes
    day: np.ndarray             # (n,)
    y0_true: np.ndarray         # (n,) true counterfactual room nights

    @property
    def tau_true_rel(self) -> np.ndarray:
        return self.tau_true / self.base


def feature_frame(df: pd.DataFrame, feats: list[str], start_day: int) -> pd.DataFrame:
    """Model features: the 10 covariates plus two calendar features (days since launch, day of week)."""
    out = df[feats].copy()
    out["days_since_launch"] = np.maximum(df["day_index"].to_numpy() - start_day + 1, 0)
    out["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    return out


def build_cate_dataset(df: pd.DataFrame, cfg: dict, treated: str = "MIA") -> CATEData:
    """Residualise every DMA-day with a leave-self-out ASCM and assemble the learning set."""
    feats = cfg["cate"]["features"]
    K = cfg["cate"]["crossfit_folds"]
    d = df.copy()
    d["dma_code"] = d["dma_code"].astype(str)
    start_day = int(d.loc[d["treatment_active"] == 1, "day_index"].min())
    T0 = start_day - 1
    from src.models.panel_att import PanelData, estimator_ascm

    codes = sorted(d["dma_code"].unique())
    resid = np.zeros((len(codes), d["day_index"].nunique()))
    scale = np.zeros(len(codes))
    full = PanelData.from_df(d, treated, outcome="room_nights_booked", start_day=start_day)
    for i, c in enumerate(codes):
        pdi = full if c == treated else full.as_treated(c)      # donors: pool excludes the exposed market
        r = estimator_ascm(pdi, cfg, inference=False)           # ridge-augmented SCM removes extrapolation error
        resid[i] = pdi.y_tr - r.y0_hat
        scale[i] = pdi.y_tr[:T0].mean()
    Y = pd.DataFrame(index=codes, columns=sorted(d["day_index"].unique()), dtype=float)
    rel = pd.DataFrame(resid / scale[:, None], index=codes, columns=Y.columns)
    d = d.sort_values(["dma_code", "day_index"]).reset_index(drop=True)
    d["y_tilde"] = [rel.at[c, t] for c, t in zip(d["dma_code"], d["day_index"])]

    ff = feature_frame(d, feats, start_day)
    Xm = ff.to_numpy(np.float32)
    D = d["treatment_active"].to_numpy(int)
    # folds: donors blocked by DMA (balanced), the treated market by day blocks so every fold sees treated rows
    fold = np.zeros(len(d), dtype=int)
    donor_codes = [c for c in codes if c != treated]
    cmap = {c: i % K for i, c in enumerate(donor_codes)}
    is_t = (d["dma_code"] == treated).to_numpy()
    fold[~is_t] = d.loc[~is_t, "dma_code"].map(cmap).to_numpy()
    day = d["day_index"].to_numpy()
    post = day >= start_day
    fold[is_t & post] = ((day[is_t & post] - start_day) // 15) % K
    fold[is_t & ~post] = ((day[is_t & ~post] - 1) // 45) % K
    base = float(scale[codes.index(treated)])
    keep = np.ones(len(d), dtype=bool)
    if cfg["cate"].get("window_only", True):
        keep = post        # the campaign only exists in the window; pre-window rows have D=0 for every market (no overlap)
    sel = np.where(keep)[0]
    return CATEData(X=Xm[sel], D=D[sel], Y=d["y_tilde"].to_numpy(float)[sel], folds=fold[sel],
                    feature_names=list(ff.columns), tau_true=d["true_att_it"].to_numpy(float)[sel], base=base,
                    eval_mask=(D[sel] == 1), dma=d["dma_code"].to_numpy()[sel], day=day[sel],
                    y0_true=d["y0_true"].to_numpy(float)[sel])


def fold_splits(folds: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in np.unique(folds)]


# ============================================================================ base class
class CATEModel:
    name = "base"
    backend = "cpu"
    native_ci = False

    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.ne = cfg["cate"]["n_estimators"]
        self.clip = tuple(cfg["cate"]["propensity_clip"])
        self.feature_names: list[str] = []

    # factories ---------------------------------------------------------------
    def reg(self, kind: str, off: int = 0, small: bool = False):
        return make_regressor(kind, seed=self.seed + off, n_estimators=self.ne, small=small)

    def clf(self, kind: str, off: int = 0, small: bool = False):
        return make_classifier(kind, seed=self.seed + off, small=small)

    def backend_kind(self, key: str) -> str:
        return self.cfg["cate"]["backends"][key]

    @property
    def uses_gpu(self) -> bool:
        return device_info()["cuda"] and self.backend in ("xgboost", "catboost", "torch")

    # API ----------------------------------------------------------------------
    def fit(self, X: np.ndarray, D: np.ndarray, Y: np.ndarray, folds: np.ndarray | None = None) -> "CATEModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_interval(self, X: np.ndarray, alpha: float = 0.05):
        return None

    def to_cpu(self) -> "CATEModel":
        """Move every GPU object to CPU so the model can be pickled and served from any thread."""
        _force_cpu(self)
        return self

    def propensity(self, X, D, kind="xgboost", off=7):
        m = self.clf(kind, off)
        m.fit(X, D)
        return m

    def _p(self, m, X):
        return np.clip(m.predict_proba(X)[:, 1], *self.clip)

    def _crossfit(self, X, D, Y, folds, kind):
        """Out-of-fold m(x)=E[Y|x], e(x)=P(D=1|x), mu0(x), mu1(x)."""
        n = len(Y)
        m, e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for tr, te in fold_splits(folds):
            mm = self.reg(kind, 1); mm.fit(X[tr], Y[tr]); m[te] = mm.predict(X[te])
            ee = self.clf(kind, 2, small=True); ee.fit(X[tr], D[tr]); e[te] = self._p(ee, X[te])
            t1, t0 = tr[D[tr] == 1], tr[D[tr] == 0]
            f1 = self.reg(kind, 3, small=True); f1.fit(X[t1], Y[t1]); mu1[te] = f1.predict(X[te])
            f0 = self.reg(kind, 4); f0.fit(X[t0], Y[t0]); mu0[te] = f0.predict(X[te])
        return m, e, mu0, mu1


def _force_cpu(obj, seen: set | None = None, depth: int = 0) -> None:
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 6:
        return
    seen.add(id(obj))
    try:
        import torch

        if isinstance(obj, torch.nn.Module):
            obj.cpu()
            return
    except Exception:
        pass
    mod = type(obj).__module__ or ""
    if mod.startswith("xgboost"):
        try:
            obj.set_params(device="cpu")
            booster = obj.get_booster()
            booster.set_param({"device": "cpu"})
        except Exception:
            pass
        return
    if isinstance(obj, (list, tuple, set)):
        for o in obj:
            _force_cpu(o, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for o in obj.values():
            _force_cpu(o, seen, depth + 1)
        return
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, int, float, np.ndarray)):
        for o in vars(obj).values():
            _force_cpu(o, seen, depth + 1)


# ============================================================================ 1-5 meta-learners
class SLearner(CATEModel):
    name = "S-Learner"

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("s")
        self.backend = self.kind

    @staticmethod
    def _aug(X, d):
        d = np.full(len(X), d) if np.isscalar(d) else d
        return np.column_stack([X, d, X * d[:, None]])

    def fit(self, X, D, Y, folds=None):
        self.m = self.reg(self.kind, 0)
        self.m.fit(self._aug(X, D.astype(np.float32)), Y)
        return self

    def predict(self, X):
        return self.m.predict(self._aug(X, 1.0)) - self.m.predict(self._aug(X, 0.0))


class TLearner(CATEModel):
    name = "T-Learner"

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("t")
        self.backend = self.kind

    def fit(self, X, D, Y, folds=None):
        self.m0 = self.reg(self.kind, 0).fit(X[D == 0], Y[D == 0])
        self.m1 = self.reg(self.kind, 1, small=True).fit(X[D == 1], Y[D == 1])
        return self

    def predict(self, X):
        return self.m1.predict(X) - self.m0.predict(X)


class XLearner(CATEModel):
    name = "X-Learner"

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("x")
        self.backend = self.kind

    def fit(self, X, D, Y, folds=None):
        t, c = D == 1, D == 0
        mu0 = self.reg(self.kind, 0).fit(X[c], Y[c])
        mu1 = self.reg(self.kind, 1, small=True).fit(X[t], Y[t])
        d1 = Y[t] - mu0.predict(X[t])                 # imputed effects on treated
        d0 = mu1.predict(X[c]) - Y[c]                 # imputed effects on controls
        self.tau1 = self.reg(self.kind, 2, small=True).fit(X[t], d1)
        self.tau0 = self.reg(self.kind, 3, small=True).fit(X[c], d0)
        self.ps = self.propensity(X, D, self.kind)
        return self

    def predict(self, X):
        g = self._p(self.ps, X)
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


class RLearner(CATEModel):
    name = "R-Learner"

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("r")
        self.backend = self.kind

    def fit(self, X, D, Y, folds=None):
        n = len(Y)
        folds = folds if folds is not None else np.arange(n) % self.cfg["cate"]["crossfit_folds"]
        m = np.zeros(n); e = np.zeros(n)
        for tr, te in fold_splits(folds):                     # Robinson residuals, cross-fitted
            mm = self.reg(self.kind, 1).fit(X[tr], Y[tr]); m[te] = mm.predict(X[te])
            ee = self.clf(self.kind, 2, small=True).fit(X[tr], D[tr]); e[te] = self._p(ee, X[te])
        resid_t = D - e
        pseudo = (Y - m) / resid_t
        lo, hi = np.percentile(pseudo[D == 1], [2, 98])
        pseudo = np.clip(pseudo, lo, hi)
        self.final = self.reg(self.kind, 5, small=True).fit(X, pseudo, sample_weight=resid_t ** 2)
        return self

    def predict(self, X):
        return self.final.predict(X)


class DRLearner(CATEModel):
    name = "DR-Learner"

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("dr")
        self.backend = self.kind

    def fit(self, X, D, Y, folds=None):
        n = len(Y)
        folds = folds if folds is not None else np.arange(n) % self.cfg["cate"]["crossfit_folds"]
        _, e, mu0, mu1 = self._crossfit(X, D, Y, folds, self.kind)
        trim = self.cfg["cate"].get("dr_trim", 0.10)          # overlap trimming (Crump et al.) for the AIPW weights
        e = np.clip(e, trim, 1 - trim)
        psi = mu1 - mu0 + D * (Y - mu1) / e - (1 - D) * (Y - mu0) / (1 - e)      # AIPW pseudo-outcome
        lo, hi = np.percentile(psi, [1, 99])
        psi = np.clip(psi, lo, hi)
        self.final = self.reg(self.kind, 6, small=True).fit(X, psi)
        return self

    def predict(self, X):
        return self.final.predict(X)


# ============================================================================ 6-8 econml
class CausalForestDMLModel(CATEModel):
    name = "Causal Forest DML"
    native_ci = True

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("cf_dml_nuisance")
        self.backend = self.kind

    def fit(self, X, D, Y, folds=None):
        from econml.dml import CausalForestDML

        cv = fold_splits(folds) if folds is not None else 3
        self.est = CausalForestDML(model_y=self.reg(self.kind, 1), model_t=self.clf(self.kind, 2, small=True),
                                   discrete_treatment=True, honest=True, n_estimators=600, min_samples_leaf=5,
                                   max_depth=8, max_samples=0.45, cv=cv, random_state=self.seed, n_jobs=4)
        self.est.fit(Y, D, X=X)
        return self

    def predict(self, X):
        return np.asarray(self.est.effect(X)).ravel()

    def predict_interval(self, X, alpha=0.05):
        lo, hi = self.est.effect_interval(X, alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class LinearDMLModel(CATEModel):
    """Linear CATE on the named effect modifiers; every other feature is a nuisance control (W).

    Fitting the linear final stage on all 12 features is unstable here: seasonality_sin/cos and days_since_launch
    are near-collinear smooth functions of time inside a 60-day window. The coefficients therefore describe the
    drivers a media planner can reason about (weekend, holiday, weather, competitor ADR, flight search, OTA).
    """

    name = "Linear DML"
    native_ci = True

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("linear_dml_nuisance")
        self.backend = self.kind

    def _split(self, X):
        return X[:, self.drv], X[:, self.ctl]

    def fit(self, X, D, Y, folds=None):
        from econml.dml import LinearDML

        names = self.feature_names or self.cfg["cate"]["features"]
        want = self.cfg["cate"]["linear_dml_drivers"]
        self.drv = [names.index(f) for f in want if f in names]
        self.ctl = [i for i in range(X.shape[1]) if i not in self.drv]
        self.p_full = X.shape[1]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = (X - self.mu) / self.sd
        Xd, Wc = self._split(Xs)
        cv = fold_splits(folds) if folds is not None else 3
        self.est = LinearDML(model_y=self.reg(self.kind, 1), model_t=self.clf(self.kind, 2, small=True),
                             discrete_treatment=True, cv=cv, random_state=self.seed)
        self.est.fit(Y, D, X=Xd, W=Wc)
        # exact linear coefficients (per +1 sd of each driver), recovered from the fitted CATE surface
        q = len(self.drv)
        base = float(np.asarray(self.est.effect(np.zeros((1, q)))).ravel()[0])
        self.intercept_ = base
        coef = np.zeros(self.p_full)
        for j, col in enumerate(self.drv):
            coef[col] = float(np.asarray(self.est.effect(np.eye(q)[[j]])).ravel()[0]) - base
        self.coef_ = coef
        return self

    def predict(self, X):
        Xd, _ = self._split((X - self.mu) / self.sd)
        return np.asarray(self.est.effect(Xd)).ravel()

    def predict_interval(self, X, alpha=0.05):
        Xd, _ = self._split((X - self.mu) / self.sd)
        lo, hi = self.est.effect_interval(Xd, alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class DomainAdaptationModel(CATEModel):
    name = "Domain Adaptation"

    def __init__(self, cfg, seed=0):
        super().__init__(cfg, seed)
        self.kind = self.backend_kind("da")
        self.backend = self.kind

    def fit(self, X, D, Y, folds=None):
        from econml.metalearners import DomainAdaptationLearner

        self.est = DomainAdaptationLearner(models=self.reg(self.kind, 1), final_models=self.reg(self.kind, 2, small=True),
                                           propensity_model=self.clf(self.kind, 3, small=True))
        self.est.fit(Y, D, X=X)
        return self

    def predict(self, X):
        return np.asarray(self.est.effect(X)).ravel()


# ============================================================================ 9-10 PyTorch
class _Net(nn.Module):
    """Shared representation Phi(x) with two outcome heads (TARNet) and an optional propensity head (DragonNet).

    Defined at module level so fitted models can be pickled with joblib.
    """

    def __init__(self, p: int, hidden: int, layers: int, dragon: bool):
        super().__init__()
        mods, d = [], p
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ELU()]
            d = hidden
        self.phi = nn.Sequential(*mods)

        def head():
            return nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ELU(), nn.Linear(hidden // 2, hidden // 4),
                                 nn.ELU(), nn.Linear(hidden // 4, 1))

        self.h0, self.h1 = head(), head()
        self.prop = nn.Linear(hidden, 1) if dragon else None
        self.eps = nn.Parameter(torch.zeros(1)) if dragon else None

    def forward(self, x):
        z = self.phi(x)
        logit = self.prop(z).squeeze(-1) if self.prop is not None else None
        return self.h0(z).squeeze(-1), self.h1(z).squeeze(-1), logit, z


def _mmd2(a, b, sigma: float = 1.0):
    """Biased RBF-kernel MMD^2 between two representation batches."""
    def k(x, y):
        return torch.exp(-torch.cdist(x, y) ** 2 / (2 * sigma ** 2))

    return k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean()


class DeepCATE(CATEModel):
    """TARNet (dragon=False, cfr=False) or CFRNet+DragonNet (dragon=True, cfr=True)."""

    backend = "torch"

    def __init__(self, cfg, seed=0, dragon=False, cfr=False, name="TARNet"):
        super().__init__(cfg, seed)
        self.dragon, self.cfr, self.name = dragon, cfr, name
        self.history: dict[str, list] = {}
        self.best_epoch = 0

    def fit(self, X, D, Y, folds=None):
        import torch

        nn_cfg = self.cfg["cate"]["nn"]
        dev = torch_device()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.y_sd = float(Y.std() + 1e-9)
        Xs = ((X - self.mu) / self.sd).astype(np.float32)
        ys = (Y / self.y_sd).astype(np.float32)
        n = len(ys)
        # stratified 10% validation split (treated rows are scarce, keep both arms in every split)
        val = np.zeros(n, dtype=bool)
        for arm in (0, 1):
            idx = rng.permutation(np.where(D == arm)[0])
            val[idx[: max(3, int(0.1 * len(idx)))]] = True
        tr_idx, va_idx = np.where(~val)[0], np.where(val)[0]
        p_t = float(D[tr_idx].mean())
        net = _Net(Xs.shape[1], nn_cfg["hidden"], nn_cfg["rep_layers"], self.dragon).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=nn_cfg["lr"], weight_decay=nn_cfg["weight_decay"])
        use_amp = bool(nn_cfg.get("amp", False)) and dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        Xt = torch.tensor(Xs, device=dev)
        yt = torch.tensor(ys, device=dev)
        dt = torch.tensor(D.astype(np.float32), device=dev)
        w_all = dt / (2 * p_t) + (1 - dt) / (2 * (1 - p_t))                           # TARNet reweighting
        a_cfr, a_d, b_d = nn_cfg["cfr_alpha"], nn_cfg["dragon_alpha"], nn_cfg["dragon_beta"]
        tr_t, tr_c = tr_idx[D[tr_idx] == 1], tr_idx[D[tr_idx] == 0]
        n_batches = max(1, int(np.ceil(len(tr_idx) / nn_cfg["batch_size"])))
        keys = ["train_loss", "val_loss", "train_outcome", "val_outcome", "train_mmd", "train_prop", "train_target"]
        self.history = {k: [] for k in keys}

        def losses(idx, train: bool):
            idx_t = torch.as_tensor(idx, device=dev)
            x, y, t, w = Xt[idx_t], yt[idx_t], dt[idx_t], w_all[idx_t]
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                y0, y1, logit, z = net(x)
            y0, y1, z = y0.float(), y1.float(), z.float()
            q = torch.where(t > 0.5, y1, y0)
            l_y = (w * (q - y) ** 2).mean()
            tot, mmd, lp, lt = l_y, torch.zeros((), device=dev), torch.zeros((), device=dev), torch.zeros((), device=dev)
            if self.cfr and (t.sum() >= 2) and ((1 - t).sum() >= 2):
                mmd = _mmd2(z[t > 0.5], z[t < 0.5])
                tot = tot + a_cfr * mmd
            if self.dragon:
                logit = logit.float()
                lp = torch.nn.functional.binary_cross_entropy_with_logits(logit, t)
                pr = torch.sigmoid(logit).clamp(0.02, 0.98)
                h = t / pr - (1 - t) / (1 - pr)
                lt = ((y - (q + net.eps * h)) ** 2).mean()
                tot = tot + a_d * lp + b_d * lt
            return tot, l_y, mmd, lp, lt

        best, best_state, bad = np.inf, None, 0
        for ep in range(nn_cfg["epochs"]):
            net.train()
            ct, cc = rng.permutation(tr_t), rng.permutation(tr_c)
            parts_t, parts_c = np.array_split(ct, n_batches), np.array_split(cc, n_batches)
            acc = np.zeros(5)
            for bt, bc in zip(parts_t, parts_c):
                idx = np.concatenate([bt, bc])
                opt.zero_grad(set_to_none=True)
                tot, l_y, mmd, lp, lt = losses(idx, True)
                scaler.scale(tot).backward()
                scaler.step(opt)
                scaler.update()
                acc += np.array([tot.item(), l_y.item(), mmd.item(), lp.item(), lt.item()])
            acc /= n_batches
            net.eval()
            with torch.no_grad():
                vtot, vy, *_ = losses(va_idx, False)
            for k, v in zip(keys, [acc[0], vtot.item(), acc[1], vy.item(), acc[2], acc[3], acc[4]]):
                self.history[k].append(float(v))
            if vtot.item() < best - 1e-5:
                best, bad, self.best_epoch = vtot.item(), 0, ep
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= nn_cfg["patience"]:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def predict(self, X):
        import torch

        dev = next(self.net.parameters()).device
        x = torch.tensor(((X - self.mu) / self.sd).astype(np.float32), device=dev)
        with torch.no_grad():
            y0, y1, _, _ = self.net(x)
        return ((y1 - y0).float().cpu().numpy()) * self.y_sd


def make_learner(name: str, cfg: dict, seed: int = 0) -> CATEModel:
    if name == "S-Learner": return SLearner(cfg, seed)
    if name == "T-Learner": return TLearner(cfg, seed)
    if name == "X-Learner": return XLearner(cfg, seed)
    if name == "R-Learner": return RLearner(cfg, seed)
    if name == "DR-Learner": return DRLearner(cfg, seed)
    if name == "Causal Forest DML": return CausalForestDMLModel(cfg, seed)
    if name == "Linear DML": return LinearDMLModel(cfg, seed)
    if name == "Domain Adaptation": return DomainAdaptationModel(cfg, seed)
    if name == "TARNet": return DeepCATE(cfg, seed, dragon=False, cfr=False, name="TARNet")
    if name == "CFRNet-DragonNet": return DeepCATE(cfg, seed, dragon=True, cfr=True, name="CFRNet-DragonNet")
    raise KeyError(name)


def fit_all(data: CATEData, cfg: dict, seed: int = 0, names: list[str] | None = None,
            keep_models: bool = False) -> dict[str, dict]:
    """Fit every learner, time it, and return CATE predictions (relative lift) on all rows."""
    log = get_logger("cate")
    out: dict[str, dict] = {}
    for name in names or LEARNER_ORDER:
        t0 = time.time()
        try:
            m = make_learner(name, cfg, seed)
            m.feature_names = list(data.feature_names)
            m.fit(data.X, data.D, data.Y, data.folds)
            tau = m.predict(data.X)
            out[name] = {"tau_rel": tau, "seconds": time.time() - t0, "gpu": bool(m.uses_gpu), "backend": m.backend,
                         "error": None, "model": m if keep_models else None,
                         "history": getattr(m, "history", None), "best_epoch": getattr(m, "best_epoch", None)}
            log.info("  %-18s %6.1fs  backend=%-8s gpu=%s", name, out[name]["seconds"], m.backend, out[name]["gpu"])
        except Exception as exc:  # one failing learner must not abort the benchmark
            out[name] = {"tau_rel": None, "seconds": time.time() - t0, "gpu": False, "backend": "-",
                         "error": f"{type(exc).__name__}: {str(exc)[:160]}", "model": None}
            log.info("  %-18s FAILED: %s", name, out[name]["error"])
    return out
