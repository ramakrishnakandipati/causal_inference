"""Aggregate and dynamic panel ATT estimators for ONE exposed market vs a donor pool.

  1. TWFE DiD (DMA + date fixed effects, covariates, DMA-clustered SE)
  2. Event study / dynamic DiD (leads and lags, reference k = -1)
  3. Callaway & Sant'Anna doubly-robust DiD (single cohort here -> see README caveat)
  4. Synthetic control (Abadie): W >= 0, sum(W) = 1
  5. Augmented SCM (Ben-Michael et al.): SCM + ridge / elastic-net bias correction
  6. Synthetic DiD (Arkhangelsky et al.)
  7. Bayesian synthetic control (NumPyro NUTS, AR(1) errors)

With a single treated unit, cluster-robust standard errors are unreliable, so every estimator also
carries *in-space placebo* inference: the same estimator is re-run with each donor as a pseudo-treated
unit and the spread of those fake effects calibrates the SE, the bands and the exact p-value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.data.generate_panel import OUTCOMES, wide
from src.models._solvers import rmspe, simplex_ls
from src.utils.logger import get_logger

TV_COVS = ["competitor_adr_usd", "flight_search_index", "temp_max_f", "precipitation_in",
           "weather_severity_index", "ota_promo_pressure_index"]
Z95 = 1.959964
ESTIMATOR_ORDER = ["TWFE DiD", "Event Study", "CS-DiD (DR)", "SCM", "ASCM (Ridge)", "SDID", "Bayesian SC"]
TRUTH_COL = {"room_nights_booked": "true_att_it", "direct_web_sessions": "true_att_direct_web_sessions",
             "branded_search_index": "true_att_branded_search_index",
             "incremental_net_revenue_usd": "true_att_incremental_net_revenue_usd"}


# ----------------------------------------------------------------------------- data container
@dataclass
class PanelData:
    outcome: str
    treated: str
    donors: list[str]
    start_day: int
    days: np.ndarray
    y_tr: np.ndarray            # (T,)
    Yc: np.ndarray              # (n,T)
    cov_tr: np.ndarray          # (T,p)
    cov_c: np.ndarray           # (n,T,p)
    cov_names: list[str]
    spend: np.ndarray           # (T,)
    cal: np.ndarray = None      # (T,2) is_weekend, is_holiday_window (identical for every DMA)

    @property
    def T(self) -> int:
        return len(self.days)

    @property
    def T0(self) -> int:
        return self.start_day - 1

    @property
    def T1(self) -> int:
        return self.T - self.T0

    @property
    def n(self) -> int:
        return len(self.donors)

    @classmethod
    def from_df(cls, df: pd.DataFrame, treated: str, donors: list[str] | None = None,
                outcome: str = "room_nights_booked", start_day: int | None = None,
                covs: list[str] | None = None) -> "PanelData":
        covs = covs or TV_COVS
        all_codes = sorted(df["dma_code"].astype(str).unique())
        donors = [c for c in (donors or all_codes) if c != treated]
        order = [treated] + donors
        Y = wide(df, outcome, order).to_numpy(float)
        days = np.array(sorted(df["day_index"].unique()))
        cov = np.stack([wide(df, c, order).to_numpy(float) for c in covs], axis=2)
        if start_day is None:
            tr = df[(df["dma_code"].astype(str) == treated) & (df["treatment_active"] == 1)]
            start_day = int(tr["day_index"].min()) if len(tr) else int(df.loc[df["post_period"] == 1, "day_index"].min())
        one = df[df["dma_code"].astype(str) == treated].sort_values("day_index")
        sp = one["ooh_ctv_spend_usd"].to_numpy(float)
        cal = one[["is_weekend", "is_holiday_window"]].to_numpy(float)
        return cls(outcome, treated, donors, int(start_day), days, Y[0], Y[1:], cov[0], cov[1:], list(covs), sp, cal)

    def as_treated(self, code: str) -> "PanelData":
        """Placebo view: a donor plays the treated unit; the remaining donors form the pool."""
        j = self.donors.index(code)
        keep = [i for i in range(self.n) if i != j]
        return PanelData(self.outcome, code, [self.donors[i] for i in keep], self.start_day, self.days,
                         self.Yc[j], self.Yc[keep], self.cov_c[j], self.cov_c[keep], self.cov_names, np.zeros(self.T),
                         self.cal)

    def restrict_donors(self, donors: list[str]) -> "PanelData":
        keep = [self.donors.index(d) for d in donors if d in self.donors]
        return PanelData(self.outcome, self.treated, [self.donors[i] for i in keep], self.start_day, self.days,
                         self.y_tr, self.Yc[keep], self.cov_tr, self.cov_c[keep], self.cov_names, self.spend, self.cal)

    def calendar_adjusted(self) -> tuple["PanelData", np.ndarray]:
        """Remove unit-specific weekend / holiday effects (estimated on the pre-period only).

        Returns the adjusted panel and the treated unit's offset (to add back to counterfactual paths).
        """
        T0 = self.T0
        Y = np.vstack([self.y_tr, self.Yc])
        Zc = np.column_stack([np.ones(T0), self.cal[:T0]])
        coef, *_ = np.linalg.lstsq(Zc, Y[:, :T0].T, rcond=None)           # (3,N)
        off = (self.cal @ coef[1:]).T                                       # (N,T)
        adj = PanelData(self.outcome, self.treated, self.donors, self.start_day, self.days, self.y_tr - off[0],
                        self.Yc - off[1:], self.cov_tr, self.cov_c, self.cov_names, self.spend, self.cal)
        return adj, off[0]


@dataclass
class PanelResult:
    name: str
    outcome: str
    att_mean: float
    se: float | None
    ci_lo: float | None
    ci_hi: float | None
    att_daily: np.ndarray               # (T1,) post-period daily effect
    y0_hat: np.ndarray                  # (T,) counterfactual path
    y_obs: np.ndarray                   # (T,)
    days: np.ndarray                    # (T,)
    T0: int
    daily_lo: np.ndarray | None = None
    daily_hi: np.ndarray | None = None
    weights: dict | None = None
    pre_rmspe: float | None = None
    post_rmspe: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def att_cum(self) -> np.ndarray:
        return np.cumsum(self.att_daily)

    @property
    def att_total(self) -> float:
        return float(self.att_daily.sum())

    def counterfactual_frame(self) -> pd.DataFrame:
        T0 = self.T0
        att = np.full(len(self.days), np.nan)
        att[T0:] = self.att_daily
        lo = np.full(len(self.days), np.nan)
        hi = np.full(len(self.days), np.nan)
        if self.daily_lo is not None:
            lo[T0:], hi[T0:] = self.daily_lo, self.daily_hi
        return pd.DataFrame({"day_index": self.days, "y_obs": self.y_obs, "y0_hat": self.y0_hat,
                             "att_hat": att, "att_lo": lo, "att_hi": hi})

    def summary(self) -> dict:
        T1 = len(self.att_daily)
        return {"estimator": self.name, "outcome": self.outcome, "att_mean": self.att_mean, "se": self.se,
                "ci_lo": self.ci_lo, "ci_hi": self.ci_hi, "att_total": self.att_total,
                "cum_lo": None if self.ci_lo is None else self.ci_lo * T1,
                "cum_hi": None if self.ci_hi is None else self.ci_hi * T1,
                "pre_rmspe": self.pre_rmspe, "post_rmspe": self.post_rmspe,
                "p_placebo": self.meta.get("p_placebo")}


def _finish(name: str, pd_: PanelData, y0_hat: np.ndarray, att_daily: np.ndarray, **kw) -> PanelResult:
    T0 = pd_.T0
    pre = rmspe(pd_.y_tr[:T0], y0_hat[:T0])
    post = rmspe(pd_.y_tr[T0:], y0_hat[T0:])
    return PanelResult(name=name, outcome=pd_.outcome, att_mean=float(att_daily.mean()), se=None, ci_lo=None,
                       ci_hi=None, att_daily=att_daily, y0_hat=y0_hat, y_obs=pd_.y_tr, days=pd_.days, T0=T0,
                       pre_rmspe=pre, post_rmspe=post, **kw)


def _with_calendar(fn: Callable[..., "PanelResult"]) -> Callable[..., "PanelResult"]:
    """DiD-family estimators run on calendar-adjusted outcomes (unit-specific weekend/holiday effects)."""
    def wrapped(pd_: PanelData, cfg: dict, **kw) -> "PanelResult":
        if not cfg["estimators"].get("calendar_adjust", True):
            return fn(pd_, cfg, **kw)
        adj, off = pd_.calendar_adjusted()
        res = fn(adj, cfg, **kw)
        res.y0_hat = res.y0_hat + off
        res.y_obs = pd_.y_tr
        res.pre_rmspe = rmspe(pd_.y_tr[:pd_.T0], res.y0_hat[:pd_.T0])
        res.post_rmspe = rmspe(pd_.y_tr[pd_.T0:], res.y0_hat[pd_.T0:])
        res.meta["calendar_adjusted"] = True
        return res
    wrapped.__name__ = fn.__name__
    return wrapped


# ----------------------------------------------------------------------------- 1. TWFE
def _dm(a: np.ndarray) -> np.ndarray:
    return a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()


@_with_calendar
def estimator_twfe(pd_: PanelData, cfg: dict, inference: bool = True, **_) -> PanelResult:
    """Y_it = a_i + l_t + b*D_it + X_it'g + e_it, DMA-clustered (CR1) SE. Exact via Frisch-Waugh."""
    Y = np.vstack([pd_.y_tr, pd_.Yc])
    N, T = Y.shape
    D = np.zeros((N, T))
    D[0, pd_.T0:] = 1.0
    X = np.concatenate([pd_.cov_tr[None], pd_.cov_c], axis=0)
    cols = [_dm(D)] + [_dm(X[:, :, k]) for k in range(X.shape[2])]
    W = np.column_stack([c.ravel() for c in cols])
    yv = _dm(Y).ravel()
    bread = np.linalg.inv(W.T @ W)
    beta = bread @ (W.T @ yv)
    e = yv - W @ beta
    se_cl = None
    if inference:
        ids = np.repeat(np.arange(N), T)
        meat = np.zeros((W.shape[1], W.shape[1]))
        for g in range(N):
            s = W[ids == g].T @ e[ids == g]
            meat += np.outer(s, s)
        K = W.shape[1] + (N - 1) + (T - 1)
        adj = N / (N - 1) * (N * T - 1) / (N * T - K)
        V = adj * bread @ meat @ bread
        se_cl = float(np.sqrt(V[0, 0]))
    b = float(beta[0])
    # no-covariate variant (reported for transparency)
    w0 = cols[0].ravel()
    b_nocov = float(w0 @ yv / (w0 @ w0))
    fitted = Y - e.reshape(N, T)
    y0_hat = fitted[0].copy()
    y0_hat[pd_.T0:] = pd_.y_tr[pd_.T0:] - b
    res = _finish("TWFE DiD", pd_, y0_hat, np.full(pd_.T1, b))
    res.meta.update({"att_nocov": b_nocov, "se_cluster": se_cl, "beta_cov": dict(zip(pd_.cov_names, beta[1:].tolist()))})
    return res


# ----------------------------------------------------------------------------- 2. Event study
def _event_study_betas(y_tr: np.ndarray, Yc: np.ndarray, cov_tr: np.ndarray, cov_c: np.ndarray,
                       T0: int, with_hc: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """beta_t for every day t (k = t - T0 relative to the first campaign day; reference k = -1)."""
    Y = np.vstack([y_tr, Yc])
    N, T = Y.shape
    X = np.concatenate([cov_tr[None], cov_c], axis=0)
    ref = T0 - 1
    tcols = [t for t in range(T) if t != ref]
    cols = []
    for t in tcols:
        E = np.zeros((N, T))
        E[0, t] = 1.0
        cols.append(_dm(E).ravel())
    cols += [_dm(X[:, :, k]).ravel() for k in range(X.shape[2])]
    W = np.column_stack(cols)
    yv = _dm(Y).ravel()
    beta, *_ = np.linalg.lstsq(W, yv, rcond=None)
    full = np.zeros(T)
    full[tcols] = beta[:len(tcols)]
    se = None
    if with_hc:
        e = yv - W @ beta
        Wd = W * e[:, None]
        bread = np.linalg.pinv(W.T @ W)
        V = len(yv) / (len(yv) - W.shape[1]) * bread @ (Wd.T @ Wd) @ bread
        s = np.sqrt(np.clip(np.diag(V)[:len(tcols)], 0, None))
        se = np.zeros(T)
        se[tcols] = s
    return full, se


@_with_calendar
def estimator_event_study(pd_: PanelData, cfg: dict, inference: bool = True, **_) -> PanelResult:
    es = cfg["estimators"]["event_study"]
    T0, T = pd_.T0, pd_.T
    beta, se_hc = _event_study_betas(pd_.y_tr, pd_.Yc, pd_.cov_tr, pd_.cov_c, T0, with_hc=inference)
    ks = np.arange(T) - T0
    win = (ks >= es["k_min"]) & (ks <= es["k_max"])
    post = ks >= 0
    res = _finish("Event Study", pd_, pd_.y_tr - beta, beta[post])
    leads = (ks >= es["k_min"]) & (ks <= -2)
    slope = float(np.polyfit(ks[leads], beta[leads], 1)[0]) if leads.sum() > 2 else float("nan")
    res.meta.update({"k": ks[win], "beta": beta[win], "se_hc1": None if se_hc is None else se_hc[win],
                     "beta_full": beta, "pretrend_rms": float(np.sqrt(np.mean(beta[leads] ** 2))),
                     "pretrend_slope": slope,
                     "post_early": float(beta[(ks >= 0) & (ks < 20)].mean()),
                     "post_late": float(beta[ks >= T - T0 - 20].mean())})
    return res


def attach_event_study_placebo(res: PanelResult, pd_: PanelData, cfg: dict) -> None:
    """Placebo event studies (each donor pseudo-treated) -> SEs, bands and a permutation pre-trend test."""
    es = cfg["estimators"]["event_study"]
    T0 = pd_.T0
    ks_full = np.arange(pd_.T) - T0
    win = (ks_full >= es["k_min"]) & (ks_full <= es["k_max"])
    leads = (ks_full >= es["k_min"]) & (ks_full <= -2)
    post = ks_full >= 0
    pl = []
    for u in pd_.donors:
        q = pd_.as_treated(u)
        if cfg["estimators"].get("calendar_adjust", True):
            q = q.calendar_adjusted()[0]
        b, _ = _event_study_betas(q.y_tr, q.Yc, q.cov_tr, q.cov_c, T0)
        pl.append(b)
    pl = np.array(pl)
    se_k = np.sqrt((pl ** 2).mean(axis=0))
    beta = res.meta["beta_full"]
    stat = np.sqrt(np.mean(beta[leads] ** 2))
    pstats = np.sqrt(np.mean(pl[:, leads] ** 2, axis=1))
    p_pre = float((1 + (pstats >= stat).sum()) / (len(pstats) + 1))
    kl = ks_full[leads]
    slopes = np.array([np.polyfit(kl, pl[i][leads], 1)[0] for i in range(len(pl))])
    p_slope = float((1 + (np.abs(slopes) >= abs(res.meta["pretrend_slope"])).sum()) / (len(slopes) + 1))
    placebo_att = pl[:, post].mean(axis=1)
    se = float(np.sqrt(np.mean(placebo_att ** 2)))
    res.se, res.ci_lo, res.ci_hi = se, res.att_mean - Z95 * se, res.att_mean + Z95 * se
    res.daily_lo, res.daily_hi = beta[post] - Z95 * se_k[post], beta[post] + Z95 * se_k[post]
    res.meta.update({"se_placebo_k": se_k[win], "placebo_betas": pl[:, win], "p_pretrend_perm": p_pre,
                     "p_pretrend_slope_perm": p_slope,
                     "p_placebo": float((1 + (np.abs(placebo_att) >= abs(res.att_mean)).sum()) / (len(placebo_att) + 1))})


# ----------------------------------------------------------------------------- 3. Callaway & Sant'Anna
@_with_calendar
def estimator_csdid(pd_: PanelData, cfg: dict, inference: bool = True, seed: int = 0, **_) -> PanelResult:
    """Doubly-robust ATT(g,t), never-treated controls, base period g-1 (Sant'Anna & Zhao 2020 panel DR).

    One cohort (g = campaign start). Outcome regression: ridge of Delta-Y on unit-level pre-period covariate
    means; propensity: strongly-ridged logistic model (one treated unit -> near-constant weights).
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    c = cfg["estimators"]["csdid"]
    Y = np.vstack([pd_.y_tr, pd_.Yc])
    N, T = Y.shape
    T0 = pd_.T0
    X = np.concatenate([pd_.cov_tr[None], pd_.cov_c], axis=0)
    Xu = np.column_stack([X[:, :T0, k].mean(axis=1) for k in range(X.shape[2])] + [Y[:, :T0].mean(axis=1)])
    mu, sd = Xu[1:].mean(0), Xu[1:].std(0) + 1e-9
    Xs = (Xu - mu) / sd
    G = np.zeros(N)
    G[0] = 1.0
    ps = LogisticRegression(C=1.0 / c.get("propensity_c", 5.0), max_iter=2000).fit(Xs, G).predict_proba(Xs)[:, 1]
    ps = np.clip(ps, 0.005, 0.995)
    dY = Y - Y[:, [T0 - 1]]
    orm = Ridge(alpha=c.get("or_ridge", 5.0)).fit(Xs[1:], dY[1:])
    r = dY - orm.predict(Xs)                                  # (N,T) outcome-regression residual of the change
    wc = ps / (1 - ps) * (1 - G)
    wc = wc / wc.sum()
    att_t = r[0] - (wc[1:, None] * r[1:]).sum(axis=0)          # ATT(g,t) for every t
    att_t[T0 - 1] = 0.0
    post = np.arange(T) >= T0
    y0_hat = pd_.y_tr - att_t
    res = _finish("CS-DiD (DR)", pd_, y0_hat, att_t[post])
    ks = np.arange(T) - T0
    res.meta.update({"k": ks, "att_gt": att_t, "propensity_treated": float(ps[0])})
    if inference:
        rng = np.random.default_rng(seed)
        psi = np.zeros((N, T))
        psi[0] = (wc[1:, None] * r[1:]).sum(axis=0)             # treated: r_0 - ATT = weighted control mean
        psi[1:] = -wc[1:, None] * r[1:]
        B = c.get("boot_reps", 500)
        a = (np.sqrt(5) + 1) / 2
        pa = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        V = np.where(rng.random((B, N)) < pa, 1 - a, a)           # Mammen multipliers
        draws = att_t[None, :] + V @ psi
        res.meta["se_boot"] = float(draws[:, post].mean(axis=1).std(ddof=1))
        res.meta["boot_band_lo"] = np.quantile(draws, 0.025, axis=0)
        res.meta["boot_band_hi"] = np.quantile(draws, 0.975, axis=0)
    return res


# ----------------------------------------------------------------------------- 4. SCM
def _unit_cov_means(pd_: PanelData) -> tuple[np.ndarray, np.ndarray]:
    T0 = pd_.T0
    return pd_.cov_tr[:T0].mean(axis=0), pd_.cov_c[:, :T0].mean(axis=1)


def fit_scm_weights(pd_: PanelData, cfg: dict, use_cov: bool | None = None, t_end: int | None = None) -> np.ndarray:
    """Abadie weights on the simplex: pre-period outcome path (+ standardised covariate means)."""
    sc = cfg["estimators"]["scm"]
    use_cov = sc["use_covariates"] if use_cov is None else use_cov
    T0 = pd_.T0 if t_end is None else t_end
    scale = max(pd_.y_tr[:T0].mean(), 1e-9)
    X = pd_.Yc[:, :T0].T / scale
    y = pd_.y_tr[:T0] / scale
    lw = np.full(T0, 1.0 / T0)
    if use_cov and sc["covariate_weight"] > 0:
        lam = sc["covariate_weight"]
        ctr, cc = _unit_cov_means(pd_)
        mu, sd = cc.mean(0), cc.std(0) + 1e-9
        Xc = ((cc - mu) / sd).T
        yc = (ctr - mu) / sd
        p = len(yc)
        X = np.vstack([X, Xc])
        y = np.concatenate([y, yc])
        lw = np.concatenate([lw * (1 - lam), np.full(p, lam / p)])
    return simplex_ls(X, y, loss_w=lw)


def covariate_balance(pd_: PanelData, w: np.ndarray) -> pd.DataFrame:
    ctr, cc = _unit_cov_means(pd_)
    sd = cc.std(0) + 1e-9
    rows = [{"covariate": n, "treated": ctr[i], "synthetic": float(w @ cc[:, i]), "donor_mean": float(cc[:, i].mean()),
             "std_diff_synthetic": float((ctr[i] - w @ cc[:, i]) / sd[i]),
             "std_diff_donor_mean": float((ctr[i] - cc[:, i].mean()) / sd[i])} for i, n in enumerate(pd_.cov_names)]
    T0 = pd_.T0
    pre_c = pd_.Yc[:, :T0].mean(axis=1)
    sdo = pre_c.std() + 1e-9
    rows.append({"covariate": "pre-period outcome mean", "treated": pd_.y_tr[:T0].mean(), "synthetic": float(w @ pre_c),
                 "donor_mean": float(pre_c.mean()), "std_diff_synthetic": float((pd_.y_tr[:T0].mean() - w @ pre_c) / sdo),
                 "std_diff_donor_mean": float((pd_.y_tr[:T0].mean() - pre_c.mean()) / sdo)})
    return pd.DataFrame(rows)


def estimator_scm(pd_: PanelData, cfg: dict, inference: bool = True, **_) -> PanelResult:
    w = fit_scm_weights(pd_, cfg)
    y0 = w @ pd_.Yc
    res = _finish("SCM", pd_, y0, (pd_.y_tr - y0)[pd_.T0:], weights=dict(zip(pd_.donors, w.tolist())))
    res.meta["w"] = w
    if inference:
        res.meta["balance"] = covariate_balance(pd_, w)
    return res


# ----------------------------------------------------------------------------- 5. ASCM
def _ridge_aug_weights(Yc_pre: np.ndarray, y_pre: np.ndarray, w: np.ndarray, lam_rel: float) -> np.ndarray:
    """w_aug = w + (Xc Xc' + lam I)^-1 Xc (x_tr - Xc' w), with column-centred donor pre-outcomes (intercept free)."""
    Xt = Yc_pre - Yc_pre.mean(axis=0, keepdims=True)          # (n,T0) centred across donors
    n = Xt.shape[0]
    K = Xt @ Xt.T
    lam = lam_rel * np.trace(K) / n
    resid = y_pre - Yc_pre.T @ w                             # (T0,) imbalance of the SCM fit
    return w + np.linalg.solve(K + lam * np.eye(n), Xt @ resid)


def _select_ridge_lambda(pd_: PanelData, cfg: dict, w_fun: Callable[[PanelData, int], np.ndarray], h: int = 30) -> float:
    """Rolling-origin validation: fit on days [0, T0-h), score the augmented counterfactual on the last h pre-days."""
    grid = cfg["estimators"]["ascm"]["ridge_grid"]
    T0 = pd_.T0
    t1 = T0 - h
    w = w_fun(pd_, t1)
    best, best_err = grid[-1], np.inf
    for lam in grid:
        wa = _ridge_aug_weights(pd_.Yc[:, :t1], pd_.y_tr[:t1], w, lam)
        err = rmspe(pd_.y_tr[t1:T0], wa @ pd_.Yc[:, t1:T0])
        if err < best_err:
            best, best_err = lam, err
    return float(best)


def estimator_ascm(pd_: PanelData, cfg: dict, inference: bool = True, kind: str = "ridge",
                   lam: float | None = None, **_) -> PanelResult:
    T0 = pd_.T0
    w0 = fit_scm_weights(pd_, cfg)
    y0 = np.empty(pd_.T)
    y0[:T0] = w0 @ pd_.Yc[:, :T0]                              # pre-period fit shown with the SCM weights
    if kind == "ridge":
        if lam is None:
            lam = _select_ridge_lambda(pd_, cfg, lambda p, te: fit_scm_weights(p, cfg, use_cov=False, t_end=te))
        wa = _ridge_aug_weights(pd_.Yc[:, :T0], pd_.y_tr[:T0], w0, lam)
        y0[T0:] = wa @ pd_.Yc[:, T0:]
        weights = dict(zip(pd_.donors, wa.tolist()))
        name = "ASCM (Ridge)"
        meta = {"lam": lam, "w_scm": w0, "w_aug": wa, "neg_weight_mass": float(-wa[wa < 0].sum())}
    else:
        from sklearn.linear_model import ElasticNet

        en = cfg["estimators"]["ascm"]["elasticnet"]
        scale = max(pd_.y_tr[:T0].mean(), 1e-9)
        Xp = pd_.Yc[:, :T0] / scale
        Yp = pd_.Yc[:, T0:] / scale
        mdl = ElasticNet(alpha=en["alpha"], l1_ratio=en["l1_ratio"], max_iter=20000, tol=1e-6).fit(Xp, Yp)
        m_tr = mdl.predict((pd_.y_tr[:T0] / scale)[None])[0] * scale
        m_c = mdl.predict(Xp) * scale
        y0[T0:] = w0 @ pd_.Yc[:, T0:] + m_tr - w0 @ m_c
        weights = dict(zip(pd_.donors, w0.tolist()))
        name = "ASCM (ElasticNet)"
        meta = {"w_scm": w0}
    res = _finish(name, pd_, y0, (pd_.y_tr - y0)[T0:], weights=weights)
    res.meta.update(meta)
    return res


# ----------------------------------------------------------------------------- 6. SDID
def estimator_sdid(pd_: PanelData, cfg: dict, inference: bool = True, **_) -> PanelResult:
    """Arkhangelsky et al. (2021): unit weights (intercept + ridge) and time weights, then weighted DiD."""
    T0, T1 = pd_.T0, pd_.T1
    Yc, y = pd_.Yc, pd_.y_tr
    sigma = np.diff(Yc[:, :T0], axis=1).std(ddof=1)
    zeta = cfg["estimators"]["sdid"]["zeta_scale"] * (T1 ** 0.25) * sigma
    Xc = (Yc[:, :T0] - Yc[:, :T0].mean(axis=1, keepdims=True)).T          # (T0,n), unit intercepts removed
    yc = y[:T0] - y[:T0].mean()
    omega = simplex_ls(Xc, yc, ridge=zeta ** 2 * T0)
    A = Yc[:, :T0] - Yc[:, :T0].mean(axis=0, keepdims=True)               # (n,T0), time intercept removed
    b = Yc[:, T0:].mean(axis=1) - Yc[:, T0:].mean()
    lam = simplex_ls(A, b, ridge=(1e-6 * sigma) ** 2 * len(b))
    intercept = float(lam @ y[:T0] - omega @ (Yc[:, :T0] @ lam))
    y0 = omega @ Yc + intercept
    res = _finish("SDID", pd_, y0, (y - y0)[T0:], weights=dict(zip(pd_.donors, omega.tolist())))
    res.meta.update({"omega": omega, "lambda_time": lam, "zeta": float(zeta), "intercept": intercept})
    return res


# ----------------------------------------------------------------------------- 7. Bayesian SC
def estimator_bayes(pd_: PanelData, cfg: dict, inference: bool = True, seed: int = 0, **_) -> PanelResult:
    """y_t = mu0 + sum_j w_j y_jt + u_t,  w ~ Dirichlet(1), u_t AR(1).  NUTS via NumPyro (JAX)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary as np_summary
    from numpyro.infer import MCMC, NUTS

    jax.config.update("jax_enable_x64", True)
    bc = cfg["estimators"]["bayes"]
    T0, T1 = pd_.T0, pd_.T1
    scale = float(pd_.y_tr[:T0].mean())
    Z = jnp.asarray(pd_.Yc[:, :T0].T / scale)
    y = jnp.asarray(pd_.y_tr[:T0] / scale)
    ar1 = bool(bc.get("ar1", True))
    ar_order = int(bc.get("ar_order", 1)) if ar1 else 0
    conc = float(bc.get("dirichlet_conc", 0.5))
    mu0_sd = float(bc.get("intercept_sd", 0.05))
    tacc = float(bc.get("target_accept", 0.97))
    rho_max = float(bc.get("rho_max", 0.98))

    def model(Z, y):
        n = Z.shape[1]
        w = numpyro.sample("w", dist.Dirichlet(jnp.full(n, conc)))
        mu0 = numpyro.sample("mu0", dist.Normal(0.0, mu0_sd)) if mu0_sd > 0 else 0.0
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))
        r = y - mu0 - Z @ w
        if ar_order == 2:                         # PACF parametrisation keeps the AR(2) stationary
            f1 = numpyro.sample("pacf1", dist.Uniform(-0.95, rho_max))
            f2 = numpyro.sample("pacf2", dist.Uniform(-0.9, 0.95))
            a1, a2 = f1 * (1.0 - f2), f2
            ll = dist.Normal(a1 * r[1:-1] + a2 * r[:-2], sigma).log_prob(r[2:]).sum()
            ll0 = dist.Normal(0.0, sigma / jnp.sqrt(1.0 - f1 ** 2)).log_prob(r[0]) + dist.Normal(f1 * r[0], sigma).log_prob(r[1])
        else:
            rho = numpyro.sample("rho", dist.Uniform(-0.5, rho_max)) if ar_order == 1 else 0.0
            ll0 = dist.Normal(0.0, sigma / jnp.sqrt(1.0 - rho ** 2)).log_prob(r[0])
            ll = dist.Normal(rho * r[:-1], sigma).log_prob(r[1:]).sum()
        numpyro.factor("lik", ll0 + ll)

    chains = int(bc["chains"])
    method = "parallel" if jax.local_device_count() >= chains else "sequential"
    mcmc = MCMC(NUTS(model, target_accept_prob=tacc), num_warmup=int(bc["warmup"]), num_samples=int(bc["samples"]),
                num_chains=chains, chain_method=method, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), Z, y, extra_fields=("diverging",))
    smp = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    grouped = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    sites = ["w", "sigma"] + (["mu0"] if mu0_sd > 0 else []) + {0: [], 1: ["rho"], 2: ["pacf1", "pacf2"]}[ar_order]
    diag = np_summary({k: grouped[k] for k in sites}, group_by_chain=True)
    rhat_max = float(max(np.max(np.asarray(d["r_hat"])) for d in diag.values()))
    ess_min = float(min(np.min(np.asarray(d["n_eff"])) for d in diag.values()))
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

    W, sig = smp["w"], smp["sigma"]
    mu0 = smp["mu0"] if mu0_sd > 0 else np.zeros_like(sig)
    S = len(sig)
    rng = np.random.default_rng(seed + 101)
    Zall = pd_.Yc.T / scale                                              # (T,n)
    mean_all = mu0[:, None] + W @ Zall.T                                  # (S,T) systematic part
    res_all = pd_.y_tr[None, :] / scale - mean_all                        # (S,T) residual path
    if ar_order == 2:
        f1s, f2s = smp["pacf1"], smp["pacf2"]
        a1, a2 = f1s * (1 - f2s), f2s
    elif ar_order == 1:
        a1, a2 = smp["rho"], np.zeros_like(sig)
    else:
        a1, a2 = np.zeros_like(sig), np.zeros_like(sig)
    rho = a1
    u = np.empty((S, T1))
    p1, p2 = res_all[:, T0 - 1], res_all[:, T0 - 2]
    for h in range(T1):
        new_u = a1 * p1 + a2 * p2 + sig * rng.standard_normal(S)
        u[:, h] = new_u
        p2, p1 = p1, new_u
    y0_post = (mean_all[:, T0:] + u) * scale                              # predictive counterfactual
    att_draws = pd_.y_tr[T0:][None, :] - y0_post                          # (S,T1)
    y0_pre = mean_all[:, :T0] * scale
    y0_hat = np.concatenate([y0_pre.mean(axis=0), y0_post.mean(axis=0)])
    att_daily = att_draws.mean(axis=0)
    mean_draws = att_draws.mean(axis=1)
    cum_draws = att_draws.sum(axis=1)
    lo, hi = np.quantile(att_draws, [0.025, 0.975], axis=0)
    # posterior predictive check on the pre-period (observed-noise intervals)
    pre_pred = y0_pre + (sig[:, None] * rng.standard_normal((S, T0))) * scale
    plo, phi = np.quantile(pre_pred, [0.025, 0.975], axis=0)
    ppc_cov = float(np.mean((pd_.y_tr[:T0] >= plo) & (pd_.y_tr[:T0] <= phi)))
    res = _finish("Bayesian SC", pd_, y0_hat, att_daily, weights=dict(zip(pd_.donors, W.mean(axis=0).tolist())))
    q = np.quantile(mean_draws, [0.025, 0.975])
    res.att_mean = float(mean_draws.mean())
    res.se = float(mean_draws.std(ddof=1))
    res.ci_lo, res.ci_hi = float(q[0]), float(q[1])
    res.daily_lo, res.daily_hi = lo, hi
    res.meta.update({"rhat_max": rhat_max, "ess_min": ess_min, "divergences": div, "ppc_pre_coverage": ppc_cov,
                     "prob_positive": float((mean_draws > 0).mean()), "rho_mean": float(np.mean(a1)),
                     "sigma_mean": float(sig.mean() * scale), "draws_mean_att": mean_draws,
                     "draws_cum": cum_draws, "att_draws": att_draws[:: max(1, S // 400)],
                     "w_sd": dict(zip(pd_.donors, W.std(axis=0).tolist())),
                     "cum_lo": float(np.quantile(cum_draws, 0.025)), "cum_hi": float(np.quantile(cum_draws, 0.975)),
                     "p_placebo": None, "jax_backend": jax.default_backend(), "chain_method": method})
    return res


# ----------------------------------------------------------------------------- placebo inference
def run_placebos(fn: Callable[..., PanelResult], pd_: PanelData, cfg: dict, **kw) -> dict:
    """Each donor plays the treated unit (donor pool = the other donors). Returns arrays over placebo units."""
    units, att, pre, post, gaps, gaps_full = [], [], [], [], [], []
    for u in pd_.donors:
        q = pd_.as_treated(u)
        r = fn(q, cfg, inference=False, **kw)
        units.append(u)
        att.append(r.att_mean)
        pre.append(r.pre_rmspe)
        post.append(r.post_rmspe)
        gaps.append(r.att_daily)
        gaps_full.append(q.y_tr - r.y0_hat)
    return {"units": units, "att_mean": np.array(att), "pre_rmspe": np.array(pre), "post_rmspe": np.array(post),
            "gap": np.array(gaps), "gap_full": np.array(gaps_full)}


def attach_placebo(res: PanelResult, pd_: PanelData, fn: Callable[..., PanelResult], cfg: dict, **kw) -> dict:
    """Calibrate SE / bands / exact p-value from in-space placebos (pre-fit filtered)."""
    pl = run_placebos(fn, pd_, cfg, **kw)
    mult = cfg["diagnostics"]["rmspe_filter_multiple"]
    keep = pl["pre_rmspe"] <= mult * max(res.pre_rmspe, 1e-9)
    if keep.sum() < 8:
        keep = np.ones_like(keep, dtype=bool)
    se = float(np.sqrt(np.mean(pl["att_mean"][keep] ** 2)))
    band = np.sqrt(np.mean(pl["gap"][keep] ** 2, axis=0))
    res.se, res.ci_lo, res.ci_hi = se, res.att_mean - Z95 * se, res.att_mean + Z95 * se
    res.daily_lo, res.daily_hi = res.att_daily - Z95 * band, res.att_daily + Z95 * band
    ratio_t = res.post_rmspe / max(res.pre_rmspe, 1e-9)
    ratio_p = pl["post_rmspe"] / np.maximum(pl["pre_rmspe"], 1e-9)
    res.meta["p_placebo"] = float((1 + (ratio_p >= ratio_t).sum()) / (len(ratio_p) + 1))
    res.meta["p_placebo_att"] = float((1 + (np.abs(pl["att_mean"]) >= abs(res.att_mean)).sum()) / (len(pl["att_mean"]) + 1))
    res.meta["placebo"] = pl
    res.meta["placebo_keep"] = keep
    return pl


ESTIMATOR_FUNCS: dict[str, Callable[..., PanelResult]] = {
    "TWFE DiD": estimator_twfe, "Event Study": estimator_event_study, "CS-DiD (DR)": estimator_csdid,
    "SCM": estimator_scm, "ASCM (Ridge)": estimator_ascm, "SDID": estimator_sdid, "Bayesian SC": estimator_bayes,
}


def run_suite(pd_: PanelData, cfg: dict, seed: int = 0, bayes: bool = True, placebo: bool = True,
              elasticnet: bool = True, only: list[str] | None = None) -> dict[str, PanelResult]:
    """Fit all estimators (optionally a subset) on one panel and calibrate inference."""
    log = get_logger("panel")
    out: dict[str, PanelResult] = {}
    names = only or ESTIMATOR_ORDER
    for name in names:
        if name == "Bayesian SC" and not bayes:
            continue
        fn = ESTIMATOR_FUNCS[name]
        res = fn(pd_, cfg, inference=True, seed=seed)
        if placebo:
            if name == "Event Study":
                attach_event_study_placebo(res, pd_, cfg)
            elif name == "Bayesian SC":
                pass
            elif name == "ASCM (Ridge)":
                attach_placebo(res, pd_, fn, cfg, lam=res.meta["lam"])
            else:
                attach_placebo(res, pd_, fn, cfg)
        out[name] = res
        log.info("  %-13s ATT %9.2f/day  (SE %s)  pre-RMSPE %s", name, res.att_mean,
                 "n/a" if res.se is None else f"{res.se:.2f}",
                 "n/a" if res.pre_rmspe is None else f"{res.pre_rmspe:.1f}")
    if elasticnet and (only is None or "ASCM (ElasticNet)" in only):
        r = estimator_ascm(pd_, cfg, kind="elasticnet")
        if placebo:
            attach_placebo(r, pd_, estimator_ascm, cfg, kind="elasticnet")
        out["ASCM (ElasticNet)"] = r
    return out


# ----------------------------------------------------------------------------- ground truth helpers
def truth_series(df: pd.DataFrame, treated: str, outcome: str, T: int) -> np.ndarray:
    """True daily ATT (zeros before the campaign) for the treated market."""
    col = TRUTH_COL[outcome]
    s = df[df["dma_code"].astype(str) == treated].sort_values("day_index")[col].to_numpy(float)
    return s[:T]


def evaluate_against_truth(res: PanelResult, truth_daily: np.ndarray, y0_true: np.ndarray | None = None) -> dict:
    T0 = res.T0
    t_post = truth_daily[T0:]
    out = {"att_true_mean": float(t_post.mean()), "att_bias": float(res.att_mean - t_post.mean()),
           "att_bias_pct": float(100 * (res.att_mean - t_post.mean()) / max(abs(t_post.mean()), 1e-9)) if abs(t_post.mean()) > 1e-9 else float("nan"),
           "daily_rmse": float(rmspe(res.att_daily, t_post)), "total_true": float(t_post.sum()),
           "covered": None if res.ci_lo is None else bool(res.ci_lo <= t_post.mean() <= res.ci_hi)}
    if y0_true is not None:
        out["cf_rmse_post"] = float(rmspe(res.y0_hat[T0:], y0_true[T0:]))
        out["cf_rmse_pre"] = float(rmspe(res.y0_hat[:T0], y0_true[:T0]))
    return out
