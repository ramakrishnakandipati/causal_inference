"""Macro geo-lift estimators of the ATT of *pausing* branded paid search on treated DMAs.

All estimators share one interface: given a DMA x day outcome matrix, a treated mask and a
pre/post cut, they return the average per-DMA-day effect on treated DMAs (ATT), a 95% interval and
the daily observed / counterfactual series for plotting.

* ``mmt``  - Matched Market Test: DTW + Mahalanobis pairs, ratio-adjusted counterfactual,
             exact paired sign-flip permutation p-value (2^pairs).
* ``bsts`` - Bayesian structural time series in NumPyro (local linear trend + day-of-week +
             holiday + control-DMA regression), forecast counterfactual with 95% credible band.
* ``sdid`` - Synthetic DiD (Arkhangelsky et al.) with ridge-regularised simplex unit/time weights.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from src.data.feature_engineering import wide
from src.utils.config import load_config

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------ data container
class GeoData:
    """Wide matrices for one outcome plus calendar covariates."""

    def __init__(self, panel: pd.DataFrame, dma: pd.DataFrame, outcome: str):
        self.outcome = outcome
        self.Y = wide(panel, outcome).loc[dma.dma_id].values.astype(float)       # N x T
        self.flight = wide(panel, "flight_idx").loc[dma.dma_id].values.astype(float)
        cal = panel[panel.dma_id == dma.dma_id.iloc[0]].sort_values("day")
        self.dow = cal.dow.values
        self.holiday = cal.holiday.values.astype(float)
        self.dates = cal.date.values
        self.T = self.Y.shape[1]
        self.treated = (dma["T"].values == 1)
        self.tier = dma.tier.values
        self.dma_names = dma.dma.values

    def split(self, cut: int, end: int | None = None, treated: np.ndarray | None = None,
              pool: np.ndarray | None = None):
        """Return dict with treated-mean series, control matrix and the cut (post = [cut:end])."""
        end = end or self.T
        tr = self.treated if treated is None else treated
        co = ~self.treated if pool is None else pool
        co = co & ~tr
        return {"y": self.Y[tr, :end].mean(0), "Yco": self.Y[co, :end], "Ytr": self.Y[tr, :end],
                "tr": tr, "co": co, "cut": cut, "end": end, "flight_tr": self.flight[tr, :end].mean(0),
                "tier_tr": self.tier[tr], "tier_co": self.tier[co]}


def _result(att, lo, hi, obs, cf, cf_lo, cf_hi, n_dmas, n_days, **extra):
    d = {"att": float(att), "ci_low": float(lo), "ci_high": float(hi),
         "cumulative": float(att * n_dmas * n_days),
         "cumulative_ci": [float(lo * n_dmas * n_days), float(hi * n_dmas * n_days)],
         "observed": np.asarray(obs), "counterfactual": np.asarray(cf),
         "cf_low": np.asarray(cf_lo), "cf_high": np.asarray(cf_hi)}
    d.update(extra)
    return d


# ------------------------------------------------------------------------------ MMT
def mmt(gd: GeoData, pairs: pd.DataFrame, cut: int, end: int | None = None, n_boot: int = 2000, seed: int = 0):
    """Matched market test on explicit (treated_idx, control_idx) pairs."""
    end = end or gd.T
    ti, ci = pairs.treated_idx.values, pairs.control_idx.values
    Yt, Yc = gd.Y[ti, :end], gd.Y[ci, :end]
    ratio = Yt[:, :cut].mean(1) / np.maximum(Yc[:, :cut].mean(1), 1e-9)          # pre-period scaling
    cf = Yc * ratio[:, None]
    eff = (Yt[:, cut:] - cf[:, cut:])                                              # pairs x post days
    pair_eff = eff.mean(1)
    att = pair_eff.mean()
    # exact paired sign-flip permutation test (two-sided)
    k = len(pair_eff)
    signs = np.array(list(itertools.product([-1, 1], repeat=k)))
    perm = (signs * pair_eff[None, :]).mean(1)
    p_perm = float((np.abs(perm) >= abs(att) - 1e-12).mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (n_boot, k))
    boot = pair_eff[idx].mean(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    day_boot = np.stack([cf[rng.integers(0, k, k)].mean(0) for _ in range(500)])
    cfm = cf.mean(0)
    return _result(att, lo, hi, Yt.mean(0), cfm, np.quantile(day_boot, 0.025, 0), np.quantile(day_boot, 0.975, 0),
                   k, end - cut, p_perm=p_perm, pair_effects=pair_eff, method="MMT")


# ------------------------------------------------------------------------------ SDID
def _proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(len(v)) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def _simplex_ridge(A, b, reg, iters=3000):
    """argmin_w ||A w - b||^2 + reg*||w||^2 s.t. w in simplex (accelerated projected gradient)."""
    n = A.shape[1]
    w = np.full(n, 1.0 / n)
    L = 2 * (np.linalg.norm(A, 2) ** 2 + reg)
    z, t = w.copy(), 1.0
    for _ in range(iters):
        g = 2 * A.T @ (A @ z - b) + 2 * reg * z
        w_new = _proj_simplex(z - g / L)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = w_new + (t - 1) / t_new * (w_new - w)
        if np.linalg.norm(w_new - w) < 1e-10:
            w = w_new
            break
        w, t = w_new, t_new
    return w


def sdid_att(Ytr_mean, Yco, cut, n_tr, zeta_scale=1.0):
    """Point estimate of SDID. Ytr_mean: (T,), Yco: (Nco x T)."""
    T_pre, T_post = cut, Yco.shape[1] - cut
    Yco_pre, Yco_post = Yco[:, :cut], Yco[:, cut:]
    zeta = zeta_scale * (n_tr * T_post) ** 0.25 * np.std(np.diff(Yco_pre, axis=1))
    # unit weights: match treated pre trend (intercept absorbed by centring over time)
    A = (Yco_pre - Yco_pre.mean(1, keepdims=True)).T                    # T_pre x Nco
    b = Ytr_mean[:cut] - Ytr_mean[:cut].mean()
    omega = _simplex_ridge(A, b, zeta ** 2 * T_pre)
    # time weights: control pre periods that predict post mean (centring over units)
    B = (Yco_pre - Yco_pre.mean(0, keepdims=True))                       # Nco x T_pre
    c = Yco_post.mean(1) - Yco_post.mean(1).mean()
    lam = _simplex_ridge(B, c, 1e-6 * Yco.shape[0] * np.var(Yco_pre))
    tr_post, tr_pre = Ytr_mean[cut:].mean(), lam @ Ytr_mean[:cut]
    co_post, co_pre = omega @ Yco_post.mean(1), omega @ Yco_pre @ lam
    return (tr_post - tr_pre) - (co_post - co_pre), omega, lam


def sdid(gd: GeoData, cut: int, end: int | None = None, n_placebo: int = 200, seed: int = 0, zeta_scale: float = 1.0,
         treated: np.ndarray | None = None):
    s = gd.split(cut, end, treated=treated)
    n_tr = int(s["tr"].sum())
    att, omega, lam = sdid_att(s["y"], s["Yco"], cut, n_tr, zeta_scale)
    # placebo-in-space standard error: fake-treat random control subsets, re-estimate
    rng = np.random.default_rng(seed)
    Yco = s["Yco"]
    plac = []
    for _ in range(n_placebo):
        pick = rng.choice(Yco.shape[0], n_tr, replace=False)
        rest = np.setdiff1d(np.arange(Yco.shape[0]), pick)
        plac.append(sdid_att(Yco[pick].mean(0), Yco[rest], cut, n_tr, zeta_scale)[0])
    plac = np.array(plac)
    se = plac.std(ddof=1)
    # counterfactual path for plotting: omega-weighted controls shifted by the pre-period gap
    cf = omega @ Yco
    cf = cf + (lam @ (s["y"][:cut] - cf[:cut]))
    n_days = s["end"] - cut
    return _result(att, att - 1.96 * se, att + 1.96 * se, s["y"], cf, cf - 1.96 * se, cf + 1.96 * se, n_tr, n_days,
                   se=float(se), placebo_atts=plac, p_value=float(((np.abs(plac) >= abs(att)).sum() + 1) / (len(plac) + 1)),
                   unit_weights=omega, time_weights=lam, method="SDID")


# ------------------------------------------------------------------------------ BSTS (NumPyro)
def _design(s, gd: GeoData, k_pc: int = 4):
    """Regressors from control DMAs only: PCs of standardised controls + tier-reweighted mean + flight idx."""
    cut = s["cut"]
    Yco = s["Yco"]
    mu, sd = Yco[:, :cut].mean(1, keepdims=True), Yco[:, :cut].std(1, keepdims=True) + 1e-9
    Z = ((Yco - mu) / sd).T                                                  # T x Nco
    Zc = Z[:cut] - Z[:cut].mean(0)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    pcs = (Z - Z[:cut].mean(0)) @ Vt[:k_pc].T
    w = np.array([np.mean(s["tier_tr"] == t) / max(np.mean(s["tier_co"] == t), 1e-9) for t in s["tier_co"]])
    w = w / w.sum()
    tiermean = w @ Yco
    X = np.column_stack([pcs, tiermean, s["flight_tr"]])
    Xm, Xs = X[:cut].mean(0), X[:cut].std(0) + 1e-9
    return (X - Xm) / Xs


def bsts(gd: GeoData, cut: int, end: int | None = None, seed: int = 0, num_warmup: int | None = None,
         num_samples: int | None = None, num_chains: int | None = None, treated: np.ndarray | None = None):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    cfg = load_config()["models"]["bsts"]
    num_warmup = num_warmup or cfg["num_warmup"]
    num_samples = num_samples or cfg["num_samples"]
    num_chains = num_chains or cfg["num_chains"]
    end = end or gd.T
    s = gd.split(cut, end, treated=treated)
    y = s["y"]
    X = _design(s, gd)
    ym, ys = y[:cut].mean(), y[:cut].std()
    yz = (y - ym) / ys
    dow_oh = np.eye(7)[gd.dow[:end]]
    hol = gd.holiday[:end]

    def model(yobs, Xr, dow, h):
        T = yobs.shape[0]
        sig_obs = numpyro.sample("sig_obs", dist.HalfNormal(0.5))
        sig_l = numpyro.sample("sig_l", dist.HalfNormal(0.03))
        sig_s = numpyro.sample("sig_s", dist.HalfNormal(0.002))
        l0 = numpyro.sample("l0", dist.Normal(0, 1))
        s0 = numpyro.sample("s0", dist.Normal(0, 0.02))
        zl = numpyro.sample("zl", dist.Normal(0, 1).expand([T]).to_event(1))
        zs = numpyro.sample("zs", dist.Normal(0, 1).expand([T]).to_event(1))
        tau = numpyro.sample("tau", dist.HalfNormal(0.5))
        beta = numpyro.sample("beta", dist.Normal(0, tau).expand([Xr.shape[1]]).to_event(1))
        draw = numpyro.sample("dow_raw", dist.Normal(0, 0.5).expand([7]).to_event(1))
        gam = numpyro.sample("hol", dist.Normal(0, 0.5))
        slope = s0 + sig_s * jnp.cumsum(zs)
        level = l0 + jnp.cumsum(slope + sig_l * zl)
        d = draw - draw.mean()
        mu = level + Xr @ beta + dow @ d + gam * h
        numpyro.deterministic("level_T", level[-1])
        numpyro.deterministic("slope_T", slope[-1])
        numpyro.sample("y", dist.Normal(mu, sig_obs), obs=yobs)

    kern = NUTS(model, target_accept_prob=0.92)
    mc = MCMC(kern, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
              chain_method="sequential", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), yz[:cut], X[:cut], dow_oh[:cut], hol[:cut])
    p = {k: np.asarray(v) for k, v in mc.get_samples().items()}
    S, P = p["sig_obs"].shape[0], end - cut
    rng = np.random.default_rng(seed)
    zl_f, zs_f = rng.normal(size=(S, P)), rng.normal(size=(S, P))
    slope_f = p["slope_T"][:, None] + p["sig_s"][:, None] * np.cumsum(zs_f, 1)
    level_f = p["level_T"][:, None] + np.cumsum(slope_f + p["sig_l"][:, None] * zl_f, 1)
    d = p["dow_raw"] - p["dow_raw"].mean(1, keepdims=True)
    mu_f = level_f + p["beta"] @ X[cut:].T + d @ dow_oh[cut:].T + p["hol"][:, None] * hol[cut:][None, :]
    cf_draws = (mu_f + rng.normal(size=(S, P)) * p["sig_obs"][:, None]) * ys + ym          # S x P (original units)
    eff_draws = y[cut:][None, :] - cf_draws
    att_draws = eff_draws.mean(1)
    lo, hi = np.quantile(att_draws, [0.025, 0.975])
    n_tr = int(s["tr"].sum())
    try:
        from numpyro.diagnostics import split_gelman_rubin
        ch = mc.get_samples(group_by_chain=True)
        rhat = float(max(np.max(split_gelman_rubin(np.asarray(ch[k]))) for k in ("sig_obs", "sig_l", "sig_s", "tau", "beta", "l0")))
    except Exception:
        rhat = float("nan")
    cf_mean = np.concatenate([np.full(cut, np.nan), cf_draws.mean(0)])
    cf_lo = np.concatenate([np.full(cut, np.nan), np.quantile(cf_draws, 0.025, 0)])
    cf_hi = np.concatenate([np.full(cut, np.nan), np.quantile(cf_draws, 0.975, 0)])
    return _result(att_draws.mean(), lo, hi, y, cf_mean, cf_lo, cf_hi, n_tr, P,
                   prob_negative=float((att_draws < 0).mean()), att_draws=att_draws, rhat_max=rhat,
                   n_regressors=int(X.shape[1]), method="BSTS")


# ------------------------------------------------------------------------------ refutations
def placebo_in_time(gd: GeoData, pairs: pd.DataFrame, weeks: int = 4, run_bsts: bool = True) -> list[dict]:
    """Pretend the pause started ``weeks`` before the real start, using pre-period data only."""
    real_cut = gd.T - 28
    cut, end = real_cut - 7 * weeks, real_cut
    out = []
    r = mmt(gd, pairs, cut, end)
    out.append({"method": "MMT", "placebo_att": r["att"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_perm"]})
    r = sdid(gd, cut, end, n_placebo=100)
    out.append({"method": "SDID", "placebo_att": r["att"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"]})
    if run_bsts:
        r = bsts(gd, cut, end)
        out.append({"method": "BSTS", "placebo_att": r["att"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "p_value": 2 * min(r["prob_negative"], 1 - r["prob_negative"])})
    for o in out:
        o["outcome"] = gd.outcome
        o["covers_zero"] = bool(o["ci_low"] <= 0 <= o["ci_high"])
    return out


def placebo_in_space(gd: GeoData, real_att: dict, n_draws: int = 300, seed: int = 0) -> pd.DataFrame:
    """Re-run MMT-style (ratio-adjusted mean-control) and SDID after fake-treating control DMAs only."""
    rng = np.random.default_rng(seed)
    cut = gd.T - 28
    ctrl = np.where(~gd.treated)[0]
    n_tr = int(gd.treated.sum())
    rows = []
    for _ in range(n_draws):
        pick = rng.choice(ctrl, n_tr, replace=False)
        tr = np.zeros(len(gd.treated), bool); tr[pick] = True
        pool = ~gd.treated
        s = gd.split(cut, treated=tr, pool=pool)
        a, _, _ = sdid_att(s["y"], s["Yco"], cut, n_tr)
        rows.append({"placebo_att": a})
    df = pd.DataFrame(rows)
    for m, v in real_att.items():
        df[f"real_{m}"] = v
    return df
