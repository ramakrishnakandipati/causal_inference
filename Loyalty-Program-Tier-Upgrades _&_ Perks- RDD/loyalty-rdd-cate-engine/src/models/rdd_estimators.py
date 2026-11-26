"""Sharp / Fuzzy RDD estimators and the validity (falsification) suite.

All estimators are weighted local polynomial fits implemented in numpy so that bandwidth
sweeps are cheap; ``rdrobust`` (CCT) and ``linearmodels`` are used as independent checks.

Convention: units with R >= cutoff are eligible (Z=1). The running variable is discrete
(whole nights), so a jump is identified from the nearest support points on each side.
"""
from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

KERNELS = ("triangular", "epanechnikov", "uniform")


# --------------------------------------------------------------------------- kernels
def kernel_weights(r: np.ndarray, c: float, h: float, kernel: str = "triangular") -> np.ndarray:
    """K((r-c)/h), zero outside the window."""
    u = np.abs((np.asarray(r, float) - c) / h)
    inside = u <= 1.0
    if kernel == "triangular":
        k = 1.0 - u
    elif kernel == "epanechnikov":
        k = 0.75 * (1.0 - u ** 2)
    elif kernel == "uniform":
        k = np.full_like(u, 0.5)
    else:
        raise ValueError(f"unknown kernel {kernel}")
    return np.where(inside, k, 0.0)


# --------------------------------------------------------------------------- WLS helpers
def _design(r: np.ndarray, c: float, p: int) -> np.ndarray:
    """[1, Z, rc, rc^2.., Z*rc, Z*rc^2..] with rc = r - c."""
    rc = r - c
    z = (r >= c).astype(float)
    cols = [np.ones_like(rc), z] + [rc ** k for k in range(1, p + 1)] + [z * rc ** k for k in range(1, p + 1)]
    return np.column_stack(cols)


def _sandwich(X: np.ndarray, w: np.ndarray, resid: np.ndarray, bread_inv: np.ndarray) -> np.ndarray:
    n, k = X.shape
    meat = (X * (w * resid)[:, None]).T @ (X * (w * resid)[:, None])
    return bread_inv @ meat @ bread_inv * n / max(n - k, 1)


def _enough_support(r: np.ndarray, w: np.ndarray, c: float, p: int) -> bool:
    left = np.unique(r[(w > 0) & (r < c)])
    right = np.unique(r[(w > 0) & (r >= c)])
    return len(left) >= p + 1 and len(right) >= p + 1


def _result(est: float, se: float, n_l: int, n_r: int, h: float, extra: Optional[dict] = None) -> dict:
    if not np.isfinite(se) or se <= 0:
        z, pv, lo, hi = np.nan, np.nan, np.nan, np.nan
    else:
        z = est / se
        pv = float(2 * (1 - stats.norm.cdf(abs(z))))
        lo, hi = est - 1.96 * se, est + 1.96 * se
    out = {"estimate": float(est), "se": float(se), "z": float(z), "p_value": pv,
           "ci_low": float(lo), "ci_high": float(hi), "n_left": int(n_l), "n_right": int(n_r), "h": float(h)}
    if extra:
        out.update(extra)
    return out


def _nan_result(h: float) -> dict:
    return _result(np.nan, np.nan, 0, 0, h, {"note": "insufficient support points in window"})


# --------------------------------------------------------------------------- sharp RDD
def sharp_rdd(y, r, c: float, h: float, p: int = 1, kernel: str = "triangular") -> dict:
    """Jump in E[y|r] at c (sharp RDD, or the ITT / reduced form in a fuzzy design)."""
    y = np.asarray(y, float); r = np.asarray(r, float)
    w = kernel_weights(r, c, h, kernel)
    keep = w > 0
    if not _enough_support(r, w, c, p):
        return _nan_result(h)
    X, yy, ww = _design(r[keep], c, p), y[keep], w[keep]
    XtW = X.T * ww
    bread_inv = np.linalg.pinv(XtW @ X)
    beta = bread_inv @ (XtW @ yy)
    cov = _sandwich(X, ww, yy - X @ beta, bread_inv)
    return _result(beta[1], np.sqrt(cov[1, 1]), int((r[keep] < c).sum()), int((r[keep] >= c).sum()), h)


# --------------------------------------------------------------------------- fuzzy RDD (2SLS)
def fuzzy_rdd(y, t, r, c: float, h: float, p: int = 1, kernel: str = "triangular") -> dict:
    """2SLS with Z = 1(R >= c) instrumenting actual tier attainment T inside the window."""
    y = np.asarray(y, float); t = np.asarray(t, float); r = np.asarray(r, float)
    w = kernel_weights(r, c, h, kernel)
    keep = w > 0
    if not _enough_support(r, w, c, p):
        return _nan_result(h)
    D = _design(r[keep], c, p)                   # columns: 1, Z, rc.., Z*rc..
    ww, yy, tt = w[keep], y[keep], t[keep]
    S = D.copy(); S[:, 1] = tt                    # structural regressors: T replaces Z
    Q = D                                         # instruments: Z plus the exogenous terms
    QtW = Q.T * ww
    Pi = np.linalg.pinv(QtW @ Q) @ (QtW @ S)
    S_hat = Q @ Pi
    SW = S_hat.T * ww
    bread_inv = np.linalg.pinv(SW @ S_hat)
    beta = bread_inv @ (SW @ yy)
    resid = yy - S @ beta
    cov = _sandwich(S_hat, ww, resid, bread_inv)
    fs = sharp_rdd(tt, r[keep], c, h, p, kernel)  # first-stage jump (same window)
    # first-stage F for the single instrument
    first_stage_f = float((fs["estimate"] / fs["se"]) ** 2) if np.isfinite(fs["se"]) and fs["se"] > 0 else np.nan
    return _result(beta[1], np.sqrt(cov[1, 1]), int((r[keep] < c).sum()), int((r[keep] >= c).sum()), h,
                   {"first_stage_jump": fs["estimate"], "first_stage_f": first_stage_f})


def rdd_estimate(df: pd.DataFrame, outcome: str, c: float, h: float, p: int = 1,
                 kernel: str = "triangular", fuzzy: bool = False, running: str = "qualifying_nights_t",
                 treatment: str = "gold_status") -> dict:
    r, y = df[running].to_numpy(float), df[outcome].to_numpy(float)
    if fuzzy:
        return fuzzy_rdd(y, df[treatment].to_numpy(float), r, c, h, p, kernel)
    return sharp_rdd(y, r, c, h, p, kernel)


# --------------------------------------------------------------------------- bandwidth selection
def ik_bandwidth(y, r, c: float) -> float:
    """Imbens-Kalyanaraman (2012) MSE-optimal bandwidth for a sharp design (triangular kernel)."""
    y = np.asarray(y, float); r = np.asarray(r, float)
    n = len(y)
    h1 = max(1.84 * r.std() * n ** (-0.2), 1.0)

    def side(mask_side, h):
        m = mask_side & (np.abs(r - c) <= h)
        return m

    left, right = r < c, r >= c
    ml, mr = side(left, h1), side(right, h1)
    if ml.sum() < 5 or mr.sum() < 5:
        return float(h1)
    f_c = (ml.sum() + mr.sum()) / (2 * h1 * n)
    var_l, var_r = y[ml].var(ddof=1), y[mr].var(ddof=1)
    # second derivatives from cubic fits on each half-sample (windows set by side medians)
    med_l, med_r = np.median(r[left]), np.median(r[right])
    def m2(mask, window_edge):
        sel = mask & (r >= window_edge[0]) & (r <= window_edge[1])
        if sel.sum() < 8 or np.unique(r[sel]).size < 4:
            return 0.0, max(sel.sum(), 1), abs(window_edge[1] - window_edge[0])
        coef = np.polyfit(r[sel] - c, y[sel], 3)
        return 2.0 * coef[1], int(sel.sum()), abs(window_edge[1] - window_edge[0])
    m2_l, n2_l, h2_l = m2(left, (med_l, c))
    m2_r, n2_r, h2_r = m2(right, (c, med_r))
    reg_l = 2160.0 * var_l / (max(n2_l, 1) * max(h2_l, 1e-6) ** 4)
    reg_r = 2160.0 * var_r / (max(n2_r, 1) * max(h2_r, 1e-6) ** 4)
    ck = 3.4375  # triangular kernel constant
    denom = f_c * ((m2_r - m2_l) ** 2 + reg_l + reg_r)
    h = ck * ((var_l + var_r) / max(denom, 1e-12)) ** 0.2 * n ** (-0.2)
    return float(np.clip(h, 1.0, 40.0))


def cct_bandwidth(y, r, c: float, t=None, kernel: str = "triangular", p: int = 1) -> float:
    """CCT MSE-optimal bandwidth via ``rdrobust.rdbwselect``; falls back to IK."""
    try:
        from rdrobust import rdbwselect
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = rdbwselect(y=np.asarray(y, float), x=np.asarray(r, float), c=c, p=p, kernel=kernel,
                             fuzzy=None if t is None else np.asarray(t, float), bwselect="mserd")
        bw = res.bws
        h = float(np.asarray(bw)[0, 0]) if hasattr(bw, "__len__") else float(bw)
        if np.isfinite(h) and h > 0:
            return h
    except Exception:
        pass
    return ik_bandwidth(y, r, c)


def rdrobust_check(y, r, c: float, h: float, t=None, kernel: str = "triangular", p: int = 1) -> dict:
    """Independent cross-check with CCT bias-corrected robust inference."""
    try:
        from rdrobust import rdrobust
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = rdrobust(y=np.asarray(y, float), x=np.asarray(r, float), c=c, p=p, kernel=kernel, h=h,
                           fuzzy=None if t is None else np.asarray(t, float))
        coef = np.asarray(res.coef).ravel(); se = np.asarray(res.se).ravel(); pv = np.asarray(res.pv).ravel()
        ci = np.asarray(res.ci)
        return {"conventional": float(coef[0]), "bias_corrected": float(coef[1]), "robust_se": float(se[2]),
                "robust_p": float(pv[2]), "robust_ci_low": float(ci[2, 0]), "robust_ci_high": float(ci[2, 1])}
    except Exception as exc:
        return {"error": str(exc)}


def linearmodels_fuzzy_check(df: pd.DataFrame, outcome: str, c: float, h: float, kernel: str = "triangular",
                             running: str = "qualifying_nights_t", treatment: str = "gold_status") -> dict:
    """Cross-check of the manual 2SLS against linearmodels.IV2SLS (p=1)."""
    from linearmodels.iv import IV2SLS
    r = df[running].to_numpy(float)
    w = kernel_weights(r, c, h, kernel)
    sub = df.loc[w > 0].copy()
    rc = sub[running] - c
    z = (sub[running] >= c).astype(float)
    exog = pd.DataFrame({"const": 1.0, "rc": rc, "rc_z": rc * z}, index=sub.index)
    res = IV2SLS(sub[outcome], exog, sub[treatment], z.rename("Z"), weights=w[w > 0]).fit(cov_type="robust")
    return {"estimate": float(res.params[treatment]), "se": float(res.std_errors[treatment])}


# --------------------------------------------------------------------------- sweeps
def bandwidth_sensitivity(df: pd.DataFrame, outcomes: Sequence[str], c: float, grid: Iterable[float],
                          kernels: Sequence[str] = ("triangular",), orders: Sequence[int] = (1,),
                          fuzzy: bool = False) -> pd.DataFrame:
    rows = []
    for y in outcomes:
        for k in kernels:
            for p in orders:
                for h in grid:
                    res = rdd_estimate(df, y, c, h, p, k, fuzzy)
                    rows.append({"outcome": y, "kernel": k, "poly_order": p, "bandwidth": h,
                                 "design": "fuzzy" if fuzzy else "sharp", **res})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- falsification suite
def mccrary_density_test(r, c: float, bandwidth: float = 10.0, kernel: str = "triangular", order: int = 2) -> dict:
    """McCrary (2008)-style test on whole-night counts.

    Counts per night are fitted by weighted local polynomial regression (quadratic by default, so
    density curvature does not masquerade as a jump) separately on each side;
    both fits are evaluated at the cutoff boundary (c - 0.5) and compared in logs.
    H0: the density is continuous at c (no bunching / manipulation).
    """
    r = np.asarray(r)
    vals, counts = np.unique(r.astype(int), return_counts=True)
    x = vals.astype(float)
    boundary = c - 0.5
    fits = {}
    for side, mask in (("left", x < c), ("right", x >= c)):
        xs, cs = x[mask], counts[mask].astype(float)
        w = kernel_weights(xs, c, bandwidth, kernel)
        sel = w > 0
        X = np.column_stack([(xs[sel] - boundary) ** k for k in range(order + 1)])
        W = w[sel]
        bread_inv = np.linalg.pinv((X.T * W) @ X)
        beta = bread_inv @ ((X.T * W) @ cs[sel])
        # Poisson sampling variance (var = count) is far more stable than residual-based HC
        # variances when only ~10 whole-night cells enter each side.
        meat = (X * (W ** 2 * cs[sel])[:, None]).T @ X
        cov = bread_inv @ meat @ bread_inv
        fits[side] = (beta[0], cov[0, 0])
    (fl, vl), (fr, vr) = fits["left"], fits["right"]
    theta = np.log(fr) - np.log(fl)
    se = np.sqrt(vl / fl ** 2 + vr / fr ** 2)
    z = theta / se
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"log_density_ratio": float(theta), "se": float(se), "z": float(z), "p_value": pval,
            "density_left": float(fl), "density_right": float(fr), "bandwidth": bandwidth,
            "reject_continuity_5pct": bool(pval < 0.05)}


def covariate_balance(df: pd.DataFrame, covariates: Sequence[str], c: float, h: float, p: int = 1,
                      kernel: str = "triangular", running: str = "qualifying_nights_t") -> pd.DataFrame:
    """Jump in each pre-treatment covariate at c, with Holm-adjusted p-values."""
    from statsmodels.stats.multitest import multipletests
    rows = [{"covariate": cov, **rdd_estimate(df, cov, c, h, p, kernel, False, running)} for cov in covariates]
    out = pd.DataFrame(rows)
    ok = out["p_value"].notna()
    out["p_holm"] = np.nan
    if ok.sum():
        out.loc[ok, "p_holm"] = multipletests(out.loc[ok, "p_value"], method="holm")[1]
    out["balanced"] = out["p_holm"] > 0.05
    return out


def placebo_cutoffs(df: pd.DataFrame, outcomes: Sequence[str], c: float, placebo: Sequence[float], h: float,
                    p: int = 1, kernel: str = "triangular", running: str = "qualifying_nights_t") -> pd.DataFrame:
    """Re-estimate the jump at false cutoffs. Each placebo only uses units on its own side of the
    true cutoff, so the real treatment change cannot contaminate it."""
    rows = []
    for cp in placebo:
        sub = df[df[running] < c] if cp < c else df[df[running] >= c]
        for y in outcomes:
            rows.append({"placebo_cutoff": cp, "outcome": y, **rdd_estimate(sub, y, cp, h, p, kernel, False, running)})
    out = pd.DataFrame(rows)
    out["null_not_rejected"] = out["p_value"] > 0.05
    return out


def donut_rdd(df: pd.DataFrame, outcomes: Sequence[str], c: float, h: float, donut: Sequence[int] = (19, 20),
              p: int = 1, kernel: str = "triangular", fuzzy: bool = False,
              running: str = "qualifying_nights_t") -> pd.DataFrame:
    sub = df[~df[running].isin(list(donut))]
    rows = []
    for y in outcomes:
        base = rdd_estimate(df, y, c, h, p, kernel, fuzzy, running)
        don = rdd_estimate(sub, y, c, h, p, kernel, fuzzy, running)
        rows.append({"outcome": y, "baseline": base["estimate"], "donut": don["estimate"],
                     "donut_se": don["se"], "donut_p": don["p_value"], "n_excluded": int(len(df) - len(sub))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- full suite
def run_rdd_suite(df: pd.DataFrame, cfg: dict) -> Dict[str, object]:
    """Run estimators + all four validity checks. ``cfg`` is the ``rdd`` section of config.yaml."""
    c = cfg["cutoff"]
    fuzzy = df.attrs.get("regime", "sharp") == "fuzzy"
    outcomes = cfg["outcomes"]
    kernel = cfg.get("default_kernel", "triangular")
    r = df["qualifying_nights_t"].to_numpy(float)
    primary = cfg.get("primary_outcome", outcomes[0])
    t_arg = df["gold_status"].to_numpy(float) if fuzzy else None
    h_cct = cct_bandwidth(df[primary].to_numpy(float), r, c, t_arg, kernel)
    h_ik = ik_bandwidth(df[primary].to_numpy(float), r, c)
    h = float(np.clip(h_cct, 3, 15))

    main = []
    for y in outcomes:
        res = rdd_estimate(df, y, c, h, 1, kernel, fuzzy)
        itt = sharp_rdd(df[y].to_numpy(float), r, c, h, 1, kernel)
        chk = rdrobust_check(df[y].to_numpy(float), r, c, h, t_arg, kernel)
        main.append({"outcome": y, "design": "fuzzy" if fuzzy else "sharp", **res,
                     "itt_estimate": itt["estimate"], "rdrobust_conventional": chk.get("conventional"),
                     "rdrobust_bias_corrected": chk.get("bias_corrected"), "rdrobust_robust_p": chk.get("robust_p")})
    return {
        "bandwidths": {"cct": float(h_cct), "ik": float(h_ik), "used": h},
        "main": pd.DataFrame(main),
        "sensitivity": bandwidth_sensitivity(df, outcomes, c, cfg["bandwidth_grid"], cfg["kernels"],
                                             cfg["polynomial_orders"], fuzzy),
        "mccrary": mccrary_density_test(r, c),
        "balance": covariate_balance(df, [v for v in df.columns if v in _COVS], c, h, 1, kernel),
        "placebo": placebo_cutoffs(df, outcomes, c, cfg["placebo_cutoffs"], max(h, 5), 1, kernel),
        "donut": donut_rdd(df, outcomes, c, h, cfg["donut"], 1, kernel, fuzzy),
    }


_COVS = ["historical_adr", "business_travel_ratio", "lead_time_avg", "ancillary_spend_t",
         "direct_booking_ratio_t", "tenure_years", "co_brand_card_holder", "home_region", "age_group",
         "weekend_stay_ratio"]
