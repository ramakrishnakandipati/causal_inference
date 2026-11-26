"""Observational ATE/ATT: propensity, PSM, IPW/SIPW, AIPW, SMD balance, refutation."""
from __future__ import annotations

import logging
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from src.common.config import load_config
from src.common.hardware import get_device_report, xgb_params

log = logging.getLogger("propensity")
EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _xgb_cfg(fast: bool) -> dict:
    c = load_config()["learners"]
    return dict(c["xgb_fast"] if fast else c["xgb"])


# ------------------------------------------------------------------ propensity
def fit_propensity(X: pd.DataFrame, T: np.ndarray, fast: bool = False, seed: int = 42):
    """Cross-fitted XGBoost propensity + Platt calibration. Returns (e_oof, full_model, calibrator)."""
    import xgboost as xgb
    cfg = load_config()["propensity"]
    p = {**_xgb_cfg(fast), **xgb_params(), "random_state": seed}
    skf = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    raw = np.zeros(len(X))
    for tr, va in skf.split(X, T):
        m = xgb.XGBClassifier(**p).fit(X.iloc[tr], T[tr])
        raw[va] = m.predict_proba(X.iloc[va])[:, 1]
    cal = LogisticRegression(C=1e6).fit(_logit(raw).reshape(-1, 1), T)
    e = cal.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    full = xgb.XGBClassifier(**p).fit(X, T)
    lo, hi = cfg["clip"]
    return np.clip(e, lo, hi), full, cal


def common_support(e: np.ndarray, T: np.ndarray) -> np.ndarray:
    lo = max(e[T == 1].min(), e[T == 0].min())
    hi = min(e[T == 1].max(), e[T == 0].max())
    return (e >= lo) & (e <= hi)


# ------------------------------------------------------------------ balance
def smd(X: pd.DataFrame, T: np.ndarray, w: np.ndarray | None = None) -> pd.Series:
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    out = {}
    for c in X.columns:
        x = X[c].values.astype(float)
        m = []
        v = []
        for g in (1, 0):
            wg = w[T == g]
            if wg.sum() <= 0:
                m.append(np.nan); v.append(np.nan); continue
            mu = np.average(x[T == g], weights=wg)
            var = np.average((x[T == g] - mu) ** 2, weights=wg)
            m.append(mu); v.append(var)
        pooled = np.sqrt((v[0] + v[1]) / 2)
        out[c] = (m[0] - m[1]) / pooled if pooled > 1e-12 else 0.0
    return pd.Series(out)


# ------------------------------------------------------------------ matching
def _knn(a_ps, b_ps, k):
    """For each a find k nearest in b (1-D logit-ps). cuML if present, else sklearn."""
    if get_device_report()["cuml"]:
        try:
            from cuml.neighbors import NearestNeighbors as CuNN
            nn = CuNN(n_neighbors=k).fit(b_ps.reshape(-1, 1).astype(np.float32))
            d, i = nn.kneighbors(a_ps.reshape(-1, 1).astype(np.float32))
            return np.asarray(d), np.asarray(i)
        except Exception as e:  # pragma: no cover
            log.warning("cuML NN failed (%s); sklearn fallback", e)
    nn = NearestNeighbors(n_neighbors=k).fit(b_ps.reshape(-1, 1))
    return nn.kneighbors(a_ps.reshape(-1, 1))


def _impute(ps_from, ps_to, y_to, k, caliper):
    """Mean outcome of k nearest neighbours (within caliper) for each unit; NaN where none."""
    d, idx = _knn(ps_from, ps_to, k)
    ok = d <= caliper
    yv = np.where(ok, y_to[idx], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        est = np.nanmean(yv, axis=1)
    return est, ok, idx


def psm(e, T, y, k=1, caliper_sd=0.2):
    """Nearest-neighbour matching on logit(e) with replacement and caliper.
    Returns dict(ate, att, matched_frac, weights_att)."""
    ps = _logit(e)
    cal = caliper_sd * ps.std()
    t, c = np.where(T == 1)[0], np.where(T == 0)[0]
    y0_t, ok_t, idx_t = _impute(ps[t], ps[c], y[c], k, cal)       # counterfactual Y0 for treated
    y1_c, ok_c, _ = _impute(ps[c], ps[t], y[t], k, cal)           # counterfactual Y1 for controls
    keep_t = ~np.isnan(y0_t)
    keep_c = ~np.isnan(y1_c)
    att = float(np.mean(y[t][keep_t] - y0_t[keep_t]))
    ite_t = y[t][keep_t] - y0_t[keep_t]
    ite_c = y1_c[keep_c] - y[c][keep_c]
    ate = float(np.concatenate([ite_t, ite_c]).mean())
    w = np.zeros(len(T))
    w[t[keep_t]] = 1.0
    for r in np.where(keep_t)[0]:
        sel = idx_t[r][ok_t[r]]
        w[c[sel]] += 1.0 / max(len(sel), 1)
    return dict(ate=ate, att=att, matched_frac=float(keep_t.mean()), weights=w)


# ------------------------------------------------------------------ weighting
def _trim_w(w, q):
    return np.minimum(w, np.quantile(w, q))


def ipw(e, T, y, trim_q=0.99):
    w1 = _trim_w(1 / e, trim_q)
    w0 = _trim_w(1 / (1 - e), trim_q)
    ate = np.mean(T * w1 * y - (1 - T) * w0 * y)
    # ATT: treated weight 1, control weight e/(1-e)
    wc = _trim_w(e / (1 - e), trim_q)
    att = y[T == 1].mean() - np.sum(wc[T == 0] * y[T == 0]) / np.sum(wc[T == 0])
    return float(ate), float(att)


def sipw(e, T, y, trim_q=0.99):
    w1 = _trim_w(1 / e, trim_q) * T
    w0 = _trim_w(1 / (1 - e), trim_q) * (1 - T)
    ate = np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0)
    wc = _trim_w(e / (1 - e), trim_q) * (1 - T)
    att = y[T == 1].mean() - np.sum(wc * y) / np.sum(wc)
    return float(ate), float(att)


def att_ipw_weights(e, T, trim_q=0.99):
    w = np.where(T == 1, 1.0, _trim_w(e / (1 - e), trim_q))
    return w


def outcome_models_crossfit(X, T, y, binary: bool, fast=False, seed=42, folds=3):
    """Cross-fitted mu1(x), mu0(x) with GPU XGBoost."""
    import xgboost as xgb
    p = {**_xgb_cfg(fast), **xgb_params(), "random_state": seed}
    mu1 = np.zeros(len(X)); mu0 = np.zeros(len(X))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, T):
        for g, mu in ((1, mu1), (0, mu0)):
            idx = tr[T[tr] == g]
            if binary:
                m = xgb.XGBClassifier(**p).fit(X.iloc[idx], y[idx])
                mu[va] = m.predict_proba(X.iloc[va])[:, 1]
            else:
                m = xgb.XGBRegressor(**p).fit(X.iloc[idx], y[idx])
                mu[va] = m.predict(X.iloc[va])
    return mu1, mu0


def aipw(e, T, y, mu1, mu0):
    """Returns (ate, se_ate, att, se_att) with influence-function standard errors."""
    psi = mu1 - mu0 + T * (y - mu1) / e - (1 - T) * (y - mu0) / (1 - e)
    ate, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))
    n1 = T.sum()
    psi_att = (T * (y - mu0) - (1 - T) * e / (1 - e) * (y - mu0)) / (n1 / len(T))
    att, se_att = psi_att.mean(), psi_att.std(ddof=1) / np.sqrt(len(psi))
    return float(ate), float(se), float(att), float(se_att)


# ------------------------------------------------------------------ orchestration
def _boot_ci(fn, n, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(idx))
    a = np.array(vals)
    return np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0)


def estimate_all(df: pd.DataFrame, X: pd.DataFrame, e: np.ndarray, fast=False, seed=42) -> dict:
    """ATE/ATT for both outcomes under naive, PSM(1:1, 1:k), IPW, SIPW, AIPW with CIs and truth."""
    cfg = load_config()["propensity"]
    nb = cfg["n_boot_fast"] if fast else cfg["n_boot"]
    T = df["treatment"].values
    res = {"n": int(len(df)), "n_treated": int(T.sum()), "outcomes": {}}
    truth = {"ancillary": ("tau_anc",), "direct": ("tau_direct",)}
    for name, col, binary in (("ancillary", "y_ancillary", False), ("direct", "y_direct", True)):
        y = df[col].values.astype(float)
        tcol = truth[name][0]
        tau = df[tcol].values
        true_ate, true_att = float(tau.mean()), float(tau[T == 1].mean())
        mu1, mu0 = outcome_models_crossfit(X, T, y, binary, fast, seed, load_config()["learners"]["folds"])
        rows = {}
        naive = y[T == 1].mean() - y[T == 0].mean()
        se_n = np.sqrt(y[T == 1].var(ddof=1) / T.sum() + y[T == 0].var(ddof=1) / (len(T) - T.sum()))
        rows["Naive"] = dict(ate=naive, att=naive, ate_lo=naive - 1.96 * se_n, ate_hi=naive + 1.96 * se_n,
                             att_lo=naive - 1.96 * se_n, att_hi=naive + 1.96 * se_n)
        for label, k in (("PSM 1:1", 1), ("PSM 1:k", cfg["knn_k"])):
            r = psm(e, T, y, k, cfg["caliper_sd"])
            lo, hi = _boot_ci(lambda i: [psm(e[i], T[i], y[i], k, cfg["caliper_sd"])[q] for q in ("ate", "att")],
                              len(y), max(nb // 2, 10), seed)
            rows[label] = dict(ate=r["ate"], att=r["att"], ate_lo=lo[0], ate_hi=hi[0], att_lo=lo[1], att_hi=hi[1],
                               matched_frac=r["matched_frac"])
        for label, fn in (("IPW", ipw), ("SIPW (Hajek)", sipw)):
            a, b = fn(e, T, y, cfg["weight_trim_q"])
            lo, hi = _boot_ci(lambda i: list(fn(e[i], T[i], y[i], cfg["weight_trim_q"])), len(y), nb, seed)
            rows[label] = dict(ate=a, att=b, ate_lo=lo[0], ate_hi=hi[0], att_lo=lo[1], att_hi=hi[1])
        a, s, b, sb = aipw(e, T, y, mu1, mu0)
        rows["AIPW (DR)"] = dict(ate=a, att=b, ate_lo=a - 1.96 * s, ate_hi=a + 1.96 * s,
                                 att_lo=b - 1.96 * sb, att_hi=b + 1.96 * sb)
        for r in rows.values():
            r["ate_bias"] = r["ate"] - true_ate
            r["att_bias"] = r["att"] - true_att
        res["outcomes"][name] = dict(true_ate=true_ate, true_att=true_att, estimators={k: {a: float(v) for a, v in r.items()} for k, r in rows.items()})
    return res


def balance_table(X: pd.DataFrame, T: np.ndarray, e: np.ndarray, trim_q=0.99, caliper_sd=0.2) -> pd.DataFrame:
    """SMD before / after PSM(1:1) / after IPW (ATE weights) / after ATT-IPW."""
    w_ate = np.where(T == 1, _trim_w(1 / e, trim_q), _trim_w(1 / (1 - e), trim_q))
    pm = psm(e, T, np.zeros(len(T)), 1, caliper_sd)
    t = pd.DataFrame({
        "smd_before": smd(X, T),
        "smd_psm": smd(X, T, pm["weights"]),
        "smd_ipw": smd(X, T, w_ate),
        "smd_ipw_att": smd(X, T, att_ipw_weights(e, T, trim_q)),
    })
    t.index.name = "feature"
    return t


# ------------------------------------------------------------------ refutation
def _aipw_est(X, T, y, fast, seed, binary):
    e, _, _ = fit_propensity(X, T, fast, seed)
    mu1, mu0 = outcome_models_crossfit(X, T, y, binary, fast, seed, 2)
    return aipw(e, T, y, mu1, mu0)[0]


def refute(df: pd.DataFrame, X: pd.DataFrame, outcome="y_ancillary", fast=False, seed=42, n_sub=8000) -> dict:
    """Refutation battery on AIPW (built-in) + DoWhy (if installed). Returns JSON-safe dict."""
    binary = outcome == "y_direct"
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), min(n_sub, len(df)), replace=False)
    d, Xs = df.iloc[idx].reset_index(drop=True), X.iloc[idx].reset_index(drop=True)
    T, y = d["treatment"].values, d[outcome].values.astype(float)
    base = _aipw_est(Xs, T, y, True, seed, binary)
    out = {"outcome": outcome, "n_used": len(d), "baseline_aipw_ate": float(base), "builtin": {}, "dowhy": None}
    # random common cause
    Xr = Xs.copy(); Xr["random_cause"] = rng.normal(size=len(Xr))
    out["builtin"]["random_common_cause"] = dict(new_effect=float(_aipw_est(Xr, T, y, True, seed, binary)))
    # placebo treatment
    plc = [float(_aipw_est(Xs, rng.permutation(T), y, True, seed, binary)) for _ in range(3)]
    out["builtin"]["placebo_treatment"] = dict(new_effect=float(np.mean(plc)), draws=plc)
    # data subset
    sub = []
    for s in range(3):
        m = rng.random(len(d)) < 0.8
        sub.append(float(_aipw_est(Xs[m].reset_index(drop=True), T[m], y[m], True, seed + s, binary)))
    out["builtin"]["data_subset"] = dict(new_effect=float(np.mean(sub)), draws=sub)
    b = out["builtin"]
    b["random_common_cause"]["pass"] = abs(b["random_common_cause"]["new_effect"] - base) < 0.15 * abs(base) + 1e-9
    b["placebo_treatment"]["pass"] = abs(b["placebo_treatment"]["new_effect"]) < 0.25 * abs(base)
    b["data_subset"]["pass"] = abs(b["data_subset"]["new_effect"] - base) < 0.20 * abs(base) + 1e-9
    out["dowhy"] = _dowhy_refute(d, Xs, outcome, seed)
    return out


def _dowhy_refute(d, Xs, outcome, seed):
    try:
        from dowhy import CausalModel
    except Exception as e:
        return {"available": False, "reason": f"dowhy import failed: {type(e).__name__}"}
    try:
        keep = ["gdummy"]
        dd = (Xs - Xs.mean()) / (Xs.std() + 1e-9)   # scale: dowhy's logistic PS model is unregularised
        dd["treatment"] = d["treatment"].astype(bool).values
        dd["outcome"] = d[outcome].values.astype(float)
        common = list(Xs.columns)
        model = CausalModel(data=dd, treatment="treatment", outcome="outcome", common_causes=common)
        est_id = model.identify_effect(proceed_when_unidentifiable=True)
        est = model.estimate_effect(est_id, method_name="backdoor.linear_regression",
                                    )
        res = {"available": True, "estimate": float(est.value), "refuters": {}}
        for name, kw in (("random_common_cause", {}), ("placebo_treatment_refuter", dict(placebo_type="permute", num_simulations=5)),
                         ("data_subset_refuter", dict(subset_fraction=0.8, num_simulations=5))):
            try:
                r = model.refute_estimate(est_id, est, method_name=name, **kw)
                res["refuters"][name] = dict(new_effect=float(r.new_effect), p_value=float(r.refutation_result.get("p_value", np.nan)) if isinstance(getattr(r, "refutation_result", None), dict) else None)
            except Exception as e:
                res["refuters"][name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        return res
    except Exception as e:
        return {"available": False, "reason": f"dowhy run failed: {type(e).__name__}: {str(e)[:160]}"}
