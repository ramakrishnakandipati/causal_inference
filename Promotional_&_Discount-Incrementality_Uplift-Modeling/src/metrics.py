"""Causal evaluation metrics: Qini / AUUC / Uplift@K / PEHE / policy value.

All curve metrics accept optional sample weights (e.g. inverse-propensity weights) so they remain valid on
observational data; on the randomized holdout weights are simply None.
"""
from __future__ import annotations

import numpy as np


def _wvec(w, n):
    return np.ones(n) if w is None else np.asarray(w, dtype=float)


def qini_curve(score, y, t, w=None, n_points: int = 101):
    """Cumulative incremental outcome per *population member* when targeting the top-x fraction by `score`.

    Q(x) = (mean_t(x) - mean_c(x)) * N(x) / N_total  (weighted); Q(1) = ATE
    Returns (x_fractions, q).
    """
    score, y, t = np.asarray(score), np.asarray(y, float), np.asarray(t)
    w = _wvec(w, len(y))
    order = np.argsort(-score, kind="mergesort")
    y, t, w = y[order], t[order], w[order]
    wt, wc = w * (t == 1), w * (t == 0)
    cum_yt, cum_yc = np.cumsum(y * wt), np.cumsum(y * wc)
    cum_nt, cum_nc = np.cumsum(wt), np.cumsum(wc)
    n = len(y)
    idx = np.unique(np.linspace(1, n, n_points).astype(int)) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        mt = np.where(cum_nt[idx] > 0, cum_yt[idx] / cum_nt[idx], 0.0)
        mc = np.where(cum_nc[idx] > 0, cum_yc[idx] / cum_nc[idx], 0.0)
    # incremental outcome per guest among the targeted top-x, scaled by the targeted share of the population
    q = (mt - mc) * (cum_nt[idx] + cum_nc[idx]) / w.sum()
    x = (idx + 1) / n
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], q])


def auuc(score, y, t, w=None) -> float:
    x, q = qini_curve(score, y, t, w)
    return float(np.trapezoid(q, x))


def random_auuc(y, t, w=None) -> float:
    """Area under the straight random-targeting line (0 -> overall ATE)."""
    x, q = qini_curve(np.zeros(len(y)), y, t, w)
    return float(0.5 * q[-1])


def normalized_qini(score, y, t, oracle_score, w=None) -> float:
    """(AUUC_model - AUUC_random) / (AUUC_oracle - AUUC_random); oracle ranks by ground-truth tau."""
    a_m = auuc(score, y, t, w)
    a_r = random_auuc(y, t, w)
    a_o = auuc(oracle_score, y, t, w)
    d = a_o - a_r
    return float((a_m - a_r) / d) if abs(d) > 1e-12 else float("nan")


def uplift_at_k(score, y, t, k: float, w=None) -> float:
    """Observed uplift (mean_treated - mean_control) within the top-k fraction ranked by score."""
    score, y, t = np.asarray(score), np.asarray(y, float), np.asarray(t)
    w = _wvec(w, len(y))
    n_top = max(int(round(k * len(y))), 2)
    top = np.argsort(-score, kind="mergesort")[:n_top]
    yt, tt, wt = y[top], t[top], w[top]
    mt, mc = tt == 1, tt == 0
    if mt.sum() == 0 or mc.sum() == 0:
        return float("nan")
    return float(np.average(yt[mt], weights=wt[mt]) - np.average(yt[mc], weights=wt[mc]))


def pehe(tau_hat, tau_true) -> float:
    d = np.asarray(tau_hat, float) - np.asarray(tau_true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def policy_value(tau_true, treat_mask, comm_cost: float = 0.0) -> dict:
    """Ground-truth expected incremental margin of a targeting policy (true tau, minus send cost)."""
    treat_mask = np.asarray(treat_mask, bool)
    gross = float(np.asarray(tau_true)[treat_mask].sum())
    cost = float(comm_cost * treat_mask.sum())
    return {"n_targeted": int(treat_mask.sum()), "incremental_margin": gross - cost, "gross": gross, "send_cost": cost}


def decile_table(score, y, t, w=None, n_bins: int = 10):
    """Observed uplift by predicted-CATE decile (decile 1 = highest predicted uplift)."""
    import pandas as pd
    score, y, t = np.asarray(score), np.asarray(y, float), np.asarray(t)
    w = _wvec(w, len(y))
    ranks = np.argsort(np.argsort(-score, kind="mergesort"), kind="mergesort")
    dec = np.minimum((ranks * n_bins) // len(y), n_bins - 1) + 1
    rows = []
    for d in range(1, n_bins + 1):
        m = dec == d
        mt, mc = m & (t == 1), m & (t == 0)
        up = (np.average(y[mt], weights=w[mt]) - np.average(y[mc], weights=w[mc])) if mt.any() and mc.any() else np.nan
        rows.append({"decile": d, "n": int(m.sum()), "uplift": up,
                     "mean_score": float(score[m].mean())})
    return pd.DataFrame(rows)
