"""Causal evaluation metrics: LATE tables, RDD diagnostics summaries, and uplift metrics.

The uplift metrics (AUUC, Qini, Uplift@K, policy value) work on the kernel-weighted local sample;
treated/control group means are weighted means.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------- RDD summaries
def late_table(suite: dict) -> pd.DataFrame:
    cols = ["outcome", "design", "estimate", "se", "ci_low", "ci_high", "p_value", "h", "n_left", "n_right"]
    return suite["main"][[c for c in cols if c in suite["main"].columns]].copy()


def bandwidth_sensitivity_table(suite: dict) -> pd.DataFrame:
    return suite["sensitivity"].pivot_table(index=["outcome", "kernel", "poly_order"], columns="bandwidth",
                                            values="estimate")


def diagnostics_summary(suite: dict) -> Dict[str, object]:
    bal, plc = suite["balance"], suite["placebo"]
    return {
        "mccrary_p_value": suite["mccrary"]["p_value"],
        "mccrary_passes": not suite["mccrary"]["reject_continuity_5pct"],
        "covariates_balanced": bool(bal["balanced"].all()),
        "min_holm_p": float(bal["p_holm"].min()),
        "placebo_all_null": bool(plc["null_not_rejected"].mean() >= 0.75),
        "placebo_rejection_share": float(1 - plc["null_not_rejected"].mean()),
        "bandwidths": suite["bandwidths"],
    }


# ----------------------------------------------------------------------------- CATE accuracy
def cate_accuracy(tau_hat: np.ndarray, tau_true: np.ndarray, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Only meaningful on synthetic data where the true individual effect is known."""
    w = np.ones_like(tau_hat) if w is None else w
    err = tau_hat - tau_true
    top_true = tau_true >= np.quantile(tau_true, 0.8)
    top_hat = tau_hat >= np.quantile(tau_hat, 0.8)
    return {
        "pehe_rmse": float(np.sqrt(np.average(err ** 2, weights=w))),
        "ate_hat": float(np.average(tau_hat, weights=w)),
        "ate_true": float(np.average(tau_true, weights=w)),
        "ate_bias": float(np.average(err, weights=w)),
        "pearson": float(np.corrcoef(tau_hat, tau_true)[0, 1]) if tau_hat.std() > 0 else 0.0,
        "spearman": float(stats.spearmanr(tau_hat, tau_true)[0]) if tau_hat.std() > 0 else 0.0,
        "top20_overlap": float((top_true & top_hat).sum() / max(top_true.sum(), 1)),
    }


# ----------------------------------------------------------------------------- uplift metrics
def _wmean(y, w):
    s = w.sum()
    return float((y * w).sum() / s) if s > 0 else 0.0


def uplift_curve(tau_hat, T, Y, w, n_points: int = 50) -> pd.DataFrame:
    """Gain curve: at top fraction q, gain(q) = q * (ybar_T - ybar_C) within the top-q group."""
    order = np.argsort(-tau_hat)
    T, Y, w = np.asarray(T)[order], np.asarray(Y)[order], np.asarray(w)[order]
    n = len(Y)
    rows = [{"fraction": 0.0, "uplift": 0.0, "gain": 0.0, "qini": 0.0}]
    cw = np.cumsum(w)
    for q in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(int(np.searchsorted(cw, q * cw[-1])) + 1, 2)
        t, y, ww = T[:k], Y[:k], w[:k]
        tm, cm = t == 1, t == 0
        if tm.sum() == 0 or cm.sum() == 0:
            up, qini = 0.0, 0.0
        else:
            up = _wmean(y[tm], ww[tm]) - _wmean(y[cm], ww[cm])
            # Radcliffe Qini: treated outcome minus control outcome rescaled to the treated group size
            qini = (float((y[tm] * ww[tm]).sum())
                    - float((y[cm] * ww[cm]).sum()) * ww[tm].sum() / ww[cm].sum()) / cw[-1]
        rows.append({"fraction": float(q), "uplift": up, "gain": up * q, "qini": qini})
    return pd.DataFrame(rows)


def uplift_metrics(tau_hat, T, Y, w, ks=(0.1, 0.3)) -> Dict[str, float]:
    curve = uplift_curve(tau_hat, T, Y, w)
    q, g, qi = curve["fraction"].to_numpy(), curve["gain"].to_numpy(), curve["qini"].to_numpy()
    auuc = float(np.trapz(g, q))
    out = {
        "auuc": auuc,
        "auuc_over_random": auuc - 0.5 * float(g[-1]),     # area between gain curve and random line
        "qini_coefficient": float(np.trapz(qi, q) - 0.5 * qi[-1]),
    }
    for k in ks:
        row = curve.iloc[(curve["fraction"] - k).abs().argmin()]
        out[f"uplift_at_{int(k * 100)}"] = float(row["uplift"])
    return out


def policy_value(tau_hat, scores, w, treat_fraction: Optional[float] = None, threshold: float = 0.0,
                 tau_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Value gain per guest of treating guests by the CATE ranking.

    ``scores`` are doubly robust pseudo-outcomes for the individual effect, so
    mean(pi * score) is an unbiased estimate of the average gain of policy pi over treating nobody.
    """
    pi = (tau_hat >= np.quantile(tau_hat, 1 - treat_fraction)).astype(float) if treat_fraction else \
         (tau_hat > threshold).astype(float)
    out = {"policy_treated_share": float(np.average(pi, weights=w)),
           "policy_value_dr": float(np.average(pi * scores, weights=w)),
           "treat_all_value_dr": float(np.average(scores, weights=w))}
    if tau_true is not None:
        best = (tau_true > 0).astype(float)
        out["policy_value_oracle_eval"] = float(np.average(pi * tau_true, weights=w))
        out["oracle_policy_value"] = float(np.average(best * tau_true, weights=w))
    return out


def r_loss(tau_hat, y_res, t_res, w) -> float:
    """Robinson residual loss: lower is better (usable without ground truth)."""
    return float(np.average((y_res - tau_hat * t_res) ** 2, weights=w))
