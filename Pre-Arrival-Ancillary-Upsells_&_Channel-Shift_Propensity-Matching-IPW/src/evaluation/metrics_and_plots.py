"""Uplift/causal evaluation metrics (observational IPW-based + oracle) and 300-DPI plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import load_config, out_dir

GRID = np.linspace(0, 1, 101)
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------ observational metrics (no truth)
def ipw_scores(T, y, e):
    """Unbiased per-unit signal for the individual effect: E[score|X] = tau(X)."""
    return T * y / e - (1 - T) * y / (1 - e)


def uplift_curve(score, T, y, e):
    """Cumulative IPW-estimated incremental outcome (per capita) when targeting top-x by `score`."""
    order = np.argsort(-score, kind="stable")
    s = ipw_scores(T, y, e)[order]
    cum = np.cumsum(s) / len(s)
    n = len(s)
    ks = np.clip((GRID * n).astype(int), 0, n)
    g = np.concatenate([[0.0], cum])[ks]
    return g  # g[-1] == IPW ATE


def qini_auuc(score, T, y, e):
    g = uplift_curve(score, T, y, e)
    auuc = float(_trapz(g, GRID))
    qini = float(_trapz(g - GRID * g[-1], GRID))
    return qini, auuc, g


def decile_uplift(score, T, y, e, k=10):
    s = ipw_scores(T, y, e)
    q = pd.qcut(pd.Series(score).rank(method="first", ascending=False), k, labels=False)
    return pd.DataFrame({"decile": np.arange(1, k + 1), "ipw_uplift": [s[q == i].mean() for i in range(k)],
                         "n": [int((q == i).sum()) for i in range(k)]})


def policy_value(score, T, y, e, frac, unit_margin, cost):
    """Estimated net value ($/guest in population) of contacting the top `frac` by score."""
    g = uplift_curve(score, T, y, e)
    i = int(round(frac * (len(GRID) - 1)))
    return float(unit_margin * g[i] - cost * frac)


def policy_curve(score, T, y, e, unit_margin, cost):
    g = uplift_curve(score, T, y, e)
    return unit_margin * g - cost * GRID


# ------------------------------------------------------------------ oracle metrics
def pehe(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def true_policy_value(score, tau_true, frac, unit_margin, cost):
    n = len(score)
    k = int(round(frac * n))
    top = np.argsort(-score, kind="stable")[:k]
    return float((unit_margin * tau_true[top].sum() - cost * k) / n)


def leaderboard(preds: dict, df_test: pd.DataFrame, e_test: np.ndarray, outcome: str, meta: dict | None = None):
    """preds: name -> tau_hat on df_test. Returns (leaderboard DF, curves DF long)."""
    b = load_config()["business"]
    col, tcol = ("y_ancillary", "tau_anc") if outcome == "ancillary" else ("y_direct", "tau_direct")
    unit = b["ancillary_margin"] if outcome == "ancillary" else b["direct_booking_margin"]
    T, y, tau = df_test["treatment"].values, df_test[col].values.astype(float), df_test[tcol].values
    rows, curves = [], []
    ref = dict(oracle=tau)
    for name, p in list(preds.items()) + [("Oracle (true tau)", tau)]:
        q, a, g = qini_auuc(p, T, y, e_test)
        row = dict(model=name, qini=q, auuc=a,
                   pehe=pehe(p, tau) if name != "Oracle (true tau)" else 0.0,
                   ate_bias=float(np.mean(p) - tau.mean()),
                   corr_true=float(np.corrcoef(p, tau)[0, 1]) if np.std(p) > 0 else 0.0,
                   policy_value_ipw=policy_value(p, T, y, e_test, b["budget_frac"], unit, b["cost_per_offer"]),
                   policy_value_true=true_policy_value(p, tau, b["budget_frac"], unit, b["cost_per_offer"]))
        if meta and name in meta:
            row.update(meta[name])
        rows.append(row)
        curves.append(pd.DataFrame({"model": name, "x": GRID, "gain": g, "random": GRID * g[-1],
                                    "net_value": policy_curve(p, T, y, e_test, unit, b["cost_per_offer"])}))
    lb = pd.DataFrame(rows).sort_values("qini", ascending=False).reset_index(drop=True)
    lb.insert(1, "outcome", outcome)
    return lb, pd.concat(curves, ignore_index=True)


# ------------------------------------------------------------------ plots
def _save(fig, name):
    p = out_dir("plots") / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def love_plot(bal: pd.DataFrame, name="love_plot.png"):
    b = bal.copy()
    b = b.reindex(b["smd_before"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(b))
    ax.scatter(b["smd_before"].abs(), y, label="Unadjusted", color="#d62728", s=40)
    ax.scatter(b["smd_psm"].abs(), y, label="After PSM 1:1", color="#1f77b4", marker="s", s=36)
    ax.scatter(b["smd_ipw"].abs(), y, label="After IPW", color="#2ca02c", marker="^", s=40)
    ax.axvline(0.1, ls="--", color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(b.index, fontsize=8)
    ax.set_xlabel("|Standardized mean difference|"); ax.set_title("Love plot: covariate balance")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, name)


def overlap_plot(e, T, channel, name="propensity_overlap.png"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(0, 1, 41)
    axs[0].hist(e[T == 1], bins, alpha=0.6, label="Treated", color="#1f77b4", density=True)
    axs[0].hist(e[T == 0], bins, alpha=0.6, label="Control", color="#ff7f0e", density=True)
    axs[0].set_title("Propensity overlap (treated vs control)"); axs[0].set_xlabel("e(X)"); axs[0].legend()
    isota = np.isin(channel, ["OTA_Expedia", "OTA_Booking"])
    axs[1].hist(e[isota], bins, alpha=0.6, label="OTA bookings", color="#9467bd", density=True)
    axs[1].hist(e[~isota], bins, alpha=0.6, label="Direct / Corporate", color="#2ca02c", density=True)
    axs[1].set_title("Propensity by booking channel"); axs[1].set_xlabel("e(X)"); axs[1].legend()
    return _save(fig, name)


def ate_comparison_plot(ate_res: dict, name="ate_comparison.png"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (o, v) in zip(axs, ate_res["outcomes"].items()):
        names = list(v["estimators"].keys())
        est = [v["estimators"][n]["ate"] for n in names]
        lo = [v["estimators"][n]["ate"] - v["estimators"][n]["ate_lo"] for n in names]
        hi = [v["estimators"][n]["ate_hi"] - v["estimators"][n]["ate"] for n in names]
        ax.errorbar(est, range(len(names)), xerr=[lo, hi], fmt="o", color="#1f77b4", capsize=3)
        ax.axvline(v["true_ate"], color="k", ls="--", label=f"True ATE = {v['true_ate']:.3f}")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"ATE: {o}")
    return _save(fig, name)


def qini_plot(curves: pd.DataFrame, outcome: str, name=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, g in curves.groupby("model"):
        style = dict(color="k", ls="--", lw=2) if m.startswith("Oracle") else dict(lw=1.3)
        ax.plot(g["x"], g["gain"], label=m, **style)
    r = curves[curves.model == curves.model.iloc[0]]
    ax.plot(r["x"], r["random"], color="grey", ls=":", label="Random")
    ax.set_xlabel("Fraction of guests targeted"); ax.set_ylabel("Cumulative incremental outcome / guest (IPW)")
    ax.set_title(f"Qini / cumulative gain: {outcome}"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    return _save(fig, name or f"qini_curves_{outcome}.png")


def policy_plot(curves: pd.DataFrame, outcome: str, name=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m, g in curves.groupby("model"):
        ax.plot(g["x"], g["net_value"], label=m, lw=1.3)
    ax.axvline(load_config()["business"]["budget_frac"], color="k", ls=":")
    ax.set_xlabel("Budget: fraction of guests contacted"); ax.set_ylabel("Net value $/guest")
    ax.set_title(f"Policy value vs budget: {outcome}"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    return _save(fig, name or f"policy_roi_{outcome}.png")


def cate_dist_plot(preds: dict, tau, outcome: str, best: list[str]):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tau, 60, alpha=0.4, color="k", density=True, label="True tau")
    for n in best:
        ax.hist(preds[n], 60, histtype="step", density=True, label=n)
    ax.set_title(f"CATE distribution: {outcome}"); ax.legend(fontsize=7); ax.set_xlabel("tau(x)")
    return _save(fig, f"cate_distribution_{outcome}.png")


def decile_plot(dec: pd.DataFrame, model: str, outcome: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(dec["decile"], dec["ipw_uplift"], color="#1f77b4")
    ax.set_xlabel("Score decile (1 = highest predicted uplift)"); ax.set_ylabel("IPW-estimated uplift")
    ax.set_title(f"Uplift by decile: {model} ({outcome})")
    return _save(fig, f"uplift_by_decile_{outcome}.png")
