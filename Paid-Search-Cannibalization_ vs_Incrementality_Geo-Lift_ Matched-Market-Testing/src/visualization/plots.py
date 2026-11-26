"""300-DPI static figures. Okabe-Ito colour-blind-safe palette; one hue per method across all charts."""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import path  # noqa: E402

log = logging.getLogger(__name__)
DPI = 300
COL = {"MMT": "#E69F00", "SDID": "#009E73", "BSTS": "#0072B2", "Truth": "#444444", "observed": "#D55E00",
       "Oracle (true tau)": "#444444", "Random": "#999999"}
ALGO_COL = {"S-Learner": "#999999", "T-Learner": "#56B4E9", "X-Learner": "#E69F00", "R-Learner": "#009E73",
            "DR-Learner": "#F0E442", "LinearDML": "#0072B2", "CausalForestDML": "#D55E00", "TARNet": "#CC79A7",
            "DragonNet": "#8B0A50"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 100})


def _save(fig, name):
    p = path("plots") / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_geolift(outcome: str = "direct_bookings", cut: int = 140):
    s = pd.read_csv(path("metrics") / "macro_series.csv")
    s = s[s.outcome == outcome]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    obs = s[s.method == "BSTS"]
    ax.plot(obs.day, obs.observed, color=COL["observed"], lw=1.6, label="Observed (holdout DMAs, mean per DMA-day)")
    for m in ("MMT", "SDID", "BSTS"):
        d = s[s.method == m].dropna(subset=["counterfactual"])
        if m == "SDID" or m == "MMT":
            d = d[d.day >= cut - 28]
        ax.plot(d.day, d.counterfactual, color=COL[m], lw=1.4, ls="--", label=f"{m} counterfactual")
        if m == "BSTS":
            d2 = d[d.day >= cut]
            ax.fill_between(d2.day, d2.cf_low, d2.cf_high, color=COL[m], alpha=0.2, label="BSTS 95% credible band")
    ax.axvline(cut, color="k", lw=1)
    ax.text(cut + 0.5, ax.get_ylim()[1] * 0.98, "pause starts", va="top", fontsize=9)
    ax.set_xlabel("Day of experiment")
    ax.set_ylabel(outcome.replace("_", " "))
    ax.set_title(f"Geo-lift: {outcome.replace('_', ' ')} in paused DMAs vs counterfactual")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, f"geolift_{outcome}.png")


def fig_estimator_forest():
    res = json.loads((path("metrics") / "macro_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (oc, ttl) in zip(axes, [("direct_bookings", "Direct bookings"), ("ota_bookings", "OTA bookings")]):
        for i, m in enumerate(("MMT", "SDID", "BSTS")):
            r = res[oc][m]
            ax.errorbar(r["att"], i, xerr=[[r["att"] - r["ci_low"]], [r["ci_high"] - r["att"]]], fmt="o", color=COL[m], capsize=4)
            ax.text(r["ci_high"], i + 0.18, f"{r['att']:.2f}", fontsize=8, color=COL[m])
        ax.axvline(res[oc]["truth"], color=COL["Truth"], ls=":", lw=1.5, label="Ground truth")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(range(3)); ax.set_yticklabels(("MMT", "SDID", "BSTS")); ax.invert_yaxis()
        ax.set_title(f"ATT of pausing: {ttl}"); ax.set_xlabel("bookings per DMA-day"); ax.legend(frameon=False, fontsize=8)
    return _save(fig, "att_estimators.png")


def fig_qini(target: str = "direct"):
    q = pd.read_csv(path("metrics") / f"qini_curves_{target}.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in q.columns:
        if c in ALGO_COL:
            ax.plot(q.index, q[c], color=ALGO_COL[c], lw=1.4, label=c)
    ax.plot(q.index, q["Oracle (true tau)"], color="#444444", lw=2, ls="--", label="Oracle (true tau)")
    ax.plot(q.index, q["Random"], color="#999999", lw=1, ls=":", label="Random")
    ax.set_xlabel("Fraction of DMA-days targeted (largest predicted pause effect first)")
    ax.set_ylabel("Cumulative Qini gain")
    ax.set_title(f"Qini curves - {target} bookings lift (out-of-fold, DMA-grouped)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, f"qini_{target}.png")


def fig_leaderboard(target: str = "direct"):
    lb = pd.read_csv(path("metrics") / f"cate_leaderboard_{target}.csv", index_col=0).sort_values("pehe_norm")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = [ALGO_COL.get(i, "#BBBBBB") for i in lb.index]
    ax.barh(lb.index, lb.pehe_norm, color=cols)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.text(1.02, -0.4, "= no better than predicting the true average effect", fontsize=8, va="bottom")
    ax.invert_yaxis(); ax.set_xlabel("PEHE / sd(true CATE)  (lower is better)")
    ax.set_title(f"CATE leaderboard - {target} bookings lift")
    return _save(fig, f"leaderboard_{target}.png")


def fig_heterogeneity():
    m = path("metrics")
    dma = pd.read_csv(m / "dma_table.csv")
    oof = pd.read_csv(m / "cate_oof_predictions_direct.csv")
    lb = pd.read_csv(m / "cate_leaderboard_direct.csv", index_col=0)
    top = [a for a in lb.index if f"tau_hat_{a}" in oof.columns][:3]
    oof["ens"] = oof[[f"tau_hat_{a}" for a in top]].mean(axis=1)
    g = oof.groupby("dma_id")[["ens", "tau_true"]].mean().join(dma.set_index("dma_id")[["tier", "ota_pressure", "organic_rank", "T"]])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(g.tau_true * 100, g.ens * 100, c=g.tier.map({1: "#D55E00", 2: "#0072B2", 3: "#009E73"}), s=35)
    lim = [min(g.tau_true.min(), g.ens.min()) * 100 - 1, max(g.tau_true.max(), g.ens.max()) * 100 + 1]
    axes[0].plot(lim, lim, "k--", lw=1); axes[0].set_xlabel("True lift of pausing (% of baseline direct bookings)")
    axes[0].set_ylabel(f"Predicted (mean of {', '.join(top)})"); axes[0].set_title("Per-DMA CATE: predicted vs truth")
    for t, c in {1: "#D55E00", 2: "#0072B2", 3: "#009E73"}.items():
        axes[0].scatter([], [], c=c, label=f"Tier {t}")
    axes[0].legend(frameon=False, fontsize=8)
    for ax, v, ttl in ((axes[1], "ota_pressure", "OTA auction pressure"), (axes[2], "organic_rank", "Organic SERP rank (1 = top)")):
        ax.scatter(g[v], g.tau_true * 100, s=28, color="#444444", label="True")
        ax.scatter(g[v], g.ens * 100, s=28, color="#D55E00", marker="x", label="Predicted")
        ax.set_xlabel(ttl); ax.set_ylabel("Direct-booking lift of pausing (%)"); ax.set_title(f"Heterogeneity by {ttl.split(' (')[0].lower()}")
        ax.legend(frameon=False, fontsize=8)
    return _save(fig, "cate_heterogeneity.png")


def fig_iroas_waterfall(method: str = "SDID"):
    e = json.loads((path("metrics") / "programme_economics.json").read_text())[method]
    g_d = e["gbv_per_direct_booking"]
    last = e["lastclick_bookings"] * g_d
    inc = e["incremental_direct_gbv_usd"]
    ota_net = e["ota_leakage_bookings_when_paused"] * e["gbv_per_ota_booking"] * (1 - e["commission_rate"])
    spend = e["branded_spend_avoided_usd"]
    steps = [("Last-click\nattributed GBV", last, "total"), ("Cannibalised\n(would book anyway)", -(last - inc), "delta"),
             ("Incremental\ndirect GBV", inc, "total"), ("OTA margin\nrecaptured if paused", -ota_net, "delta"),
             ("Branded spend", -spend, "delta"), ("Net value of\nkeeping SEM", inc - ota_net - spend, "total")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    run = 0
    for i, (lab, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, color="#0072B2" if v >= 0 else "#D55E00"); run = v
            ax.text(i, v, f"${v/1e3:,.0f}k", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        else:
            ax.bar(i, v, bottom=run, color="#D55E00" if v < 0 else "#009E73");
            ax.text(i, run + v / 2, f"-${abs(v)/1e3:,.0f}k" if v < 0 else f"+${v/1e3:,.0f}k", ha="center", va="center", color="white", fontsize=9)
            run += v
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(top=last * 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1e3:,.0f}k"))
    ax.set_ylabel("USD over the 4-week holdout (10 DMAs)")
    ax.set_title(f"iROAS {e['iroas']:.1f}x vs last-click ROAS {e['last_click_roas']:.1f}x  ({method}; cannibalisation {e['cannibalization_rate']:.0%})", pad=14, fontsize=11)
    return _save(fig, "iroas_waterfall.png")


def fig_balance():
    b = pd.read_csv(path("metrics") / "balance_table.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(b))
    ax.scatter(b.smd_all_controls, y, color="#999999", label="All 40 controls", s=40)
    ax.scatter(b.smd_matched, y, color="#009E73", label="Matched controls", s=40)
    ax.axvline(0, color="k", lw=0.6); ax.axvspan(-0.1, 0.1, color="#009E73", alpha=0.1)
    ax.set_yticks(y); ax.set_yticklabels(b.covariate); ax.invert_yaxis()
    ax.set_xlabel("Standardised mean difference (holdout - control)"); ax.set_title("Covariate balance (shaded: |SMD| < 0.1)")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "balance_smd.png")


def fig_placebo():
    m = path("metrics")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sp = pd.read_csv(m / "placebo_space_direct_bookings.csv")
    axes[0].hist(sp.placebo_att, bins=30, color="#BBBBBB")
    axes[0].axvline(sp.real_SDID.iloc[0], color=COL["SDID"], lw=2, label=f"Real SDID ATT {sp.real_SDID.iloc[0]:.2f}")
    axes[0].set_title("Placebo-in-space (SDID, fake holdouts among controls)"); axes[0].set_xlabel("placebo ATT, bookings per DMA-day"); axes[0].legend(frameon=False, fontsize=8)
    pt = pd.read_csv(m / "placebo_in_time.csv")
    pt = pt[pt.outcome == "direct_bookings"]
    for i, r in enumerate(pt.itertuples()):
        axes[1].errorbar(r.placebo_att, i, xerr=[[r.placebo_att - r.ci_low], [r.ci_high - r.placebo_att]], fmt="o", color=COL[r.method], capsize=4)
    axes[1].axvline(0, color="k", lw=0.8); axes[1].set_yticks(range(len(pt))); axes[1].set_yticklabels(pt.method); axes[1].invert_yaxis()
    axes[1].set_title("Placebo-in-time (fake pause 4 weeks early, pre-data only)"); axes[1].set_xlabel("placebo ATT")
    return _save(fig, "placebo_tests.png")


def fig_montecarlo():
    mc = pd.read_csv(path("metrics") / "macro_montecarlo.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(("MMT", "SDID", "BSTS")):
        e = mc[mc.method == m].err
        ax.scatter(np.full(len(e), i) + np.random.default_rng(0).uniform(-0.1, 0.1, len(e)), e, color=COL[m], alpha=0.8)
        ax.hlines(e.mean(), i - 0.25, i + 0.25, color=COL[m], lw=2)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(["MMT", "SDID", "BSTS"])
    ax.set_ylabel("Estimate - truth (bookings per DMA-day)"); ax.set_title("Monte-Carlo error across fresh experiments (bar = mean bias)")
    return _save(fig, "montecarlo_error.png")


def fig_policy():
    pol = pd.read_csv(path("metrics") / "dma_policy_recommendations.csv")
    pol = pol.sort_values("true_ncm_change_if_paused_day")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pol))
    ax.bar(x, pol.true_ncm_change_if_paused_day, color=np.where(pol.true_ncm_change_if_paused_day > 0, "#009E73", "#D55E00"), alpha=0.85, label="True effect of pausing")
    ax.scatter(x, pol.pred_ncm_change_if_paused_day, color="k", s=14, zorder=3, label="Model-predicted (OOF ensemble)")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(pol.dma, rotation=90, fontsize=7)
    ax.set_ylabel("Change in net contribution margin, $/day if paused")
    ax.set_title("Per-DMA policy: pause when the (green) effect is positive"); ax.legend(frameon=False, fontsize=8)
    return _save(fig, "policy_by_dma.png")


def make_all():
    out = [fig_geolift("direct_bookings"), fig_geolift("ota_bookings"), fig_estimator_forest(), fig_qini("direct"),
           fig_qini("ota"), fig_leaderboard("direct"), fig_leaderboard("ota"), fig_heterogeneity(),
           fig_iroas_waterfall("SDID"), fig_balance(), fig_placebo(), fig_montecarlo(), fig_policy()]
    log.info("wrote %d figures", len(out))
    return out


def verify_outputs() -> bool:
    from PIL import Image
    ok = True
    for p in sorted(path("plots").glob("*.png")):
        im = Image.open(p)
        dpi = im.info.get("dpi", (0, 0))[0]
        if p.stat().st_size == 0 or round(dpi) < 299:
            log.error("bad figure %s dpi=%s", p.name, dpi); ok = False
    need = ["cate_leaderboard_direct.csv", "cate_leaderboard_ota.csv", "macro_results.json", "dma_policy_recommendations.csv",
            "placebo_in_time.csv", "programme_economics.json", "policy_value.json"]
    for n in need:
        if not (path("metrics") / n).exists():
            log.error("missing metric %s", n); ok = False
    log.info("verify: %s", "OK" if ok else "FAILED")
    return ok
