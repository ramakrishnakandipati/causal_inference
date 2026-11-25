"""300-DPI causal evaluation & business-impact plots -> outputs/plots/."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import FEATURES, get_logger, load_config, resolve
from src.metrics import decile_table, qini_curve

log = get_logger()
DPI = 300
C = {"treated": "#0072B2", "control": "#E69F00", "pos": "#009E73", "neg": "#D55E00", "grey": "#6b7280", "ink": "#1f2937"}
FAMILY = {"S-Learner": ("#0072B2", "-"), "T-Learner": ("#56B4E9", "-"), "X-Learner": ("#005f8f", "-"), "R-Learner": ("#7fb3d5", "-"),
          "DR-Learner": ("#009E73", "--"), "LinearDML": ("#8fd0b4", "--"), "NonParamDML": ("#CC79A7", "--"),
          "CausalForestDML": ("#D55E00", "-"), "UpliftRF": ("#E69F00", "-."), "TARNet": ("#6b21a8", ":"), "DragonNet": ("#111827", ":")}


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.bbox": "tight"})


def _save(fig, name, cfg):
    out = resolve(cfg["paths"]["plots"]); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)
    log.info("saved %s", out / name)


def plot_overlap(cfg, logs):
    prop = pd.read_parquet(logs / "propensity_train.parquet")
    diag = json.load(open(logs / "propensity_diagnostics.json"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 41)
    ax[0].hist(prop.loc[prop["T"] == 1, "e_hat"], bins, alpha=0.6, color=C["treated"], label="Treated (T=1)", density=True)
    ax[0].hist(prop.loc[prop["T"] == 0, "e_hat"], bins, alpha=0.6, color=C["control"], label="Control (T=0)", density=True)
    ax[0].set(xlabel="Estimated propensity  e(x) = P(T=1|X)", ylabel="Density", title="Propensity overlap (out-of-fold)")
    ax[0].legend()
    smd = pd.DataFrame({"raw": diag["smd_raw"], "ipw": diag["smd_ipw"]}).sort_values("raw")
    y = np.arange(len(smd))
    ax[1].barh(y + 0.2, smd["raw"], 0.38, color=C["neg"], label="Unweighted")
    ax[1].barh(y - 0.2, smd["ipw"], 0.38, color=C["pos"], label="IPW-weighted")
    ax[1].axvline(0.1, color=C["ink"], ls="--", lw=1)
    ax[1].text(0.102, len(smd) - 0.6, "|SMD| = 0.1", fontsize=8)
    ax[1].set_yticks(y); ax[1].set_yticklabels(smd.index)
    ax[1].set(xlabel="|Standardized mean difference|", title="Covariate balance before / after IPW")
    ax[1].legend(loc="lower right")
    fig.suptitle("Selection bias diagnostics", fontsize=13, fontweight="bold")
    _save(fig, "01_propensity_overlap_balance.png", cfg)


def plot_qini(cfg, logs):
    te = pd.read_parquet(logs / "test_set.parquet"); preds = pd.read_parquet(logs / "test_predictions.parquet")
    w = pd.read_parquet(logs / "test_propensity.parquet")["w_ipw"].values
    dec = pd.read_parquet(logs / "test_decisions.parquet")
    y, t = te["Y_margin"].values, te["T"].values
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, (col, ls) in FAMILY.items():
        if name in preds:
            x, q = qini_curve(preds[name].values, y, t, w)
            ax.plot(x * 100, q, color=col, ls=ls, lw=1.8, label=name)
    x, q = qini_curve(dec["resp_prob"].values, y, t, w)
    ax.plot(x * 100, q, color=C["grey"], lw=2, label="Propensity-to-buy baseline")
    x, q = qini_curve(np.zeros(len(y)), y, t, w)
    ax.plot(x * 100, q, color="black", lw=1.2, ls="--", label="Random")
    x, q = qini_curve(te["tau_margin_true"].values, y, t, w)
    ax.plot(x * 100, q, color="#999999", lw=1.2, ls=(0, (1, 3)), label="Oracle (true tau)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set(xlabel="% of guests targeted (ranked by predicted CATE)", ylabel="Cumulative incremental margin per guest ($)",
           title="Qini curves - IPW-weighted test set, 11 CATE estimators vs baselines")
    ax.legend(ncol=2, fontsize=8, loc="best")
    _save(fig, "02_multi_model_qini_curves.png", cfg)


def plot_deciles(cfg, logs):
    te = pd.read_parquet(logs / "test_set.parquet"); dec = pd.read_parquet(logs / "test_decisions.parquet")
    w = pd.read_parquet(logs / "test_propensity.parquet")["w_ipw"].values
    best = json.load(open(logs / "summary.json"))["best_model"]
    dm = decile_table(dec["tau_hat"].values, te["Y_margin"].values, te["T"].values, w)
    dc = decile_table(dec["tau_hat"].values, te["Y_conv"].values, te["T"].values, w)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, d, lab, scale, fmt in [(ax[0], dm, "Observed incremental margin per guest ($)", 1, "${:.1f}"),
                                  (ax[1], dc, "Observed incremental conversion (pp)", 100, "{:.1f}")]:
        v = d["uplift"].values * scale
        a.bar(d["decile"], v, color=[C["pos"] if u >= 0 else C["neg"] for u in v])
        span = np.nanmax(np.abs(v))
        for xi, u in zip(d["decile"], v):
            a.text(xi, u + 0.03 * span * (1 if u >= 0 else -2.5), fmt.format(u), ha="center", fontsize=7.5)
        a.axhline(0, color="black", lw=0.8); a.set_xticks(range(1, 11))
        a.set(xlabel="Predicted-CATE decile (1 = highest)", ylabel=lab)
    ax[0].annotate("Decile 10: Sleeping Dogs /\nmargin-eroding discounts", xy=(10, dm["uplift"].iloc[-1]),
                   xytext=(6.5, dm["uplift"].min() * 0.85), arrowprops=dict(arrowstyle="->", color=C["neg"]),
                   color=C["neg"], fontsize=9)
    fig.suptitle(f"Uplift by predicted CATE decile - {best} (IPW-adjusted, test set)", fontsize=13, fontweight="bold")
    _save(fig, "03_uplift_by_decile.png", cfg)


def plot_shap(cfg, logs):
    import shap
    sv = pd.read_parquet(logs / "shap_values.parquet"); X = pd.read_parquet(logs / "shap_sample.parquet")
    best = json.load(open(logs / "shap_summary.json"))["model"]
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(sv.values, X[FEATURES], show=False, max_display=10)
    plt.title(f"Drivers of CATE heterogeneity - SHAP on {best} tau-hat(x) surrogate", fontsize=12, fontweight="bold")
    plt.xlabel("SHAP value (impact on predicted incremental margin, $)")
    _save(plt.gcf(), "04_cate_heterogeneity_shap.png", cfg)


def plot_training(cfg, logs):
    h = json.load(open(logs / "deep_training_history.json"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, name in zip(ax, ["TARNet", "DragonNet"]):
        if name not in h:
            continue
        e = np.arange(1, len(h[name]["train"]) + 1)
        a.plot(e, h[name]["train"], color=C["treated"], lw=2, label="Train (total loss)")
        a.plot(e, h[name]["val"], color=C["neg"], lw=2, label="Validation (total loss)")
        a.plot(e, h[name]["train_outcome"], color=C["treated"], lw=1, ls="--", label="Train (outcome MSE)")
        a.plot(e, h[name]["val_outcome"], color=C["neg"], lw=1, ls="--", label="Val (outcome MSE)")
        a.axvline(h[name]["best_epoch"], color=C["grey"], ls=":", lw=1)
        a.text(h[name]["best_epoch"], a.get_ylim()[1] * 0.97, " best epoch", fontsize=8, color=C["grey"], va="top")
        a.set(title=f"{name} - CUDA + AMP", xlabel="Epoch", ylabel="Loss (standardised margin)")
        a.legend(fontsize=8)
    _save(fig, "05_deep_uplift_training_curves.png", cfg)


def plot_policies(cfg, logs):
    pol = pd.read_csv(logs / "policy_simulation.csv")
    labels = [s.replace(" (", "\n(") for s in pol["policy"]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    v = pol["incremental_margin"]
    ax[0].bar(range(len(pol)), v, color=[C["pos"] if u >= 0 else C["neg"] for u in v])
    rng_ = max(abs(v.max()), abs(v.min()), 1)
    for i, (u, p) in enumerate(zip(v, pol["pct_targeted"])):
        ax[0].text(i, u + (0.02 if u >= 0 else -0.09) * rng_, f"${u:,.0f}\n({p:.0%} targeted)", ha="center", fontsize=8.5)
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_xticks(range(len(pol))); ax[0].set_xticklabels(labels, fontsize=8.5)
    ax[0].set(ylabel="Net incremental margin on test set ($)", title="Net incremental margin by CRM policy")
    ax[0].set_ylim(min(v.min() * 1.3, -1), max(v.max() * 1.35, 1))
    x = np.arange(len(pol))
    ax[1].bar(x - 0.27, pol["discount_cannibalization"], 0.27, color=C["neg"], label="Discount cannibalization ($ paid to would-have-booked guests)")
    ax[1].bar(x, pol["incremental_bookings"] * 100, 0.27, color=C["treated"], label="Incremental bookings (x100)")
    ax[1].bar(x + 0.27, pol["send_cost"], 0.27, color=C["grey"], label="Send cost ($)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8.5)
    ax[1].set(ylabel="$ / bookings x100", title="Cannibalization vs incremental volume")
    ax[1].legend(fontsize=8)
    fig.suptitle("CRM policy comparison - ground-truth counterfactual simulation", fontsize=13, fontweight="bold")
    _save(fig, "06_crm_policy_margin_waterfall.png", cfg)


def run(cfg=None):
    cfg = cfg or load_config(); _style()
    logs = resolve(cfg["paths"]["logs"])
    for f in (plot_overlap, plot_qini, plot_deciles, plot_shap, plot_training, plot_policies):
        try:
            f(cfg, logs)
        except Exception as ex:
            log.error("%s failed: %r", f.__name__, ex)


if __name__ == "__main__":
    run()
