"""Five EDA charts (300 DPI) into outputs/visualizations/eda/.   Run: python -m src.pipelines.eda_visualizations"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from data.generate_hospitality_data import INSTRUMENTS, OUTCOME, TREATMENT  # noqa: E402
from src.utils import plotstyle as ps  # noqa: E402
from src.utils.gpu_utils import load_config, resolve  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("eda", "eda.log")


def _fit_slope(x, y):
    return np.polyfit(x, y, 1)


def main() -> None:
    cfg = load_config()
    ps.apply()
    out = resolve(cfg["paths"]["eda_plots"])
    df = pd.read_parquet(resolve(cfg["paths"]["data"]) / "hospitality_bookings.parquet")
    log.info("EDA on %s rows", len(df))
    s = df.sample(min(6000, len(df)), random_state=0)

    # 1 Simpson's paradox
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(s[TREATMENT], s[OUTCOME], s=5, alpha=0.25, color=ps.MUTED, linewidths=0)
    b = _fit_slope(df[TREATMENT], df[OUTCOME])
    xs = np.linspace(df[TREATMENT].quantile(.01), df[TREATMENT].quantile(.99), 50)
    ax.plot(xs, np.polyval(b, xs), color=ps.CAT[4], label=f"Pooled OLS slope = {b[0]:+.2f}")
    ax.set(xlabel="ln net ADR", ylabel="log demand", title="Pooled: price and demand rise together")
    ax.legend(loc="upper left")
    ax = axes[1]
    # partial out the observed confounders X (Frisch-Waugh) so the only remaining driver of the co-movement is U
    from data.generate_hospitality_data import FEATURES
    xm = np.column_stack([np.ones(len(df)), df[FEATURES].to_numpy(float)])
    res = {k: df[k].to_numpy() - xm @ np.linalg.lstsq(xm, df[k].to_numpy(), rcond=None)[0] for k in (TREATMENT, OUTCOME)}
    df["t_res"], df["y_res"] = res[TREATMENT], res[OUTCOME]
    nb = 10
    df["u_q"] = pd.qcut(df["demand_shock_u"], nb, labels=False)
    cols = sns.light_palette(ps.CAT[0], nb + 3)[3:]
    sl = []
    for c, (q, g) in zip(cols, df.groupby("u_q")):
        bb = _fit_slope(g["t_res"], g["y_res"])
        sl.append(bb[0])
        xx = np.linspace(g["t_res"].quantile(.02), g["t_res"].quantile(.98), 20)
        ax.plot(xx, np.polyval(bb, xx), color=c, linewidth=1.4)
    ax.plot([], [], color=ps.CAT[0], label=f"Within-U-decile slopes: mean {np.mean(sl):+.2f} (low U lightest, surge darkest)")
    ax.set(xlabel="ln net ADR (net of X)", ylabel="log demand (net of X)",
           title="Holding the demand shock U fixed: slopes turn negative")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("The endogeneity trap (Simpson's paradox)", fontsize=14, fontweight="semibold")
    fig.savefig(out / "01_endogeneity_trap_simpsons_paradox.png")
    plt.close(fig)

    # 2 first stage
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    names = {"competitor_room_outage_pct": "Competitor room outage (%)",
             "airline_seat_capacity_shock": "Airline seat shock (x10%)",
             "pricing_algo_rounding_threshold": "Rate-rounding trigger (0/1)"}
    for ax, z in zip(axes, INSTRUMENTS):
        if z == "pricing_algo_rounding_threshold":
            sns.violinplot(data=s, x=z, y=TREATMENT, ax=ax, color=ps.CAT[0], inner="quartile", linewidth=0.8)
        else:
            ax.scatter(s[z], s[TREATMENT], s=5, alpha=0.2, color=ps.CAT[0], linewidths=0)
            bz = _fit_slope(df[z], df[TREATMENT])
            xx = np.linspace(df[z].min(), df[z].max(), 20)
            ax.plot(xx, np.polyval(bz, xx), color=ps.CAT[1])
        r = df[z].corr(df[TREATMENT])
        ax.set(xlabel=names[z], title=f"corr(Z, ln ADR) = {r:+.2f}", ylabel="ln net ADR" if z == INSTRUMENTS[0] else "")
    fig.suptitle("Instrument relevance: each Z moves the offered rate", fontsize=14, fontweight="semibold")
    fig.savefig(out / "02_instrument_relevance_first_stage.png")
    plt.close(fig)

    # 3 distributions
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    feats = [("lead_time", "Lead time (days)"), ("los", "Length of stay"), ("seasonality", "Seasonality index"), ("adr", "Offered ADR ($)")]
    order = ["leisure", "corporate", "group"]
    for j, (f, lab) in enumerate(feats):
        sns.violinplot(data=s, x="segment", y=f, hue="segment", palette=ps.SEG_COLOR, ax=axes[0, j], inner="quartile",
                       linewidth=0.8, legend=False, order=order, hue_order=order)
        axes[0, j].set(title=f"{lab} by guest segment", xlabel="", ylabel="")
        sns.violinplot(data=s, x="property_tier", y=f, hue="property_tier", palette=ps.CAT[:3], ax=axes[1, j],
                       inner="quartile", linewidth=0.8, legend=False)
        axes[1, j].set(title=f"{lab} by property tier", xlabel="", ylabel="")
        axes[1, j].tick_params(axis="x", labelsize=8)
    fig.suptitle("Feature distributions by guest segment and property tier", fontsize=14, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out / "03_feature_distributions_by_segment.png")
    plt.close(fig)

    # 4 correlation heatmap
    cols = INSTRUMENTS + ["demand_shock_u", TREATMENT, OUTCOME, "lead_time", "los", "seasonality", "loyalty_tier", "hist_adr_sensitivity"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    ax.grid(False)
    ax.set_title("Relevance corr(Z, T) != 0   |   Exogeneity corr(Z, U) ~ 0")
    fig.savefig(out / "04_confounder_instrument_correlation_heatmap.png")
    plt.close(fig)

    # 5 elasticity landscape
    d = df.copy()
    d["lead_bin"] = pd.cut(d["lead_time"], [-1, 3, 7, 14, 30, 60, 90, 400], labels=["0-3", "4-7", "8-14", "15-30", "31-60", "61-90", "90+"])
    piv = d.pivot_table(index="loyalty_tier", columns="lead_bin", values="true_elasticity", aggfunc="mean", observed=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="Blues_r", ax=ax, cbar_kws={"label": "true price elasticity"})
    ax.grid(False)
    ax.set(xlabel="Lead time (days)", ylabel="Loyalty tier", title="Ground-truth heterogeneous elasticity: lead time x loyalty tier")
    fig.savefig(out / "05_true_heterogeneous_elasticity_landscape.png")
    plt.close(fig)
    log.info("Saved 5 EDA plots to %s", out)


if __name__ == "__main__":
    main()
