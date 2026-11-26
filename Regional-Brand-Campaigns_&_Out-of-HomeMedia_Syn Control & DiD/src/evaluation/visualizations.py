"""Eight publication-grade 300-DPI figures written to outputs/figures/.

Every plot function reads the metrics saved by the pipeline, so plotting is decoupled from training.
Colour roles (validated reference palette, first three slots): blue = synthetic-control family, orange =
DiD family, aqua = Bayesian / DML family; black dashed = ground truth (known because the panel is simulated).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.utils.config import ensure_dir, read_json, resolve  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

DPI = 300
INK, INK2, GRID, SURF = "#191919", "#666666", "#e6e6e6", "#ffffff"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#257de0", "#f47230", "#00b894", "#6a4ce2", "#9c9c9c"
FAMILY = {"TWFE DiD": ORANGE, "Event Study": ORANGE, "CS-DiD (DR)": ORANGE, "SCM": BLUE, "ASCM (Ridge)": BLUE,
          "ASCM (ElasticNet)": BLUE, "SDID": BLUE, "Bayesian SC": AQUA}
LEARNER_FAMILY = {"S-Learner": BLUE, "T-Learner": BLUE, "X-Learner": BLUE, "R-Learner": BLUE, "DR-Learner": BLUE,
                  "Causal Forest DML": AQUA, "Linear DML": AQUA, "Domain Adaptation": AQUA,
                  "TARNet": ORANGE, "CFRNet-DragonNet": ORANGE}
HEAD = "mia_all24"
PRIMARY = "room_nights_booked"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 8.5,
    })


def _save(fig, name: str, cfg: dict) -> None:
    out = ensure_dir(cfg["paths"]["figures"]) / name
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def _load(cfg: dict) -> dict:
    m = resolve(cfg["paths"]["metrics"])
    M: dict = {"panel": pd.read_parquet(resolve(cfg["panel"]["parquet_path"]))}
    M["panel"]["dma_code"] = M["panel"]["dma_code"].astype(str)
    for name, f in [("cf", "counterfactual_paths.parquet"), ("pgap", "placebo_gap_paths.parquet"),
                    ("cate_pred", "cate_predictions.parquet")]:
        if (m / f).exists():
            M[name] = pd.read_parquet(m / f)
    for name, f in [("summary", "panel_att_summary.csv"), ("es", "event_study.csv"), ("es_meta", "event_study_pretrend.csv"),
                    ("weights", "donor_weights.csv"), ("balance", "covariate_balance.csv"), ("placebo", "placebo_in_space.csv"),
                    ("placebo_p", "placebo_pvalues.csv"), ("mc", "panel_monte_carlo.csv"), ("cate", "cate_benchmark.csv"),
                    ("econ", "economics.csv")]:
        if (m / f).exists():
            M[name] = pd.read_csv(m / f)
    for name, f in [("drivers", "cate_drivers.json"), ("nn", "nn_training_history.json")]:
        if (m / f).exists():
            M[name] = read_json(m / f)
    return M


def _dates(M: dict) -> pd.Series:
    p = M["panel"]
    return p[p["dma_code"] == "MIA"].sort_values("day_index").set_index("day_index")["date"]


def _roll(x: np.ndarray, w: int = 7) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().to_numpy()


def _fortnight_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))


def _footer(fig, text: str, y: float = 0.01) -> None:
    fig.text(0.01, y, text, fontsize=8.3, color=INK2, va="bottom", ha="left")


def _launch(ax, dates, T0: int, label: bool = True) -> None:
    ax.axvline(dates.iloc[T0], color=INK2, lw=1.0, ls=(0, (4, 3)))
    if label:
        ax.text(dates.iloc[T0], ax.get_ylim()[1], "  campaign launch", va="top", ha="left", fontsize=8, color=INK2)


def _est(M: dict, est: str, outcome: str = PRIMARY, scen: str = HEAD) -> pd.DataFrame:
    c = M["cf"]
    return c[(c.scenario == scen) & (c.outcome == outcome) & (c.estimator == est)].sort_values("day_index")


# ----------------------------------------------------------------------------- 01
def fig01(cfg: dict, M: dict) -> None:
    p, T0 = M["panel"], cfg["panel"]["pre_days"]
    dates = _dates(M)
    wide = p.pivot(index="day_index", columns="dma_code", values=PRIMARY)
    idx = wide / wide.iloc[:T0].mean() * 100
    donors = [c for c in wide.columns if c != "MIA"]
    scm = _est(M, "SCM")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.05]})
    a.plot(dates, _roll(idx[donors].mean(axis=1).to_numpy()), color=GRAY, lw=2, label="Average of all 24 control DMAs")
    a.plot(dates, _roll(scm["y0_hat"].to_numpy() / wide["MIA"].iloc[:T0].mean() * 100), color=BLUE, lw=2, label="Synthetic control (SCM)")
    a.plot(dates, _roll(idx["MIA"].to_numpy()), color=INK, lw=2, label="Miami (observed)")
    a.set_title("Raw donor average is not a valid counterfactual")
    a.set_ylabel("Room nights, index (pre-period mean = 100), 7-day mean")
    _launch(a, dates, T0)
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    es = M["es"][(M["es"].scenario == HEAD) & (M["es"].outcome == PRIMARY)]
    meta = M["es_meta"][(M["es_meta"].scenario == HEAD) & (M["es_meta"].outcome == PRIMARY)].iloc[0]
    k = es["k"].to_numpy()
    b.axhline(0, color=INK2, lw=1)
    b.fill_between(k, es["beta"] - 1.96 * es["se_placebo"], es["beta"] + 1.96 * es["se_placebo"], color=ORANGE, alpha=0.12, lw=0)
    b.plot(k, es["beta"], color=ORANGE, lw=2, marker="o", ms=3.5, label="Event study, all-donor DiD (placebo 95% band)")
    d = _est(M, "ASCM (Ridge)")
    gap = (d["y_obs"] - d["y0_hat"]).to_numpy()
    kk = np.arange(len(gap)) - T0
    msk = (kk >= es["k"].min()) & (kk <= es["k"].max())
    b.plot(kk[msk], gap[msk], color=BLUE, lw=2, label="Gap to augmented synthetic control")
    b.plot(k[k >= 0], es["true_att"][k >= 0], color=INK, lw=1.6, ls=(0, (5, 3)), label="True daily effect (simulation truth)")
    b.axvline(0, color=INK2, lw=1.0, ls=(0, (4, 3)))
    b.set_xlabel("Days relative to campaign launch (k)")
    b.set_ylabel("Room nights / day vs. reference")
    b.set_title("Event study: pre-period leads and post-period dynamics")
    b.legend(loc="upper left", fontsize=8)
    b.text(0.985, 0.04, f"Pre-trend (leads k=-14..-2): RMS {meta.pretrend_rms:.0f} rn, slope {meta.pretrend_slope:+.1f} rn/day\n"
                        f"permutation p (RMS) = {meta.p_pretrend_perm:.2f}; reference k = -1",
           transform=b.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    fig.suptitle("Parallel trends: why Miami needs a synthetic comparison, not a plain donor average", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    _footer(fig, "Miami climbs into winter faster than the average control DMA, so a plain DiD would credit the campaign with the seasonal gap; "
                 "the synthetic control tracks Miami's own pre-period path.")
    _save(fig, "01_parallel_trends_event_study.png", cfg)


# ----------------------------------------------------------------------------- 02
def fig02(cfg: dict, M: dict) -> None:
    p, T0 = M["panel"], cfg["panel"]["pre_days"]
    dates = _dates(M)
    mia = p[p.dma_code == "MIA"].sort_values("day_index")
    asc, bay, scm = _est(M, "ASCM (Ridge)"), _est(M, "Bayesian SC"), _est(M, "SCM")
    T = len(mia)
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.32, wspace=0.18)
    a = fig.add_subplot(gs[0, :])
    a.plot(dates, _roll(mia["y0_true"].to_numpy()), color=GRAY, lw=1.8, ls=(0, (5, 3)), label="True counterfactual (simulation truth)")
    a.plot(dates, _roll(scm["y0_hat"].to_numpy()), color=ORANGE, lw=1.6, label="SCM counterfactual")
    a.plot(dates, _roll(asc["y0_hat"].to_numpy()), color=BLUE, lw=2, label="ASCM counterfactual")
    a.plot(dates, _roll(bay["y0_hat"].to_numpy()), color=AQUA, lw=1.6, label="Bayesian SC counterfactual (post: 95% band)")
    a.fill_between(dates.iloc[T0:], _roll(bay["y_obs"].to_numpy()[T0:] - bay["att_hi"].to_numpy()[T0:]),
                   _roll(bay["y_obs"].to_numpy()[T0:] - bay["att_lo"].to_numpy()[T0:]), color=AQUA, alpha=0.16, lw=0)
    a.plot(dates, _roll(mia["room_nights_booked"].to_numpy()), color=INK, lw=2.2, label="Miami observed")
    a.set_ylabel("Room nights booked per day (7-day mean)")
    a.set_title("Observed vs. counterfactual: the gap after launch is the campaign's incremental room nights")
    a.legend(loc="upper left", ncol=2)
    a.set_ylim(a.get_ylim()[0], a.get_ylim()[1] * 1.04)
    _launch(a, dates, T0)
    b = fig.add_subplot(gs[1, 0])
    d_post = dates.iloc[T0:]
    true_att = mia["true_att_it"].to_numpy()[T0:]
    b.fill_between(d_post, bay["att_lo"].to_numpy()[T0:], bay["att_hi"].to_numpy()[T0:], color=AQUA, alpha=0.16, lw=0)
    b.plot(d_post, bay["att_hat"].to_numpy()[T0:], color=AQUA, lw=1.6, label="Bayesian SC (95% band)")
    b.plot(d_post, asc["att_hat"].to_numpy()[T0:], color=BLUE, lw=2, label="ASCM")
    b.plot(d_post, true_att, color=INK, lw=1.6, ls=(0, (5, 3)), label="True effect")
    b.axhline(0, color=INK2, lw=1)
    b.set_ylabel("Incremental room nights / day")
    b.set_title("Daily effect (ATT)")
    b.legend(loc="lower center", ncol=3, fontsize=8)
    c = fig.add_subplot(gs[1, 1])
    cum_true = np.cumsum(true_att)
    cum_b = np.cumsum(bay["att_hat"].to_numpy()[T0:])
    cum_a = np.cumsum(asc["att_hat"].to_numpy()[T0:])
    summ = M["summary"]
    srow = summ[(summ.scenario == HEAD) & (summ.outcome == PRIMARY) & (summ.estimator == "Bayesian SC")].iloc[0]
    n = np.arange(1, len(cum_b) + 1)
    c.fill_between(d_post, srow.ci_lo * n, srow.ci_hi * n, color=AQUA, alpha=0.16, lw=0)
    c.plot(d_post, cum_b, color=AQUA, lw=1.8, label="Bayesian SC (95% CI on mean effect)")
    c.plot(d_post, cum_a, color=BLUE, lw=2, label="ASCM")
    c.plot(d_post, cum_true, color=INK, lw=1.6, ls=(0, (5, 3)), label="True cumulative effect")
    c.set_ylabel("Cumulative incremental room nights")
    c.set_title("Cumulative effect over the 60-day campaign")
    c.legend(loc="upper left", fontsize=8)
    c.text(d_post.iloc[-1], cum_true[-1], f"  {cum_true[-1]:,.0f}", va="center", fontsize=8.5, color=INK)
    for ax in (b, c):
        _fortnight_axis(ax)
    fig.suptitle("Synthetic control counterfactual for Miami (room nights)", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=0.96)
    _save(fig, "02_synthetic_control_counterfactual.png", cfg)


# ----------------------------------------------------------------------------- 03
def fig03(cfg: dict, M: dict) -> None:
    w = M["weights"]
    w = w[(w.scenario == HEAD) & (w.outcome == PRIMARY)]
    piv = w.pivot(index="donor", columns="estimator", values="weight")
    sd = w[w.estimator == "Bayesian SC"].set_index("donor")["weight_sd"]
    planted = cfg["dgp"]["planted"]["MIA"]
    order = piv["SCM"].sort_values(ascending=False).index.tolist()
    keep = [d for d in order if max(abs(piv.loc[d]).max(), planted.get(d, 0)) > 0.02][:12]
    keep = keep[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(keep))
    a.barh(y, piv.loc[keep, "Bayesian SC"], height=0.5, color="#d9e8fb", label="Bayesian SC posterior mean (bar) +/- 1 sd", zorder=2)
    a.errorbar(piv.loc[keep, "Bayesian SC"], y, xerr=sd.loc[keep], fmt="none", ecolor=AQUA, elinewidth=1.4, capsize=2.5, zorder=3)
    for est, col, mk, off in [("SCM", BLUE, "o", -0.14), ("ASCM (Ridge)", ORANGE, "s", 0.0), ("SDID", AQUA, "D", 0.14)]:
        a.scatter(piv.loc[keep, est], y + off, s=44, color=col, marker=mk, edgecolor=SURF, linewidth=1.5, zorder=4, label=est)
    a.scatter([planted.get(d, 0) for d in keep], y, s=70, marker="X", color=INK, edgecolor=SURF, linewidth=1.2, zorder=5,
              label="Planted mix (simulation truth)")
    a.axvline(0, color=INK2, lw=1)
    a.set_yticks(y, keep)
    a.set_xlabel("Donor weight")
    a.set_title("Donor weights (top donors, headline analysis)")
    a.legend(loc="lower right", fontsize=8)
    bal = M["balance"]
    bal = bal[(bal.scenario == HEAD) & (bal.outcome == PRIMARY)].iloc[::-1]
    yy = np.arange(len(bal))
    b.axvspan(-0.25, 0.25, color="#eef3fb", lw=0, zorder=0)
    b.axvline(0, color=INK2, lw=1)
    for yi, r in zip(yy, bal.itertuples()):
        b.plot([r.std_diff_synthetic, r.std_diff_donor_mean], [yi, yi], color=GRID, lw=2, zorder=1)
    b.scatter(bal["std_diff_donor_mean"], yy, s=52, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=3, label="vs. average of 24 donors")
    b.scatter(bal["std_diff_synthetic"], yy, s=52, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=4, label="vs. synthetic control")
    b.set_yticks(yy, [c.replace("_", " ") for c in bal["covariate"]])
    b.set_xlabel("Standardised difference in pre-period mean (treated - comparison)")
    b.set_title("Covariate balance (shaded: |d| < 0.25)")
    b.legend(loc="lower right", fontsize=8)
    fig.suptitle("Who Miami is compared with: donor weights and covariate balance", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _footer(fig, "Weights are only weakly identified: several donor mixes fit the pre-period equally well, so compare counterfactuals, not weights."
                 + "\nThe synthetic control matches the pre-period outcome path; covariates are shown as a diagnostic, not matched.")
    _save(fig, "03_donor_weights_and_covariate_balance.png", cfg)


# ----------------------------------------------------------------------------- 04
def fig04(cfg: dict, M: dict) -> None:
    T0 = cfg["panel"]["pre_days"]
    dates = _dates(M)
    pg = M["pgap"]
    est = "ASCM (Ridge)"
    g = pg[pg.estimator == est].sort_values("day_index")
    pl = M["placebo"]
    pl = pl[(pl.scenario == HEAD) & (pl.outcome == PRIMARY) & (pl.estimator == est)]
    keep = set(pl[pl.kept_pre_fit & (pl.unit != "TREATED")].unit)
    pre_t = float(pl.loc[pl.unit == "TREATED", "pre_rmspe"].iloc[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    units = [c for c in g.columns if c not in ("estimator", "day_index", "TREATED")]
    ymax = np.abs(g["TREATED"]).max() * 1.6
    for u in units:
        a.plot(dates, g[u], color=(GRAY if u in keep else "#d6d6d6"), lw=1.0 if u in keep else 0.8, zorder=1)
    a.plot(dates, g["TREATED"], color=BLUE, lw=2.4, zorder=3)
    a.axhline(0, color=INK2, lw=1)
    a.set_ylim(-ymax, ymax)
    _launch(a, dates, T0)
    a.set_ylabel("Gap: observed - counterfactual (room nights / day)")
    a.set_title(f"Placebo gaps: Miami vs. {len(units)} control DMAs treated as if exposed (ASCM)")
    a.legend(handles=[Line2D([0], [0], color=BLUE, lw=2.4, label="Miami (actual campaign)"),
                      Line2D([0], [0], color=GRAY, lw=1.2, label=f"Placebo DMAs with decent pre-fit ({len(keep)})"),
                      Line2D([0], [0], color="#d6d6d6", lw=1.0, label="Placebo DMAs with poor pre-fit (dropped in filtered p)")],
             loc="upper left", fontsize=8)
    pl = pl.sort_values("ratio", ascending=False).reset_index(drop=True)
    colors = [BLUE if u == "TREATED" else ("#b9b9b9" if k else "#dddddd") for u, k in zip(pl["unit"], pl["kept_pre_fit"])]
    b.bar(np.arange(len(pl)), pl["ratio"], color=colors, width=0.7, zorder=2)
    b.set_xticks(np.arange(len(pl)), ["MIA" if u == "TREATED" else u for u in pl["unit"]], rotation=90, fontsize=7.5)
    for tick, u in zip(b.get_xticklabels(), pl["unit"]):
        tick.set_fontweight("bold" if u == "TREATED" else "normal")
    b.set_ylabel("Post / pre RMSPE ratio")
    pp = M["placebo_p"]
    pp = pp[(pp.scenario == HEAD) & (pp.outcome == PRIMARY) & (pp.estimator == est)].iloc[0]
    b.set_title("Exact permutation test")
    b.text(0.97, 0.95, f"Miami ranks {int(pp.rank_of_treated)} of {int(pp.n_units)}\nexact p = {pp.p_exact_ratio:.3f}"
                        f"\n(pre-fit-filtered p = {pp.p_exact_ratio_filtered:.3f})\nsmallest attainable p = {pp.min_possible_p:.3f}",
           transform=b.transAxes, ha="right", va="top", fontsize=9, color=INK)
    fig.suptitle("In-space placebo permutation across all 24 control DMAs", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "04_in_space_placebo_permutation.png", cfg)


# ----------------------------------------------------------------------------- 05
def fig05(cfg: dict, M: dict) -> None:
    s = M["summary"]
    s = s[(s.scenario == HEAD) & (s.outcome == PRIMARY)]
    order = ["TWFE DiD", "Event Study", "CS-DiD (DR)", "SCM", "ASCM (Ridge)", "ASCM (ElasticNet)", "SDID", "Bayesian SC"]
    s = s.set_index("estimator").loc[[o for o in order if o in set(s.estimator)]]
    truth = float(s["att_true_mean"].iloc[0])
    has_mc = "mc" in M
    fig, axes = plt.subplots(1, 2 if has_mc else 1, figsize=(13.5 if has_mc else 8.5, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1]} if has_mc else None)
    a = axes[0] if has_mc else axes
    y = np.arange(len(s))[::-1]
    a.axvline(truth, color=INK, lw=1.6, ls=(0, (5, 3)), zorder=1)
    for yi, (name, r) in zip(y, s.iterrows()):
        col = FAMILY[name]
        a.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        a.scatter([r.att_mean], [yi], s=70, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        a.text(max(r.ci_hi, r.att_mean) + 8, yi, f"{r.att_mean:,.0f}   [{r.ci_lo:,.0f}, {r.ci_hi:,.0f}]", va="center", fontsize=8.5, color=INK)
    a.set_yticks(y, list(s.index))
    a.set_xlabel("Average incremental room nights per day (95% interval)")
    a.set_title("Headline analysis: one panel, seven estimators (+ an elastic-net ASCM variant)")
    a.set_xlim(min(s.ci_lo.min(), 0) - 20, s.ci_hi.max() + 150)
    a.legend(handles=[Line2D([0], [0], color=ORANGE, lw=3, label="DiD family (parallel trends)"),
                      Line2D([0], [0], color=BLUE, lw=3, label="Synthetic-control family"),
                      Line2D([0], [0], color=AQUA, lw=3, label="Bayesian synthetic control"),
                      Line2D([0], [0], color=INK, lw=1.6, ls=(0, (5, 3)), label=f"True ATT = {truth:.0f} (simulation truth)")], loc="lower right", fontsize=8)
    if has_mc:
        b = axes[1]
        mc = M["mc"].set_index("estimator")
        mc = mc.loc[[o for o in order if o in mc.index]]
        yy = np.arange(len(mc))[::-1]
        b.axvline(0, color=INK2, lw=1)
        for yi, (name, r) in zip(yy, mc.iterrows()):
            b.barh(yi, r.mean_bias, height=0.5, color=FAMILY[name], zorder=2)
            b.text(max(r.mean_bias, 0) + 4, yi, f"{r.mean_bias:+.0f}  (RMSE {r.rmse:.0f}, cover {100 * r.ci_coverage:.0f}%)", va="center", fontsize=8.2)
        b.set_yticks(yy, [])
        b.set_xlabel("Mean bias, room nights / day")
        n_mc = int(pd.read_csv(resolve(cfg["paths"]["metrics"]) / "panel_monte_carlo_runs.csv")["seed"].nunique())
        b.set_title(f"Monte-Carlo check: {n_mc} fresh panels")
        b.set_xlim(min(0, mc.mean_bias.min()) - 5, mc.mean_bias.max() + 105)
    fig.suptitle("Panel ATT estimators on the same 25 x 240 panel", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "05_panel_att_estimators_comparison.png", cfg)


# ----------------------------------------------------------------------------- 06
def fig06(cfg: dict, M: dict) -> None:
    c = M["cate"].copy()
    ref = c[c.reference]
    learners = c[~c.reference].sort_values("pehe")
    fig = plt.figure(figsize=(13.8, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.85, 1.05], wspace=0.34)
    a = fig.add_subplot(gs[0])
    y = np.arange(len(learners))[::-1]
    for yi, r in zip(y, learners.itertuples()):
        a.barh(yi, r.pehe, height=0.55, color=LEARNER_FAMILY[r.learner], zorder=2)
        sd = 0 if np.isnan(r.pehe_sd) else r.pehe_sd
        a.plot([r.pehe - sd, r.pehe + sd], [yi, yi], color=INK, lw=1.2, zorder=3)
        a.text(r.pehe + sd + 1.5, yi, f"{r.pehe:.1f}", va="center", fontsize=8.5)
    ref_handles = []
    for r in ref.itertuples():
        ls = (0, (5, 3)) if "Constant" in r.learner else (0, (1, 2))
        a.axvline(r.pehe, color=INK, lw=1.4, ls=ls)
        ref_handles.append(Line2D([0], [0], color=INK, lw=1.4, ls=ls,
                                  label=("Constant effect (oracle mean)" if "Constant" in r.learner else "Oracle regression on features") + f": {r.pehe:.0f}"))
    a.set_yticks(y, list(learners["learner"]))
    a.set_xlabel("PEHE: RMSE vs. true daily effect (room nights)")
    a.set_title("(a) Accuracy, lower is better\n(mean +/- sd over seeds)")
    a.legend(handles=[Line2D([0], [0], color=BLUE, lw=6, label="Meta-learners"), Line2D([0], [0], color=AQUA, lw=6, label="DML / forest / domain"),
                      Line2D([0], [0], color=ORANGE, lw=6, label="Deep (PyTorch)")] + ref_handles, loc="upper center",
             bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    b = fig.add_subplot(gs[1])
    lq = learners.set_index("learner").loc[learners["learner"], :]
    b.barh(y, lq["qini_oracle"], height=0.55, color=[LEARNER_FAMILY[k] for k in lq.index], zorder=2)
    for yi, v in zip(y, lq["qini_oracle"]):
        b.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8.5)
    b.set_yticks(y, [])
    b.set_xlim(min(0, lq["qini_oracle"].min()) - 0.02, 1.05)
    b.set_xlabel("Qini coefficient (0 = random, 1 = oracle)")
    b.set_title("(b) Ranking campaign days\nby predicted lift")
    d = fig.add_subplot(gs[2])
    pr = M["cate_pred"]
    best = learners["learner"].iloc[0]
    d.plot([pr.tau_true.min(), pr.tau_true.max()], [pr.tau_true.min(), pr.tau_true.max()], color=INK2, lw=1, ls=(0, (5, 3)), zorder=1)
    cf = "Causal Forest DML"
    if cf in pr and cf + " lo" in pr:
        d.errorbar(pr.tau_true, pr[cf], yerr=[pr[cf] - pr[cf + " lo"], pr[cf + " hi"] - pr[cf]], fmt="none", ecolor="#bfeee2", elinewidth=1, zorder=2)
        d.scatter(pr.tau_true, pr[cf], s=22, color=AQUA, edgecolor=SURF, linewidth=0.8, zorder=3, label="Causal Forest DML (95% CI)")
    if best != cf:
        d.scatter(pr.tau_true, pr[best], s=22, color=LEARNER_FAMILY[best], edgecolor=SURF, linewidth=0.8, zorder=4, label=f"{best} (best PEHE)")
    d.set_xlabel("True daily effect (room nights)")
    d.set_ylabel("Predicted daily effect")
    d.set_title("(c) Predicted vs. true effect\nper campaign day")
    d.legend(loc="upper left", fontsize=8)
    fig.suptitle("Ten Geo-CATE learners on the Miami campaign window", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.0)
    _save(fig, "06_cate_meta_learners_comparison.png", cfg)


# ----------------------------------------------------------------------------- 07
def fig07(cfg: dict, M: dict) -> None:
    dr, pr = M["drivers"], M["cate_pred"]
    pdp = dr["partial_dependence"]
    feats = [("days_since_launch", "Days since launch"), ("flight_search_index", "Flight-search index"),
             ("competitor_adr_usd", "Competitor ADR (USD)"), ("ota_promo_pressure_index", "OTA promo pressure index")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    axes = axes.ravel()
    best = dr["best_learner"]
    for ax, (f, lab) in zip(axes[:4], feats):
        g = np.array(pdp[f]["grid"])
        ax.plot(g, pdp[f]["tau_hat"], color=BLUE, lw=2.2, marker="o", ms=4.5)
        x, tt = pr[f].to_numpy(), pr["tau_true"].to_numpy()
        bins = np.unique(np.quantile(x, np.linspace(0, 1, 6)))
        ids = np.clip(np.digitize(x, bins[1:-1]), 0, len(bins) - 2)
        xs = [x[ids == i].mean() for i in range(len(bins) - 1)]
        ys = [tt[ids == i].mean() for i in range(len(bins) - 1)]
        ax.scatter(xs, ys, s=52, color=INK, marker="D", edgecolor=SURF, linewidth=1.3, zorder=4)
        ax.set_xlabel(lab)
        ax.set_ylabel("Effect, room nights / day")
        ax.set_title(lab, fontsize=10)
    # subgroup panel: estimated vs. true mean effect for the planner-relevant splits
    ax = axes[4]
    sub = dr["subgroups"]
    groups = [("Weekend", "is_weekend", "est_high", "true_high"), ("Weekday", "is_weekend", "est_low", "true_low"),
              ("Holiday window", "is_holiday_window", "est_high", "true_high"), ("Other days", "is_holiday_window", "est_low", "true_low"),
              ("Stormy (severity > 5)", "weather_severity_index", "est_high", "true_high"), ("Calm", "weather_severity_index", "est_low", "true_low")]
    xg = np.arange(len(groups))
    est = [sub[k][e] for _, k, e, _ in groups]
    tru = [sub[k][t] for _, k, _, t in groups]
    ax.bar(xg - 0.2, est, width=0.38, color=BLUE, zorder=2, label=f"{best} estimate")
    ax.bar(xg + 0.2, tru, width=0.38, color="#bdbdbd", zorder=2, label="True mean effect")
    for xi, (e_, t_) in enumerate(zip(est, tru)):
        ax.text(xi - 0.2, e_ + 2, f"{e_:.0f}", ha="center", fontsize=7.8)
        ax.text(xi + 0.2, t_ + 2, f"{t_:.0f}", ha="center", fontsize=7.8, color=INK2)
    ax.set_xticks(xg, [g[0] for g in groups], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Effect, room nights / day")
    ax.set_title("Subgroup effects: estimated vs. true", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.8, ncol=1)
    ax.set_ylim(0, max(est + tru) * 1.25)
    ax = axes[5]
    coef = {k: v for k, v in dr["linear_dml"]["coef_room_nights_per_sd"].items() if abs(v) > 0}
    names = sorted(coef, key=lambda k: abs(coef[k]))
    vals = [coef[k] for k in names]
    ax.barh(range(len(names)), vals, color=[AQUA if v >= 0 else ORANGE for v in vals], height=0.6, zorder=2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=8)
    ax.set_xlabel("Change in effect per +1 pooled sd (room nights/day)")
    ax.set_title("Linear DML on the named drivers", fontsize=10)
    fig.suptitle("What drives lift? Market-condition heterogeneity in the campaign effect", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    fig.legend(handles=[Line2D([0], [0], color=BLUE, lw=2.2, marker="o", ms=5, label=f"{best}: partial dependence"),
                        Line2D([0], [0], color=INK, lw=0, marker="D", ms=6, label="True mean effect by quintile")],
               loc="upper right", bbox_to_anchor=(0.995, 0.975), ncol=2, fontsize=8.5)
    _save(fig, "07_cate_heterogeneity_drivers.png", cfg)


# ----------------------------------------------------------------------------- 08
def fig08(cfg: dict, M: dict) -> None:
    nn = M["nn"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, name, col in [(axes[0], "TARNet", ORANGE), (axes[1], "CFRNet-DragonNet", BLUE)]:
        h = nn[name]["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=col, lw=2, label="train (total loss)")
        ax.plot(ep, h["val_loss"], color=INK, lw=1.8, label="validation")
        be = nn[name]["best_epoch"] + 1
        ax.axvline(be, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(be, ax.get_ylim()[0], f" best epoch {be}", fontsize=8, va="bottom", color=INK2)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    h = nn["CFRNet-DragonNet"]["history"]
    ep = np.arange(1, len(h["train_outcome"]) + 1)
    for key, lab, col in [("train_outcome", "factual outcome (weighted MSE)", BLUE), ("train_mmd", "MMD representation balance", AQUA),
                          ("train_prop", "propensity head (BCE)", ORANGE), ("train_target", "targeted regularisation", VIOLET)]:
        ax.plot(ep, np.maximum(h[key], 1e-6), color=col, lw=1.8, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_title("CFRNet-DragonNet loss components")
    ax.legend(loc="upper right", fontsize=7.8)
    fig.suptitle("PyTorch uplift networks on the GPU: training curves", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "08_pytorch_uplift_training_curves.png", cfg)


FIGURES = [("01", fig01), ("02", fig02), ("03", fig03), ("04", fig04), ("05", fig05), ("06", fig06), ("07", fig07), ("08", fig08)]


def run(cfg: dict) -> list[str]:
    log = get_logger("viz")
    _style()
    M = _load(cfg)
    done = []
    for tag, fn in FIGURES:
        try:
            fn(cfg, M)
            done.append(tag)
            log.info("figure %s written", tag)
        except Exception as exc:  # one broken figure must not stop the others
            import traceback

            log.info("figure %s FAILED: %s: %s\n%s", tag, type(exc).__name__, exc, traceback.format_exc(limit=3))
    return done


if __name__ == "__main__":
    from src.utils.config import load_config

    run(load_config())
