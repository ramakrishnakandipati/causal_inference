"""Publication-grade (300 DPI) diagnostic and business charts.

Palette: validated categorical slots from the reference design (blue, orange, aqua, violet),
recessive hairline grid, text in ink tokens (never series colour), legend whenever >= 2 series.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.models.rdd_estimators import kernel_weights

# ---- tokens ---------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#257de0", "#f47230", "#00b894", "#6a4ce2", "#ee7fac"
CAT = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
DPI = 300

OUTCOME_LABEL = {
    "next_year_room_nights": ("Next-year room nights", "nights"),
    "next_year_ancillary_spend": ("Next-year ancillary spend", "USD"),
    "next_year_direct_booking_ratio": ("Next-year direct-booking ratio", "share"),
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9.5,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans", "savefig.dpi": DPI,
    })


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _fmt(outcome: str, v: float, signed: bool = True) -> str:
    sign = "+" if signed and v >= 0 else ""
    if outcome == "next_year_ancillary_spend":
        return f"{sign}${v:,.0f}"
    if outcome == "next_year_direct_booking_ratio":
        return f"{sign}{v:.3f}"
    return f"{sign}{v:.2f} nights"


# ---- local polynomial curve with CI ----------------------------------------------------------
def local_poly_curve(r, y, grid, hb: float = 4.0, p: int = 1, lo_side: bool = True):
    """Boundary-aware local polynomial fit (triangular kernel) with robust 95% bands.

    ``lo_side`` marks which side of the cutoff the grid lies on; data from the other side is never used.
    """
    fit, lo, hi = [], [], []
    for x0 in grid:
        w = kernel_weights(r, x0, hb, "triangular")
        m = w > 0
        if m.sum() < p + 3 or np.unique(r[m]).size < p + 1:
            fit.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        X = np.column_stack([(r[m] - x0) ** k for k in range(p + 1)])
        W = w[m]
        bi = np.linalg.pinv((X.T * W) @ X)
        beta = bi @ ((X.T * W) @ y[m])
        res = y[m] - X @ beta
        meat = (X * (W * res)[:, None]).T @ (X * (W * res)[:, None])
        cov = bi @ meat @ bi * m.sum() / max(m.sum() - p - 1, 1)
        se = np.sqrt(max(cov[0, 0], 0))
        fit.append(beta[0]); lo.append(beta[0] - 1.96 * se); hi.append(beta[0] + 1.96 * se)
    return np.array(fit), np.array(lo), np.array(hi)


# ================================================================================ 01 RDD main
def plot_rdd_main(df: pd.DataFrame, suite: dict, cutoff: float, out: Path, window=(4, 36)) -> str:
    _style()
    outcomes = list(suite["main"]["outcome"])
    fig, axes = plt.subplots(1, len(outcomes), figsize=(14.5, 4.6))
    r_all = df["qualifying_nights_t"].to_numpy(float)
    for ax, oc in zip(np.atleast_1d(axes), outcomes):
        y = df[oc].to_numpy(float)
        m = (r_all >= window[0]) & (r_all <= window[1])
        r, yy = r_all[m], y[m]
        bins = pd.DataFrame({"r": r, "y": yy}).groupby("r")["y"].mean()
        ax.scatter(bins.index, bins.values, s=22, color=INK2, alpha=0.85, zorder=3, label="Mean per night")
        for side, color, name in ((r < cutoff, BLUE, "Below Gold threshold"), (r >= cutoff, ORANGE, "At or above threshold")):
            grid = np.linspace(r[side].min() + 0.5 if color == BLUE else cutoff, cutoff if color == BLUE else r[side].max() - 0.5, 60)
            f, lo, hi = local_poly_curve(r[side], yy[side], grid, hb=5.0)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
            ax.plot(grid, f, color=color, lw=2, zorder=4, label=name)
        ax.axvline(cutoff, color=INK, lw=1, ls=(0, (4, 3)), zorder=1)
        row = suite["main"].set_index("outcome").loc[oc]
        label, _ = OUTCOME_LABEL[oc]
        fuzzy = row["design"] == "fuzzy"
        txt = f"{'Fuzzy RDD' if fuzzy else 'Sharp RDD'} effect: {_fmt(oc, row['estimate'])}\n95% CI [{_fmt(oc, row['ci_low'], False)}, {_fmt(oc, row['ci_high'], False)}]"
        if fuzzy:
            txt += f"\nITT jump: {_fmt(oc, row['itt_estimate'])}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID))
        ax.set_title(label)
        ax.set_xlabel("Qualifying nights in year t")
        ax.set_ylabel(OUTCOME_LABEL[oc][1])
        ax.set_xlim(window[0] - 0.5, window[1] + 0.5)
    h = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(*h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Reaching Gold at 20 nights lifts every outcome: sharp jumps at the cutoff, smooth trends elsewhere",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "01_rdd_discontinuity_main.png")


# ================================================================================ 02 McCrary
def _count_fits(r, c, bw=10, order=2):
    vals, counts = np.unique(np.asarray(r).astype(int), return_counts=True)
    x = vals.astype(float); boundary = c - 0.5
    curves = {}
    for side, mask in (("left", x < c), ("right", x >= c)):
        xs, cs = x[mask], counts[mask].astype(float)
        w = kernel_weights(xs, c, bw, "triangular"); sel = w > 0
        coef = np.polyfit(xs[sel] - boundary, cs[sel], order, w=np.sqrt(w[sel]))
        grid = np.linspace(xs[sel].min() - 0.4, boundary, 30) if side == "left" else np.linspace(boundary, xs[sel].max() + 0.4, 30)
        curves[side] = (grid, np.polyval(coef, grid - boundary))
    return x, counts, curves


def plot_mccrary(df: pd.DataFrame, mc: dict, cutoff: float, out: Path, df_manip: Optional[pd.DataFrame] = None,
                 mc_manip: Optional[dict] = None, bw: float = 10) -> str:
    _style()
    ncols = 3 if df_manip is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6))
    r = df["qualifying_nights_t"].to_numpy(float)
    ax = axes[0]
    vals, counts = np.unique(r.astype(int), return_counts=True)
    ax.bar(vals, counts, width=0.9, color=BLUE, alpha=0.35, linewidth=0)
    kde = stats.gaussian_kde(r, bw_method=0.18)
    xs = np.linspace(1, 45, 300)
    ax.plot(xs, kde(xs) * len(r), color=BLUE, lw=2, label="Smooth density (KDE)")
    ax.axvline(cutoff, color=INK, lw=1, ls=(0, (4, 3)))
    ax.set_title("Qualifying nights are smooth through 20")
    ax.set_xlabel("Qualifying nights in year t"); ax.set_ylabel("Guests per night count")
    ax.legend(loc="upper right")

    def zoom(ax, data, res, title, color):
        x, cnt, curves = _count_fits(data["qualifying_nights_t"], cutoff, bw)
        keep = (x >= cutoff - bw) & (x <= cutoff + bw)
        ax.scatter(x[keep], cnt[keep], s=26, color=INK2, zorder=3, label="Guests per night")
        ax.plot(*curves["left"], color=BLUE, lw=2, label="Local fit, below")
        ax.plot(*curves["right"], color=ORANGE, lw=2, label="Local fit, at/above")
        ax.axvline(cutoff - 0.5, color=INK, lw=1, ls=(0, (4, 3)))
        verdict = "no sign of bunching" if not res["reject_continuity_5pct"] else "bunching detected"
        top = max(cnt[keep].max(), curves["left"][1].max(), curves["right"][1].max())
        ax.set_ylim(0, top * 1.38)                       # headroom keeps the annotation clear of the data
        ax.text(0.97, 0.97, f"McCrary-style test\nlog density ratio {res['log_density_ratio']:+.3f}\np = {res['p_value']:.3f} ({verdict})",
                transform=ax.transAxes, va="top", ha="right", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID))
        ax.set_title(title); ax.set_xlabel("Qualifying nights in year t"); ax.set_ylabel("Guests per night count")
        ax.set_xlim(cutoff - bw - 0.5, cutoff + bw + 0.5)

    zoom(axes[1], df, mc, "Clean data: density test passes", BLUE)
    axes[1].legend(loc="lower left", fontsize=7.5)
    if df_manip is not None:
        zoom(axes[2], df_manip, mc_manip, "Simulated gaming: test fails", ORANGE)
    fig.suptitle("Running-variable density is continuous at the Gold cutoff", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "02_mccrary_density_test.png")


# ================================================================================ 03 sensitivity
def plot_bandwidth_sensitivity(suite: dict, out: Path, primary_only: bool = False) -> str:
    _style()
    s = suite["sensitivity"]
    outcomes = list(s["outcome"].unique())
    fig, axes = plt.subplots(1, len(outcomes), figsize=(14.5, 4.2))
    combos = [(k, p) for k in s["kernel"].unique() for p in sorted(s["poly_order"].unique())]
    for ax, oc in zip(np.atleast_1d(axes), outcomes):
        for i, (k, p) in enumerate(combos):
            d = s[(s.outcome == oc) & (s.kernel == k) & (s.poly_order == p)].dropna(subset=["estimate"])
            off = (i - len(combos) / 2) * 0.07
            ax.errorbar(d["bandwidth"] + off, d["estimate"], yerr=1.96 * d["se"], fmt="o-", ms=4, lw=1.4, capsize=2,
                        color=CAT[i % len(CAT)], label=f"{k}, p={p}")
        ax.set_title(OUTCOME_LABEL[oc][0]); ax.set_xlabel("Bandwidth h (nights)"); ax.set_ylabel("Estimated jump")
        ax.set_xticks(sorted(s["bandwidth"].unique()))
    np.atleast_1d(axes)[0].legend(loc="best", fontsize=7.5)
    fig.suptitle("Estimates are stable across bandwidths, kernels and polynomial orders", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "03_bandwidth_sensitivity.png")


# ================================================================================ 04 falsification
def plot_falsification(suite: dict, out: Path) -> str:
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    b = suite["balance"].sort_values("estimate")
    z = b["estimate"] / b["se"]
    a1.barh(b["covariate"], z, color=[BLUE if q > 0.05 else ORANGE for q in b["p_holm"]], height=0.55)
    a1.axvline(0, color=AXIS, lw=1)
    for v in (-1.96, 1.96):
        a1.axvline(v, color=INK2, lw=0.8, ls=(0, (3, 3)))
    a1.set_title("Covariates are balanced at the cutoff"); a1.set_xlabel("Jump at cutoff in standard errors (dashed = ±1.96)")
    pl = suite["placebo"]
    pl = pl[pl["outcome"] == pl["outcome"].iloc[0]]
    a2.set_xticks(pl["placebo_cutoff"].astype(float))
    a2.errorbar(pl["placebo_cutoff"].astype(float), pl["estimate"], yerr=1.96 * pl["se"], fmt="o", color=BLUE, capsize=3, ms=6)
    a2.axhline(0, color=INK2, lw=1)
    a2.set_title("Placebo cutoffs show no effect"); a2.set_xlabel("False cutoff (nights)")
    a2.set_ylabel(f"Jump in {OUTCOME_LABEL[pl['outcome'].iloc[0]][0].lower()}")
    fig.suptitle("Falsification checks: the jump appears only at 20 nights", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "04_covariate_balance_placebo.png")


# ================================================================================ 05 leaderboard
def plot_leaderboard(lb: pd.DataFrame, out: Path, outcome_label: str = "room nights") -> str:
    _style()
    d = lb.dropna(subset=["pearson"]).sort_values("pearson")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 0.42 * len(d) + 2.4), sharey=True)
    for ax, col, title, color in ((axes[0], "pearson", "Agreement with true effect (corr.)", BLUE),
                                  (axes[1], "pehe_rmse", "Individual-effect error (RMSE, lower is better)", ORANGE),
                                  (axes[2], "auuc_over_random", "Targeting gain over random (AUUC)", AQUA)):
        ax.barh(d["learner"], d[col], color=color, height=0.6)
        ax.set_title(title, fontsize=10)
        for y, v in enumerate(d[col]):
            ax.text(v, y, f" {v:.2f}", va="center", fontsize=8, color=INK)
    fig.suptitle(f"CATE learner leaderboard on {outcome_label} (out-of-fold, kernel-weighted window around 20 nights)",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "05_cate_leaderboard.png")


# ================================================================================ 06 segments
def plot_segments(oof: pd.DataFrame, learner: str, out: Path, outcome_label: str = "room nights") -> str:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    d = oof.copy()
    d["biz"] = pd.cut(d["business_travel_ratio"], [0, 0.15, 0.3, 0.45, 0.6, 1.0], include_lowest=True)
    d["adr"] = pd.cut(d["historical_adr"], [0, 100, 140, 180, 240, 2000], include_lowest=True)
    for ax, key, title, xl in ((axes[0], "biz", "Corporate-leaning guests respond least", "Business-travel share of stays"),
                               (axes[1], "adr", "Mid-tier ADR guests respond most", "Historical ADR (USD)")):
        g = d.groupby(key, observed=True).apply(lambda x: pd.Series({
            "est": np.average(x[learner], weights=x["w"]), "true": np.average(x["tau_true"], weights=x["w"]), "n": len(x)}))
        xs = np.arange(len(g))
        ax.bar(xs - 0.2, g["true"], 0.38, color=INK2, alpha=0.55, label="True effect (synthetic)")
        ax.bar(xs + 0.2, g["est"], 0.38, color=BLUE, label=f"Estimated ({learner})")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{max(i.left, 0):g}-{i.right:g}" if i.right < 1000 else f">{i.left:g}" for i in g.index])
        ax.set_ylim(0, max(g["true"].max(), g["est"].max()) * 1.3)     # headroom for the legend
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(f"Gold effect on {outcome_label}")
    axes[0].legend(loc="upper left")
    fig.suptitle("Who benefits most from Gold perks? Estimated vs true effect by guest segment", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "06_cate_by_segment.png")


# ================================================================================ 07 uplift
def plot_uplift(curves: Dict[str, pd.DataFrame], out: Path) -> str:
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for i, (name, c) in enumerate(curves.items()):
        a1.plot(c["fraction"], c["gain"], color=CAT[i % len(CAT)], lw=2, label=name)
        a2.plot(c["fraction"], c["qini"], color=CAT[i % len(CAT)], lw=2, label=name)
    for ax, key in ((a1, "gain"), (a2, "qini")):
        first = next(iter(curves.values()))
        ax.plot([0, 1], [0, first[key].iloc[-1]], color=INK2, lw=1, ls=(0, (4, 3)), label="Random targeting")
        ax.set_xlabel("Share of guests targeted (ranked by predicted effect)")
    a1.set_ylabel("Cumulative uplift"); a1.set_title("Uplift curve")
    a2.set_ylabel("Qini value"); a2.set_title("Qini curve")
    a2.legend(loc="lower right")
    fig.suptitle("Targeting by predicted effect beats random: out-of-fold uplift and Qini curves", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "07_uplift_qini_curves.png")


# ================================================================================ 08 loss curves
def plot_loss_curves(histories: Dict[str, Dict[str, list]], out: Path) -> str:
    _style()
    fig, axes = plt.subplots(1, max(len(histories), 1), figsize=(6.2 * max(len(histories), 1), 4.2), squeeze=False)
    for ax, (name, h) in zip(axes[0], histories.items()):
        ax.plot(h["train"], color=BLUE, lw=2, label="Train")
        ax.plot(h["val"], color=ORANGE, lw=2, label="Validation")
        best = int(np.argmin(h["val"]))
        ax.scatter([best], [h["val"][best]], color=INK, zorder=5, s=24)
        ax.annotate(f"best epoch {best + 1}", (best, h["val"][best]), textcoords="offset points", xytext=(8, 10), fontsize=8)
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted loss")
        ax.legend()
    fig.suptitle("Deep uplift networks converge and early-stop on validation loss", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    return _save(fig, out / "08_training_loss_curves.png")
