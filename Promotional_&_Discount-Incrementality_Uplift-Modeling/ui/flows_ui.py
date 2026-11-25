"""Streamlit rendering for the five end-to-end flows (see src/flows.py for the numbers behind them)."""
from __future__ import annotations

import json
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.common import LOYALTY_LEVELS
from src.flows import FLOWS, QUADRANT_WHY, campaign_report, guest_report, load_flow_data, pick_guest

POS, NEG, BLUE, GREY, INK = "#009E73", "#D55E00", "#0072B2", "#6b7280", "#1f2937"
BADGE = {"Persuadable": "🟢", "Always-Taker": "🟡", "Lost Cause": "⚪", "Sleeping Dog": "🔴"}
STEPPER = "**① Profile** → **② Selection bias** → **③ CATE, 11 estimators** → **④ Persona & decision** → **⑤ Business impact**"
STEPPER_CAMPAIGN = "**① CRM list** → **② Batch scoring** → **③ Targeting quality** → **④ Policy comparison** → **⑤ Scale to database**"


def _plain(fn):
    """Build figures under matplotlib's default style: other imported libs (econml/shap) leave a ggplot-like rcParams."""
    def wrapper(*a, **k):
        with plt.style.context("default"), plt.rc_context({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                                                           "axes.facecolor": "white", "figure.facecolor": "white"}):
            return fn(*a, **k)
    wrapper.__name__ = fn.__name__
    return wrapper


def _usd(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 0 else f"-${-v:,.0f}"))


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.grid(axis="x", alpha=0.25)


def _show(container, fig):
    """Render at the figure's natural size (100 px/inch) so fonts stay readable instead of stretching to the column."""
    container.pyplot(fig, clear_figure=True, width=int(fig.get_figwidth() * 100))


@contextmanager
def step(fid: str, k: int, title: str, sub: str = ""):
    with st.container(border=True, key=f"{fid}_step{k}"):
        st.markdown(f"##### Step {k} · {title}")
        if sub:
            st.caption(sub)
        yield


@st.cache_data(show_spinner=False)
def _flow_data(_cfg):
    return load_flow_data(_cfg)


@st.cache_data(show_spinner="Scoring with all 11 estimators…")
def _guest(_scorer, flow_id: str, est: str):
    te, _ = _flow_data(_scorer.cfg)
    f = next(x for x in FLOWS if x["id"] == flow_id)
    row = pick_guest(te, f["persona"])
    return guest_report(_scorer, row, estimator=est), int(row.name)


@st.cache_data(show_spinner="Scoring the CRM list…")
def _campaign(_scorer, est: str, n: int, thr: float):
    te, dec = _flow_data(_scorer.cfg)
    return campaign_report(_scorer, te, dec, est, n, thr)


def _money(v: float, dec: int = 0) -> str:
    return f"-${abs(v):,.{dec}f}" if v < 0 else f"${v:,.{dec}f}"


# ----------------------------------------------------------------------------------------------------- guest flows
@_plain
def _fig_estimators(rep: dict, thr: float):
    items, head, truth = rep["per_estimator"], rep["estimator"], rep["truth"]["tau_margin"]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ys = np.arange(len(items))[::-1]
    for yi, it in zip(ys, items):
        col = POS if it["tau"] > thr else NEG
        if it["lo"] is not None:
            ax.hlines(yi, it["lo"], it["hi"], color=col, lw=1.6, alpha=0.5)
        ax.plot(it["tau"], yi, "o", color=col, ms=7, mec="white", mew=1.4, zorder=3)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.axvline(truth, color=INK, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([("★ " if it["estimator"] == head else "") + it["estimator"] for it in items])
    for lab, it in zip(ax.get_yticklabels(), items):
        lab.set_fontweight("bold" if it["estimator"] == head else "normal")
    ax.set_xlabel("Estimated incremental 30-day margin of the 20%-off offer ($; line = 95% CI)")
    _usd(ax, "x")
    _clean(ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=POS, lw=0, label="τ̂ above threshold (mail)"),
                       Line2D([], [], marker="o", color=NEG, lw=0, label="τ̂ at/below threshold (suppress)"),
                       Line2D([], [], color=INK, ls="--", label=f"true τ = {_money(truth, 1)} (ground truth)")],
              fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=1.2, handletextpad=0.3)
    fig.tight_layout()
    return fig


@_plain
def _fig_impact(rep: dict):
    dec = rep["head"]["decision"]
    vals = [rep["mass_blast_value"] * 1000, rep["chosen_value"] * 1000]
    labels = ["Mass blast\n(20% off to all)", f"Model decision\n({dec})"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(labels, vals, width=0.45, color=[POS if v >= 0 else NEG for v in vals])
    span = max(abs(v) for v in vals) or 1
    for i, v in enumerate(vals):
        ax.text(i, v + (0.03 if v >= 0 else -0.03) * span, _money(v), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9.5, color=INK, fontweight="bold")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(min(min(vals) * 1.35, -0.1 * span), max(max(vals) * 1.35, 0.1 * span))
    ax.set_ylabel("Net incremental margin per\n1,000 look-alike guests")
    _usd(ax)
    _clean(ax)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    return fig


def _render_guest_flow(flow: dict, scorer, est: str, cfg: dict):
    fid = flow["id"]
    rep, idx = _guest(scorer, fid, est)
    te, _ = _flow_data(cfg)
    h, tr = rep["head"], rep["truth"]
    thr = cfg["offer"]["cate_threshold"]
    with st.container(key=f"{fid}_full"):
        st.markdown(f"### Flow {fid[-1]} — {flow['title']}  ·  designed persona: *{flow['persona']}*")
        st.write(flow["story"])
        st.markdown(STEPPER)

        with step(fid, 1, "Guest profile", f"A real held-out guest (test-set row {idx}), picked as the most typical {flow['persona']}."):
            c1, c2 = st.columns([1, 1])
            p = rep["payload"]
            tbl = pd.DataFrame({"Value": [f"{p['hist_booking_freq']:.1f}", f"{p['days_since_last_stay']:.0f}", p["loyalty_tier"],
                                          f"{p['hist_price_sensitivity']:.2f}", f"{p['avg_lead_time']:.0f}", f"${p['avg_adr']:.0f}",
                                          f"{p['weekend_stay_ratio']:.2f}", f"{p['cancellation_rate']:.2f}",
                                          f"{p['email_open_rate_90d']:.2f}", f"{p['points_balance']:,.0f}"]},
                               index=["Bookings / year", "Days since last stay", "Loyalty tier", "Price sensitivity", "Avg lead time (days)",
                                      "Avg daily rate", "Weekend-stay ratio", "Cancellation rate", "Email open rate (90d)", "Points balance"])
            c1.table(tbl)
            c2.markdown("**The same request the API serves** (`POST /score`)")
            c2.code(json.dumps({"guest": p, "estimator": est}, indent=1), language="json")

        with step(fid, 2, "Selection-bias check", "Historically, who got mailed was not random: the CRM favoured engaged, loyal guests."):
            e = rep["propensity"]
            pop = float(te["T"].mean())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Propensity e(x) = P(mailed | X)", f"{e:.2f}", f"{e - pop:+.2f} vs CRM average {pop:.2f}", delta_color="off")
            m2.metric("IPW weight if mailed", f"{rep['w_if_mailed']:.2f}")
            m3.metric("IPW weight if not mailed", f"{rep['w_if_not_mailed']:.2f}")
            lo, hi = cfg["evaluation"]["propensity_clip"]
            m4.metric("Overlap", "OK" if lo < e < hi else "Poor", f"valid range {lo:.2f}–{hi:.2f}", delta_color="off")
            if e > pop + 0.1:
                msg = "Guests like this were mailed **more than average**, so a naive mailed-vs-unmailed comparison would credit the promo with engagement that was already there."
            elif e < pop - 0.1:
                msg = "Guests like this were mailed **less than average**, so there is less treated evidence nearby; the estimators lean on the propensity / outcome models here."
            else:
                msg = "This guest sits near the average mailing probability, so treated and untreated look-alikes are plentiful."
            st.info(msg + " All 11 estimators correct for this through cross-fitted nuisance models or IPW.")

        with step(fid, 3, "Estimated treatment effect from all 11 estimators", "τ̂ = predicted extra 30-day margin if the guest receives the 20%-off offer."):
            c1, c2 = st.columns([1.6, 1])
            _show(c1, _fig_estimators(rep, thr))
            taus = np.array([it["tau"] for it in rep["per_estimator"]])
            n_pos = int((taus > thr).sum())
            c2.metric("Estimators that say “mail”", f"{n_pos} / {len(taus)}")
            c2.metric(f"Headline: {rep['estimator']}", f"{_money(h['tau_margin_hat'], 2)}",
                      None if h["ci95_low"] is None else f"95% CI [{h['ci95_low']:.1f}, {h['ci95_high']:.1f}]", delta_color="off")
            c2.metric("Spread across estimators", f"{_money(taus.min())} … {_money(taus.max())}")
            c2.caption("A tight cluster on one side of 0 means a robust call; a straddle means treat the decision with caution.")

        with step(fid, 4, "Persona & CRM decision"):
            q, d = h["quadrant"], h["decision"]
            c1, c2 = st.columns([1, 1.4])
            c1.markdown(f"### {BADGE[q]} {q}")
            (c1.success if d.startswith("SEND") else c1.error)(f"CRM decision: **{d}**")
            c2.markdown(QUADRANT_WHY[q].format(tc=100 * h["tau_conv_hat"], p0=100 * h["baseline_conv_prob"]))
            k1, k2, k3, k4 = c2.columns(4)
            k1.metric("τ̂ margin, 20% off", _money(h["tau_margin_hat"], 2))
            k2.metric("τ̂ margin, 2,000 pts*", _money(h["tau_margin_points_hat"], 2))
            k3.metric("τ̂ conversion", f"{100 * h['tau_conv_hat']:+.1f} pp")
            k4.metric("Baseline conv.", f"{100 * h['baseline_conv_prob']:.0f}%")
            c2.caption("*Points arm is modelled from the estimated conversion lift (80% of the discount lift), not observed in the data.")

        with step(fid, 5, "Business impact vs ground truth", "The synthetic data knows both potential outcomes, so every call can be audited."):
            c1, c2 = st.columns([1, 1])
            ok = h["quadrant"] == tr["persona"]
            cmp = pd.DataFrame({"Model": [h["quadrant"], f"{100 * h['tau_conv_hat']:+.1f} pp", _money(h["tau_margin_hat"], 1),
                                          f"{100 * h['baseline_conv_prob']:.0f}%"],
                                "Ground truth": [tr["persona"] + ("  ✅" if ok else "  ❌"), f"{100 * tr['tau_conv']:+.1f} pp",
                                                 _money(tr["tau_margin"], 1), f"{100 * tr['p0']:.0f}%"]},
                               index=["Persona", "Conversion effect", "Margin effect (20% off)", "Baseline conversion"])
            c1.table(cmp)
            gain = rep["chosen_value"] - rep["mass_blast_value"]
            c1.metric("Model decision vs mass blast", f"{_money(gain * 1000)} per 1,000 look-alikes",
                      "margin protected" if gain >= 0 else "margin lost", delta_color="normal" if gain >= 0 else "inverse")
            _show(c2, _fig_impact(rep))


# ----------------------------------------------------------------------------------------------------- campaign flow
@_plain
def _fig_mix(counts: dict):
    order = ["SEND 20% OFF", "SEND 2,000 POINTS", "SUPPRESS"]
    cols = {"SEND 20% OFF": BLUE, "SEND 2,000 POINTS": POS, "SUPPRESS": GREY}
    vals = [counts.get(k, 0) for k in order]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.barh(order[::-1], vals[::-1], height=0.5, color=[cols[k] for k in order[::-1]])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("Guests")
    _clean(ax)
    fig.tight_layout()
    return fig


@_plain
def _fig_policies(pol: pd.DataFrame):
    labels = [p.replace(" (", "\n(") for p in pol["Policy"]]
    vals = pol["Net incr. margin ($)"].values
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(labels, vals, width=0.5, color=[POS if v >= 0 else NEG for v in vals])
    span = max(np.abs(vals).max(), 1)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.03 if v >= 0 else -0.03) * span, _money(v), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(min(vals.min() * 1.3, -0.1 * span), max(vals.max() * 1.3, 0.1 * span))
    ax.set_ylabel("Net incremental margin")
    _usd(ax)
    ax.tick_params(axis="x", labelsize=8)
    _clean(ax)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    return fig


def _render_campaign_flow(flow: dict, scorer, est: str, cfg: dict):
    fid = flow["id"]
    te, _ = _flow_data(cfg)
    with st.container(key=f"{fid}_full"):
        st.markdown(f"### Flow 5 — {flow['title']}")
        st.write(flow["story"])
        st.markdown(STEPPER_CAMPAIGN)

        with step(fid, 1, "Choose the CRM list", "A random sample of held-out guests stands in for the list you would export from the CRM."):
            c1, c2 = st.columns([1, 1.6])
            n = c1.slider("Guests on the list", 500, 5000, 2000, 250, key="f5_n")
            thr = c1.slider("CATE targeting threshold ($)", -5.0, 40.0, float(cfg["offer"]["cate_threshold"]), 0.5, key="f5_thr")
            c1.caption(f"Estimator driving the policy: **{est}** (change it above).")
            rep = _campaign(scorer, est, n, thr)
            smp = rep["sample"].head(6)
            show = pd.DataFrame({"Bookings/yr": smp["hist_booking_freq"].map("{:.1f}".format),
                                 "Days since stay": smp["days_since_last_stay"].map("{:.0f}".format),
                                 "Tier": smp["loyalty_tier"].map(lambda i: LOYALTY_LEVELS[int(i)]),
                                 "ADR": smp["avg_adr"].map("${:.0f}".format),
                                 "Weekend ratio": smp["weekend_stay_ratio"].map("{:.2f}".format),
                                 "Cancel rate": smp["cancellation_rate"].map("{:.2f}".format),
                                 "Open rate": smp["email_open_rate_90d"].map("{:.2f}".format)})
            c2.markdown("**First rows of the list** (features only; the persona is hidden from the models)")
            c2.table(show)

        with step(fid, 2, "Batch scoring", "Same code path as `POST /score/batch`."):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Guests scored", f"{rep['n']:,}")
            m2.metric("Total latency", f"{rep['latency_ms_total']:.0f} ms")
            m3.metric("Per guest", f"{rep['latency_ms_total'] / rep['n']:.2f} ms")
            m4.metric("Mailed", f"{rep['n_send']:,}", f"{100 * rep['n_send'] / rep['n']:.0f}% of list", delta_color="off")
            _show(st, _fig_mix(rep["decision_counts"]))

        with step(fid, 3, "Targeting quality", "Who did the model decide to mail, and did it protect the guests a discount would hurt?"):
            c1, c2 = st.columns([1.5, 1])
            c1.markdown("**Assigned persona (columns) by true persona (rows)**")
            c1.dataframe(rep["confusion"].style.format("{:.0%}").background_gradient(cmap="Greens", axis=None), width="stretch")
            mix = rep["mailed_mix"]
            c2.metric("Always-Takers + Sleeping Dogs left alone", f"{100 * rep['harmful_suppressed']:.0f}%")
            c2.metric("Mailed list that is Persuadable", f"{100 * mix.get('Persuadable', 0):.0f}%")
            c2.metric("Mailed list that is Neutral", f"{100 * mix.get('Neutral', 0):.0f}%")

        with step(fid, 4, "Policy comparison (ground-truth counterfactuals)", "Propensity targeting gets the same mailing budget as the causal policy."):
            c1, c2 = st.columns([1.4, 1])
            _show(c1, _fig_policies(rep["policies"]))
            pol = rep["policies"].copy()
            pol["Net incr. margin ($)"] = pol["Net incr. margin ($)"].map(lambda v: _money(v))
            pol["Wasted offers ($)"] = pol["Wasted offers ($)"].map(lambda v: _money(v))
            c2.table(pol.set_index("Policy"))
            c2.caption("“Wasted offers” = discount/points handed to guests who would have booked anyway. The points arm is modelled.")

        with step(fid, 5, "Scale to the CRM database"):
            db = st.number_input("CRM database size", 10_000, 50_000_000, int(cfg["offer"]["crm_database_size"]), 100_000, key="f5_db")
            sc = db / rep["n"]
            p = rep["policies"].set_index("Policy")["Net incr. margin ($)"]
            mass, causal = p.iloc[0], p.iloc[3]
            k1, k2, k3 = st.columns(3)
            k1.metric("Mass blast", _money(mass * sc))
            k2.metric(f"Causal policy ({est})", _money(causal * sc))
            k3.metric("Margin protected vs mass blast", _money((causal - mass) * sc), f"{100 * rep['n_send'] / rep['n']:.0f}% of guests mailed", delta_color="off")
            st.caption(f"Scaled linearly from {rep['n']:,} sampled guests to {db:,}. Mail cost ${rep['cc']:.2f} per send is included.")


# ----------------------------------------------------------------------------------------------------- tab entry point
def render_flows_tab(scorer, cfg: dict):
    st.subheader("End-to-end flows")
    st.caption("Five worked scenarios on real held-out guests: profile → selection bias → CATE → decision → impact. "
               "Every number is computed live by the same code the API uses.")
    c1, c2 = st.columns([3, 1.3])
    short = [f["short"] for f in FLOWS]
    pick = c1.radio("Flow", short, horizontal=True, key="flow_choice")
    est = c2.selectbox("Headline estimator", scorer.order, index=scorer.order.index(scorer.best), key="flow_est")
    flow = FLOWS[short.index(pick)]
    if flow["kind"] == "guest":
        _render_guest_flow(flow, scorer, est, cfg)
    else:
        _render_campaign_flow(flow, scorer, est, cfg)
