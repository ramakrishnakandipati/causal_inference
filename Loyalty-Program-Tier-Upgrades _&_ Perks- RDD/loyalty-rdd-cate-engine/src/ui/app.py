"""Streamlit dashboard (compatible with Streamlit 1.12).

Run:  streamlit run src/ui/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.api.service import ModelService, ModelsNotTrained, SHORT_TO_OUTCOME, UNITS
from src.evaluation.visualizer import local_poly_curve

BLUE, ORANGE, AQUA, GREY, INK = "#257de0", "#f47230", "#00b894", "#666666", "#191919"
OUT_LABEL = {"next_year_room_nights": "Room nights", "next_year_ancillary_spend": "Ancillary spend (USD)",
             "next_year_direct_booking_ratio": "Direct-booking ratio"}

PERSONAS = {
    "Leisure family traveller": dict(historical_adr=170.0, business_travel_ratio=0.08, lead_time_avg=70.0, ancillary_spend_t=900.0,
                                     direct_booking_ratio_t=0.30, tenure_years=5.0, co_brand_card_holder=False,
                                     home_region="North America", age_group="40-49", weekend_stay_ratio=0.55),
    "Bleisure couple": dict(historical_adr=175.0, business_travel_ratio=0.40, lead_time_avg=40.0, ancillary_spend_t=700.0,
                            direct_booking_ratio_t=0.35, tenure_years=4.0, co_brand_card_holder=False,
                            home_region="Europe", age_group="30-39", weekend_stay_ratio=0.40),
    "Corporate road warrior": dict(historical_adr=210.0, business_travel_ratio=0.88, lead_time_avg=8.0, ancillary_spend_t=350.0,
                                   direct_booking_ratio_t=0.40, tenure_years=9.0, co_brand_card_holder=True,
                                   home_region="North America", age_group="40-49", weekend_stay_ratio=0.05),
    "Budget solo explorer": dict(historical_adr=75.0, business_travel_ratio=0.10, lead_time_avg=25.0, ancillary_spend_t=150.0,
                                 direct_booking_ratio_t=0.15, tenure_years=1.5, co_brand_card_holder=False,
                                 home_region="Asia-Pacific", age_group="18-29", weekend_stay_ratio=0.45),
    "Luxury resort loyalist": dict(historical_adr=420.0, business_travel_ratio=0.20, lead_time_avg=90.0, ancillary_spend_t=2600.0,
                                   direct_booking_ratio_t=0.60, tenure_years=12.0, co_brand_card_holder=True,
                                   home_region="Middle East & Africa", age_group="50-64", weekend_stay_ratio=0.50),
}

st.set_page_config(page_title="Loyalty Tier RDD + CATE", page_icon="🏨", layout="wide")


@st.experimental_singleton
def load_service():
    return ModelService(os.environ.get("LOYALTY_OUTPUTS"))


@st.experimental_memo
def threshold_sweep(outcome_short: str, cost: float):
    """Net value for every candidate threshold; memoised because Streamlit reruns the whole script on each widget change."""
    svc_ = load_service()
    rows = []
    for cc in range(12, 31):
        s = svc_.simulate_threshold(cc, outcome_short, None, cost)
        rows.append({"threshold": cc, "net": s["net_value_usd"] or 0.0, "affected": s["guests_affected"]})
    return rows


def fmt(outcome_short: str, v: float, signed: bool = True) -> str:
    s = "+" if signed and v >= 0 else ""
    if outcome_short == "ancillary_spend":
        return f"{s}${v:,.0f}"
    if outcome_short == "direct_booking_ratio":
        return f"{s}{v:.3f}"
    return f"{s}{v:.2f} nights"


try:
    svc = load_service()
except ModelsNotTrained as exc:
    st.error(f"{exc}")
    st.stop()

c = svc.cutoff
st.title("Does reaching Gold change how guests behave?")
st.caption(f"Regression discontinuity at {c:.0f} qualifying nights plus heterogeneous-effect learners · "
           f"{svc.regime} design · {len(svc.panel):,} synthetic loyalty members")

tab_rdd, tab_valid, tab_cate, tab_sim, tab_models = st.tabs(
    ["Gold effect (RDD)", "Validity checks", "Guest explorer (CATE)", "Threshold simulator", "Model leaderboard"])

# ------------------------------------------------------------------------------------------ RDD
with tab_rdd:
    s1, s2, s3, s4 = st.columns(4)
    outcome = s1.selectbox("Outcome", list(OUT_LABEL), format_func=OUT_LABEL.get, key="rdd_outcome")
    kernel = s2.selectbox("Kernel", ["triangular", "epanechnikov", "uniform"], key="rdd_kernel")
    default_h = float(svc.meta["bandwidths"]["used"])
    h = s3.slider("Bandwidth h (nights)", 4.0, 15.0, float(round(default_h, 1)), 0.5, key="rdd_h")
    order = s4.selectbox("Polynomial order", [1, 2], key="rdd_p")
    res = svc.late(outcome, kernel, h, order)
    oc_short = outcome.replace("next_year_", "")
    m1, m2, m3 = st.columns(3)
    if res["estimate"] is None:
        m1.warning("Not enough support points for this bandwidth and polynomial order.")
    else:
        m1.metric("Estimated Gold effect", fmt(oc_short, res["estimate"]),
                  help="Fuzzy RDD (2SLS) for guests whose status follows the cutoff" if svc.regime == "fuzzy" else "Sharp RDD")
        m2.metric("95% interval", f"{fmt(oc_short, res['ci_low'], False)} to {fmt(oc_short, res['ci_high'], False)}")
        m3.metric("Known truth (synthetic)", fmt(oc_short, res["true_local_late"]))

    r_all = svc.panel["qualifying_nights_t"].to_numpy(float)
    y_all = svc.panel[outcome].to_numpy(float)
    sel = (r_all >= 4) & (r_all <= 36)
    r, y = r_all[sel], y_all[sel]
    bins = pd.DataFrame({"r": r, "y": y}).groupby("r")["y"].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bins.index, y=bins.values, mode="markers", name="Mean per night",
                             marker=dict(color=GREY, size=7)))
    for left, color, name in ((True, BLUE, "Below Gold threshold"), (False, ORANGE, "At or above threshold")):
        side = r < c if left else r >= c
        grid = np.linspace(r[side].min() + 0.5, c, 50) if left else np.linspace(c, r[side].max() - 0.5, 50)
        f, lo, hi = local_poly_curve(r[side], y[side], grid, hb=5.0)
        fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="none",
                                 fillcolor="rgba(37,125,224,0.18)" if left else "rgba(244,114,48,0.18)",
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid, y=f, mode="lines", name=name, line=dict(color=color, width=3)))
    fig.add_vline(x=c, line_dash="dash", line_color=INK)
    fig.update_layout(height=470, template="plotly_white", xaxis_title="Qualifying nights in year t",
                      yaxis_title=OUT_LABEL[outcome], legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dots are average outcomes per qualifying-night count; lines are local polynomial fits with 95% bands. "
               "The vertical gap at the dashed line is the causal effect of Gold status for guests at the threshold.")
    with st.expander("How the estimate is built"):
        st.write(f"Local polynomial of order {order}, {kernel} kernel, bandwidth {h:g} nights "
                 f"({res['n_left']:,} guests below, {res['n_right']:,} above). "
                 + ("Fuzzy design: eligibility (R ≥ 20) instruments actual Gold attainment, so the estimate is a "
                    "complier effect. First-stage jump in Gold take-up: "
                    f"{res.get('first_stage_jump', float('nan')):.2f}." if svc.regime == "fuzzy" else ""))

# ------------------------------------------------------------------------------------------ validity
with tab_valid:
    diag = svc.rdd
    mc = diag["mccrary"]
    v1, v2, v3 = st.columns(3)
    v1.metric("Density test p-value", f"{mc['p_value']:.3f}", "passes" if not mc["reject_continuity_5pct"] else "bunching detected")
    bal = diag["balance"]
    v2.metric("Covariates balanced", f"{int(bal['balanced'].sum())} of {len(bal)}")
    plc = diag["placebo"]
    v3.metric("Placebo cutoffs with no effect", f"{int(plc['null_not_rejected'].sum())} of {len(plc)}")
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Covariate balance at the cutoff")
        b = bal[["covariate", "estimate", "se", "p_value", "p_holm", "balanced"]].round(3)
        st.dataframe(b)
    with cB:
        st.subheader("Placebo cutoffs")
        p = plc[["placebo_cutoff", "outcome", "estimate", "se", "p_value"]].round(3)
        st.dataframe(p)
    st.subheader("Donut hole (excluding 19 and 20 nights)")
    st.dataframe(diag["donut"].round(3))
    img = svc.out / "plots" / "02_mccrary_density_test.png"
    if img.exists():
        st.image(str(img), caption="Density of qualifying nights around the cutoff")

# ------------------------------------------------------------------------------------------ CATE
with tab_cate:
    trained = svc.outcomes
    if not trained:
        st.info("No CATE models trained yet.")
    else:
        st.subheader("Who responds most to Gold perks?")
        left, right = st.columns([1, 1.2])
        with left:
            persona = st.selectbox("Guest persona", list(PERSONAS), key="persona")
            base = PERSONAS[persona]
            k = persona  # widget keys change with the persona so sliders reset to the preset
            prof = dict(
                historical_adr=st.slider("Average daily rate (USD)", 40.0, 700.0, float(base["historical_adr"]), 5.0, key="adr" + k),
                business_travel_ratio=st.slider("Business-travel share", 0.0, 1.0, float(base["business_travel_ratio"]), 0.01, key="biz" + k),
                lead_time_avg=st.slider("Average lead time (days)", 0.0, 200.0, float(base["lead_time_avg"]), 1.0, key="lead" + k),
                ancillary_spend_t=st.slider("Ancillary spend this year (USD)", 0.0, 5000.0, float(base["ancillary_spend_t"]), 50.0, key="anc" + k),
                direct_booking_ratio_t=st.slider("Direct-booking share", 0.0, 1.0, float(base["direct_booking_ratio_t"]), 0.01, key="dir" + k),
                tenure_years=st.slider("Tenure (years)", 0.0, 25.0, float(base["tenure_years"]), 0.5, key="ten" + k),
                co_brand_card_holder=st.checkbox("Co-brand credit card holder", bool(base["co_brand_card_holder"]), key="cb" + k),
                home_region=st.selectbox("Home region", svc.meta["home_regions"], svc.meta["home_regions"].index(base["home_region"]), key="reg" + k),
                age_group=st.selectbox("Age group", svc.meta["age_groups"], svc.meta["age_groups"].index(base["age_group"]), key="age" + k),
                weekend_stay_ratio=st.slider("Weekend-stay share", 0.0, 1.0, float(base["weekend_stay_ratio"]), 0.01, key="wk" + k),
            )
        with right:
            oc_short = st.selectbox("Outcome", trained, format_func=lambda s: OUT_LABEL[SHORT_TO_OUTCOME[s]], key="cate_oc")
            learner = st.selectbox("Learner", ["(best)"] + svc.learners(oc_short), key="cate_learner")
            out = svc.cate(prof, oc_short, None if learner == "(best)" else learner)
            # Streamlit 1.12 does not allow columns nested inside a column, so the metrics are stacked
            st.metric("Predicted Gold effect", fmt(oc_short, out["predicted_effect"]),
                      help=f"Learner: {out['learner']}")
            st.metric("Compared with average guest", f"{out['percentile_among_guests']:.0f}th percentile",
                      f"avg {fmt(oc_short, out['population_average_effect'])}")
            st.write(f"**Segment:** {out['segment']}")
            if out.get("interval_95"):
                st.write(f"**95% interval:** {fmt(oc_short, out['interval_95'][0], False)} to {fmt(oc_short, out['interval_95'][1], False)}")
            (st.success if out["percentile_among_guests"] >= 60 else st.warning if out["percentile_among_guests"] < 35 else st.info)(out["recommendation"])
            alt = pd.Series(out.get("all_learner_estimates", {})).sort_values()
            if len(alt):
                fig = go.Figure(go.Bar(x=alt.values, y=alt.index, orientation="h", marker_color=BLUE))
                fig.update_layout(height=60 + 28 * len(alt), template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                                  title="Estimate by learner", xaxis_title=UNITS[oc_short])
                st.plotly_chart(fig, use_container_width=True)
        seg = svc.out / "plots" / "06_cate_by_segment.png"
        if seg.exists():
            st.image(str(seg), caption="Estimated vs true effect by guest segment (out-of-fold)")

# ------------------------------------------------------------------------------------------ simulator
with tab_sim:
    st.subheader("What if the Gold threshold moved?")
    s1, s2, s3 = st.columns(3)
    new_c = s1.slider("Candidate threshold (nights)", 10, 35, int(c) - 3, 1, key="sim_c")
    cost = s2.slider("Perk cost per activated member (USD)", 0, 600, 120, 10, key="sim_cost")
    sim_oc = s3.selectbox("Outcome", [o for o in svc.outcomes if o == "room_nights"] or svc.outcomes, key="sim_oc")
    sim = svc.simulate_threshold(new_c, sim_oc, None, float(cost))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Guests affected", f"{sim['guests_affected']:,}", f"{sim['share_of_members_affected'] * 100:.1f}% of members")
    k2.metric("Avg effect per guest", fmt(sim_oc, sim["avg_effect_per_guest"]))
    k3.metric("Incremental " + UNITS[sim_oc], f"{sim['incremental_total']:+,.0f}")
    if sim["net_value_usd"] is not None:
        k4.metric("Net value (USD)", f"${sim['net_value_usd']:+,.0f}", help="Incremental room-night revenue (at each guest's ADR) minus perk cost")
    st.write(f"**Direction:** {sim['direction']} · **Gold take-up factor:** {sim['take_up_factor']:.2f} · **Model:** {sim['learner']}")
    for w in sim["warnings"]:
        st.warning(w)
    sw = pd.DataFrame(threshold_sweep(sim_oc, float(cost)))
    fig = go.Figure(go.Bar(x=sw["threshold"], y=sw["net"], marker_color=[AQUA if v >= 0 else ORANGE for v in sw["net"]]))
    fig.add_vline(x=c, line_dash="dash", line_color=INK)
    fig.update_layout(height=340, template="plotly_white", xaxis_title="Gold threshold (nights)",
                      yaxis_title="Net value vs today (USD)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Effects are measured at the 20-night cutoff and extrapolated for other thresholds; treat moves of more than "
               "about 5 nights as directional only.")

# ------------------------------------------------------------------------------------------ models
with tab_models:
    for oc_short, lb in svc.leaderboards.items():
        st.subheader(f"CATE learners · {oc_short.replace('_', ' ')}")
        show = lb[["learner", "status", "pearson", "pehe_rmse", "ate_hat", "ate_true", "auuc_over_random",
                   "qini_coefficient", "uplift_at_10", "uplift_at_30", "policy_value_dr", "r_loss_oof", "seconds"]].round(3)
        st.dataframe(show)
    st.caption("Correlation, RMSE and the average effects compare each learner with the true individual effect, which is "
               "known only because the data are synthetic. AUUC, Qini, uplift and policy value use observed outcomes only.")
    for name in ("05_cate_leaderboard.png", "07_uplift_qini_curves.png", "08_training_loss_curves.png"):
        p = svc.out / "plots" / name
        if p.exists():
            st.image(str(p))
