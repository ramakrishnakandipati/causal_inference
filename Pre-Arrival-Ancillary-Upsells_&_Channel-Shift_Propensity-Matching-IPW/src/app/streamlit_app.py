"""Streamlit dashboard: observational bias, CATE leaderboard, guest simulator, campaign optimiser."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import serving
from src.common.config import load_config, out_dir
from src.common.hardware import get_device_report
from src.data import synthetic_generator as sg

st.set_page_config(page_title="Hospitality Causal Upsell Engine", layout="wide", page_icon="🏨")
CFG = load_config()
M = out_dir("metrics")
PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"]

if not (M / "campaign_summary.json").exists():
    st.error("Artifacts missing. Run `python scripts/run_pipeline.py` first.")
    st.stop()


@st.cache_data
def load_all():
    return dict(ate=json.load(open(M / "ate_comparison.json")), bal=pd.read_csv(M / "balance_smd.csv"),
                lb=pd.read_csv(M / "cate_leaderboard.csv"), cv=pd.read_csv(M / "uplift_curves.csv"),
                dec=pd.read_csv(M / "uplift_deciles.csv"), ref=json.load(open(M / "refutation.json")),
                prop=pd.read_parquet(out_dir("data") / "train_propensity.parquet"),
                summ=serving.campaign_summary(), data=json.load(open(M / "data_summary.json")))


D = load_all()
dev = get_device_report()

st.title("🏨 Pre-Arrival Ancillary Upsells & Direct-Channel Shift")
st.caption(f"Causal inference & uplift engine · synthetic hospitality data with known ground truth · "
           f"compute: {'CUDA – ' + str(dev['gpu_name']) if dev['cuda'] else 'CPU'}")

t1, t2, t3, t4 = st.tabs(["1 · Observational bias & balance", "2 · CATE leaderboard & uplift", "3 · Guest simulator", "4 · Campaign optimiser"])

# ============================================================ TAB 1
with t1:
    st.subheader("Naive lift vs. causal estimators (ATE)")
    outcome = st.radio("Outcome", ["ancillary", "direct"], horizontal=True, key="o1",
                       format_func=lambda o: "Ancillary spend ($)" if o == "ancillary" else "Direct re-booking within 180d (probability)")
    v = D["ate"]["outcomes"][outcome]
    est = v["estimators"]
    names = list(est.keys())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[est[n]["ate"] for n in names], y=names, mode="markers", marker=dict(size=11, color=PAL[0]),
                             error_x=dict(type="data", symmetric=False, array=[est[n]["ate_hi"] - est[n]["ate"] for n in names],
                                          arrayminus=[est[n]["ate"] - est[n]["ate_lo"] for n in names]), name="Estimate (95% CI)"))
    fig.add_vline(x=v["true_ate"], line_dash="dash", line_color="black", annotation_text=f"True ATE {v['true_ate']:.3f}")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="ATE", yaxis=dict(autorange="reversed"))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    tbl = pd.DataFrame(est).T[["ate", "att", "ate_bias", "att_bias"]].rename(columns=str.upper)
    tbl.insert(0, "True ATE / ATT", f"{v['true_ate']:.3f} / {v['true_att']:.3f}")
    c2.dataframe(tbl.round(3), width="stretch")
    naive = est["Naive"]["ate"]
    c2.metric("Naive over-statement vs truth", f"{(naive / v['true_ate'] - 1) * 100:+.0f}%", help="Selection bias: engaged, tech-savvy guests are more likely to be reached")
    c2.metric("AIPW error vs truth", f"{(est['AIPW (DR)']['ate'] / v['true_ate'] - 1) * 100:+.1f}%")

    st.subheader("Covariate balance (Love plot) and propensity overlap")
    bal = D["bal"].set_index("feature")
    bal = bal.reindex(bal.smd_before.abs().sort_values().index)
    lf = go.Figure()
    for col, lab, sym, colr in (("smd_before", "Unadjusted", "circle", "#d62728"), ("smd_psm", "After PSM 1:1", "square", "#1f77b4"), ("smd_ipw", "After IPW", "triangle-up", "#2ca02c")):
        lf.add_trace(go.Scatter(x=bal[col].abs(), y=bal.index, mode="markers", name=lab, marker=dict(symbol=sym, size=9, color=colr)))
    lf.add_vline(x=0.1, line_dash="dash", line_color="black")
    lf.update_layout(height=520, xaxis_title="|SMD|", margin=dict(l=10, r=10, t=10, b=10))
    b1, b2 = st.columns(2)
    b1.plotly_chart(lf, width="stretch")
    P = D["prop"]
    P = P.assign(group=np.where(P.treatment == 1, "Treated", "Control"),
                 booking=np.where(P.channel.isin(["OTA_Expedia", "OTA_Booking"]), "OTA", "Direct / Corporate"))
    mode = b2.radio("Overlap split", ["Treated vs control", "OTA vs Direct/Corporate"], horizontal=True)
    hf = px.histogram(P, x="e_hat", color="group" if mode.startswith("Treated") else "booking", barmode="overlay", nbins=40,
                      histnorm="probability density", opacity=0.6, color_discrete_sequence=PAL)
    hf.update_layout(height=470, xaxis_title="Estimated propensity e(X)", margin=dict(l=10, r=10, t=10, b=10))
    b2.plotly_chart(hf, width="stretch")
    bs = D["ate"]["balance"]
    st.info(f"Covariates with |SMD| < 0.10 — before: **{bs['share_below_0p1_before']:.0%}**, after PSM: **{bs['share_below_0p1_psm']:.0%}**, "
            f"after IPW: **{bs['share_below_0p1_ipw']:.0%}** · common support kept {D['ate']['common_support_kept']:.1%} of stays.")
    with st.expander("Refutation tests (placebo treatment, random common cause, data subset)"):
        for o, r in D["ref"].items():
            st.markdown(f"**{o}** · baseline AIPW ATE = {r['baseline_aipw_ate']:.3f} (n={r['n_used']})")
            st.dataframe(pd.DataFrame(r["builtin"]).T[["new_effect", "pass"]], width="stretch")
            dw = r.get("dowhy") or {}
            if dw.get("available"):
                st.caption("DoWhy (backdoor linear regression)")
                st.dataframe(pd.DataFrame(dw["refuters"]).T, width="stretch")
            else:
                st.caption(f"DoWhy unavailable: {dw.get('reason')}")

# ============================================================ TAB 2
with t2:
    st.subheader("CATE model leaderboard")
    o2 = st.radio("Outcome", ["ancillary", "direct"], horizontal=True, key="o2",
                  format_func=lambda o: "Ancillary spend ($)" if o == "ancillary" else "Direct re-booking (binary)")
    lb = D["lb"][D["lb"].outcome == o2].reset_index(drop=True)
    cols = ["model", "qini", "auuc", "pehe", "ate_bias", "corr_true", "policy_value_ipw", "policy_value_true", "fit_seconds", "latency_ms_per_1k"]
    st.dataframe(lb[cols].round(4), width="stretch", hide_index=True)
    st.caption("Qini / AUUC use IPW-weighted observed outcomes only (no truth). PEHE, ATE bias, corr and *policy_value_true* use the synthetic ground truth. "
               "Oracle row = ranking by true τ(x) (ceiling).")
    cv = D["cv"][D["cv"].outcome == o2]
    sel = st.multiselect("Models on Qini chart", list(lb.model), default=list(lb.model[:4]) + ["Oracle (true tau)"] if "Oracle (true tau)" not in list(lb.model[:4]) else list(lb.model[:5]))
    qf = go.Figure()
    for i, m in enumerate(sel):
        g = cv[cv.model == m]
        qf.add_trace(go.Scatter(x=g.x, y=g.gain, name=m, mode="lines", line=dict(width=3 if m.startswith("Oracle") else 1.8, dash="dash" if m.startswith("Oracle") else "solid", color="black" if m.startswith("Oracle") else PAL[i % len(PAL)])))
    if len(cv):
        g0 = cv[cv.model == cv.model.iloc[0]]
        qf.add_trace(go.Scatter(x=g0.x, y=g0.random, name="Random", line=dict(color="grey", dash="dot")))
    qf.update_layout(height=430, xaxis_title="Fraction of guests targeted", yaxis_title="Cumulative incremental outcome / guest (IPW)", margin=dict(l=10, r=10, t=10, b=10))
    c1, c2 = st.columns(2)
    c1.plotly_chart(qf, width="stretch")
    real = lb[~lb.model.str.startswith("Oracle")]
    sf = px.scatter(real, x="latency_ms_per_1k", y="pehe", color="model", size=np.clip(real.qini.abs() / max(real.qini.abs().max(), 1e-9) * 25 + 8, 8, 40),
                    log_x=True, color_discrete_sequence=PAL, labels={"latency_ms_per_1k": "Inference latency (ms / 1k guests)", "pehe": "PEHE (lower is better)"})
    sf.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    c2.plotly_chart(sf, width="stretch")
    dec = D["dec"][D["dec"].outcome == o2]
    st.markdown(f"**Uplift by decile — {dec.model.iloc[0]}** (best Qini)")
    st.plotly_chart(px.bar(dec, x="decile", y="ipw_uplift", color_discrete_sequence=[PAL[0]]).update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10)), width="stretch")

# ============================================================ TAB 3
with t3:
    st.subheader("Guest simulator — one-click personas")
    FIELDS = dict(channel=sg.CHANNELS, room_tier=sg.TIERS, device=sg.DEVICES)
    if "g" not in st.session_state:
        st.session_state.g = dict(list(serving.PERSONAS.values())[0])

    def pick(name):
        st.session_state.g = dict(serving.PERSONAS[name])
        for k, v in st.session_state.g.items():
            st.session_state[f"w_{k}"] = v

    pc = st.columns(3)
    for i, name in enumerate(serving.PERSONAS):
        pc[i % 3].button(name, key=f"p{i}", on_click=pick, args=(name,), width="stretch")
    g = st.session_state.g
    f1, f2, f3, f4, f5 = st.columns(5)
    guest = dict(
        channel=f1.selectbox("Booking channel", sg.CHANNELS, index=sg.CHANNELS.index(g["channel"]), key="w_channel"),
        room_tier=f1.selectbox("Room tier", sg.TIERS, index=sg.TIERS.index(g["room_tier"]), key="w_room_tier"),
        lead_time=f2.number_input("Lead time (days)", 0, 365, int(g["lead_time"]), key="w_lead_time"),
        los=f2.number_input("Length of stay", 1, 30, int(g["los"]), key="w_los"),
        adults=f3.number_input("Adults", 1, 8, int(g["adults"]), key="w_adults"),
        children=f3.number_input("Children", 0, 8, int(g["children"]), key="w_children"),
        device=f4.selectbox("Device", sg.DEVICES, index=sg.DEVICES.index(g["device"]), key="w_device"),
        hist_spend=f4.number_input("Historical ancillary spend ($)", 0.0, 10000.0, float(g["hist_spend"]), key="w_hist_spend"),
        loyalty_tier=f5.selectbox("Loyalty tier", [0, 1, 2, 3], index=int(g["loyalty_tier"]), key="w_loyalty_tier", format_func=lambda i: ["None", "Silver", "Gold", "Platinum"][i]),
        resort=f5.selectbox("Resort property", [0, 1], index=int(g["resort"]), key="w_resort", format_func=lambda i: "Yes" if i else "No"))
    gdf = pd.DataFrame([guest])
    names_a, names_d = serving.model_names("ancillary"), serving.model_names("direct")
    m1, m2 = st.columns(2)
    ma = m1.selectbox("Ancillary model", names_a, index=names_a.index(serving.best_model("ancillary")))
    md = m2.selectbox("Direct-shift model", names_d, index=names_d.index(serving.best_model("direct")))
    r = serving.predict_guests(gdf, ma, md).iloc[0]
    tr_ = sg.true_effects(gdf).iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("τ̂ ancillary ($)", f"{r.tau_ancillary:+.2f}", help=f"Synthetic truth: {tr_.tau_anc:+.2f}")
    k2.metric("τ̂ direct re-booking (pp)", f"{r.tau_direct * 100:+.1f}", help=f"Synthetic truth: {tr_.tau_direct * 100:+.1f}")
    k3.metric("Expected net value ($)", f"{r.incremental_value:+.2f}")
    k4.metric("Predicted segment", r.segment_pred)
    st.caption(f"Synthetic ground truth for this guest: τ_anc = {tr_.tau_anc:+.2f} $, τ_direct = {tr_.tau_direct * 100:+.1f} pp, true segment = **{tr_.segment}**, "
               f"P(contacted) = {tr_.e_true:.2f}.")
    if "tau_ancillary_lo" in r.index:
        st.caption(f"Causal Forest 95% interval — ancillary: [{r.tau_ancillary_lo:+.2f}, {r.tau_ancillary_hi:+.2f}]")
    rows = []
    for o, nm in (("ancillary", names_a), ("direct", names_d)):
        for n in nm:
            kw = dict(model_anc=n) if o == "ancillary" else dict(model_dir=n)
            rr = serving.predict_guests(gdf, **kw).iloc[0]
            rows.append(dict(outcome=o, model=n, tau=rr[f"tau_{o}"] * (100 if o == "direct" else 1)))
    cmp = pd.DataFrame(rows)
    truth = {"ancillary": tr_.tau_anc, "direct": tr_.tau_direct * 100}
    cc = st.columns(2)
    for c, o, ttl in ((cc[0], "ancillary", "τ̂ ancillary ($) by model"), (cc[1], "direct", "τ̂ direct re-booking (pp) by model")):
        d = cmp[cmp.outcome == o]
        ff = px.bar(d, x="tau", y="model", orientation="h", title=ttl, color_discrete_sequence=[PAL[0]])
        ff.add_vline(x=truth[o], line_dash="dash", line_color="black", annotation_text="truth")
        ff.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        c.plotly_chart(ff, width="stretch")

# ============================================================ TAB 4
with t4:
    st.subheader("Campaign optimiser under a marketing budget")
    b = CFG["business"]
    SCEN = {"Baseline (30% budget)": dict(bf=0.30, cost=b["cost_per_offer"], am=b["ancillary_margin"], dm=b["direct_booking_margin"], ch=[]),
            "Peak season — tight 10% budget": dict(bf=0.10, cost=b["cost_per_offer"], am=b["ancillary_margin"], dm=b["direct_booking_margin"], ch=[]),
            "Aggressive 60% (cheap SMS, $0.40)": dict(bf=0.60, cost=0.40, am=b["ancillary_margin"], dm=b["direct_booking_margin"], ch=[]),
            "OTA-only push (commission 20%)": dict(bf=0.30, cost=b["cost_per_offer"], am=b["ancillary_margin"], dm=75.0, ch=["OTA_Expedia", "OTA_Booking"]),
            "Expensive offers ($6/guest)": dict(bf=0.30, cost=6.0, am=b["ancillary_margin"], dm=b["direct_booking_margin"], ch=[])}
    if "s4" not in st.session_state:
        st.session_state.s4 = SCEN["Baseline (30% budget)"]

    def load_s(n):
        s = SCEN[n]
        st.session_state.s4 = s
        st.session_state.update(bf=s["bf"], cost=s["cost"], am=s["am"], dm=s["dm"], ch=s["ch"])

    sc_cols = st.columns(len(SCEN))
    for c, n in zip(sc_cols, SCEN):
        c.button(n, on_click=load_s, args=(n,), key="sc_" + n, width="stretch")
    s = st.session_state.s4
    q1, q2, q3, q4, q5 = st.columns(5)
    bf = q1.slider("Budget: share of guests contacted", 0.05, 1.0, float(s["bf"]), 0.05, key="bf")
    cost = q2.number_input("Cost per offer ($)", 0.0, 50.0, float(s["cost"]), 0.1, key="cost")
    am = q3.slider("Ancillary margin", 0.1, 1.0, float(s["am"]), 0.05, key="am")
    dm = q4.number_input("Value of a direct booking ($)", 0.0, 500.0, float(s["dm"]), 5.0, key="dm")
    chs = q5.multiselect("Channels (empty = all)", sg.CHANNELS, default=s["ch"], key="ch")
    only_pos = st.checkbox("Only contact guests with positive expected net value", True)
    res = serving.optimize_campaign(bf, cost, am, dm, only_pos, chs or None)
    k = st.columns(5)
    k[0].metric("Guests contacted", f"{res['n_selected']:,}", f"{res['share_selected']:.0%} of {res['n_guests']:,}")
    k[1].metric("Spend", f"${res['spend']:,.0f}")
    k[2].metric("Estimated net value", f"${res['est_net_value']:,.0f}")
    k[3].metric("True net value (synthetic)", f"${res['true_net_value']:,.0f}")
    k[4].metric("ROI (true)", f"{res['roi_true'] * 100:,.0f}%" if res["roi_true"] is not None else "—")
    comp = pd.DataFrame({"policy": ["Causal model targeting", "Random, same size", "Oracle (true τ)", "Contact everyone"],
                         "net_value": [res["true_net_value"], res["random_net_value"], res["oracle_net_value"], res["treat_all_net_value"]]})
    cf = px.bar(comp, x="policy", y="net_value", color="policy", color_discrete_sequence=PAL, text_auto=".3s")
    cf.update_layout(height=350, showlegend=False, yaxis_title="True net value ($, test guests)", margin=dict(l=10, r=10, t=10, b=10))
    d1, d2 = st.columns([3, 2])
    d1.plotly_chart(cf, width="stretch")
    d2.markdown("**Contacted guests by channel**")
    d2.dataframe(pd.DataFrame(res["by_channel"]).round(3), width="stretch", hide_index=True)
    d2.markdown("**By true segment (synthetic)**")
    d2.dataframe(pd.DataFrame(res["by_true_segment"]), width="stretch", hide_index=True)
    st.caption("Treat-all can beat a tight-budget policy: the budget constraint is what makes targeting valuable. "
               "Compare the model to *random of the same size* and to the *oracle* for the same budget.")
